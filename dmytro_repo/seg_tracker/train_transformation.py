import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import defaultdict
import madgrad
import config

import common_utils
import dataset_transform
import models_transformation
from common_utils import DotDict, timeit_context
import offset_grid_to_transform

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(seed=42)


def build_model(cfg):
    model_params = cfg['model_params']
    model: nn.Module = models_transformation.__dict__[model_params['model_cls']](cfg=model_params, pretrained=True)
    return model


def train(experiment_name: str, fold: int, continue_epoch: int = -1):
    model_str = experiment_name
    distributed = False

    cfg = common_utils.load_config_data(experiment_name)

    model_params = DotDict(cfg["model_params"])
    model_type = model_params.model_type
    train_params = DotDict(cfg["train_params"])

    checkpoints_dir = f"../output/checkpoints/{model_str}/{fold}"
    tensorboard_dir = f"../output/tensorboard/{model_type}/{model_str}_{fold}"
    oof_dir = f"../output/oof/{model_str}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)
    print("\n", experiment_name, "\n")

    logger = SummaryWriter(log_dir=tensorboard_dir)

    scaler = torch.cuda.amp.GradScaler()

    dataset_train = dataset_transform.DatasetTransform(
        stage=dataset_transform.DatasetTransform.STAGE_TRAIN,
        fold=fold,
        cfg_data=cfg['dataset_params'],
        return_torch_tensors=True,
        small_subset=False
    )

    dataset_valid = dataset_transform.DatasetTransform(
        stage=dataset_transform.DatasetTransform.STAGE_VALID,
        fold=fold,
        cfg_data=cfg['dataset_params'],
        return_torch_tensors=True,
        small_subset=False
    )

    batch_size = cfg['train_data_loader']['batch_size']
    nb_samples_per_epoch = cfg['train_params']['nb_samples_per_epoch']
    nb_samples_valid = cfg['train_params']['nb_samples_valid']

    data_loaders = {
        "train": DataLoader(
            dataset_train,
            num_workers=cfg['train_data_loader']['num_workers'],
            shuffle=False,
            batch_size=batch_size,
            sampler=torch.utils.data.RandomSampler(dataset_train, num_samples=nb_samples_per_epoch, replacement=True)
        ),
        "val": DataLoader(
            dataset_valid,
            num_workers=cfg['val_data_loader']['num_workers'],
            shuffle=False,
            batch_size=cfg['val_data_loader']['batch_size'],
            sampler=torch.utils.data.RandomSampler(dataset_valid, num_samples=nb_samples_valid, replacement=True)
        ),
    }

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.train()

    initial_lr = float(train_params.initial_lr)
    if train_params.optimizer == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "madgrad":
        optimizer = madgrad.MADGRAD(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, nesterov=True)
    else:
        raise RuntimeError("Invalid optimiser" + train_params.optimizer)

    nb_epochs = train_params.nb_epochs
    if train_params.scheduler == "steps":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=train_params.optimiser_milestones,
            gamma=0.2,
            last_epoch=continue_epoch
        )
    elif train_params.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = common_utils.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=train_params.scheduler_period,
            T_mult=train_params.get('scheduler_t_mult', 1),
            eta_min=initial_lr / 1000.0,
            last_epoch=-1,
            first_epoch_lr_scale=0.01
        )
        for i in range(continue_epoch + 1):
            scheduler.step()
    else:
        raise RuntimeError("Invalid scheduler name")

    if continue_epoch > -1:
        print(f"{checkpoints_dir}/{continue_epoch:03}.pt")
        checkpoint = torch.load(f"{checkpoints_dir}/{continue_epoch:03}.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    grad_clip_value = train_params.get("grad_clip", 8.0)
    print("grad clip:", grad_clip_value)
    print(f"Num training samples: {len(dataset_train)} validation samples: {len(dataset_valid)}")

    cr_mse = torch.nn.MSELoss(reduction='none')

    for epoch_num in range(continue_epoch + 1, nb_epochs + 1):
        for phase in ["train", "val"]:
            model.train(phase == "train")

            epoch_loss = common_utils.AverageMeter()
            epoch_metric = common_utils.AverageMeter()
            epoch_loss_separate = defaultdict(common_utils.AverageMeter)

            data_loader = data_loaders[phase]

            optimizer.zero_grad()

            nb_batches = 0
            data_iter = tqdm(data_loader)
            for data in data_iter:
                with torch.set_grad_enabled(phase == "train"):
                    cur_frame = data['cur_img'].cuda()
                    prev_frame = data['prev_img'].cuda()

                    cur_points = data['cur_points_grid'].cuda()
                    prev_points = data['prev_points_grid'].cuda()

                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        heatmap, offsets = model(prev_frame, cur_frame)
                        gt_offsets = (cur_points - prev_points)[..., 2:-2, 2:-2]
                        loss = (cr_mse(offsets, gt_offsets) * heatmap).sum(dim=(1, 2, 3)).mean()

                    epoch_metric.update(0, batch_size)

                    if phase == "train":
                        scaler.scale(loss).backward()

                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(optimizer)
                        if grad_clip_value > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                        scaler.step(optimizer)
                        scaler.update()

                    if phase == "val":
                        # save predictions visualisation
                        pass

                    epoch_loss.update(loss.detach().item(), batch_size)

                    data_iter.set_description(
                        f"{epoch_num} {phase[0]}"
                        f" Loss {epoch_loss.avg:1.4f}"
                    )

            if epoch_num > 0:
                logger.add_scalar(f"loss_{phase}", epoch_loss.avg, epoch_num)

                if phase == "train":
                    logger.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_num)
                logger.flush()

            if phase == "train":
                scheduler.step()
                if (epoch_num % train_params.save_period == 0) or (epoch_num == nb_epochs):
                    torch.save(
                        {
                            "epoch": epoch_num,
                            "model_state_dict": model.module.state_dict() if distributed else model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        f"{checkpoints_dir}/{epoch_num:03}.pt",
                    )



def check(experiment_name: str, fold: int, epoch: int = -1):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"../output/checkpoints/{model_str}/{fold}"
    print("\n", experiment_name, "\n")

    cfg['dataset_params']['synthetic_img_ratio'] = 0.0

    dataset_valid = dataset_transform.DatasetTransform(
        stage=dataset_transform.DatasetTransform.STAGE_VALID,
        fold=fold,
        cfg_data=cfg['dataset_params'],
        return_torch_tensors=True,
        small_subset=False
    )

    data_loader = DataLoader(
            dataset_valid,
            num_workers=2,
            shuffle=False,
            batch_size=1,
            sampler=torch.utils.data.RandomSampler(dataset_valid, num_samples=1024, replacement=True)
        )

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()
    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cr_mse = torch.nn.MSELoss(reduction='none')

    data_iter = tqdm(data_loader)
    for data in data_iter:
        with torch.set_grad_enabled(False):
            cur_frame = data['cur_img'].cuda()
            prev_frame = data['prev_img'].cuda()
            cur_points = data['cur_points_grid'][..., 2:-2, 2:-2].cuda()
            prev_points = data['prev_points_grid'][..., 2:-2, 2:-2].cuda()

            heatmap, offsets = model(prev_frame, cur_frame)
            gt_offsets = (cur_points - prev_points)
            loss = (cr_mse(offsets, gt_offsets) * heatmap).sum(dim=(1, 2, 3)).mean()
            print(f'loss {float(loss):0.1f}')
            print(f'heatmap sum {float(heatmap[0].sum()):0.1f}')
            print(f'mean gt offset {float(gt_offsets[0, 0].mean()):0.1f} {float(gt_offsets[0, 1].mean()):0.1f}')
            print(f'mean pr offset {float(offsets[0, 0].mean()):0.1f} {float(offsets[0, 1].mean()):0.1f}')
            weighted_offset = (offsets[0] * heatmap[0]).sum(dim=(1, 2))
            print(f'mean pw offset {float(weighted_offset[0]):0.1f} {float(weighted_offset[1]):0.1f}')

            center_offset = np.array([512.0, 512.0])[:, None]

            cur_points_pred = (prev_points + offsets).detach().cpu().numpy()
            dx, dy, angle, err = offset_grid_to_transform.offset_grid_to_transform_params(
                prev_frame_points=prev_points.cpu().numpy().reshape(2, -1) - center_offset,
                cur_frame_points=cur_points_pred.reshape(2, -1) - center_offset,
                points_weight=heatmap[0].detach().cpu().numpy().reshape(-1)
            )
            dx2, dy2, angle2, err2 = offset_grid_to_transform.offset_grid_to_transform_params(
                prev_frame_points=prev_points.cpu().numpy().reshape(2, -1) - center_offset,
                cur_frame_points=cur_points_pred.reshape(2, -1) - center_offset,
                points_weight=heatmap[0].detach().cpu().numpy().reshape(-1) ** 2
            )

            tr, _ = offset_grid_to_transform.offset_grid_to_transform(
                prev_frame_points=prev_points.cpu().numpy().reshape(2, -1),
                cur_frame_points=cur_points_pred.reshape(2, -1),
                points_weight=heatmap[0].detach().cpu().numpy().reshape(-1) ** 2
            )

            print(f'Estimated angle {angle:0.2f} offset {dx:0.1f},{dy:0.1f} err {err:0.2f}')
            print(f'Estimated angle {angle2:0.2f} offset {dx2:0.1f},{dy2:0.1f} err {err2:0.2f} w2')

            print(f"GT        angle {float(data['angle'][0]):0.2f} offset {float(data['dx'][0]):0.1f},{float(data['dy'][0]):0.1f} "
                  f"scale {float(data['scale'][0]):0.3f} synt: {data['is_synthetic'][0]}")

            if float(loss) < 20:
                continue

            fig, ax = plt.subplots(2, 3)
            ax[0, 0].imshow(data['cur_img'][0].numpy())
            ax[1, 0].imshow((heatmap[0, 0] / heatmap[0, 0].max()).detach().cpu().numpy())

            ax[0, 1].imshow(gt_offsets[0, 0].cpu().numpy())
            ax[0, 2].imshow(gt_offsets[0, 1].cpu().numpy())

            ax[1, 1].imshow(offsets[0, 0].detach().cpu().numpy())
            ax[1, 2].imshow(offsets[0, 1].detach().cpu().numpy())
            plt.figure()

            cur_img = data['cur_img'][0].numpy()
            prev_img = data['prev_img'][0].numpy()

            h, w = cur_img.shape

            prev_tr2 = common_utils.build_geom_transform(
                dst_w=w,
                dst_h=h,
                src_center_x=w / 2 + dx2,
                src_center_y=h / 2 + dy2,
                scale_x=1.0,
                scale_y=1.0,
                angle=angle2,
                return_params=True
            )

            prev_img_aligned2 = cv2.warpAffine(
                prev_img,
                prev_tr2[:2, :],  # the same crop as with the cur image
                dsize=(w, h),
                flags=cv2.INTER_LINEAR)

            plt.imshow(np.stack([cur_img, prev_img_aligned2, cur_img], axis=2))
            plt.figure()
            print(tr[:2, :])

            prev_img_aligned = cv2.warpAffine(
                prev_img,
                tr[:2, :],  # the same crop as with the cur image
                dsize=(cur_img.shape[1], cur_img.shape[0]),
                flags=cv2.INTER_LINEAR)

            plt.imshow(np.stack([cur_img, prev_img_aligned, cur_img], axis=2))
            plt.show()


def export_model(experiment_name: str, fold: int, epoch: int):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)

    checkpoints_dir = f"../output/checkpoints/{model_str}/{fold}"
    print("\n", experiment_name, "\n")

    dst_dir = '../output/models'
    os.makedirs(dst_dir, exist_ok=True)

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()

    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    torch.save(
        {
            "model_state_dict": model.state_dict()
        },
        f'{dst_dir}/{experiment_name}_{epoch}_{fold}.pth'
    )

    # with torch.jit.optimized_execution(True):
    #     traced_model = torch.jit.trace(model, (torch.rand(1, 1024, 1024).cuda(), torch.rand(1, 1024, 1024).cuda()))
    #
    # torch.jit.save(traced_model, f'{dst_dir}/{experiment_name}_{epoch}_{fold}j.pth')
    #
    # with torch.jit.optimized_execution(True):
    #     traced_model = torch.jit.trace(model.half(), (torch.rand(1, 1024, 1024).half().cuda(), torch.rand(1, 1024, 1024).half().cuda()))
    #
    # torch.jit.save(traced_model, f'{dst_dir}/{experiment_name}_{epoch}_{fold}jh.pth')


from load_images_pipe import *

def find_transforms(model, img_crops):
    all_frames = torch.from_numpy(np.stack(img_crops, axis=0) / 255.0).float().cuda()
    cur_frame = all_frames[1:]
    prev_frame = all_frames[:-1]

    prev_points = np.zeros((2, 32, 32), dtype=np.float32)
    prev_points[0, :, :] = np.arange(16, 1024, 32)[None, :]
    prev_points[1, :, :] = np.arange(16, 1024, 32)[:, None]
    prev_points = prev_points[..., 2:-2, 2:-2]

    center_offset = np.array([512.0, 512.0])[:, None]

    heatmap, offsets = model(prev_frame, cur_frame)
    heatmap = heatmap.detach().cpu().numpy()
    offsets = offsets.detach().cpu().numpy()

    res = []
    for i in range(heatmap.shape[0]):
        cur_points = prev_points + offsets[i]

        dx, dy, angle, err = dataset_transform.offset_grid_to_transform_params(
            prev_frame_points=prev_points.reshape(2, -1) - center_offset,
            cur_frame_points=cur_points.reshape(2, -1) - center_offset,
            points_weight=heatmap[i].reshape(-1) ** 2
        )
        # print(f'Estimated angle {angle:0.2f} offset {dx:0.1f},{dy:0.1f} err {err:0.2f}')

        res.append([dx, dy, angle, err])
    return res



def predict_dataset_offsets(part='part1'):
    experiment_name = config.TRANSFORM_MODEL
    fold = config.TRANSFORM_MODEL_FOLD
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"../output/checkpoints/{model_str}/{fold}"
    print("\n", experiment_name, "\n")
    torch.set_grad_enabled(False)

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()
    print(f"{checkpoints_dir}/{config.TRANSFORM_MODEL_EPOCH:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{config.TRANSFORM_MODEL_EPOCH:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(f'{config.DATA_DIR}/{part}/ImageSets/groundtruth.csv')
    flight_ids = list(sorted(df['flight_id'].unique()))

    load_img_stage = mpipe.OrderedStage(load_img, 1)
    decode_img_stage = mpipe.OrderedStage(decode_img, 16)
    load_img_stage.link(decode_img_stage)
    pipe = mpipe.Pipeline(load_img_stage)

    for flight_id in tqdm(flight_ids):
        df_flight = df[df.flight_id == flight_id][['img_name', 'frame']].drop_duplicates().reset_index(drop=True)
        file_names = [f'{config.DATA_DIR}/{part}/Images/part{part}{flight_id}/{fn[:-4]}.{config.IMG_FORMAT}' for fn in df_flight.img_name] #ADDED {part} in front of {flight_id}

        crop_w = 1024
        crop_h = 1024
        batch_size = 4

        last_img = np.zeros((1024, 1024), dtype=np.uint8)

        load_failed = []

        frame_transforms = [[0, 0, 0, 0]]  # dx, dy, angle, err

        img_crops = []  # not processed images yet of one frame + batch size
        for fn, img in tqdm(limited_pipe(pipe, file_names, 32), total=len(file_names)):
            load_failed.append(img is None)
            if img is None:
                img = last_img
            last_img = img

            h, w = img.shape

            y0 = (h - crop_h) // 2
            x0 = (w - crop_w) // 2
            img_crop = img[y0:y0+crop_h, x0: x0+crop_w]
            img_crops.append(img_crop)

            if len(img_crops) >= batch_size:
                frame_transforms += find_transforms(model, img_crops)
                img_crops = [img_crops[-1]]

        if len(img_crops) > 1:
            frame_transforms += find_transforms(model, img_crops)

        del img_crops

        dst_dir = f'{config.DATA_DIR}/frame_transforms/{part}'
        os.makedirs(dst_dir, exist_ok=True)

        frame_transforms = np.array(frame_transforms)
        df_flight['load_failed'] = load_failed
        df_flight['dx'] = frame_transforms[:, 0]
        df_flight['dy'] = frame_transforms[:, 1]
        df_flight['angle'] = frame_transforms[:, 2]
        df_flight['error'] = frame_transforms[:, 3]

        df_flight.to_pickle(f'{dst_dir}/{flight_id}.pkl')
        df_flight.to_csv(f'{dst_dir}/{flight_id}.csv', index=False)
    pipe.put(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default="check")
    parser.add_argument("experiment", type=str, default="")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--part", type=str, default='')

    args = parser.parse_args()
    action = args.action
    experiment_name = common_utils.normalize_experiment_name(args.experiment)

    if action == "train":
        train(
            experiment_name=experiment_name,
            continue_epoch=args.epoch,
            fold=args.fold
        )

    if action == "check":
        check(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )

    if action == "export_model":
        export_model(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )

    if action == "predict_dataset_offsets":
        predict_dataset_offsets(args.part)
