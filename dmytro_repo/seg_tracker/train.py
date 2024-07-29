import argparse
import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import defaultdict
import madgrad
# import sam
import focal_loss
import copy

import common_utils
import dataset_tracking
import seg_prediction_to_items
import models_segmentation
from common_utils import DotDict, timeit_context
import pandas as pd #ADDED

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# seed_everything(seed=49)


def build_model(cfg):
    model_params = cfg['model_params']
    # if model_params['model_type'] == 'models_segmentation':
    model: nn.Module = models_segmentation.__dict__[model_params['model_cls']](cfg=model_params, pretrained=True)
    return model


class TrackPreferredSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: dataset_tracking.TrackingDataset, num_samples: int, tracking_samples_share: 0.4):
        super().__init__(dataset)
        print('idk man')
        self.data_source = dataset
        self.num_samples = num_samples
        self.tracking_samples_share = tracking_samples_share
        self.generator = torch.Generator()
        self.generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

    def __iter__(self) -> Iterator[int]:
        block_size = 16
        for _ in range(self.num_samples // block_size):
            nb_tracking = round(block_size * self.tracking_samples_share)
            samples_tracking = np.random.choice(self.data_source.frame_nums_with_match_items, nb_tracking)
            # samples_other = np.random.choice(self.data_source.frame_nums_with_items, block_size-nb_tracking)
            samples_other = np.random.randint(low=0, high=len(self.data_source.frames), size=block_size - nb_tracking)
            yield from samples_tracking
            yield from samples_other

    def __len__(self):
        return self.num_samples


class TrackPreferredSamplerWithFP(torch.utils.data.Sampler[int]):
    def __init__(self,
                 dataset: dataset_tracking.TrackingDataset,
                 num_samples: int,
                 fp_samples,
                 tracking_samples_share: 0.25,
                 fp_samples_share: 0.25):
        super().__init__(dataset)
        self.data_source = dataset
        self.num_samples = num_samples
        self.tracking_samples_share = tracking_samples_share
        self.fp_samples_share = fp_samples_share

        self.fp_samples_p = np.array([s[0]+0.1 for s in fp_samples])
        self.fp_samples_p /= self.fp_samples_p.sum()
        self.fp_samples_idx = np.array([s[1] for s in fp_samples])

    def __iter__(self) -> Iterator[int]:
        block_size = 64
        for _ in range(self.num_samples // block_size):
            nb_tracking = round(block_size * self.tracking_samples_share)
            nb_fp = round(block_size * self.fp_samples_share)

            samples_tracking = np.random.choice(self.data_source.frame_nums_with_match_items, nb_tracking)
            samples_fp = np.random.choice(self.fp_samples_idx, nb_fp, p=self.fp_samples_p)
            samples_other = np.random.randint(low=0, high=len(self.data_source.frames), size=block_size - nb_tracking - nb_fp)

            all_samples = np.concatenate([samples_tracking, samples_fp, samples_other])
            np.random.shuffle(all_samples)
            yield from all_samples

    def __len__(self):
        return self.num_samples


class TrackPreferredSamplerValid(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: dataset_tracking.TrackingDataset, num_samples: int, tracking_samples_share: 0.75):
        super().__init__(dataset)
        self.data_source = dataset
        self.num_samples = num_samples
        self.tracking_samples_share = tracking_samples_share

        nb_tracking = round(num_samples * self.tracking_samples_share)
        step = len(self.data_source.frame_nums_with_match_items) // (2*nb_tracking)
        self.samples_tracking = self.data_source.frame_nums_with_match_items[::step][:nb_tracking]

        step = len(self.data_source.frame_nums_with_items) // (2*(num_samples - nb_tracking))
        self.samples_other = self.data_source.frame_nums_with_items[::step][:num_samples - nb_tracking]

        print('TrackPreferredSamplerValid', len(self.samples_tracking), len(self.samples_other))

    def __iter__(self) -> Iterator[int]:
        yield from list(self.samples_tracking)
        yield from list(self.samples_other)

    def __len__(self):
        return self.num_samples


class ItemsWithPlaneSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: dataset_tracking.TrackingDataset, step=8):
        super().__init__(dataset)
        self.data_source = dataset
        self.items = list(self.data_source.frame_nums_with_match_items[::step])

    def __iter__(self) -> Iterator[int]:
        yield from self.items

    def __len__(self):
        return len(self.items)


def combine_images(data, input_frames):
    image = data['image'].float().cuda()
    image_prev0 = data['prev_image_aligned0'].float().cuda()

    images = [image_prev0, image]
    for img_idx in range(1, input_frames - 1):
        images.append(data[f'prev_image_aligned{img_idx}'].float().cuda())

    return torch.stack(images, dim=1)


class SkipNSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: dataset_tracking.TrackingDataset, skip_step: int):
        super().__init__(dataset)
        self.data_source = dataset
        self.samples = list(range(np.random.randint(skip_step), len(self.data_source), skip_step))
        self.num_samples = len(self.samples)
        print('SkipNSampler', self.num_samples)

    def __iter__(self) -> Iterator[int]:
        yield from self.samples

    def __len__(self):
        return self.num_samples


from check_frame_level_prediction import Box, DetectedItem, extend_bounding_boxes, calc_iou


def find_fp_samples(model, dataset: dataset_tracking.TrackingDataset, step: int, batch_size: int = 8, input_frames: int = 2):
    """
    Find the false positive samples for dataset with index step.
    :param model:
    :param dataset:
    :param step:
    :param batch_size:
    :return:

    [
            [max_conf, frame_index, [[conf, cx, cy], [conf, cx, cy], ...]]
    ]
    """

    sampler = SkipNSampler(dataset, step)
    print('total frames:', len(sampler))
    data_loader = DataLoader(
        dataset,
        num_workers=8,
        shuffle=False,
        batch_size=batch_size,
        sampler=sampler
    )
    model.eval()

    res = []

    data_iter = tqdm(data_loader)
    for data in data_iter:
        with torch.set_grad_enabled(False):
            images = combine_images(data, input_frames)
            with torch.cuda.amp.autocast():
                pred = model(images)

            pred['mask'] = torch.sigmoid(pred['mask'])
            pred = {k: pred[k].cpu().detach().numpy() for k in pred.keys()}

            for i in range(images.shape[0]):
                x_offset = float(data['crop_x'][i])
                y_offset = float(data['crop_y'][i])
                comb_pred = pred['mask'][i, 0]
                detected_objects = seg_prediction_to_items.pred_to_items(
                    comb_pred=comb_pred,
                    offset=pred['offset'][i],
                    size=pred['size'][i],
                    tracking=pred['tracking'][i],
                    distance=pred['distance'][i, 0],
                    above_horizon=pred['above_horizon'][i, 0],
                    conf_threshold=0.12,
                    pred_scale=8.0,
                    x_offset=x_offset,
                    y_offset=y_offset
                )
                item_idx = int(data['idx'][i])
                gt_items = [extend_bounding_boxes(copy.copy(it)) for it in dataset.frames[item_idx].items]
                pred_items = [
                    extend_bounding_boxes(DetectedItem(
                        item_id='airborne',
                        cx=it['cx'],
                        cy=it['cy'],
                        w=it['w'],
                        h=it['h'],
                        confidence=it['conf'],
                        distance=it['distance'],
                        dx=0,
                        dy=0
                    ))
                    for it in detected_objects
                ]
                not_matched_objects = []
                for pred_item in pred_items:
                    matched = False
                    for gt_item in gt_items:
                        iou = calc_iou(pred_item, gt_item)
                        if iou > 0.1:
                            matched = True
                            break

                    if not matched:
                        not_matched_objects.append((pred_item.confidence, pred_item.cx, pred_item.cy))

                if len(not_matched_objects):
                    max_conf = max(o[0] for o in not_matched_objects)
                    res.append((max_conf, item_idx, tuple(not_matched_objects)))
    return res


def check_find_fp_samples(experiment_name, epoch):
    model_str = experiment_name
    fold = 0
    cfg = common_utils.load_config_data(experiment_name)
    checkpoints_dir = f"../output/checkpoints/{model_str}/{fold}"
    print("\n", experiment_name, "\n")

    dataset_train_full_size = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_TRAIN,
        parts=[1], #[2, 3],
        cfg_data={'dataset_params': dict(
            back_steps=[1],
            scale=1,
            crop_with_plane_ratio=0,
            crop_size=(2432, 2048)
        )},
        return_torch_tensors=True,
        small_subset=False
    )

    model = build_model(cfg)
    print(model.__class__.__name__)

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
    model.eval()

    res = find_fp_samples(model=model, dataset=dataset_train_full_size, step=577)
    print(len(res))


def train(experiment_name: str, fold: int, continue_epoch: int = -1):
    print('made it to train()')
    model_str = experiment_name
    distributed = False
    store_epoch = []
    store_cls = []
    store_size = []
    store_offset = []
    store_distance = []
    store_tracking = []
    store_horizon = []
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
    dataset_train = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_TRAIN,
        cfg_data=cfg,
        return_torch_tensors=True,
        small_subset=False
    )

    # dataset_valid = dataset_tracking.TrackingDataset(
    #     stage=dataset_tracking.BaseDataset.STAGE_VALID,
    #     fold=fold,
    #     cfg_data=cfg,
    #     return_torch_tensors=True,
    #     small_subset=False,
    #     cache_samples=True
    # )

    cfg_full_size = copy.deepcopy(cfg)
    cfg_full_size['dataset_params']['crop_with_plane_ratio'] = 0
    cfg_full_size['dataset_params']['crop_size'] = (2432, 2048)

    dataset_train_full_size = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_TRAIN,
        cfg_data=cfg_full_size,
        return_torch_tensors=True,
        small_subset=False
    )
    dataset_train_full_size.is_training = False  # disable augmentations etc

    batch_size = cfg['train_data_loader']['batch_size']
    nb_samples_per_epoch = cfg['train_params']['nb_samples_per_epoch']
    nb_samples_valid = cfg['train_params']['nb_samples_valid']
    mask_loss_name = cfg['train_params'].get('mask_loss', 'bce')
    tracking_samples_share = cfg['train_params'].get('tracking_samples_share', 0.25)
    fp_samples_share = cfg['train_params'].get('fp_samples_share', 0.25)
    fpm_batch_size = cfg['train_params'].get('fpm_batch_size', 4)
    fpm_epochs = cfg['train_params'].get('fpm_epochs', {})

    data_loaders = {
        "train": DataLoader(
            dataset_train,
            num_workers=cfg['train_data_loader']['num_workers'],
            shuffle=False,
            batch_size=batch_size,
            sampler=TrackPreferredSampler(dataset_train, num_samples=nb_samples_per_epoch, tracking_samples_share=tracking_samples_share)
        ),
        # "val": DataLoader(
        #     dataset_valid,
        #     num_workers=cfg['val_data_loader']['num_workers'],
        #     shuffle=False,
        #     batch_size=cfg['val_data_loader']['batch_size'],
        #     sampler=TrackPreferredSamplerValid(dataset_train, num_samples=nb_samples_valid, tracking_samples_share=0.75)
        # ),
    }
    print('building model')
    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.train()
    print('model in training mode')
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
    print('optimizer set')
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

    print('scheduler set')
    if continue_epoch > -1:
        print(f"{checkpoints_dir}/{continue_epoch:03}.pt")
        checkpoint = torch.load(f"{checkpoints_dir}/{continue_epoch:03}.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    grad_clip_value = train_params.get("grad_clip", 8.0)
    freeze_backbone_steps = train_params.get("freeze_backbone_steps", 0)
    print(f"grad clip: {grad_clip_value} freeze_backbone_steps {freeze_backbone_steps}")
    print(f"Num training samples: {len(dataset_train)} tracking samples share: {tracking_samples_share} fp samples {fp_samples_share}")

    cr_mse = torch.nn.MSELoss(reduction='none')
    cr_mae = torch.nn.L1Loss(reduction='none')
    cr_bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    cr_fl = focal_loss.FocalLossV2(reduction='none')

    if mask_loss_name == 'bce':
        mask_loss = cr_bce
    elif mask_loss_name == 'fl':
        mask_loss = cr_fl
    else:
        raise RuntimeError('Invalid mask loss name '+mask_loss_name)

    #checkpoints_to_save = [250, 370, 535, 770, 1100, 1535, 2220]
    #checkpoints_to_save = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    checkpoints_to_save = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print('bout to calculate loss')
    for epoch_num in range(continue_epoch + 1, nb_epochs + 1):
        for phase in ["train"]:  # , "val"]:
            model.train(phase == "train")
            #print('1')
            if freeze_backbone_steps > 0:
                if epoch_num == 0:
                    model.freeze_encoder()
                if epoch_num == freeze_backbone_steps:
                    model.unfreeze_encoder()

            epoch_loss = common_utils.AverageMeter()
            epoch_metric = common_utils.AverageMeter()
            epoch_loss_separate = defaultdict(common_utils.AverageMeter)

            data_loader = data_loaders[phase]
            print(data_loader)
            optimizer.zero_grad()
            #print('2')
            nb_batches = 0
            #print('2.1')
            data_iter = tqdm(data_loader)
            #print('2.2')
            for data in data_iter:
                print(data)
                #print('2.3')
                with torch.set_grad_enabled(phase == "train"):
                    #print('2.5')
                    images = combine_images(data, model_params.input_frames)
                    #print('3')
                    label_cls = data['cls'].float().cuda()
                    label_cls_planned = data['cls_planned'].float().cuda()
                    reg_mask = data['reg_mask'].float().cuda()
                    reg_offset = data['reg_offset'].float().cuda()
                    reg_size = data['reg_size'].float().cuda()
                    reg_tracking = data['reg_tracking'].float().cuda()
                    reg_distance = data['reg_distance'].float().cuda()
                    reg_above_horizon = data['reg_above_horizon'].float().cuda()
                    reg_tracking_mask = data['reg_tracking_mask'].float().cuda()
                    reg_distance_mask = data['reg_distance_mask'].float().cuda()
                    reg_offset_mask = data['reg_offset_mask'].float().cuda()
                    reg_above_horizon_mask = data['reg_above_horizon_mask'].float().cuda()

                    optimizer.zero_grad()

                    if 'loss_scale' in train_params:
                        loss_items_scale = train_params['loss_scale']
                    else:
                        loss_items_scale = {
                            'cls': 2000,
                            'size': 1,
                            'offset': 0.2,
                            'distance': 1,
                            'tracking': 1,
                            'above_horizon': 1
                        }

                    with torch.cuda.amp.autocast():
                        pred = model(images)
                        #print('3.5')
                        loss_cls_all = mask_loss(pred['mask'], 1.0 - label_cls[:, :1])
                        loss_cls_planned = mask_loss(pred['mask'], 1.0 - label_cls_planned[:, :1])

                        if epoch_num > 75:
                            loss_cls = torch.minimum(loss_cls_all, loss_cls_planned).mean()
                        else:
                            loss_cls = loss_cls_all.mean()

                        reg_mask = reg_mask[:, None, :, :]

                        mask_size = reg_mask.sum() + 0.1
                        loss_size = cr_mse(pred['size'] * reg_mask, reg_size * reg_mask).sum() / mask_size

                        reg_offset_mask = reg_offset_mask[:, None, :, :]
                        reg_offset_mask_size = reg_offset_mask.sum() + 0.1
                        loss_offset = (cr_mse(pred['offset'] * reg_offset_mask, reg_offset * reg_offset_mask)).sum() / reg_offset_mask_size

                        loss_distance = cr_mse(pred['distance'][:, 0] * reg_distance_mask,
                                               reg_distance * reg_distance_mask).sum() / (reg_distance_mask.sum() + 0.1)

                        mask_tracking_size = reg_tracking_mask.sum() + 0.1
                        reg_tracking_mask = reg_tracking_mask[:, None, :, :]
                        loss_tracking = cr_mae(pred['tracking'] * reg_tracking_mask, reg_tracking * reg_tracking_mask).sum() / (
                                    mask_tracking_size.sum() + 0.1)

                        losses = [
                            ('cls', loss_cls),
                            ('size', loss_size),
                            ('offset', loss_offset),
                            ('distance', loss_distance),
                            ('tracking', loss_tracking)
                        ]

                        if model_params.get('output_above_horizon', False):
                            reg_above_horizon_mask_size = reg_above_horizon_mask.sum()
                            if reg_above_horizon_mask_size > 0:
                                loss_horizon = cr_bce(pred['above_horizon'][reg_above_horizon_mask > 0],
                                                      reg_above_horizon[reg_above_horizon_mask > 0]).mean()
                            else:
                                loss_horizon = torch.zeros((1,)).float().cuda()
                            losses.append(('above_horizon', loss_horizon))

                        loss = sum([loss_val * loss_items_scale[loss_name] for loss_name, loss_val in losses])

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
                    print('updated loss, bout to store vals')
                    store_epoch.append(epoch_num)
                    store_cls.append(loss_cls.detach().cpu().numpy())
                    store_size.append(loss_size.detach().cpu().numpy())
                    store_offset.append(loss_offset.detach().cpu().numpy())
                    store_distance.append(loss_distance.detach().cpu().numpy())
                    store_tracking.append(loss_tracking.detach().cpu().numpy())
                    store_horizon.append(loss_horizon.detach().cpu().numpy())
                    for loss_name, loss_val in losses:
                        # store_losses_row.append(loss_val)#ADDED
                        # store_losses['epoch_num'] = epoch_num
                        # store_losses[]
                        epoch_loss_separate[loss_name].update(loss_val.detach().item(), batch_size)



                    data_iter.set_description(
                        f"{epoch_num} {phase[0]}"
                        f" Loss {epoch_loss.avg:1.4f}" + ' '.join(
                            f" {loss_name} {loss_val.avg * loss_items_scale[loss_name]:1.4f}" for loss_name, loss_val in epoch_loss_separate.items()
                        )
                    )

                    del loss, losses, loss_cls, loss_size, loss_distance, loss_items_scale, loss_offset, loss_tracking
            print('4')
            if epoch_num > 0:
                logger.add_scalar(f"loss_{phase}", epoch_loss.avg, epoch_num)
                logger.add_scalar(f"metrics_{phase}", epoch_metric.avg, epoch_num)

                for loss_name, loss_val in epoch_loss_separate.items():
                    logger.add_scalar(f"loss_{loss_name}_{phase}", loss_val.avg, epoch_num)

                if phase == "train":
                    logger.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_num)
                logger.flush()

            if phase == "train":
                scheduler.step()
                # if (epoch_num % train_params.save_period == 0) or (epoch_num == nb_epochs):
                if epoch_num in checkpoints_to_save:
                    torch.save(
                        {
                            "epoch": epoch_num,
                            "model_state_dict": model.module.state_dict() if distributed else model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        f"{checkpoints_dir}/{epoch_num:03}.pt",
                    )
        if epoch_num in fpm_epochs:
            fn = f"{checkpoints_dir}/fp_samples_{epoch_num:03}.pt"
            if os.path.exists(fn):
                fp_samples = torch.load(fn)
            else:
                fp_samples = find_fp_samples(model, dataset=dataset_train_full_size, step=fpm_epochs[epoch_num],
                                             batch_size=fpm_batch_size, input_frames=model_params.input_frames)
                torch.save(fp_samples, fn)
            print()
            print(f'Found fp samples: {len(fp_samples)} share {len(fp_samples)/(len(dataset_train_full_size)//fpm_epochs[epoch_num])}')
            dataset_train.set_fpm_samples(fp_samples)
            data_loaders['train'] = DataLoader(
                dataset_train,
                num_workers=cfg['train_data_loader']['num_workers'],
                shuffle=False,
                batch_size=batch_size,
                sampler=TrackPreferredSamplerWithFP(
                    dataset_train,
                    num_samples=nb_samples_per_epoch,
                    fp_samples=fp_samples,
                    tracking_samples_share=tracking_samples_share,
                    fp_samples_share=fp_samples_share
                )
            )
    store = {} #tensor.detach().cpu().numpy()
    store['epoch_num'] = store_epoch#.detach().cpu().numpy()
    store['loss_cls'] = store_cls#.detach().cpu().numpy()
    store['loss_size'] = store_size#.detach().cpu().numpy()
    store['loss_offset'] = store_offset#.detach().cpu().numpy()
    store['loss_distance'] = store_distance#.detach().cpu().numpy()
    store['loss_tracking'] = store_tracking#.detach().cpu().numpy()
    store['loss_horizon'] = store_horizon#.detach().cpu().numpy()
    df = pd.DataFrame(store)
    df.to_csv(f"/gv1/projects/GRIP_Precog_Opt/data_loading/dmytro-airborne-detection-starter-kit-master/output/{model_str}_loss_nums.csv")
    print('done')
    


def check(experiment_name: str, fold: int, epoch: int = -1):
    model_str = experiment_name
    cfg = common_utils.load_config_data(experiment_name)

    model_params = DotDict(cfg["model_params"])

    checkpoints_dir = f"../output/checkpoints/{model_str}/{fold}"
    print("\n", experiment_name, "\n")

    # dataset_valid = dataset_tracking.TrackingDataset(
    #     stage=dataset_tracking.BaseDataset.STAGE_VALID,
    #     cfg_data=cfg,
    #     return_torch_tensors=True,
    #     small_subset=False
    # )

    cfg_full_size = copy.deepcopy(cfg)
    cfg_full_size['dataset_params']['crop_with_plane_ratio'] = 0
    # cfg_full_size['dataset_params']['crop_size'] = (2432, 2048)
    padding = (16 + 32 + 64) // 2
    # padding = -8
    cfg_full_size['dataset_params']['crop_size'] = (2448 + padding*2, 2048)

    dataset_valid = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_VALID,
        # parts=parts,
        cfg_data=cfg_full_size,
        return_torch_tensors=True,
        small_subset=False
    )

    batch_size = cfg['train_data_loader']['batch_size']
    nb_samples_per_epoch = cfg['train_params']['nb_samples_per_epoch']
    nb_samples_valid = cfg['train_params']['nb_samples_valid']

    data_loader = DataLoader(
            dataset_valid,
            num_workers=4,
            shuffle=False,
            batch_size=1,
            sampler=TrackPreferredSamplerValid(dataset_valid, num_samples=nb_samples_valid, tracking_samples_share=0.75)
        )

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_scale = dataset_valid.pred_scale

    with torch.set_grad_enabled(False):
        data_iter = tqdm(data_loader)
        for data in data_iter:
            image = data['image'].float().cuda()

            images = combine_images(data, model_params.input_frames)
            pred = model(images)

            w = float(data['center_item_w'][0])
            h = float(data['center_item_h'][0])
            distance = float(data['center_item_distance'][0])

            idx = data['idx'][0]
            print(f'\ncenter item size {w:.1f}x{h:.1f} distance {distance:.1f} idx {idx}')

            pred['mask'] = torch.sigmoid(pred['mask'])

            reg_size = data['reg_size'].float().numpy()[0]
            reg_distance = data['reg_distance'].float().numpy()[0]
            reg_offset = data['reg_offset'].float().numpy()[0]

            size = pred['size'][0].detach().cpu().numpy()
            distance = pred['distance'][0].detach().cpu().numpy()
            offset = pred['offset'][0].detach().cpu().numpy()

            output_binary_mask = model_params.get('output_binary_mask', False)

            pred_cls = pred['mask'][0, 0].detach().cpu().numpy()

            fig, ax = plt.subplots(2, 3, figsize=(24, 16))
            gs = ax[0, 0].get_gridspec()
            ax[0, 0].remove()
            ax[0, 1].remove()
            ax[1, 0].remove()
            ax[1, 1].remove()

            axbig = fig.add_subplot(gs[:, :2])
            axbig.imshow(image[0].cpu().numpy(), cmap='gray')

            ax[0, 2].imshow(2 - data['cls'][0, 0].cpu().numpy() - data['cls_planned'][0, 0].cpu().numpy(), vmin=0, vmax=2, cmap='gray')
            ax[1, 2].imshow(pred_cls, vmin=0, vmax=1, cmap='gray')


            # ax[0, 0].imshow(torch.stack([image_prev0, image, image_prev1], dim=3)[0].cpu().numpy())
            # ax[0, 0].imshow(image[0].cpu().numpy(), cmap='gray')
            # ax[0, 1].imshow(image[0].cpu().numpy(), cmap='gray')
            # ax[0, 2].imshow(image[0].cpu().numpy(), cmap='gray')
            #
            # ax[1, 0].imshow(1 - data['cls'][0, 0].cpu().numpy(), vmin=0, vmax=1, cmap='gray')
            # ax[1, 1].imshow(1 - data['cls_planned'][0, 0].cpu().numpy(), vmin=0, vmax=1, cmap='gray')
            # ax[1, 2].imshow(pred_cls, vmin=0, vmax=1, cmap='gray')

            # ax[1, 3].imshow(torch.sigmoid(pred['above_horizon'][0, 0]).detach().cpu().numpy(), vmin=0, vmax=1, cmap='gray')

            # ax[0, 3].imshow(image[0].cpu().numpy(), cmap='gray')
            # ax[0, 3].imshow(cv2.resize(1-data['cls'][0, 0].cpu().numpy(), None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST), vmin=0, vmax=1, alpha=0.2)

            # ax[1, 0].imshow(np.power(2, size[0]))
            # ax[1, 1].imshow(np.power(2, size[1]))
            # ax[1, 0].imshow(offset[0])
            # ax[1, 1].imshow(offset[1])
            # ax[1, 2].imshow(np.power(2, distance[0]))

            # ax[1, 3].imshow(image[0].cpu().numpy(), cmap='gray')
            # ax[1, 3].imshow(cv2.resize(pred_cls, None, fx=4.0, fy=4.0,
            #                            interpolation=cv2.INTER_NEAREST), vmin=0, vmax=1, alpha=0.2)
            #
            # ax[2, 0].imshow(np.power(2, reg_size[0]))
            # ax[2, 1].imshow(np.power(2, reg_size[1]))
            # # ax[2, 0].imshow(reg_offset[0])
            # # ax[2, 1].imshow(reg_offset[1])
            # ax[2, 2].imshow(np.power(2, reg_distance))

            mask = pred['mask'][0, 0].cpu().detach().numpy()
            pred = seg_prediction_to_items.pred_to_items(
                comb_pred=mask,
                offset=pred['offset'][0].cpu().detach().numpy(),
                size=pred['size'][0].cpu().detach().numpy(),
                tracking=pred['tracking'][0].cpu().detach().numpy(),
                distance=pred['distance'][0, 0].cpu().detach().numpy(),
                above_horizon=pred['above_horizon'][0, 0].cpu().detach().numpy(),
                conf_threshold=0.1,
                pred_scale=pred_scale,
                x_offset=-padding
            )

            # ax[0, 0].scatter([256], [256], c='g', s=1, label='gt')

            for p in pred:
                print(p)

                if p['conf'] > 0.1:
                    xc = p['cx'] + p['offset'][0] + padding
                    yc = p['cy'] + p['offset'][1]
                    w = p['w']
                    h = p['h']
                    rect = matplotlib.patches.Rectangle((xc-w/2, yc-h/2), w, h, linewidth=1, edgecolor='y', facecolor='none')
                    axbig.add_patch(rect)
                    axbig.text(xc-w/2, yc-h/2, f'{int(p["conf"] * 100)}', c='y')

            print()

            # if len(pred):
            #     p = pred[0]
            #     ax[0, 0].scatter([p['cx']], [p['cy']], c='b', s=1, label='pred')
            #     ax[0, 0].scatter([p['cx']+p['offset'][0]], [p['cy']+p['offset'][1]], c='r', s=1, label='pred+offset')
            #     ax[0, 0].legend()
            plt.show()


def check_center(experiment_name: str, fold: int, epoch: int):
    model_str = experiment_name
    cfg = common_utils.load_config_data(experiment_name)

    model_params = DotDict(cfg["model_params"])
    checkpoints_dir = f"../output/checkpoints/{model_str}/{fold}"
    print("\n", experiment_name, "\n")

    dataset_valid = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_VALID,
        cfg_data=cfg,
        return_torch_tensors=True,
        small_subset=False
    )

    dataset_valid.crop_with_plane_ratio = 1.0

    data_loader = DataLoader(
            dataset_valid,
            num_workers=16,
            shuffle=False,
            batch_size=8,
            sampler=ItemsWithPlaneSampler(dataset_valid)
        )

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    centers_single_point = []
    centers_single_point_with_offset = []

    centers_item_sizes = []

    with torch.set_grad_enabled(False):
        data_iter = tqdm(data_loader)
        for data in data_iter:
            image = data['image'].float().cuda()
            images = combine_images(data, model_params.input_frames)

            model_pred = model(images)
            model_pred['cls'] = torch.sigmoid(model_pred['cls'])

            for i in range(image.shape[0]):
                w = float(data['center_item_w'][i])
                h = float(data['center_item_h'][i])

                pred = seg_prediction_to_items.pred_to_items(
                    comb_pred=model_pred['mask'][i, 0].cpu().detach().numpy(),
                    offset=model_pred['offset'][i].cpu().detach().numpy(),
                    size=model_pred['size'][i].cpu().detach().numpy(),
                    tracking=model_pred['tracking'][i].cpu().detach().numpy(),
                    distance=model_pred['distance'][i, 0].cpu().detach().numpy(),
                    above_horizon=model_pred['above_horizon'][i, 0].cpu().detach().numpy(),
                    conf_threshold=0.25)

                if w > 0 and h > 0:
                    centers_item_sizes.append(np.array([w, h]))
                    if len(pred):
                        p = pred[0]

                        p_single_point = np.array([p['cx'] - 256, p['cy'] - 256])
                        p_single_point_with_offset = p_single_point + p['offset']

                        centers_single_point.append(p_single_point)
                        centers_single_point_with_offset.append(p_single_point_with_offset)
                    else:
                        centers_single_point.append(np.array([100, 100]))
                        centers_single_point_with_offset.append(np.array([100, 100]))

                if len(centers_single_point) % 10000 == 0:
                    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(24, 16))

                    centers_single_point_a = np.array(centers_single_point)
                    centers_single_point_with_offset_a = np.array(centers_single_point_with_offset)
                    centers_item_sizes_a = np.clip(np.array(centers_item_sizes, dtype=np.float32), 8, 1000)

                    mask = (np.abs(centers_single_point_with_offset_a/centers_item_sizes_a).max(axis=1) < 2)
                    print(mask.shape, centers_single_point_a.shape, mask.mean())

                    centers_single_point_a = centers_single_point_a[mask, :]
                    centers_single_point_with_offset_a = centers_single_point_with_offset_a[mask, :]
                    centers_item_sizes_a = centers_item_sizes_a[mask, :]

                    ax[0, 0].hist(centers_single_point_a[:, 0], bins=256, color='b', range=(-16, 16))
                    ax[0, 0].hist(centers_single_point_with_offset_a[:, 0], bins=256, color='r', alpha=0.5, range=(-16, 16))

                    ax[1, 0].hist(centers_single_point_a[:, 1], bins=256, color='b', range=(-16, 16))
                    ax[1, 0].hist(centers_single_point_with_offset_a[:, 1], bins=256, color='r', alpha=0.5, range=(-16, 16))

                    ax[0, 1].hist(centers_single_point_a[:, 0] / centers_item_sizes_a[:, 0], bins=256, color='b')
                    ax[0, 1].hist(centers_single_point_with_offset_a[:, 0] / centers_item_sizes_a[:, 0], bins=256, color='r', alpha=0.5)

                    ax[1, 1].hist(centers_single_point_a[:, 1] / centers_item_sizes_a[:, 1], bins=256, color='b')
                    ax[1, 1].hist(centers_single_point_with_offset_a[:, 1] / centers_item_sizes_a[:, 1], bins=256, color='r', alpha=0.5)

                    plt.show()


def check_sample_crops(experiment_name: str, fold: int, epoch_num: int):
    model_str = experiment_name
    cfg = common_utils.load_config_data(experiment_name)

    model_params = DotDict(cfg["model_params"])
    checkpoints_dir = f"../output/checkpoints/{model_str}/{fold}"

    fn = f"{checkpoints_dir}/fp_samples_{epoch_num:03}.pt"
    fp_samples = torch.load(fn)
    fpm_epochs = cfg['train_params'].get('fpm_epochs', {})

    dataset_train = dataset_tracking.TrackingDataset(
        stage=dataset_tracking.BaseDataset.STAGE_TRAIN,
        # parts=parts,
        cfg_data=cfg,
        return_torch_tensors=True,
        small_subset=False
    )

    # cfg_full_size = copy.deepcopy(cfg)
    # cfg_full_size['dataset_params']['crop_with_plane_ratio'] = 0
    # cfg_full_size['dataset_params']['crop_size'] = (2432, 2048)
    #
    # dataset_train_full_size = dataset_tracking.TrackingDataset(
    #     stage=dataset_tracking.BaseDataset.STAGE_TRAIN,
    #     # parts=parts,
    #     cfg_data=cfg_full_size,
    #     return_torch_tensors=True,
    #     small_subset=False
    # )
    # dataset_train_full_size.is_training = False  # disable augmentations etc

    print()
    print(f'Found fp samples: {len(fp_samples)} share {len(fp_samples) / (len(dataset_train) // fpm_epochs[epoch_num])}')
    dataset_train.set_fpm_samples(fp_samples)

    # data_loader = DataLoader(
    #     dataset_train,
    #     num_workers=cfg['train_data_loader']['num_workers'],
    #     shuffle=False,
    #     batch_size=8,
    #     sampler=TrackPreferredSamplerWithFP(
    #         dataset_train,
    #         num_samples=2000,
    #         fp_samples=fp_samples,
    #         tracking_samples_share=0.25,
    #         fp_samples_share=0.25
    #     )
    # )

    # nb_fpm = 0
    # nb_total = 0
    # for d in tqdm(data_loader):
    #     nb_total += len(d['is_fpm_sample'])
    #     nb_fpm += sum(d['is_fpm_sample'])
    #
    # print(f'fpm share: {nb_fpm}/{nb_total} = {nb_fpm/nb_total}')

    max_conf, frame_index, items = fp_samples[0]
    print(max_conf, frame_index, items)

    for i in range(128):
        data = dataset_train[frame_index if i < 32 else frame_index+1]
        print(data['idx'], data['crop_x'], data['crop_y'], data['is_fpm_sample'])

        plt.imshow(np.stack([
                data['prev_image_aligned0'].numpy(),
                data['image'].numpy(),
                data['prev_image_aligned0'].numpy()
            ], axis=2), vmin=0, vmax=1, cmap='gray')

        plt.show()



def export_model(experiment_name: str, fold: int, epoch: int):
    model_str = experiment_name

    cfg = common_utils.load_config_data(experiment_name)

    checkpoints_dir = f"../output/checkpoints/{model_str}/{fold}"
    print("\n", experiment_name, "\n")

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()

    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dst_dir = '../output/models'
    os.makedirs(dst_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict()
        },
        f'{dst_dir}/{experiment_name}_{epoch}.pth'
    )


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default="check")
    parser.add_argument("experiment", type=str, default="")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--sample", type=int, default=-1)

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

    if action == "check_center":
        check_center(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )

    if action == "check_find_fp_samples":
        check_find_fp_samples(
            experiment_name=experiment_name,
            epoch=args.epoch
        )

    if action == "check_sample_crops":
        check_sample_crops(
            experiment_name=experiment_name,
            fold=args.fold,
            epoch_num=args.epoch
        )

    if action == "export_model":
        export_model(
            experiment_name=experiment_name,
            epoch=args.epoch,
            fold=args.fold
        )
