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
import loss_asl
import pickle
from dataclasses import dataclass

import common_utils
import dataset_tracking
import seg_prediction_to_items
from dataset_tracking import Frame, DetectionItem
import models_segmentation
from common_utils import DotDict, timeit_context

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized

from train import seed_everything, build_model, combine_images
from load_images_pipe import *
import config
import scipy.stats.mstats
import check_frame_level_prediction
from check_frame_level_prediction import FlightFrames, load_flights_for_part, load_flights_for_time_based_validation

seed_everything(seed=42)

enable_visualisation = False


def predict_detections(model, frame_numbers, images, flight_transforms, cache_dir, back_steps):
    assert back_steps[0] == 1
    max_back_steps = max(back_steps)
    batch_size = len(images) - max_back_steps
    h, w = images[-1].shape
    X = np.zeros((batch_size, len(back_steps)+1, h, w), dtype=np.uint8)

    # dx = prev_img_transforms['dx'].values
    # dy = prev_img_transforms['dy'].values
    # angle = prev_img_transforms['angle'].values
    # print(dx, dy, angle)

    cache_fn = f'{cache_dir}/{frame_numbers[0]}_{frame_numbers[-1]}.npz'

    for i in range(batch_size):
        prev_step_transforms = {}

        for back_step in range(1, max_back_steps+1):
            cur_frame = frame_numbers[i] - back_step + 1
            flight_transforms_row = flight_transforms[flight_transforms.frame == cur_frame]
            if len(flight_transforms_row):
                dx = flight_transforms_row.iloc[0]["dx"]
                dy = flight_transforms_row.iloc[0]["dy"]
                angle = flight_transforms_row.iloc[0]["angle"]
            else:
                dx = 0
                dy = 0
                angle = 0

            # print(back_step, dx, dy, angle)
            prev_tr = common_utils.build_geom_transform(
                dst_w=w,
                dst_h=h,
                src_center_x=w / 2 + dx,
                src_center_y=h / 2 + dy,
                scale_x=1.0,
                scale_y=1.0,
                angle=angle,
                return_params=True
            )

            if back_step > 1:
                prev_tr = prev_step_transforms[back_step-1] @ prev_tr
            prev_step_transforms[back_step] = prev_tr

        prev_img_aligned = cv2.warpAffine(
            images[i + max_back_steps - 1],
            prev_step_transforms[1][:2, :],  # the same crop as with the cur image
            dsize=(w, h),
            flags=cv2.INTER_LINEAR)

        X[i, 0, :, :] = prev_img_aligned
        X[i, 1, :, :] = images[i + max_back_steps]

        for bsi, back_step in enumerate(back_steps[1:]):
            prev_img_aligned = cv2.warpAffine(
                images[i + max_back_steps - back_step],
                prev_step_transforms[back_step][:2, :],  # the same crop as with the cur image
                dsize=(w, h),
                flags=cv2.INTER_LINEAR)

            X[i, bsi+2, :, :] = prev_img_aligned

        # common_utils.print_stats('x', np.transpose(X[i, :3, :, :], (1, 2, 0)))
        # plt.imshow(np.transpose(X[i, :3, :, :], (1, 2, 0)))
        # plt.figure()
        #
        # plt.imshow(np.stack([images[i + max_back_steps-1], images[i + max_back_steps], images[i + max_back_steps - 2]], axis=2))
        # plt.show()

    X = torch.from_numpy(X).cuda()
    X = X.float() / 255.0

    with torch.cuda.amp.autocast():
        pred = model(X)

    pred['mask'] = torch.sigmoid(pred['mask'])

    for k in list(pred.keys()):
        pred[k] = pred[k].cpu().detach().numpy()

    mask_mul_full = (pred['mask'] > 0.05).astype(np.float32)
    mask_mul = (pred['mask'] > 0.1).astype(np.float32)

    np.savez_compressed(
        cache_fn,
        mask=(pred['mask'] * mask_mul_full).astype(np.float16),
        offset=(pred['offset'] * mask_mul).astype(np.float16),
        size=(pred['size'] * mask_mul).astype(np.float16),
        tracking=(pred['tracking'] * mask_mul).astype(np.float16),
        distance=(pred['distance'] * mask_mul).astype(np.float16),
        above_horizon=(pred['above_horizon'] * mask_mul).astype(np.float16)
    )


def detect_objects(frame_numbers, cache_dir):
    batch_size = len(frame_numbers) - 1

    cache_fn = f'{cache_dir}/{frame_numbers[0]}_{frame_numbers[-1]}.npz'

    try:
        data = np.load(cache_fn)
        pred = {
            k: data[k].astype(np.float32) for k in ['mask', 'offset', 'size', 'tracking', 'distance', 'above_horizon']
        }
    except Exception as e:
        print(f'failed to load \n{cache_fn}\n')
        raise e

    res = []

    for i in range(batch_size):
        comb_pred = pred['mask'][i, 0]
        detected_objects = seg_prediction_to_items.pred_to_items(
            comb_pred=comb_pred,
            # cls=comb_pred[None, :, :],
            offset=pred['offset'][i],
            size=pred['size'][i],
            tracking=pred['tracking'][i],
            distance=pred['distance'][i, 0],
            above_horizon=pred['above_horizon'][i, 0],
            conf_threshold=0.1,
            pred_scale=8
        )

        res.append(detected_objects)
    return res


def detect_objects_from_ensemble(frame_numbers, cache_dirs):
    batch_size = len(frame_numbers) - 1

    data_ens = [np.load(f'{cache_dir}/{frame_numbers[0]}_{frame_numbers[-1]}.npz') for cache_dir in cache_dirs]

    pred = {
        k: data_ens[0][k].astype(np.float32) for k in ['offset', 'size', 'tracking', 'distance', 'above_horizon']
    }

    # pred['mask'] = scipy.stats.mstats.gmean([d['mask'].astype(np.float32)+0.05 for d in data_ens], axis=0) - 0.05
    pred['mask'] = np.array([d['mask'].astype(np.float32) for d in data_ens]).mean(axis=0)
    #
    # print(pred['mask'].shape, data_ens[0]['mask'].shape)

    res = []

    for i in range(batch_size):
        comb_pred = pred['mask'][i, 0]
        detected_objects = seg_prediction_to_items.pred_to_items(
            comb_pred=comb_pred,
            # cls=comb_pred[None, :, :],
            offset=pred['offset'][i],
            size=pred['size'][i],
            tracking=pred['tracking'][i],
            distance=pred['distance'][i, 0],
            above_horizon=pred['above_horizon'][i, 0],
            conf_threshold=0.1,
            pred_scale=8
        )

        res.append(detected_objects)
    return res


def save_predictions(experiment_name: str, fold: int, epoch: int, part='part1', nb_flights=50, from_flight=0, step=1):
    model_str = experiment_name
    cfg = common_utils.load_config_data(experiment_name)
    back_steps = cfg['dataset_params']['back_steps']
    max_back_step = max(back_steps)

    checkpoints_dir = f"../output/checkpoints/{model_str}/{fold}"
    oof_dir = f"../output/oof_cache/{model_str}_{epoch}"
    os.makedirs(oof_dir, exist_ok=True)
    print("\n", experiment_name, epoch, "\n")

    batch_size = 4
    torch.set_grad_enabled(False)

    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.eval()

    print(f"{checkpoints_dir}/{epoch:03}.pt")
    checkpoint = torch.load(f"{checkpoints_dir}/{epoch:03}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    if part == 'val':
        flights: List[FlightFrames] = load_flights_for_time_based_validation()
    else:
        flights: List[FlightFrames] = load_flights_for_part(part)

    flights = flights[from_flight:from_flight+nb_flights:step]

    load_img_stage = mpipe.OrderedStage(load_img, 1)
    decode_img_stage = mpipe.OrderedStage(decode_img, 16)
    load_img_stage.link(decode_img_stage)
    pipe = mpipe.Pipeline(load_img_stage)

    for flight in tqdm(flights):
        flight_id = flight.flight_id
        part = flight.part
        file_names = flight.file_names
        frame_numbers = flight.frame_numbers

        cache_dir = f'{oof_dir}/cache/{part}_{flight_id}'
        os.makedirs(cache_dir, exist_ok=True)

        dst_fn = f'{oof_dir}/{part}_{flight_id}.pkl'
        if os.path.exists(dst_fn):
            continue
        frame_transforms = pd.read_pickle(f'{config.DATA_DIR}/frame_transforms/{part}/{flight_id}.pkl')

        crop_w = 2432
        crop_h = 2048

        last_img = np.zeros((crop_h, crop_w), dtype=np.uint8)

        load_failed = []

        frame_predictions = []

        images = []  # not processed images yet of one frame + batch size

        for img_num, (fn, img) in tqdm(enumerate(limited_pipe(pipe, file_names, 16)), total=len(file_names)):
            load_failed.append(img is None)
            if img is None:
                img = last_img
            last_img = img

            h, w = img.shape

            y0 = (h - crop_h) // 2
            x0 = (w - crop_w) // 2
            img_crop = img[y0:y0 + crop_h, x0: x0 + crop_w]

            images.append(img_crop)

            if img_num == 0:
                for i in range(max_back_step):
                    images.append(img_crop)  # add the same image as prev frame for the first time

            if len(images) >= batch_size+max_back_step:
                predict_detections(
                    model,
                    frame_numbers=frame_numbers[len(frame_predictions): len(frame_predictions) + batch_size + 1],
                    images=images,
                    flight_transforms=frame_transforms,
                    cache_dir=cache_dir,
                    back_steps=back_steps
                )

                frame_predictions += detect_objects(frame_numbers=frame_numbers[len(frame_predictions): len(frame_predictions) + batch_size + 1],
                                                    cache_dir=cache_dir)

                images = images[-max_back_step:]

        if len(images) > max_back_step:
            predict_detections(
                model,
                frame_numbers=frame_numbers[len(frame_predictions): len(frame_predictions) + len(images)],
                images=images,
                flight_transforms=frame_transforms,
                cache_dir=cache_dir,
                back_steps=back_steps
            )

            frame_predictions += detect_objects(frame_numbers=frame_numbers[len(frame_predictions): len(frame_predictions) + len(images)],
                                                cache_dir=cache_dir)

        del images

        pickle.dump(frame_predictions, open(dst_fn, 'wb'))

    pipe.put(None)



def save_predictions_from_cache(experiment_name: str, fold: int, epoch: int, part='part1', nb_flights=50, from_flight=0):
    model_str = experiment_name
    cfg = common_utils.load_config_data(experiment_name)

    checkpoints_dir = f"../output/checkpoints/{model_str}/{fold}"
    oof_dir_cache = f"../output/oof_cache/{model_str}_{epoch}"
    oof_dir_out = f"../output/oof_cache_filt1/{model_str}_{epoch}"
    os.makedirs(oof_dir_out, exist_ok=True)
    print("\n", experiment_name, "\n")
    print(oof_dir_cache)
    print(oof_dir_out)

    batch_size = 4
    torch.set_grad_enabled(False)

    if part == 'val':
        flights: List[FlightFrames] = load_flights_for_time_based_validation()
    else:
        flights: List[FlightFrames] = load_flights_for_part(part)

    flights = flights[from_flight:from_flight + nb_flights]


    for flight in tqdm(flights):
        flight_id = flight.flight_id
        part = flight.part
        file_names = flight.file_names
        frame_numbers = flight.frame_numbers

        cache_dir = f'{oof_dir_cache}/cache/{part}_{flight_id}'
        if not os.path.exists(cache_dir):
            cache_dir = f'{oof_dir_cache}/cache/{flight_id}'

        crop_w = 2432
        crop_h = 2048

        last_img = np.zeros((crop_h, crop_w), dtype=np.uint8)

        frame_predictions = []

        images = []  # not processed images yet of one frame + batch size

        for fn in tqdm(file_names):
            images.append(last_img)
            if len(images) == 1:
                images.append(last_img)  # add the same image as prev frame for the first time

            if len(images) >= batch_size+1:
                frame_predictions += detect_objects(
                    frame_numbers=frame_numbers[len(frame_predictions): len(frame_predictions) + len(images)],
                    cache_dir=cache_dir)
                images = [images[-1]]

        if len(images) > 1:
            frame_predictions += detect_objects(frame_numbers=frame_numbers[len(frame_predictions): len(frame_predictions) + len(images)],
                                                cache_dir=cache_dir)

        del images

        pickle.dump(frame_predictions, open(f'{oof_dir_out}/{flight_id}.pkl', 'wb'))


def save_predictions_from_ensemble(experiments: list, part='part1', experiment_name=None, nb_flights=50):
    if experiment_name is None:
        experiment_name = experiments[0][0] + f'_ens_{experiments[0][1]}'
    model_str = experiment_name

    oof_dst_dir = f"../output/oof_cache/{model_str}"
    os.makedirs(oof_dst_dir, exist_ok=True)

    oof_dirs = [f"../output/oof_cache/{experiment[0]}_{experiment[1]}" for experiment in experiments]
    print("\n", experiment_name, "\n")
    print(oof_dst_dir)

    batch_size = 4

    if part == 'val':
        flights: List[FlightFrames] = load_flights_for_time_based_validation()
    else:
        flights: List[FlightFrames] = load_flights_for_part(part)

    flights = flights[:nb_flights]


    for flight in tqdm(flights):
        flight_id = flight.flight_id
        part = flight.part
        file_names = flight.file_names
        frame_numbers = flight.frame_numbers

        cache_dirs = [f'{oof_dir}/cache/{part}_{flight_id}' for oof_dir in oof_dirs]
        if not os.path.exists(cache_dirs[0]):
            cache_dirs = [f'{oof_dir}/cache/{flight_id}' for oof_dir in oof_dirs]

        crop_w = 2432
        crop_h = 2048

        last_img = np.zeros((crop_h, crop_w), dtype=np.uint8)

        frame_predictions = []

        images = []  # not processed images yet of one frame + batch size

        for fn in tqdm(file_names):
            images.append(last_img)
            if len(images) == 1:
                images.append(last_img)  # add the same image as prev frame for the first time

            if len(images) >= batch_size + 1:
                frame_predictions += detect_objects_from_ensemble(
                    frame_numbers=frame_numbers[len(frame_predictions): len(frame_predictions) + len(images)],
                    cache_dirs=cache_dirs)
                images = [images[-1]]

        if len(images) > 1:
            frame_predictions += detect_objects_from_ensemble(frame_numbers=frame_numbers[len(frame_predictions): len(frame_predictions) + len(images)],
                                                cache_dirs=cache_dirs)

        del images

        pickle.dump(frame_predictions, open(f'{oof_dst_dir}/{part}_{flight_id}.pkl', 'wb'))



if __name__ == "__main__":

    # to generate predictions from the ensemble, after predicting individual models:
    # save_predictions_from_ensemble(
    #     experiments=[
    #         ('130_hrnet32_all', 2220),
    #         ('130_hrnet48_all', 2220),
    #         ('130_gernet_m_b2_all', 2220),
    #         ('130_dla60_256_sgd', 1100),
    #     ],
    #     part='val',
    #     experiment_name='130_mix6_10',
    #     nb_flights=1311
    # )
    # check_frame_level_prediction.check_frame_level_predictions('130_mix6_10', nb_flights=1311, part='val')


    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default="predict")
    parser.add_argument("experiment", type=str, default="")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=2220)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--flights", type=int, default=2000)
    parser.add_argument("--from_flight", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--part", type=str, default='val')

    args = parser.parse_args()
    action = args.action
    experiment_name = common_utils.normalize_experiment_name(args.experiment)

    if action == "predict":
        save_predictions(
            experiment_name=experiment_name,
            fold=0,
            epoch=args.epoch,
            part=args.part,
            nb_flights=args.flights,
            from_flight=args.from_flight,
            step=args.step
        )
        check_frame_level_prediction.check_frame_level_predictions(
            experiment_name=experiment_name,
            epoch=args.epoch,
            part=args.part,
            nb_flights=args.flights
        )

    if action == "predict_from_cache":
        save_predictions_from_cache(
            experiment_name=experiment_name,
            fold=0,
            epoch=args.epoch,
            part=args.part,
            nb_flights=args.flights,
            from_flight=args.from_flight
        )
        check_frame_level_prediction.check_frame_level_predictions(
            experiment_name=experiment_name,
            epoch=args.epoch,
            part=args.part,
            nb_flights=args.flights
        )

