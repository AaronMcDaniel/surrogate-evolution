import argparse
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from collections import defaultdict
import pickle
import re

import common_utils
from common_utils import DotDict, timeit_context
from typing import Iterator, Optional, Sequence, List, Dict, TypeVar, Generic, Sized
from dataclasses import dataclass
from load_images_pipe import *
import config
from config import MIN_OBJECT_AREA

enable_visualisation = False

@dataclass
class Box:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def left(self):
        return self.cx - self.w / 2

    @property
    def right(self):
        return self.cx + self.w / 2

    @property
    def top(self):
        return self.cy - self.h / 2

    @property
    def bottom(self):
        return self.cy + self.h / 2


@dataclass
class GtItem(Box):
    item_id: str
    distance: float
    matched_conf: float = 0.0


@dataclass
class DetectedItem(Box):
    item_id: str
    distance: float

    confidence: float
    dx: float
    dy: float

    is_matched_planned: bool = False
    is_matched_unplanned: bool = False
    matched_item_id: str = ''
    track_id: int = -1
    add_to_submit: bool = False


@dataclass
class FrameItems:
    predicted: List[DetectedItem]
    gt_planned: List[GtItem]
    gt_unplanned: List[GtItem]

    frame_img_fn: str = ''
    frame_img_prev_fn: str = ''

    transform_dx: float = 0
    transform_dy: float = 0
    transform_angle: float = 0
    transform_error: float = 0


def render_frame(frame_items: FrameItems):
    print(f'{frame_items.frame_img_fn} prev: {frame_items.frame_img_prev_fn}')
    img = cv2.imread(frame_items.frame_img_fn, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    ax = plt.gca()

    def add_rect(p, c):
        rect = matplotlib.patches.Rectangle((p.left, p.top), p.w, p.h, linewidth=1, edgecolor=c, facecolor='none')
        ax.add_patch(rect)

    for p in frame_items.gt_unplanned:
        add_rect(p, 'gray')

    for p in frame_items.gt_planned:
        add_rect(p, 'b')
        print(p.distance)

    for p in frame_items.predicted:
        if p.confidence < 0.25:
            continue
        if p.is_matched_planned:
            c = 'g'
        elif p.is_matched_unplanned:
            c = 'y'
        else:
            c = 'r'
        add_rect(p, c)
        plt.text(p.left, p.top, f'{int(p.confidence*100)}', c='yellow')

    plt.show()

def extend_bounding_boxes(item: Box):
    res = item

    if item.w * item.h > MIN_OBJECT_AREA:
        return res

    orig_aspect_ratio = item.w / item.h
    extended_width = np.sqrt(MIN_OBJECT_AREA * orig_aspect_ratio)
    extended_height = MIN_OBJECT_AREA / extended_width

    res.w = extended_width
    res.h = extended_height
    return res


def calc_iou(i1: Box, i2: Box):
    ix_min = max(i1.left, i2.left)
    iy_min = max(i1.top, i2.top)
    ix_max = min(i1.right, i2.right)
    iy_max = min(i1.bottom, i2.bottom)

    iw = max(ix_max - ix_min, 0.)
    ih = max(iy_max - iy_min, 0.)

    intersections = iw * ih
    unions = (i1.w * i1.h + i2.w * i2.h - intersections)

    iou = intersections / unions
    return iou


@dataclass
class FlightFrames:
    part: str
    flight_id: str
    frame_numbers: List[int]
    file_names: List[str]


def df_to_flight_frames(df, part) -> List[FlightFrames]:
    res = {}
    for _, row in df.iterrows():
        flight_id = row['flight_id']
        key = (part, flight_id)
        fn = f"{config.DATA_DIR}/{part}/Images/part{part}{flight_id}/{row['img_name'][:-4]}.png" #ADDED {part} in front of {flight_id} and changed.jpg to .png
        if key not in res:
            res[key] = FlightFrames(part, flight_id, [], [])
        res[key].frame_numbers.append(row['frame'])
        res[key].file_names.append(fn)

    return [res[k] for k in sorted(res.keys())]


def load_flights_for_part(part) -> List[FlightFrames]:
    cache_fn = f'{config.DATA_DIR}/flight_frames_{part}.csv'
    if os.path.exists(cache_fn):
        return pickle.load(open(cache_fn, 'rb'))

    df = pd.read_csv(f'{config.DATA_DIR}/{part}/ImageSets/groundtruth.csv')
    df_frames = df[['flight_id', 'img_name', 'frame']].drop_duplicates().reset_index(drop=True)
    res = df_to_flight_frames(df_frames, part)
    return res


def load_flights_for_time_based_validation() -> List[FlightFrames]:
    cache_fn = f'{config.DATA_DIR}/flight_frames_val.pkl'
    if os.path.exists(cache_fn):
        return pickle.load(open(cache_fn, 'rb'))

    train_val_ds = pd.read_csv(f'{config.DATA_DIR}/train_val_flights.csv')
    res = []

    for part in ['part1', 'part2', 'part3']:
        df = pd.read_csv(f'{config.DATA_DIR}/{part}/ImageSets/groundtruth.csv')
        train_val_ds_part = train_val_ds[train_val_ds['part'] == part].reset_index(drop=True)

        df_frames = df[['flight_id', 'img_name', 'frame']].drop_duplicates().reset_index(drop=True)
        df_frames = pd.merge(df_frames, train_val_ds_part[['flight_id', 'is_validation']], how='left', on='flight_id')
        df_frames = df_frames[df_frames['is_validation'] == True].reset_index(drop=True)
        res += df_to_flight_frames(df_frames, part)

    pickle.dump(res, open(cache_fn, 'wb'))
    return res


def check_frame_level_predictions(experiment_name: str, epoch=None, part='part1',
                                  preview=False, preview_fp=False,
                                  nb_flights=50,
                                  from_flight=0
                                  ):
    model_str = experiment_name
    if epoch is not None:
        oof_dir = f"../output/oof_cache/{model_str}_{epoch}"
    else:
        oof_dir = f"../output/oof_cache/{model_str}"

    df_by_part = {
        part: pd.read_csv(f'{config.DATA_DIR}/{part}/ImageSets/groundtruth.csv')
        for part in ['part1', 'part2', 'part3']
    }

    with common_utils.timeit_context('load df'):
        if part == 'val':
            flights: List[FlightFrames] = load_flights_for_time_based_validation()
        else:
            flights: List[FlightFrames] = load_flights_for_part(part)

    flights = flights[from_flight:from_flight+nb_flights]

    print(model_str, epoch)
    print(oof_dir)

    gt_planned_conf = []
    gt_unplanned_conf = []
    pred_match_planned_conf = []
    pred_match_unplanned_conf = []
    pred_unmatched_conf = []
    nb_frames = 0

    for flight in tqdm(flights):
        flight_id = flight.flight_id
        part = flight.part
        df = df_by_part[part]

        try:
            frame_predictions = pickle.load(open(f'{oof_dir}/{part}_{flight_id}.pkl', 'rb'))
        except FileNotFoundError:
            try:
                frame_predictions = pickle.load(open(f'{oof_dir}/{flight_id}.pkl', 'rb'))
            except FileNotFoundError:
                print(f'{oof_dir}/{part}_{flight_id}.pkl')
                break

        df_flight = df[df.flight_id == flight_id].reset_index(drop=True)
        df_flight_frames_with_files = df_flight[['img_name', 'frame']].drop_duplicates().reset_index(drop=True)
        df_flight_frames = df_flight_frames_with_files['frame'].values
        df_flight_image_names = df_flight_frames_with_files['img_name'].values

        x_pred_offset = (2448 - 2432) / 2
        frame_transforms = pd.read_pickle(f'{config.DATA_DIR}/frame_transforms/{part}/{flight_id}.pkl')

        frame_items: Dict[int, FrameItems] = {}

        for index, row in df_flight.iterrows():
            frame = row['frame']
            if frame not in frame_items:
                cur_frame_transform = frame_transforms[frame_transforms.frame == frame].iloc[0]
                frame_items[frame] = FrameItems(
                    predicted=[], gt_planned=[], gt_unplanned=[],
                    frame_img_fn=f'{config.DATA_DIR}/{part}/Images/part{part}{flight_id}/{row["img_name"][:-4]}.png', #ADDED {part} in front of {flight_id} and changed.jpg to .png
                    transform_dx=cur_frame_transform['dx'],
                    transform_dy=cur_frame_transform['dy'],
                    transform_angle=cur_frame_transform['angle'],
                    transform_error=cur_frame_transform['error']
                )

            if not isinstance(row['id'], str):
                continue

            left = row['gt_left']
            right = row['gt_right']
            top = row['gt_top']
            bottom = row['gt_bottom']

            item = GtItem(
                cx=(left + right) / 2,
                cy=(top + bottom) / 2,
                w=right - left,
                h=bottom - top,
                item_id=row['id'],
                distance=row['range_distance_m']
            )

            is_planned = not np.isnan(item.distance) and item.distance < 700
            item: GtItem = extend_bounding_boxes(item)

            if is_planned:
                frame_items[frame].gt_planned.append(item)
            else:
                frame_items[frame].gt_unplanned.append(item)

        for frame, detected_items, image_name in zip(df_flight_frames, frame_predictions, df_flight_image_names):
            cur_frame_items = frame_items[frame]
            for it in detected_items:
                item_det = DetectedItem(
                    item_id='Airborne',
                    cx=it['cx'] + x_pred_offset + it['offset'][0],
                    cy=it['cy'] + it['offset'][1],
                    # cx=it['cx'] + x_pred_offset,
                    # cy=it['cy'],
                    w=it['w'],
                    h=it['h'],
                    confidence=it['conf'],
                    distance=it['distance'],
                    dx=it['tracking'][0],
                    dy=it['tracking'][1],
                )

                # if it['above_horizon'] > 0:
                #     item_det.confidence = item_det.confidence ** 0.9
                # else:
                #     item_det.confidence = item_det.confidence ** 1.1

                if item_det.distance > 800:
                    continue

                # if item_det.cx < 80 and item_det.cy < 300:
                #     continue

                item_det: DetectedItem = extend_bounding_boxes(item_det)

                for item_gt in cur_frame_items.gt_planned:
                    if item_gt.matched_conf > 0:
                        continue

                    iou = calc_iou(item_det, item_gt)
                    if iou > config.IS_MATCH_MIN_IOU_THRESH:
                        item_gt.matched_conf = item_det.confidence
                        item_det.is_matched_planned = True
                        break

                    if iou > config.IS_NO_MATCH_MAX_IOU_THRESH:
                        item_det.is_matched_unplanned = True
                        break

                if not item_det.is_matched_planned and not item_det.is_matched_unplanned:
                    for item_gt in cur_frame_items.gt_unplanned:
                        if item_gt.matched_conf > 0:
                            continue

                        iou = calc_iou(item_det, item_gt)
                        if iou > config.IS_NO_MATCH_MAX_IOU_THRESH:
                            item_det.is_matched_unplanned = True
                            item_gt.matched_conf = item_det.confidence
                            break

                cur_frame_items.predicted.append(item_det)

        nb_displayed = 0
        for cur_frame_items in frame_items.values():
            has_fn = (len(cur_frame_items.gt_planned) and cur_frame_items.gt_planned[0].matched_conf < 0.7)
            has_fp = (len(cur_frame_items.predicted) and cur_frame_items.predicted[0].confidence > 0.7 and
                not (cur_frame_items.predicted[0].is_matched_planned or cur_frame_items.predicted[0].is_matched_unplanned))

            if has_fp and preview_fp and nb_displayed < 16:
                render_frame(cur_frame_items)
                nb_displayed += 1


            gt_planned_conf += [item_gt.matched_conf for item_gt in cur_frame_items.gt_planned]
            gt_unplanned_conf += [item_gt.matched_conf for item_gt in cur_frame_items.gt_unplanned]

            pred_match_planned_conf += [item_pred.confidence for item_pred in cur_frame_items.predicted
                                        if item_pred.is_matched_planned]
            pred_match_unplanned_conf += [item_pred.confidence for item_pred in cur_frame_items.predicted
                                          if item_pred.is_matched_unplanned]
            pred_unmatched_conf += [item_pred.confidence for item_pred in cur_frame_items.predicted
                                    if not item_pred.is_matched_unplanned and not item_pred.is_matched_planned]
            nb_frames += 1

    gt_planned_conf = np.array(sorted(gt_planned_conf))
    gt_unplanned_conf = np.array(sorted(gt_unplanned_conf))
    pred_match_planned_conf = np.array(sorted(pred_match_planned_conf))
    pred_match_unplanned_conf = np.array(sorted(pred_match_unplanned_conf))
    pred_unmatched_conf = np.array(sorted(pred_unmatched_conf))
    pred_unmatched_conf = pred_unmatched_conf[::-1]

    for threshold in [0.0002, 0.0003, 0.0004, 0.0005]:
        nb_allowed_pred_unmatched = int(nb_frames * threshold)
        conf_threshold = pred_unmatched_conf[nb_allowed_pred_unmatched]
        print(threshold)
        print('conf threshold:', conf_threshold)
        print('AFDR:', (gt_planned_conf > conf_threshold).mean()*100)
        print(f'{conf_threshold:0.3f} {(gt_planned_conf > conf_threshold).mean()*100:0.1f}')

    for conf_threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        print(f'{conf_threshold:0.2f} {(gt_planned_conf > conf_threshold).mean() * 100:0.1f}')

    if preview:
        plt.plot(np.arange(len(pred_unmatched_conf)) / nb_frames, pred_unmatched_conf, label='unmatched conf threshold')
        plt.axvline(0.0)
        plt.axvline(0.0005)
        plt.axhline(conf_threshold)
        plt.legend()
        plt.title(model_str)
        plt.figure()

        plt.plot(gt_planned_conf[::-1], np.arange(len(gt_planned_conf)) / len(gt_planned_conf), label='gt_planned_conf')
        plt.axvline(conf_threshold)
        plt.legend()

        plt.title(model_str)
        plt.show()

    #
    # plt.hist(gt_planned_conf, bins=100, label='gt planned')
    # plt.legend()
    # plt.figure()
    #
    # plt.hist(gt_unplanned_conf, bins=100, label='gt unplanned')
    # plt.legend()
    # plt.figure()
    #
    # plt.hist(pred_match_planned_conf, bins=100, label='match planned')
    # plt.legend()
    # plt.figure()
    #
    # plt.hist(pred_match_unplanned_conf, bins=100, label='match unplanned')
    # plt.legend()
    # plt.figure()
    #
    # plt.hist(pred_unmatched_conf, bins=100, label='not matched')
    # plt.legend()
    # plt.show()


def match_predictions_to_gt(experiment_name: str,
                            epoch=None,
                            part='part1',
                            nb_flights=50,
                            from_flight=0):

    model_str = experiment_name
    if epoch is not None:
        oof_dir = f"../output/oof_cache/{model_str}_{epoch}"
    else:
        oof_dir = f"../output/oof_cache/{model_str}"

    df_by_part = {
        part: pd.read_csv(f'{config.DATA_DIR}/{part}/ImageSets/groundtruth.csv')
        for part in ['part1', 'part2', 'part3']
    }

    with common_utils.timeit_context('load df'):
        if part == 'val':
            flights: List[FlightFrames] = load_flights_for_time_based_validation()
        else:
            flights: List[FlightFrames] = load_flights_for_part(part)

    flights = flights[from_flight:from_flight + nb_flights]

    print(model_str, epoch)
    print(oof_dir)

    all_flights_frame_items = {}

    for flight in tqdm(flights):
        flight_id = flight.flight_id
        part = flight.part
        df = df_by_part[part]

        try:
            frame_predictions = pickle.load(open(f'{oof_dir}/{part}_{flight_id}.pkl', 'rb'))
        except FileNotFoundError:
            try:
                frame_predictions = pickle.load(open(f'{oof_dir}/{flight_id}.pkl', 'rb'))
            except FileNotFoundError:
                print(f'{oof_dir}/{part}_{flight_id}.pkl')
                break

        df_flight = df[df.flight_id == flight_id].reset_index(drop=True)
        df_flight_frames_with_files = df_flight[['img_name', 'frame']].drop_duplicates().reset_index(drop=True)
        df_flight_frames = df_flight_frames_with_files['frame'].values
        df_flight_image_names = df_flight_frames_with_files['img_name'].values

        x_pred_offset = (2448 - 2432) / 2
        frame_transforms = pd.read_pickle(f'{config.DATA_DIR}/frame_transforms/{part}/{flight_id}.pkl')

        frame_items: Dict[int, FrameItems] = {}
        all_flights_frame_items[(part, flight_id)] = frame_items

        for index, row in df_flight.iterrows():
            frame = row['frame']
            if frame not in frame_items:
                cur_frame_transform = frame_transforms[frame_transforms.frame == frame].iloc[0]
                frame_items[frame] = FrameItems(
                    predicted=[], gt_planned=[], gt_unplanned=[],
                    frame_img_fn=f'{config.DATA_DIR}/{part}/Images/part{part}{flight_id}/{row["img_name"][:-4]}.png', #ADDED {part} in front of {flight_id} and changed.jpg to .png
                    transform_dx=cur_frame_transform['dx'],
                    transform_dy=cur_frame_transform['dy'],
                    transform_angle=cur_frame_transform['angle'],
                    transform_error=cur_frame_transform['error']
                )

            if not isinstance(row['id'], str):
                continue

            left = row['gt_left']
            right = row['gt_right']
            top = row['gt_top']
            bottom = row['gt_bottom']

            item = GtItem(
                cx=(left + right) / 2,
                cy=(top + bottom) / 2,
                w=right - left,
                h=bottom - top,
                item_id=row['id'],
                distance=row['range_distance_m']
            )

            is_planned = not np.isnan(item.distance) and item.distance < 700
            item: GtItem = extend_bounding_boxes(item)

            if is_planned:
                frame_items[frame].gt_planned.append(item)
            else:
                frame_items[frame].gt_unplanned.append(item)

        for frame, detected_items, image_name in zip(df_flight_frames, frame_predictions, df_flight_image_names):
            cur_frame_items = frame_items[frame]
            for it in detected_items:
                item_det = DetectedItem(
                    item_id='Airborne',
                    cx=it['cx'] + x_pred_offset + it['offset'][0],
                    cy=it['cy'] + it['offset'][1],
                    w=it['w'],
                    h=it['h'],
                    confidence=it['conf'],
                    distance=it['distance'],
                    dx=it['tracking'][0],
                    dy=it['tracking'][1],
                )

                # if it['above_horizon'] > 0:
                #     item_det.confidence = item_det.confidence ** 0.9
                # else:
                #     item_det.confidence = item_det.confidence ** 1.1

                if item_det.distance > 950:
                    continue

                item_det: DetectedItem = extend_bounding_boxes(item_det)

                for item_gt in cur_frame_items.gt_planned:
                    if item_gt.matched_conf > 0:
                        continue

                    iou = calc_iou(item_det, item_gt)
                    if iou > config.IS_MATCH_MIN_IOU_THRESH:
                        item_gt.matched_conf = item_det.confidence
                        item_det.is_matched_planned = True
                        item_det.matched_item_id = item_gt.item_id
                        break

                    if iou > config.IS_NO_MATCH_MAX_IOU_THRESH:
                        item_det.is_matched_unplanned = True
                        item_det.matched_item_id = item_gt.item_id
                        break

                if not item_det.is_matched_planned and not item_det.is_matched_unplanned:
                    for item_gt in cur_frame_items.gt_unplanned:
                        if item_gt.matched_conf > 0:
                            continue

                        iou = calc_iou(item_det, item_gt)
                        if iou > config.IS_NO_MATCH_MAX_IOU_THRESH:
                            item_det.is_matched_unplanned = True
                            item_gt.matched_conf = item_det.confidence
                            item_det.matched_item_id = item_gt.item_id
                            break

                cur_frame_items.predicted.append(item_det)

    return all_flights_frame_items


def check_score():
    check_frame_level_predictions(
        '130_gernet_m_b2_only_planned', 2220,
        # part='part1',
        part='val',
        preview_fp=False,
        preview=True,
        nb_flights=572,
    )


def save_matched_predictions():
    res2 = match_predictions_to_gt(
        '100_mix6_2',
        part='part1',
        nb_flights=1311
    )
    pickle.dump(res2, open(f'{config.DATA_DIR}/pred_matched_oof_100_mix6_2.pkl', 'wb'))


if __name__ == "__main__":
    check_score()
    # save_matched_predictions()
