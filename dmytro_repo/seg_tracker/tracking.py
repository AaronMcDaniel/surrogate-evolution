import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import copy
import json
import math


import config
import common_utils
from prediction_structure import *


def build_transform(frame_items):
    w = 2448
    h = 2048

    prev_tr = common_utils.build_geom_transform(
        dst_w=w,
        dst_h=h,
        src_center_x=w / 2 + frame_items.transform_dx,
        src_center_y=h / 2 + frame_items.transform_dy,
        scale_x=1.0,
        scale_y=1.0,
        angle=frame_items.transform_angle,
        return_params=True
    )

    return prev_tr


def track_items_trivial(flight_items):
    for frame_num, frame_items in flight_items.items():
        nb_frame_detections = 0
        for detected_object in frame_items.predicted:
            detected_object.add_to_submit = False

            if detected_object.distance > 800:
                continue
            if detected_object.confidence < 0.8:
                continue

            detected_object.add_to_submit = True
            detected_object.track_id = nb_frame_detections + 1
            nb_frame_detections += 1

    return flight_items


class SimpleOffsetTracker:
    def __init__(self,
                 min_track_size=8,
                 threshold_to_find=0, #0.8,
                 threshold_to_continue=0, #0.7,
                 threshold_distance=32,
                 min_distance=850):
        self.prev_frames_items: List[List[DetectedItem]] = [[]]
        self.min_track_size = min_track_size
        self.threshold_to_find = threshold_to_find
        self.threshold_to_continue = threshold_to_continue
        self.threshold_distance = threshold_distance
        self.track_ids_in_use = set()
        self.min_distance = min_distance

    def distance(self, cur_item: DetectedItem, prev_item: DetectedItem, frames_offset):
        diag = (cur_item.w ** 2 + cur_item.h ** 2) ** 0.5

        # allow some level of uncertainty for longer distance prediction
        predict_distance = ((cur_item.dx * frames_offset) ** 2 + (cur_item.dx * frames_offset) ** 2) ** 0.5

        diag32 = max(32.0, diag, predict_distance * 0.5) / 32.0
        # diag32_prev = max(32, (prev_item.w ** 2 + prev_item.h ** 2) ** 0.5) / 32.0
        diff_w = abs(math.log2(max(16, cur_item.w) / max(16, prev_item.w)))
        diff_h = abs(math.log2(max(16, cur_item.h) / max(16, prev_item.h)))

        offset_x = cur_item.cx + cur_item.dx * frames_offset - prev_item.cx
        offset_y = cur_item.cy + cur_item.dy * frames_offset - prev_item.cy

        offset = (offset_x**2 + offset_y**2) ** 0.5

        res = offset / diag32 + diff_w * 16 + diff_h * 16 + (1.0 - prev_item.confidence) * 8
        return res

    def allocate_track_id(self):
        for i in range(1, 64):
            if i not in self.track_ids_in_use:
                self.track_ids_in_use.add(i)
                return i
        return 1

    def process_frame_detections(self, detections: List[DetectedItem], transform: np.ndarray) -> List[DetectedItem]:
        # project previous items to align with the current frame
        for prev_frame_items in self.prev_frames_items:
            prev_items_pos = np.array([[it.cx, it.cy] for it in prev_frame_items])
            if len(prev_items_pos) == 0:
                continue

            prev_items_pos_projected = (transform[:2, :2] @ prev_items_pos.T).T + transform[:2, 2]

            for i, item in enumerate(prev_frame_items):
                item.cx, item.cy = prev_items_pos_projected[i, :]

        items_to_keep = []  # to be added to history

        detections = copy.deepcopy(detections)
        for cur_item in detections:
            if cur_item.confidence < self.threshold_to_continue:
                continue

            if cur_item.distance > self.min_distance:
                continue

            cur_item.prev_item_idx = None
            cur_item.next_item_idx = None
            cur_item.track_id = -1
            cur_item.items_in_track = 1

            min_distance = 1e10
            prev_item_distances = []

            # check up to 3 frames back
            for frames_offset in [1, 2, 3]:
                if len(self.prev_frames_items) < frames_offset:
                    break

                prev_item_distances = [
                    self.distance(cur_item, prev_item, frames_offset) if prev_item.next_item_idx is None else 1000
                    for prev_item in self.prev_frames_items[-frames_offset]
                ]

                min_distance = min(prev_item_distances) if prev_item_distances else 1e10
                if min_distance < self.threshold_distance * frames_offset:
                    break

            if min_distance < self.threshold_distance * frames_offset:
                prev_item_idx = int(np.nanargmin(prev_item_distances))
                prev_item = self.prev_frames_items[-frames_offset][prev_item_idx]
                prev_item.next_item_idx = (frames_offset, len(items_to_keep))
                cur_item.prev_item_idx = (-frames_offset, prev_item_idx)

                if cur_item.confidence > self.threshold_to_find:
                    cur_item.items_in_track = prev_item.items_in_track + 1
                else:
                    cur_item.items_in_track = prev_item.items_in_track

                cur_item.track_id = prev_item.track_id

                if cur_item.items_in_track > self.min_track_size:
                    cur_item.add_to_submit = True

                    if cur_item.track_id < 0:
                        cur_item.track_id = self.allocate_track_id()

            items_to_keep.append(cur_item)

        self.prev_frames_items.append(items_to_keep)

        self.prev_frames_items = self.prev_frames_items[-3:]  # only keep the last 3 frames in history

        self.track_ids_in_use = {cur_item.track_id for cur_item in sum(self.prev_frames_items, [])}

        return [cur_item for cur_item in items_to_keep if cur_item.add_to_submit]


def track_items_offset(flight_items):
    tracker = SimpleOffsetTracker()

    res = {}
    for frame_num, frame_items in flight_items.items():
        transform = build_transform(frame_items)
        items_to_submit = tracker.process_frame_detections(frame_items.predicted, transform)
        frame_items.predicted = copy.deepcopy(items_to_submit)
        res[frame_num] = frame_items

    return res


def prepare_submission_json(all_items):
    res = []
    for (part, flight_id), flight_items in all_items.items():
        for frame_num, frame_items in flight_items.items():
            frame_detections = []
            for detected_object in frame_items.predicted:
                if detected_object.add_to_submit:
                    frame_detections.append(dict(
                        x=float(detected_object.cx),
                        y=float(detected_object.cy),
                        w=float(detected_object.w),
                        h=float(detected_object.h),
                        track_id=detected_object.track_id,
                        s=float(detected_object.confidence),
                        n='airborne'
                    ))
            res.append({
                'img_name': frame_items.frame_img_fn.split('/')[-1][:-3] + 'png',
                'detections': frame_detections
            })
    return res


def run_tracking():
    with common_utils.timeit_context('load data'):
        matched_items = pickle.load(open(f'{config.DATA_DIR}/pred_matched_oof_100_mix6_2.pkl', 'rb'))

    matched_items_processed = {}
    with common_utils.timeit_context('process tracking'):
        for (part, flight_id), flight_items in tqdm(matched_items.items()):
            # matched_items_processed[(part, flight_id)] = track_items_trivial(flight_items)
            matched_items_processed[(part, flight_id)] = track_items_offset(flight_items)

    with common_utils.timeit_context('generate results'):
        results_pred = prepare_submission_json(matched_items_processed)

    with common_utils.timeit_context('save results'):
        os.makedirs('../output/pred_tracking/track_offset/', exist_ok=True)
        with open("../output/pred_tracking/track_offset/result_mix_6.2_offset_0.8-0.7-d32-n8-3steps_v2.json", 'w') as fp:
            json.dump(results_pred, fp)
        # with open("../output/pred_tracking/track_offset/result_trivial_0.8.json", 'w') as fp:
        #     json.dump(results_pred, fp)

    # check score with
    # python core/metrics/run_airborne_metrics.py --dataset-folder data/evaluation/gt_part1 --results-folder ../output/pred_tracking/track_offset --summaries-folder ../output/pred_tracking/summaries_offset


if __name__ == '__main__':
    run_tracking()
