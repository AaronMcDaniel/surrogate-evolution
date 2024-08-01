# This file is the entrypoint for your submission.
# You can modify this file to include your code or directly call your functions/modules from here.
import random
import cv2
from evaluator.airborne_detection import AirbornePredictor

from tqdm import tqdm

import sys
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.patches
import time

from seg_tracker import config as config
from typing import List, Dict
from seg_tracker.prediction_structure import DetectedItem
import numpy as np
import random
import torch
import sys
import pickle
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, '/gv1/projects/GRIP_Precog_Opt')
import importlib

Dataset = importlib.import_module('data_loading.airborne-detection-starter-kit-master.data.dataset')

seed = 42
random.seed(seed)

current_path = os.getcwd()
sys.path.append(f'{current_path}/seg_tracker')

from seg_tracker.seg_tracker import SegTracker, SegDetector, SegTrackerFromOffset

MIN_TRACK_LEN = 30
MIN_SCORE = 0.985


class SegPredictor(AirbornePredictor):
    training_data_path = None
    test_data_path = None
    vocabulary_path = None

    """
    PARTICIPANT_TODO:
    You can do any preprocessing required for your codebase here like loading up models into memory, etc.
    """
    def inference_setup(self, job_id=None):
        self.detector = SegDetector(job_id=job_id)
        if config.DETECTOR_ONLY != True:
            self.tracker = SegTrackerFromOffset(detector=self.detector)
        self.visualize_pred = False

    def get_all_frame_images(self, flight_id):
        frames = []
        flight_folder = join(self.test_data_path, flight_id)
        for frame in sorted(listdir(flight_folder)):
            if isfile(join(flight_folder, frame)):
                frames.append(frame)
        return frames

    def flight_started(self):
        self.track_id_results = {}
        self.visited_frame = {}
        self.track_len_so_far = {}
        self.tracker = SegTrackerFromOffset(detector=self.detector)

    def proxy_register_object_and_location(self, class_name, track_id, bbox, confidence, img_name):
        # MIN_TRACK_LEN check
        if track_id not in self.track_len_so_far:
            self.track_len_so_far[track_id] = 0
        self.track_len_so_far[track_id] += 1
        if self.track_len_so_far[track_id] <= MIN_TRACK_LEN:
            return

        # MIN_SCORE check
        if confidence < MIN_SCORE:
            return

        if img_name not in self.visited_frame:
            self.visited_frame[img_name] = []
        if track_id in self.visited_frame[img_name]:
            raise Exception('two entities  within the same frame {} have the same track id'.format(img_name))
        self.visited_frame[img_name].append(track_id)

        self.register_object_and_location(class_name, track_id, bbox, confidence, img_name)

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
    
    def predict(self, image, prev_image):
        if config.DETECTOR_ONLY == False:
            detected_objects, prev_tr = self.detector.detect_objects(image=image, prev_image=prev_image)
        else:
            detected_objects = self.detector.detect_objects(image=image, prev_image=prev_image)

        detected_items = [
            DetectedItem(
                cx=it['cx'] + it['offset'][0],
                cy=it['cy'] + it['offset'][1],
                w=it['w'],
                h=it['h'],
                distance=it['distance'],
                confidence=it['conf'],
                track_id=-1,
                dx=it['tracking'][0],
                dy=it['tracking'][1],
                item_id=''
            )
            for it in detected_objects
            if it['conf'] >= self.threshold_to_continue
        ]

        if config.DETECTOR_ONLY == False:
            items_to_submit = process_frame_detections(detected_items, prev_tr)
        else:
            items_to_submit = process_frame_detections(detected_items)

        res = [
            dict(
                cx=it.cx,
                cy=it.cy,
                w=it.w,
                h=it.h,
                track_id=it.track_id,
                conf=it.confidence,
                offset=[0.0, 0.0]
            )
            for it in items_to_submit
        ]

        return res
    
    
    
    
    """
    PARTICIPANT_TODO:
    During the evaluation all combinations for flight_id and flight_folder_path will be provided one by one.
    """
    def inference(self, flight_id):
        self.flight_started()
        
        prev_frame = None
        x = 0
        #print(flight_id)
        # print('HERE2 num of frame:', len(self.get_all_frame_images(flight_id)))
        # print('FRAME START', )
        all_frames = self.get_all_frame_images(flight_id)
        check = True
        while check:
            index = random.randint(1, len(all_frames) - 1)
            if flight_id not in self.flight_cache.keys() or index not in self.flight_cache[flight_id]:
                check = False
        self.flight_cache[flight_id] = []
        self.flight_cache[flight_id].append(index)
        self.flight_cache[flight_id].append(index - 1)
        
        frame_image = all_frames[index]
        prev_frame_image = all_frames[index - 1]
        #print(frame_image, prev_frame_image)
        #for frame_image in tqdm(self.get_all_frame_images(flight_id)):
        frame_image_path = self.get_frame_image_location(flight_id, frame_image)
        prev_frame_image_path = self.get_frame_image_location(flight_id, prev_frame_image)
        frame = cv2.imread(frame_image_path, cv2.IMREAD_GRAYSCALE)
        prev_frame = cv2.imread(prev_frame_image_path, cv2.IMREAD_GRAYSCALE)
        #print('frame shape:', frame.shape)
        x += 1
        #print(x)

        # if prev_frame is None:
        #     prev_frame = frame
            #continue

       # if config.DETECTOR_ONLY != True:
        results = self.tracker.predict(image=frame, prev_image=prev_frame)
        #print('inference results', len(results))
        # else:
        #     results = predict(image=frame, prev_image=prev_frame)
        #prev_frame = frame
        #print('results:', results)
        class_name = 'airborne'

        if self.visualize_pred and len(results):
            plt.imshow(frame)
            ax = plt.gca()
        #print(len(results))
        #print(results)
        with open('mapping.pkl', 'rb') as file:
            mapping = pickle.load(file)
        #dataset = Dataset.AOTDataset('val', string=1)
        #label = dataset.find_image_from_filename(frame_image, flight_id)
        frame_id = mapping[flight_id][frame_image]
        #print(frame_id)
        combined_results = {'boxes': [], 'scores': [], 'labels': []}
        for res in results:
            #print(res.keys())
            item = res #['detections'][0]
            box = [item['cx'], item['cy'], item['cx'] + item['w'], item['cy'] + item['h']]
            score = np.float64(item['conf'])
            label = 0
            combined_results['boxes'].append(box)
            combined_results['scores'].append(score)
            combined_results['labels'].append(label)
        self.custom_results[(flight_id, frame_id)] = combined_results
        
        #print(combined_results)
        # if combined_results['boxes'].dim() == 1:
        #     boxes = boxes.unsqueeze(0)
        # # adds w, h to x, y coordinate to obtain x2, y2
        # converted_box = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4]], dim=1)

        # for res in results:
        #     track_id = int(res['track_id'])
        #     confidence = float(res['conf'])
        #     cx = res['cx'] + res['offset'][0]
        #     cy = res['cy'] + res['offset'][1]
        #     w = res['w']
        #     h = res['h']

        #     bbox = [float(cx - w/2), float(cy - h/2), float(cx + w/2), float(cy + h/2)]
        #     # bbox = [float(cy - h / 2), float(cx - w / 2), float(cy + h / 2), float(cx + w / 2)]
        #     self.register_object_and_location(class_name, track_id, bbox, confidence, frame_image)

        #     if self.visualize_pred:
        #         rect = matplotlib.patches.Rectangle(bbox[:2], w, h, linewidth=1, edgecolor='b', facecolor='none')
        #         ax.add_patch(rect)
        #         plt.text(bbox[0], bbox[1], f'{int(confidence * 100)}', c='yellow')

        # if self.visualize_pred and len(results):
        #     plt.show()


# Transfer generated results to metrics codebase bbox format
def convert_and_copy_generated_results_to_metrics_folder():
    import json
    flight_results = json.loads(open("data/results/run0/result.json").read())
    for i in range(len(flight_results)):
        for j in range(len(flight_results[i]['detections'])):
            x = flight_results[i]['detections'][j]['x']
            y = flight_results[i]['detections'][j]['y']
            w = flight_results[i]['detections'][j]['w'] - x
            h = flight_results[i]['detections'][j]['h'] - y

            flight_results[i]['detections'][j]['x'] = x + w/2
            flight_results[i]['detections'][j]['y'] = y + h/2
            flight_results[i]['detections'][j]['w'] = w
            flight_results[i]['detections'][j]['h'] = h

    with open("data/evaluation/result/result.json", 'w') as fp:
        json.dump(flight_results, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Dmytro predictions.")
    parser.add_argument('--job_id', type=int, default=None, help='num of job in array')
    args = parser.parse_args()
    job_id = args.job_id
    if job_id == 11:
        job_id = None
    submission = SegPredictor()
    submission.run(job_id=job_id)
    submission.save_results(job_id=job_id)
    # convert_and_copy_generated_results_to_metrics_folder()

