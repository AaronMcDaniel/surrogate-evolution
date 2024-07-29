import pickle

import pandas as pd
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import config
import os
import common_utils
import matplotlib.pyplot as plt
import cv2
import math
import re

from dataclasses import dataclass
from typing import List


@dataclass
class DetectionItem:
    cls_name: str
    item_id: int
    distance: float
    cx: float
    cy: float
    w: float
    h: float
    above_horizon: float

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

    def apply_img_scale(self, scale):
        return DetectionItem(
            cls_name=self.cls_name,
            item_id=self.item_id,
            distance=self.distance,
            cx=self.cx / scale,
            cy=self.cy / scale,
            w=self.w / scale,
            h=self.h / scale,
            above_horizon=self.above_horizon
        )

    def apply_transform(self, transform, scale):
        w = self.w
        h = self.h

        x = self.cx - w / 2
        y = self.cy - h / 2

        points = np.array([
            [x, y + h / 3],
            [x, y + h * 2 / 3],
            [x + w, y + h / 3],
            [x + w, y + h * 2 / 3],
            [x + w / 3, y],
            [x + w * 2 / 3, y],
            [x + w / 3, y + h],
            [x + w * 2 / 3, y + h],
        ])

        # points = transform().inverse(points)

        points = (transform[:2, :2] @ points.T).T + transform[:2, 2]

        p0 = np.min(points, axis=0)
        p1 = np.max(points, axis=0)

        return DetectionItem(
            cls_name=self.cls_name,
            item_id=self.item_id,
            distance=self.distance / scale,
            cx=(p0[0] + p1[0]) / 2,
            cy=(p0[1] + p1[1]) / 2,
            w=p1[0] - p0[0],
            h=p1[1] - p0[1],
            above_horizon=self.above_horizon
        )


@dataclass
class Frame:
    part: int
    flight_id: str
    frame_num: int
    img_name: str

    items: List[DetectionItem]
    have_match_item: bool = False


class BaseDataset(torch.utils.data.Dataset):
    STAGE_TRAIN = "train"
    STAGE_VALID = "valid"

    def __init__(
            self,
            stage,
            cfg_data,
            return_torch_tensors=False,
            small_subset=True
    ):
        self.stage = stage
        self.is_training = False
        self.return_torch_tensors = return_torch_tensors
        self.dataset_params = cfg_data['dataset_params']

        self.scale = self.dataset_params['scale']
        self.scale_str = '2' if self.scale == 2 else ''

        self.w = 2496 // self.scale
        self.h = 2048 // self.scale

        if stage == self.STAGE_TRAIN:
            self.is_training = True
        else:  # if stage == self.STAGE_VALID:
            self.is_training = False

        parts = [1] #, 2, 3]

        self.frames: List[Frame] = []
        self.frame_nums_with_items: List[int] = []
        self.frame_nums_with_match_items: List[int] = []

        self.train_on_all_samples = self.dataset_params.get('train_on_all_samples', False)

        for part in parts:
            with common_utils.timeit_context('load ds ' + str(part)):
                frames, frame_nums_with_items, frame_nums_with_match_items = self.load_ds(part,
                                                                                          stage,
                                                                                          small_subset=small_subset)
            l = len(self.frames)
            self.frames += frames
            self.frame_nums_with_items += [i + l for i in frame_nums_with_items]
            self.frame_nums_with_match_items += [i + l for i in frame_nums_with_match_items]

        self.parts = parts

        print(stage, len(self.frames), len(self.frame_nums_with_items), len(self.frame_nums_with_match_items))

    def img_fn(self, part, flight_id, img_name):
        return f'{config.DATA_DIR}/part{part}/Images{self.scale_str}/part{part}{flight_id}/{img_name}.{config.IMG_FORMAT}' #ADDED {part} in front of {flight_id}

    def prepare_train_val_split(self):
        """
            Prepare the train_val_flights.csv file, selecting the isolated in time group of 572 flights,
            between 280 and 325 days from the first flight
        """
        fn = f'{config.DATA_DIR}/train_val_flights.csv'
        if not os.path.exists(fn):
            print('Prepare train/val split')
            df_part1 = pd.read_csv(f'{config.DATA_DIR}/part1/ImageSets/groundtruth.csv')
            df_part2 = pd.read_csv(f'{config.DATA_DIR}/part2/ImageSets/groundtruth.csv')
            #df_part3 = pd.read_csv(f'{config.DATA_DIR}/part3/ImageSets/groundtruth.csv')
            df_part1['part'] = 'part1'
            df_part2['part'] = 'part2'
            #df_part3['part'] = 'part3'
            df = pd.concat([df_part1, df_part2]) #, df_part3])
            first_fly = df.drop_duplicates(['flight_id', 'part'], ignore_index=True)
            first_fly = first_fly.reset_index(drop=True)
            ts_s = np.array(first_fly.time.values)*1e-9
            ts_d = ts_s / 60 / 60 / 24
            first_fly['ts_d'] = ts_d - ts_d.min()
            val_samples_mask = (first_fly['ts_d'] > 280) & (first_fly['ts_d'] < 325)
            first_fly['is_validation'] = val_samples_mask

            print(f'Validation flights: {val_samples_mask.sum()}')
            first_fly[['part', 'flight_id', 'ts_d', 'is_validation']].to_csv(fn, index=False)

    def load_ds(self, part, stage, small_subset=False):
        if small_subset:
            cache_fn = f'{config.DATA_DIR}/ds{self.scale_str}_{part}_small_{stage}.pkl'
        else:
            suffix = '_all' if self.train_on_all_samples else ''
            cache_fn = f'{config.DATA_DIR}/ds{self.scale_str}_{part}_{stage}{suffix}.pkl'
        print(cache_fn)
        self.prepare_train_val_split()

        if not os.path.exists(cache_fn):
            frames_dict = {}
            df = pd.read_csv(f'{config.DATA_DIR}/part{part}/ImageSets/groundtruth.csv')
            print(df.shape)
            if not self.train_on_all_samples:
                train_val_ds = pd.read_csv(f'{config.DATA_DIR}/train_val_flights.csv')
                train_val_ds = train_val_ds[train_val_ds['part'] == f'part{part}'].reset_index(drop=True)

                df = pd.merge(df, train_val_ds[['flight_id', 'is_validation']], how='left', on='flight_id')
                print(df.shape)
                df = df[df['is_validation'] == (stage == self.STAGE_VALID)].reset_index(drop=True)
                print(df.shape)

            # discard frames with no objects for now
            # df = df[~df['id'].isna()].reset_index(drop=True)

            if small_subset:
                df = df[:len(df) // 64]

            rx = re.compile(r'(\D+)(\d+)')

            for _, row in df.iterrows():
                flight_id = row['flight_id']
                img_name = row['img_name'][:-4]
                frame_num = row['frame']
                range_distance_m = row['range_distance_m']
                gt_left = row['gt_left']
                gt_right = row['gt_right']
                gt_top = row['gt_top']
                gt_bottom = row['gt_bottom']
                item_id = row['id']

                if not os.path.exists(self.img_fn(part, flight_id, img_name)):
                    print('skip missing img', self.img_fn(part, flight_id, img_name))
                    continue

                key = (flight_id, frame_num)

                if key not in frames_dict:
                    frames_dict[key] = Frame(
                        part=part,
                        flight_id=flight_id,
                        frame_num=frame_num,
                        img_name=img_name,
                        items=[]
                    )

                if not isinstance(item_id, str) and math.isnan(item_id):
                    continue

                m = rx.search(item_id)
                cls_name = m.group(1)
                item_id_num = int(m.group(2))

                if cls_name == 'BIrd':
                    cls_name = 'Bird'

                frame = frames_dict[key]
                frame.items.append(
                    DetectionItem(
                        cls_name=cls_name,
                        item_id=item_id_num,
                        distance=range_distance_m,
                        cx=(gt_left + gt_right) / 2,
                        cy=(gt_top + gt_bottom) / 2,
                        w=gt_right - gt_left,
                        h=gt_bottom - gt_top,
                        above_horizon=row['is_above_horizon']
                    )
                )

                if not np.isnan(range_distance_m) and range_distance_m < config.UPPER_BOUND_MAX_DIST_SELECTED_TRAIN:
                    frame.have_match_item = True

            frames = [frames_dict[k] for k in sorted(frames_dict.keys())]
            frame_nums_with_items = [i for i, f in enumerate(frames) if len(f.items)]
            frame_nums_with_match_items = [i for i, f in enumerate(frames) if f.have_match_item]

            pickle.dump((frames, frame_nums_with_items, frame_nums_with_match_items), open(cache_fn, 'wb'))

        frames, frame_nums_with_items, frame_nums_with_match_items = pickle.load(open(cache_fn, 'rb'))
        return frames, frame_nums_with_items, frame_nums_with_match_items

    def __len__(self) -> int:
        return len(self.frames)


def gaussian2D(shape, sigma_x, sigma_y):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-x * x / (2 * sigma_x * sigma_x) - y * y / (2 * sigma_y * sigma_y))
    h[h < 1e-4] = 0
    cy = (shape[0] - 1) // 2
    cx = (shape[1] - 1) // 2
    h[cy-1:cy+2, cx-1:cx+2] = 1.0
    return h, y + np.zeros_like(x), x + np.zeros_like(y)


def render_y(items, prev_step_items, w, h, pred_scale, non_important_items_scale=1.0):
    w4 = w // pred_scale
    h4 = h // pred_scale

    cls = np.zeros((config.NB_CLASSES, h4, w4), dtype=np.float32)
    cls_planned = np.zeros((config.NB_CLASSES, h4, w4), dtype=np.float32)

    reg_mask = np.zeros((h4, w4), dtype=np.float32)
    reg_size = np.zeros((2, h4, w4), dtype=np.float32)

    reg_offset = np.zeros((2, h4, w4), dtype=np.float32)
    reg_offset_mask = np.zeros((h4, w4), dtype=np.float32)

    reg_tracking = np.zeros((2, h4, w4), dtype=np.float32)
    reg_tracking_mask = np.zeros((h4, w4), dtype=np.float32)

    reg_distance = np.zeros((h4, w4), dtype=np.float32)
    reg_distance_mask = np.zeros((h4, w4), dtype=np.float32)

    reg_above_horizon = np.zeros((1, h4, w4), dtype=np.float32) + 0.5
    reg_above_horizon_mask = np.zeros((1, h4, w4), dtype=np.float32)

    for item in items:
        distance = item.distance
        have_distance = not np.isnan(distance)
        item_important = have_distance and (distance < config.UPPER_BOUND_MAX_DIST_SELECTED_TRAIN)
        if distance > config.MAX_PREDICT_DISTANCE:
            distance = config.MAX_PREDICT_DISTANCE

        cx = item.cx
        cy = item.cy

        cx_img = math.floor(cx / pred_scale)
        cy_img = math.floor(cy / pred_scale)

        orig_w = max(10.0, item.w)
        orig_h = max(10.0, item.h)

        w = math.ceil(orig_w / pred_scale)
        h = math.ceil(orig_h / pred_scale)
        w = max(3, w // 2 * 2 + 1)
        h = max(3, h // 2 * 2 + 1)

        item_mask, ogrid_y, ogrid_x = gaussian2D((h, w), sigma_x=w / 4, sigma_y=h / 4)

        if not item_important:
            item_mask = item_mask * non_important_items_scale

        x_offset = (cx / pred_scale - (cx_img + 0.5))
        y_offset = (cy / pred_scale - (cy_img + 0.5))

        if (0 < cx_img < w4 - 1) and (0 < cy_img < h4 - 1):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    reg_offset[0, cy_img + dy, cx_img + dx] = x_offset - dx
                    reg_offset[1, cy_img + dy, cx_img + dx] = y_offset - dy
                    reg_offset_mask[cy_img + dy, cx_img + dx] = 1

        log_distance = np.log2(max(40.0, distance))

        # clip masks
        w2 = (w - 1) // 2
        h2 = (h - 1) // 2

        dst_x = cx_img - w2
        dst_y = cy_img - h2

        if dst_x < 0:
            item_mask = item_mask[:, -dst_x:]
            dst_x = 0

        if dst_y < 0:
            item_mask = item_mask[-dst_y:, :]
            dst_y = 0

        mask_h, mask_w = item_mask.shape
        if dst_x + mask_w > w4:
            mask_w = w4 - dst_x
            item_mask = item_mask[:, :mask_w]

        if dst_y + mask_h > h4:
            mask_h = h4 - dst_y
            item_mask = item_mask[:mask_h, :]

        if mask_w > 0 and mask_h > 0:
            res_slice = np.s_[dst_y:dst_y + mask_h, dst_x:dst_x + mask_w]

            cls_name = item.cls_name
            if cls_name in ('Airplane', 'Drone', 'Helicopter'):
                cls_name = 'Airborne'

            if cls_name == 'Flock':
                cls_name = 'Bird'

            if cls_name == 'Bird':
                continue

            cls_crop = cls[config.CLASSES.index(cls_name)][res_slice]
            np.maximum(cls_crop, item_mask, out=cls_crop)

            if have_distance and item.distance < config.UPPER_BOUND_MAX_DIST * 1.4:
                cls_planned_crop = cls_planned[config.CLASSES.index(cls_name)][res_slice]
                np.maximum(cls_planned_crop, item_mask, out=cls_planned_crop)

            orig_mask_crop = reg_mask[res_slice]
            pixels_to_update = item_mask > orig_mask_crop

            np.maximum(orig_mask_crop, item_mask, out=orig_mask_crop)

            reg_size[0][res_slice][pixels_to_update] = np.log2(orig_w)
            reg_size[1][res_slice][pixels_to_update] = np.log2(orig_h)

            if (0 < cx_img < w4 - 1) and (0 < cy_img < h4 - 1):
                if item.above_horizon > 0.5:
                    reg_above_horizon[0, cy_img, cx_img] = 1.0
                    reg_above_horizon_mask[0, cy_img, cx_img] = 1.0

                if item.above_horizon < -0.5:
                    reg_above_horizon[0, cy_img, cx_img] = 0.0
                    reg_above_horizon_mask[0, cy_img, cx_img] = 1.0

            if have_distance:
                reg_distance[res_slice][pixels_to_update] = log_distance
                distance_map_crop = reg_distance_mask[res_slice]
                np.maximum(distance_map_crop, item_mask, out=distance_map_crop)

            best_distance = 1e20
            for prev_item in prev_step_items:
                if item.cls_name == prev_item.cls_name and item.item_id == prev_item.item_id:
                    distance = ((prev_item.cx - item.cx) ** 2 + (prev_item.cy - item.cy) ** 2) ** 0.5

                    if best_distance < 1e19:
                        print('Found multiple matches')

                    if distance < best_distance:
                        reg_tracking[0][res_slice][pixels_to_update] = np.clip(
                            (prev_item.cx - item.cx) / config.OFFSET_SCALE, -1, 1)
                        reg_tracking[1][res_slice][pixels_to_update] = np.clip(
                            (prev_item.cy - item.cy) / config.OFFSET_SCALE, -1, 1)

                        best_distance = distance
                        tracking_map_crop = reg_tracking_mask[res_slice]
                        np.maximum(tracking_map_crop, item_mask, out=tracking_map_crop)

    cls[0] = np.clip(1.0 - np.sum(cls[1:], axis=0), 0.0, 1.0)
    cls_planned[0] = np.clip(1.0 - np.sum(cls_planned[1:], axis=0), 0.0, 1.0)

    return dict(
        cls=cls,
        cls_planned=cls_planned,
        reg_mask=reg_mask,
        reg_offset=reg_offset,
        reg_size=reg_size,
        reg_tracking=reg_tracking,
        reg_distance=reg_distance,
        reg_above_horizon=reg_above_horizon,
        reg_above_horizon_mask=reg_above_horizon_mask,

        reg_tracking_mask=reg_tracking_mask,
        reg_distance_mask=reg_distance_mask,
        reg_offset_mask=reg_offset_mask
    )


def check_render_y():
    w = 256
    h = 200

    items = [
        DetectionItem(
            cls_name='Airplane',
            item_id=1,
            distance=500,
            cx=100,
            cy=100,
            w=6,
            h=6,
            above_horizon=1.0
        ),

        DetectionItem(
            cls_name='Airplane',
            item_id=1,
            distance=300,
            cx=150,
            cy=120,
            w=400,
            h=300,
            above_horizon=1.0
        ),

        DetectionItem(
            cls_name='Airplane',
            item_id=1,
            distance=100,
            cx=170,
            cy=170,
            w=8,
            h=16,
            above_horizon=-1.0
        ),

        DetectionItem(
            cls_name='Drone',
            item_id=1,
            distance=100,
            cx=500,
            cy=100,
            w=8,
            h=16,
            above_horizon=-1.0
        ),

        DetectionItem(
            cls_name='Airplane',
            item_id=1,
            distance=np.nan,
            cx=w * 2 + 10,
            cy=h * 2 + 10,
            w=200,
            h=200,
            above_horizon=1.0
        ),
    ]

    y = render_y(items=items, prev_step_items=items[:2], pred_scale=8, w=w, h=h)

    for k, v in y.items():
        common_utils.print_stats(k, v)

    fig, ax = plt.subplots(2, 5)
    ax[0, 0].imshow(y['cls'][0])
    ax[0, 1].imshow(y['cls'][config.CLASSES.index('Airplane')])
    ax[0, 2].imshow(y['reg_mask'])
    ax[0, 3].imshow(y['reg_distance'])
    ax[0, 4].imshow(y['reg_distance_mask'])
    # ax[1, 0].imshow(y['reg_size'][0])
    # ax[1, 1].imshow(y['reg_size'][1])
    ax[1, 0].imshow(y['reg_above_horizon'][0])
    ax[1, 1].imshow(y['reg_above_horizon_mask'][0])
    ax[1, 2].imshow(y['reg_offset'][0])
    ax[1, 3].imshow(y['reg_offset'][1])
    ax[1, 4].imshow(y['reg_offset_mask'])

    plt.show()


def test_render_y_position_offset():
    w = 256
    h = 200

    cx = 112
    cy = 133

    items = [
        DetectionItem(
            cls_name='Airplane',
            item_id=1,
            distance=500,
            cx=cx,
            cy=cy,
            w=32,
            h=32
        ),
    ]

    y = render_y(items=items, scale=2, pred_scale=4, w=w, h=h)

    reg_offset = y['reg_offset']
    img_cx = math.floor(cx / 8)
    img_cy = math.floor(cy / 8)

    assert y['reg_mask'][img_cy, img_cx] == 1.0

    for img_x in range(img_cx - 1, img_cx + 2):
        for img_y in range(img_cy - 1, img_cy + 2):
            pred_x = (img_x + 0.5) * 8 + reg_offset[0, img_y, img_x] * 8
            pred_y = (img_y + 0.5) * 8 + reg_offset[1, img_y, img_x] * 8

            err_x = pred_x - cx
            err_y = pred_y - cy

            print(img_x, img_y, reg_offset[:, img_y, img_x], pred_x, pred_y, err_x, err_y)

            assert abs(err_x) < 0.1
            assert abs(err_y) < 0.1


class TrackingDataset(BaseDataset):
    def __init__(
            self,
            stage,
            cfg_data,
            parts=None,
            return_torch_tensors=False,
            small_subset=False,
            cache_samples=False
    ):
        super().__init__(stage, cfg_data, return_torch_tensors=return_torch_tensors, small_subset=small_subset)
        self.back_steps = self.dataset_params['back_steps']

        self.crop_w, self.crop_h = self.dataset_params.get('crop_size', (-1, -1))
        self.pred_scale = self.dataset_params.get('pred_scale', 8)
        self.pos_offset_sigma = self.dataset_params.get('pos_offset_sigma', 0)

        self.crop_enabled = self.crop_w > 0
        self.crop_with_plane_ratio = self.dataset_params.get('crop_with_plane_ratio', 0.9) if self.is_training else 1.0

        self.val_values_cache = {}
        print('START HERE')
        print(self.val_values_cache)
        print('END HERE')
        self.cache_samples = cache_samples

        self.transforms = {}
        for part in self.parts:
            with common_utils.timeit_context('load transforms ' + str(part)):
                self.transforms[part] = {}
                transforms_dir = f'{config.DATA_DIR}/frame_transforms/part{part}'
                for fn in os.listdir(transforms_dir):
                    if fn.endswith('.pkl'):
                        flight_id = fn[:-4]
                        self.transforms[part][flight_id] = pd.read_pickle(f'{transforms_dir}/{flight_id}.pkl')

        self.fpm_samples = {}  # frame_index -> [[conf, cx, cy], [conf, cx, cy], ...]]

    def set_fpm_samples(self, samples):
        """
        :param samples: fpm samples, in format:
        [
            [max_conf, frame_index, [[conf, cx, cy], [conf, cx, cy], ...]]
        ]
        :return:
        """
        self.fpm_samples = {frame_index: points for conf, frame_index, points in samples}

    def __getitem__(self, index):
        if self.cache_samples and index in self.val_values_cache:
            return self.val_values_cache[index]

        frame = self.frames[index]

        back_steps_candidates = self.back_steps
        frame_back_steps = []
        prev_frames = []

        for back_steps in back_steps_candidates:
            while back_steps > 0:
                if back_steps < index:
                    prev_frame = self.frames[index - back_steps]
                    if prev_frame.flight_id == frame.flight_id:
                        break
                back_steps -= 1

            frame_back_steps.append(back_steps)
            prev_frames.append(self.frames[index - back_steps])

        cur_img = cv2.imread(self.img_fn(frame.part, frame.flight_id, frame.img_name), cv2.IMREAD_GRAYSCALE)

        if cur_img is None:
            return self.__getitem__(index + 1)

        prev_images = []
        for prev_frame in prev_frames:
            prev_img = cv2.imread(self.img_fn(prev_frame.part, prev_frame.flight_id, prev_frame.img_name),
                                  cv2.IMREAD_GRAYSCALE)
            if prev_img is None:
                prev_img = cur_img.copy()
            prev_images.append(prev_img)

        h, w = cur_img.shape
        if self.is_training:
            scale_aug = 2 ** np.random.normal(loc=0.0, scale=0.25)
            scale_aug_x = 2 ** np.random.normal(loc=0.0, scale=0.1)
            scale_aug_y = 2 ** np.random.normal(loc=0.0, scale=0.1)
            scale_aug_combined = scale_aug * (scale_aug_x * scale_aug_y) ** 0.5
            angle = np.random.normal(loc=0.0, scale=7.0)
            shear = 0  # np.random.normal(loc=0.0, scale=3.0)

            # print(f'scale {scale_aug} scale_x {scale_aug_x} scale_y {scale_aug_y} angle {angle}')

            transform = common_utils.build_geom_transform(
                dst_w=w,
                dst_h=h,
                src_center_x=w / 2 + np.random.uniform(-16, 16),
                src_center_y=h / 2 + np.random.uniform(-16, 16),
                scale_x=scale_aug * scale_aug_x,
                scale_y=scale_aug * scale_aug_y,
                angle=angle,
                shear=shear,
                hflip=np.random.choice([True, False]),
                vflip=False,
            )
        else:
            scale_aug = 1.0
            scale_aug_x = 1.0
            scale_aug_y = 1.0
            scale_aug_combined = scale_aug * (scale_aug_x * scale_aug_y) ** 0.5

            transform = common_utils.build_geom_transform(
                dst_w=w,
                dst_h=h,
                src_center_x=w / 2,
                src_center_y=h / 2,
                scale_x=scale_aug * scale_aug_x,
                scale_y=scale_aug * scale_aug_y,
                angle=0,
                shear=0,
                hflip=False,
                vflip=False,
            )

        cur_img_transform = np.linalg.pinv(transform.params)
        # print(cur_img_transform)

        frames_transform = {0: cur_img_transform}
        flight_transforms = self.transforms[frame.part][frame.flight_id]

        for frame_back_step in range(1, max(frame_back_steps) + 1):
            frame_num = frame.frame_num - frame_back_step + 1
            flight_transforms_row = flight_transforms[flight_transforms.frame == frame_num]
            # print(flight_transforms_row.iloc[0]["dx"], flight_transforms_row.iloc[0]["dy"])

            if len(flight_transforms_row):
                dx = flight_transforms_row.iloc[0]["dx"]
                dy = flight_transforms_row.iloc[0]["dy"]
                angle = flight_transforms_row.iloc[0]["angle"]

                if self.is_training and self.pos_offset_sigma > 0:
                    dx += np.random.normal(0, self.pos_offset_sigma)
                    dy += np.random.normal(0, self.pos_offset_sigma)

                transform = common_utils.build_geom_transform(
                    dst_w=w,
                    dst_h=h,
                    src_center_x=w / 2 + dx,
                    src_center_y=h / 2 + dy,
                    scale_x=1.0,
                    scale_y=1.0,
                    angle=angle,
                    return_params=True
                )
                transform = frames_transform[frame_back_step - 1] @ transform
                frames_transform[frame_back_step] = transform
            else:
                transform = frames_transform[frame_back_step - 1]
                frames_transform[frame_back_step] = transform

        if 1 not in frames_transform:
            frames_transform[1] = frames_transform[0]

        cur_step_items = [f.apply_img_scale(self.scale).apply_transform(cur_img_transform, scale=scale_aug_combined)
                          for f in frame.items]
        prev_step_items = [f.apply_img_scale(self.scale).apply_transform(frames_transform[1], scale=scale_aug_combined)
                           for f in prev_frames[0].items]

        center_on_plane = len(cur_step_items) > 0 and np.random.rand() < self.crop_with_plane_ratio
        if center_on_plane:
            if self.is_training:
                item = np.random.choice(cur_step_items)
            else:
                item = cur_step_items[0]

            selected_item = item

            crop_cx = item.cx
            crop_cy = item.cy

            if self.is_training:
                crop_cx += np.random.normal(loc=0.0, scale=self.crop_w / 6)
                crop_cy += np.random.normal(loc=0.0, scale=self.crop_h / 6)
        else:
            selected_item = None

            if self.is_training:
                crop_cx = np.random.uniform(self.crop_w // 2 - 8, w - self.crop_w // 2 + 8)
                crop_cy = np.random.uniform(self.crop_h // 2 - 8, h - self.crop_h // 2 + 8)
            else:
                crop_cx = w / 2
                crop_cy = h / 2

        if index in self.fpm_samples:
            fpm_sample = self.fpm_samples[index]
            samples_conf = np.array([c[0] for c in fpm_sample])
            sample_idx = np.random.choice(len(fpm_sample), p=samples_conf/samples_conf.sum())
            _, crop_cx, crop_cy = fpm_sample[sample_idx]
            crop_cx += np.random.normal(loc=0.0, scale=self.crop_w / 4)
            crop_cy += np.random.normal(loc=0.0, scale=self.crop_h / 4)

        pad_x = max(16, (self.crop_w - w) // 2)
        pad_y = max(16, (self.crop_h - h) // 2)
        crop_x = round(crop_cx - self.crop_w // 2)
        crop_y = round(crop_cy - self.crop_h // 2)
        crop_x = np.clip(crop_x, -pad_x, w - self.crop_w + pad_x)
        crop_y = np.clip(crop_y, -pad_y, h - self.crop_h + pad_y)

        for item in cur_step_items:
            item.cx -= crop_x
            item.cy -= crop_y

        for item in prev_step_items:
            item.cx -= crop_x
            item.cy -= crop_y

        crops = []

        for img, frame_back_step in zip([cur_img] + prev_images, [0] + frame_back_steps):
            transform = frames_transform[frame_back_step]
            t = np.array([[1, 0, -crop_x],
                          [0, 1, -crop_y],
                          [0, 0, 1]]) @ transform

            crop = cv2.warpAffine(
                img,
                t[:2, :],
                dsize=(self.crop_w, self.crop_h),
                flags=cv2.INTER_LINEAR)
            crops.append(crop)

        res = render_y(cur_step_items, prev_step_items=prev_step_items, w=self.crop_w, h=self.crop_h, pred_scale=self.pred_scale)

        res['idx'] = index
        res['crop_x'] = crop_x
        res['crop_y'] = crop_y
        res['transform'] = cur_img_transform
        res['is_fpm_sample'] = index in self.fpm_samples

        res['image'] = crops[0]
        # res['image2'] = cur_img_transformed
        # res['image_orig'] = cur_img
        # res['prev_image'] = prev_images[0]
        for i, prev_img in enumerate(crops[1:]):
            res[f'prev_image_aligned{i}'] = prev_img

        if selected_item is not None:
            res['center_item_w'] = selected_item.w
            res['center_item_h'] = selected_item.h
            res['center_item_distance'] = selected_item.distance
        else:
            res['center_item_w'] = 0
            res['center_item_h'] = 0
            res['center_item_distance'] = 0

        gamma_aug = 2 ** np.random.normal(0.0, 0.2)

        if self.return_torch_tensors:
            for k in list(res.keys()):
                if k == 'image' or k.startswith('prev_image_aligned'):
                    res[k] = torch.from_numpy(res[k].astype(np.float32) / 255.0).float()

                    if self.is_training:
                        res[k] = torch.pow(res[k], gamma_aug)

                elif isinstance(res[k], np.ndarray):
                    res[k] = torch.from_numpy(res[k].astype(np.float32)).float()

        # if self.cache_samples and len(self.val_values_cache) < 1000:
        #     self.val_values_cache[index] = res

        return res


def check_dataset():
    scale = 1

    ds = TrackingDataset(
        stage=BaseDataset.STAGE_VALID,
        cfg_data={'dataset_params': dict(
            train_on_all_samples=True,
            back_steps=[1],
            scale=scale,
            crop_size=(512, 512),
            pos_offset_sigma=0,
        )},
        small_subset=True
    )

    idx_to_check = [1836028, 1570557, 1289913, 2663822, 1289921, 1836028, 672784, 728191, 1100405, 672784]
    idx_to_check = []

    for idx in idx_to_check + ds.frame_nums_with_match_items[::32]:
        sample = ds[idx]

        if ds.frames[idx].items[0].distance > 900:
            continue

        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                common_utils.print_stats(k, v)

        centers = []
        for item in ds.frames[idx].items:
            print(item.distance)
            centers.append([item.cx / scale, item.cy / scale])

        print(sample['idx'])
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(sample['image'])

        # ax[0, 1].imshow(sample['image_orig'])
        ax[0, 1].scatter(np.array(centers)[:, 0], np.array(centers)[:, 1])

        ax[0, 2].imshow(sample['cls'][0])

        ax[1, 2].imshow(np.stack([
            sample['prev_image_aligned0'],
            sample['image'],
            sample['prev_image_aligned0']
        ], axis=2)
        )

        ax[1, 1].imshow(np.stack([
            sample['prev_image_aligned0'],
            sample['image'],
            sample['prev_image_aligned0']
        ], axis=2)
        )

        ax[1, 1].imshow(cv2.resize(sample['cls'][0], None, fx=8.0, fy=8.0, interpolation=cv2.INTER_NEAREST), alpha=0.2)

        # ax[0, 1].imshow(gaussian2D((128, 128), sigma_x=128/3*0.54, sigma_y=128/3*0.54))
        # ax[0, 2].imshow(gaussian2D((6, 6), sigma_x=6./6, sigma_y=6./6))
        # ax[0, 3].imshow(gaussian2D((3, 3), sigma_x=3. / 6, sigma_y=3. / 6))
        # ax[1, 1].imshow(gaussian2D((5, 5), sigma_x=5. / 6, sigma_y=5. / 6))
        # ax[1, 2].imshow(gaussian2D((17, 17), sigma_x=17. / 6, sigma_y=17. / 6))

        plt.show()

    # ds = BaseDataset(stage=BaseDataset.STAGE_TRAIN,
    #                  fold=1,
    #                  cfg_data={'dataset_params': dict()}
    #                  )

    # for sample in ds:
    #     print(sample['inchi'])
    #     # common_utils.print_stats('image', sample['image'])
    #     # common_utils.print_stats('image_generated', sample['image_generated'])
    #     print('nb atoms: ', sample['y_nb_atoms'])
    #     print('nb bonds: ', sample['y_nb_bonds'])
    #
    #     fig, ax = plt.subplots(2, 4)
    #     ax[0, 0].imshow(sample['image'])
    #     ax[0, 1].imshow(sample['image_generated'])
    #     ax[0, 2].imshow(sample['y_atom_num'])
    #     ax[0, 3].imshow(sample['y_atom_mask'])
    #
    #     ax[1, 0].imshow(sample['y_bond_type'])
    #     ax[1, 1].imshow(sample['y_bond_mask'])
    #     ax[1, 2].imshow(sample['y_bond_len'][0])
    #     ax[1, 3].imshow(sample['y_bond_dir'][0])
    #     plt.show()


def check_dataset_for_fpm():
    ds = TrackingDataset(
        stage=BaseDataset.STAGE_VALID,
        cfg_data={'dataset_params': dict(
            back_steps=[1],
            scale=1,
            crop_with_plane_ratio=0,
            crop_size=(2432, 2048)
        )},
        small_subset=False
    )

    scale = 1

    # idx_to_check = [1836028, 1570557, 1289913, 2663822, 1289921, 1836028, 672784, 728191, 1100405, 672784]
    idx_to_check = []

    for idx in idx_to_check + ds.frame_nums_with_match_items[::32]:
        sample = ds[idx]

        if ds.frames[idx].items[0].distance > 900:
            continue

        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                common_utils.print_stats(k, v)

        centers = []
        for item in ds.frames[idx].items:
            print(item.distance)
            centers.append([item.cx / scale, item.cy / scale])

        print(sample['idx'])
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(sample['image'])

        ax[0, 1].imshow(sample['image_orig'])
        ax[0, 1].scatter(np.array(centers)[:, 0], np.array(centers)[:, 1])

        ax[0, 2].imshow(sample['cls'][0])

        ax[1, 2].imshow(np.stack([
            sample['prev_image_aligned0'],
            sample['image'],
            sample['prev_image_aligned0']
        ], axis=2)
        )

        ax[1, 1].imshow(np.stack([
            sample['prev_image_aligned0'],
            sample['image'],
            sample['prev_image_aligned0']
        ], axis=2)
        )

        ax[1, 1].imshow(cv2.resize(sample['cls'][0], None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST), alpha=0.2)

        # ax[0, 1].imshow(gaussian2D((128, 128), sigma_x=128/3*0.54, sigma_y=128/3*0.54))
        # ax[0, 2].imshow(gaussian2D((6, 6), sigma_x=6./6, sigma_y=6./6))
        # ax[0, 3].imshow(gaussian2D((3, 3), sigma_x=3. / 6, sigma_y=3. / 6))
        # ax[1, 1].imshow(gaussian2D((5, 5), sigma_x=5. / 6, sigma_y=5. / 6))
        # ax[1, 2].imshow(gaussian2D((17, 17), sigma_x=17. / 6, sigma_y=17. / 6))

        plt.show()

    # ds = BaseDataset(stage=BaseDataset.STAGE_TRAIN,
    #                  fold=1,
    #                  cfg_data={'dataset_params': dict()}
    #                  )

    # for sample in ds:
    #     print(sample['inchi'])
    #     # common_utils.print_stats('image', sample['image'])
    #     # common_utils.print_stats('image_generated', sample['image_generated'])
    #     print('nb atoms: ', sample['y_nb_atoms'])
    #     print('nb bonds: ', sample['y_nb_bonds'])
    #
    #     fig, ax = plt.subplots(2, 4)
    #     ax[0, 0].imshow(sample['image'])
    #     ax[0, 1].imshow(sample['image_generated'])
    #     ax[0, 2].imshow(sample['y_atom_num'])
    #     ax[0, 3].imshow(sample['y_atom_mask'])
    #
    #     ax[1, 0].imshow(sample['y_bond_type'])
    #     ax[1, 1].imshow(sample['y_bond_mask'])
    #     ax[1, 2].imshow(sample['y_bond_len'][0])
    #     ax[1, 3].imshow(sample['y_bond_dir'][0])
    #     plt.show()


def check_dataset_set_fpm_samples():
    ds = TrackingDataset(
        stage=BaseDataset.STAGE_VALID,
        cfg_data={'dataset_params': dict(
            back_steps=[1],
            scale=1,
            crop_with_plane_ratio=0,
            crop_size=(512, 512)
        )},
        small_subset=False
    )

    frame_idx = 1
    ds.set_fpm_samples([
        [0.5, frame_idx, [[1.0, 0, 0]]]
    ])

    sample = ds[frame_idx]
    print('crop pos:', sample['crop_x'], sample['crop_y'])
    plt.imshow(sample['image'])
    plt.show()

    ds.set_fpm_samples([
        [0.5, frame_idx, [[1.0, 0, 2000]]]
    ])

    sample = ds[frame_idx]
    print('crop pos:', sample['crop_x'], sample['crop_y'])
    plt.imshow(sample['image'])
    plt.show()


def check_performance():
    ds = TrackingDataset(
        stage=BaseDataset.STAGE_VALID,
        cfg_data={'dataset_params': dict(back_steps=[1], scale=1, crop_size=(512, 512))}
    )

    # for idx in ds.frame_nums_with_match_items[::32]:
    #     sample = ds[idx]
    nb_samples = 0
    for sample in tqdm(ds):
        nb_samples += 1
        if nb_samples > 512:
            break


if __name__ == '__main__':
    check_dataset()
    # test_render_y_position_offset()
    # check_render_y()
    # check_performance()
    # check_dataset_for_fpm()
    # find_frames_transform_distribution()
    # check_dataset_set_fpm_samples()

