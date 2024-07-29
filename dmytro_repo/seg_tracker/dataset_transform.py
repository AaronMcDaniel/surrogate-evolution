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
import scipy
import scipy.optimize
from typing import List
from offset_grid_to_transform import offset_grid_to_transform_params


def find_transform_between_images(cur_img, prev_img):
    downscale_candidates = [2, 4, 8, 16]

    h, w = prev_img.shape
    cur_pts = []
    for downscale_candidate in downscale_candidates:
        cur_img_2 = cv2.resize(cur_img[:h * 4 // 5, :], None,
                               fx=1.0 / downscale_candidate,
                               fy=1.0 / downscale_candidate,
                               interpolation=cv2.INTER_AREA)
        try:
            cur_pts = cv2.goodFeaturesToTrack(cur_img_2,
                                              maxCorners=200,
                                              qualityLevel=0.01,
                                              minDistance=30,
                                              blockSize=3) * downscale_candidate
            if len(cur_pts) > 32:
                break

        except TypeError:
            continue

    if len(cur_pts) < 32:
        raise RuntimeError('Failed to find points')

    prev_pts, status, err = cv2.calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, None)

    if prev_pts is None:
        raise RuntimeError('Failed to track points')

    # Filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    cur_pts = cur_pts[idx]

    # Find transformation matrix
    transform, inlier = cv2.estimateAffinePartial2D(prev_pts, cur_pts)

    if transform is not None and transform.shape == (2, 3):
        transform = np.vstack([transform, [0., 0., 1.]])
    else:
        raise RuntimeError('transform failed')

    estimated_scale = (transform[0, 0] ** 2 + transform[0, 1] ** 2) ** 0.5
    estimated_angle = math.atan2(transform[0, 1], transform[0, 0]) * 180 / math.pi

    offset_xy = np.linalg.pinv(transform) @ np.array([w / 2, h / 2, 1]).T

    estimated_offset_x = offset_xy[0] - w / 2
    estimated_offset_y = offset_xy[1] - h / 2

    return dict(
        scale=estimated_scale,
        angle=estimated_angle,
        dx=estimated_offset_x,
        dy=estimated_offset_y
    )


class DatasetTransform(torch.utils.data.Dataset):
    """
    Dataset for transform T = build_geom_transform(offset, rotation, scale around center of img) so
    prev_aligned = T * prev_image
    cur_img is aligned with prev_aligned

    """

    STAGE_TRAIN = "train"
    STAGE_VALID = "valid"

    def __init__(
            self,
            stage,
            fold,
            cfg_data,
            return_torch_tensors=False,
            small_subset=False,
            always_return_synth_img=False,
    ):
        self.stage = stage
        self.fold = fold
        self.is_training = False
        self.return_torch_tensors = return_torch_tensors
        self.dataset_params = cfg_data
        self.synthetic_img_ratio = self.dataset_params['synthetic_img_ratio']
        self.always_return_synth_img = always_return_synth_img

        self.downscale = self.dataset_params['downscale']
        self.crop_w, self.crop_h = self.dataset_params['crop_size']

        self.sigma_scale = self.dataset_params['sigma_scale']
        self.sigma_angle = self.dataset_params['sigma_angle']
        self.sigma_offset = self.dataset_params['sigma_offset']

        if stage == self.STAGE_TRAIN:
            parts = [i for i in [1, 2, 3] if i != fold + 1]
            self.is_training = True
        else:  # if stage == self.STAGE_VALID:
            parts = [fold + 1] if fold >= 0 else [0]
            self.is_training = False

        frames = []
        for part in parts:
            with common_utils.timeit_context('load ds ' + str(part)):
                frames.append(self.load_ds(part, small_subset=small_subset))

        self.frames = pd.concat(frames, axis=0, ignore_index=True).reset_index(drop=True)
        print(stage, len(self.frames))

    def img_fn(self, part, flight_id, img_name):
        return f'{config.DATA_DIR}/part{part}/Images/part{part}{flight_id}/{img_name}.{config.IMG_FORMAT}' #ADDED {part} in front of {flight_id}

    def load_ds(self, part, small_subset=False) -> pd.DataFrame:
        if small_subset:
            cache_fn = f'{config.DATA_DIR}/ds_transform_{part}_small.pkl'
        else:
            cache_fn = f'{config.DATA_DIR}/ds_transform_{part}.pkl'

        if not os.path.exists(cache_fn):
            frames_dict = {}
            df = pd.read_csv(f'{config.DATA_DIR}/part{part}/ImageSets/groundtruth.csv')

            if small_subset:
                df = df[:len(df) // 64]

            for _, row in df.iterrows():
                flight_id = row['flight_id']
                img_name = row['img_name'][:-4]
                frame_num = row['frame']

                if not os.path.exists(self.img_fn(part, flight_id, img_name)):
                    print('skip missing img', self.img_fn(part, flight_id, img_name))
                    continue

                key = (flight_id, frame_num)

                if key not in frames_dict:
                    frames_dict[key] = dict(
                        part=part,
                        flight_id=flight_id,
                        frame_num=frame_num,
                        img_name=img_name,
                    )

            frames = pd.DataFrame([frames_dict[k] for k in sorted(frames_dict.keys())])
            frames.to_pickle(cache_fn)

        frames = pd.read_pickle(cache_fn)
        return frames

    def __len__(self) -> int:
        return len(self.frames)

    def load_prev_image_and_transform(self, index, cur_img):
        if index < 1:
            raise RuntimeError('no prev frame')

        frame = self.frames.iloc[index]

        prev_frame = self.frames.iloc[index - 1]
        if prev_frame['flight_id'] != frame['flight_id']:
            raise RuntimeError('no prev frame')

        prev_img = cv2.imread(self.img_fn(prev_frame.part, prev_frame.flight_id, prev_frame.img_name),
                              cv2.IMREAD_GRAYSCALE)
        if prev_img is None:
            raise RuntimeError('no prev frame')

        tr = find_transform_between_images(cur_img, prev_img)
        return prev_img, tr

    def synthetic_img_for_transform_params(self, cur_img, prev_tr_params):
        h, w = cur_img.shape
        scale = prev_tr_params['scale']
        prev_img_transform = common_utils.build_geom_transform(
            dst_w=w,
            dst_h=h,
            src_center_x=w / 2 + prev_tr_params['dx'],
            src_center_y=h / 2 + prev_tr_params['dy'],
            scale_x=scale,
            scale_y=scale,
            angle=prev_tr_params['angle'],
            return_params=True
        )

        prev_img_transform_scale = common_utils.build_geom_transform(
            dst_w=self.crop_w,
            dst_h=self.crop_h,
            src_center_x=w / 2,
            src_center_y=h / 2,
            scale_x=1.0 / self.downscale,
            scale_y=1.0 / self.downscale,
            angle=0,
            return_params=True
        )

        combined_tr = prev_img_transform_scale @ np.linalg.pinv(prev_img_transform)

        prev_img = cv2.warpAffine(
            cur_img,
            combined_tr[:2, :],
            dsize=(self.crop_w, self.crop_h),
            flags=cv2.INTER_AREA if self.downscale > 1 else cv2.INTER_LINEAR)

        return prev_img

    def __getitem__(self, index):
        frame = self.frames.iloc[index]
        use_synthetic = np.random.rand() < self.synthetic_img_ratio

        cur_img = cv2.imread(self.img_fn(frame.part, frame.flight_id, frame.img_name), cv2.IMREAD_GRAYSCALE)

        if cur_img is None:
            return self.__getitem__(index + 1)

        h, w = cur_img.shape

        prev_tr_params = {}

        if not use_synthetic:
            try:
                prev_img, prev_tr_params = self.load_prev_image_and_transform(index, cur_img)

                if abs(prev_tr_params['scale'] - 1) > 3 * self.sigma_scale:
                    use_synthetic = True

                if abs(prev_tr_params['angle']) > 3 * self.sigma_angle:
                    use_synthetic = True

                if abs(prev_tr_params['dx']) > 3 * self.sigma_offset:
                    use_synthetic = True

                if abs(prev_tr_params['dy']) > 3 * self.sigma_offset:
                    use_synthetic = True

            except RuntimeError:
                use_synthetic = True

        cur_img_tr = common_utils.build_geom_transform(
            dst_w=self.crop_w,
            dst_h=self.crop_h,
            src_center_x=w/2,
            src_center_y=h/2,
            scale_x=1.0/self.downscale,
            scale_y=1.0/self.downscale,
            angle=0,
            return_params=True
        )

        cur_img_crop = cv2.warpAffine(
            cur_img,
            cur_img_tr[:2, :],
            dsize=(self.crop_w, self.crop_h),
            flags=cv2.INTER_AREA if self.downscale > 1 else cv2.INTER_LINEAR)

        if use_synthetic:
            prev_tr_params = dict(
                scale=np.exp(np.random.normal(0, self.sigma_scale*2)),
                angle=np.random.normal(0, self.sigma_angle*2),
                dx=np.random.normal(0, self.sigma_offset*2),
                dy=np.random.normal(0, self.sigma_offset*2)
            )

            prev_img_crop = self.synthetic_img_for_transform_params(cur_img=cur_img, prev_tr_params=prev_tr_params)
            if self.always_return_synth_img:
                prev_img_crop_synth = prev_img_crop
        else:
            prev_img_crop = cv2.warpAffine(
                prev_img,
                cur_img_tr[:2, :],  # the same crop as with the cur image
                dsize=(self.crop_w, self.crop_h),
                flags=cv2.INTER_AREA if self.downscale > 1 else cv2.INTER_LINEAR)

            if self.always_return_synth_img:
                prev_img_crop_synth = self.synthetic_img_for_transform_params(cur_img=cur_img, prev_tr_params=prev_tr_params)

        prev_points = np.zeros((2, 32, 32), dtype=np.float32)
        prev_points[0, :, :] = np.arange(16, 1024, 32)[None, :]
        prev_points[1, :, :] = np.arange(16, 1024, 32)[:, None]

        prev_points_1d = prev_points.reshape((2, -1))

        transform = common_utils.build_geom_transform(
            dst_w=self.crop_w,
            dst_h=self.crop_h,
            src_center_x=self.crop_w / 2 + prev_tr_params['dx'],
            src_center_y=self.crop_w / 2 + prev_tr_params['dy'],
            scale_x=prev_tr_params['scale'],
            scale_y=prev_tr_params['scale'],
            angle=prev_tr_params['angle'],
            return_params=True
        )

        cur_points = ((transform[:2, :2] @ prev_points_1d).T + transform[:2, 2]).T
        cur_points = cur_points.reshape((2, 32, 32))

        res = prev_tr_params
        res['is_synthetic'] = use_synthetic
        res['index'] = index
        res['cur_img'] = cur_img_crop
        res['prev_img'] = prev_img_crop

        res['cur_points_grid'] = cur_points
        res['prev_points_grid'] = prev_points

        if self.always_return_synth_img:
            res['prev_img_synth'] = prev_img_crop_synth

        if self.return_torch_tensors:
            res['cur_img'] = torch.from_numpy(res['cur_img'] / 255.0).float()
            res['prev_img'] = torch.from_numpy(res['prev_img'] / 255.0).float()

            res['cur_points_grid'] = torch.from_numpy(res['cur_points_grid']).float()
            res['prev_points_grid'] = torch.from_numpy(res['prev_points_grid']).float()

        return res


def check_transform_ds():
    w = 1024
    h = 1024

    ds = DatasetTransform(
        stage=DatasetTransform.STAGE_TRAIN,
        fold=0,
        return_torch_tensors=False,
        small_subset=True,
        always_return_synth_img=True,
        cfg_data=dict(
            synthetic_img_ratio=0.0,
            downscale=1,
            crop_size=(w, h),
            sigma_scale=0.1,
            sigma_angle=5.0,
            sigma_offset=30.0
        )
    )

    for idx in range(0, len(ds), 128):
        sample = ds[idx]

        print(f"{sample['index']} {sample['is_synthetic']} scale {sample['scale']:0.3f} angle {sample['angle']:0.2f} offset {sample['dx']:0.1f},{sample['dy']:0.1f}")
        if sample['is_synthetic']:
            continue

        if abs(sample['angle']) > 0.4 or abs(sample['scale']-1) > 0.005:
            fig, ax = plt.subplots(2, 4)
            ax[0, 0].imshow(sample['cur_img'])
            ax[0, 1].imshow(sample['prev_img'])
            ax[0, 1].imshow(sample['prev_img_synth'])

            # expected to be aligned
            ax[1, 0].imshow(np.stack([sample['prev_img_synth'], sample['prev_img'], sample['prev_img_synth']], axis=2))
            # expected to be not aligned
            ax[1, 1].imshow(np.stack([sample['prev_img_synth'], sample['cur_img'], sample['prev_img_synth']], axis=2))

            prev_points = sample['prev_points_grid'].reshape((2, -1))
            cur_points = sample['cur_points_grid'].reshape((2, -1))

            # ax[1, 1].scatter(prev_points[0, :], prev_points[1, :], color='blue', s=1)
            # ax[1, 1].scatter(cur_points[0, :], cur_points[1, :], color='green', s=1)
            for i in range(prev_points.shape[1]):
                ax[1, 1].arrow(prev_points[0, i], prev_points[1, i],
                               prev_points[0, i] - cur_points[0, i], prev_points[1, i]-cur_points[1, i],
                               head_width=0.1, head_length=0.2, fc='k', ec='k')

            prev_tr = common_utils.build_geom_transform(
                dst_w=w,
                dst_h=h,
                src_center_x=w/2 + sample['dx'],
                src_center_y=h/2 + sample['dy'],
                scale_x=sample['scale'],
                scale_y=sample['scale'],
                angle=sample['angle'],
                return_params=True
            )

            prev_img_aligned = cv2.warpAffine(
                sample['prev_img'],
                prev_tr[:2, :],  # the same crop as with the cur image
                dsize=(w, h),
                flags=cv2.INTER_LINEAR)

            prev_synth_aligned = cv2.warpAffine(
                sample['prev_img_synth'],
                prev_tr[:2, :],  # the same crop as with the cur image
                dsize=(w, h),
                flags=cv2.INTER_LINEAR)

            # expected to be aligned
            ax[1, 2].imshow(np.stack([sample['cur_img'], prev_img_aligned, sample['cur_img']], axis=2))
            # expected to be perfectly aligned
            ax[1, 3].imshow(np.stack([sample['cur_img'], prev_synth_aligned, sample['cur_img']], axis=2))
            plt.show()


if __name__ == '__main__':
    check_transform_ds()


