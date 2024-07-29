import os
import logging
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import urllib
import zipfile

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from siammot.configs.defaults import cfg
from siammot.modelling.rcnn import build_siammot
from siammot.data.adapters.augmentation.build_augmentation import build_siam_augmentation

import cv2

class SiamMOTTracker:
    """
    Implement a wrapper to call tracker
    """

    def __init__(self,
                 config_file,
                 model_path,
                 gpu_id=0):

        self.device = torch.device("cuda:{}".format(gpu_id))

        cfg.merge_from_file(config_file)
        self.cfg = cfg
        self.model_path = model_path

        self.transform = build_siam_augmentation(cfg, is_train=False)
        self.tracker = self._build_and_load_tracker()
        self.tracker.eval()
        self.tracker.reset_siammot_status()

    def _preprocess(self, frame):

        # frame is RGB-Channel
        frame = Image.fromarray(frame, 'RGB')
        dummy_bbox = torch.tensor([[0, 0, 1, 1]])
        dummy_boxlist = BoxList(dummy_bbox, frame.size, mode='xywh')
        frame, _ = self.transform(frame, dummy_boxlist)

        return frame

    def _build_and_load_tracker(self):
        tracker = build_siammot(self.cfg)
        tracker.to(self.device)
        checkpointer = DetectronCheckpointer(cfg, tracker,
                                              save_dir=self.model_path)
        if os.path.isfile(self.model_path):
            _ = checkpointer.load(self.model_path)
        elif os.path.isdir(self.model_path):
            _ = checkpointer.load(use_latest=True)
        else:
            raise ValueError("No model parameters are loaded.")

        return tracker

    def process(self, frame):
        orig_h, orig_w, _ = frame.shape
        # frame should be RGB image
        frame = self._preprocess(frame)

        with torch.no_grad():
            results = self.tracker(frame.to(self.device))

        assert (len(results) == 1)
        results = results[0].to('cpu')
        results = results.resize([orig_w, orig_h]).convert('xywh')

        return results
