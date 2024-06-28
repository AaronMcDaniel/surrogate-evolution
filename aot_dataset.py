import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import cv2
from torch.utils.data import DataLoader
import torch
import json
import time

class AOTDataset(Dataset):
    def __init__(self, mode, string=None):
        
        if string == None:
            if mode == 'train':
                self.pickle_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part1/pickles/labels/part1_NOSTRING_TENSOR_labels.pkl'
                self.image_folder = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part1/'
            elif mode == 'val':
                self.pickle_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part2/pickles/labels/part2_NOSTRING_TENSOR_labels.pkl'
                self.image_folder = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part2/'
            elif mode == 'test':
                self.pickle_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part3/pickles/labels/part3_NOSTRING_TENSOR_labels.pkl'
                self.image_folder = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part3/'
        else:
            if mode == 'train':
                self.pickle_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part1/pickles/labels/part1_STRING_TENSOR_labels.pkl'
                self.image_folder = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part1/'
            elif mode == 'val':
                self.pickle_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part2/pickles/labels/part2_STRING_TENSOR_labels.pkl'
                self.image_folder = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part2/'
            elif mode == 'test':
                self.pickle_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part3/pickles/labels/part3_STRING_TENSOR_labels.pkl'
                self.image_folder = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part3/'

        with open(self.pickle_path, 'rb') as f:
            self.labels = pickle.load(f)
        print(mode, 'labels loaded')

        self.string = string
        self.class_mapping_all = {'Airborne1': 0, 'Airborne2': 1, 'Airborne3': 2, 'Airborne4': 3, 'Airborne5': 4, 'Airborne6': 5, 'Airborne7': 6, 'Airborne9': 7, 'Airborne10': 8, 'Airborne11': 9, 'Airborne12': 10, 'Airborne14': 11, 'Airborne16': 12, 'Airborne17': 13, 'Airborne18': 14, 'Airplane1': 15, 'Airplane2': 16, 'Airplane3': 17, 'Airplane4': 18, 'Airplane5': 19, 'Airplane6': 20, 'Airplane7': 21, 'Airplane8': 22, 'Airplane10': 23, 'Bird1': 24, 'Bird2': 25, 'Bird3': 26, 'Bird4': 27, 'Bird5': 28, 'Bird6': 29, 'Bird7': 30, 'Bird8': 31, 'Bird9': 32, 'Bird10': 33, 'Bird15': 34, 'Bird23': 35, 'Drone1': 36, 'Flock1': 37, 'Flock2': 38, 'Flock3': 39, 'Helicopter1': 40, 'Helicopter2': 41, 'Helicopter3': 42}
        self.class_mapping = {'Airborne' : 0, 'Airplane' : 1, 'Bird' : 2, 'Drone' : 3, 'Flock' : 4, 'Helicopter' : 5}
        self.last = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        if self.string is not None:
            label = eval_label(label)
        image_path = label['path']
        try:
            image = cv2.imread(self.image_folder + image_path)
            image = np.array(image)
        except:
            label = self.labels[self.last]
            if self.string is not None:
                label = eval_label(label)
            image_path = label['path']
            image = cv2.imread(self.image_folder + image_path)
        image = cv2.imread(self.image_folder + image_path)
        image = np.array(image, dtype=np.int32)
        image = torch.from_numpy(image)
        self.last = idx
        return image, label

    def find_image(self, frame_id, flight_id):
        if self.string is not None:
            labels = eval_batch(self.labels)
        else:
            labels = self.labels
        for label in labels:
            if label['flight_id'] == flight_id and label['frame_id'] == frame_id:
                output = self.image_folder + label['path']
                return output

def my_collate(batch):
    images = []
    labels = []
    for i, item in enumerate(batch):
        images.append(batch[i][0])
        labels.append(batch[i][1])
    return [images, labels]

def eval_label(label):
    nan = np.nan
    def array(x):
        return np.array(x)
    def tensor(x, dtype=torch.int64):
        return torch.tensor(x)
    return eval(label) 

def eval_batch(batch):
    batch = list(batch)
    for i, label in enumerate(batch):
        batch[i] = eval_label(batch[i])
    return batch