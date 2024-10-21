"""
AOT dataset and sampler classes.
"""

import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import cv2
from torch.utils.data import DataLoader, Sampler, SequentialSampler, BatchSampler, RandomSampler
import torch
import json
import time
from tqdm import tqdm
import datetime
from torchvision.datasets import DatasetFolder
import glob

class AOTDataset(Dataset):
    def __init__(self, mode, string=None, bad_data=None, cache_thresh=1, max_size=None, seed=None):
        # decides which pickle_path and image_folder to use
        if bad_data == None:
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
                    self.pickle_path = 'aot_data/train/Labels/part1_STRING_TENSORV2_SHRUNKEN_labels.pkl'
                    self.image_folder = 'aot_data/train'
                elif mode == 'val':
                    self.pickle_path = 'aot_data/val/Labels/part2_STRING_TENSORV2_SHRUNKEN_labels.pkl'
                    self.image_folder = 'aot_data/val'
                elif mode == 'test':
                    self.pickle_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part3/pickles/labels/part3_STRING_TENSORV2_labels.pkl'
                    self.image_folder = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part3/'
        else:
            self.pickle_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part1/pickles/labels/part1_STRING_TENSOR_CATERED_ERRORS_labels.pkl'
            self.image_folder = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part1/'

        self.string = string
        # loads the pickle file to get image labels
        with open(self.pickle_path, 'rb') as f:
            self.labels = pickle.load(f)
        
        print(f'{mode} LABELS LOADED!')
        # TEMPORARY FIX
        init_len = len(self.labels)
        self.labels = [label for label in self.labels if self.image_exists(label)]
        final_len = len(self.labels)
        print(f"Filtered {init_len - final_len} labels with missing images.")

        # initialize a random generator
        self.np_gen = np.random.default_rng(seed)

        # a cache to store faulty filenames and the time threshold they are valid for
        self.flight_mapping = {}
        self.cache_thresh = cache_thresh

        self.class_mapping_all = {'Airborne1': 0, 'Airborne2': 1, 'Airborne3': 2, 'Airborne4': 3, 'Airborne5': 4, 'Airborne6': 5, 'Airborne7': 6, 'Airborne9': 7, 'Airborne10': 8, 'Airborne11': 9, 'Airborne12': 10, 'Airborne14': 11, 'Airborne16': 12, 'Airborne17': 13, 'Airborne18': 14, 'Airplane1': 15, 'Airplane2': 16, 'Airplane3': 17, 'Airplane4': 18, 'Airplane5': 19, 'Airplane6': 20, 'Airplane7': 21, 'Airplane8': 22, 'Airplane10': 23, 'Bird1': 24, 'Bird2': 25, 'Bird3': 26, 'Bird4': 27, 'Bird5': 28, 'Bird6': 29, 'Bird7': 30, 'Bird8': 31, 'Bird9': 32, 'Bird10': 33, 'Bird15': 34, 'Bird23': 35, 'Drone1': 36, 'Flock1': 37, 'Flock2': 38, 'Flock3': 39, 'Helicopter1': 40, 'Helicopter2': 41, 'Helicopter3': 42}
        self.class_mapping = {'Airborne' : 0, 'Airplane' : 1, 'Bird' : 2, 'Drone' : 3, 'Flock' : 4, 'Helicopter' : 5}
        
        # if max size is set, take a subset of the labels 
        if max_size is not None and max_size < self.__len__():
            labels_subset = self.labels[:]
            labels_subset = np.array(labels_subset)
            self.np_gen.shuffle(labels_subset)
            self.labels = list(labels_subset[:max_size])

    def image_exists(self, label):
        if isinstance(label, str):
            label = eval_label(label)
        image_path = os.path.join(self.image_folder, label['path'])
        return os.path.exists(image_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        # if labels were serialized as a string, evaluate it back into a dictionary
        if self.string is not None:
            label = eval_label(label)
        # load the image as a tensor using the path stored in the label  
        image_path = os.path.join(self.image_folder, label['path'])   
        # image_path_str = self.image_folder + label['path']
        # if not os.path.exists(os.path.join(image_path_str)):
        #     print('Image does not exist')
        #     return None
        image = cv2.imread(image_path)
        image = np.array(image)
        image = torch.from_numpy(image)
        
        return image, label

    def get_label(self, idx):
        # same as getitem but doesn't return the image
        label = self.labels[idx]
        if self.string is not None:
            label = eval_label(label)
        return label

    def find_image(self, frame_id, flight_id):
        # returns the image when given frame_id and flight_id
        for label in self.labels:
            if self.string is not None:
                label = eval_label(label)
            if label['flight_id'] == flight_id and label['frame_id'] == frame_id:
                    image = cv2.imread(self.image_folder + label['path'])
                    image = np.array(image)
                    image = torch.from_numpy(image)
                    return image, label

    def filter_labels(self):
        blacklist = []
        labels = self.labels
        if self.string is not None:
            labels = eval_batch(labels)
        for label in labels:
            path = label['path']
            file_path = self.image_folder + path


    def quick_filter(self, blacklist):
        for label in self.labels:
            if self.string is not None:
                label = eval_label(label)
            if 'Images/' + label['path'] in blacklist:
                self.labels.remove(label)
    
    def is_valid_file(self, label):
        label = eval_label(label)
        path = label['path']
        parts = path.split('/')
        root_dir = self.image_folder + parts[0] + '/' + parts[1] + '/'
        test = parts[2] + '*'
        files = glob.glob(test, root_dir=root_dir, include_hidden=False)
        if len(files) > 1:
            return False
        else:
            return True

        


def my_collate(batch):
    # makes the dataloader return a list of dictionaries instead of the default dictionary of lists
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
    return eval(label) #, {'nan': np.nan, 'array': self.array})

def eval_batch(batch):
    batch = list(batch)
    for i, label in enumerate(batch):
        batch[i] = eval_label(batch[i])
    return batch



class AOTSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=None):
        self.dataset = dataset
        self.torch_gen = torch.Generator()
        if seed is not None:
            self.torch_gen.manual_seed(seed)
        self.sampler = RandomSampler(dataset, replacement=False, generator=self.torch_gen)
        self.batch_size = batch_size

    def __len__(self):
        return int(self.dataset.__len__() / self.batch_size)

    def __iter__(self):
        sampler_iter = iter(self.sampler)
        while True:
            try:
                # constructs a batch of only valid images
                batch = []
                while len(batch) < self.batch_size:
                    test = next(sampler_iter)
                    # keeps gathering samples until it finds a valid one to add to the batch
                    while self.eval_idx(test) == False:
                        test = next(sampler_iter) 
                    batch.append(test)
                yield batch
            except StopIteration:
                break

    def eval_idx(self, idx):
        label = self.dataset.get_label(idx)
        flight_id = label['flight_id']
        img_name = label['path'].split('/')[2]
       
        # checks cache first to see if the cached flight is still valid based on time threshold
        if flight_id in self.dataset.flight_mapping:
            old_time = self.dataset.flight_mapping[flight_id][1]
            new_time = time.perf_counter()
            if (new_time - old_time) / 60 / 60 >= self.dataset.cache_thresh:
                del self.dataset.flight_mapping[flight_id]
        # if flight is not in cache, create a list of all non-unique files and add them to dict
        if flight_id not in self.dataset.flight_mapping:
            root_dir = self.dataset.image_folder + '/Images/'
            files = os.listdir(root_dir + flight_id)
            #print(files)
            unique = []
            non_unique_files = []

            files_np = np.array(files)
            unique_files, counts = np.unique(files_np, return_counts=True)
            non_unique_mask = counts > 1
            non_unique_files = unique_files[non_unique_mask]
            self.dataset.flight_mapping[flight_id] = (non_unique_files, time.perf_counter())
        #m checks if specific image is in cache to verify it
        if img_name in self.dataset.flight_mapping[flight_id][0]:
            return False
        else:
            return True