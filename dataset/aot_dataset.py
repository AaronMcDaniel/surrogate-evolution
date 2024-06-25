import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import cv2
from torch.utils.data import DataLoader

class AOTDataset(Dataset):
    def __init__(self, mode):
        if mode == 'train':
            self.pickle_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part1/pickles/labels/part1_TOTAL_labels.pkl'
            self.image_folder = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part1/'
        elif mode == 'val':
            self.pickle_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part2/pickles/labels/part2_TOTAL_labels.pkl'
            self.image_folder = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part2/'
        elif mode == 'test':
            self.pickle_path = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part3/pickles/labels/part3_TOTAL_labels.pkl'
            self.image_folder = '/gv1/projects/GRIP_Precog_Opt/data_loading/airborne-detection-starter-kit-master/data/part3/'

        with open(self.pickle_path, 'rb') as f:
            self.labels = pickle.load(f)
        print('labels loaded')

        self.class_mapping = {'Airborne1': 0, 'Airborne2': 1, 'Airborne3': 2, 'Airborne4': 3, 'Airborne5': 4, 'Airborne6': 5, 'Airborne7': 6, 'Airborne9': 7, 'Airborne10': 8, 'Airborne11': 9, 'Airborne12': 10, 'Airborne14': 11, 'Airborne16': 12, 'Airborne17': 13, 'Airborne18': 14, 'Airplane1': 15, 'Airplane2': 16, 'Airplane3': 17, 'Airplane4': 18, 'Airplane5': 19, 'Airplane6': 20, 'Airplane7': 21, 'Airplane8': 22, 'Airplane10': 23, 'Bird1': 24, 'Bird2': 25, 'Bird3': 26, 'Bird4': 27, 'Bird5': 28, 'Bird6': 29, 'Bird7': 30, 'Bird8': 31, 'Bird9': 32, 'Bird10': 33, 'Bird15': 34, 'Bird23': 35, 'Drone1': 36, 'Flock1': 37, 'Flock2': 38, 'Flock3': 39, 'Helicopter1': 40, 'Helicopter2': 41, 'Helicopter3': 42}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_path = label['path']
        image = cv2.imread(self.image_folder + image_path)
        image = np.array(image)
        return image, label

dataset = AOTDataset('train')
trainloader = DataLoader(dataset, batch_size=64, shuffle=True)
total = 0
for image_batch, label_batch in trainloader:
    #img = cv2.imread(image_batch[0])
    for i, x in enumerate(label_batch['classes']):
        for j, y in enumerate(x):
            #print(int(y.numpy()))

            if int(y.numpy()) == 42:
                print(total)
                print(label_batch['ranges'][i][j])
                cv2.imwrite('THISONETHISONE.jpg', image_batch[i].numpy())
                total += 1
                break