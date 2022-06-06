import os
import torch
import cv2
import config
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class LoadData(Dataset):
    def __init__(self, img_dir):
        self.dir = img_dir
        self.images_path = os.listdir(self.dir)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = np.array(Image.open(f'{self.dir}/{self.images_path[item]}')) / 255.0
        y = img[:, :600, :]
        y = cv2.resize(y, config.IMAGE_SHAPE)
        x = img[:, 600:, :]
        x = cv2.resize(x, config.IMAGE_SHAPE)
        return torch.tensor(x.reshape(3, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]),
                            dtype=torch.float32), torch.tensor(
            y.reshape(3, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), dtype=torch.float32)


# data = LoadData('/home/prasanna/dataset/pix2pix-data/maps/maps/val')
# print(data.__len__())
