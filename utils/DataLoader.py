import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image

import glob
import os
import random


class Loader(data.DataLoader):
    def __init__(self, dataset_dir, styles, cls, transforms):
        super(Loader, self).__init__(self)
        self.dataset_dir = dataset_dir
        self.styles = styles
        folder_A = glob.glob(f'{os.path.join(dataset_dir, styles, cls[0])}/*')
        folder_B = glob.glob(f'{os.path.join(dataset_dir, styles, cls[1])}/*')

        self.transform = transforms
        self.image_path = []

        for i in range(len(folder_A)):
            self.image_path.append([folder_A[i], 0])

        for i in range(len(folder_B)):
            self.image_path.append([folder_B[i], 1])

    def __getitem__(self, index):

        item = self.transform(Image.open(self.image_path[index][0]))
        label = self.image_path[index][1]

        return [item, label]

    def __len__(self):
        return len(self.image_path)