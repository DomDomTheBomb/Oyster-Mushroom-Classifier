# Purpose: takes in the directory of the data and creates a
# data set class that can be used to retrieve images from
# @author: Dominic Sobocinski

import os
import numpy as np
from skimage import io
from skimage import transform as trans
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class OysterMushroom(Dataset):
    def __init__(self, img_dir: list, target: list, use_cache = False, transform=None, img_size=256):
        self.img_labels = target
        self.img_dir = img_dir
        self.data = []
        self.cache = use_cache
        self.img_size = img_size

        if self.cache:
            for i, dir in enumerate(img_dir):
                print(str(i) + "/" + str(len(img_dir)) + ": " + dir)
                image = io.imread(dir)
                image = trans.resize(image, (img_size, img_size))
                self.data.append(image)

        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.cache:
            image = self.data[idx]
        else:
            image = io.imread(self.img_dir[idx])
            #image = image.astype(np.float32) / 255
            #image = np.transpose(image, (2, 0, 1))
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = image.astype(np.float32) / 255
            image = np.transpose(image, (2, 0, 1))
        return image, label


# when this program is ran as main, it basically selects a random image
# from the dataset and shows how it will look like for training the model
if __name__ == "__main__":

    mush_dir = "images/oyster/"
    non_mush_dir = "images/background/"
    img_dir = []
    img_labels = []

    for file in os.listdir(mush_dir):
        img_labels.append(1)
        img_dir.append(mush_dir + file)

    # list of augmentations that we would apply to the images for training
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size = (5, 9), sigma = (0.1, 5)),
        transforms.RandomRotation(degrees = (0, 12)),
        transforms.RandomHorizontalFlip(p = 0.3),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    dataset = OysterMushroom(img_dir, img_labels, transform = transformations)

    import random

    # randomly select an image to visualize itS
    for i in range(0, len(dataset)):
        image = np.array(dataset[random.randint(0, len(dataset))][0])
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)
        plt.show()

