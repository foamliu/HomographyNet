import pickle
import os
import random

import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms

from config import image_folder

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class DeepHNDataset(Dataset):
    def __init__(self, split):
        filename = '{}.pkl'.format(split)
        with open(filename, 'rb') as file:
            samples = pickle.load(file)

        self.samples = samples

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        image, H_four_points = sample
        print(image.shape)
        print(H_four_points)
        return image

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    train = DeepHNDataset('train')
    print('num_train: ' + str(len(train)))
    valid = DeepHNDataset('valid')
    print('num_valid: ' + str(len(valid)))

    print(train[0])
    print(valid[0])
