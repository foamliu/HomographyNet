import pickle

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from config import im_size


class DeepHNDataset(Dataset):
    def __init__(self, split):
        filename = 'data/{}.pkl'.format(split)
        print('loading {}...'.format(filename))
        with open(filename, 'rb') as file:
            samples = pickle.load(file)
        np.random.shuffle(samples)
        self.split = split
        self.samples = samples

    def __getitem__(self, i):
        sample = self.samples[i]
        image, four_points, perturbed_four_points = sample
        img0 = image[:, :, 0]
        img0 = cv.resize(img0, (im_size, im_size))
        img1 = image[:, :, 1]
        img1 = cv.resize(img1, (im_size, im_size))
        img = np.zeros((im_size, im_size, 3), np.float32)
        img[:, :, 0] = img0 / 255.
        img[:, :, 1] = img1 / 255.
        img = np.transpose(img, (2, 0, 1))  # HxWxC array to CxHxW
        H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
        target = np.reshape(H_four_points, (8,))
        return img, target

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    train = DeepHNDataset('train')
    print('num_train: ' + str(len(train)))
    valid = DeepHNDataset('valid')
    print('num_valid: ' + str(len(valid)))

    print(train[0])
    print(valid[0])
