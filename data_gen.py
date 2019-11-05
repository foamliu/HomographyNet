import json
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
        filename = '{}.json'.format(split)
        with open(filename, 'r') as file:
            samples = json.load(file)

        self.samples = samples

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        before = sample['before']
        full_path = os.path.join(image_folder, before)
        img_0 = cv.imread(full_path)
        img_0 = img_0[..., ::-1]  # RGB
        img_0 = transforms.ToPILImage()(img_0)
        img_0 = self.transformer(img_0)

        after = sample['after']
        full_path = os.path.join(image_folder, after)
        img_1 = cv.imread(full_path)
        img_1 = img_1[..., ::-1]  # RGB
        img_1 = transforms.ToPILImage()(img_1)
        img_1 = self.transformer(img_1)

        # the second input should be ranked higher
        if random.random() > 0.5:
            return img_0, img_1, 0.
        else:
            return img_1, img_0, 1.

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    train = DeepHNDataset('train')
    print('num_train: ' + str(len(train)))
    valid = DeepHNDataset('valid')
    print('num_valid: ' + str(len(valid)))

    print(train[0])
    print(valid[0])
