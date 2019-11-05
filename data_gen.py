import pickle

import numpy as np
from torch.utils.data import Dataset


class DeepHNDataset(Dataset):
    def __init__(self, split):
        filename = 'data/{}.pkl'.format(split)
        with open(filename, 'rb') as file:
            samples = pickle.load(file)

        self.samples = samples

    def __getitem__(self, i):
        sample = self.samples[i]
        image, H_four_points = sample
        img = np.zeros((128, 128, 3), np.float32)
        img[:, :, 0:2] = image / 255.
        img = np.transpose(img, (2, 0, 1))  # HxWxC array to CxHxW
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
