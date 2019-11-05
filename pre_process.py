import os
import pickle
import random

import cv2 as cv
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

from config import image_folder, rho, four_points, top_point, bottom_point
from config import num_train, num_valid, num_test, train_file, valid_file, test_file


### This function is provided by Mez Gebre's repository "deep_homography_estimation"
#   https://github.com/mez/deep_homography_estimation
#   Dataset_Generation_Visualization.ipynb
def process(files):
    samples = []
    for f in tqdm(files):
        fullpath = os.path.join(image_folder, f)
        img = cv.imread(fullpath, 0)
        img = cv.resize(img, (320, 240))
        test_image = img.copy()
        perturbed_four_points = []
        for point in four_points:
            perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

        H = cv.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
        H_inverse = inv(H)

        warped_image = cv.warpPerspective(img, H_inverse, (320, 240))

        Ip1 = test_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
        Ip2 = warped_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]

        training_image = np.dstack((Ip1, Ip2))
        H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
        datum = (training_image, H_four_points)
        samples.append(datum)

    return samples


if __name__ == "__main__":
    files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
    np.random.shuffle(files)
    samples = process(files)
    print(len(samples))

    train = samples[:num_train]
    valid = samples[num_train:num_train + num_valid]
    test = samples[num_train + num_valid:num_train + num_valid + num_test]

    print(len(train))
    print(len(valid))
    print(len(test))

    with open(train_file, 'wb') as f:
        pickle.dump(train, f)
    with open(valid_file, 'wb') as f:
        pickle.dump(valid, f)
    with open(test_file, 'wb') as f:
        pickle.dump(test, f)
