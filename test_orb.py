import pickle

import cv2
import numpy as np
from tqdm import tqdm

MIN_MATCH_COUNT = 10


def compute_homo(img1, img2):
    # Initiate SIFT detector
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    print('len(good): ' + str(len(good)))

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    return None


def compute_mse(H):
    four_points = np.float32([[0, 0], [0, 128], [128, 128], [128, 0]])
    four_points = np.array([four_points])
    print(four_points)
    print(H)
    predicted_four_pints = cv2.perspectiveTransform(four_points, H)
    print('predicted_four_pints.shape: ' + str(predicted_four_pints.shape))
    error = np.subtract(np.array(predicted_four_pints), np.array(four_points))
    mse = (np.square(error)).mean()
    return mse


def test():
    filename = 'data/test.pkl'
    with open(filename, 'rb') as file:
        samples = pickle.load(file)

    mse_list = []
    for sample in tqdm(samples):
        image, H_four_points = sample
        img1 = image[:, :, 0]
        img2 = image[:, :, 1]
        H = compute_homo(img1, img2)
        mse = compute_mse(H)
        mse_list.append(mse)

    print('MSE: {:5f}'.format(np.mean(mse_list)))


if __name__ == "__main__":
    test()
