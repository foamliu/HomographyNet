import pickle
import argparse
import cv2
import numpy as np

from config import print_freq
from utils import AverageMeter

MIN_MATCH_COUNT = 10


def compute_homo(img1, img2, args):
    H = np.identity(3)
    if args.type == 'surf':
        try:
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

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        except cv2.error as err:
            print(err)

    elif args.type == 'identity':
        pass

    return H


def compute_mace(H, perturbed_four_points):
    four_points = np.float32([[64, 64], [64, 320], [320, 320], [320, 64]])
    # print('four_points: ' + str(four_points))
    # print('perturbed_four_points: ' + str(perturbed_four_points))
    # print(four_points.shape)
    # print(H)
    pred_four_pints = cv2.perspectiveTransform(np.array([four_points]), H)
    # print('predicted_four_pints: ' + str(pred_four_pints))
    # print(pred_four_pints.shape)
    # print('predicted_four_pints.shape: ' + str(predicted_four_pints.shape))
    error = np.subtract(pred_four_pints, perturbed_four_points)
    # print('error: ' + str(error))
    mace = (np.abs(error)).mean()
    return mace


def test(args):
    filename = 'data/test.pkl'
    with open(filename, 'rb') as file:
        samples = pickle.load(file)

    mace_list = []
    maces = AverageMeter()
    for i, sample in enumerate(samples):
        image, four_points, perturbed_four_points = sample
        img1 = np.zeros((640, 480), np.uint8)
        img1[64:320, 64:320] = image[:, :, 0]
        img2 = np.zeros((640, 480), np.uint8)
        img2[64:320, 64:320] = image[:, :, 1]

        H = compute_homo(img2, img1, args)
        try:
            mace = compute_mace(H, perturbed_four_points)
            mace_list.append(mace)
            maces.update(mace)
        except cv2.error as err:
            print(err)
        if i % print_freq == 0:
            print('[{0}/{1}]\tMean Average Corner Error {mace.val:.5f} ({mace.avg:.5f})'.format(i, len(samples),
                                                                                                mace=maces))

    print('MSE: {:5f}'.format(np.mean(mace_list)))
    print('len(mse_list): ' + str(len(mace_list)))


def parse_args():
    parser = argparse.ArgumentParser(description='Test with SURF+RANSAC')
    # general
    parser.add_argument('--type', type=str, default='surf', help='surf or identity')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    test(args)
