import random

import cv2 as cv
import numpy as np
from numpy.linalg import inv

from test_orb import compute_homo

rho = 64
patch_size = 256
top_point = (rho, rho)
left_point = (patch_size + rho, rho)
bottom_point = (patch_size + rho, patch_size + rho)
right_point = (rho, patch_size + rho)
four_points = [top_point, left_point, bottom_point, right_point]

if __name__ == "__main__":
    # np.random.seed(7)
    # random.seed(7)
    fullpath = '000000523955.jpg'
    img = cv.imread(fullpath, 0)
    img = cv.resize(img, (640, 480))
    test_image = img.copy()
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))
    print('perturbed_four_points: ' + str(perturbed_four_points))

    H = cv.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    print(H)
    H_inverse = inv(H)

    warped_image = cv.warpPerspective(img, H_inverse, (640, 480))
    # warped_image = cv.warpPerspective(img, H, (640, 480))

    Ip1 = test_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    Ip2 = warped_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]

    warped_image = cv.polylines(warped_image, [np.int32(four_points)], True, 255, 3, cv.LINE_AA)
    test_image = cv.polylines(test_image, [np.int32(perturbed_four_points)], True, 255, 3, cv.LINE_AA)

    Ip1_new = np.zeros((320, 320), np.uint8)
    Ip1_new[64:, 64:] = Ip1
    Ip2_new = np.zeros((320, 320), np.uint8)
    Ip2_new[64:, 64:] = Ip2

    # pred_H = compute_homo(Ip1, Ip2)
    pred_H = compute_homo(Ip1_new, Ip2_new)
    print(pred_H)
    error = np.reshape(H - pred_H, (9,))
    print(error)
    print(np.square(error).mean())

    # pred_four_pints = cv.perspectiveTransform(np.array(four_points), pred_H)
    Ip3 = cv.warpPerspective(Ip1, pred_H, (256, 256))

    cv.imshow('test_image', test_image)
    cv.imshow('warped_image', warped_image)
    cv.imshow('Ip1', Ip1)
    cv.imshow('Ip2', Ip2)
    cv.imshow('Ip3', Ip3)
    cv.waitKey(0)

    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    print(H_four_points)
