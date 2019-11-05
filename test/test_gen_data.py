import random

import cv2 as cv
import numpy as np
from numpy.linalg import inv

from config import rho, four_points, top_point, bottom_point

if __name__ == "__main__":
    fullpath = '000000523955.jpg'
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

    cv.imshow('Ip1', Ip1)
    cv.imshow('Ip2', Ip2)
    cv.waitKey(0)

    H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    print(H_four_points)
