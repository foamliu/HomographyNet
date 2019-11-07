import cv2 as cv

from pre_process import get_datum

if __name__ == "__main__":
    fullpath = '000000523955.jpg'
    size = (320, 240)
    rho = 32
    patch_size = 128

    img = cv.imread(fullpath, 0)
    img = cv.resize(img, size)
    test_image = img.copy()

    for top_point in [(0 + 32, 0 + 32), (128 + 32, 0 + 32), (0 + 32, 48 + 32), (128 + 32, 48 + 32), (64 + 32, 24 + 32)]:
        # top_point = (rho, rho)
        datum = get_datum(img, test_image, size, rho, top_point, patch_size)
        img1 = datum[0][:, :, 0]
        img2 = datum[0][:, :, 1]
        print('img1.shape: ' + str(img1.shape))
        print('img2.shape: ' + str(img2.shape))
        cv.imshow('img1', img1)
        cv.imshow('img2', img2)
        cv.waitKey(0)
