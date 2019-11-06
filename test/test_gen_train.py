import cv2 as cv

from pre_process import get_datum

if __name__ == "__main__":
    fullpath = '000000523955.jpg'
    img = cv.imread(fullpath, 0)
    img = cv.resize(img, (320, 240))
    test_image = img.copy()
    size = (320, 240)
    rho = 32
    patch_size = 128

    for top_point in [(32, 32), (160, 32), (32, 128), (160, 128), (128, 88)]:
        # top_point = (rho, rho)
        datum = get_datum(img, test_image, size, rho, top_point, patch_size)
        img1 = datum[0][:, :, 0]
        img2 = datum[0][:, :, 1]
        print('img1.shape: ' + str(img1.shape))
        print('img2.shape: ' + str(img2.shape))
        cv.imshow('img1', img1)
        cv.imshow('img2', img2)
        cv.waitKey(0)
