# HomographyNet

This is a deep convolutional neural network for estimating the relative homography between a pair of images. 
Deep Image Homography Estimation [paper](https://arxiv.org/abs/1606.03798) implementation in PyTorch.

## Features

- Backbone: MobileNetV2
- Dataset: MSCOCO 2014 training set

## DataSet

- Train/valid: generated 500,000/41,435 pairs of image patches sized 128x128(rho=32).
- Test: generated 10,000 pairs of image patches sized 256x256(rho=64).


## Dependencies

- Python 3.6.8
- PyTorch 1.3.0


## Usage
### Data Pre-processing
Extract training images:
```bash
$ python3 extract.py
$ python3 pre_process.py
```

### Train
```bash
$ python3 train.py --lr 0.005 --batch-size 64
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

## Test
Homography Estimation Comparison on Warped MS-COCO 14 Test Set.
```bash
$ python3 test.py
$ python3 test_orb.py --type surf
$ python3 test_orb.py --type identity
```
### Result
|Method|Mean Average Corner Error (pixels)|
|---|---|
|HomographyNet|3.53|
|SURF + RANSAC|8.83|
|Identity Homography|32.13|

### Graph
![image](https://gitee.com/foamliu/HomographyNet/raw/master/images/result.jpg)