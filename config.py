import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 256
batch_size = 32

num_samples = 118287
num_train = 98287
num_valid = 10000
num_test = 10000
image_folder = 'data/train2017'
train_file = 'data/train.pkl'
valid_file = 'data/valid.pkl'
test_file = 'data/test.pkl'

# Training parameters
num_workers = 8  # for data-loading
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data gen parameters
rho = 64
patch_size = 256
top_point = (64, 64)
left_point = (patch_size + 64, 64)
bottom_point = (patch_size + 64, patch_size + 64)
right_point = (64, patch_size + 64)
four_points = [top_point, left_point, bottom_point, right_point]
