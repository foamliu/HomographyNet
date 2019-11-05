import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 128

num_samples = 118287
num_train = 98287
num_valid = 10000
num_test = 10000
image_folder = 'data/train2017'
data_file = 'data/data.pkl'

# Training parameters
num_workers = 8  # for data-loading
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data gen parameters
rho = 32
patch_size = 128
top_point = (32, 32)
left_point = (patch_size + 32, 32)
bottom_point = (patch_size + 32, patch_size + 32)
right_point = (32, patch_size + 32)
four_points = [top_point, left_point, bottom_point, right_point]
