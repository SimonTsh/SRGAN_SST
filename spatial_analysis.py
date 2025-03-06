import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import pickle
import gc
import shutil

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
from data_utils import TestTensorDataset, denormalize
from model import Generator

parser = argparse.ArgumentParser(description='Analyse Spatial features')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_95.pth', type=str, help='generator model epoch name') # SRGAN: 101; WGAN: 198
parser.add_argument('--crop_size', default=128, type=int, help='testing images crop size') # 64, 256
parser.add_argument('--test_loc', default='aus', type=str, help='<aus>: Australian West, <scs>: South China Sea')
opt = parser.parse_args()

def img2freq(img_2d):
    sr_image_f = np.fft.fft2(img_2d)
    sr_image_fshifted = np.fft.fftshift(sr_image_f)
    sr_image_mag = np.log(np.abs(sr_image_fshifted) + 1)

    return sr_image_mag


UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name
CROP_SIZE = opt.crop_size
TEST_LOC = opt.test_loc

# Load model and test dataset
if TEST_LOC == 'aus':
    data_filename = f'train_1y_Australia2_test_data_{CROP_SIZE}.pkl'
elif TEST_LOC == 'scs':
    data_filename = f'sc_256_2y_5_test_data_{CROP_SIZE}.pkl'
else:
    KeyError('Not a valid test location')
data_name, extension = os.path.splitext(data_filename)
results = {data_name: {'psnr': [], 'ssim': [], 'psnr_bicubic': [], 'ssim_bicubic': []}}

model = Generator(in_channels=1, out_channels=1, scale_factor=UPSCALE_FACTOR).eval() #Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    model = model.cuda()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

data_source = 'di-lab'
data_dir = 'data/%s/' % (data_source)
with open(f'{data_dir}{data_filename}','rb') as f:
    test_data = pickle.load(f)
gc.enable()
test_HR = test_data['HR']
test_LR = test_data['LR']
test_HR_interp = test_data['HR_interp']

# test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
test_set = TestTensorDataset(test_HR, test_LR, test_HR_interp, upscale_factor=UPSCALE_FACTOR, crop_size=CROP_SIZE)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/di-lab_%s/' % str(UPSCALE_FACTOR)
if not os.path.exists(out_path): # check if file exists
    os.makedirs(out_path)
else:
    if os.listdir(out_path): # check if directory is not empty
        for item in os.listdir(out_path): # remove all contents of the directory
            item_path = os.path.join(out_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        print('directory is empty, nothing to remove')

# Compare reconstructed image with original hr ground truth
index = 0
for lr_image, hr_bicubic_image, hr_image in test_bar:
    image_name = f'test{index}'

    with torch.no_grad():
        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    sr_image = model(lr_image)

    # perform spatial frequency analysis on norm image
    hr_image_mag = img2freq(hr_image.cpu().detach().numpy().squeeze(0).squeeze(0))
    sr_image_mag = img2freq(sr_image.cpu().detach().numpy().squeeze(0).squeeze(0))
    hr_bicubic_img_mag = img2freq(hr_bicubic_image.detach().numpy().squeeze(0).squeeze(0))

    # denormalization with max pixel
    hr_image_denorm = denormalize(hr_image, test_HR[index].min(), test_HR[index].max()) # GLOBAL_HR_MIN, GLOBAL_HR_MAX) # hr_image.cpu().numpy() * test_HR_max.numpy()
    sr_image_denorm = denormalize(sr_image, test_HR[index].min(), test_HR[index].max()) # GLOBAL_HR_MIN, GLOBAL_HR_MAX) # sr_image.cpu().detach().numpy() * test_HR_max.numpy()
    hr_bicubic_img_denorm = denormalize(hr_bicubic_image, test_HR[index].min(), test_HR[index].max()) # GLOBAL_HR_MIN, GLOBAL_HR_MAX) # self-implemented version

    test_images = torch.stack(
        [(hr_bicubic_img_denorm.detach().squeeze(0)),
         (sr_image_denorm.cpu().detach().squeeze(0)),
         (hr_image_denorm.cpu().detach().squeeze(0))])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    for i in range(3):
        axes_flat[i].imshow(test_images[i].squeeze(0), cmap='viridis')
        axes_flat[i].set_axis_off()
    axes_flat[3].imshow(hr_bicubic_img_mag, cmap='gray'); axes_flat[3].set_axis_off()
    axes_flat[4].imshow(sr_image_mag, cmap='gray'); axes_flat[4].set_axis_off()
    axes_flat[5].imshow(hr_image_mag, cmap='gray'); axes_flat[5].set_axis_off()
    
    fig.savefig(f'{out_path}{image_name}_spatialF', dpi=300, bbox_inches='tight')

    plt.close('all')
    index += 1
