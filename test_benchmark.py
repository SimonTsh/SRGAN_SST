import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import pickle
import gc
import shutil

import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestTensorDataset, display_transform # ,TestDatasetFromFolder
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_93.pth', type=str, help='generator model epoch name') # 100: non-aug, 35: aug, 93: wgan-gp
opt = parser.parse_args()

def load_data(data):
    data_size = len(data)
    c, patch_hr_h, patch_hr_w = data[0][0].shape # need to include channel size allocation
    c, patch_lr_h, patch_lr_w = data[0][1].shape

    image_HR = torch.empty(data_size, c, patch_hr_h, patch_hr_w)
    image_LR = torch.empty(data_size, c, patch_lr_h, patch_lr_w)
    for index, value in enumerate(data):
        image_HR[index,:,:,:], image_LR[index,:,:,:] = value

    return image_HR, image_LR


data_name = 'di-lab'
results = {data_name: {'psnr': [], 'ssim': []}} #,'Set5': {'psnr': [], 'ssim': []}

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

model = Generator(in_channels=1, out_channels=1, scale_factor=UPSCALE_FACTOR).eval() #Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

data_dir = 'data/%s/' % (data_name)
test_dir = data_dir + 'test_data.pkl'
with open(test_dir,'rb') as f:
    test_data = pickle.load(f)
gc.enable()
test_HR, test_LR = load_data(test_data) # test_data[0], test_data[1]

# test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
test_set = TestTensorDataset(test_HR, test_LR, upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=1, shuffle=False) # num_workers = 4
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/di-lab_%s/' % str(UPSCALE_FACTOR)
if not os.path.exists(out_path):
    os.makedirs(out_path)
else:
    if os.listdir(out_path): # check if directory is not empty
        for item in os.listdir(out_path): # remove all contents of the directory
            item_path = os.path.join(out_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)


index = 0
for lr_image, hr_restore_img, hr_image in test_bar:
    with torch.no_grad():
        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    sr_image = model(lr_image)
    mse = ((hr_image - sr_image) ** 2).data.mean()
    psnr = 10 * log10(1 / mse)
    ssim = pytorch_ssim.ssim(sr_image, hr_image).item() #.data[0]

    test_images = torch.stack(
        [display_transform()(hr_restore_img.squeeze(0)), 
         display_transform()(hr_image.data.cpu().squeeze(0)),
         display_transform()(sr_image.data.cpu().squeeze(0))])
    image = utils.make_grid(test_images, nrow=3, padding=5)
    image_name = 'test' + str(index)
    utils.save_image(image, out_path + image_name + '_psnr_%.4f_ssim_%.4f.png' % (psnr, ssim), padding=5)
    # utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) + image_name.split('.')[-1], padding=5)

    # save psnr\ssim
    results[data_name]['psnr'].append(psnr) #0
    results[data_name]['ssim'].append(ssim) #0
    # results[image_name.split('_')[2]]['psnr'].append(psnr)
    # results[image_name.split('_')[2]]['ssim'].append(ssim)
    index += 1

out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + data_name + '_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')
