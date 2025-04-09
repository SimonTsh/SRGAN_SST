import argparse
import os
from math import log10

import numpy as np
import pickle
import gc

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from data_utils import clear_directory, denormalize, load_original_data, TestTensorDataset
from model import Generator

parser = argparse.ArgumentParser(description='Test Global Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_89.pth', type=str, help='generator model epoch name') # SRGAN: 101; WGAN: 198
parser.add_argument('--crop_size', default=256, type=int, help='testing images crop size') # 64, 256
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name
CROP_SIZE = opt.crop_size

# Load model and test dataset
data_filename = 'train_1y_Australia2.pkl' # 'sc_256_2y_5.pkl' #load original dataset
data_name, extension = os.path.splitext(data_filename)
results = {data_name: {'dist_sr_hr': [], 'dist_bi_hr': [], 'time': []}}

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
data_HR_interp, data_HR, data_LR = load_original_data(test_data)
test_HR = []; test_LR = []; test_HR_interp = []; pos = []; norm_time = []
for i, data in enumerate(data_HR):
    test_HR.append(data[0][0])
    test_LR.append(data_LR[i][0])
    test_HR_interp.append(data_HR_interp[i][0].squeeze(0))
    pos.append(data[1])
    norm_time.append(data[2])

num_sample = np.shape(test_HR)[0] # // 10 # for debugging
test_HR = test_HR[:num_sample]; test_LR = test_LR[:num_sample]
# test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
test_set = TestTensorDataset(test_HR, test_LR, test_HR_interp, upscale_factor=UPSCALE_FACTOR, crop_size=CROP_SIZE)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/di-lab_%s/' % str(UPSCALE_FACTOR)
clear_directory(out_path)

# Create global map of reconstructed images
index = 0; sst_image = []; hr_image_all = []; lr_image_all = []
for lr_image, hr_bicubic_image, hr_image in test_bar:
    with torch.no_grad():
        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    sr_image = model(lr_image)

    # denormalization with max pixel
    hr_image_denorm = denormalize(hr_image, test_HR[index].min(), test_HR[index].max()) # hr_image.cpu().numpy() * test_HR_max.numpy()
    sr_image_denorm = denormalize(sr_image, test_HR[index].min(), test_HR[index].max()) # sr_image.cpu().detach().numpy() * test_HR_max.numpy()
    bi_image_denorm = denormalize(hr_bicubic_image, test_HR[index].min(), test_HR[index].max())
    lr_image_denorm = denormalize(lr_image, test_LR[index].min(), test_LR[index].max()) # lr_image.cpu().numpy() * test_LR_max.numpy()
    
    _, _, img_w, img_h = hr_image_denorm.shape
    sr_image_sst = sr_image_denorm.cpu().squeeze(0).squeeze(0)[img_w//2][img_h//2]
    hr_image_sst = hr_image_denorm.cpu().squeeze(0).squeeze(0)[img_w//2][img_h//2]
    bi_image_sst = bi_image_denorm.cpu().squeeze(0).squeeze(0)[img_w//2][img_h//2]
    dist_sr_hr_sst = abs(sr_image_sst.detach() - hr_image_sst)
    dist_bi_hr = abs(bi_image_sst.detach() - hr_image_sst)
    results[data_name]['dist_sr_hr'].append(dist_sr_hr_sst)
    results[data_name]['dist_bi_hr'].append(dist_bi_hr)
    results[data_name]['time'].append(norm_time[index])

    index += 1

out_path = 'statistics/'
dates = [(datetime(2023, 1, 1) + timedelta(days=int(t*360))).date().isoformat() for t in results[data_name]['time']] # Precompute all date strings

# plotting dist-to-coast trend
fig, ax = plt.subplots()
ax.set_xlabel('norm time')
ax.scatter(dates, np.array(results[data_name]['dist_bi_hr']), color='tab:blue', s=15) # alpha=0.7) # dist_sr_hr
ax.set_ylabel('MAE (degrees)')
ax.grid(True)
if len(dates) > 1000:
    n = len(dates) // 1000
else:
    n = len(dates) // 100 
ax.xaxis.set_major_locator(plt.MultipleLocator(n))
plt.xticks(rotation=70)

fig.savefig(f'{out_path}{data_name}_temporalAnalysis.png', dpi=300, bbox_inches='tight')
plt.close('all')