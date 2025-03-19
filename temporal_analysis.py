import argparse
import os
import zipfile

import numpy as np
import pickle
import gc

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

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
data_filename = 'train_1y_Australia2.pkl' # 'sc_256_2y_5.pkl' # load original dataset
data_name, extension = os.path.splitext(data_filename)
results = {data_name: {'psnr': [], 'ssim': []}} #,'Set5': {'psnr': [], 'ssim': []}

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
