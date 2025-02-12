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

import matplotlib.pyplot as plt
import pytorch_ssim
from data_utils import TestTensorDataset, denormalize, denormalize_mean_std
from model import Generator

parser = argparse.ArgumentParser(description='Test Real Measurement Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_85.pth', type=str, help='generator model epoch name') # SRGAN: 101; WGAN: 198
parser.add_argument('--crop_size', default=64, type=int, help='testing images crop size') # 64, 256
parser.add_argument('--test_loc', default='scs', type=str, help='<aus>: Australian West, <scs>: South China Sea')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name
CROP_SIZE = opt.crop_size
TEST_LOC = opt.test_loc
GLOBAL_HR_MIN = 279
GLOBAL_HR_MAX = 306
GLOBAL_MEAN = 0.509
GLOBAL_STD = 0.194

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
test_HR_bicubic = test_data['HR_interp']

# test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
test_set = TestTensorDataset(test_HR, test_LR, upscale_factor=UPSCALE_FACTOR, crop_size=CROP_SIZE)
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
    with torch.no_grad():
        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    sr_image = model(lr_image)

    # denormalization with max pixel
    # hr_image_denorm = denormalize_mean_std(hr_image, GLOBAL_MEAN, GLOBAL_STD)
    # sr_image_denorm = denormalize_mean_std(sr_image, GLOBAL_MEAN, GLOBAL_STD)
    # hr_bicubic_img_denorm = denormalize_mean_std(hr_bicubic_image, GLOBAL_MEAN, GLOBAL_STD)
    hr_image_denorm = denormalize(hr_image, test_HR[index].min(), test_HR[index].max()) # GLOBAL_HR_MIN, GLOBAL_HR_MAX) # hr_image.cpu().numpy() * test_HR_max.numpy()
    sr_image_denorm = denormalize(sr_image, test_HR[index].min(), test_HR[index].max()) # GLOBAL_HR_MIN, GLOBAL_HR_MAX) # sr_image.cpu().detach().numpy() * test_HR_max.numpy()
    hr_bicubic_img_denorm = denormalize(hr_bicubic_image, test_HR[index].min(), test_HR[index].max()) # GLOBAL_HR_MIN, GLOBAL_HR_MAX) # self-implemented version
    # hr_bicubic_img_denorm = denormalize(normalize_to_01(test_HR_bicubic[index]), test_HR[index].min(), test_HR[index].max()).unsqueeze(0).unsqueeze(0) # from dataset
    # lr_image_denorm = denormalize(lr_image, test_LR[index].min(), test_LR[index].max()) # lr_image.cpu().numpy() * test_LR_max.numpy()
    
    # calculate superresolution parameters
    # mse = ((hr_image_denorm - sr_image_denorm) ** 2).mean() # denorm
    # psnr = 10 * log10(test_HR[index].max() ** 2 / mse) # denorm # GLOBAL_HR_MAX ** 2 / mse) # 
    # # print(f'mse, psnr (denorm) = {mse}, {psnr}')

    mse = ((hr_image - sr_image) ** 2).data.mean() # norm
    psnr = 10 * log10(1 / mse) # norm
    # print(f'mse, psnr (norm) = {mse}, {psnr}')
    ssim = pytorch_ssim.ssim(sr_image, hr_image).item() #.data[0]; norm

    # calculate bicubic parameters
    # mse_bicubic = ((hr_image_denorm.cpu() - hr_bicubic_img_denorm) ** 2).data.mean()
    # psnr_bicubic = 10 * log10(test_HR[index].max() ** 2 / mse_bicubic) # GLOBAL_HR_MAX ** 2 / mse_bicubic) # 
    # # print(f'mse, psnr (denorm) = {mse_bicubic}, {psnr_bicubic}')

    mse_bicubic = ((hr_image.data.cpu() - hr_bicubic_image) ** 2).data.mean()
    psnr_bicubic = 10 * log10(1 / mse_bicubic)
    # print(f'mse, psnr (norm) = {mse_bicubic}, {psnr_bicubic}')
    ssim_bicubic = pytorch_ssim.ssim(hr_bicubic_image, hr_image.data.cpu()).item()

    # save psnr/ssim
    results[data_name]['psnr'].append(psnr)
    results[data_name]['ssim'].append(ssim)
    results[data_name]['psnr_bicubic'].append(psnr_bicubic)
    results[data_name]['ssim_bicubic'].append(ssim_bicubic)

    # save test images
    test_images = torch.stack(
        [(hr_bicubic_img_denorm.squeeze(0)),
         (sr_image_denorm.cpu().squeeze(0)),
         (hr_image_denorm.cpu().squeeze(0))])
    image_name = f'test{index}'
    # image = utils.make_grid(test_images, nrow=3, padding=5)
    # utils.save_image(test_images, out_path + image_name + '_psnr_%.4f_ssim_%.4f_psnrBC_%.4f_ssimBC_%.4f.png' % (psnr, ssim, psnr_bicubic, ssim_bicubic), padding=5)
    # # utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) + image_name.split('.')[-1], padding=5) # TODO: change to colour plot
    
    image = test_images.detach()
    fig, axes = plt.subplots(1, 3, figsize=(20, 20))
    for i, ax in enumerate(axes):
        im = ax.imshow(image[i].squeeze(0), cmap='viridis')
        ax.set_axis_off()
        cax = fig.add_axes([ax.get_position().x1 + 0.005, ax.get_position().y0, 0.02, ax.get_position().height])
        fig.colorbar(im, cax=cax)
    fig.savefig(out_path + image_name + '_psnr_%.4f_ssim_%.4f_psnrBC_%.4f_ssimBC_%.4f.png' 
                % (psnr, ssim, psnr_bicubic, ssim_bicubic), 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')
    index += 1

out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': [], 'psnr_bicubic': [], 'ssim_bicubic': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    psnr_bicubic = np.array(item['psnr_bicubic'])
    ssim_bicubic = np.array(item['ssim_bicubic'])
    if (len(psnr) == 0) or (len(ssim) == 0) or (len(psnr_bicubic) == 0) or (len(ssim_bicubic) == 0):
        psnr = 'No data'
        ssim = 'No data'
        psnr_bicubic = 'No data'
        ssim_bicubic = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
        psnr_bicubic = psnr_bicubic.mean()
        ssim_bicubic = ssim_bicubic.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)
    saved_results['psnr_bicubic'].append(psnr_bicubic)
    saved_results['ssim_bicubic'].append(ssim_bicubic)
saved_results['model'] = MODEL_NAME
saved_results['crop_size'] = CROP_SIZE

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame_filepath = out_path + data_source + '_' + str(UPSCALE_FACTOR) + '_test_results.csv'
data_frame.to_csv(data_frame_filepath, index_label='DataSet', mode='a', header=not os.path.exists(data_frame_filepath))
