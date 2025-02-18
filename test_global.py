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
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta

import pytorch_ssim
from data_utils import denormalize, load_original_data, TestTensorDataset
from model import Generator

parser = argparse.ArgumentParser(description='Test Real Measurement Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_85.pth', type=str, help='generator model epoch name') # SRGAN: 101; WGAN: 198
parser.add_argument('--crop_size', default=256, type=int, help='testing images crop size') # 64, 256
opt = parser.parse_args()


def animate(i):
    masks = times == unique_time[i]
    masks_true = np.where(masks)[0]

    ax.clear()
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='--')
    ax.set_extent([min_lon-4, max_lon+4, min_lat-2, max_lat+2])
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    for mask in masks_true:
        img = ax.imshow(sst_images[mask], extent=[pos_latlon[mask,0,1], pos_latlon[mask,1,1], 
                                            pos_latlon[mask,0,0], pos_latlon[mask,2,0]], 
                                            origin='lower', cmap='viridis', 
                                            vmin=sst_images.min(), vmax=sst_images.max())
        # Create or update the colorbar
        animate.cbar = plt.colorbar(img, cax=cax) # Create the colorbar if it doesn't exist
        img.set_clim(vmin=sst_images.min(), vmax=sst_images.max())  # Update the image limits
        animate.cbar.update_normal(img)  # Update the colorbar
        fig.canvas.draw_idle()  # Redraw the figure
    
    ax.set_title(f'Time: {(datetime(int(2023), 1, 1) + timedelta(days=int(unique_time[i]*360))).date()}')    

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name
CROP_SIZE = opt.crop_size

# Load model and test dataset
data_filename = 'train_1y_Australia2.pkl' # 'sc_256_2y_5.pkl' # load original dataset # 
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

num_sample = np.shape(test_HR)[0] # for debugging
test_HR = test_HR[:num_sample]; test_LR = test_LR[:num_sample]
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
    sr_bicubic_img_denorm = denormalize(hr_bicubic_image, test_HR[index].min(), test_HR[index].max())
    lr_image_denorm = denormalize(lr_image, test_LR[index].min(), test_LR[index].max()) # lr_image.cpu().numpy() * test_LR_max.numpy()
    
    # calculate superresolution parameters
    mse = ((hr_image_denorm - sr_image_denorm) ** 2).mean() # denorm
    psnr = 10 * log10(test_HR[index].max() ** 2 / mse) # denorm
    # print(f'psnr (denorm) = {psnr}')

    mse = ((hr_image - sr_image) ** 2).data.mean() # norm
    psnr = 10 * log10(1 / mse) # norm
    # print(f'psnr (norm) = {psnr}')
    ssim = pytorch_ssim.ssim(sr_image, hr_image).item() #.data[0]; norm

    # calculate bicubic parameters
    mse_bicubic = ((hr_image.data.cpu() - hr_bicubic_image) ** 2).data.mean()
    psnr_bicubic = 10 * log10(1 / mse_bicubic)
    ssim_bicubic = pytorch_ssim.ssim(hr_bicubic_image, hr_image.data.cpu()).item()

    # save psnr/ssim
    results[data_name]['psnr'].append(psnr)
    results[data_name]['ssim'].append(ssim)

    # save test images
    test_images = torch.stack(
        [(sr_bicubic_img_denorm.squeeze(0)),
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

    sst_image.append(sr_image_denorm.cpu().detach().squeeze(0).squeeze(0))
    hr_image_all.append(hr_image_denorm.cpu().detach().squeeze(0).squeeze(0))
    lr_image_all.append(sr_bicubic_img_denorm.cpu().detach().squeeze(0).squeeze(0))
    index += 1

out_path = 'statistics/'

# Create a figure with a map projection function to animate
sst_images = np.array(sst_image)[:num_sample] # sst_image # hr_image_all # lr_image_all
times = np.array(norm_time).squeeze(-1)[:num_sample]
pos_latlon = np.array(pos)[:num_sample]
unique_time = np.unique(times)
min_lat = pos_latlon[:,:,0].min()
max_lat = pos_latlon[:,:,0].max()
min_lon = pos_latlon[:,:,1].min()
max_lon = pos_latlon[:,:,1].max()

# Add map features
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ani = animation.FuncAnimation(fig, animate, frames=len(unique_time), interval=500)
plt.show()
writer = PillowWriter(fps=5)
ani.save(f'{out_path}{data_name}_map.gif', writer=writer)

# fig.savefig(out_path + data_name + '_map.png', dpi=300)
plt.close('all')

# Save quantitative results
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
saved_results['model'] = MODEL_NAME
saved_results['crop_size'] = CROP_SIZE

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame_filepath = out_path + data_source + '_' + str(UPSCALE_FACTOR) + '_test_results.csv'
data_frame.to_csv(data_frame_filepath, index_label='DataSet', mode='a', header=not os.path.exists(data_frame_filepath))
