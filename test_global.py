import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import pickle
import gc

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta

import pytorch_ssim
from data_utils import clear_directory, denormalize, load_original_data, TestTensorDataset
from model import Generator

parser = argparse.ArgumentParser(description='Test Global Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_89.pth', type=str, help='generator model epoch name') # SRGAN: 101; WGAN: 198
parser.add_argument('--crop_size', default=256, type=int, help='testing images crop size') # 64, 256
opt = parser.parse_args()


def animate(i):
    masks = times == unique_time[i]
    masks_true = np.where(masks)[0]

    for j, mask in enumerate(masks_true): # update images in-place
        if j < len(images):
            extents = [pos_latlon[mask,0,1], pos_latlon[mask,1,1],
                       pos_latlon[mask,0,0], pos_latlon[mask,2,0]]
            images[j].set_extent(extents)
            images[j].set_data(sst_images[mask])
            images[j].set_visible(True)
            images[j].set_clim(vmin=sst_images.min(), 
                               vmax=sst_images.max())
        else:  # handle case with more patches than pre-allocated images
            img = ax.imshow(sst_images[mask],
                            extent=extents,
                            origin='lower',
                            cmap='viridis',
                            vmin=sst_images.min(), 
                            vmax=sst_images.max())
            images.append(img)
    
    for j in range(len(masks_true), len(images)): # hide unused images from previous frames
        images[j].set_visible(False)
    
    ax.set_title(f'Time: {dates[i]}')
    return images + [ax.title]


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
unique_time = np.unique(times)
pos_latlon = np.array(pos)[:num_sample]
min_lat = pos_latlon[:,:,0].min()
max_lat = pos_latlon[:,:,0].max()
min_lon = pos_latlon[:,:,1].min()
max_lon = pos_latlon[:,:,1].max()

# Add static map features once
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='--')
ax.set_extent([min_lon-4, max_lon+4, min_lat-2, max_lat+2])

cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height] # create colorbar once
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=sst_images.min(), vmax=sst_images.max()))
fig.colorbar(sm, cax=cax)

dates = [(datetime(2023, 1, 1) + timedelta(days=int(t*360))).date().isoformat() for t in unique_time] # Precompute all date strings

images = [] # Create initial empty image collection
for _ in range(len(unique_time)):
    images.append(ax.imshow(np.empty((0,0)), visible=False, origin='lower', cmap='viridis'))

ani = animation.FuncAnimation(fig, animate, frames=len(unique_time), interval=500, blit=True, repeat=False)

# writer = PillowWriter(fps=5)
ani.save(f'{out_path}{data_name}_map.gif', writer='pillow', fps=5, savefig_kwargs={'facecolor': fig.get_facecolor()})
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
