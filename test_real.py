import argparse
import os
from math import log10, sqrt

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
from data_utils import clear_directory, denormalize, TestTensorDataset
from model import Generator

parser = argparse.ArgumentParser(description='Test Real Measurement Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_85.pth', type=str, help='generator model epoch name')
parser.add_argument('--crop_size', default=256, type=int, help='testing images crop size') # 64, 256
parser.add_argument('--loc_name', default='right', type=str, help='location of test data (aus, right)')

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def find_closest_match(df, mean_latlon):
    df['distance'] = df.apply(lambda row: haversine_distance(row['lat'], row['lon'], mean_latlon[0], mean_latlon[1]), axis=1)
    closest_match = df.loc[df['distance'].idxmin(), ['lat', 'lon', 'date', 'sst']]
    return closest_match

def load_data(data):
    data_size = len(data)
    c, patch_hr_h, patch_hr_w = data[0][0].shape # need to include channel size allocation
    c, patch_lr_h, patch_lr_w = data[0][7].shape # dist_to_coast = data[0][6] # [306, 306]
    image_HR_interp = torch.empty(data_size, c, patch_hr_h, patch_hr_w)
    image_HR        = torch.empty(data_size, c, patch_hr_h, patch_hr_w)
    image_LR        = torch.empty(data_size, c, patch_lr_h, patch_lr_w)
    pos = []; norm_time = []; dist_to_coast = []

    for index, value in enumerate(data):
        if value:
            image_HR_interp[index,:,:,:], image_HR[index,:,:,:], _, _, _, _, _, image_LR[index,:,:,:], _ = value
            pos.append(value[2])
            norm_time.append(value[3])
            dist_to_coast.append(value[6])
        else:
            pos.append(None)
            norm_time.append(None)
            dist_to_coast.append(None)
            continue

    return image_HR_interp.squeeze(1), image_HR.squeeze(1), image_LR.squeeze(1), pos, norm_time, dist_to_coast

def animate(i):
    masks = times == unique_time[i]
    masks_true = np.where(masks)[0]

    for j, mask in enumerate(masks_true): # update images in-place
        if j < len(images):
            extents = [pos_latlon[mask,0,1], pos_latlon[mask,1,1],
                       pos_latlon[mask,0,0], pos_latlon[mask,2,0]]
            images[j].set_extent(extents)
            images[j].set_data(temperatures[mask])
            images[j].set_visible(True)
            images[j].set_clim(vmin=np.nanmin(temperatures),
                               vmax=np.nanmax(temperatures))
        else:  # handle case with more patches than pre-allocated images
            img = ax.imshow(temperatures[mask],
                            extent=extents,
                            origin='lower',
                            cmap='viridis',
                            vmin=np.nanmin(temperatures),
                            vmax=np.nanmax(temperatures))
            images.append(img)
    
    for j in range(len(masks_true), len(images)): # hide unused images from previous frames
        images[j].set_visible(False)
    
    ax.set_title(f'Time: {dates[i]}')
    return images + [ax.title]


opt = parser.parse_args()
UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name
CROP_SIZE = opt.crop_size
LOC_NAME = opt.loc_name

# Load measurement test dataset (_real) & 'sea' truth measurement point
# loc = 'aus' # 'aus' # 'right # (Australia, India)
if LOC_NAME == 'aus':
    csv_dir = 'iQuam_20210101-20211231_combined-enti-Australia_new.csv'
    image_size = '' # load 256x256
elif LOC_NAME == 'right':
    csv_dir = 'iQuam_20200101-20221231_surface_IndiOcean-right_new.csv'
    image_size = '_256-Ind' # load 256x256
else:
    KeyError('Not a defined location')

data_filename = f'{LOC_NAME}_real{image_size}.pkl'
data_name, extension = os.path.splitext(data_filename)
results = {data_name: {'psnr': [], 'ssim': [], 'diff_sr': [], 'diff_hr': [], 'dist_to_coast': []}}

# Load pre-trained model of choice
model = Generator(in_channels=1, out_channels=1, scale_factor=UPSCALE_FACTOR).eval() #Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    model = model.cuda()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

data_source = 'di-lab'
data_dir = 'data/%s/' % (data_source)
df = pd.read_csv(f'{data_dir}{csv_dir}') # load sea truth data
with open(f'{data_dir}{data_filename}','rb') as f:
    test_data = pickle.load(f) # load image data
gc.enable()
test_HR_interp, test_HR, test_LR, pos, norm_time, dist_to_coast = load_data(test_data)
if LOC_NAME == 'right':
    test_HR_interp = torch.flip(test_HR_interp, dims=(1,))
num_sample = np.shape(test_HR)[0] # 200
test_HR_interp = test_HR_interp[:num_sample]
test_HR = test_HR[:num_sample]
test_LR = test_LR[:num_sample]

# test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
# test_HR = test_HR[:3]; test_LR = test_HR[:3]
test_set = TestTensorDataset(test_HR, test_LR, test_HR_interp, upscale_factor=UPSCALE_FACTOR, crop_size=CROP_SIZE)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/di-lab_%s/' % str(UPSCALE_FACTOR)
clear_directory(out_path)

# Compare reconstructed image with original hr ground truth
index = 0; sr_images = []
for lr_image, hr_restore_img, hr_image in test_bar:
    if torch.isnan(hr_image).all():
        print(f'Skipping index {index}...')
        index += 1
        continue

    with torch.no_grad():
        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    sr_image = model(lr_image)

    # denormalization with max pixel
    hr_image_denorm = denormalize(hr_image, test_HR[index].min(), test_HR[index].max())
    sr_image_denorm = denormalize(sr_image, test_HR[index].min(), test_HR[index].max())
    hr_restore_img_denorm = denormalize(hr_restore_img, test_HR[index].min(), test_HR[index].max())
    lr_image_denorm = denormalize(hr_image, test_LR[index].min(), test_LR[index].max())

    # calculate test parameters
    mse = ((hr_image - sr_image) ** 2).data.mean() # norm
    rmse = sqrt(mse)
    psnr = 10 * log10(1 / mse) # norm
    ssim = pytorch_ssim.ssim(sr_image, hr_image).item() #.data[0]; norm

    mse_bicubic = ((hr_image.data.cpu() - hr_restore_img) ** 2).data.mean()
    rmse_bicubic = sqrt(mse_bicubic)
    psnr_bicubic = 10 * log10(1 / mse_bicubic)
    ssim_bicubic = pytorch_ssim.ssim(hr_restore_img, hr_image.data.cpu()).item()

    # save psnr/ssim
    results[data_name]['psnr'].append(psnr)
    results[data_name]['ssim'].append(ssim)
    # results[image_name.split('_')[2]]['psnr'].append(psnr)
    # results[image_name.split('_')[2]]['ssim'].append(ssim)

    # Compare with 'sea' truth
    _, _, img_w, img_h = hr_restore_img.shape
    sst = df['sst'][index] # closest_match['sst'] # iQuam sea truth data
    # closest_match = find_closest_match(df, np.mean(latlon_HR[index],0))
    # print(f"test{index}: ground_truth: {closest_match['lat']}, {closest_match['lon']}; test_image: {np.mean(latlon_HR[index],0)}")
    
    # Compare point measurements
    sr_image_sst = sr_image_denorm.cpu().squeeze(0).squeeze(0)[img_w//2][img_h//2]
    hr_image_sst = hr_image_denorm.cpu().squeeze(0).squeeze(0)[img_w//2][img_h//2]
    dist_sr_sst = abs(sr_image_sst.detach() - sst)
    dist_hr_sst = abs(hr_image_sst - sst)
    results[data_name]['diff_sr'].append(dist_sr_sst)
    results[data_name]['diff_hr'].append(dist_hr_sst)

    # Save dist_to_coast
    l, w = dist_to_coast[index].shape
    results[data_name]['dist_to_coast'].append(dist_to_coast[index][l//2][w//2]) # ~ [309, 309]

    # Save test images
    test_images = torch.stack(
        [(hr_restore_img.squeeze(0)),
         (sr_image.data.cpu().squeeze(0)),
         (hr_image.data.cpu().squeeze(0))])
    image = test_images # utils.make_grid(test_images, nrow=3, padding=5)
    image_name = f'test{index}'
    fig, axes = plt.subplots(1, 3, figsize=(20, 20))
    for i, ax in enumerate(axes):
        im = ax.imshow(image[i].squeeze(0), cmap='viridis') # test_HR_max
        ax.set_axis_off()
        cax = fig.add_axes([ax.get_position().x1 + 0.005, ax.get_position().y0, 0.02, ax.get_position().height])
        fig.colorbar(im, cax=cax)
    fig.savefig(out_path + image_name + '_psnr_%.4f_ssim_%.4f_psnrBC_%.4f_ssimBC_%.4f_dist_sr_%.2f_dist_hr_%2f.png' 
                % (psnr, ssim, psnr_bicubic, ssim_bicubic, dist_sr_sst, dist_hr_sst), 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')
    # utils.save_image(image, out_path + image_name + '_psnr_%.4f_ssim_%.4f_psnrBC_%.4f_ssimBC_%.4f.png' % (psnr, ssim, psnr_bicubic, ssim_bicubic), padding=5)
    # utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) + image_name.split('.')[-1], padding=5) # TODO: change to colour plot
    
    sr_images.append(sr_image_denorm.cpu().detach().numpy().squeeze(0).squeeze(0))
    index += 1

out_path = 'statistics/'

# plotting dist-to-coast trend
fig, ax1 = plt.subplots()
ax1.set_xlabel('dist to coast (km)')
ax1.scatter(np.array(results[data_name]['dist_to_coast']), 
            np.array(results[data_name]['diff_sr']), 
            color='tab:red', label='SR', alpha=0.7)
ax1.set_ylabel('SR-InSitu MAE (deg)', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.scatter(np.array(results[data_name]['dist_to_coast']), 
            np.array(results[data_name]['diff_hr']), 
            color='tab:blue', label='HR', alpha=0.7)
ax2.set_ylabel('HR-InSitu MAE (deg)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

ax1.grid(True)
fig.savefig(f'{out_path}{data_name}_distToCoast.png', dpi=300, bbox_inches='tight')
plt.close('all')

# plotting temporal trend
times = np.array([x for x in norm_time if x is not None]).squeeze(-1)[:num_sample]
unique_time = np.unique(times)
pos_latlon = np.array([x for x in pos if x is not None])[:num_sample]
min_lat = pos_latlon[:,:,0].min()
max_lat = pos_latlon[:,:,0].max()
min_lon = pos_latlon[:,:,1].min()
max_lon = pos_latlon[:,:,1].max()

# Create a figure with a map projection
patch_lats = np.array(df['lat'])#[:3]
patch_lons = np.array(df['lon'])#[:3]
temperatures = np.array(sr_images)[:num_sample]

# Define the size of the patch in degrees
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Add map features
ax.coastlines()
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='--')
ax.set_extent([min_lon-4, max_lon+4, min_lat-2, max_lat+2])

cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height] # create colorbar once
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=np.nanmin(temperatures), vmax=np.nanmax(temperatures)))
fig.colorbar(sm, cax=cax)

dates = [(datetime(2023, 1, 1) + timedelta(days=int(t*360))).date().isoformat() for t in unique_time]

images = [] # Create initial empty image collection
for _ in range(len(unique_time)):
    images.append(ax.imshow(np.empty((0,0)), visible=False, origin='lower', cmap='viridis'))
    
ani = animation.FuncAnimation(fig, animate, frames=len(unique_time), interval=500, blit=True, repeat=False)

ani.save(f'{out_path}{data_name}_map.gif', writer='pillow', fps=5, savefig_kwargs={'facecolor': fig.get_facecolor()})
# fig.savefig(out_path + data_name + '_map.png', dpi=300)
plt.close('all')


# Save quantitative results
saved_results = {'psnr': [], 'ssim': [], 'diff_sr': [], 'diff_hr': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    dist_sr_sst = np.array([i.detach().numpy() for i in item['diff_sr']]) # np.array(item['dist_sr'])
    dist_hr_sst = np.array([i.detach().numpy() for i in item['diff_hr']]) # np.array(item['dist_hr'])
    if (len(psnr) == 0) or (len(ssim) == 0) or (len(dist_sr_sst) == 0) or (len(dist_hr_sst) == 0):
        psnr = 'No data'
        ssim = 'No data'
        dist_sr_sst = 'No data'
        dist_hr_sst = 'No data'
    else:
        psnr = np.nanmean(psnr)
        ssim = np.nanmean(ssim)
        dist_sr_sst = np.nanmean(dist_sr_sst)
        dist_hr_sst = np.nanmean(dist_hr_sst)
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)
    saved_results['diff_sr'].append(dist_sr_sst)
    saved_results['diff_hr'].append(dist_hr_sst)
saved_results['model'] = MODEL_NAME

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame_filepath = out_path + data_source + '_' + str(UPSCALE_FACTOR) + '_test_results.csv'
data_frame.to_csv(data_frame_filepath, index_label='DataSet', mode='a', header=not os.path.exists(data_frame_filepath))
