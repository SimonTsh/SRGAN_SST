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
import matplotlib.cm as cm
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pytorch_ssim
from data_utils import TestTensorDataset, CustomDataset, denormalize
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_101.pth', type=str, help='generator model epoch name')
parser.add_argument('--crop_size', default=64, type=int, help='testing images crop size') # 64, 256
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
    c, patch_lr_h, patch_lr_w = data[0][7].shape
    image_HR_interp = torch.empty(data_size, c, patch_hr_h, patch_hr_w)
    image_HR        = torch.empty(data_size, c, patch_hr_h, patch_hr_w)
    image_LR        = torch.empty(data_size, c, patch_lr_h, patch_lr_w)

    for index, value in enumerate(data):
        if value:
            image_HR_interp[index,:,:,:], image_HR[index,:,:,:], _, _, _, _, _, image_LR[index,:,:,:], _ = value
        else:
            continue

    return image_HR_interp.squeeze(1), image_HR.squeeze(1), image_LR.squeeze(1)

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
results = {data_name: {'psnr': [], 'ssim': [], 'dist_sr': [], 'dist_hr': []}} #,'Set5': {'psnr': [], 'ssim': []}

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
test_HR_interp, test_HR, test_LR = load_data(test_data)

# test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
# test_HR = test_HR[:3]; test_LR = test_HR[:3]
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
index = 0; sr_images = []
for lr_image, hr_restore_img, hr_image in test_bar:
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
    psnr = 10 * log10(1 / mse) # norm
    ssim = pytorch_ssim.ssim(sr_image, hr_image).item() #.data[0]; norm

    mse_bicubic = ((hr_image.data.cpu() - hr_restore_img) ** 2).data.mean()
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
    dist_sr_sst = abs(sr_image_sst - sst)
    dist_hr_sst = abs(hr_image_sst - sst)
    results[data_name]['dist_sr'].append(dist_sr_sst)
    results[data_name]['dist_hr'].append(dist_hr_sst)

    # Save test images
    test_images = torch.stack(
        [(hr_restore_img.squeeze(0)), 
         (hr_image.data.cpu().squeeze(0)),
         (sr_image.data.cpu().squeeze(0))])
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

# Create a figure with a map projection
patch_lats = np.array(df['lat'])#[:3]
patch_lons = np.array(df['lon'])#[:3]
temperatures = np.array(sr_images)

# Define the size of the patch in degrees (example: 0.5 degrees x 0.5 degrees)
patch_size_degrees = 0.1 # 0.05
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Add map features
ax.coastlines()
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='--')
ax.set_extent([patch_lons.min(), patch_lons.max(), patch_lats.min(), patch_lats.max()]) # Set extent to focus on the South China Sea

for lat_center, lon_center, temp in zip(patch_lats, patch_lons, temperatures):
    # Set extent to focus on the area around the patch
    # ax.set_extent([lon_center - patch_size_degrees/2, lon_center + patch_size_degrees/2, 
    #                lat_center - patch_size_degrees/2, lat_center + patch_size_degrees/2])
    ax.imshow(temp, extent=[lon_center - patch_size_degrees/2, lon_center + patch_size_degrees/2, 
                            lat_center - patch_size_degrees/2, lat_center + patch_size_degrees/2], 
                            origin='lower', cmap='viridis')
plt.show()
fig.savefig(out_path + data_name + '_map.png', dpi=300)

# # Create a color map
# cmap = cm.get_cmap('jet')
# norm = colors.Normalize(vmin=np.min(temperatures), vmax=np.max(temperatures))
# # Plot temperature values at patch locations
# for lat, lon, temp in zip(patch_lats, patch_lons, temperatures):
#     # ax.plot(lon, lat, marker='o', markersize=5, color='red')
#     ax.scatter(lon, lat, s=50, c=[cmap(norm(temp.mean()))], transform=ccrs.PlateCarree())
#     ax.annotate(f"{temp.mean():.1f}°C", (lon, lat), textcoords="offset points", xytext=(0,10), ha='center')


# Save quantitative results
saved_results = {'psnr': [], 'ssim': [], 'dist_sr': [], 'dist_hr': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    dist_sr_sst = np.array([i.detach().numpy() for i in item['dist_sr']]) # np.array(item['dist_sr'])
    dist_hr_sst = np.array([i.detach().numpy() for i in item['dist_hr']]) # np.array(item['dist_hr'])
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
    saved_results['dist_sr'].append(dist_sr_sst)
    saved_results['dist_hr'].append(dist_hr_sst)
saved_results['model'] = MODEL_NAME

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame_filepath = out_path + data_source + '_' + str(UPSCALE_FACTOR) + '_test_results.csv'
data_frame.to_csv(data_frame_filepath, index_label='DataSet', mode='a', header=not os.path.exists(data_frame_filepath))
