import argparse
import os
import zipfile

import numpy as np
import pickle
import gc
from netCDF4 import Dataset
import xarray as xr

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
data_filename = 'train_1y_Australia2.pkl' # 'sc_256_2y_5.pkl' # load original datasets #
data_name, extension = os.path.splitext(data_filename)
results = {'dist_sr_hr': [], 'dist_to_coast': []}

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
test_HR = []; test_LR = []; test_HR_interp = []; pos = []; norm_time = []; dist_to_coast = []
for i, data in enumerate(data_HR):
    test_HR.append(data[0][0])
    test_LR.append(data_LR[i][0])
    test_HR_interp.append(data_HR_interp[i][0].squeeze(0))
    pos.append(data[1])
    norm_time.append(data[2])
    dist_to_coast.append(data[3])

# Load ancilliary data i.e. dist-to-coast (for AUS)
data_filename2 = 'dist_to_GSHHG_v2.3.7_1m.nc' # 'dist_to_GSHHG_v2.3.7_1m.zip' # global netCDF grid
data_name2, extension2 = os.path.splitext(data_filename2)
lat_filter = [np.amin(np.amin(pos, axis=1)), np.amax(np.amin(pos, axis=2))] # -90 to 90 --> -45.1, -15.0
lon_filter = [np.amin(np.amax(pos, axis=2)), np.amax(np.amax(pos, axis=2))] # 0 to 360 --> 90.0, 125.1
if extension2 == '.zip': # os.path.exists(f'{data_dir}{data_filename2}'):
    with zipfile.ZipFile(f'{data_dir}{data_filename2}', 'r') as f:
        f.extractall()
        os.remove(f'{data_dir}{data_filename2}')
        print(f'{data_dir}{data_filename2} extracted and ZIP deleted')
elif extension2 == '.nc':
    # nc_file = Dataset(f'{data_dir}{data_filename2}', 'r')
    # print(nc_file.variables.keys()) # lon, lat, dist
    global_nc = xr.open_dataset(f'{data_dir}{data_filename2}')
    if not np.any(dist_to_coast): # e.g. aus dataset
        aus_nc = global_nc.sel(lat = slice(lat_filter[0], lat_filter[1]),
                                  lon = slice(lon_filter[0], lon_filter[1]))
        ocean_mask = aus_nc["dist"] < 0 # keep only ocean points (i.e. negative distances)
        dist_to_coast = aus_nc.where(ocean_mask, drop=True)
        # lat = nc_file.variables['lat'][:].data
        # lat = lat[(lat > lat_filter[0]) & (lat < lat_filter[1])]
        # lon = nc_file.variables['lon'][:].data
        # lon = lon[(lon > lon_filter[0]) & (lon < lon_filter[1])]
        # dist = nc_file.variables['dist'][:].data
        # dist_to_coast = xr.DataArray(
        #     data=dist,
        #     dims=["y", "x"],
        #     coords={
        #         "lat": ("y", lat),  # Assign 1D lat array
        #         "lon": ("x", lon),  # Assign 1D lon array
        #     }
        # )
else:
    print(f'{data_dir}{data_filename2} does not exist')

num_sample = np.shape(test_HR)[0] # 200
test_HR = test_HR[:num_sample]
test_LR = test_LR[:num_sample]

test_set = TestTensorDataset(test_HR, test_LR, test_HR_interp, upscale_factor=UPSCALE_FACTOR, crop_size=CROP_SIZE)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/di-lab_%s/' % str(UPSCALE_FACTOR)
clear_directory(out_path)

index = 0
for lr_image, hr_bicubic_image, hr_image in test_bar:
    with torch.no_grad():
        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
        hr_image = hr_image.cuda()

    sr_image = model(lr_image)
    sr_image_denorm = denormalize(sr_image, test_HR[index].min(), test_HR[index].max())
    hr_image_denorm = denormalize(hr_image, test_HR[index].min(), test_HR[index].max())

    # investigate the relation between prediction accuracy and distance to the coast (in AUS, if enough time also SCS)
    sr_image_sst = sr_image_denorm.cpu().squeeze(0).squeeze(0)[CROP_SIZE//2][CROP_SIZE//2]
    hr_image_sst = hr_image_denorm.cpu().squeeze(0).squeeze(0)[CROP_SIZE//2][CROP_SIZE//2]
    dist_sr_hr = abs(sr_image_sst.detach() - hr_image_sst)
    results['dist_sr_hr'].append(dist_sr_hr)#.item())

    index += 1

times = np.array(norm_time).squeeze(-1)[:num_sample]
unique_time = np.unique(times)
pos_latlon = np.array(pos)[:num_sample]
for i in range(num_sample):
    if isinstance(dist_to_coast,xr.Dataset):
        pos_center = np.mean(pos[i], axis=0)
        pos_closest = dist_to_coast.sel(lat=pos_center[0], lon=pos_center[1], method='nearest') # lat, lon is a general grid
        results['dist_to_coast'].append(pos_closest.dist.item())
    else:
        l, w = dist_to_coast[i].shape
        results['dist_to_coast'].append(dist_to_coast[i][l//2][w//2]) # [309, 306] for scs

# plotting trend
out_path = 'statistics/'
fig, ax = plt.subplots()
ax.scatter(np.array(results['dist_to_coast']), np.array(results['dist_sr_hr']), s=20)
ax.set_ylabel('accuracy (deg)')
ax.set_xlabel('dist to coast (km)')
ax.grid(True)
fig.savefig(f'{out_path}{data_name}_distToCoast.png', dpi=300, bbox_inches='tight') 

plt.close('all')