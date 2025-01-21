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
from data_utils import CustomDataset, TestTensorDataset, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_230.pth', type=str, help='generator model epoch name') # 100: non-aug, 35: aug, 93: wgan-gp
opt = parser.parse_args()

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
    c, patch_hr_h, patch_hr_w = data[0][0][0].shape # need to include channel size allocation
    c, patch_lr_h, patch_lr_w = torch.unsqueeze(data[0][1][0],0).shape

    image_HR = torch.empty(data_size, c, patch_hr_h, patch_hr_w); image_LR = torch.empty(data_size, c, patch_lr_h, patch_lr_w)
    latlon_HR = np.empty((data_size, 4, 2)); latlon_LR = np.empty((data_size, 4, 2))
    time_HR = torch.empty(data_size); time_LR = torch.empty(data_size)
    for index, value in enumerate(data):
        image_HR[index,:,:,:] = value[0][0]; image_LR[index,:,:,:] = torch.unsqueeze(value[1][0],0)
        latlon_HR[index,:,:] = value[0][1]; latlon_LR[index,:,:] = value[1][1]
        time_HR[index] = value[0][2]; time_LR[index] = value[1][2]

    return image_HR, image_LR, latlon_HR, latlon_LR, time_HR, time_LR


# Load model and test dataset
data_filename = 'train_1y_Australia2_test_data.pkl' # 'sc_256_2y_5_test_data.pkl' # 
data_name, extension = os.path.splitext(data_filename)
results = {data_name: {'psnr': [], 'ssim': [], 'dist_hr_restore': [], 'dist_hr': []}} #,'Set5': {'psnr': [], 'ssim': []}

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

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
test_HR, test_LR, latlon_HR, latlon_LR, time_HR, time_LR = load_data(test_data)
test_HR_max = test_HR.max(); test_HR_min = test_HR.min(); test_HR_avg = test_HR.mean()
test_LR_max = test_LR.max(); test_LR_min = test_LR.min(); test_LR_avg = test_LR.mean()

# test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
test_set = TestTensorDataset(test_HR, test_LR, upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
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

# Load 'sea' truth
csv_dir = 'iQuam_20210101-20211231_combined-enti-Australia_new.csv' #'iQuam_20200101-20221231_surface_IndiOcean-right_new.csv'
df = pd.read_csv(f'{data_dir}{csv_dir}')

# Compare reconstructed image with original hr ground truth
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

    mse_bicubic = ((hr_image.data.cpu() - hr_restore_img) ** 2).data.mean()
    psnr_bicubic = 10 * log10(1 / mse_bicubic)
    ssim_bicubic = pytorch_ssim.ssim(hr_restore_img, hr_image.data.cpu()).item()

    # save psnr/ssim
    results[data_name]['psnr'].append(psnr)
    results[data_name]['ssim'].append(ssim)
    # results[image_name.split('_')[2]]['psnr'].append(psnr)
    # results[image_name.split('_')[2]]['ssim'].append(ssim)

    # compare with 'sea' truth
    _, _, img_w, img_h = hr_restore_img.shape
    norm_time = time_HR[index] # how to restore normalisation?

    closest_match = find_closest_match(df, np.mean(latlon_HR[index],0))
    sst = closest_match['sst'] # iQuam sea truth data
    # print(f"ground_truth: {closest_match['lat']}, {closest_match['lon']}; test_image: {np.mean(latlon_HR[index],0)}")
    
    hr_restore_img_sst = hr_restore_img[0][0][img_w//2][img_h//2].numpy() * test_HR_avg.numpy()
    hr_image_sst = hr_image.data.cpu()[0][0][img_w//2][img_h//2].numpy() * test_HR_avg.numpy()
    dist_hr_restore_sst = abs(hr_restore_img_sst - sst)
    dist_hr_sst = abs(hr_image_sst - sst)
    results[data_name]['dist_hr_restore'].append(dist_hr_restore_sst)
    results[data_name]['dist_hr'].append(dist_hr_sst)

    # save test images
    test_images = torch.stack(
        [display_transform()(hr_restore_img.squeeze(0)), 
         display_transform()(hr_image.data.cpu().squeeze(0)),
         display_transform()(sr_image.data.cpu().squeeze(0))]).squeeze()
    image = test_images # utils.make_grid(test_images, nrow=3, padding=5)
    image_name = 'test' + str(index)
    fig, axes = plt.subplots(1, 3, figsize=(20, 20))
    for i, ax in enumerate(axes):
        im = ax.imshow(image[i].numpy() * test_HR_avg.numpy(), cmap='viridis') # test_HR_max
        ax.set_axis_off()
        cax = fig.add_axes([ax.get_position().x1 + 0.005, ax.get_position().y0, 0.02, ax.get_position().height])
        fig.colorbar(im, cax=cax)
    fig.savefig(out_path + image_name + '_psnr_%.4f_ssim_%.4f_psnrBC_%.4f_ssimBC_%.4f_dist_hrR_%.2f_dist_hr_%2f.png' 
                % (psnr, ssim, psnr_bicubic, ssim_bicubic, dist_hr_restore_sst, dist_hr_sst), 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')
    # utils.save_image(image, out_path + image_name + '_psnr_%.4f_ssim_%.4f_psnrBC_%.4f_ssimBC_%.4f.png' % (psnr, ssim, psnr_bicubic, ssim_bicubic), padding=5)
    # utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) + image_name.split('.')[-1], padding=5) # TODO: change to colour plot
    
    index += 1

out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': [], 'dist_hr_restore': [], 'dist_hr': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    dist_hr_restore_sst = np.array(item['dist_hr_restore'])
    dist_hr_sst = np.array(item['dist_hr'])
    if (len(psnr) == 0) or (len(ssim) == 0) or (len(dist_hr_restore_sst) == 0) or (len(dist_hr_sst) == 0):
        psnr = 'No data'
        ssim = 'No data'
        dist_hr_restore_sst = 'No data'
        dist_hr_sst = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
        dist_hr_restore_sst = dist_hr_restore_sst.mean()
        dist_hr_sst = dist_hr_sst.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)
    saved_results['dist_hr_restore'].append(dist_hr_restore_sst)
    saved_results['dist_hr'].append(dist_hr_sst)
saved_results['model'] = MODEL_NAME

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame_filepath = out_path + data_source + '_' + str(UPSCALE_FACTOR) + '_test_results.csv'
data_frame.to_csv(data_frame_filepath, index_label='DataSet', mode='a', header=not os.path.exists(data_frame_filepath))
