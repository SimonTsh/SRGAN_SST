import os
import shutil
import pickle
import zipfile

import torch
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, ToTensor, ToPILImage
from torch.utils.data import random_split, TensorDataset

import numpy as np
import matplotlib
matplotlib.use('Agg')  # or 'QtAgg' if you installed PyQt
import matplotlib.pyplot as plt


class AugmentedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for data_HR, data_LR in self.dataset:
            data_aug_HR = transform()(data_HR)
            data_aug_LR = transform()(data_LR)

            data_combi_HR = torch.cat((data_HR, data_aug_HR), dim=0)
            data_combi_LR = torch.cat((data_LR, data_aug_LR), dim=0)
            
            yield data_combi_HR, data_combi_LR

    def __len__(self):
        return len(self.dataset)

def transform():
    return Compose([
        ToPILImage(),
        RandomRotation((-180, 180)),  # Random rotation of full possible 360deg
        RandomHorizontalFlip(),  # Random horizontal flip
        ToTensor()
        # transforms.FiveCrop(224),  # Five crop
])

def move_hr_pngs(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file is a PNG and has 'HR' in its name
        if filename.lower().endswith('.png') and 'LR' in filename:
            # Construct full file paths
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            
            # Move the file
            shutil.move(source_file, destination_file)
            print(f"Moved: {filename}")

def load_data(data):
    data_size = len(data)
    c, patch_hr_h, patch_hr_w = data[0][0].shape # need to include channel size allocation
    # c, patch_lr_h, patch_lr_w = data[0][7].shape

    image_HR_interp = torch.empty(data_size, c, patch_hr_h, patch_hr_w)
    image_HR        = torch.empty(data_size, c, patch_hr_h, patch_hr_w)
    image_LR        = [] #torch.empty(data_size, c, patch_lr_h, patch_lr_w)
    for index, value in enumerate(data):
        image_HR_interp[index,:,:,:], image_HR[index,:,:,:], *arg = value
        image_LR.append(value[7][0,:,:]) # since LR has different image sizes

    return image_HR_interp, image_HR, image_LR

def extract_data(dataset):
    # Function to extract HR and LR data from a dataset
    hr_data = torch.stack([item[0] for item in dataset])
    lr_data = torch.stack([item[1] for item in dataset])
    return [hr_data, lr_data]

def rearrange_dict_arrays(original_dict, new_order):
    rearranged_dict = {}
    for key, array in original_dict.items():
        rearranged_dict[key] = torch.from_numpy(array[new_order])
    return rearranged_dict


## Define parameters
to_visualise = 1 # 0 or 1
source_folder = "data/di-lab/" # upsample factor: 2 or 4
# destination_folder = "data/test/data" # HR: target, LR: data
# move_hr_pngs(source_folder, destination_folder)

## To separate dataset into train, test, val
# Load the data from the .pkl file
data_filename = 'sc_256_2y_5.pkl' # 'SCS_data.zip' # 'train_1y_Australia2.pkl'
root, extension = os.path.splitext(data_filename)
data_dir = source_folder + data_filename

if extension == '.pkl':
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
elif extension == '.zip':
    with zipfile.ZipFile(data_dir, 'r') as f:
        print(f.namelist())
else:
    KeyError('Not a recognised file type')

data_HR_interp, data_HR, data_LR = load_data(data)
min_size = min({len(data_lr) for data_lr in data_LR})
data_LR_cut = torch.empty(data_HR.shape[0], data_HR.shape[1], min_size, min_size)
for i, data_lr in enumerate(data_LR):
    data_LR_cut[i,0,:,:] = data_lr[:min_size,:min_size]
print('data loaded successfully')
# Store in a dictionary or tuple for clarity
data = {
    'HR': data_HR,
    'LR': data_LR_cut
}

# Set a seed for reproducibility
torch.manual_seed(42)
# Define the sizes for train, validation, and test sets
dataset = TensorDataset(data['HR'], data['LR'])
total_size = len(dataset)
train_size = int(0.7 * total_size)  # 70% for training
val_size = int(0.15 * total_size)   # 15% for validation
test_size = total_size - train_size - val_size  # 15% for testing

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
print('data split successfully')

# Perform augmentation of dataset
train_augment = AugmentedDataset(train_dataset)
train_combi = {
    'HR': [],
    'LR': []
}
for batch_idx, (data_combi_HR, data_combi_LR) in enumerate(train_augment):
    train_combi['HR'].append(data_combi_HR)
    train_combi['LR'].append(data_combi_LR)
    # print(f"Batch {batch_idx}: Data HR shape: {data_HR.shape}, Data (LR) shape: {data_LR}")
print('train data augmented successfully')

val_augment = AugmentedDataset(val_dataset)
val_combi = {
    'HR': [],
    'LR': []
}
for batch_idx, (data_combi_HR, data_combi_LR) in enumerate(val_augment):
    val_combi['HR'].append(data_combi_HR)
    val_combi['LR'].append(data_combi_LR)
print('val data augmented successfully')
# train_combined = torch.cat((train_dataset, train_augment.dataset), dim=0) # return TypeError: expected Tensor as element 0 in argument 0, but got Subset
# val_combined = torch.cat((val_dataset, val_augment.dataset), dim=0)

# Perform shuffling using randomly generated indices
num_samples, multi_fac, w, h = np.shape(train_combi['HR'])
train_combi['HR'] = np.array(train_combi['HR']).reshape(num_samples*multi_fac, w, h)
num_samples, multi_fac, w, h = np.shape(train_combi['LR']) # w, h different
train_combi['LR'] = np.array(train_combi['LR']).reshape(num_samples*multi_fac, w, h)
indices = np.random.permutation(num_samples*multi_fac) # for numpy array
train_data = rearrange_dict_arrays(train_combi, indices)
print('train data shuffled successfully')

num_samples, multi_fac, w, h = np.shape(val_combi['HR'])
val_combi['HR'] = np.array(val_combi['HR']).reshape(num_samples*multi_fac, w, h)
num_samples, multi_fac, w, h = np.shape(val_combi['LR']) # w, h different
val_combi['LR'] = np.array(val_combi['LR']).reshape(num_samples*multi_fac, w, h)
indices = np.random.permutation(num_samples*multi_fac)
val_data = rearrange_dict_arrays(val_combi, indices)
print('val data shuffled successfully')

test_data = test_dataset

if to_visualise:
    # To visualise a subset of images (e.g., first 16)
    num_images = 16
    images = train_data['HR'][:num_images]

    # Create a grid of images
    rows, cols = round(np.sqrt(num_images)), round(np.sqrt(num_images))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].cpu().numpy(), cmap='viridis')
        ax.axis('off')
    # grid = vutils.make_grid(images, nrow=4, normalize=True, padding=2)
    # grid = grid.permute(1, 2, 0) # Convert to numpy and adjust dimensions

    # Display the grid
    plt.tight_layout()
    plt.savefig(source_folder+f'{root}_plot.png')

# Save each set as a separate .pkl file
with open(source_folder+f'{root}_train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open(source_folder+f'{root}_val_data.pkl', 'wb') as f:
    pickle.dump(val_data, f)

with open(source_folder+f'{root}_test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)
