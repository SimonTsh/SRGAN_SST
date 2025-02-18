from os import listdir
from os.path import join

from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, RandomRotation, RandomHorizontalFlip
import torch.nn.functional as F


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def train_hr_transformTensor():
    return Compose([
        ToPILImage(),
        # RandomCrop(crop_size),
        ToTensor()
    ])

def train_lr_transformTensor(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

def augment_transform():
    return Compose([
        ToPILImage(),
        RandomRotation((-180, 180)),  # Random rotation of full possible 360deg
        RandomHorizontalFlip(),  # Random horizontal flip
        ToTensor()
        # transforms.FiveCrop(224),  # Five crop
])

def augment_tensor(tensor):
    # Ensure input is a 4D tensor [B, C, H, W]
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Random rotation
    angle = torch.rand(1) * 360 - 180  # Random angle between -180 and 180
    tensor = rotate_tensor(tensor, angle)
    
    # Random horizontal flip
    if torch.rand(1) > 0.5:
        tensor = torch.flip(tensor, [3])  # Flip along the width dimension
    
    # Remove batch dimension if it was added
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0).squeeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    
    return tensor

def rotate_tensor(tensor, angle):
    # Convert angle to radians
    angle = angle * torch.pi / 180
    
    # Create affine grid for rotation
    theta = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0]
    ], dtype=tensor.dtype).unsqueeze(0)
    
    grid = F.affine_grid(theta, tensor.size(), align_corners=False)
    
    # Apply rotation using grid_sample
    rotated = F.grid_sample(tensor, grid, align_corners=False)
    
    return rotated

def center_crop_tensor(tensor, crop_size):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    
    h, w = tensor.shape
    th, tw = crop_size
    
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    
    return tensor[i:i+th, j:j+tw]

def resize_tensor(tensor, scale_factor, mode='bicubic'):
    # Ensure input is a 4D tensor [B, C, H, W]
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Calculate new size
    _, _, h, w = tensor.shape
    new_h = int(h / scale_factor)
    new_w = int(w / scale_factor)
    
    # Perform interpolation
    resized = F.interpolate(tensor, size=(new_h, new_w), mode=mode, align_corners=False)
    
    # Remove batch dimension if it was added
    if resized.dim() == 4:
        resized = resized.squeeze(0).squeeze(0)
    elif resized.dim() == 3:
        resized = resized.squeeze(0)
    
    return resized

def normalize_to_01(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

def normalize_to_01_global(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)

def normalize_mean_std(tensor, mean, std):
    return (tensor - mean) / std

def denormalize(tensor, original_min, original_max):
    return tensor * (original_max - original_min) + original_min

def denormalize_mean_std(tensor, mean, std):
    return (tensor * std) + mean

def load_original_data(data):
    image_HR_interp = []
    image_HR        = []
    image_LR        = []

    for _, value in enumerate(data):
        image_HR_interp.append([value[0], value[2], value[3]]) # patch, coordinates, norm_time
        image_HR.append([value[1], value[2], value[3]])
        image_LR.append([value[7][0,:,:], value[2], value[3]]) # since LR has different image sizes

    return image_HR_interp, image_HR, image_LR


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class TrainTensorDataset(Dataset):
    def __init__(self, data_hr, data_lr, crop_size, upscale_factor):
        super(TrainTensorDataset, self).__init__()
        self.data_hr = data_hr
        self.data_lr = data_lr
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.data_hr_min = 279 # self.data_hr.min()
        self.data_hr_max = 306 # self.data_hr.max()
        self.data_hr_mean = 0.509
        self.data_hr_std = 0.194
        # crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        # self.hr_transform = train_hr_transformTensor()
        # self.lr_transform = train_lr_transformTensor(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = normalize_to_01(self.data_hr[index])
        # hr_image = normalize_to_01_global(self.data_hr[index], self.data_hr_min, self.data_hr_max)
        # hr_image = normalize_mean_std(hr_image, self.data_hr_mean, self.data_hr_std)
        lr_image = resize_tensor(hr_image, self.upscale_factor, mode='bicubic')
        # lr_image = F.interpolate(hr_image.unsqueeze(0).unsqueeze(0), size=(self.crop_size // self.upscale_factor, self.crop_size // self.upscale_factor), mode='bicubic', align_corners=False)
        return lr_image.unsqueeze(0), hr_image.unsqueeze(0)
    
    def __len__(self):
        return len(self.data_hr)
    

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

class ValTensorDataset(Dataset):
    def __init__(self, data_hr, data_lr, upscale_factor):
        super(ValTensorDataset, self).__init__()
        self.data_hr = data_hr
        self.data_lr = data_lr
        self.upscale_factor = upscale_factor
        self.data_hr_min = 279 # self.data_hr.min()
        self.data_hr_max = 306 # self.data_hr.max()
        self.data_hr_mean = 0.509
        self.data_hr_std = 0.194

    def __getitem__(self, index):
        hr_image = normalize_to_01(self.data_hr[index])
        # hr_image = normalize_to_01_global(self.data_hr[index], self.data_hr_min, self.data_hr_max)
        # hr_image = normalize_mean_std(hr_image, self.data_hr_mean, self.data_hr_std)
        
        w, h = hr_image.size()
        # hr_scale = Resize(w, interpolation=Image.BICUBIC)
        # lr_scale = Resize(w // self.upscale_factor, interpolation=Image.BICUBIC)
        # crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        
        # hr_image = CenterCrop(crop_size)(hr_image)
        # hr_image = ToPILImage()(hr_image)
        # lr_image = F.interpolate(hr_image.unsqueeze(0).unsqueeze(0), size=(w // self.upscale_factor, h // self.upscale_factor), mode='bicubic', align_corners=False)
        lr_image = resize_tensor(hr_image, self.upscale_factor, mode='bicubic')
        hr_restore_img = F.interpolate(lr_image.unsqueeze(0).unsqueeze(0), size=(w, h), mode='bicubic', align_corners=False)
        return lr_image.unsqueeze(0), hr_restore_img.squeeze(0), hr_image.unsqueeze(0)

    def __len__(self):
        return len(self.data_hr)
    
    
class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index): # apply on a batch when method is called e.g. in Dataloader
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(sorted(self.lr_filenames, key=lambda x: int(''.join(filter(str.isdigit, x))))[index])
        w, h = lr_image.size
        hr_image = Image.open(sorted(self.hr_filenames, key=lambda x: int(''.join(filter(str.isdigit, x))))[index])
        # hr_image = CenterCrop((self.upscale_factor * h, self.upscale_factor * w))(hr_image)
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC) # why is it inverted h, w
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
    
class TestTensorDataset(Dataset):
    def __init__(self, hr_data, lr_data, test_hr_bicubic, upscale_factor, crop_size):
        super(TestTensorDataset, self).__init__()
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.hr_bicubic = test_hr_bicubic
        self.upscale_factor = upscale_factor
        self.crop_size = crop_size
        # self.data_hr_min = 279 # self.data_hr.min()
        # self.data_hr_max = 306 # self.data_hr.max()
        # self.data_hr_mean = 0.509
        # self.data_hr_std = 0.194

    def __getitem__(self, index):
        lr_image = normalize_to_01(self.lr_data[index])
        hr_image = normalize_to_01(self.hr_data[index])
        hr_bicubic = normalize_to_01(self.hr_bicubic[index])
        # lr_image = normalize_to_01_global(self.lr_data[index], self.data_hr_min, self.data_hr_max)
        # hr_image = normalize_to_01_global(self.hr_data[index], self.data_hr_min, self.data_hr_max)
        # hr_image = normalize_mean_std(hr_image, self.data_hr_mean, self.data_hr_std)

        w_lr, h_lr = lr_image.size()
        w_hr, h_hr = hr_image.size()
        w_hr_bicubic, h_hr_bicubic = hr_bicubic.size()
        # lr_image = ToPILImage()(lr_image)
        # hr_image = ToPILImage()(hr_image)

        # check for image size consistency
        if w_hr != self.crop_size:
            # hr_image = CenterCrop(self.crop_size)(hr_image)
            hr_image = center_crop_tensor(hr_image, self.crop_size)
            w_hr, h_hr = hr_image.size()
        if w_hr_bicubic != self.crop_size:
            hr_bicubic = center_crop_tensor(hr_bicubic, self.crop_size)
            w_hr_bicubic, h_hr_bicubic = hr_bicubic.size()
        if w_lr != (self.crop_size // 4): # given 256 --> 64
            lr_image = center_crop_tensor(lr_image, self.crop_size // 4)
            w_lr, l_hr = lr_image.size()
        lr_image = resize_tensor(hr_image, self.upscale_factor, mode='bicubic')
        hr_restore_img = hr_bicubic.unsqueeze(0).unsqueeze(0) # F.interpolate(lr_image.unsqueeze(0).unsqueeze(0), size=(w_hr, h_hr), mode='bicubic', align_corners=False)
        # hr_scale = Resize((self.upscale_factor * w_lr, self.upscale_factor * h_lr), interpolation=Image.BICUBIC        
        # hr_scale = Resize((w_hr, h_hr), interpolation=Image.BICUBIC)
        # hr_restore_img = hr_scale(lr_image)

        return lr_image.unsqueeze(0), hr_restore_img.squeeze(0), hr_image.unsqueeze(0)

    def __len__(self):
        return len(self.hr_data)


class CustomDataset(Dataset):
    def __init__(self, hr_data, lr_data, hr_interp_data, downscale_factor):
        assert len(hr_data) == len(lr_data) == len(hr_interp_data), "All lists must have the same length"
        self.hr_data = hr_data
        self.lr_data = lr_data
        self.hr_interp_data = hr_interp_data
        self.downscale_factor = downscale_factor

    def __getitem__(self, index):
        lr_image = self.lr_data[index][0]
        hr_image = self.hr_data[index][0]
        hr_interp_image = self.hr_interp_data[index][0]

        _, w, h = hr_image.size()
        hr_crop_size = min(w, h) // self.downscale_factor
        w, h = lr_image.size()
        lr_crop_size = min(w, h) // self.downscale_factor
        
        hr_image = CenterCrop(hr_crop_size)(hr_image)
        lr_image = CenterCrop(lr_crop_size)(hr_image)
        hr_interp_image = CenterCrop(hr_crop_size)(hr_interp_image)

        return hr_image, lr_image, hr_interp_image
    
    def __len__(self):
        return len(self.hr_data)

class AugmentedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        # self.data_hr_min = 279 # self.data_hr.min() # does not work with global values
        # self.data_hr_max = 306 # self.data_hr.max()

    def __iter__(self):
        for data_HR, data_LR, data_HR_interp in self.dataset:
            data_aug_HRs = augment_tensor(torch.cat((normalize_to_01(data_HR), normalize_to_01(data_HR_interp))))
            data_aug_HR = data_aug_HRs[0] # augment_tensor(normalize_to_01(data_HR)) # augment_transform()(data_HR)
            data_aug_HR_interp = data_aug_HRs[1] # augment_tensor(normalize_to_01(data_HR_interp)) # augment_transform()(data_HR_interp)
            # data_aug_LR = augment_tensor(normalize_to_01(data_LR)) # augment_transform()(data_LR)

            data_combi_HR = torch.cat((data_HR, denormalize(data_aug_HR.unsqueeze(0), data_HR.min(), data_HR.max())), dim=0)
            data_combi_HR_interp = torch.cat((data_HR_interp, denormalize(data_aug_HR_interp.unsqueeze(0), data_HR.min(), data_HR.max())), dim=0)
            # data_combi_LR = torch.cat((data_LR, denormalize(data_aug_LR.unsqueeze(0), data_LR.min(), data_LR.max())), dim=0)
            
            yield data_combi_HR, data_LR, data_combi_HR_interp

    def __len__(self):
        return len(self.dataset)
