from os import listdir
from os.path import join

from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


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


def train_hr_transformTensor(crop_size):
    return Compose([
        ToPILImage(),
        RandomCrop(crop_size),
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
    def __init__(self, dataset, crop_size, upscale_factor):
        super(TrainTensorDataset, self).__init__()
        self.dataset = dataset
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transformTensor(crop_size)
        self.lr_transform = train_lr_transformTensor(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(self.dataset[index]) # already ToPILImage() inside transform function
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image
    
    def __len__(self):
        return len(self.dataset)
    

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
    def __init__(self, dataset, upscale_factor):
        super(ValTensorDataset, self).__init__()
        self.dataset = dataset
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        hr_image = self.dataset[index]
        w, h = hr_image.size()
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)

        hr_image = ToPILImage()(hr_image)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.dataset)
    
    
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
    def __init__(self, hr_data, lr_data, upscale_factor):
        super(TestTensorDataset, self).__init__()
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        lr_image = self.lr_data[index]
        hr_image = self.hr_data[index]
        _, w_lr, h_lr = lr_image.size()
        _, w_hr, h_hr = hr_image.size()

        lr_image = ToPILImage()(lr_image)
        hr_image = ToPILImage()(hr_image)
        lr_image = hr_image.resize((int(w_hr / self.upscale_factor), int(h_hr / self.upscale_factor)), Image.LANCZOS)
        
        # hr_scale = Resize((self.upscale_factor * w_lr, self.upscale_factor * h_lr), interpolation=Image.BICUBIC)
        hr_scale = Resize((w_hr, h_hr), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)

        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_data)
