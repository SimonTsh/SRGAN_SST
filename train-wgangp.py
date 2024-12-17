import argparse
import os
from math import log10

import pandas as pd
import pickle
import gc
import shutil

import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm
from tensorboard_logger import configure, log_value

import pytorch_ssim
from data_utils import TrainTensorDataset, ValTensorDataset, display_transform # ,TrainDatasetFromFolder, ValDatasetFromFolder
from loss import GeneratorLoss, compute_gradient_penalty
from model import Generator, Discriminator_WGAN

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--epoch_start', default=113, type=int, help='epoch to load from')
parser.add_argument('--crop_size', default=64, type=int, help='training images crop size') # 88 # 96 # 128
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=250, type=int, help='train epoch number') # 30,100,500
parser.add_argument('--g_learning_rate', default=1e-4, type=float, help='learning rate for generator') # 0.0001
parser.add_argument('--d_learning_rate', default=1e-5, type=float, help='learning rate for discriminator') # 0.0002
parser.add_argument('--b1', default=0.0, type=float, help='adam: decay of first order momentum of gradient') # 0.5
parser.add_argument('--b2', default=0.9, type=float, help='adam: decay of second order momentum of gradient') # 0.999
parser.add_argument('--decay_epoch', default=100, type=int, help='start lr decay every decay_epoch epochs') # 30,50,70,250
parser.add_argument('--gamma', default=0.1, type=float, help='multiplicative factor of learning rate decay') # 0.5

def load_data(data):
    data_size = len(data)
    c, patch_hr_h, patch_hr_w = data[0][0].shape # need to include channel size allocation
    c, patch_lr_h, patch_lr_w = data[0][1].shape

    image_HR = torch.empty(data_size, c, patch_hr_h, patch_hr_w)
    image_LR = torch.empty(data_size, c, patch_lr_h, patch_lr_w)
    for index, value in enumerate(data):
        image_HR[index,:,:,:], image_LR[index,:,:,:], _ = value

    return image_HR, image_LR

def open_pkl_file(data_dir, data_source, data_type):
    data_filepath = data_dir + f'{data_source}_{data_type}.pkl'
    with open(data_filepath,'rb') as f:
        data = pickle.load(f)
    
    return data


if __name__ == '__main__':
    opt = parser.parse_args()
    
    EPOCH_START = opt.epoch_start
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    G_LEARNING_RATE = opt.g_learning_rate
    D_LEARNING_RATE = opt.d_learning_rate
    B1 = opt.b1
    B2 = opt.b2
    DECAY_EPOCH = opt.decay_epoch
    GAMMA = opt.gamma

    use_tensorboard = True

    data_dir = 'data/di-lab/'
    data_source1 = 'train_1y_Australia2'
    train_data1 = open_pkl_file(data_dir, data_source1, 'train_data')
    val_data1 = open_pkl_file(data_dir, data_source1, 'val_data')

    data_source2 = 'sc_256_2y_5'
    train_data2 = open_pkl_file(data_dir, data_source2, 'train_data')
    val_data2 = open_pkl_file(data_dir, data_source2, 'val_data')

    gc.enable()

    train_HR, train_LR = ConcatDataset([train_data1['HR'], train_data2['HR']]), ConcatDataset([train_data1['LR'],train_data2['HR']]) #load_data(train_data)
    val_HR, val_LR = ConcatDataset([val_data1['HR'], val_data2['HR']]), ConcatDataset([val_data1['LR'], val_data2['LR']]) #load_data(val_data)
    
    train_set = TrainTensorDataset(train_HR, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValTensorDataset(val_HR, upscale_factor=UPSCALE_FACTOR)
    
    train_loader = DataLoader(train_set, num_workers=1, batch_size=16, shuffle=True) # batch_size=32, 64, 128 # num_workers=4 --> 16 @ epoch 112 onwards
    val_loader = DataLoader(val_set, num_workers=1, batch_size=1, shuffle=False) # num_workers=4
    
    netG = Generator(in_channels=1, out_channels=1, scale_factor=UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator_WGAN()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    if use_tensorboard:
        configure('log', flush_secs=5)
    
    if EPOCH_START != 0:
        print('Starting from epoch %d...' % (EPOCH_START + 1))
        netG.load_state_dict(torch.load('epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, EPOCH_START)))
        netD.load_state_dict(torch.load('epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, EPOCH_START)))

    optimizerG = optim.Adam(netG.parameters(), lr=G_LEARNING_RATE, betas=(B1, B2))
    optimizerD = optim.Adam(netD.parameters(), lr=D_LEARNING_RATE, betas=(B1, B2))

    schedulerG = MultiStepLR(optimizerG, milestones=[DECAY_EPOCH], gamma=GAMMA)
    schedulerD = MultiStepLR(optimizerD, milestones=[DECAY_EPOCH], gamma=GAMMA)
    
    numIterationsDPerG = 5 # number of discriminator iterations per generator iteration
    # best_psnr = 0 # to track best achievable psnr
    # patience = 10  # number of epochs to wait before stopping
    # patience_counter = 0 # to be refreshed

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    
    for epoch in range(EPOCH_START + 1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            optimizerG.zero_grad()
            
            real_img = target
            if torch.cuda.is_available():
                real_img = real_img.float().cuda()
            z = data
            if torch.cuda.is_available():
                z = z.float().cuda()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            optimizerG.step()
            schedulerG.step()

            ############################
            # (2) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            for _ in range(numIterationsDPerG):
                optimizerD.zero_grad()
                real_out = netD(real_img).mean()
                fake_out = netD(fake_img.detach()).mean()
                gradient_penalty = compute_gradient_penalty(netD, real_img, fake_img)
                lamda = 10  # penalty coefficient
                d_loss = fake_out - real_out + lamda * gradient_penalty

                # Backward pass for discriminator
                # d_loss.backward()

                params = [p for p in netD.parameters() if p.requires_grad]
                grads = torch.autograd.grad(d_loss, params, create_graph=True, retain_graph=True)

                for param, grad in zip(params, grads):
                    param.data = param.data.contiguous()
                    
                    param.grad = grad # manually set gradients for each parameter in order to retain graph
                    if param.grad is not None:
                        param.grad.data = param.grad.data.contiguous()
                
                optimizerD.step()
                schedulerD.step()
                
                fake_img = netG(z)
                fake_out = netD(fake_img).mean()

                # loss for current batch before optimization 
                running_results['g_loss'] += g_loss.item() * batch_size
                running_results['d_loss'] += d_loss.item() * batch_size

            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
    
        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
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
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            validating_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                validating_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.float().cuda()
                    hr = hr.float().cuda()
                sr = netG(lr)
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                validating_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                validating_results['ssims'] += batch_ssim * batch_size
                validating_results['psnr'] = 10 * log10((hr.max()**2) / (validating_results['mse'] / validating_results['batch_sizes']))
                validating_results['ssim'] = validating_results['ssims'] / validating_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        validating_results['psnr'], validating_results['ssim']))
        
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), 
                     display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
                
                # current_psnr = validating_results['psnr'] # check for increasing psnr
                # if current_psnr > best_psnr:
                #     best_psnr = current_psnr
                #     patience_counter = 0

                #     # Save the best model
                #     torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
                #     torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
                # else:
                #     patience_counter += 1
                #     if patience_counter >= patience:
                #         print(f"Early stopping at epoch {epoch} due to no improvement in PSNR for {patience} epochs")
                #         break

            val_images = torch.stack(val_images)
            # Ensure we have a multiple of 15 images
            num_images = 15
            if val_images.size(0) % num_images != 0:
                # Pad with blank images or duplicate the last image
                pad_size = num_images - (val_images.size(0) % num_images)
                val_images = torch.cat([val_images, val_images[-1].unsqueeze(0).repeat(pad_size, 1, 1, 1)])
            val_images = torch.chunk(val_images, val_images.size(0) // num_images)
            
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                nrow = 3
                image = utils.make_grid(image, nrow=nrow, padding=5, normalize=True)

                if index % 300 == 0: # %100
                    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1

        if use_tensorboard:
            log_value('d_loss', running_results['d_loss'] / running_results['batch_sizes'], epoch)
            log_value('g_loss', running_results['g_loss'] / running_results['batch_sizes'], epoch)
            log_value('d_score', running_results['d_score'] / running_results['batch_sizes'], epoch)
            log_value('g_score', running_results['g_score'] / running_results['batch_sizes'], epoch)
            log_value('ssim', validating_results['psnr'], epoch)
            log_value('psnr', validating_results['ssim'], epoch)

        # save model parameters
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(validating_results['psnr'])
        results['ssim'].append(validating_results['ssim'])
    
        if epoch != 0: # and epoch % 10 == 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(EPOCH_START + 1, epoch + 1))
            data_frame.to_csv(out_path + 'di-lab_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
