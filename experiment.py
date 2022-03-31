import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchgan.losses import LeastSquaresDiscriminatorLoss, LeastSquaresGeneratorLoss
from torch.nn.functional import interpolate

from model import *
from data import PatchSet, Mode
from utils import *

import shutil
from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd

from splicing import image_coor,cut_image,splicing_image
from math import exp
import random

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average,
                       full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models,
    # not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

class Experiment(object):
    def __init__(self, option):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = option.image_size

        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)
        self.best = self.train_dir / 'best.pth'
        self.last_g = self.train_dir / 'generator.pth'
        self.last_pd = self.train_dir / 'nlayerdiscriminator.pth'
        self.ifAdaIN = option.ifAdaIN
        self.ifAttention = option.ifAttention
        self.ifTwoInput = option.ifTwoInput
        self.ifAmplificat = option.ifAmplificat
        self.cut_num_h = option.cut_num[0]
        self.cut_num_w = option.cut_num[1]
        self.a = option.a
        self.b = option.b
        self.c = option.c
        self.d = option.d

        self.logger = get_logger()
        self.logger.info('Model initialization')

        self.generator = CombinFeatureGenerator(ifAdaIN=self.ifAdaIN, ifAttention=self.ifAttention,ifTwoInput = self.ifTwoInput).to(self.device)
        self.nlayerdiscriminator = NLayerDiscriminator(input_nc = 12, getIntermFeat = True).to(self.device)

        self.pd_loss = GANLoss().to(self.device)


        device_ids = [i for i in range(option.ngpu)]
        if option.cuda and option.ngpu > 1:
            self.generator = nn.DataParallel(self.generator, device_ids)
            self.nlayerdiscriminator = nn.DataParallel(self.nlayerdiscriminator, device_ids)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=option.lr)
        self.pd_optimizer = optim.Adam(self.nlayerdiscriminator.parameters(), lr=option.lr)

        def lambda_rule(epoch):
            lr_l = 1.0
            return lr_l

        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda_rule)
        self.pd_scheduler = torch.optim.lr_scheduler.LambdaLR(self.pd_optimizer, lr_lambda=lambda_rule)


        n_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for generator.')
        n_params = sum(p.numel() for p in self.nlayerdiscriminator.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for nlayerdiscriminator.')
        self.logger.info(str(self.generator))
        self.logger.info(str(self.nlayerdiscriminator))

    def train_on_epoch(self, n_epoch, data_loader):
        self.g_scheduler.step()
        self.pd_scheduler.step()
        self.generator.train()
        self.nlayerdiscriminator.train()
        epg_loss = AverageMeter()
        eppd_loss = AverageMeter()
        epg_error = AverageMeter()

        batches = len(data_loader)
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):
            t_start = timer()
            data = [im.to(self.device) for im in data]
            if self.ifAmplificat == True:
                if random.randint(0, 1) == 0:
                    pass
                else:
                    data = [data[2],data[3],data[0],data[1]]
            else:
                pass
            data[0] = interpolate(interpolate(data[0], scale_factor=1 / 16),scale_factor=16,mode='bicubic')
            data[2] = interpolate(interpolate(data[2], scale_factor=1 / 16), scale_factor=16, mode='bicubic')
            inputs, target = data[:-1], data[-1]
            ############################
            # (1) Update D network
            ###########################
            self.nlayerdiscriminator.zero_grad()
            self.generator.zero_grad()
            prediction = self.generator(inputs)

            pred_fake = self.nlayerdiscriminator(torch.cat((prediction.detach(), inputs[2]), dim=1))
            pred_real1 = self.nlayerdiscriminator(torch.cat((target, inputs[2]), dim=1))
            pd_loss = (self.pd_loss(pred_fake,False) +
                       self.pd_loss(pred_real1,True))  * 0.5

            pd_loss.backward()
            self.pd_optimizer.step()

            eppd_loss.update(pd_loss.item())

            ############################
            # (2) Update G network
            ###########################
            prediction = self.generator(inputs)
            pred_fake = self.nlayerdiscriminator(torch.cat((prediction, inputs[2]), dim=1))

            loss_G_GAN = self.pd_loss(pred_fake, True) * self.a

            loss_G_l1 = F.l1_loss(prediction, target) * self.b \
                        + (1.0 - msssim(prediction, target, normalize=True)) * self.d\
                        + (1.0 - torch.mean(F.cosine_similarity(prediction, target, 1))) * self. c

            g_loss = loss_G_l1 + loss_G_GAN

            g_loss.backward()
            self.g_optimizer.step()
            epg_loss.update(g_loss.item())

            mse = F.mse_loss(prediction.detach(), target).item()
            epg_error.update(mse)
            t_end = timer()
            self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                             f'G-Loss: {g_loss.item():.6f} - '
                             f'PD-Loss: {pd_loss.item():.6f} - '
                             f'MSE: {mse:.6f} - '
                             f'Time: {t_end - t_start}s')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        save_checkpoint(self.generator, self.g_optimizer, self.last_g)
        save_checkpoint(self.nlayerdiscriminator, self.pd_optimizer, self.last_pd)
        return epg_loss.avg, eppd_loss.avg, epg_error.avg

    def train(self, train_dir, val_dir, patch_stride, batch_size,
              num_workers=0, epochs=50, resume=True):
        last_epoch = -1
        least_error = float('inf')
        if resume and self.history.exists():
            df = pd.read_csv(self.history)
            last_epoch = int(df.iloc[-1]['epoch'])
            least_error = df['val_error'].min()
            load_checkpoint(self.last_g, self.generator, optimizer=self.g_optimizer)
            load_checkpoint(self.last_pd, self.nlayerdiscriminator, optimizer=self.pd_optimizer)
        start_epoch = last_epoch + 1

        # 加载数据
        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size, PATCH_SIZE, patch_stride)
        val_set = PatchSet(val_dir, self.image_size, PATCH_SIZE)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

        self.logger.info('Training...')
        for epoch in range(start_epoch, epochs + start_epoch):

            self.logger.info(f"Learning rate for Generator: "
                             f"{self.g_optimizer.param_groups[0]['lr']}")
            self.logger.info(f"Learning rate for Discriminator: "
                             f"{self.pd_optimizer.param_groups[0]['lr']}")
            train_g_loss, train_pd_loss, train_g_error = self.train_on_epoch(epoch, train_loader)
            val_error = self.test_on_epoch(val_loader)
            csv_header = ['epoch', 'train_g_loss', 'train_pd_loss', 'train_g_error','val_error']
            csv_values = [epoch, train_g_loss, train_pd_loss, train_g_error, val_error]
            log_csv(self.history, csv_values, header=csv_header)

            if val_error < least_error:
                least_error = val_error
                shutil.copy(str(self.last_g), str(self.best))

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        self.generator.eval()
        self.nlayerdiscriminator.eval()
        epoch_error = AverageMeter()
        for data in data_loader:
            data = [im.to(self.device) for im in data]
            inputs, target = data[:-1], data[-1]
            prediction = self.generator(inputs)
            g_loss = F.mse_loss(prediction, target)
            epoch_error.update(g_loss.item())
        return epoch_error.avg

    @torch.no_grad()
    def test(self, test_dir, image_size, patch_size, num_workers=0):
        self.generator.eval()
        load_checkpoint(self.best, model=self.generator)
        self.logger.info('Testing...')
        image_dirs = iter([p for p in test_dir.iterdir() if p.is_dir()])
        test_set = PatchSet(test_dir, self.image_size, image_size)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)
        pixel_scale = 10000
        t_start = timer()
        for inputs in test_loader:
            inputs = [im.to(self.device) for im in inputs]
            inputs[0] = interpolate(interpolate(inputs[0], scale_factor=1 / 16), scale_factor=16, mode='bicubic')
            inputs[2] = interpolate(interpolate(inputs[2], scale_factor=1 / 16), scale_factor=16, mode='bicubic')
            coor = image_coor(inputs[0], self.cut_num_h, self.cut_num_w, patch_size)
            input_patch_list = []
            for input in inputs:
                input_patch_list.append(cut_image(input,coor))
            result_list = []
            for i in range(len(input_patch_list[0])):
                input_patch = [input_patch_list[0][i],input_patch_list[1][i],input_patch_list[2][i],input_patch_list[3][i]]
                prediction_patch = self.generator(input_patch)
                prediction_patch = prediction_patch.squeeze().cpu().numpy()
                prediction_patch = prediction_patch.transpose(1,2,0)
                result_list.append(prediction_patch * pixel_scale)
            result = splicing_image(result_list,coor)
            result = result.astype(np.int16)
            result = result.transpose(2,0,1)
            metadata = {
                        'driver': 'GTiff',
                        'width': self.image_size[1],
                        'height': self.image_size[0],
                        'count': NUM_BANDS,
                        'dtype': np.int16
                    }
            name = f'PRED_{next(image_dirs).stem}.tif'
            save_array_as_tif(result, self.test_dir / name, metadata)
            t_end = timer()
            self.logger.info(f'Time cost: {t_end - t_start}s on {name}')
            t_start = timer()

