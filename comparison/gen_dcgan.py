'''
initial version
在v2的基础上为生成器的卷积模块之前添加了一个全连接模块
在v2-1的基础上将生成器和判别器的激活函数全部换成ELU
'''


import argparse
import os
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

torch.cuda.set_device(0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.cuda.FloatTensor(np.ones(d_interpolates.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


class Generator(nn.Module):
    def __init__(self, mode='train'):
        super(Generator, self).__init__()

        if mode == 'train':
            prob = 0.25
        else:
            prob = 0

        self.full_layer = nn.Sequential(nn.Sequential(nn.Linear(80, 64 * 1 * 10)))

        self.model = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (2, 9), stride=(1, 3)),  # [24, 32, 2, 28]
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True), nn.Dropout2d(p=prob),
            nn.ConvTranspose2d(32, 16, (3, 6), stride=(1, 3)),  # [24, 16, 4, 90]
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True), nn.Dropout2d(p=prob),
            nn.ConvTranspose2d(16, 8, (4, 5), stride=(2, 3)),  # [24, 8, 10, 186]
            nn.BatchNorm2d(8),
            nn.ELU(inplace=True), nn.Dropout2d(p=prob),
            nn.ConvTranspose2d(8, 8, (4, 5), stride=(2, 3)),  # [24, 1, 22, 303]
            nn.BatchNorm2d(8),
            nn.ELU(inplace=True), nn.Dropout2d(p=prob),
            nn.Conv2d(8, 1, (1, 8), stride=(1, 1)),  # [24, 1, 22, 607]
        )

    def forward(self, noise):
        feature = self.full_layer(noise)
        feature = feature.view(feature.shape[0], 64, 1, 10)
        out_img = self.model(feature)
        # print(out_img.size())

        return out_img



class Discriminator(nn.Module):
    def __init__(self, mode='train'):
        super(Discriminator, self).__init__()

        if mode == 'train':
            prob = 0.25
        else:
            prob = 0

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, (1, 23), stride=(1, 1)),  # [24, 8, 22, 353]
            # nn.BatchNorm2d(8),
            nn.ELU(inplace=True),
            nn.Conv2d(8, 16, (22, 1), stride=1),  # [24, 16, 1, 353]
            # nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, (1, 17), stride=(1, 3)),  # [24, 32, 1, 113]
            # nn.BatchNorm2d(32),
            nn.ELU(inplace=True), nn.Dropout2d(p=prob),
            nn.Conv2d(32, 64, (1, 7), stride=(1, 3)),  # [24, 64, 1, 36]
            # nn.BatchNorm2d(64),
            nn.ELU(inplace=True), nn.Dropout2d(p=prob),
            nn.Conv2d(64, 64, (1, 7), stride=(1, 3)),  # [24, 128, 1, 10]
            # nn.BatchNorm2d(64),
            nn.ELU(inplace=True), nn.Dropout2d(p=prob),
            nn.Conv2d(64, 1, (1, 1), stride=(1, 1)),  # [24, 128, 1, 10]
        )

    def forward(self, imgB):
        feature = self.model(imgB)
        # print(feature.size())

        return feature


class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.num = data.shape[0]

    def __getitem__(self, index):

        data = self.data[index % self.num]
        # std = np.std(data)
        # mean = np.mean(data)
        # data = (data - mean) / std

        return data

    def __len__(self):
        return self.num



class ExGAN():
    def __init__(self, nsub, label, ordata):
        super(ExGAN, self).__init__()
        self.batch_size = 6
        self.n_epochs = 1000

        self.channels = 1
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.lambda_pixel = 0

        self.lambda_gp = 10
        self.nSub = nsub
        self.label = label
        self.data = ordata
        self.active_function = nn.Tanh()

        self.start_epoch = 0

        self.pretrain = False

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.Generator = Generator(mode='train')
        # self.Classifier = Classifier(c_dim=self.c_dim, mode='train')
        self.Discriminator = Discriminator(mode='train')

        self.Generator = self.Generator.cuda()
        # self.Classifier = self.Classifier.cuda()
        self.Discriminator = self.Discriminator.cuda()

        self.centers = {}

        self.train_dataloader = DataLoader(
            TrainDataset(data=self.data),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )



    def sample_images(self, s, l):
        """Saves a generated sample from the validation set"""
        gen = Generator(mode='test')
        gen = gen.cuda()
        gen.load_state_dict(torch.load("Generator_weights-FD-cnn-dav-learning-standard-v0.pth"))

        gendata = np.zeros((1, 1, 22, 1000))
        for _ in range(250):
            noise = Variable(self.Tensor(np.random.normal(0, 1, (5, 80))))
            fake = self.Generator(noise)

            gendata = np.concatenate((gendata, (fake.data.cpu()).numpy()))
        np.save("./gen_data/S%d_class%d.npy" % (s, l), gendata[1:])
        print("S%d_class%d.npy" % (s, l))



    def train(self):
        self.Generator.apply(weights_init_normal)
        self.Discriminator.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.Generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.Discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Train the cnn model
        for e in range(self.n_epochs):
            for i, (data) in enumerate(self.train_dataloader):

                # Model inputs
                noise = Variable(self.Tensor(np.random.normal(0, 1, (data.shape[0], 80))))
                dataB = Variable(data.type(self.Tensor))

                # --------------
                #  Train the generator
                # --------------
                fake_img= self.Generator(noise)
                pre_fake = self.Discriminator(fake_img)

                self.optimizer_G.zero_grad()

                # Adversarial loss
                loss_GAN = -torch.mean(pre_fake)

                # Total loss
                loss_G = loss_GAN

                loss_G.backward()
                self.optimizer_G.step()

                # --------------
                #  Train the discriminator
                # --------------

                self.optimizer_D.zero_grad()

                pre_real = self.Discriminator(dataB)
                pre_fake = self.Discriminator(fake_img.detach())

                # Adversarial loss
                gradient_penalty = compute_gradient_penalty(self.Discriminator, dataB.data, fake_img.data)
                loss_D_GAN = -torch.mean(pre_real) + torch.mean(pre_fake) + self.lambda_gp * gradient_penalty

                # Total loss
                loss_D = loss_D_GAN

                loss_D.backward()
                self.optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------
                # Determine approximate time left
                print(
                        "\r[Epoch:%d  Batch %d/%d] [Total loss_G: %f ] [loss_GAN: %f ] [Total loss_D: %f ]"
                        % (
                            e,
                            i,
                            len(self.train_dataloader),
                            loss_G.item(),
                            loss_GAN.item(),
                            loss_D.item(),
                        )
                )

            torch.save(self.Generator.state_dict(), "Generator_weights-FD-cnn-dav-learning-standard-v0.pth")
            torch.save(self.Discriminator.state_dict(), "Discriminator_weights-FD-cnn-dav-learning-standard-v0.pth")

            if e % 500 == 0:
                self.sample_images(self.nSub, self.label)



def main():
    root = '/document/yanglie/Code_of_Yonghao_Song/T_separate/'
    for i in range(7, 9):
        total_data = scipy.io.loadmat(root + 'A0%dT.mat' % (i + 1))
        data = total_data['data']
        data = data.transpose((2, 1, 0))
        label = total_data['label']

        for l in range(4):
            ordata = np.zeros((1, 1, 22, 1000))
            for k in range(len(label)):
                if label[k] == l+1:
                    d = data[k]
                    d = d[np.newaxis, :]
                    d = d[np.newaxis, :]
                    ordata = np.concatenate((ordata, d))

            exgan = ExGAN(i + 1, l + 1, ordata)
            exgan.train()



if __name__ == "__main__":
    main()
