"""
model of MI_gen
generate the eeg with the constrains of csp
tried to use the same or another discriminator to discriminate the csp-processed fake eeg data
"""


import argparse
import numpy as np
import random
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import cv2 as cv
import torchvision
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3, 4, 5'
# gpus = [0, 1, 2, 3]
# device = torch.cuda.set_device('cuda:{}'.format(gpus[0]))

writer = SummaryWriter('./TensorBoardX')  # the tb of 25 is out of Lab4

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1650, help="number of epochs of training")  # 500 is ok
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.1, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image samples")
parser.add_argument('--nz', type=int, default=64, help="size of the latent z vector used as the generator input.")
opt = parser.parse_args()
# latent space means the space of noise which is used to generate fake sample

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0,1,2,3,4' if torch.cuda.is_available() else 'cpu')


batch_size = 5
dropout_level = 0.05
nz = opt.nz
img_shape = (16, 1000)
T = 4.0


def weights_init(m):

    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)




class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.nz = opt.nz
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 5), stride=(4, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 13), stride=(1, 3)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 15), stride=(1, 3)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.layer6 = nn.Sequential(
            # H_out=(H_in?1)×stride[0]?2×padding[0]+kernel_size[0]+output_padding[0]
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(1, 15), stride=(1, 3)),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 13), stride=(1, 3)),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 5), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.layer9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(4, 5), stride=(4, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU()
        )
        self.layer10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 2), stride=(1, 1)),
            nn.BatchNorm2d(1),
            # nn.Sigmoid()
            nn.Tanh()
        )
        '''
        self.layer0 = nn.Sequential(
            # nn.Linear(100, 1600),
            # nn.LeakyReLU(0.2),
            nn.Linear(1600, 25600),
            nn.LeakyReLU(0.2)
        )
        self.layer1 = nn.Sequential(
            # H_out=(H_in?1)×stride[0]?2×padding[0]+kernel_size[0]+output_padding[0]
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 15), stride=(1, 3)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 13), stride=(1, 3)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 5), stride=(1, 2)),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 5), stride=(2, 1)),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 2), stride=(1, 1)),
            # nn.BatchNorm2d(1),
            # nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, z):
        out = self.layer0(z)
        out = out.view(out.size(0), 128, 4, 50)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, 23), stride=(1, 1)),
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=30, kernel_size=(22, 1), stride=(1, 1)),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(1, 17), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6)),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(1, 7), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6)),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2)
        )
        '''
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, 23), stride=(1, 1)),
            # nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=30, kernel_size=(22, 1), stride=(1, 1)),
            # nn.BatchNorm2d(30),
            nn.LeakyReLU(0.2)
            # nn.Dropout(0.5)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(1, 17), stride=(1, 1)),
            # nn.BatchNorm2d(30),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6))
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(1, 7), stride=(1, 1)),
            # nn.BatchNorm2d(30),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6))
        )

        self.layer22 = nn.Sequential(  # it's not very ok for csp
            nn.Conv2d(in_channels=10, out_channels=30, kernel_size=(4, 1), stride=(4, 1)),
            #  nn.BatchNorm2d(30),
            nn.LeakyReLU(0.2)
            # nn.Dropout(0.5)
        )
        self.layer222 = nn.Sequential(  # it's not very ok for csp
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(4, 1), stride=(1, 1)),
            # nn.BatchNorm2d(30),
            nn.LeakyReLU(0.2)
            # nn.Dropout(0.5)
        )
        self.layer23 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(1, 17), stride=(1, 1)),
            # nn.BatchNorm2d(30),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6))
        )
        self.layer24 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(1, 7), stride=(1, 1)),
            # nn.BatchNorm2d(30),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6))
        )

        self.dense_layers = nn.Sequential(
            # nn.Dropout(0.3),
            # nn.Linear(106624, 600),  # 61440
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),
            nn.Linear(750, 1)
        )
        self.dense_layers_csp = nn.Sequential(
            # nn.Dropout(0.3),
            # nn.Linear(3000, 200),  # 30720 68992
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),
            nn.Linear(750, 1)
        )

    def forward(self, x):
        out = self.layer1(x)

        if x.shape[2] == 22:
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.view(out.size(0), -1)
            out = self.dense_layers(out)
        elif x.shape[2] == 16:
            out = self.layer22(out)
            out = self.layer222(out)
            out = self.layer23(out)
            out = self.layer24(out)
            out = out.view(out.size(0), -1)
            out = self.dense_layers_csp(out)
        return out


# Loss weight for gradient penalty
lambda_gp = 10


def wgan(datatrain, cspdatatrain, label, nclass, nseed, sub_index, Cov, Dis_mean, Dis_std, P, B, Wb):
    # Initialize generator and discriminator
    discriminator = Discriminator()
    # discriminator2 = Discriminator()
    generator = Generator()
    discriminator.apply(weights_init)
    # discriminator2.apply(weights_init)
    generator.apply(weights_init)

    discriminator = discriminator.cuda()
    # discriminator2 = discriminator2.cuda()
    generator = generator.cuda()
    discriminator = nn.DataParallel(discriminator, device_ids=[0, 1, 2, 3, 4])
    # discriminator2 = nn.DataParallel(discriminator2, device_ids=[0, 1, 2, 3, 4])
    generator = nn.DataParallel(generator, device_ids=[0, 1, 2, 3, 4])
    discriminator.to(device)
    # discriminator2.to(device)
    generator.to(device)
    print('Generator')
    print(generator)
    print('Discriminator')
    print(discriminator)

    random.seed(nseed)
    np.random.seed(nseed)
    torch.manual_seed(nseed)
    torch.cuda.manual_seed(nseed)

    datatrain = torch.from_numpy(datatrain)
    cspdatatrain = torch.from_numpy(cspdatatrain)
    label = torch.from_numpy(label)

    dataset = torch.utils.data.TensorDataset(datatrain, cspdatatrain, label)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    Wb = torch.Tensor(Wb.transpose()).cuda()

    def compute_gradient_penalty(D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        '''
        real_samples = np.array(real_samples.cpu())
        real_samples = np.squeeze(real_samples)
        real_samples = torch.from_numpy(real_samples).cuda()
        fake_samples = np.array(fake_samples.cpu())
        fake_samples = np.squeeze(fake_samples)
        fake_samples = torch.from_numpy(fake_samples).cuda()
        '''
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))  # (5,1,1)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        # print (interpolates.shape)
        # interpolates = interpolates.cpu().detach().numpy()
        # interpolates = np.expand_dims(interpolates, axis=1)
        # interpolates = torch.from_numpy(interpolates).cuda()
        d_interpolates = D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # ----------
    #  Training
    # ----------
    new_data = []
    batches_done = 0
    discriminator.train()
    # discriminator2.train()
    generator.train()
    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader, 0):  # 50 data

            imgs, csp_imgs, _ = data
            imgs = imgs.cuda()
            csp_imgs = csp_imgs.cuda()

            # Configure input
            real_data = Variable(imgs.type(Tensor))
            real_csp_data = Variable(csp_imgs.type(Tensor))

            if i % 1 == 0:
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                # optimizer_D2.zero_grad()
                # Sample noise as generator input
                # no noise, but use part of the origin input randomly
                '''
                seg5 = np.zeros((imgs.shape[0], 1, 16, 1000))
                for o5 in range(imgs.shape[0]):
                    rand_trial_number = np.random.randint(0, np.shape(datatrain)[0])
                    seg5[o5, 0, :, :] = datatrain[rand_trial_number, 0, :, :]
                z = torch.randn(imgs.shape[0], 1, 16, 1000).cuda()
                for zi in range(imgs.shape[0]):
                    for zj in range(16):
                        for zk in range(1000):
                            z[zi, 0, zj, zk] = seg5[zi, 0, zj, zk]
                '''
                # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
                z = torch.randn(imgs.shape[0], 1600).cuda()
                # Generate a batch of images
                # !!! directly generate from randn
                fake_imgs = generator(z)

                # fake_csp_imgs = [Wb.mm(fake_imgs[fci_index, 0, :, :]) for fci_index in range(5)]
                fake_csp_imgs = torch.randn(fake_imgs.shape[0], 1, 16, 1000).cuda()
                for fci_index in range(fake_imgs.shape[0]):
                    fake_csp_imgs[fci_index, 0, :, :] = Wb.mm(fake_imgs[fci_index, 0, :, :])
                # Real images
                # ttt = discriminator(fake_csp_imgs)
                real_validity = discriminator(real_data)
                real_csp_validity = discriminator(real_csp_data)
                # Fake images
                fake_validity = discriminator(fake_imgs)
                fake_csp_validity = discriminator(fake_csp_imgs)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_imgs.data)
                csp_gradient_penalty = compute_gradient_penalty(discriminator, real_csp_data.data, fake_csp_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                writer.add_scalar('Train/Discriminator_eeg', d_loss, epoch)
                writer.flush()

                d_csp_loss = -torch.mean(real_csp_validity) + torch.mean(fake_csp_validity) + lambda_gp * csp_gradient_penalty

                writer.add_scalar('Train/Discriminator_csp', d_csp_loss, epoch)
                writer.flush()

                d_loss += d_csp_loss * 0.1
                d_loss.backward()
                optimizer_D.step()
                # d_csp_loss.backward()
                # optimizer_D2.step()
                # use for tensorboardX
                # dd = d_loss + d_csp_loss
                writer.add_scalar('Train/Discriminator', d_loss, epoch)
                writer.flush()
                torch.cuda.empty_cache()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
                optimizer_G.zero_grad()
                # -----------------
                #  Train Generator
                # -----------------
                '''
                seg5 = np.zeros((imgs.shape[0], 1, 16, 1000))
                for o5 in range(imgs.shape[0]):
                    rand_trial_number = np.random.randint(0, np.shape(datatrain)[0])
                    seg5[o5, 0, :, :] = datatrain[rand_trial_number, 0, :, :]

                z = torch.randn(imgs.shape[0], 1, 16, 1000).cuda()
                for zi in range(imgs.shape[0]):
                    for zj in range(16):
                        for zk in range(1000):
                            z[zi, 0, zj, zk] = seg5[zi, 0, zj, zk]
                grid = torchvision.utils.make_grid(z[0, :, :, :])
                writer.add_image('Input real data', grid, global_step=0)
                '''
                # z = torch.randn(imgs.shape[0], 100).cuda()

                # Generate a batch of images
                fake_imgs = generator(z)

                if epoch > 1398:
                    print(epoch)
                    fake_data = fake_imgs.data[:25].cpu().numpy()
                    new_data.append(fake_data)


                fake_csp_imgs = torch.randn(fake_imgs.shape[0], 1, 16, 1000).cuda()
                for fci_index in range(fake_imgs.shape[0]):
                    fake_csp_imgs[fci_index, 0, :, :] = Wb.mm(fake_imgs[fci_index, 0, :, :])
                # writer.add_graph(generator, z)

                # the constrains of the covariance matrix and eigenvalue
                tmp_fake_imgs = np.array(fake_imgs.cpu().detach())
                cov_loss = []
                ev_loss = []
                for cov_index in range(imgs.shape[0]):
                    one_fake_imgs = tmp_fake_imgs[cov_index, 0, :, :]
                    oneone = np.dot(one_fake_imgs, one_fake_imgs.transpose())
                    one_cov = oneone/np.trace(oneone)
                    one_dis = np.sqrt(np.sum(np.power(one_cov - Cov, 2)))
                    one_cov_loss = np.abs(one_dis - Dis_mean) / Dis_std
                    if one_cov_loss <= 1:
                        one_cov_loss = 0
                    cov_loss.append(one_cov_loss)

                    BTP = np.dot(B.transpose(), P)
                    one_ev = np.dot(BTP, one_cov)
                    one_ev = np.dot(one_ev, BTP.transpose())
                    one_ev_four = np.diag(one_ev)[0:4]
                    one_ev_loss = np.mean(one_ev_four)
                    one_ev_loss = np.abs(np.log(one_ev_loss))
                    ev_loss.append(one_ev_loss)

                cov_loss = np.mean(cov_loss).astype(np.float32)
                writer.add_scalar('Train/G_Cov_loss', cov_loss, epoch)
                writer.flush()

                ev_loss = np.mean(ev_loss).astype(np.float32)
                writer.add_scalar('Train/G_Ev_loss', ev_loss, epoch)
                writer.flush()

                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                fake_csp_validity = discriminator(fake_csp_imgs)
                g_loss = -torch.mean(fake_validity) - torch.mean(fake_csp_validity) * 0.1
                writer.add_scalar('Train/G_g_loss', g_loss, epoch)
                writer.flush()

                g_loss.data = g_loss.data + torch.tensor(3 * cov_loss).cuda() + torch.tensor(10 * ev_loss).cuda()
                g_loss.backward()
                optimizer_G.step()

                # use for tensorboardX
                writer.add_scalar('Train/Generator', g_loss, epoch)
                writer.flush()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                # save the generate data each 5-epoch
                if epoch % 5 == 0:
                    save_fake_img = tmp_fake_imgs[0, 0, :, :]
                    # cv.imwrite('/home/syh/Documents/MI/code/MI-GAN/Lab4/process/cv3/class' + str(nclass) +
                    #            '_epoch_' + str(epoch) + '.png', save_fake_img)
                    plt.imshow(save_fake_img, cmap='Greys', aspect='auto', origin='lower')
                    plt.savefig('/home/syh/Documents/MI/code/MI-GAN/Lab4/process/justT/plt' + str(sub_index) +
                                '/class' + str(nclass) + '_epoch_' + str(epoch) + '.jpg')


                # writer.add_graph(generator, z)
                grid0 = torchvision.utils.make_grid(fake_imgs[0, 0, :, :])
                writer.add_image('output fake data0', grid0, global_step=0)


    # writer.close()
    torch.save(discriminator, '/home/syh/Documents/MI/code/MI-GAN/model/justT/S'
               + str(sub_index) + '_D_class' + str(nclass) + '.pth')
    torch.save(generator, '/home/syh/Documents/MI/code/MI-GAN/model/justT/S'
               + str(sub_index) + '_G_class' + str(nclass) + '.pth')
    discriminator.eval()
    generator.eval()

    """
    for epoch in range(250):
        for i, data in enumerate(dataloader, 0):
            imgs, csp_imgs, _ = data
            imgs = imgs.cuda()
            '''
            seg5 = np.zeros((imgs.shape[0], 1, 16, 1000))
            for o5 in range(imgs.shape[0]):
                rand_trial_number = np.random.randint(0, np.shape(datatrain)[0])
                seg5[o5, 0, :, :] = datatrain[rand_trial_number, 0, :, :]

            z = torch.randn(imgs.shape[0], 1, 16, 1000).cuda()
            for zi in range(imgs.shape[0]):
                for zj in range(16):
                    for zk in range(1000):
                        z[zi, 0, zj, zk] = seg5[zi, 0, zj, zk]
            '''
            z = torch.randn(imgs.shape[0], 1600).cuda()
            # Generate a batch of images
            fake_imgs = generator(z)

            # the constrain of the cov-matrix
            tmp_fake_imgs = np.array(fake_imgs.cpu().detach())
            cov_loss = []
            ev_loss = []
            for cov_index in range(imgs.shape[0]):
                one_fake_imgs = tmp_fake_imgs[cov_index, 0, :, :]
                oneone = np.dot(one_fake_imgs, one_fake_imgs.transpose())
                one_cov = oneone / np.trace(oneone)
                one_dis = np.sqrt(np.sum(np.power(one_cov - Cov, 2)))
                one_cov_loss = np.abs(one_dis - Dis_mean) / Dis_std
                if one_cov_loss <= 1:
                    one_cov_loss = 0
                cov_loss.append(one_cov_loss)

                BTP = np.dot(B.transpose(), P)
                one_ev = np.dot(BTP, one_cov)
                one_ev = np.dot(one_ev, BTP.transpose())
                one_ev_four = np.diag(one_ev)[0:4]
                one_ev_loss = np.mean(one_ev_four)
                one_ev_loss = np.abs(np.log(one_ev_loss))
                ev_loss.append(one_ev_loss)

            cov_loss = np.mean(cov_loss).astype(np.float32)
            writer.add_scalar('Generate/Cov_loss', cov_loss, epoch)
            writer.flush()

            ev_loss = np.mean(ev_loss).astype(np.float32)
            writer.add_scalar('Generate/Ev_loss', ev_loss, epoch)
            writer.flush()

            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)
            writer.add_scalar('Generate/g_loss', g_loss, epoch)
            writer.flush()

            g_loss.data = g_loss.data + torch.tensor(cov_loss).cuda() + torch.tensor(3 * ev_loss).cuda()
            writer.add_scalar('Generate/Generator', g_loss, epoch)
            writer.flush()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), g_loss.item())
            )

            # !!! what is that use for?
            # one epoch get one fake data
            # change to 5
            if i % opt.sample_interval == 0:
                # print(batches_done % opt.sample_interval)
                fake_data = fake_imgs.data[:25].cpu().numpy()
                new_data.append(fake_data)

            # fake_data = fake_imgs.data[:25].cpu().numpy()
            # new_data.append(fake_data)
    """
    new_data = np.concatenate(new_data)
    new_data = np.asarray(new_data)
    writer.close()
    return new_data

