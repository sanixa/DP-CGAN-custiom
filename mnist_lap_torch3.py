import sys, os
sys.path.append('../pyvacy')

import argparse
import time
import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset

from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

import pyvacy
from pyvacy import optim, analysis, sampling
from pyvacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent

from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

from torchsummary import summary
from tqdm import tqdm
def same_seeds(seed):
    # Python built-in random module
    # random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1)

class Unflatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], 1, 28, 28)

class Unflatten_7(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1, 7, 7)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Sequential(
            nn.Embedding(10, 50),
            nn.Linear(50, 784),
            Unflatten(),
        )

        self.model = nn.Sequential(
            nn.Conv2d(2, 128, 3, 2, 1), #[128, 14, 14]
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, 3, 2, 1), #[128, 7, 7]
            nn.LeakyReLU(),

            Flatten(),

            nn.Dropout(0.4),
            nn.Linear(7*7*128, 1),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, imgs, labels):
        labels = self.label_emb(labels)
        data = torch.cat((imgs, labels), axis=1)
        return self.model(data)

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        
        self.label_emb = nn.Sequential(
            nn.Embedding(10, 50),
            nn.Linear(50, 49),
            Unflatten_7(),
        )

        self.linear = nn.Sequential(
            nn.Linear(self.z_dim, 7*7*128),
            nn.LeakyReLU(),
            Unflatten_7(),
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(129, 128, 4, 2, 1), #[128, 14, 14]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1), #[128, 28, 28]
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, 7, 1, 3),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, z, labels):
        labels, linear = self.label_emb(labels), self.linear(z)
        data = torch.cat((linear, labels), axis=1)
        return self.model(data)


FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

def Laplacian_smoothing(net, sigma=1):
    ## after add dp noise
    for p_net in net.parameters():
        size_param = torch.numel(p_net)
        if size_param < 3:
            pass
        else:
           tmp = p_net.grad.view(-1, size_param)

           c = np.zeros(shape=(1, size_param))
           c[0, 0] = -2.; c[0, 1] = 1.; c[0, -1] = 1.
           c = torch.Tensor(c).cuda()
           zero_N = torch.zeros(1, size_param).cuda()
           c_fft = torch.rfft(c, 1, onesided=False)
           coeff = 1./(1.-sigma*c_fft[...,0])
           ft_tmp = torch.rfft(tmp, 1, onesided=False)
           tmp = torch.zeros_like(ft_tmp)
           tmp[...,0] = ft_tmp[...,0]*coeff
           tmp[...,1] = ft_tmp[...,1]*coeff
           tmp = torch.irfft(tmp, 1, onesided=False)
           tmp = tmp.view(p_net.grad.size())
           p_net.grad.data = tmp

    return net


def generate_image_mnist(iter, netG, fix_noise, save_dir, num_classes=10,
                   img_w=28, img_h=28):
    batchsize = fix_noise.size()[0]
    nrows = 10
    ncols = num_classes
    figsize = (ncols, nrows)
    noise = fix_noise

    sample_list = []
    for class_id in range(num_classes):
        label = torch.full((nrows,), class_id, dtype=torch.long).cuda()
        sample = netG(noise, label)
        sample = sample.view(batchsize, img_w, img_h)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3])
    samples = np.reshape(samples, [nrows * ncols, img_w, img_h])

    plt.figure(figsize=figsize)
    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(samples[i], cmap='gray')
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'samples_{}.png'.format(iter)), dpi=150, format='png')

    del label, noise, sample
    torch.cuda.empty_cache()



try:
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--classes', type=int, default=10)
        parser.add_argument('--class_epoch', type=int, default=20)
        parser.add_argument('--g_dim', type=int, default=100)
        parser.add_argument('--test-batch', type=int, default=256)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--file', type=str, default='./acc_result/acc_mnist')
        parser.add_argument('--clip', type=float, default=1.0)
        parser.add_argument('--batch', type=int, default=64)
        parser.add_argument('--iter', type=int, default=20000)
        parser.add_argument('--noise', type=float, default=0.11145)
        parser.add_argument('--delta', type=float, default=1e-5)
        parser.add_argument('--exp_name', type=str)
        args = parser.parse_args()


        transform = transforms.ToTensor()
        train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
        train_loader = DataLoader(train_set, 
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=len(train_set),
                    sample_rate=args.batch/len(train_set),
            ),
        )

        if (args.noise >= 0):
            G = Generator(args.g_dim).cuda()
            G.train()

            D = Discriminator().cuda()
            D.train()

            optimizerD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
            optimizerG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

            privacy_engine = PrivacyEngine(
                D,
                sample_rate=args.batch/len(train_set),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=args.noise,
                max_grad_norm=args.clip,
            )
            privacy_engine.attach(optimizerD)

            '''
            def compute_epsilon(steps,nm):
                orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
                sampling_probability = args.batch/len(train_set)
                rdp = compute_rdp(q=sampling_probability,
                                noise_multiplier=nm,
                                steps=steps,
                                orders=orders)
                #Delta is set to 1e-5 because MNIST has 60000 training points.
                return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

            eps = compute_epsilon(args.iter,args.noise)
            print(eps)
            '''

            criterion = nn.BCELoss()
            bar = tqdm(range(args.iter+1))
            for iteration in bar:
                img, label = next(iter(train_loader)) 
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                batch_size = img.size()[0]

                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake  = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                r_img = Variable(img.type(FloatTensor))
                label = Variable(label.type(LongTensor))

                noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.g_dim))))
                gen_label = Variable(LongTensor(np.random.randint(0, args.classes, batch_size)))

                
                r_logit = D(r_img, label)
                r_loss_D = criterion(r_logit, valid)

                gen_img = G(noise, gen_label)
                f_logit = D(gen_img, gen_label)
                f_loss_D = criterion(f_logit, fake)
                loss_D = (r_loss_D + f_loss_D) / 2

                loss_D.backward()
                D = Laplacian_smoothing(D)
                optimizerD.step()	

                gen_img = G(noise, gen_label)
                f_logit = D(gen_img, gen_label)
                loss_G = criterion(f_logit, valid)

                loss_G.backward()
                optimizerG.step()
                		
                bar.set_description(f"Epoch [{iteration+1}/{args.iter}] d_loss: {loss_D.item():.5f} g_loss: {loss_G.item():.5f}")
                if((iteration+1) %1000 == 0):
                    save_dir = "./checkpoint/"+ args.exp_name
                    print(f"save dir:{save_dir}")
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    fix_noise = FloatTensor(np.random.normal(0, 1, (10, args.g_dim)))
                    generate_image_mnist(iteration+1, G, fix_noise, save_dir)
                    torch.save(G.state_dict(), f"./checkpoint/"+args.exp_name+f"/iteration{(iteration+1)}.ckpt")
                    torch.save(D.state_dict(), f"./checkpoint/"+args.exp_name+f"/D_iteration{(iteration+1)}.ckpt")
                
                if((iteration+1) %args.iter == 0):
                    break

except:
    raise




