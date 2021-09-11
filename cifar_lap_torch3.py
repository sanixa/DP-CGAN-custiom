
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


def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


def pixel_norm(x, eps=1e-10):
    '''
    Pixel normalization
    :param x:
    :param eps:
    :return:
    '''
    return x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=1, keepdim=True) + eps)


def l2_norm(v, eps=1e-10):
    '''
    L2 normalization
    :param v:
    :param eps:
    :return:
    '''
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    '''
    Spectral Normalization
    '''

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2_norm(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2_norm(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
        del u, v, w

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2_norm(u.data)
        v.data = l2_norm(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


#@torchsnooper.snoop()
class GeneratorDCGAN_cifar(nn.Module):
    def __init__(self, z_dim=10, model_dim=64, num_classes=10, outact=nn.Tanh()):
        super(GeneratorDCGAN_cifar, self).__init__()

        self.cdist = nn.CosineSimilarity(dim = 1, eps= 1e-9)
        self.grads = []
        self.grad_dict = {}

        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        fc = nn.Linear(z_dim + num_classes, z_dim * 1 * 1)
        deconv1 = nn.ConvTranspose2d(z_dim, model_dim * 4, 4, 1, 0, bias=False)
        deconv2 = nn.ConvTranspose2d(model_dim * 4, model_dim * 2, 4, 2, 1, bias=False)
        deconv3 = nn.ConvTranspose2d(model_dim * 2, model_dim, 4, 2, 1, bias=False)
        deconv4 = nn.ConvTranspose2d(model_dim, 1, 4, 2, 1, bias=False)

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        self.deconv4 = deconv4
        self.BN_1 = nn.BatchNorm2d(model_dim * 4)
        self.BN_2 = nn.BatchNorm2d(model_dim * 2)
        self.BN_3 = nn.BatchNorm2d(model_dim)
        self.fc = fc
        self.relu = nn.ReLU()
        self.outact = outact

        ''' reference by https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch/blob/master/gan_cifar.py
        nn.ConvTranspose2d(z_dim, model_dim * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(model_dim * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(model_dim * 8, model_dim * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(model_dim * 4, model_dim * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(model_dim * 2, model_dim, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(model_dim, nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 64 x 64
        '''

    def forward(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)
        output = self.fc(z_in)
        output = output.view(-1, self.z_dim, 1, 1)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv1(output)
        output = self.BN_1(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv2(output)
        output = self.BN_2(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv3(output)
        output = self.BN_3(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv4(output)
        output = self.outact(output)

        return output.view(-1, 32 * 32)

#@torchsnooper.snoop()
class DiscriminatorDCGAN_cifar(nn.Module):
    def __init__(self, model_dim=64, num_classes=10, if_SN=False):
        super(DiscriminatorDCGAN_cifar, self).__init__()

        self.model_dim = model_dim
        self.num_classes = num_classes

        if if_SN:
            self.conv1 = SpectralNorm(nn.Conv2d(1, model_dim, 4, 2, 1, bias=False))
            self.conv2 = SpectralNorm(nn.Conv2d(model_dim, model_dim * 2, 4, 2, 1, bias=False))
            self.conv3 = SpectralNorm(nn.Conv2d(model_dim * 2, model_dim * 4, 4, 2, 1, bias=False))
            #self.conv4 = SpectralNorm(nn.Conv2d(model_dim * 4, 1, 4, 1, 0, bias=False))
            self.BN_1 = nn.BatchNorm2d(model_dim * 2)
            self.BN_2 = nn.BatchNorm2d(model_dim * 4)
            self.linear = SpectralNorm(nn.Linear(4 * 4 * 4 * model_dim, 1))
            self.linear_y = SpectralNorm(nn.Embedding(num_classes, 4 * 4 * 4 * model_dim))
        else:
            self.conv1 = nn.Conv2d(1, model_dim, 4, 2, 1, bias=False)
            self.conv2 = nn.Conv2d(model_dim, model_dim * 2, 4, 2, 1, bias=False)
            self.conv3 = nn.Conv2d(model_dim * 2, model_dim * 4, 4, 2, 1, bias=False)
            #self.conv4 = nn.Conv2d(model_dim * 4, 1, 4, 1, 0, bias=False)
            self.BN_1 = nn.BatchNorm2d(model_dim * 2)
            self.BN_2 = nn.BatchNorm2d(model_dim * 4)
            self.linear = nn.Linear(4 * 4 * 4 * model_dim, 1)
            self.linear_y = nn.Embedding(num_classes, 4 * 4 * 4 * model_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.Sigmoid = nn.Sigmoid()

        ''' reference by https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch/blob/master/gan_cifar.py
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        '''

    def forward(self, input, y):
        input = input.view(-1, 1, 32, 32)
        h = self.LeakyReLU(self.conv1(input))
        h = self.LeakyReLU(self.BN_1(self.conv2(h)))
        h = self.LeakyReLU(self.BN_2(self.conv3(h)))
        #h = self.Sigmoid(self.conv4(h))
        h = h.view(-1, 4 * 4 * 4 * self.model_dim)
        out = self.linear(h)
        out += torch.sum(self.linear_y(y) * h, dim=1, keepdim=True)
        out = self.Sigmoid(out)
        return out


FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor


def generate_image_cifar(iter, netG, fix_noise, save_dir, num_classes=10,
                   img_w=32, img_h=32):
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

try:
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--classes', type=int, default=10)
        parser.add_argument('--class_epoch', type=int, default=20)
        parser.add_argument('--g_dim', type=int, default=100)
        parser.add_argument('--test-batch', type=int, default=256)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--file', type=str, default='./acc_result/acc_mnist')
        parser.add_argument('--clip', type=float, default=1.0)
        parser.add_argument('--batch', type=int, default=64)
        parser.add_argument('--iter', type=int, default=20000)
        parser.add_argument('--noise', type=float, default=0.11145)
        parser.add_argument('--delta', type=float, default=1e-5)
        parser.add_argument('--exp_name', type=str)
        args = parser.parse_args()


        transform = transforms.Compose([
                            transforms.Grayscale(1),
                            transforms.ToTensor()])
        train_set = CIFAR10(root="CIFAR10", download=True, train=True, transform=transform)
        train_loader = DataLoader(train_set, 
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=len(train_set),
                    sample_rate=args.batch/len(train_set),
            ),
        )

        if (args.noise >= 0):
            G = GeneratorDCGAN_cifar(z_dim=args.g_dim).cuda()
            G.train()

            D = DiscriminatorDCGAN_cifar()
            D.train()

            #G = convert_batchnorm_modules(G)
            D = convert_batchnorm_modules(D).cuda()

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
                for i in range(2):
                    img, label = next(iter(train_loader)) 
                    optimizerD.zero_grad()
                    optimizerG.zero_grad()

                    batch_size = img.size()[0]
                    valid = Variable(FloatTensor(batch_size).fill_(1.0), requires_grad=False)
                    fake  = Variable(FloatTensor(batch_size).fill_(0.0), requires_grad=False)

                    r_img = Variable(img.type(FloatTensor))
                    label = Variable(label.type(LongTensor))

                    noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.g_dim))))
                    gen_label = Variable(LongTensor(np.random.randint(0, args.classes, batch_size)))


                    r_logit = D(r_img, label).view(-1, 1).squeeze(1)
                    r_loss_D = criterion(r_logit, valid)

                    gen_img = G(noise.detach(), gen_label)
                    f_logit = D(gen_img, gen_label).view(-1, 1).squeeze(1)
                    f_loss_D = criterion(f_logit, fake)
                    loss_D = (r_loss_D + f_loss_D) / 2

                    loss_D.backward()
                    D = Laplacian_smoothing(D)
                    optimizerD.step()
                iteration = iteration + 1
                
                for i in range(3):
                    noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, args.g_dim))))
                    gen_label = Variable(LongTensor(np.random.randint(0, args.classes, batch_size)))
                    
                    gen_img = G(noise, gen_label)
                    f_logit = D(gen_img, gen_label).view(-1, 1).squeeze(1)
                    loss_G = criterion(f_logit, valid)

                    loss_G.backward()
                    optimizerG.step()

                bar.set_description(f"Epoch [{iteration+1}/{args.iter}] d_loss: {loss_D.item():.5f} g_loss: {loss_G.item():.5f}")
                if((iteration+1) %1000 == 0):
                    save_dir = "./checkpoint_cifar/"+ args.exp_name
                    print(f"save dir:{save_dir}")
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    fix_noise = FloatTensor(np.random.normal(0, 1, (10, args.g_dim)))
                    generate_image_cifar(iteration+1, G, fix_noise.detach(), save_dir)
                    torch.save(G.state_dict(), f"./checkpoint_cifar/"+args.exp_name+f"/iteration{(iteration+1)}.ckpt")
                    torch.save(D.state_dict(), f"./checkpoint_cifar/"+args.exp_name+f"/D_iteration{(iteration+1)}.ckpt")

                if((iteration+1) %args.iter == 0):
                    break


except:
    raise






