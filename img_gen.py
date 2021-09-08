import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import DataLoader, Dataset, TensorDataset
from opacus.utils.module_modification import convert_batchnorm_modules

from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

class Unflatten_7(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1, 7, 7)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Sequential(
            nn.Embedding(10, 50),
            nn.Linear(50, 49),
            Unflatten_7(),
        )

        self.linear = nn.Sequential(
            nn.Linear(100, 7*7*128),
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

    def forward(self, z, labels):
        labels, linear = self.label_emb(labels), self.linear(z)
        data = torch.cat((linear, labels), axis=1)
        return self.model(data)

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

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--classes', type=int, default=10)
	parser.add_argument('--gan_epoch', type=int, default=1000)
	parser.add_argument('--g_dim', type=int, default=100)
	parser.add_argument('--model-path', type=str, default='checkpoint/DPCGAN_MNIST.bin')
	parser.add_argument('--save-dir', type=str)
	parser.add_argument('--dataset', type=str)
	args = parser.parse_args()

	#eps = args.model_path.split('_')[-1].split('.')[0].split('s')[-1]
	#print(f"eps = {eps}")
	#dir_path = f'data/CGAN_eps{eps}'
	if not os.path.isfile(args.model_path):
		sys.exit("model does not exist")

	print('loading model...')
	if args.dataset == 'mnist':
		netG = Generator().cuda()
	elif args.dataset == 'cifar_10':
		netG = GeneratorDCGAN_cifar(z_dim=args.g_dim).cuda()
		netG = convert_batchnorm_modules(netG)
	netG.load_state_dict(torch.load(args.model_path))

	print(f"save dir:{args.save_dir}")
	if not os.path.isdir(args.save_dir):
		os.mkdir(args.save_dir)

	for i in tqdm(range(1, 6)):
		noise = Variable(FloatTensor(np.random.normal(0, 1, (2000, args.g_dim))).cuda())
		label = Variable(LongTensor(np.tile(np.arange(10), 200)).cuda())
		image = Variable(netG(noise, label))

		if (i == 1):
			new_image = image.cpu().detach().numpy()
			new_label = label.cpu().detach().numpy()
			new_noise = noise.cpu().detach().numpy()
		else:
			new_image = np.concatenate((new_image, image.cpu().detach().numpy()), axis=0)
			new_label = np.concatenate((new_label, label.cpu().detach().numpy()), axis=0)
			new_noise = np.concatenate((new_noise, noise.cpu().detach().numpy()), axis=0)

	np.savez_compressed(f"{args.save_dir}/generated.npz", noise=new_noise, img_r01=new_image)
	'''
	np.save(f"{args.save_dir}/train_data.npy", new_image)
	np.save(f"{args.save_dir}/train_label.npy", new_label)


	for i in tqdm(range(1, 6)):
		noise = Variable(FloatTensor(np.random.normal(0, 1, (2000, args.g_dim))).cuda())
		label = Variable(LongTensor(np.tile(np.arange(10), 200)).cuda())
		image = Variable(netG(noise, label))

		if (i == 1):
			new_image = image.cpu().detach().numpy()
			new_label = label.cpu().detach().numpy()
			new_noise = noise.cpu().detach().numpy()
		else:
			new_image = np.concatenate((new_image, image.cpu().detach().numpy()), axis=0)
			new_label = np.concatenate((new_label, label.cpu().detach().numpy()), axis=0)
			new_noise = np.concatenate((new_noise, noise.cpu().detach().numpy()), axis=0)

	np.save(f"{args.save_dir}/test_data.npy", new_image)
	np.save(f"{args.save_dir}/test_label.npy", new_label)
	'''
	train_set, test_set = None, None
	if args.dataset == 'mnist':
		transform = transforms.ToTensor()
		train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
		test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)
	elif args.dataset == 'cifar_10':
		transform = transforms.Compose([
						transforms.Grayscale(1),
						transforms.ToTensor()])
		train_set = CIFAR10(root="CIFAR10", download=True, train=True, transform=transform)
		test_set = CIFAR10(root="CIFAR10", download=True, train=False, transform=transform)


	train_loader = torch.utils.data.DataLoader(train_set, batch_size=10000, shuffle=True)
	for data, target in train_loader:
		np.save(f"{args.save_dir}/train_data.npy", data.cpu().detach().numpy())
		np.save(f"{args.save_dir}/train_label.npy", target.cpu().detach().numpy())

	test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=True)
	for data, target in test_loader:
		np.save(f"{args.save_dir}/test_data.npy", data.cpu().detach().numpy())
		np.save(f"{args.save_dir}/test_label.npy", target.cpu().detach().numpy())