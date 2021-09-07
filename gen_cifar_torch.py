import sys
import argparse
import time
import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset, TensorDataset

from torchvision import transforms
from torchvision.datasets import CIFAR10

from torchsummary import summary
from tqdm import tqdm

class Flatten(nn.Module):
	def forward(self, data):
		return data.view(data.size()[0], -1)

class Unflatten(nn.Module):
	def forward(self, data):
		return data.view(data.size()[0], 512, 2, 2)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.label_emb = nn.Sequential(
			nn.Embedding(10, 2048),
			Flatten(),
		)

		self.img_model = nn.Sequential(
			nn.utils.spectral_norm(nn.Conv2d(3, 64, 3, 2, 1)), #[64, 16, 16]
			#nn.Conv2d(3, 32, 3, 2, 1),
			nn.LeakyReLU(0.2),

			nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, 2, 1)), #[128, 8, 8]
			#nn.Conv2d(32, 64, 3, 2, 1),
			#nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),	

			nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 2, 1)), #[256, 4, 4]
			#nn.Conv2d(64, 128, 3, 2, 1),
			#nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),

			nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, 2, 1)), #[512, 2, 2]
			#nn.Conv2d(128, 256, 3, 2, 1),
			#nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),

			Flatten(),
		)

		self.model = nn.Sequential(
			nn.Dropout(0.3),
			nn.Linear(2048, 1),
			nn.Sigmoid(),
		)

	def forward(self, img, label):
		img, label = self.img_model(img), self.label_emb(label)
		model_input = torch.mul(img, label)
		return self.model(model_input)

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		self.label_emb = nn.Sequential(
			nn.Embedding(10, args.g_dim),
			nn.Flatten(),
		)

		self.model = nn.Sequential(
			nn.Linear(args.g_dim, 2048),
			Unflatten(),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1),

			nn.ConvTranspose2d(512, 256, 4, 2, 1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1),

			nn.ConvTranspose2d(256, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1),

			nn.ConvTranspose2d(128, 64, 4, 2, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.1),

			nn.ConvTranspose2d(64, 3, 4, 2, 1),
			nn.Tanh(),	
		)

	def forward(self, noise, label):
		label = self.label_emb(label)
		model_input = torch.mul(noise, label)
		return self.model(model_input)

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()

		#input [3, 32, 32]
		self.model = nn.Sequential(
			nn.Conv2d(3, 16, 3, 2, 1), #[32, 16, 16]
			nn.ReLU(),

			nn.Conv2d(16, 32, 3, 2, 1), #[32, 8, 8]
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 64, 3, 2, 1), #[64, 4, 4]
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.Conv2d(64, 128, 3, 2, 1), #[128, 2, 2]
			nn.BatchNorm2d(128),
			nn.ReLU(),

			Flatten(),

			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Dropout(0.4),
			nn.Linear(128, 10),
		)

	def forward(self, data):
		return self.model(data)

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

def sample(epoch):
	noise = Variable(FloatTensor(np.random.normal(0, 1, (args.classes, args.g_dim))))
	labels = np.array([num for num in range(args.classes)])
	labels = Variable(LongTensor(labels))
	gen_imgs = G(noise, labels)
	for i in range(args.classes):
		img = ((gen_imgs[i].cpu().detach().numpy()+1)/2).transpose(1,2,0)
		img = (img*255).astype(np.uint8)
		plt.imsave(f'./checkImg/epoch{epoch}_{i}.png', img)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--classes', type=int, default=10)
	parser.add_argument('--gen_num', type=int, default=10000)
	parser.add_argument('--batch', type=int, default=50)
	parser.add_argument('--g_dim', type=int, default=2048)
	parser.add_argument('--test-batch', type=int, default=256)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--save_dir', type=str, default='./checkImg3')
	parser.add_argument('--g_ckpt',type = str,default='./DPCGAN_GG.bin')
	args = parser.parse_args()

	G = Generator().cuda()
	G.load_state_dict(torch.load(args.g_ckpt))
	opt_G = Adam(G.parameters(), lr=args.lr)


	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5), (0.5)),
	])

	train_set = CIFAR10(root="CIFAR10", download=True, train=True, transform=transform)
	train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)

	G.eval()

	for i in tqdm(range(500)):
		gen_labels = Variable(LongTensor(np.random.randint(0, args.classes, args.gen_num)))
		noise = Variable(FloatTensor(np.random.normal(0, 1, (args.gen_num, args.g_dim))))
		synthesized = G(noise, gen_labels)

		if (i == 0):
			new_data = synthesized.cpu().detach()
			new_label = gen_labels.cpu().detach()
		else:
			new_data = torch.cat((new_data, synthesized.cpu().detach()), 0)
			new_label = torch.cat((new_label, gen_labels.cpu().detach()), 0)
	os.makedirs(f"{args.save_dir}",exist_ok=True)
	new_data = torch.clamp(new_data, min=-1., max=1.).permute(0,2,3,1).numpy()
	new_label = new_label.numpy()
	print(f"new_data.shape:{new_data.shape}")
	print(f"new_label.shape:{new_label.shape}")
	np.save(f"{args.save_dir}/label.npy",new_label)

	for i in range(10):
		os.makedirs(f"{args.save_dir}/{i}", exist_ok = True)
	for i in range(args.gen_num*500):
		cv2.imwrite(f"{args.save_dir}/{new_label[i]}/{i}.png",(new_data[i]+1)*255/2)
		

