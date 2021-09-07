import sys
import argparse
import time
import cv2
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

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
	parser.add_argument('--gan_epoch', type=int, default=100)
	parser.add_argument('--class_epoch', type=int, default=40)
	parser.add_argument('--batch', type=int, default=50)
	parser.add_argument('--g_dim', type=int, default=2048)
	parser.add_argument('--test-batch', type=int, default=256)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--file', type=str, default='./acc_result/acc_cifar.txt')
	args = parser.parse_args()

	G = Generator().cuda()
	G.train()
	opt_G = Adam(G.parameters(), lr=args.lr)

	D = Discriminator().cuda()
	D.train()
	opt_D = Adam(D.parameters(), lr=args.lr)

	criterion = nn.BCELoss()

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5), (0.5)),
	])

	train_set = CIFAR10(root="CIFAR10", download=True, train=True, transform=transform)
	train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)

	for epoch in range(args.gan_epoch):
		bar = tqdm(enumerate(train_loader))
		for i, (imgs, labels) in bar:
			#imgs = imgs.cuda()
			bs = imgs.size(0)

			valid = Variable(FloatTensor(bs, 1).fill_(1.0), requires_grad=False)
			fake  = Variable(FloatTensor(bs, 1).fill_(0.0), requires_grad=False)

			r_imgs = Variable(imgs.type(FloatTensor))
			labels = Variable(labels.type(LongTensor))

			noise = Variable(FloatTensor(np.random.normal(0, 1, (bs, args.g_dim))))
			gen_labels = Variable(LongTensor(np.random.randint(0, args.classes, bs)))

			gen_imgs = G(noise, gen_labels)
			f_logit = D(gen_imgs, gen_labels)
			loss_G = criterion(f_logit, valid)

			opt_G.zero_grad()
			loss_G.backward()
			opt_G.step()

			r_logit = D(r_imgs, labels)
			r_loss_D = criterion(r_logit, valid)

			gen_imgs = G(noise, gen_labels)
			f_logit = D(gen_imgs, gen_labels)
			f_loss_D = criterion(f_logit, fake)
			loss_D = (r_loss_D + f_loss_D) / 2

			opt_D.zero_grad()
			loss_D.backward()
			opt_D.step()


			bar.set_description(f"Epoch [{epoch+1}/{args.gan_epoch}] d_loss: {loss_D.item():.5f} g_loss: {loss_G.item():.5f}")

		G.eval()
		sample(epoch)
		G.train()

	torch.save(G.state_dict(), f"DPCGAN_GG.bin")
	G.eval()

	for i in tqdm(range(500)):
		gen_labels = Variable(LongTensor(np.random.randint(0, args.classes, 100)))
		noise = Variable(FloatTensor(np.random.normal(0, 1, (100, args.g_dim))))
		synthesized = G(noise, gen_labels)

		if (i == 0):
			new_data = synthesized.cpu().detach()
			new_label = gen_labels.cpu().detach()
		else:
			new_data = torch.cat((new_data, synthesized.cpu().detach()), 0)
			new_label = torch.cat((new_label, gen_labels.cpu().detach()), 0)

	new_data = torch.clamp(new_data, min=-1., max=1.)

	C = Classifier().cuda()
	C.train()
	opt_C = SGD(C.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()

	gen_set = TensorDataset(new_data.cuda(), new_label.cuda())
	gen_loader = DataLoader(gen_set, batch_size=args.batch, shuffle=True)

	for epoch in range(args.class_epoch):
		train_acc = 0.0
		train_loss = 0.0
		for i, (data, label) in enumerate(gen_loader):
			pred = C(data)
			loss = criterion(pred, label)

			opt_C.zero_grad()
			loss.backward()
			opt_C.step()

			train_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == label.cpu().numpy())
			train_loss += loss.item()
	    
		print(f'acc: {train_acc/gen_set.__len__():.3f}  loss: {train_loss/gen_set.__len__():.4f}')

	test_set = CIFAR10(root='CIFAR10', download=False, train=False, transform=transform)
	test_loader = DataLoader(test_set, batch_size=args.test_batch, shuffle=False)

	test_acc = 0.0
	C.eval()
	for i, (data, label) in enumerate(test_loader):
		data = Variable(data.type(FloatTensor))
		label = Variable(label.type(LongTensor))
		pred = C(data)
		test_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == label.cpu().numpy())

	print(f'the final result of test accuracy = {test_acc/test_set.__len__():.3f}')
	with open(args.file, 'w') as f:
		f.write(f'the final result of test accuracy = {test_acc/test_set.__len__():.3f}')
