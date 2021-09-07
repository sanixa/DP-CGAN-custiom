import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

class Unflatten_7(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1, 7, 7)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        '''
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
        '''
        self.unflatten_7 = Unflatten_7()
        self.emb = nn.Embedding(10, 50)
        self.linear_1 = nn.Linear(50, 49)

        self.linear_2 = nn.Linear(100, 7*7*128)

        self.conv_tran_1 = nn.ConvTranspose2d(129, 128, 4, 2, 1)
        self.conv_tran_2 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.conv = nn.Conv2d(128, 1, 7, 1, 3)


    def forward(self, z, labels):
        '''
        labels, linear = self.label_emb(labels), self.linear(z)
        data = torch.cat((linear, labels), axis=1)
        return self.model(data)
        '''

        labels = self.emb(labels)
        labels = self.linear_1(labels)
        labels = self.unflatten_7(labels)

        linear = self.linear_2(z)
        linear = nn.LeakyReLU()(linear)
        linear = self.unflatten_7(linear)

        data = torch.cat((linear, labels), axis=1)
        data = self.conv_tran_1(data)
        data = nn.LeakyReLU()(data)
        data = self.conv_tran_2(data)
        data = nn.LeakyReLU()(data)
        data = self.conv(data)
        data = nn.Tanh()(data)

        return data

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--classes', type=int, default=10)
	parser.add_argument('--gan_epoch', type=int, default=1000)
	parser.add_argument('--g_dim', type=int, default=100)
	parser.add_argument('--model-path', type=str, default='checkpoint/DPCGAN_MNIST.bin')
	parser.add_argument('--save-dir', type=str)
	args = parser.parse_args()

	#eps = args.model_path.split('_')[-1].split('.')[0].split('s')[-1]
	#print(f"eps = {eps}")
	#dir_path = f'data/CGAN_eps{eps}'
	if not os.path.isfile(args.model_path):
		sys.exit("model does not exist")

	print('loading model...')
	netG = Generator().cuda()
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


