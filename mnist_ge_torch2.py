import sys
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
        self.cdist = nn.CosineSimilarity(dim = 1, eps= 1e-9)
        self.grads = []
        self.grad_dict = {}

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
    def __init__(self):
        super(Generator, self).__init__()
        self.cdist = nn.CosineSimilarity(dim = 1, eps= 1e-9)
        self.grads = []
        self.grad_dict = {}

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
        self.apply(weights_init)

    def forward(self, z, labels):
        labels, linear = self.label_emb(labels), self.linear(z)
        data = torch.cat((linear, labels), axis=1)
        return self.model(data)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        #input [1, 28, 28]
        self.model = nn.Sequential(
            Flatten(),

            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, input):
        return self.model(input)

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
		cv2.imwrite(f'./checkImg/epoch{epoch}_{i}.png', img)

log2 = [1000,5000,10000,20000]
log1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
log1_ = [False for i in range(len(log1))]

def collect(args, collect_iter, G, D, train_set, opt_G, opt_D, G_grad_dict_num, D_grad_dict_num):

    mini_loader, micro_loader = pyvacy.sampling.get_data_loaders(args.mini, args.micro, collect_iter)
    criterion = nn.BCELoss()

    bar = tqdm(enumerate(mini_loader(train_set)))
    for iteration, (x_mini, y_mini) in bar:
        opt_D.zero_grad()
        opt_G.zero_grad()
        for img, label in micro_loader(TensorDataset(x_mini, y_mini)):
            valid = Variable(FloatTensor(args.micro, 1).fill_(1.0), requires_grad=False)
            fake  = Variable(FloatTensor(args.micro, 1).fill_(0.0), requires_grad=False)

            r_img = Variable(img.type(FloatTensor))
            label = Variable(label.type(LongTensor))

            noise = Variable(FloatTensor(np.random.normal(0, 1, (args.micro, args.g_dim))))
            gen_label = Variable(LongTensor(np.random.randint(0, args.classes, args.micro)))

            gen_img = G(noise, gen_label)
            f_logit = D(gen_img, gen_label)
            loss_G = criterion(f_logit, valid)

            opt_G.zero_microbatch_grad()
            loss_G.backward()
            opt_G.microbatch_step()

            r_logit = D(r_img, label)
            r_loss_D = criterion(r_logit, valid)

            gen_img = G(noise, gen_label)
            f_logit = D(gen_img, gen_label)
            f_loss_D = criterion(f_logit, fake)
            loss_D = (r_loss_D + f_loss_D) / 2

            opt_D.zero_microbatch_grad()
            loss_D.backward()
            opt_D.microbatch_step()

        opt_G.step()
        opt_D.step()			
        bar.set_description(f"Epoch [{iteration+1}/{args.iter}] d_loss: {loss_D.item():.5f} g_loss: {loss_G.item():.5f}")

        if torch.rand(1) < 0.2:
            G.grad_dict[G_grad_dict_num] = [p.grad.clone() for p in G.parameters()]
            g_vec = torch.cat([g.view(-1) for g in G.grad_dict[G_grad_dict_num]])
            G.grads.append(g_vec)
            G_grad_dict_num += 1

            D.grad_dict[D_grad_dict_num] = [p.grad.clone() for p in D.parameters()]
            g_vec = torch.cat([g.view(-1) for g in D.grad_dict[D_grad_dict_num]])
            D.grads.append(g_vec)
            D_grad_dict_num += 1


    return G_grad_dict_num, D_grad_dict_num


try:
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--classes', type=int, default=10)
        parser.add_argument('--class_epoch', type=int, default=20)
        parser.add_argument('--g_dim', type=int, default=100)
        parser.add_argument('--test-batch', type=int, default=256)
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--file', type=str, default='./acc_result/acc_mnist')
        parser.add_argument('--clip', type=float, default=1.2)
        parser.add_argument('--micro', type=int, default=1)
        parser.add_argument('--mini', type=int, default=32)
        parser.add_argument('--iter', type=int, default=20000)
        parser.add_argument('--collect-iter', type=int, default=1000)
        parser.add_argument('--noise', type=float, default=0.11145)
        parser.add_argument('--delta', type=float, default=1e-5)
        args = parser.parse_args()

        transform = transforms.ToTensor()
        test_set = MNIST(root='MNIST', download=False, train=False, transform=transform)
        test_loader = DataLoader(test_set, batch_size=args.test_batch, shuffle=False)

        if (args.noise > 0):
            G = Generator().cuda()
            G.train()
            opt_G = pyvacy.optim.DPAdam(
                    l2_norm_clip = 1e3,
                    noise_multiplier = 0,
                    minibatch_size = args.mini,
                    microbatch_size = args.micro,
                    params = G.parameters(),
                    lr = args.lr,
                    betas = (0.5, 0.999),
                )
            
            D = Discriminator().cuda()
            D.train()
            opt_D = pyvacy.optim.DPAdam(
                    l2_norm_clip = args.clip,
                    noise_multiplier = args.noise,
                    minibatch_size = args.mini,
                    microbatch_size = args.micro,
                    params = D.parameters(),
                    lr = args.lr,
                    betas = (0.5, 0.999),
                )

            transform = transforms.ToTensor()
            train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
            mini_loader, micro_loader = pyvacy.sampling.get_data_loaders(args.mini, args.micro, args.iter)

            eps = pyvacy.analysis.epsilon(len(train_set), args.mini, args.noise, args.iter, args.delta)
            print(f'Achieves ({eps:.2f}, {args.delta})-DP')
            dire = "eps_"+str(int(eps))
            print(dire)
            fl1 = open("./checkpoint/"+dire+"/diff_iter/acc_different_iter.txt","w")
            fl2 = open("./checkpoint/"+dire+"/diff_acc/iter_acc.txt","w")
            fl3 = open("./checkpoint/"+dire+"/record.txt","w")            
            fl4 = open("./checkpoint/"+dire+"/gradient_encode_G.txt","w")
            fl5 = open("./checkpoint/"+dire+"/gradient_encode_D.txt","w")            
            args.file = f"{args.file}_eps{eps:.1f}.txt"

            G_grad_dict_num, D_grad_dict_num = 0, 0
            collect_iter = args.collect_iter
            G_grad_dict_num, D_grad_dict_num = collect(args, collect_iter, G, D, train_set, opt_G, opt_D, G_grad_dict_num, D_grad_dict_num)
            G.grads = torch.stack(G.grads)
            D.grads = torch.stack(D.grads)
            print("finish collecting")

            criterion = nn.BCELoss()

            bar = tqdm(enumerate(mini_loader(train_set)))
            for iteration, (x_mini, y_mini) in bar:
                opt_D.zero_grad()
                opt_G.zero_grad()
                for img, label in micro_loader(TensorDataset(x_mini, y_mini)):
                    valid = Variable(FloatTensor(args.micro, 1).fill_(1.0), requires_grad=False)
                    fake  = Variable(FloatTensor(args.micro, 1).fill_(0.0), requires_grad=False)

                    r_img = Variable(img.type(FloatTensor))
                    label = Variable(label.type(LongTensor))

                    noise = Variable(FloatTensor(np.random.normal(0, 1, (args.micro, args.g_dim))))
                    gen_label = Variable(LongTensor(np.random.randint(0, args.classes, args.micro)))

                    gen_img = G(noise, gen_label)
                    f_logit = D(gen_img, gen_label)
                    loss_G = criterion(f_logit, valid)

                    opt_G.zero_microbatch_grad()
                    loss_G.backward()
                    opt_G.microbatch_step()

                    r_logit = D(r_img, label)
                    r_loss_D = criterion(r_logit, valid)

                    gen_img = G(noise, gen_label)
                    f_logit = D(gen_img, gen_label)
                    f_loss_D = criterion(f_logit, fake)
                    loss_D = (r_loss_D + f_loss_D) / 2

                    opt_D.zero_microbatch_grad()
                    loss_D.backward()
                    opt_D.microbatch_step()

                    g_vec = torch.cat([p.grad.view(-1) for p in G.parameters()])
                    dist_mat = G.cdist(g_vec.unsqueeze(0), G.grads)
                    #import ipdb; ipdb.set_trace()
                    closest_idx = int(dist_mat.argmax().data.cpu().numpy())
                    if torch.rand(1) < 0.01:
                        fl4.write("{:.4f}\n".format(max(dist_mat.data.cpu().numpy())))
                        
                    for idx, p in enumerate(G.parameters()):
                        p.grad = G.grad_dict[closest_idx][idx]

                    g_vec = torch.cat([p.grad.view(-1) for p in D.parameters()])
                    dist_mat = D.cdist(g_vec.unsqueeze(0), D.grads)
                    #import ipdb; ipdb.set_trace()
                    closest_idx = int(dist_mat.argmax().data.cpu().numpy())
                    if torch.rand(1) < 0.01:
                        fl5.write("{:.4f}\n".format(max(dist_mat.data.cpu().numpy())))
                        
                    for idx, p in enumerate(D.parameters()):
                        p.grad = D.grad_dict[closest_idx][idx]

                opt_G.step()
                opt_D.step()			
                bar.set_description(f"Epoch [{iteration+1}/{args.iter}] d_loss: {loss_D.item():.5f} g_loss: {loss_G.item():.5f}")
                if((iteration+1) %100 ==0):
                    G.eval()
                    for i in tqdm(range(50)):
                        gen_labels = Variable(LongTensor(np.random.randint(0, args.classes, 1000)))
                        noise = Variable(FloatTensor(np.random.normal(0, 1, (1000, args.g_dim))))
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
                    opt_C = torch.optim.Adam(C.parameters(), lr=1e-3,weight_decay=1e-3,amsgrad = True)
                    criterion2 = nn.CrossEntropyLoss()

                    gen_set = TensorDataset(new_data.cuda(), new_label.cuda())
                    gen_loader = DataLoader(gen_set, batch_size=args.mini, shuffle=True)

                    for epoch in range(args.class_epoch):
                        train_acc = 0.0
                        train_loss = 0.0
                        for i, (data, label) in enumerate(gen_loader):
                            pred = C(data)
                            loss = criterion2(pred, label)

                            opt_C.zero_grad()
                            loss.backward()
                            opt_C.step()

                            train_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == label.cpu().numpy())
                            train_loss += loss.item()
                        
                        print(f'acc: {train_acc/gen_set.__len__():.3f}  loss: {train_loss/gen_set.__len__():.4f}')

                    test_acc = 0.0
                    C.eval()
                    for i, (data, label) in enumerate(test_loader):
                        data = Variable(data.type(FloatTensor))
                        label = Variable(label.type(LongTensor))
                        pred = C(data)
                        test_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == label.cpu().numpy())

                    print(f'iteration: {iteration+1} result of test accuracy = {test_acc/test_set.__len__():.3f}')
                    fl3.write(f'iteration: {iteration+1} result of test accuracy = {test_acc/test_set.__len__():.3f}\n')
                    if((iteration+1) in log2):
                        fl2.write(f"iteration:{(iteration+1):5d}\tacc:{test_acc/test_set.__len__()}\n")
                        torch.save(G.state_dict(), f"./checkpoint/"+dire+f"/DPCGAN_MNIST_eps{eps:.1f}_iteration{(iteration+1)}.ckpt")

                    for i in range(len(log1)):
                        if(test_acc/test_set.__len__() >log1[i] and log1_[i] ==False):
                            fl1.write(f"iteration:{iteration+1:5d}\tacc:{test_acc/test_set.__len__()}\n")
                            log1_[i] = True
                            aaa = test_acc/test_set.__len__()
                            torch.save(G.state_dict(), f"./checkpoint/"+dire+f"/DPCGAN_MNIST_eps{eps:.1f}_acc{aaa:.1f}.ckpt")
                    del C

            torch.save(G.state_dict(), f"checkpoint/DPCGAN_MNIST_eps{eps:.1f}.bin")

        else:
            dire = "eps_inf"
            print(dire)
            fl1 = open("./checkpoint/"+dire+"/diff_iter/acc_different_iter.txt","w")
            fl2 = open("./checkpoint/"+dire+"/diff_acc/iter_acc.txt","w")
            fl3 = open("./checkpoint/"+dire+"/record.txt","w")              
            G = Generator().cuda()
            G.train()
            opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

            D = Discriminator().cuda()
            D.train()
            opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

            args.file = f"{args.file}.txt"
            criterion = nn.BCELoss()

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
            train_loader = DataLoader(train_set, batch_size=args.mini, shuffle=True)
            iterations = 0
            gan_epoch = (args.iter * args.mini) // len(train_set)
            for epoch in range(gan_epoch):
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


                    bar.set_description(f"Epoch [{epoch+1}/{gan_epoch}] d_loss: {loss_D.item():.5f} g_loss: {loss_G.item():.5f}")
                    if((iterations+1) %100 ==0):
                        G.eval()
                        for j in tqdm(range(50)):
                            gen_labels = Variable(LongTensor(np.random.randint(0, args.classes, 1000)))
                            noise = Variable(FloatTensor(np.random.normal(0, 1, (1000, args.g_dim))))
                            synthesized = G(noise, gen_labels)

                            if (j == 0):
                                new_data = synthesized.cpu().detach()
                                new_label = gen_labels.cpu().detach()
                            else:
                                new_data = torch.cat((new_data, synthesized.cpu().detach()), 0)
                                new_label = torch.cat((new_label, gen_labels.cpu().detach()), 0)

                        new_data = torch.clamp(new_data, min=-1., max=1.)

                        C = Classifier().cuda()
                        C.train()
                        opt_C = torch.optim.SGD(C.parameters(), lr=1e-3)
                        criterion2 = nn.CrossEntropyLoss()

                        gen_set = TensorDataset(new_data.cuda(), new_label.cuda())
                        gen_loader = DataLoader(gen_set, batch_size=args.mini, shuffle=True)

                        for e in range(args.class_epoch):
                            train_acc = 0.0
                            train_loss = 0.0
                            for j, (data, label) in enumerate(gen_loader):
                                pred = C(data)
                                loss = criterion2(pred, label)

                                opt_C.zero_grad()
                                loss.backward()
                                opt_C.step()

                                train_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == label.cpu().numpy())
                                train_loss += loss.item()
                            
                            print(f'acc: {train_acc/gen_set.__len__():.3f}  loss: {train_loss/gen_set.__len__():.4f}')

                        test_acc = 0.0
                        C.eval()
                        for j, (data, label) in enumerate(test_loader):
                            data = Variable(data.type(FloatTensor))
                            label = Variable(label.type(LongTensor))
                            pred = C(data)
                            test_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == label.cpu().numpy())

                        print(f'iteration: {iterations+1} result of test accuracy = {test_acc/test_set.__len__():.3f}')
                        if(iterations+1 in log2):
                            fl2.write(f"iteration:{iterations+1:5d}\tacc:{test_acc/test_set.__len__()}\n")
                            torch.save(G.state_dict(), f"./checkpoint/"+dire+f"/DPCGAN_MNIST_eps_inf_iteration{iterations+1}.ckpt")

                        for j in range(len(log1)):
                            if(test_acc/test_set.__len__() >log1[j] and log1_[j] ==False):
                                fl1.write(f"iteration:{iterations+1:5d}\tacc:{test_acc/test_set.__len__()}\n")
                                log1_[j] = True
                                aaa = test_acc/test_set.__len__()
                                torch.save(G.state_dict(), f"./checkpoint/"+dire+f"/DPCGAN_MNIST_eps_inf_acc{aaa:.1f}.ckpt")
                    iterations +=1
                G.eval()
                sample(epoch)
                G.train()

            torch.save(G.state_dict(), f"checkpoint/DPCGAN_MNIST.bin")


        G.eval()
        for i in tqdm(range(50)):
            gen_labels = Variable(LongTensor(np.random.randint(0, args.classes, 1000)))
            noise = Variable(FloatTensor(np.random.normal(0, 1, (1000, args.g_dim))))
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
        opt_C = torch.optim.SGD(C.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        gen_set = TensorDataset(new_data.cuda(), new_label.cuda())
        gen_loader = DataLoader(gen_set, batch_size=args.mini, shuffle=True)

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
        fl1.close()
        fl2.close()
        fl3.close()
        fl4.close()
        fl5.close()
except:
    raise
    fl1.close()
    fl2.close()
    fl3.close()
    fl4.close()
    fl5.close()





