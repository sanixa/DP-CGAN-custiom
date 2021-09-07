import os

import numpy as np
import argparse
import glob
import cv2
import time
import sys
import scipy
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data_utils



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

def print_elapsed_time():
    end_time = int(time.time())
    d = divmod(end_time-start_time,86400)  # days
    h = divmod(d[1],3600)  # hours
    m = divmod(h[1],60)  # minutes
    s = m[1]  # seconds

    print('Elapsed Time: %d days, %d hours, %d minutes, %d seconds' % (d[0],h[0],m[0],s))

def calculate_results_matrices(distances_real_vs_sample,distances_real_vs_train, d_min=0.1):

    results_sample = np.zeros((len(distances_real_vs_sample),4))
    for i in range(len(results_sample)):
        # indicate that dataset is a sample
        results_sample[i][0] = 0
        
        integral_approx = 0
        integral_approx_log = 0
        integral_approx_eps = 0
        for eps in distances_real_vs_sample[i]:
            if eps < d_min:
                integral_approx = integral_approx + d_min/eps
                integral_approx_log = integral_approx_log + (-np.log(eps/d_min))
                integral_approx_eps = integral_approx_eps + 1

        integral_approx = integral_approx/len(distances_real_vs_sample[0])
        integral_approx_log = integral_approx_log/len(distances_real_vs_sample[0])
        integral_approx_eps = integral_approx_eps/len(distances_real_vs_sample[0])

        results_sample[i][1] = integral_approx_log
        results_sample[i][2] = integral_approx_eps
        results_sample[i][3] = integral_approx

    results_train = np.zeros((len(distances_real_vs_train),4))
    for i in range(len(results_train)):
        # indicate that dataset is a training data set
        results_train[i][0] = 1
        
        integral_approx = 0
        integral_approx_log = 0
        integral_approx_eps = 0
        for eps in distances_real_vs_train[i]:
            if eps < d_min:
                integral_approx = integral_approx + d_min/eps
                integral_approx_log = integral_approx_log + (-np.log(eps/d_min))
                integral_approx_eps = integral_approx_eps + 1

        integral_approx = integral_approx/len(distances_real_vs_train[0])
        integral_approx_log = integral_approx_log/len(distances_real_vs_train[0])
        integral_approx_eps = integral_approx_eps/len(distances_real_vs_train[0])

        results_train[i][1] = integral_approx_log
        results_train[i][2] = integral_approx_eps
        results_train[i][3] = integral_approx
        
    return results_sample,results_train

def mc_attack_sample(results_sample, results_train):
    ###single MI
    results = np.concatenate((results_sample, results_train))
    np.random.shuffle(results)
    mc_attack_log = results[results[:,1].argsort()][:,0][-len(results_train):].mean()
    np.random.shuffle(results)
    mc_attack_eps = results[results[:,2].argsort()][:,0][-len(results_train):].mean()
    np.random.shuffle(results)
    mc_attack_frac = results[results[:,3].argsort()][:,0][-len(results_train):].mean()
    ###set MI
    successfull_set_attack_1 = results_train[:,1].sum() > results_sample[:,1].sum()
    successfull_set_attack_2 = results_train[:,2].sum() > results_sample[:,2].sum()
    successfull_set_attack_3 = results_train[:,3].sum() > results_sample[:,3].sum()

    return mc_attack_log, mc_attack_eps, mc_attack_frac, successfull_set_attack_1, successfull_set_attack_2, successfull_set_attack_3

def mc_attack(results_sample, results_train):

    mc_attack_log, mc_attack_eps, mc_attack_frac, successfull_set_attack_1, successfull_set_attack_2, successfull_set_attack_3 = mc_attack_sample(results_sample, results_train)

    print('50_perc_mc_attack_log: %.3f'%(mc_attack_log))
    print('50_perc_mc_attack_eps: %.3f'%(mc_attack_eps))
    print('50_perc_mc_attack_frac: %.3f'%(mc_attack_frac))
    print('successfull_set_attack_1: %.3f'%(successfull_set_attack_1))
    print('successfull_set_attack_2: %.3f'%(successfull_set_attack_2))
    print('successfull_set_attack_3: %.3f'%(successfull_set_attack_3))

    iterations = 1000
    results_attacks = np.zeros((iterations, 3))

    for i in range(len(results_attacks)):
        np.random.shuffle(results_train)
        res = mc_attack_sample(results_sample, results_train[0:10])
        results_attacks[i][0] = res[0]
        results_attacks[i][1] = res[1]
        results_attacks[i][2] = res[2]

    print('11_perc_mc_attack_log: %.3f'%(results_attacks[:,0].mean()))
    print('11_perc_mc_attack_eps: %.3f'%(results_attacks[:,1].mean()))
    print('11_perc_mc_attack_frac: %.3f'%(results_attacks[:,2].mean()))

    return mc_attack_log, mc_attack_eps, mc_attack_frac, results_attacks[:,0].mean(), results_attacks[:,1].mean(), results_attacks[:,2].mean(), successfull_set_attack_1, successfull_set_attack_2, successfull_set_attack_3

'''
def calc_hist(image):
    vMin = np.amin(image)
    vMax = np.amax(image)

    image = (image-vMin)/(vMax-vMin)*255
    hist = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16],[0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist,hist).flatten()
    return hist

def calc_batch_hist(images):
    features = np.zeros((len(images),4096))
    
    for i in range(len(images)):
        features[i,:] = calc_hist(images[i])
        
    return features
'''
## to grayscale and both dataset image szie is 28*28
def calc_hist(image):
    vMin = np.amin(image)
    vMax = np.amax(image)

    image = (image-vMin)/(vMax-vMin)*255
    hist = cv2.calcHist([image], [0], None, [16],[0, 256])
    hist = cv2.normalize(hist,hist).flatten()

    return hist

def calc_batch_hist(images):
    features = np.zeros((len(images),16))
    
    for i in range(len(images)):
        features[i,:] = calc_hist(images[i])
        
    return features

def color_hist_attack(args, experiment_results, netG, trX, trY, vaX, vaY, teX, teY, mc_no_batches, mc_sample_size, exp_no, percentiles):
    vaX = vaX.permute(0, 2, 3, 1).cpu().detach().numpy()
    trX = trX.permute(0, 2, 3, 1).cpu().detach().numpy()

    feature_matrix_vaX = calc_batch_hist(vaX)
    feature_matrix_trX = calc_batch_hist(trX)

    distances_trX = np.zeros((len(feature_matrix_trX), mc_no_batches*mc_sample_size))
    distances_vaX = np.zeros((len(feature_matrix_vaX), mc_no_batches*mc_sample_size))

    for i in range(mc_no_batches):

        print('Working on %d/%d'%(i, mc_no_batches))

        ### generate img
        ###euclidean_generated_samples = gan.sample()
        use_cuda = torch.cuda.is_available()
        devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(1)]
        device0 = devices[0]
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        g_dim = 100

        noise = Variable(FloatTensor(np.random.normal(0, 1, (1000, g_dim))).to(device0))
        label = Variable(LongTensor(np.tile(np.arange(10), 100)).to(device0))
        image = Variable(netG(noise, label))
        generated_samples = image.cpu().detach().numpy()

        feature_matrix_generated = calc_batch_hist(generated_samples)

        distances_trX_partial = scipy.spatial.distance.cdist(feature_matrix_trX, feature_matrix_generated, 'euclidean')
        distances_vaX_partial = scipy.spatial.distance.cdist(feature_matrix_vaX, feature_matrix_generated, 'euclidean')

        # optimized, better than concatenate
        distances_trX[:,i*mc_sample_size:(i+1)*mc_sample_size] = distances_trX_partial
        distances_vaX[:,i*mc_sample_size:(i+1)*mc_sample_size] = distances_vaX_partial

        print_elapsed_time()

    for percentile in percentiles:
        print_elapsed_time()
        print('Calculating Results Matrices for '+str(percentile)+' Percentile...')
        d_min = np.percentile(np.concatenate((distances_trX,distances_vaX)),percentile)
        results_sample,results_train = calculate_results_matrices(distances_vaX, distances_trX,d_min)
        mc_attack_results = mc_attack(results_sample, results_train)
        
        # save data
        new_row = np.zeros(7)
        new_row[0] = exp_no
        new_row[1] = mc_attack_results[0]
        new_row[2] = mc_attack_results[1]
        new_row[3] = mc_attack_results[2]
        new_row[4] = mc_attack_results[6]
        new_row[5] = mc_attack_results[7]
        new_row[6] = mc_attack_results[8]
        experiment_results.append(new_row)

        exp_dir = os.path.join(args.exp, args.dataset)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        np.savetxt(os.path.join(exp_dir, args.name + '.csv'), np.array(experiment_results), fmt='%1.3f', delimiter=',')

    print('Calculating Results Matrices for flexible d_min...')
    distances = np.concatenate((distances_trX,distances_vaX))
    d_min = np.median([distances[i].min() for i in range(len(distances))])
    results_sample,results_train = calculate_results_matrices(distances_vaX, distances_trX,d_min)
    mc_attack_results = mc_attack(results_sample, results_train)

    # save data
    new_row = np.zeros(7)
    new_row[0] = exp_no
    new_row[1] = mc_attack_results[0]
    new_row[2] = mc_attack_results[1]
    new_row[3] = mc_attack_results[2]
    new_row[4] = mc_attack_results[6]
    new_row[5] = mc_attack_results[7]
    new_row[6] = mc_attack_results[8]
    experiment_results.append(new_row)

    exp_dir = os.path.join(args.exp, args.dataset)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    np.savetxt(os.path.join(exp_dir, args.name + '.csv'), np.array(experiment_results), fmt='%1.3f', delimiter=',')

    return experiment_results

def euclidean_PCA_mc_attack(args, experiment_results, n_components_pca, netG, trX, trY, vaX, vaY, teX, teY, exp_no, mc_euclidean_no_batches, mc_sample_size, percentiles):
    pca = PCA(n_components=n_components_pca)
    vaX = vaX.permute(0, 2, 3, 1)
    trX = trX.permute(0, 2, 3, 1)
    teX = teX.permute(0, 2, 3, 1)

    pca.fit_transform(teX.reshape((len(teX), 784)))

    euclidean_trX = np.reshape(trX, (len(trX), 784))
    euclidean_trX = pca.transform(euclidean_trX)

    euclidean_vaX = np.reshape(vaX, (len(vaX), 784))
    euclidean_vaX = pca.transform(euclidean_vaX)

    distances_trX = np.zeros((len(euclidean_trX), mc_euclidean_no_batches*mc_sample_size))
    distances_vaX = np.zeros((len(euclidean_vaX), mc_euclidean_no_batches*mc_sample_size))

    for i in range(mc_euclidean_no_batches):

        print('Working on %d/%d'%(i, mc_euclidean_no_batches))
        ### generate img
        ###euclidean_generated_samples = gan.sample()
        use_cuda = torch.cuda.is_available()
        devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(1)]
        device0 = devices[0]
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        g_dim = 100

        noise = Variable(FloatTensor(np.random.normal(0, 1, (1000, g_dim))).to(device0))
        label = Variable(LongTensor(np.tile(np.arange(10), 100)).to(device0))
        image = Variable(netG(noise, label))
        euclidean_generated_samples = image.cpu().detach().numpy()

        euclidean_generated_samples = np.reshape(euclidean_generated_samples, (len(euclidean_generated_samples),784))
        euclidean_generated_samples = pca.transform(euclidean_generated_samples)

        distances_trX_partial = scipy.spatial.distance.cdist(euclidean_trX, euclidean_generated_samples, 'euclidean')
        distances_vaX_partial = scipy.spatial.distance.cdist(euclidean_vaX, euclidean_generated_samples, 'euclidean')

        # optimized, better than concatenate
        distances_trX[:,i*mc_sample_size:(i+1)*mc_sample_size] = distances_trX_partial
        distances_vaX[:,i*mc_sample_size:(i+1)*mc_sample_size] = distances_vaX_partial
        
        print_elapsed_time()

    for percentile in percentiles:
        print_elapsed_time()
        print('Calculating Results Matrices for '+str(percentile)+' Percentile...')

        d_min = np.percentile(np.concatenate((distances_trX,distances_vaX)),percentile)
        results_sample,results_train = calculate_results_matrices(distances_vaX, distances_trX,d_min)
        
        mc_attack_results = mc_attack(results_sample, results_train)

        new_row = np.zeros(7)
        new_row[0] = exp_no
        new_row[1] = mc_attack_results[0]
        new_row[2] = mc_attack_results[1]
        new_row[3] = mc_attack_results[2]
        new_row[4] = mc_attack_results[6]
        new_row[5] = mc_attack_results[7]
        new_row[6] = mc_attack_results[8]
        experiment_results.append(new_row)

        exp_dir = os.path.join(args.exp, args.dataset)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        np.savetxt(os.path.join(exp_dir, args.name + '.csv'), np.array(experiment_results), fmt='%1.3f', delimiter=',')
    
    print('Calculating Results Matrices for flexible d_min...')
    distances = np.concatenate((distances_trX,distances_vaX))
    d_min = np.median([distances[i].min() for i in range(len(distances))])
    results_sample,results_train = calculate_results_matrices(distances_vaX, distances_trX,d_min)

    mc_attack_results = mc_attack(results_sample, results_train)

    new_row = np.zeros(7)
    new_row[0] = exp_no
    new_row[1] = mc_attack_results[0]
    new_row[2] = mc_attack_results[1]
    new_row[3] = mc_attack_results[2]
    new_row[4] = mc_attack_results[6]
    new_row[5] = mc_attack_results[7]
    new_row[6] = mc_attack_results[8]
    experiment_results.append(new_row)

    exp_dir = os.path.join(args.exp, args.dataset)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    np.savetxt(os.path.join(exp_dir, args.name + '.csv'), np.array(experiment_results), fmt='%1.3f', delimiter=',')
    
    return experiment_results

def discriminate_for_wb(data_to_be_discriminated, training_indicator):
    disc_results = np.zeros((len(data_to_be_discriminated),2))
    
    disc_results[:,1] = training_indicator
    disc_results[:,0] = sess.run(D_real, feed_dict={inputs:data_to_be_discriminated}).reshape((100,))
    
    return disc_results

def wb_attack_sample(disc_results_train, disc_results_validate):
    results = np.concatenate((disc_results_train,disc_results_validate))
    np.random.shuffle(results)
    results = results[results[:,0].argsort()]

    return results[-len(disc_results_train):,1].mean()

def wb_attack(trX_inds, vaX_inds, exp_no):

    disc_results_train = discriminate_for_wb(trX[trX_inds],1)
    disc_results_validate = discriminate_for_wb(vaX[vaX_inds],0)

    fifty_perc_wb_attack = wb_attack_sample(disc_results_train, disc_results_validate)

    #iterations = 1000
    #results_attacks = np.zeros((iterations, ))

    #for i in range(len(results_attacks)):
    #    np.random.shuffle(disc_results_train)
    #    results_attacks[i] = wb_attack_sample(disc_results_train[0:10], disc_results_validate)

    eleven_perc_wb_attack = 0#results_attacks.mean()

    print('50_perc_wb_attack: %.3f'%(fifty_perc_wb_attack))
    #print('11_perc_wb_attack: %.3f'%(eleven_perc_wb_attack))

    # white box
    new_row = np.zeros(1, dtype = dt)[0]
    new_row['instance_no'] = instance_no
    new_row['exp_no'] = exp_no
    new_row['method'] = 1 # white box
    new_row['percentage_of_data'] = 0.1
    new_row['50_perc_white_box'] = fifty_perc_wb_attack
    new_row['11_perc_white_box'] = eleven_perc_wb_attack
    experiment_results.append(new_row)
    np.savetxt(experiment+'.csv', np.array(experiment_results, dtype = dt))
    
    #return fifty_perc_wb_attack
'''
start_time = int(time.time())

for exp_no in range(exp_nos):

    trX_inds = np.arange(len(trX))
    np.random.shuffle(trX_inds)
    trX_inds = trX_inds[0:100]

    vaX_inds = np.arange(len(vaX))
    np.random.shuffle(vaX_inds)
    vaX_inds = vaX_inds[0:100]

    # euclidean PCA
    #euclidean_PCA_mc_attack(200, trX_inds, vaX_inds, exp_no, 100, 10000, [1,0.1,0.01,0.001])
    #print(experiment+': Finished PCA Monte Carlo 200 in experiment %d of %d'%(exp_no+1, exp_nos))

    euclidean_PCA_mc_attack(120, trX_inds, vaX_inds, exp_no, 10, 10000, [])
    print(experiment+': Finished PCA Monte Carlo 120 in experiment %d of %d'%(exp_no+1, exp_nos))

    #euclidean_PCA_mc_attack(40, trX_inds, vaX_inds, exp_no, 100, 10000, [1,0.1,0.01,0.001])
    #print(experiment+': Finished PCA Monte Carlo 40 in experiment %d of %d'%(exp_no+1, exp_nos))

    # color_hist_attack
    # 10000 cannot be changed easily!
    # color_hist_attack(300, 10000, trX_inds, vaX_inds, exp_no, [1, 0.1, 0.01, 0.001])
    # print(experiment+': Finished Color Hist in experiment %d of %d'%(exp_no+1, exp_nos))

    # white box
    wb_attack(trX_inds, vaX_inds, exp_no)
    print(experiment+': Finished White Box in experiment %d of %d'%(exp_no+1, exp_nos))
    
    print_elapsed_time()
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help='exp name')
    parser.add_argument('--exp', type=str, default=None, help='exp output floder name')
    parser.add_argument('--n', type=int, default=100, help='number of exp iteration')
    parser.add_argument('--dataset', type=str, default='cifar_10', help='dataset name')
    parser.add_argument('--load_dir', type=str, default=None, help='load generator path')

    args = parser.parse_args()

    ### Data loaders
    if args.dataset == 'mnist':
        transform_train = transforms.Compose([
        transforms.CenterCrop((28, 28)),
        transforms.ToTensor(),
        #transforms.Grayscale(),
        ])
    elif args.dataset == 'cifar_10':
        transform_train = transforms.Compose([
        transforms.CenterCrop((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        ])

    if args.dataset == 'mnist':
        IMG_DIM = 784
        NUM_CLASSES = 10
        trainset = datasets.MNIST(root=os.path.join('./', 'MNIST'), train=True, download=True,
                              transform=transform_train)
        testset = datasets.MNIST(root=os.path.join('./', 'MNIST'), train=False,
                              transform=transform_train)

    elif args.dataset == 'cifar_10':
        IMG_DIM = 784
        NUM_CLASSES = 10
        trainset = datasets.CIFAR10(root=os.path.join('./', 'CIFAR10'), train=True, download=True,
                              transform=transform_train)
        testset = datasets.CIFAR10(root=os.path.join('./', 'CIFAR10'), train=False,
                              transform=transform_train)
    else:
        raise NotImplementedError

    ###load Generator
    print('loading model...')
    use_cuda = torch.cuda.is_available()
    devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(1)]
    device0 = devices[0]
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    netG = Generator().to(device0)
    netG.load_state_dict(torch.load(args.load_dir))


    ### follow representation from mc_attack
    ########################################################################
    indices = torch.arange(44000)
    va = data_utils.Subset(trainset, indices)

    indices = torch.arange(44000,50000)
    te = data_utils.Subset(trainset, indices)

    tr = testset

    start_time = int(time.time())
    experiment_results = []

    for exp_no in range(args.n):

        trloader = DataLoader(tr, batch_size=100, shuffle=True)
        valoader = DataLoader(va, batch_size=100, shuffle=True)
        teloader = DataLoader(te, batch_size=6000, shuffle=True)

        trX, trY = next(iter(trloader))
        vaX, vaY = next(iter(valoader))
        teX, teY = next(iter(teloader))

        experiment_results = euclidean_PCA_mc_attack(args, experiment_results, 120, netG, trX, trY, vaX, vaY, teX, teY, exp_no, 300, 1000, [])
        print(args.name+': Finished PCA Monte Carlo 120 in experiment %d of %d'%(exp_no+1, args.n))

        # color_hist_attack
        # 10000 cannot be changed easily!
        #experiment_results = color_hist_attack(args, experiment_results, netG, trX, trY, vaX, vaY, teX, teY, 300, 1000, exp_no, [])#[1, 0.1, 0.01, 0.001])
        #print(args.name+': Finished Color Hist in experiment %d of %d'%(exp_no+1, args.n))

        print_elapsed_time()