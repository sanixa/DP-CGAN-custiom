import time
import sys
import os
from absl import logging
import collections
import numpy as np
import cv2
from numba import cuda 

import torch
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, TensorDataset

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.analysis import privacy_ledger
from tensorflow_privacy.privacy.dp_query import gaussian_query

def _random_choice(inputs, n_samples):
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)

    ind = tf.random.categorical(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

    return tf.gather(inputs, ind, name="random_choice")


def make_optimizer_class(cls):
    parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__
    child_code = cls.compute_gradients.__code__
    GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name
    if child_code is not parent_code:
        logging.warning(
            'WARNING: Calling make_optimizer_class() on class %s that overrides '
            'method compute_gradients(). Check to ensure that '
            'make_optimizer_class() does not interfere with overridden version.',
            cls.__name__)

    class DPOptimizerClass(cls):
        _GlobalState = collections.namedtuple('_GlobalState', ['l2_norm_clip', 'stddev'])
    
        def __init__(self, dp_sum_query, num_microbatches=None, unroll_microbatches=False, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)
            self._dp_sum_query = dp_sum_query
            self._num_microbatches = num_microbatches
            self._global_state = self._dp_sum_query.initial_global_state()
            self._unroll_microbatches = unroll_microbatches

        def compute_gradients(self,
                                loss,
                                var_list,
                                gate_gradients=GATE_OP,
                                aggregation_method=None,
                                colocate_gradients_with_ops=False,
                                grad_loss=None,
                                gradient_tape=None,
                                curr_noise_mult=0,
                                curr_norm_clip=1):

            self._dp_sum_query = gaussian_query.GaussianSumQuery(curr_norm_clip, curr_norm_clip*curr_noise_mult)
            self._global_state = self._dp_sum_query.make_global_state(curr_norm_clip, curr_norm_clip*curr_noise_mult)
      
            if not gradient_tape:
                raise ValueError('When in Eager mode, a tape needs to be passed.')

            vector_loss = loss()
            if self._num_microbatches is None:
                self._num_microbatches = tf.shape(input=vector_loss)[0]
            sample_state = self._dp_sum_query.initial_sample_state(var_list)
            microbatches_losses = tf.reshape(vector_loss, [self._num_microbatches, -1])
            sample_params = (self._dp_sum_query.derive_sample_params(self._global_state))

            def process_microbatch(i, sample_state):
                microbatch_loss = tf.reduce_mean(input_tensor=tf.gather(microbatches_losses, [i]))
                grads = gradient_tape.gradient(microbatch_loss, var_list)
                sample_state = self._dp_sum_query.accumulate_record(sample_params, sample_state, grads)
                return sample_state
    
            for idx in range(self._num_microbatches):
                sample_state = process_microbatch(idx, sample_state)

            if curr_noise_mult > 0:
                grad_sums, self._global_state = (self._dp_sum_query.get_noised_result(sample_state, self._global_state))
            else:
                grad_sums = sample_state

            def normalize(v):
                return v / tf.cast(self._num_microbatches, tf.float32)

            final_grads = tf.nest.map_structure(normalize, grad_sums)
            grads_and_vars = final_grads#list(zip(final_grads, var_list))
    
            return grads_and_vars

    return DPOptimizerClass

def make_gaussian_optimizer_class(cls):

    class DPGaussianOptimizerClass(make_optimizer_class(cls)):
        def __init__(self,
                        l2_norm_clip,
                        noise_multiplier,
                        num_microbatches=None,
                        ledger=None,
                        unroll_microbatches=False,
                        *args,  # pylint: disable=keyword-arg-before-vararg
                        **kwargs):
            dp_sum_query = gaussian_query.GaussianSumQuery(l2_norm_clip, l2_norm_clip * noise_multiplier)

            if ledger:
                dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query, ledger=ledger)

            super(DPGaussianOptimizerClass, self).__init__(dp_sum_query,
                                                            num_microbatches,
                                                            unroll_microbatches,
                                                            *args,
                                                            **kwargs)

        @property
        def ledger(self):
            return self._dp_sum_query.ledger

    return DPGaussianOptimizerClass

GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
DPGradientDescentGaussianOptimizer_NEW = make_gaussian_optimizer_class(GradientDescentOptimizer)

checkpoint_dir = './checkpoint'

def checkpoint_name(title):  
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt__" + str(title))
    return(checkpoint_prefix)


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
train_labels = train_labels.reshape((60000, 1))
COND_num_classes = 10 # Number of classes, set to 10 for MNIST dataset
train_labels_vec = np.zeros((len(train_labels), COND_num_classes), dtype='float32')
for i, label in enumerate(train_labels):
    train_labels_vec[i, int(train_labels[i])] = 1.0

Z_DIM = 100
def make_generator_model_FCC():
    in_label = layers.Input(shape=(COND_num_classes,))
    in_lat = layers.Input(shape=(Z_DIM,))

    merge = layers.concatenate([in_lat, in_label], axis=1)

    ge1 = layers.Dense(128, use_bias=True)(merge)
    ge1 = layers.ReLU()(ge1)
    ge2 = layers.Dense(784, use_bias=True, activation="tanh")(ge1)
    out_layer = layers.Reshape((28, 28, 1))(ge2)

    model = models.Model([in_lat, in_label], out_layer)
    return model

def make_discriminator_model_FCC():
    in_label = layers.Input(shape=(COND_num_classes,))
    in_image = layers.Input(shape=(28, 28, 1))
    in_image_b = layers.Flatten()(in_image)

    merge = layers.concatenate([in_image_b, in_label], axis=1)

    ge1 = layers.Dense(128, use_bias=True)(merge)
    ge1 = layers.ReLU()(ge1)
    out_layer = layers.Dense(1, use_bias=True)(ge1)

    model = models.Model([in_image, in_label], out_layer)
    return model

cross_entropy_DISC = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
cross_entropy_GEN = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Notice the use of `tf.function`: This annotation causes the function to be "compiled".
@tf.function
def train_step_DISC(images, labels, noise, labels_to_gen):    
    with tf.GradientTape(persistent=True) as disc_tape_real:
        # This dummy call is needed to obtain the var list.
        dummy = discriminator([images, labels], training=True)
        var_list = discriminator.trainable_variables
        
        # In Eager mode, the optimizer takes a function that returns the loss.
        def loss_fn_real():
            real_output = discriminator([images, labels], training=True)
            disc_real_loss = cross_entropy_DISC(tf.ones_like(real_output), real_output)
            return disc_real_loss
        
        grads_and_vars_real = discriminator_optimizer.compute_gradients(loss_fn_real, 
                                                                        var_list, 
                                                                        gradient_tape=disc_tape_real, 
                                                                        curr_noise_mult=NOISE_MULT[config],
                                                                        curr_norm_clip=NORM_CLIP)
        
        # In Eager mode, the optimizer takes a function that returns the loss.
        def loss_fn_fake():
            generated_images = generator([noise, labels_to_gen], training=True)
            fake_output = discriminator([generated_images, labels_to_gen], training=True)
            disc_fake_loss = cross_entropy_DISC(tf.zeros_like(fake_output), fake_output)
            return disc_fake_loss
        
        grads_and_vars_fake = discriminator_optimizer.compute_gradients(loss_fn_fake,
                                                                        var_list, 
                                                                        gradient_tape=disc_tape_real,
                                                                        curr_noise_mult=0,
                                                                        curr_norm_clip=NORM_CLIP)
        disc_loss_r = loss_fn_real()
        disc_loss_f = loss_fn_fake()
        
        s_grads_and_vars = [(grads_and_vars_real[idx] + grads_and_vars_fake[idx])
                            for idx in range(len(grads_and_vars_real))]
        sanitized_grads_and_vars = list(zip(s_grads_and_vars, var_list))
        
        discriminator_optimizer.apply_gradients(sanitized_grads_and_vars)
        
    return(disc_loss_r, disc_loss_f)

# Notice the use of `tf.function`: This annotation causes the function to be "compiled".
@tf.function
def train_step_GEN(labels, noise):
    with tf.GradientTape() as gen_tape:
        generated_images = generator([noise, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)
        # if the generator is performing well, the discriminator will classify the fake images as real (or 1)
        gen_loss = cross_entropy_GEN(tf.ones_like(fake_output), fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    return(gen_loss)

def train(dataset, title, verbose):
    for epoch in range(EPOCHS):
        start = time.time()

        i_gen = 0
        for image_batch, label_batch in dataset:          
            noise = tf.random.normal([BATCH_SIZE, Z_DIM])
            labels_to_gen = _random_choice(labels_gen_vec, BATCH_SIZE)
    
            d_loss_r, d_loss_f = train_step_DISC(image_batch, label_batch, noise, labels_to_gen)

            if (i_gen + 1) % N_DISC == 0:
                g_loss_f = train_step_GEN(labels_to_gen, noise)

            i_gen = i_gen + 1
            
        print (f'Time for epoch {epoch + 1} is {time.time()-start: .2f} sec')

        # Save the model
        if epoch == EPOCHS - 1:
            checkpoint.save(file_prefix = checkpoint_name(title))

BUFFER_SIZE = 60000 # Total size of training data
BATCH_SIZE = 600
NR_MICROBATCHES = 600 # Each batch of data is split in smaller units called microbatches.

NORM_CLIP = 1.1 # Does NOT affect EPSILON, but increases NOISE on gradients
NOISE_MULT = [7.75, 1.81, 1.12, 0.585, 0.4865]
#NOISE_MULT = 0.4865 #7.75 for eps=1, 1.81 for eps=5, 1.12 for eps=10, 0.585 for eps=50, 0.4865 for eps=100

DP_DELTA = 1e-5 # Needs to be smaller than 1/BUFFER_SIZE
EPOCHS = 249

N_DISC = 1 # Number of times we train DISC before training GEN once

config = int(sys.argv[1])
configuration = [1, 5, 10, 50, 100]

# Learning Rate for DISCRIMINATOR
LR_DISC = tf.compat.v1.train.polynomial_decay(learning_rate=0.150,
                                              global_step=tf.compat.v1.train.get_or_create_global_step(),
                                              decay_steps=10000,
                                              end_learning_rate=0.052,
                                              power=1)

if BATCH_SIZE % NR_MICROBATCHES != 0:
    raise ValueError('Batch size should be an integer multiple of the number of microbatches')

# Obtain DP_EPSILON
compute_dp_sgd_privacy.compute_dp_sgd_privacy(n = BUFFER_SIZE, 
                                              batch_size = BATCH_SIZE, 
                                              noise_multiplier = NOISE_MULT[config], 
                                              epochs = EPOCHS, 
                                              delta = DP_DELTA)

generator_optimizer = tf.keras.optimizers.Adam()

discriminator_optimizer = DPGradientDescentGaussianOptimizer_NEW(
   learning_rate = LR_DISC,
   l2_norm_clip = NORM_CLIP,
   noise_multiplier = NOISE_MULT[config],
   num_microbatches = NR_MICROBATCHES)

generator = make_generator_model_FCC()
discriminator = make_discriminator_model_FCC()

# Create checkpoint structure
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

tf.random.set_seed(1)

# Batch and random shuffle training data
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels_vec)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Fix some seeds to help visualize progress
seed = tf.random.normal([10, Z_DIM])
seed_labels = tf.Variable(np.diag(np.full(10,1)).reshape((10,10)), dtype='float32')

# To be used for sampling random labels to pass to generator
labels_gen_vec = np.zeros((10, COND_num_classes), dtype='float32')
for i in [0,1,2,3,4,5,6,7,8,9]:
    labels_gen_vec[i, int(i)] = 1.0

training_title = 'eps100'
train(train_dataset, training_title, False)


for i in range(20000):
    noise = tf.Variable(tf.random.normal([1, Z_DIM]))
    noise_label = np.random.randint(10)
    noise_onehot_label = tf.Variable(np.eye(10)[noise_label].reshape((1, 10)), dtype='float32')
    if (i == 0):
        generated_labels = [noise_label]
        generated_images = generator([noise, noise_onehot_label], training=False)
    else:
        generated_labels.append(noise_label)
        generated_images = tf.concat([generated_images, generator([noise, noise_onehot_label], training=False)], 0)

generated_images = generated_images.numpy()
generated_images = torch.from_numpy(generated_images)
generated_labels = torch.Tensor(generated_labels).type(torch.LongTensor)

torch.save(generated_images, f'./fake_data/eps{configuration[config]}_image.pt')
torch.save(generated_labels, f'./fake_data/eps{configuration[config]}_label.pt')

device = cuda.get_current_device()
device.reset()

class Flatten(nn.Module):
    def forward(self, data):
        return data.view(data.size()[0], -1)

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

network = Classifier().cuda()
network.train()
opt = optim.SGD(network.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
epochs = 15
batch_size = 64

gen_set = TensorDataset(generated_images, generated_labels)
gen_loader = DataLoader(gen_set, batch_size = batch_size, shuffle = True)

for epoch in range(epochs):
    train_acc = 0.0
    train_loss = 0.0
    for i, (data, label) in enumerate(gen_loader):
        pred = network(data.cuda())
        loss = criterion(pred, label.cuda())

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == label.numpy())
        train_loss += loss.item()
    
    print(f'acc: {train_acc/gen_set.__len__():.3f}  loss: {train_loss/gen_set.__len__():.4f}')

transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))
])

test_set = MNIST(root='./MNIST', download=True, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

test_acc = 0.0
network.eval()
for i, (data, label) in enumerate(test_loader):
    data = torch.FloatTensor(data)
    pred = network(data.cuda())
    test_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == label.numpy())

print(f'the final result of test accuracy = {test_acc/test_set.__len__():.3f}')
with open(f'./acc_result/eps{configuration[config]}.txt', 'w') as f:
    f.write(f'the final result of test accuracy = {test_acc/test_set.__len__():.3f}')



