import torch 
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

from .kernel_mmd import mmd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def save_model(model, fn):
    torch.save(model.state_dict(), fn)

def load_model(model, fn):
    model.load_state_dict(torch.load(fn))
    return model


def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))

    return 

def sample_noise(batch_size, dim):

    noise = 2 * torch.rand((batch_size,dim)) - 1
    return noise

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=64, H=16, W=16):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count

def gan_generate(dtype, G, noise_size=96, batch_size=64):
    g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
    fake_images = G(g_fake_seed).type(dtype)
    return fake_images.data.cpu().numpy()


def fake_img_collection(dtype, model, N_real, img_size=64):
    
    fake_imgs = np.zeros((N_real, img_size*img_size))
    fake_count, img_count = 0, 0
    while fake_count < N_real:
        batch_fake =  gan_generate(dtype, model)
        for i in range(batch_fake.shape[0]):
            if img_count < N_real:
                fake_imgs[img_count,:] = batch_fake[i,:]
                img_count += 1
        fake_count += batch_fake.shape[0] 
    return fake_imgs


def calculate_gap(dtype, generator, val, train):
    
    fake_imgs = fake_img_collection(dtype, generator, train.shape[0])
    fake_imgs[fake_imgs < 0.0] = 0.0
    val_metric = mmd(fake_imgs, val, 25.0)
    fake_imgs = fake_img_collection(dtype, generator, train.shape[0])
    fake_imgs[fake_imgs < 0.0] = 0.0
    train_metric = mmd(fake_imgs, train, 25.0)
    verification = mmd(train,val,25.0)
    print(val_metric, train_metric, verification)
    return np.abs(val_metric - train_metric)
            
    
    
    
def run_a_gan(loader_train, dtype, D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=100, batch_size=64, noise_size=96, num_epochs=50, sz=64, show=True, label_smoothing=False, instance_noise=False, validation=False, validation_set=None, train_set=None):
    """
    Train a GAN!
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    d_loss_list, g_loss_list = [], []
    iter_count = 0
    sigma = 1.0e-2
    lin_decay_rate = sigma / num_epochs
    
    if validation:
        gap_list = []
    
    for epoch in range(num_epochs):
        if validation and epoch > 0:
            gap_list.append(calculate_gap(dtype, G, validation_set, train_set))
            
        sigma_n = sigma - (lin_decay_rate * epoch)
        for x, _ in loader_train:
            x = x[:,0,:,:]
            N, H, W = list(x.size())
            x = x.view(N,1,H,W)
            # Add instance noise to input images x
            noise = torch.randn((N,1,H,W)) * sigma_n
            if instance_noise:
                x = x + noise
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.type(dtype)

            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed).detach()
            noise = torch.randn((N,1*H*W),dtype=torch.float, device=torch.device('cuda:0')) * sigma_n
            if instance_noise:
                fake_images = fake_images + noise
            
            logits_real = D(2* (real_data - 0.5)).type(dtype)
            logits_fake = D(fake_images.view(batch_size, 1, sz, sz))
            rn = np.random.uniform(0,1)
            if rn >= 0.90 and label_smoothing:
                d_total_error = discriminator_loss(logits_fake, logits_real, dtype)
            else:
                d_total_error = discriminator_loss(logits_real, logits_fake, dtype)
                
            d_total_error.backward()        
            D_solver.step()
           
            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)
            gen_logits_fake = D(fake_images.view(batch_size, 1, sz, sz))
            g_error = generator_loss(gen_logits_fake, dtype)
            g_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0 and show):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
                imgs_numpy = fake_images.data.cpu().numpy()
                show_images(imgs_numpy[0:32])
                plt.show()
                print()
            iter_count += 1
        g_loss_list.append(g_error.item())
        d_loss_list.append(d_total_error.item())
    if validation:
        return gap_list, D, G
    return d_loss_list, g_loss_list, fake_images.data.cpu().numpy(), D, G
