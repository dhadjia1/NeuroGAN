import torch 
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

from .network_utils import Flatten
from .network_utils import Unflatten

def discriminator(sz=64):
    """
    Build and return a PyTorch model implementing the architecture above.
    """
    model = nn.Sequential(Flatten(),nn.Linear(sz*sz, 256,bias=True) , nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Linear(256, 256, bias=True),
                          nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Linear(256, 1, bias=True)
    )
    return model


def generator(noise_dim=96,sz=64):
    """
    Build and return a PyTorch model implementing the architecture above.
    """
    model = nn.Sequential(nn.Linear(noise_dim, 1024,bias=True) , nn.ReLU(inplace=True), nn.Linear(1024, 1024, bias=True),
                          nn.ReLU(inplace=True), nn.Linear(1024, sz*sz, bias=True), nn.Tanh()
    )
    return model

def dc_discriminator(batch_size, sz=64):
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    return nn.Sequential(
        Unflatten(batch_size, 1, sz, sz),
        nn.Conv2d(1,32, kernel_size=5, stride=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32,64, kernel_size=5, stride=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(64,64, kernel_size=6, stride=1),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(1024, 1024),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Linear(1024, 1)     
    )

def dc_generator(batch_size, noise_dim=96, sz=64):
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described above.
    """
    return nn.Sequential(
       nn.Linear(noise_dim, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 8*8*batch_size),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(8*8*batch_size),
        Unflatten(batch_size, batch_size, 8, 8),
        nn.ConvTranspose2d(batch_size, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(8),
        nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),
        nn.Tanh(),
        Flatten()
    )


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def vanilla_discriminator_loss(logits_real, logits_fake, dtype):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
  
    
    loss = None
    N,_ = logits_real.size()
    real_loss = bce_loss(logits_real, torch.autograd.Variable(torch.ones(N)).type(dtype))
    N,_ = logits_fake.size()
    fake_loss = bce_loss(logits_fake, torch.autograd.Variable(torch.zeros(N)).type(dtype))
    return fake_loss + real_loss

def vanilla_generator_loss(logits_fake, dtype):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    N,_ = logits_fake.size()
    return bce_loss(logits_fake, torch.autograd.Variable(torch.ones(N)).type(dtype))

def ls_discriminator_loss(scores_real, scores_fake, dtype):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    N, _ = scores_real.size()
    
    loss_real = torch.pow(torch.autograd.Variable(torch.ones(N)).type(dtype) - scores_real, 2)
    loss_real = 0.5 * torch.mean(loss_real)
    
    N, _ = scores_fake.size()
    loss_fake = torch.pow(scores_fake,2)
    loss_fake = 0.5 * torch.mean(loss_fake)
    
    loss = loss_fake + loss_real    
    return loss

def ls_generator_loss(scores_fake, dtype):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    N, _ = scores_fake.size()
    loss = torch.pow(torch.autograd.Variable(torch.ones(N)).type(dtype) - scores_fake, 2)
    loss = 0.5 * torch.mean(loss)
    return loss




def get_optimizer(model, lr=1e-3, beta1=0.5, beta2=0.999):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999. These values can be chaged.
    
    Input:
    - model: A PyTorch model that we want to optimize.
    
    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    return optimizer

