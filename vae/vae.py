from torch import nn
import torch

from .base_vae import BaseVAE
from torch.nn import functional as F


class ResNetBlock(nn.Module):
    def __init__(self, c_dim):
        super(ResNetBlock, self).__init__()
        
        # Manual depth=2
        layers = []
        expand_dim = 64
        layers.append(nn.Conv2d(c_dim, expand_dim*c_dim, kernel_size = 3, padding=1))
        layers.append(nn.BatchNorm2d(expand_dim*c_dim))
        layers.append(nn.ReLU())
            
        layers.append(nn.Conv2d(expand_dim*c_dim, c_dim, kernel_size = 3, padding=1))
        layers.append(nn.BatchNorm2d(c_dim))
        layers.append(nn.ReLU())

        self.module_list = nn.ModuleList(layers)

    def forward(self, x):

        x_prime = x
        for layer in self.module_list[:-1]:
            x_prime = layer(x_prime)

        return self.module_list[-1](x + x_prime)           
    
class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        self.down_sample = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.down_sample(x)

class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        
        self.up_sample = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.up_sample(x)

class Encoder(nn.Module):
    def __init__(self, in_channel, dim, n_step = 2) :
        super(Encoder, self).__init__()
        
        channel_dims =  [(1,1) for _ in range(n_step)]
        
        layers = []

        if (dim!=in_channel):
            layers.append(nn.Conv2d(in_channel, dim, kernel_size = 3, padding=1))

        for c_dim in channel_dims:
            layers.append(ResNetBlock(c_dim[0]))
            layers.append(DownSample(c_dim[0], c_dim[1]))

        layers.append(nn.Conv2d(c_dim[1], c_dim[1], kernel_size = 3, padding=1))

        self.module_list = nn.ModuleList(layers)

    def forward(self, x):
        print(x.shape)
        for layer in self.module_list:
            print(x.shape)
            x = layer(x)

        return torch.split(x, x.shape[2]//2, dim=2)

class Decoder(nn.Module):
    def __init__(self, out_channel, dim, n_step=2):
        super(Decoder, self).__init__()
        
        channel_dims =  [(1,1) for i in range(n_step)]
        
        layers = []
        for c_dim in channel_dims:
            layers.append(ResNetBlock(c_dim[0]))
            layers.append(UpSample(c_dim[0], c_dim[1]))

        if (dim!=out_channel):
            layers.append(nn.Conv2d(dim, out_channel, kernel_size = 3, padding=1))
        
        self.module_list = nn.ModuleList(layers)
    
    def forward(self,x):
        for layer in self.module_list:
            x = layer(x)
        return x
    

class ConvVAE(BaseVAE):
    def __init__(self, inp_shape, n_step, dim) -> None:
        super(ConvVAE, self).__init__()

        if dim==None:
            dim = inp_shape[0]
        
        self.encoder = Encoder(in_channel = inp_shape[0], dim = dim, n_step = n_step)
        self.decoder = Decoder(out_channel = inp_shape[0], dim = dim, n_step = n_step)
        
        self.latent_dim = (inp_shape[1]/ 2**(n_step), 1)
        

    def forward(self,x):

        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        r = self.decoder(z)
        return r, x, mu, log_var

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps*std + mu
    
    def loss_function(self, recons, x, mu, log_var, beta, kld_weight):
        
        recons_loss = F.mse_loss(recons , x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = (1,2)), dim = 0)

        loss = recons_loss + beta * kld_weight * kld_loss


        return {'loss': loss, 'recon_loss':recons_loss, 'KLD_loss':kld_loss}


    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim[0], self.latent_dim[1], self.latent_dim[2])
        return self.decoder(z) 
    
    def generate(self, x):
        return self.forward(x)[0]
    
    
