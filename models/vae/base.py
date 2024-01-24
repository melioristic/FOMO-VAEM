from abc import abstractmethod

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .dist import log_normal_diag, log_categorical, log_bernoulli, log_normal_diag_prop


class RoundStraightThrough(torch.autograd.Function):
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input):
        rounded = torch.round(input, out=None)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super(Encoder, self).__init__()

        self.encoder = encoder_net

        #! Understand better
        self.round = RoundStraightThrough.apply

    def reparameterization(self, mu_e, log_var_e):
        std = torch.exp(0.5*log_var_e)
        eps = torch.randn_like(std)
        return mu_e + std * eps

    def encode(self, x):
        h_e = self.encoder(x)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        return mu_e, log_var_e

    def sample(self, x=None, mu_e=None, log_var_e=None):
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        else:
            if (mu_e is None) or (log_var_e is None):
                raise ValueError('mu and log-scale can`t be None!')
            z = self.reparameterization(mu_e, log_var_e)
        return z

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError('mu, log-scale and z can`t be None!')

        return log_normal_diag(z, mu_e, log_var_e)

    def forward(self, x, type='log_prob'):
        assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x)
        else:
            return self.sample(x)
        

class Decoder(nn.Module):
    def __init__(self, decoder_net, distribution='gaussian', num_vals=None):
        super(Decoder, self).__init__()

        self.decoder = decoder_net
        self.distribution = distribution
        self.num_vals=num_vals

    def decode(self, z):
        h_d = self.decoder(z)

        if self.distribution == 'gaussian':
            b = h_d.shape[0]
            d = h_d.shape[1]
            h_d = h_d.view(b, d)
            mu_d = h_d
            return [mu_d]
        
        elif self.distribution == 'categorical':
            b = h_d.shape[0]
            d = h_d.shape[1]//self.num_vals
            h_d = h_d.view(b, d, self.num_vals)
            mu_d = torch.softmax(h_d, 2)
            return [mu_d]


        elif self.distribution == 'bernoulli':
            mu_d = torch.sigmoid(h_d)
            return [mu_d]

        else:
            raise ValueError('Either `gaussian` or `categorical` or `bernoulli`')

    def sample(self, z):
        outs = self.decode(z)

        if self.distribution == 'gaussian':
            mu_d = outs[0]
            x_new = mu_d
        
        elif self.distribution == 'categorical':
            mu_d = outs[0]
            b = mu_d.shape[0]
            m = mu_d.shape[1]
            mu_d = mu_d.view(mu_d.shape[0], -1, self.num_vals)
            p = mu_d.view(-1, self.num_vals)
            x_new = torch.multinomial(p, num_samples=1).view(b, m)

        elif self.distribution == 'bernoulli':
            mu_d = outs[0]
            x_new = torch.bernoulli(mu_d)

        else:
            raise ValueError('Either `categorical` or `bernoulli`')

        return x_new

    def log_prob(self, x, z):

        outs = self.decode(z)

        if self.distribution == 'gaussian':
            mu_d = outs[0]
            log_p = log_normal_diag_prop(x, mu_d, reduction='sum', dim=-1).sum(-1)
        elif self.distribution == 'categorical':
            mu_d = outs[0]
            log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)

        elif self.distribution == 'bernoulli':
            mu_d = outs[0]
            log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)

        else:
            raise ValueError('Either `categorical` or `bernoulli`')

        return log_p

    def forward(self, z, x=None, type='log_prob'):
        assert type in ['decoder', 'log_prob'], 'Type could be either decode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x, z)
        else:
            return self.sample(z)
        

class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior, num_vals=None, L=16, likelihood_type='gaussian', beta=1, kld_weight=1):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.num_vals = num_vals


        self.likelihood_type = likelihood_type

    def forward(self, x):
        # encoder
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        r = self.decoder.sample(z)

        return r, x, z, mu_e, log_var_e 
        # ELBO
        
    def loss(self, x, reduction = "avg"):
        r, x, z, mu_e, log_var_e = self.forward(x)

        RE = self.decoder.log_prob(x, z)
        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)

        error = 0
        if np.isnan(RE.cpu().detach().numpy()).any():
            print('RE {}'.format(RE))
            error = 1
        if np.isnan(KL.cpu().detach().numpy()).any():
            print('RE {}'.format(KL))
            error = 1

        if error == 1:
            raise ValueError()

        if reduction == 'sum':
            return -RE.sum(), -KL.sum()
        else:
            return -RE.mean(), -KL.mean()
        
    def sample(self, batch_size=64):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)
    
class BaseVAE(nn.Module):
    def __init__(self,) -> None:
        super(BaseVAE, self).__init__()

    def encode(self,):
        raise NotImplementedError
    
    def decode(self,):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError
    
    def generate(self):
        raise NotImplementedError
    
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def loss_function(self):
        pass
