from torch import nn

class EncoderMLP(nn.Module):
    def __init__(self, inp_dim, latent_dim, hidden_neurons):
        super(EncoderMLP, self).__init__()
        D, M, L = inp_dim, hidden_neurons, latent_dim
        self.layers = nn.Sequential(nn.Linear(D,M), nn.LeakyReLU(),
                                    nn.Linear(M,M), nn.LeakyReLU(),
                                    nn.Linear(M, 2*L))
        
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)

        return x
    
class DecoderMLP(nn.Module):
    def __init__(self, out_dim, latent_dim, hidden_neurons):
        super(DecoderMLP, self).__init__()
        D, M, L = out_dim, hidden_neurons, latent_dim
        self.layers = nn.Sequential(nn.Linear(L,M), nn.LeakyReLU(),
                                    nn.Linear(M,M), nn.LeakyReLU(),
                                    nn.Linear(M, D))
        
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)

        return x