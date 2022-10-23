import torch
import torch.nn as nn
import torch.nn.functional as F



## nonlinear activation function
class Poly_Tunable(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1)*0.01)
        self.b = nn.Parameter(torch.ones(1)*1.0)
        self.c = nn.Parameter(torch.ones(1)*0.01)

    def forward(self, z):
        return self.a * z * z + self.b * z + self.c



#####################################
class PGBlock(nn.Module):
    """docstring for ResBlock"""
    def __init__(self, dim=16, channel=4):
        super(PGBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv = nn.Sequential(nn.ConvTranspose1d(1, channel, 5, stride=2, padding=2, output_padding=1, bias=False),
                                    nn.BatchNorm1d(channel),
                                    nn.LeakyReLU(0.2),
                                    #Poly_Tunable(),
                                    nn.Conv1d(channel, 1, 1, stride=1, bias=False))
        #self.deconv = nn.Sequential(nn.ConvTranspose1d(1, 1, 5, stride=2, padding=2, output_padding=1, bias=False),
        #                            nn.BatchNorm1d(channel),
        #                            nn.LeakyReLU(0.2))
        #self.alpha = nn.Parameter(torch.ones(1)*0.05)

    def forward(self, x, alpha):

        net = self.upsample(x) * (1 -alpha) + self.deconv(x) * alpha

        return net

## ResNet Generator 

# dim is the dimension of both input noise and output device
class PGGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=4):
        super().__init__()

        self.num_layers = num_layers
        self.dims = [64, 128, 256, 512, 1024]
        self.channels = [16, 8, 4, 2, 1]
        self.alphas = [0, 0, 0, 0]
        self.PGBLOCK = nn.ModuleList()
        for i in range(num_layers):
            self.PGBLOCK.append(PGBlock(self.dims[i], self.channels[i]))

        self.iniFC = nn.Sequential(nn.Linear(in_dim, self.dims[0]),
                                nn.BatchNorm1d(self.dims[0]))
        self.endFC = nn.Linear(self.dims[-1], out_dim, bias=False)


    def forward(self, z):
        x = self.iniFC(z).unsqueeze(1)
        for i in range(self.num_layers):
            x = self.PGBLOCK[i](x, self.alphas[i])
        x = self.endFC(torch.squeeze(x))
        return torch.tanh(x)

    def update_alpha(self, i, alpha):
        self.alphas[i] = alpha


