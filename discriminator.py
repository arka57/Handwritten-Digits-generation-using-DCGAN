import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,channels_image,features_d):
        super(Discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Conv2d(channels_image,features_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_d,features_d*2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d*2,features_d*4,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(features_d*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d*4,features_d*8,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(features_d*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_d*8,1,kernel_size=4,stride=2,padding=0),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.dis(x)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"        