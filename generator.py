import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,channels_noise,channel_image,features_g):
        super(Generator,self).__init__()
        self.gen=nn.Sequential(
            nn.ConvTranspose2d(channels_noise,features_g*16,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(features_g*16),
            nn.ReLU(),
            nn.ConvTranspose2d(features_g*16,features_g*8,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(features_g*8),
            nn.ReLU(),
            nn.ConvTranspose2d(features_g*8,features_g*4,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(features_g*4),
            nn.ReLU(),
            nn.ConvTranspose2d(features_g*4,features_g*2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(features_g*2),
            nn.ReLU(),
            nn.ConvTranspose2d(features_g*2,channel_image,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )

    def forward(self,x):
        return self.gen(x)    