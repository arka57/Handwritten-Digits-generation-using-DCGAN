import torch
import torch.nn as nn



class Discriminator(nn.Module):
    def __init__(self,channels_image,features_d):
        super(Discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Conv2d(channels_image,features_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d,features_d*2,4,2,1),
            self._block(features_d*2,features_d*4,4,2,1),
            self._block(features_d*4,features_d*8,4,2,1),
            nn.Conv2d(features_d*8,1,kernel_size=4,stride=2,padding=0),
            nn.Sigmoid()
        )

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,out_channels,kernel_size,stride,padding,bias=False
            ),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.dis(x)  

class Generator(nn.Module):
    def __init__(self,channels_noise,channels_image,features_g):
        super(Generator,self).__init__()
        self.gen=nn.Sequential(
            self._block(channels_noise,features_g*16,4,1,0),#4x4
            self._block(features_g*16,features_g*8,4,2,1),#8x8
            self._block(features_g*8,features_g*4,4,2,1),#16x16
            self._block(features_g*4,features_g*2,4,2,1),#32x32
            nn.ConvTranspose2d(features_g*2,channels_image,kernel_size=4,stride=2,padding=1),#64x64
            nn.Tanh()

        )              

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d( in_channels,out_channels,kernel_size,stride,padding,bias=False 
            ),
            nn.ReLU()
        )    

    def forward(self,x):
        return self.gen(x)


def weight_initialize(model):

    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()