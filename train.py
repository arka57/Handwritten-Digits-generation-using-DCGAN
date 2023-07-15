import torch
import torch.nn as nn
import torch.optim as opt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,random_split

from torchvision.utils import save_image
from model import Discriminator,Generator,weight_initialize

#Hyperparameters
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE=2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 3
FEATURES_DISC = 64
FEATURES_GEN = 64
folder="test"
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

#Creating the dataset --MNIST
dataset=datasets.MNIST(root="dataset/",train=True,transform=transforms,download=True)
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

#Initializing the network
dis=Discriminator(CHANNELS_IMG,FEATURES_DISC).to(device)
gen=Generator(NOISE_DIM,CHANNELS_IMG,FEATURES_GEN).to(device)
weight_initialize(dis)
weight_initialize(gen)

opt_d=opt.Adam(dis.parameters(),lr=LEARNING_RATE,betas=(0.5, 0.999))
opt_g=opt.Adam(gen.parameters(),lr=LEARNING_RATE,betas=(0.5, 0.999))
criterion=nn.BCELoss()

fixed_noise=torch.randn(32,NOISE_DIM,1,1).to(device)


gen.train()
dis.train()

#Training

for epoch in range(NUM_EPOCHS):
    for batch_index,(real,_) in enumerate(dataloader):
        real=real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        #print("noise",noise.shape)
        fake=gen(noise)
        #print("fake",fake.shape)

        #Train Discriminator
        real_dis=dis(real).reshape(-1)
        #print("real_dis",real_dis.shape)
        fake_dis=dis(fake.detach()).reshape(-1)
        #print("fake_dis",fake_dis.shape)

        loss_dis_real=criterion(real_dis,torch.ones_like(real_dis))
        loss_dis_fake=criterion(fake_dis,torch.zeros_like(fake_dis))

        loss_D=(loss_dis_fake+loss_dis_real)/2

        opt_d.zero_grad()
        loss_D.backward()
        opt_d.step()

        #Train Generator
        output=dis(fake).reshape(-1)
        #print("output",output.shape)
        loss_G=criterion(output,torch.ones_like(output))
        opt_g.zero_grad()
        loss_G.backward()
        opt_g.step()


         # Print losses occasionally and print to tensorboard
        if batch_index % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_index}/{len(dataloader)} \
                  Loss D: {loss_D:.4f}, loss G: {loss_G:.4f}"
            )


checkpoint_gen = {
        "gen_state_dict": gen.state_dict(),
        "gen_optimizer": opt_g.state_dict(),
    }
torch.save(checkpoint_gen, "gen.pth.tar")

checkpoint_dis = {
        "dis_state_dict": dis.state_dict(),
        "dis_optimizer": opt_d.state_dict(),
    }
torch.save(checkpoint_gen, "dis.pth.tar")
           


gen.eval()
with torch.no_grad():
    test_image=gen(fixed_noise)
    test_image=test_image*0.5+0.5
    #test_image=test_image.permute(0,2,3,1)
    save_image(test_image,folder+f"/test_1.png")
