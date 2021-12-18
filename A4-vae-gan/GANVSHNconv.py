import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from models import calculate_fc_size_decoder, calculate_fc_size_encoder
import torchvision

  


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        img_h: int,
        img_w: int,
        n_fil_1: int = 32,
        n_fil_2: int = 64,
        kernel_size: int = 3,
        fc_size: int = 128,
    ) -> None:
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.D = latent_dim

        self.conv1_n_fil = n_fil_1
        self.conv2_n_fil = n_fil_2

        self.conv_kernel_size = kernel_size

        self.lin1 = nn.Linear(latent_dim, fc_size)
        self.drop1 = nn.Dropout(0.5)
        fc_size_2 = calculate_fc_size_decoder(self.img_h, self.img_w, self.conv_kernel_size, self.conv2_n_fil)
        self.lin2 = nn.Linear(fc_size, fc_size_2)
        self.drop2 = nn.Dropout(0.25)
        self.dconv1 = nn.ConvTranspose2d(
            in_channels=self.conv2_n_fil, out_channels=self.conv1_n_fil, kernel_size=self.conv_kernel_size, stride=1
        )
        self.dconv2 = nn.ConvTranspose2d(
            in_channels=self.conv1_n_fil, out_channels=in_channels, kernel_size=self.conv_kernel_size, stride=1
        )
    def generate(self,N):
        #print("generator shape")
        z = torch.randn(size=(N, self.D))
        x_gen=self.lin1(z)
        x_gen=F.relu(x_gen)
        x_gen=self.drop1(x_gen)

        x_gen=self.lin2(x_gen)
        x_gen = F.relu(x_gen)
        x_gen=self.drop2(x_gen)

        conv1_size_reduction = self.conv_kernel_size - 1
        conv2_size_reduction = self.conv_kernel_size - 1
        total_reduction = conv1_size_reduction + conv2_size_reduction

        x_gen = x_gen.view(-1, self.conv2_n_fil, (self.img_h - total_reduction), (self.img_w - total_reduction))
        x_gen=self.dconv1(x_gen)
        x_gen=self.dconv2(x_gen)
        #print(x_gen.shape)
        return x_gen



    def forward(self, x):
        x_gen=self.generate(N=x.shape[0])
        return x_gen

    def gen_loss(self, d_gen):
        return torch.log(1. - d_gen)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int, #D
        img_h: int,
        img_w: int,
        out_size: int = 1,
        n_fil_1: int = 64,
        n_fil_2: int = 32,
        kernel_size: int = 3,
        mp_size: int = 2,
    ) -> None:
        super().__init__()
        # Image size
        self.img_h = img_h
        self.img_w = img_w

        # Number of filters in conv layers
        self.conv1_n_fil = n_fil_1
        self.conv2_n_fil = n_fil_2

        self.conv_kernel_size = kernel_size
        self.max_pool_kernel_size = mp_size

        # Encoder layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=self.conv1_n_fil, kernel_size=self.conv_kernel_size, stride=1
        ) #here D
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1_n_fil, out_channels=self.conv2_n_fil, kernel_size=self.conv_kernel_size, stride=1
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=self.max_pool_kernel_size)
        self.drop1 = nn.Dropout(0.25)

        fc_size = (6272)
            #calculate_fc_size_encoder(self.img_h, self.img_w, self.conv_kernel_size, self.conv2_n_fil, self.max_pool_kernel)
        self.lin1 = nn.Linear(fc_size, out_size)
        self.drop2 = nn.Dropout(0.5)

    def discriminate(self, x):
        #print("discriminator shapes")
        #print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = F.relu(x)
        x = self.max_pool1(x)
        #print(x.shape)
        x = self.drop1(x)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        x = self.lin1(x)
        #print(x.shape)
        x = F.relu(x)
        x = self.drop2(x)
        x = torch.sigmoid(x)
        return x

    def forward(self,x):
        d_gen=self.discriminate(x)
        return d_gen

    def dis_loss(self, d_real, d_gen):
        # We maximize wrt. the discriminator, but optimizers minimize!
        # We need to include the negative sign!
        return -(torch.log(d_real) + torch.log(1. - d_gen))





from data import *
import torch

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
print(dev)


batch_size = 64
svhn =torchvision.datasets.SVHN(root= "A4-vae-gan/data/VSHN", download = True)#root: str, split: str = 'train', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
#print(svhn.data)
print(svhn.data.shape)
x_train = svhn.data
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size)

#test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size)


#data = train_iterator.dataset.data
#shape = train_iterator.dataset.data.shape
#datatype = train_iterator.dataset.data.dtype

#print(shape)
#print(datatype)
import tqdm
def train(G, D, train_loader, optimizer_D,optimizer_G, epochs, device, log_interval):
    #model.train()
    #train_D_loss, train_G_loss = 0, 0
    train_loss = 0
    lossfunc = nn.BCELoss()
    #all_zeros = torch.zeros_like(batch_size, dtype=torch.float32)
    #all_ones = torch.ones_like(batch_size, dtype=torch.float32)
    #for epoch in tqdm(range(epochs), desc='epochs'):
    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            dis_accum_loss,gen_accum_loss = 0, 0
            data = data.to(device)

            # FIRST TRAIN DISCRIMINATOR
            optimizer_D.zero_grad()
            probReal = D(data) #[0,1]


            all_ones = torch.ones_like(probReal, dtype = torch.float32)
            d_loss_real = lossfunc(probReal, all_ones)


            # NOW GENERATE
            generated = G(data)
            probGen = D(generated)
            all_zeros = torch.zeros_like(probGen, dtype=torch.float32)
            d_loss_fake = lossfunc(probGen, all_zeros)

            d_loss = d_loss_real+d_loss_fake
            dis_accum_loss += d_loss
            #d_loss = Discriminator.dis_loss(Discriminator, probReal, probGen)

            d_loss.backward()
            optimizer_D.step()



            # Optim generator:
            optimizer_G.zero_grad()
            fakes = G(data)
            D_fakes = D(fakes)
            g_loss = lossfunc(D_fakes, all_ones)
            #g_loss=Generator.gen_loss(Generator,D_fakes)
            gen_accum_loss+=g_loss

            g_loss.backward()
            optimizer_G.step()






            #train_D_loss += d_loss.item()
            #train_G_loss += g_loss.item()

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\D-Loss: {:.6f}, G-loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        d_loss.item() / len(data),
                        g_loss.item()/ len(data),
                    )
                )

        print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_D_loss / len(train_loader.dataset)))

G = Generator(3,128,32,32).to(dev)
D = Discriminator(3,28,28).to(dev)

  
optimizer_G=torch.optim.Adam(G.parameters(),lr=0.001)
optimizer_D = torch.optim.Adam(D.parameters(), lr = 0.001)

train(G = G, D=D, train_loader =train_loader, optimizer_D=optimizer_D ,optimizer_G=optimizer_G,epochs= 1,device= dev, log_interval =100)



def view_images(images_list, idx):
  fig, axes = plt.subplots(figsize=(7,7), nrows=4,
    ncols=4, sharey=True, sharex=True)
  for ax, img in zip(axes.flatten(), images_list[idx]):
    img = img.detach()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    im = ax.imshow(img.reshape((28,28)), cmap='gray_r')
  plt.show()

print("\nGenerating 16 images using trained generator ")
num_images = 16
zz = torch.normal(0.0, 1.0, size=(num_images, 28*28))
rand_images = G(zz)
images_list = [rand_images]
view_images(images_list, 0)

print("\nEnd GAN MNIST demo ")