
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torch

batch_size = 64
svhn =torchvision.datasets.SVHN(root= "A4-vae-gan/data/VSHN", download = True)#root: str, split: str = 'train', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)
#print(svhn.data)
#print(svhn.data.shape)
x_train = svhn.data
x_train = (x_train.astype(np.float32) ) / 255

x_train=x_train.reshape(-1,32*32)

#print(x_train.shape)

train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size)






class GAN(nn.Module):
    def __init__(self, D, M):
        super(GAN, self).__init__()
        self.D = D
        self.M = M

        self.gen1 = nn.Linear(in_features=self.M, out_features=300)
        self.gen2 = nn.Linear(in_features=300, out_features=self.D)
        self.dis1 = nn.Linear(in_features=self.D, out_features=300)
        self.dis2 = nn.Linear(in_features=300, out_features=1)

    def generate(self, N):
        z = torch.randn(size=(N, self.D))
        x_gen = self.gen1(z)
        x_gen = nn.functional.relu(x_gen)
        x_gen = self.gen2(x_gen)
        return x_gen

    def discriminate(self, x):
        y = self.dis1(x)
        y = nn.functional.relu(y)
        y = self.dis2(y)
        y = torch.sigmoid(y)
        return y

    def gen_loss(self, d_gen):
        return torch.log(1. - d_gen)

    def dis_loss(self, d_real, d_gen):
        # We maximize wrt. the discriminator, but optimizers minimize!
        # We need to include the negative sign!
        return -(torch.log(d_real) + torch.log(1. - d_gen))

    def forward(self, x_real):
        print('x_real.shape = ', x_real.shape)

        x_gen = self.generate(N=x_real.shape[0])
        print('x_gen.shape = ', x_gen.shape)

        d_real = self.discriminate(x_real)
        print('d_real.shape = ', d_real.shape)

        d_gen = self.discriminate(x_gen)
        print('d_gen.shape = ', d_gen.shape)
        return d_real, d_gen


class Generator(nn.Module):
    def __init__(self, D, M):
        super(Generator, self).__init__()

        self.D = D
        self.M = M  # latent dimension

        self.gen1 = nn.Linear(in_features=self.M, out_features=300)
        self.gen2 = nn.Linear(in_features=300, out_features=self.D)

    def generate(self, N):
        z = torch.randn(size=(N, self.D))
        x_gen = self.gen1(z)
        x_gen = nn.functional.relu(x_gen)
        x_gen = self.gen2(x_gen)
        #print("x_gen shape")
        #print(x_gen.shape)
        return x_gen

    def gen_loss(self, d_gen):
        return torch.log(1. - d_gen)

    def forward(self, x_real):
        # print('x_real.shape[0] = ',x_real.shape[0])

        x_gen = self.generate(N=x_real.shape[0])
        # print('x_gen.shape = ',x_gen.shape)

        return x_gen


class Discriminator(nn.Module):
    def __init__(self, D):
        super(Discriminator, self).__init__()
        self.D = D

        self.dis1 = nn.Linear(in_features=self.D, out_features=300)
        self.dis2 = nn.Linear(in_features=300, out_features=1)

    def discriminate(self, x):
        #print(x.shape)
        y = self.dis1(x)

        y = nn.functional.relu(y)
        y = self.dis2(y)
        y = torch.sigmoid(y)
        return y

    def dis_loss(self, d_real, d_gen):
        # We maximize wrt. the discriminator, but optimizers minimize!
        # We need to include the negative sign!
        return -(torch.log(d_real) + torch.log(1. - d_gen))

    def forward(self, x):
        # print('x_real.shape = ',x.shape)

        d_gen = self.discriminate(x)
        # print('d_gen.shape = ',d_gen.shape)
        return d_gen


from data import *

import torch

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
print(dev)





def train(G, D, train_loader, optimizer_D, optimizer_G, epochs, device, log_interval):
    # model.train()
    train_D_loss, train_G_loss = 0, 0
    train_loss = 0
    lossfunc = nn.BCELoss()
    # for epoch in tqdm(range(epochs), desc='epochs'):
    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            # print(data.shape)
            # data.view(-1, 28*28)
            data = data.to(device)

            # FIRST TRAIN DISCRIMINATOR
            optimizer_D.zero_grad()
            probReal = D(data)

            all_ones = torch.ones_like(probReal, dtype=torch.float32)
            d_loss_real = lossfunc(probReal, all_ones)

            # NOW GENERATE
            generated = G(data)
            probGen = D(generated)
            all_zeros = torch.zeros_like(probGen, dtype=torch.float32)
            d_loss_fake = lossfunc(probGen, all_zeros)

            d_loss = ((d_loss_real + d_loss_fake) / 2).mean()

            d_loss.backward()
            optimizer_D.step()

            # Optim generator:
            optimizer_G.zero_grad()
            fakes = G(data)
            D_fakes = D(fakes)
            g_loss = lossfunc(D_fakes, all_ones)
            g_loss.backward()
            optimizer_G.step()

            train_D_loss += d_loss.item()
            train_G_loss += g_loss.item()

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\D-Loss: {:.6f}, G-loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        d_loss.item() / len(data),
                        g_loss.item() / len(data),
                    )
                )

        print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_D_loss / len(train_loader.dataset)))


G = Generator(D=32*32 , M=32*32 ).to(dev)
D = Discriminator(D=32*32 )

optimizer_G = torch.optim.Adam(G.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.001)

train(G=G, D=D, train_loader=train_loader, optimizer_D=optimizer_D, optimizer_G=optimizer_G, epochs=10, device=dev,
      log_interval=100)





def view_images(images_list, idx, filename=None):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=5,
                             ncols=2, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), images_list[idx]):
        img = img.detach()

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32, 32)), cmap='gray_r')

    if filename is not None:
        plt.savefig(filename)
    plt.show()


def get_interpolations(noise1, noise2, num_interpolates):
    x = [((1.0 - (i / num_interpolates)) * noise1) + ((i / num_interpolates) * noise2) for i in
         range(num_interpolates + 1)]
    return torch.cat(x, dim=0)


def interpolate(num_samples, model, input_dim):
    noise1 = torch.randn((1, input_dim), device=dev)
    noise2 = torch.randn((1, input_dim), device=dev)

    to_sample = get_interpolations(noise1, noise2, num_samples)
    samples = model(num_samples, to_sample)
    print(samples.shape)

    return samples


print("\nGenerating 16 images using trained generator ")
num_images = 16
zz = torch.normal(0.0, 1.0, size=(num_images, 32 * 32))
rand_images = G(zz).to('cpu')
images_list = [rand_images]
view_images(images_list, 0, 'demo.png')

for i in range(10):
    interpolate_samples = interpolate(10, G, G.D).cpu()
    samples_list = [interpolate_samples]
    view_images(samples_list, 0, 'MNIST_GAN' + str(i) + '.png')
print("\nEnd GAN MNIST demo ")