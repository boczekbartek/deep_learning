import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


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
        return x_gen


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
        y = self.dis1(x)
        y = nn.functional.relu(y)
        y = self.dis2(y)
        y = torch.sigmoid(y)
        return y



    def forward(self, x):
        # print('x_real.shape = ',x.shape)

        d_gen = self.discriminate(x)
        # print('d_gen.shape = ',d_gen.shape)
        return d_gen


from data import *
from MNISTDATA import *
import torch

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
print(dev)

batch_size = 64
(x_train, y_train), (x_test, y_test), lol = load_mnist()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size)

# data = train_iterator.dataset.data
# shape = train_iterator.dataset.data.shape
# datatype = train_iterator.dataset.data.dtype

# print(shape)
# print(datatype)
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

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
            #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            generated = G(data)
            #adversarial loss
            loss_D = -torch.mean(D(data)) + torch.mean(D(generated))
            loss_D.backward()

            optimizer_D.step()

            #CLIP WEIGHT
            for p in D.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            #if batch_idx % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

            optimizer_G.zero_grad()

                # Generate a batch of images
            gen_imgs = G(data)
                # Adversarial loss
            loss_G = -torch.mean(D(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\D-Loss: {:.6f}, G-loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss_D.item() / len(data),
                        loss_G.item() / len(data),
                    )
                )

        print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_D_loss / len(train_loader.dataset)))



G = Generator(D=28 * 28, M=28 * 28).to(dev)
D = Discriminator(D=28 * 28)

#optimizer_G = torch.optim.Adam(G.parameters(), lr=0.001)
#optimizer_D = torch.optim.Adam(D.parameters(), lr=0.001)
optimizer_G = torch.optim.RMSprop(G.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(D.parameters(), lr=opt.lr)



train(G=G, D=D, train_loader=train_loader, optimizer_D=optimizer_D, optimizer_G=optimizer_G, epochs=10, device=dev,
      log_interval=100)


def view_images(images_list, idx):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4,
                             ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), images_list[idx]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='gray_r')
    plt.show()


print("\nGenerating 16 images using trained generator ")
num_images = 16
zz = torch.normal(0.0, 1.0, size=(num_images, 28 * 28))
rand_images = G(zz)
images_list = [rand_images]
view_images(images_list, 0)

print("\nEnd GAN MNIST demo ")