import numpy as np
import torch.nn as nn
import torchvision as tv
import time
import matplotlib.pyplot as plt
import torch

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)


class Discriminator(nn.Module):
    def __init__(self,D):
        super(Discriminator, self).__init__()
        self.D = D
        self.dis1 = nn.Linear(in_features=self.D, out_features=300)
        self.dis2 = nn.Linear(in_features=300, out_features=1)


    def forward(self, x):
        x = x.view(-1, 28 * 28)  # flatten image(s)
        y = self.dis1(x)
        y = nn.functional.relu(y)
        y = self.dis2(y)
        y = torch.sigmoid(y)
        return y

    def dis_loss(self, d_real, d_gen):
        # We maximize wrt. the discriminator, but optimizers minimize! # We need to include the negative sign!
        return -(torch.log(d_real) + torch.log(1. - d_gen))


# -----------------------------------------------------------

class Generator(nn.Module):  # 100-32-64-128-784
    def __init__(self):
        super(Generator, self).__init__()
        self.M = M
        self.gen1 = nn.Linear(in_features=self.M, out_features=300)
        self.gen2 = nn.Linear(in_features=300, out_features= self.D)

    def forward(self, N):
        z = torch.randn(size=(N, self.D))
        x_gen = self.gen1(z)
        x_gen = nn.functional.relu(x_gen)
        x_gen = self.gen2(x_gen)
        return x_gen

    def gen_loss(self, d_gen):
        return torch.log(1. - d_gen)



# -----------------------------------------------------------

def view_images(images_list, idx):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4,
                             ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), images_list[idx]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='gray_r')
    plt.show()


# -----------------------------------------------------------

def main():
    # 0. get started
    print("\nBegin GAN MNIST demo ")
    np.random.seed(1)
    #nn.manual_seed(1)

    # 1. create MNIST DataLoader object
    print("\nCreating MNIST Dataset and DataLoader ")

    bat_size = 64
    # requested size, not necessarily actual
    # 60,000 train images / 64 = 937 batches + 1 of size 32

    trfrm = tv.transforms.ToTensor()  # also divides by 255
    train_ds = tv.datasets.MNIST(root="data", train=True,
                                 download=True, transform=trfrm)
    train_ldr = nn.utils.data.DataLoader(train_ds,
                                        batch_size=bat_size, shuffle=True, drop_last=True)

    # 2. create networks
    dis = Discriminator().to(device)  # 784-128-64-32-1
    gen = Generator().to(device)  # 100-32-64-128-784

    # 3. train GAN model
    max_epochs = 100
    ep_log_interval = 10
    lrn_rate = 0.002  # small for Adam

    dis.train()  # set mode
    gen.train()
    dis_optimizer = nn.optim.Adam(dis.parameters(), lrn_rate)
    gen_optimizer = nn.optim.Adam(gen.parameters(), lrn_rate)
    loss_func = nn.nn.BCELoss()
    all_ones = nn.ones(bat_size, dtype=T.float32).to(device)
    all_zeros = nn.zeros(bat_size, dtype=T.float32).to(device)

    # -----------------------------------------------------------

    print("\nStarting training ")
    for epoch in range(0, max_epochs):
        for batch_idx, (real_images, _) in enumerate(train_ldr):
            dis_accum_loss = 0.0  # to display progress
            gen_accum_loss = 0.0

            # 1a. train discriminator (0/1) using real images
            dis_optimizer.zero_grad()
            dis_real_oupt = dis(real_images)  # [0, 1]
            dis_real_loss = loss_func(dis_real_oupt.squeeze(),
                                      all_ones)

            # 1b. train discriminator using fake images
            zz = nn.normal(0.0, 1.0,
                          size=(bat_size, 100)).to(device)
            fake_images = gen(zz)
            dis_fake_oupt = dis(fake_images)
            dis_fake_loss = loss_func(dis_fake_oupt.squeeze(),
                                      all_zeros)
            dis_loss_tot = dis_real_loss + dis_fake_loss
            dis_accum_loss += dis_loss_tot

            dis_loss_tot.backward()
            dis_optimizer.step()

            # 2. train gen with fake images and flipped labels
            gen_optimizer.zero_grad()
            zz = nn.normal(0.0, 1.0,
                          size=(bat_size, 100)).to(device)
            fake_images = gen(zz)
            dis_fake_oupt = dis(fake_images)
            gen_loss = loss_func(dis_fake_oupt.squeeze(), all_ones)
            gen_accum_loss += gen_loss

            gen_loss.backward()
            gen_optimizer.step()

        if epoch % ep_log_interval == 0 or epoch == max_epochs - 1:
            print(" epoch: %4d | dis loss: %0.4f | gen loss: %0.4f " \
                  % (epoch, dis_accum_loss, gen_accum_loss))
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            fn = ".\\Models\\" + str(dt) + str("-") + "epoch_" + \
                 str(epoch) + "_gan_mnist_model.pt"
            T.save(gen.state_dict(), fn)

    print("Training commplete ")

    # -----------------------------------------------------------

    print("\nGenerating 16 images using trained generator ")
    num_images = 16
    zz = nn.normal(0.0, 1.0, size=(num_images, 100)).to(device)
    gen.eval()
    rand_images = gen(zz)
    images_list = [rand_images]
    view_images(images_list, 0)

    print("\nEnd GAN MNIST demo ")


if __name__ == "__main__":
    main()