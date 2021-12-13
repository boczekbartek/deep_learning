import torch.nn as nn

from data import *


class GAN(nn.Module):
        def __init__(self, D, M):
                super(GAN, self).__init__()
                self.D = D
                self.M = M
                self.gen1 = nn.Linear(in_features= self.M, out_features=300)
                self.gen2 = nn.Linear(in_features=300, out_features= self.D)
                self.dis1 = nn.Linear(in_features= self.D, out_features=300)
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
        # We maximize wrt. the discriminator, but optimizers minimize! # We need to include the negative sign!
                return -(torch.log(d_real) + torch.log(1. - d_gen))

        def forward(self, x_real):
                x_gen = self.generate(N=x_real.shape[0])
                d_real = self.discriminate(x_real)
                d_gen = self.discriminate(x_gen)
                return d_real, d_gen


        def gen_loss(self, d_gen):
                return torch.log(1. - d_gen)
        def dis_loss(self, d_real, d_gen):
        # We maximize wrt. the discriminator, but optimizers minimize! # We need to include the negative sign!
                return -(torch.log(d_real) + torch.log(1. - d_gen))
        def forward(self, x_real):
                x_gen = self.generate(N=x_real.shape[0])
                d_real = self.discriminate(x_real)
                d_gen = self.discriminate(x_gen)
                return d_real, d_gen


def train(model, train_loader, optimizer, epoch, device, log_interval):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )

    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / len(train_loader.dataset)))

