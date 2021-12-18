from __future__ import print_function
import argparse
import pathlib
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

class IWAE_1(nn.Module):
    def __init__(self, dim_h1, dim_image_vars):
        super(IWAE_1, self).__init__()
        self.dim_h1 = dim_h1
        self.dim_image_vars = dim_image_vars

        ## encoder
        self.encoder_h1 = Block(dim_image_vars, 400, dim_h1)

        ## decoder
        self.decoder_x = nn.Sequential(nn.Linear(dim_h1, 400),#20
                                       nn.ReLU(),
                                       nn.Linear(400, dim_image_vars),#784
                                       nn.Sigmoid())


    def encoder(self, x):
        mu_h1, sigma_h1 = self.encoder_h1(x.view(-1, 784))
        #mu_h1, sigma_h1 = self.encoder_h1(x)
        eps = Variable(sigma_h1.data.new(sigma_h1.size()).normal_())
        h1 = mu_h1 + sigma_h1 * eps
        return h1, mu_h1, sigma_h1, eps

    def decoder(self, h1):

        p = self.decoder_x(h1)
        return p

    def forward(self, x):
        h1,mu_h1, sigma_h1, eps = self.encoder(x.view(-1, 784))
       # h1, mu_h1, sigma_h1, eps = self.encoder(x)
        p = self.decoder(h1)
        return (h1, mu_h1, sigma_h1, eps), (p)

    def train_loss(self, inputs):
        h1, mu_h1, sigma_h1, eps = self.encoder(inputs)
        # log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1)/sigma_h1)**2 - torch.log(sigma_h1), -1)
        log_Qh1Gx = torch.sum(-0.5 * (eps) ** 2 - torch.log(sigma_h1), -1)
        inputs=inputs.view(-1,784)
        p = self.decoder(h1)
        log_Ph1 = torch.sum(-0.5 * h1 ** 2, -1)
        log_PxGh1 = torch.sum(inputs * torch.log(p) + (1 - inputs) * torch.log(1 - p), -1)

        log_weight = log_Ph1 + log_PxGh1 - log_Qh1Gx
        log_weight = log_weight - torch.max(log_weight, 0)[0]
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = Variable(weight.data, requires_grad=False)
        loss = -torch.mean(torch.sum(weight * (log_Ph1 + log_PxGh1 - log_Qh1Gx), 0))
        return loss

    def test_loss(self, inputs):
        h1, mu_h1, sigma_h1, eps = self.encoder(inputs)
        # log_Qh1Gx = torch.sum(-0.5*((h1-mu_h1)/sigma_h1)**2 - torch.log(sigma_h1), -1)
        log_Qh1Gx = torch.sum(-0.5 * (eps) ** 2 - torch.log(sigma_h1), -1)

        p = self.decoder(h1)
        log_Ph1 = torch.sum(-0.5 * h1 ** 2, -1)
        log_PxGh1 = torch.sum(inputs * torch.log(p) + (1 - inputs) * torch.log(1 - p), -1)

        log_weight = log_Ph1 + log_PxGh1 - log_Qh1Gx
        weight = torch.exp(log_weight)
        loss = -torch.mean(torch.log(torch.mean(weight, 0)))
        return loss

class Block(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim):
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.transform = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.transform(x)
        #print(out.shape)
        mu = self.fc_mu(out)
        logsigma = self.fc_logsigma(out)
        sigma = torch.exp(logsigma)
        return mu, sigma






def main(batch_size=128, epochs=100, no_cuda=False, seed=1, log_interval=10):
    # parser = argparse.ArgumentParser(description='VAE MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=128, metavar='N',
    #                     help='input batch size for training (default: 128)')
    # parser.add_argument('--epochs', type=int, default=10, metavar='N',
    #                     help='number of epochs to train (default: 10)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # args = parser.parse_args()
    pathlib.Path("results").mkdir(exist_ok=True)
    cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    model = IWAE_1(20, 784).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch, device, log_interval)
        #test(model, test_loader, epoch, batch_size, device)
        with torch.no_grad():
            #sample = torch.randn(64, 20).to(device)
            sample = torch.randn(64,20).to(device)
            sample = model.decoder(sample).cpu()
            print(sample.shape)
            save_image(sample.view(64,1,28,28), "results/sample_" + str(epoch) + ".png")



def train(model, train_loader, optimizer, epoch, device, log_interval):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        #recon_batch, mu, logvar = model(data)
        loss = model.train_loss(data)
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



if __name__ == '__main__':
    main()