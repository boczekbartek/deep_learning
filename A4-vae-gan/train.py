import logging
import pathlib

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from data import load_mnist
from models import VAEGauss
from criterions import elbo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def test(model, testloader, device, batch_size, epoch, results_dir: pathlib.Path):
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(testloader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)

            # TODO use elbo
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), results_dir / f"reconstruction_{epoch}.png", nrow=n)
        test_loss /= len(testloader.dataset)
        print("====> Test set loss: {:.4f}".format(test_loss))


def train_vae_gauss_mnist(epochs, batch_size, lr=0.001, cuda=True, log_interval=10):
    results_dir = pathlib.Path("vae_gauss_mnist")
    results_dir.mkdir(exist_ok=True)

    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logging.info(f"Device: {device}")
    trainloader, testloader = load_mnist(batch_size, cuda)

    x0, _ = next(iter(trainloader))
    b, c, h, w = x0.shape
    model = VAEGauss(in_channels=c, latent_dim=128, img_h=h, img_w=w)

    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = 0
    for epoch in range(1, epochs):
        logging.info(f"Epoch {epoch}")
        for batch_idx, (x, _) in enumerate(trainloader):
            x = x.to(device)
            # x = x.reshape(-1, 784)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)

            # TODO use elbo
            loss = loss_function(x_hat, x, mu, logvar)
            # loss = elbo(x, x_hat, z, mu, log_std)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                logging.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(x),
                        len(trainloader.dataset),
                        100.0 * batch_idx / len(trainloader),
                        loss.item() / len(x),
                    )
                )
        logging.info("====> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / len(trainloader.dataset)))
        test(model, testloader, device, batch_size, epoch, results_dir)


if __name__ == "__main__":
    train_vae_gauss_mnist(10, 64, log_interval=100, cuda=True)
