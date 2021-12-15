import logging
import pathlib

import torch
import pandas as pd
from torchvision.utils import save_image

from data import data_factory
from criterions import criterions_factory
from models import models_factory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def compute_grad_norm(model) -> float:
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def test(model, testloader, device, epoch, results_dir: pathlib.Path, loss_function):
    test_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(testloader):
            x = x.to(device)
            x_hat, mu, std, z = model(x)

            test_loss += loss_function(x_hat, x, mu, std, z).item()
            if i == 0:
                x_hat_01 = torch.where(x_hat > 0.5, 1.0, 0.0)
                n = min(x.shape[0], 16)
                comparison = torch.cat([x[:n], x_hat[:n], x_hat_01[:n]])
                save_image(comparison.cpu(), results_dir / f"reconstruction_{epoch}.png", nrow=n)
        test_loss /= len(testloader.dataset)
        logging.info("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss


def train_vae_gauss(
    epochs, batch_size, model_name: str, dataset: str, criterion: str, lr=0.001, cuda=True, log_interval=10,
):
    results_dir = pathlib.Path(f"{model_name}_{dataset}_{criterion}")
    results_dir.mkdir(exist_ok=True)
    logging.info(f"Results dir: {results_dir}")

    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logging.info(f"Device: {device}")

    logging.info(f"Loading dataset: {dataset}")
    data_loading_function = data_factory(dataset)
    trainloader, testloader = data_loading_function(batch_size, cuda)

    x0, _ = next(iter(trainloader))
    b, c, h, w = x0.shape

    logging.info(f"Creating model: {model_name}")
    model = models_factory(model_name)(in_channels=c, img_h=h, img_w=w)

    logging.debug(f"Sending model to: {device}")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = criterions_factory(criterion)
    data = list()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        logging.info(f"Epoch {epoch}")
        batch_idx = 0
        for batch_idx, (x, _) in enumerate(trainloader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, std, z = model(x)

            loss = loss_function(x_hat, x, mu, std, z)
            loss /= torch.flatten(x_hat).shape[0] / len(x)

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
                        loss.item(),
                    )
                )
        train_loss /= batch_idx + 1
        logging.info(f"====> Epoch: {epoch} Average loss: {train_loss:.4f}")
        test_loss = test(model, testloader, device, epoch, results_dir, loss_function)
        data.append(
            {"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "grad_norm": compute_grad_norm(model)}
        )
    model_file = results_dir / "model.pt"
    progress_file = results_dir / "progress.csv"

    logging.info(f"Saving final model to: {model_file}")
    torch.save(model, model_file)
    pd.DataFrame(data).to_csv(progress_file)


if __name__ == "__main__":
    train_vae_gauss(30, 128, "vae-gauss-base", "mnist", "elbo", lr=1e-3, log_interval=50, cuda=True)
    train_vae_gauss(30, 64, "vae-gauss-big", "svhn", "elbo", lr=1e-3, log_interval=50, cuda=True)
