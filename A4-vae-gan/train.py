import logging
import pathlib

from typing import Tuple

import torch
import pandas as pd
from torchvision.utils import save_image
from pytorch_model_summary import summary

from data import data_factory
from criterions import criterions_factory
from models import models_factory
from evaluate import VAESamples

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def compute_grad_norm(model) -> float:
    with torch.no_grad():
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm


def test(inference_and_loss, model, testloader, device, epoch, results_dir: pathlib.Path, loss_function, batch_size):
    test_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(testloader):
            x = x.to(device)

            x_hat, loss = inference_and_loss(model, loss_function, x)
            loss /= torch.flatten(x_hat).shape[0] / len(x)

            test_loss += loss.item()
            if i == 0:
                # x_hat_01 = torch.where(x_hat > 0.5, 1.0, 0.0)
                n = min(x.shape[0], 16)
                comparison = torch.cat([x[:n], x_hat[:n]])
                save_image(comparison.cpu(), results_dir / f"reconstruction_{epoch}.png", nrow=n)
        test_loss /= batch_size
        logging.info("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss


def sample(model, n, results_dir, epoch, preload_batch_size=16):
    logging.info(f"Sampling {n} images from the model...")
    with torch.no_grad():
        dataset = VAESamples(model, total=n, preload_batch_size=preload_batch_size)
        x_hat = list(dataset[i][0].cpu() for i in range(n))
        save_image(x_hat, results_dir / f"samples_{epoch}.png", nrow=n)


def inference_and_loss_vae_gauss(model, loss_function, x) -> Tuple[torch.Tensor, torch.Tensor]:
    x_hat, mu, std, z = model(x)
    loss = loss_function(x_hat, x, mu, std, z)
    return x_hat, loss


def inference_and_loss_vae_realnvp(model, loss_function, x) -> Tuple[torch.Tensor, torch.Tensor]:
    x_hat, log_prob_z0, log_prob_zk, log_det = model(x)
    loss = loss_function(x_hat, x, log_prob_z0, log_prob_zk, log_det)
    return x_hat, loss


def inference_and_loss_vae_realnvp_jt(model, loss_function, x) -> Tuple[torch.Tensor, torch.Tensor]:
    x_hat, log_det_J, log_prob_z = model(x)
    loss = loss_function(x_hat, x, log_det_J, log_prob_z)
    return x_hat, loss


def inference_and_loss_vae_realnvp_jt2(model, loss_function, x) -> Tuple[torch.Tensor, torch.Tensor]:
    x_hat, log_det_J, log_prob_z, z0, mu, std = model(x)
    loss = loss_function(x_hat, x, z0, mu, std, log_det_J, log_prob_z)
    return x_hat, loss


inference_and_loss_functions = {
    "vae-gauss-base": inference_and_loss_vae_gauss,
    "vae-gauss-big": inference_and_loss_vae_gauss,
    "vae-gauss-base-sigm": inference_and_loss_vae_gauss,
    "vae-gauss-sigm-big": inference_and_loss_vae_gauss,
    "vae-realnvp-base": inference_and_loss_vae_realnvp,
    "vae-realnvp-base-jt": inference_and_loss_vae_realnvp_jt,
    "vae-realnvp-base-jt-4flows": inference_and_loss_vae_realnvp_jt,
    "vae-realnvp-base-jt-8flows": inference_and_loss_vae_realnvp_jt,
    "vae-realnvp-base-jt-4flows2": inference_and_loss_vae_realnvp_jt2,
    "vae-realnvp-base-jt-8flows2": inference_and_loss_vae_realnvp_jt2,
}


def train(
    epochs,
    batch_size,
    model_name: str,
    dataset: str,
    criterion: str,
    lr=0.001,
    cuda=True,
    log_interval=10,
    summary_only=False,
):
    results_dir = pathlib.Path(f"{model_name}_{dataset}_{criterion}")
    results_dir.mkdir(exist_ok=True)
    logging.info(f"Results dir: {results_dir}")

    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logging.info(f"Device: {device}")

    logging.info(f"Loading dataset: {dataset}")
    data_loading_function = data_factory(dataset)
    trainloader, valloader, testloader = data_loading_function(batch_size, cuda)

    x0, _ = next(iter(trainloader))
    b, c, h, w = x0.shape

    logging.info(f"Creating model: {model_name}")
    model = models_factory(model_name)(in_channels=c, img_h=h, img_w=w)

    logging.info(summary(model, torch.zeros((b, c, h, w)), show_input=False, show_hierarchical=True))
    if summary_only:
        return

    logging.debug(f"Sending model to: {device}")
    model = model.to(device)
    model.train()

    inference_and_loss = inference_and_loss_functions[model_name]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = criterions_factory(criterion)
    data = list()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        logging.info(f"Epoch {epoch}")
        batch_idx = 0
        for batch_idx, (x, _) in enumerate(trainloader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, loss = inference_and_loss(model, loss_function, x)
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
        model.eval()
        with torch.no_grad():
            test_loss = test(
                inference_and_loss, model, valloader, device, epoch, results_dir, loss_function, batch_size
            )
            data.append({"epoch": epoch, "train_loss": train_loss, "val_loss": test_loss})
            sample(model, 16, results_dir, epoch)
    model_file = results_dir / "model.pt"
    progress_file = results_dir / "progress.csv"

    logging.info(f"Saving final model to: {model_file}")
    torch.save(model, model_file)
    pd.DataFrame(data).to_csv(progress_file)


if __name__ == "__main__":
    # train(30, 128, "vae-gauss-base", "mnist", "elbo", lr=1e-3, log_interval=50, cuda=True)
    # train(30, 128, "vae-realnvp-base-jt-4flows", "mnist", "elbo_flows", lr=1e-3, log_interval=50, cuda=True)

    train(30, 128, "vae-gauss-base", "svhn", "elbo", lr=1e-3, log_interval=50, cuda=True)
    train(30, 128, "vae-realnvp-base-jt-4flows", "svhn", "elbo_flows", lr=1e-3, log_interval=50, cuda=True)

    train(30, 128, "vae-gauss-base-sigm", "mnist", "elbo", lr=1e-3, log_interval=50, cuda=True)
    train(30, 128, "vae-gauss-big-sigm", "svhn", "elbo", lr=1e-3, log_interval=50, cuda=True)

    # train(30, 128, "vae-gauss-base", "mnist", "elbo", lr=1e-3, log_interval=50, cuda=True)
    # train(30, 64, "vae-gauss-big", "svhn", "elbo", lr=1e-3, log_interval=50, cuda=True)

    # # train(30, 128, "vae-realnvp-base", "mnist", "nll", lr=1e-3, log_interval=50, cuda=True)
    # # train(30, 128, "vae-realnvp-base", "svhn", "nll", lr=1e-3, log_interval=50, cuda=True)

    # train(30, 128, "vae-realnvp-base-jt", "mnist", "elbo_flows", lr=1e-3, log_interval=50, cuda=True)
    # train(30, 128, "vae-realnvp-base-jt-4flows", "mnist", "elbo_flows", lr=1e-3, log_interval=50, cuda=True)
    # train(30, 128, "vae-realnvp-base-jt-4flows2", "mnist", "elbo_flows2", lr=1e-3, log_interval=50, cuda=True)

    # train(30, 128, "vae-realnvp-base-jt-8flows", "mnist", "elbo_flows", lr=1e-3, log_interval=50, cuda=True)
    # train(30, 128, "vae-realnvp-base-jt", "svhn", "elbo_flows", lr=1e-3, log_interval=50, cuda=True)
    # train(30, 128, "vae-realnvp-base-jt-8flows", "svhn", "elbo_flows", lr=1e-3, log_interval=50, cuda=True)
