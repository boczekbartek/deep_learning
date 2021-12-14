import pytest
import torch

from typing import Tuple

from torch._C import PyTorchFileReader

from models import VAEGauss
from data import load_mnist

BATCHES = [1, 2, 4, 16, 32]


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("latent_dim", [128])
def test_vae_gauss_mnist(batch_size, latent_dim):
    train_loader, test_loader = load_mnist(batch_size=batch_size, cuda=False)
    x0, _ = next(iter(train_loader))
    b, ch, h, w = x0.shape
    model = VAEGauss(in_channels=ch, latent_dim=latent_dim, img_h=h, img_w=w)

    gen_x = model.forward(x0)
    assert gen_x.shape == x0.shape


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("img_h, img_w", [(28, 28), (32, 32)])
@pytest.mark.parametrize("latent_dim", [128])
def test_vae_gauss_encoder(batch_size: int, in_channels: int, img_h: int, img_w: int, latent_dim: int):
    model = VAEGauss(in_channels=in_channels, latent_dim=latent_dim, img_h=img_h, img_w=img_w)

    # Generate random input batch
    x = torch.randn(batch_size, in_channels, img_h, img_w)

    z = model.encode(x)
    assert z.shape == (batch_size, latent_dim)


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("img_h, img_w", [(28, 28), (32, 32)])
@pytest.mark.parametrize("latent_dim", [128])
def test_vae_gauss_decoder(batch_size: int, in_channels: int, img_h: int, img_w: int, latent_dim: int):
    model = VAEGauss(in_channels=in_channels, latent_dim=latent_dim, img_h=img_h, img_w=img_w)

    # Generate random latent vector
    z = torch.randn(batch_size, latent_dim)

    x_gen = model.decode(z)

    assert x_gen.shape == (batch_size, in_channels, img_h, img_w)
