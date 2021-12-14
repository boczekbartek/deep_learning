import pytest
import torch

from typing import Tuple

from models import VAEGauss
from data import load_mnist


@pytest.mark.parametrize("batch_size", [1, 2, 4, 16, 32])
@pytest.mark.xfail
def test_vae_gauss(batch_size):
    train_loader, test_loader = load_mnist(batch_size=batch_size, cuda=False)
    x0, _ = next(iter(train_loader))
    model = VAEGauss(in_channels=1, latent_dim=128)

    gen_x = model.forward(x0)
    assert gen_x.shape == x0.shape


@pytest.mark.parametrize("batch_size", [1, 2, 4, 16, 32])
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("img_h, img_w", [(28, 28), (32, 32)])
@pytest.mark.parametrize("latent_dim", [128])
def test_vae_gauss_encoder(batch_size: int, in_channels: int, img_h: int, img_w: int, latent_dim: int):
    model = VAEGauss(in_channels=in_channels, latent_dim=latent_dim, img_h=img_h, img_w=img_w)

    # Generate random input batch
    x = torch.randn(batch_size, in_channels, img_h, img_w)

    expected_shape = (batch_size, latent_dim)
    z = model.encode(x)
    assert z.shape == expected_shape
