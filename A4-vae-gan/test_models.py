import pytest
import torch

from models import VAEGaussBase
from data import load_mnist

BATCHES = [1, 2, 4, 16, 32]


@pytest.mark.parametrize("batch_size", BATCHES)
def test_vae_gauss_base_mnist(batch_size):
    train_loader, test_loader = load_mnist(batch_size=batch_size, cuda=False)
    x0, _ = next(iter(train_loader))
    b, ch, h, w = x0.shape
    model = VAEGaussBase(in_channels=ch, img_h=h, img_w=w)

    x_hat, mu, std, z = model.forward(x0)
    assert x_hat.shape == x0.shape


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("img_h, img_w", [(28, 28), (32, 32)])
def test_vae_gauss_encoder(batch_size: int, in_channels: int, img_h: int, img_w: int):
    model = VAEGaussBase(in_channels=in_channels, img_h=img_h, img_w=img_w)
    assert model.latent_dim == 128
    # Generate random input batch
    x = torch.randn(batch_size, in_channels, img_h, img_w)

    mu, logvar = model.encode(x)
    assert mu.shape == (batch_size, model.latent_dim)
    assert logvar.shape == (batch_size, model.latent_dim)


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("img_h, img_w", [(28, 28), (32, 32)])
def test_vae_gauss_decoder(batch_size: int, in_channels: int, img_h: int, img_w: int):
    model = VAEGaussBase(in_channels=in_channels, img_h=img_h, img_w=img_w)

    # Generate random latent vector
    z = torch.randn(batch_size, model.latent_dim)

    x_gen = model.decode(z)

    assert x_gen.shape == (batch_size, in_channels, img_h, img_w)


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("latent_dim", [1, 5, 128, 256])
def test_vae_gauss_reparametrize(batch_size, latent_dim):
    mu = torch.randn(batch_size, latent_dim)
    logvar = torch.randn(batch_size, latent_dim)

    z, std = VAEGaussBase.reparameterize(mu, logvar)

    assert z.shape == (batch_size, latent_dim)
    assert std.shape == (batch_size, latent_dim)
