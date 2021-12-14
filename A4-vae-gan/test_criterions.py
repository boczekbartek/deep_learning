import pytest
import torch
import pytest

from criterions import elbo
from test_models import BATCHES


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("img_h, img_w", [(28, 28), (32, 32)])
@pytest.mark.parametrize("latent_dim", [128])
def test_elbo_works(batch_size: int, img_h: int, img_w: int, latent_dim: int):
    x = torch.randn(batch_size, img_h, img_w)
    x_hat = torch.randn(batch_size, img_h, img_w)

    z = torch.randn(batch_size, latent_dim)

    mu = torch.randn(batch_size, latent_dim) ** 2
    std = torch.randn(batch_size, latent_dim)

    loss = elbo(x, x_hat, z, mu, std)
    float(loss)


# TODO figure out if this test makes sense
@pytest.mark.xfail
@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("img_h, img_w", [(28, 28), (32, 32)])
@pytest.mark.parametrize("latent_dim", [128])
def test_elbo_lower_for_same_than_for_different(batch_size: int, img_h: int, img_w: int, latent_dim: int):
    x = torch.randn(batch_size, img_h, img_w)

    x_diff = torch.randn(batch_size, img_h, img_w)
    x_same = x.clone()

    z = torch.randn(batch_size, latent_dim)
    # Disable KL factor
    mu = torch.randn(batch_size, latent_dim) ** 2
    std = torch.randn(batch_size, latent_dim)

    elbo_same = elbo(x, x_same, z, mu, std)
    elbo_diff = elbo(x, x_diff, z, mu, std)
    assert float(elbo_same) < float(elbo_diff)
