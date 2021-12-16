from models import VAERealNvpRadialBase, VAERealNvpJTBase
from data import load_mnist
import pytest
import torch

from test_models import BATCHES
from test_data import cuda_opts


@pytest.mark.parametrize("batch_size", BATCHES)
def test_vae_gauss_base_mnist(batch_size):
    train_loader, test_loader = load_mnist(batch_size=batch_size, cuda=False)
    x0, _ = next(iter(train_loader))
    b, ch, h, w = x0.shape
    model = VAERealNvpRadialBase(in_channels=ch, img_h=h, img_w=w)

    x_hat, mu, std, z = model.forward(x0)
    assert x_hat.shape == x0.shape


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("cuda", cuda_opts)
@pytest.mark.parametrize("n_flows", [1, 2, 3, 4, 5])
def test_vae_realnvp_jt_base_mnist(batch_size, cuda, n_flows):
    device = torch.device("cuda" if cuda else "cpu")

    train_loader, test_loader = load_mnist(batch_size=batch_size, cuda=cuda)
    x0, _ = next(iter(train_loader))
    b, ch, h, w = x0.shape
    model = VAERealNvpJTBase(in_channels=ch, img_h=h, img_w=w, n_flows=n_flows).to(device)

    x_hat, log_det_J, log_prob_z = model.forward(x0.to(device))
    assert x_hat.shape == x0.shape
