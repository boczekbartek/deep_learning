import pytest

from models import VAEGauss
from data import load_mnist


@pytest.mark.parametrize("batch_size", [1, 2, 4, 16, 32])
def test_vau_gauss(batch_size):
    train_loader, test_loader = load_mnist(batch_size=batch_size, cuda=False)
    x0, _ = next(iter(train_loader))
    model = VAEGauss(in_channels=1, latent_dim=128)

    gen_x = model.forward(x0)
    assert gen_x.shape == x0.shape
