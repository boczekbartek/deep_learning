import pytest
import torch

import torch
from test_models import BATCHES
from data import RescaleTensor


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("img_h, img_w", [(28, 28), (32, 32)])
@pytest.mark.parametrize("out_shape", [(3, 299, 299), (3, 32, 32)])
def test_rescale_tensor(batch_size, channels, img_h, img_w, out_shape):
    tensor = torch.randn(batch_size, channels, img_h, img_w)
    transform = RescaleTensor(out_shape)
    out = transform(tensor)
    assert out.shape == (batch_size, *out_shape)


@pytest.mark.parametrize("channels", [2, 4, 5, 6])
@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("img_h, img_w", [(28, 28), (32, 32)])
@pytest.mark.parametrize("out_shape", [(3, 299, 299), (3, 32, 32)])
def test_rescale_raises_assert_on_wrong_channels(channels, batch_size, img_h, img_w, out_shape):
    """ Rescaling should raise assertion error when wrong number of channels is passed """
    assert (
        channels not in RescaleTensor.compatible_channels
    ), f"Testing compatible number of channels but expecting to fail - fix test params"

    tensor = torch.randn(batch_size, channels, img_h, img_w)
    transform = RescaleTensor(out_shape)
    with pytest.raises(AssertionError):
        out = transform(tensor)
