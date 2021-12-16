import pytest
import torch
from data import load_inceptionv3_mnist, load_mnist, load_svhn, load_inceptionv3_svhn
from test_models import BATCHES


cuda_opts = [True, False] if torch.cuda.is_available() else [False]


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("cuda", cuda_opts)
def test_loading_mnist(batch_size, cuda):
    trainloader, testloader = load_mnist(batch_size, cuda)
    x0, _ = next(iter(trainloader))
    assert x0.shape == (batch_size, 1, 28, 28)


@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("cuda", cuda_opts)
def test_loading_svhn(batch_size, cuda):
    trainloader, testloader = load_svhn(batch_size, cuda)
    x0, _ = next(iter(trainloader))
    assert x0.shape == (batch_size, 3, 32, 32)


@pytest.mark.parametrize("test_or_train,idx", [("train", 0), ("test", 1)])
def test_rescaling_mnist_to_match_inceptionv3_correct_shape(test_or_train, idx):
    expected_shape = (3, 299, 299)
    loader = load_inceptionv3_mnist(batch_size=32, cuda=False)[idx]

    for (data, _) in loader:
        for image in data:
            assert image.shape == expected_shape, f"{test_or_train} | Wrong shape in train image: {image.shape}"
        break


@pytest.mark.parametrize("test_or_train,idx", [("train", 0), ("test", 1)])
def test_rescaling_svhn_to_match_inceptionv3_correct_shape(test_or_train, idx):
    expected_shape = (3, 299, 299)
    loader = load_inceptionv3_svhn(batch_size=4, cuda=False)[idx]

    for (data, _) in loader:
        for image in data:
            assert image.shape == expected_shape, f"{test_or_train} | Wrong shape in train image: {image.shape}"
        break


@pytest.mark.parametrize("test_or_train,idx", [("train", 0), ("test", 1)])
def test_rescaling_mnits_to_match_inceptionv3_all_channels_same(test_or_train, idx):
    expected_num_channels = 3
    loader = load_inceptionv3_mnist(batch_size=32, cuda=False)[idx]

    for (data, _) in loader:
        for image in data:
            assert (
                image.shape[0] == expected_num_channels
            ), f"{test_or_train} | Too few channels. Image should be in shape (3,h,w)"
            ch0 = image[0, :, :]
            for ch in range(1, expected_num_channels):
                assert torch.equal(ch0, image[ch, :, :]), f"Channel {ch} is different than channel 0"
        break
