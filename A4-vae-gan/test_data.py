import pytest
import torch
from data import load_inceptionv3_mnist


@pytest.mark.parametrize("test_or_train,idx", [("train", 0), ("test", 1)])
def test_rescaling_mnits_to_match_inceptionv3_correct_shape(test_or_train, idx):
    expected_shape = (3, 32, 32)
    loader = load_inceptionv3_mnist(batch_size=32, cuda=False)[idx]

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
