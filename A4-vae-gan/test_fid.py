import torch
import pytest
import numpy as np

from torch.utils.data import DataLoader

from fid import FID
from test_models import BATCHES

cuda_opts = [True, False] if torch.cuda.is_available() else [False]


@pytest.mark.parametrize("dataset_size", [2, 5, 64])
@pytest.mark.parametrize("cuda", cuda_opts)
def test_fid_encode(dataset_size, cuda):
    """ Test encding with InceptionV3 model """
    device = torch.device("cuda" if cuda else "cpu")
    x = torch.randn(dataset_size, 3, 299, 299)
    fid = FID().to(device)
    features = fid.encode(x)
    assert features.shape == (dataset_size, 2048)


@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("dataset_size", [2, 5, 64])
@pytest.mark.parametrize("cuda", cuda_opts)
def test_fid_encode_batched(batch_size, dataset_size, cuda):
    """ Test encding with InceptionV3 model """
    device = torch.device("cuda" if cuda else "cpu")

    x = torch.randn(dataset_size, 3, 299, 299)
    y = torch.randn(dataset_size, 3)
    dataset = list(zip(x, y))

    dataloader = DataLoader(dataset, batch_size=batch_size)
    fid = FID().to(device)
    features = fid.batched_encode(dataloader)
    assert features.shape == (dataset_size, 2048)


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("cuda", cuda_opts)
def test_fid_lower_for_same(batch_size, cuda):
    device = torch.device("cuda" if cuda else "cpu")
    x1 = torch.randn(batch_size, 3, 299, 299).to(device)
    x2 = torch.randn(batch_size, 3, 299, 299).to(device)

    fid = FID().to(device)
    fid_same = fid.score(x1, x1)
    fid_different = fid.score(x1, x2)

    assert fid_same < fid_different


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_fid_symmetric(batch_size):
    x1 = torch.randn(batch_size, 3, 299, 299)
    x2 = torch.randn(batch_size, 3, 299, 299)

    fid = FID()

    fid1 = fid.score(x1, x2)
    fid2 = fid.score(x2, x1)

    assert np.isclose(fid1, fid2)
