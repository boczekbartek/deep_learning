from fid import FID
import torch
import pytest

from test_models import BATCHES

cuda_opts = [True, False] if torch.cuda.is_available() else [False]


@pytest.mark.parametrize("batch_size", BATCHES)
def test_fid_encode(batch_size):
    """ Test encding with InceptionV3 model """
    x = torch.randn(batch_size, 3, 299, 299)
    fid = FID()
    features = fid.encode(x)
    assert features.shape == (batch_size, 2048)


@pytest.mark.parametrize("batch_size", BATCHES)
@pytest.mark.parametrize("cuda", cuda_opts)
def test_fid_lower_for_same(batch_size, cuda):
    device = torch.device("cuda" if cuda else "cpu")
    x1 = torch.randn(batch_size, 3, 299, 299).to(device)
    x2 = torch.randn(batch_size, 3, 299, 299).to(device)

    fid = FID().to(device)
    fid_same = fid.score(x1, x1).cpu()
    fid_different = fid.score(x1, x2).cpu()

    assert fid_same < fid_different
