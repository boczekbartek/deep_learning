import torch

from fid import FID
from data import RescaleTensor
from data import data_factory
from models import AbstractSamplingModel


def evaluate_with_fid(model: AbstractSamplingModel, dataset: str, n_test_samples: int, device: torch.device) -> float:
    _, testloader = data_factory(dataset)(batch_size=n_test_samples, cuda=False)
    test_data, _ = next(iter(testloader))
    rescaler = RescaleTensor((3, 299, 299))

    samples = model.sample(n=n_test_samples).cpu()
    samples = rescaler(samples)

    fid = FID().to(device)
    return float(fid.score(test_data, samples).cpu())
