import torch

from fid import FID
from data import RescaleTensor
from data import data_factory
from models import AbstractSamplingModel
from torch.utils.data import Dataset, DataLoader


class VAESamples(Dataset):
    def __init__(self, model: AbstractSamplingModel, total: int, preload_batch_size: int):
        self.model = model
        self.transform = RescaleTensor((3, 299, 299))
        self.total = total
        self.preload_batch_size = preload_batch_size
        self.samples_buffer = []
        self.samples_buffer_iter = iter([])

    def __len__(self):
        return self.total

    def __preload_samples(self):
        """ Dataset must return only 1 image, but VAE works best with batches, so we pre-genearate some images and then return the one by one"""
        self.samples_buffer = self.model.sample(n=self.preload_batch_size)
        self.samples_buffer_iter = iter(self.samples_buffer)

    def __getitem__(self, idx):
        try:
            sample = next(self.samples_buffer_iter)

        except StopIteration:
            self.__preload_samples()
            sample = next(self.samples_buffer_iter)
        c, h, w = sample.shape
        sample = sample.view(1, c, h, w)
        sample = sample.squeeze(0)
        mock_y = torch.Tensor([1, 2, 3])
        return (sample, mock_y)


class VAESamples2Inception(Dataset):
    def __init__(self, model: AbstractSamplingModel, total: int, preload_batch_size: int):
        self.model = model
        self.transform = RescaleTensor((3, 299, 299))
        self.total = total
        self.preload_batch_size = preload_batch_size
        self.samples_buffer = []
        self.samples_buffer_iter = iter([])

    def __len__(self):
        return self.total

    def __preload_samples(self):
        """ Dataset must return only 1 image, but VAE works best with batches, so we pre-genearate some images and then return the one by one"""
        self.samples_buffer = self.model.sample(n=self.preload_batch_size)
        self.samples_buffer_iter = iter(self.samples_buffer)

    def __getitem__(self, idx):
        try:
            sample = next(self.samples_buffer_iter)

        except StopIteration:
            self.__preload_samples()
            sample = next(self.samples_buffer_iter)
        sample = sample.view(1, 1, 28, 28)
        sample = self.transform(sample)
        sample = sample.squeeze(0)
        mock_y = torch.Tensor([1, 2, 3])
        return (sample, mock_y)


def evaluate_with_fid(model: AbstractSamplingModel, dataset: str, batch_size: int, device: torch.device) -> float:
    _, testloader = data_factory(dataset)(batch_size=batch_size, cuda=False)

    model = model.to(device)
    vae_data = VAESamples2Inception(model, total=len(testloader.dataset), preload_batch_size=batch_size)
    samples_loader = DataLoader(vae_data, batch_size=batch_size, num_workers=0)

    fid = FID().to(device)
    return fid.score_batched(samples_loader, testloader)
