import torch
import torch.nn as nn
import torch.utils.data
from torchvision.models.inception import inception_v3
from data import Rescale
from tqdm import tqdm


def rescale(x):
    res = Rescale((3, 299, 299))
    return res(x)


class FID(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inception = inception_v3(pretrained=True)

        # Disable last (softmax) layer in the incpetion model
        self.inception.fc = torch.nn.Identity()
        self.inception.eval()

    def get_device(self) -> torch.device:
        return next(self.inception.parameters()).device

    @torch.no_grad()
    def encode(self, x):
        x = x.to(self.get_device())
        return self.inception(x)

    @torch.no_grad()
    def batched_encode(self, xloader):
        result = torch.empty(len(xloader.dataset), 2048)

        i = 0
        for x, _ in tqdm(xloader, desc="InceptionV3 Encoding"):
            x = x.to(self.get_device())
            enc = self.inception(x).cpu()
            result[i : i + len(x), :] = enc
            i += len(x)
        return result

    def score_batched(self, x1_loader: torch.utils.data.DataLoader, x2_loader: torch.utils.data.DataLoader) -> float:
        """ Allowes to score much bigger datasets. Is limited by available RAM - 2 covariance matrices of size len(x1_loader) x len(x2_loader) will be created """
        enc1 = self.batched_encode(x1_loader)
        enc2 = self.batched_encode(x2_loader)
        return self._score_fid(enc1, enc2)

    def score(self, x1: torch.Tensor, x2: torch.Tensor):
        """ Simple scoring function - if model is on cuda - limited size of x1 and x2 is usable, because it must fit in the GPU memory """
        enc1 = self.encode(x1)
        enc2 = self.encode(x2)
        return self._score_fid(enc1, enc2)

    def _score_fid(self, enc1: torch.Tensor, enc2: torch.Tensor) -> float:
        mu1 = enc1.mean(axis=0)
        mu2 = enc2.mean(axis=0)

        sigma1 = torch.cov(enc1)
        sigma2 = torch.cov(enc2)

        if sigma1.shape == ():
            sigma1 = sigma1.view(1,)

        if sigma2.shape == ():
            sigma2 = sigma2.view(1,)

        covmean = torch.sqrt(torch.matmul(sigma1, sigma2))

        if covmean.shape == ():
            p2 = sigma1 + sigma2 - 2.0 * covmean
        else:
            p2 = torch.trace(sigma1 + sigma2 - 2.0 * covmean)

        fid = torch.sum((mu1 - mu2) ** 2.0) + p2

        return fid.item()

    def forward(self, x1, x2):
        return self.score(x1, x2)
