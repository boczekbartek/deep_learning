import torch
import torch.nn as nn
import torch.utils.data
from torchvision.models.inception import inception_v3
from data import Rescale
from tqdm import tqdm
import numpy as np
from scipy import linalg
import logging


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

    def _score_fid(self, enc1: torch.Tensor, enc2: torch.Tensor, eps=1e-6) -> float:
        enc1 = enc1.cpu().numpy()
        enc2 = enc2.cpu().numpy()

        mu1 = enc1.mean(axis=0)
        mu2 = enc2.mean(axis=0)

        sigma1 = np.cov(enc1, rowvar=False)
        sigma2 = np.cov(enc2, rowvar=False)

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            logging.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            covmean = torch.sqrt(torch.matmul(sigma1, sigma2))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def forward(self, x1, x2):
        return self.score(x1, x2)
