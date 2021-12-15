import torch
import torch.nn as nn

from torchvision.models.inception import inception_v3
from data import Rescale


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

    def encode(self, x):
        return self.inception(x)

    def score(self, x1: torch.Tensor, x2: torch.Tensor):
        with torch.no_grad():
            enc1 = self.encode(x1)
            enc2 = self.encode(x2)

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

        return fid

    def forward(self, x1, x2):
        return self.score(x1, x2)
