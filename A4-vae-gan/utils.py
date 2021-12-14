import torch


def log_normal(z: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor):
    mu = z + mu
    dist = torch.distributions.LogNormal(mu, torch.pow(log_std, 2))
    return dist.sample()
