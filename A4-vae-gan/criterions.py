import torch
import torch.distributions
import torch.nn.functional as F

from utils import log_normal


def log_normal(z, mu, std):
    n = torch.distributions.Normal(mu, std)
    return n.log_prob(z)


def elbo(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, z: torch.Tensor) -> torch.Tensor():
    # reconstruction error
    RE = F.mse_loss(x, x_hat, reduction="sum")

    # Kullback-Leibler divergence
    KL = log_normal(z, mu, std) - log_normal(z, 0, 1)

    loss = RE + KL.sum()
    return loss


def loss_function(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor):
    BCE = F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def criterions_factory(name):
    return {"elbo": elbo, "torch": loss_function}[name]
