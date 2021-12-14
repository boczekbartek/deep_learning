import torch
import torch.distributions
import torch.nn.functional as F

from utils import log_normal


def elbo(
    x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor
) -> torch.Tensor():
    # reconstruction error
    RE = F.mse_loss(x, x_hat)
    # KL = encoder - marginal prior
    KL = log_normal(z, mu, log_std) - log_normal(
        z, torch.zeros_like(mu), torch.ones_like(log_std)
    )  # TODO KL slides way
    # KL = -0.5 * torch.sum(1 + log_std - mu.pow(2) - log_std.exp())

    # REMEMBER! We maximize ELBO, but optimizers minimize.
    # Therefore, we need to take the negative sign!
    return -(RE - KL).sum()
