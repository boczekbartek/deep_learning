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


def elbo_flows(x_hat, x, log_prob_z, log_det_J):
    RE = F.mse_loss(x, x_hat, reduction="sum")
    KL = -(log_prob_z + log_det_J).sum()
    return RE + KL


def elbo_flows2(x_hat, x, z, mu, std, log_prob_z, log_det_J):
    RE = F.mse_loss(x, x_hat, reduction="sum")
    KL1 = log_normal(z, mu, std) - log_normal(z, 0, 1)

    KL2 = -(log_prob_z + log_det_J)
    return RE + KL1.sum() + KL2.sum()


def loss_function(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor):
    BCE = F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def nll_flow_loss(
    x_hat: torch.Tensor, x: torch.Tensor, log_prob_z0: torch.Tensor, log_prob_zk: torch.Tensor, log_det: torch.Tensor
):
    RE = F.mse_loss(x, x_hat, reduction="sum")

    loss = log_prob_z0.mean() + RE - log_prob_zk.mean() - log_det.mean()

    return loss


def criterions_factory(name):
    return {
        "elbo": elbo,
        "torch": loss_function,
        "nll": nll_flow_loss,
        "elbo_flows": elbo_flows,
        "elbo_flows2": elbo_flows2,
    }[name]
