import torch
import torch.nn as nn


class RadialFlow(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.z0 = nn.Parameter(torch.empty(latent_dim))
        self.log_alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))
        self.latent_dim = latent_dim

        nn.init.normal_(self.z0)
        nn.init.normal_(self.log_alpha)
        nn.init.normal_(self.beta)

    def forward(self, z: torch.Tensor):
        z_sub = z - self.z0
        alpha = torch.exp(self.log_alpha)
        r = torch.norm(z_sub)
        h = 1 / (alpha + r)
        f = z + self.beta * h * z_sub
        log_det = (self.latent_dim - 1) * torch.log(1 + self.beta * h) + torch.log(
            1 + self.beta * h + self.beta - self.beta * r / (alpha + r) ** 2
        )

        return f, log_det
