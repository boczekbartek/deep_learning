import logging
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractclassmethod


class AbstractSamplingModel(ABC):
    @abstractclassmethod
    def sample(self, n: int) -> torch.Tensor:
        """ Sample n samples from the model """
        pass


class VAEGauss(AbstractSamplingModel, nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, img_h: int, img_w: int) -> None:
        """Convolutional Variational Autoencoder with Gaussian prior

        Args:
            in_channels (int): Number of channels in the input image
            latent_dim (int): Size of each parameter of the latent space. [0:half] is mu, [half:] is log_std.
            img_h (int): Height of input image
            img_w (int): Width of input image
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Image size
        self.img_h = img_h
        self.img_w = img_w

        # Number of filters in conv layers
        self.conv1_n_fil = 32
        self.conv2_n_fil = 64

        self.conv_kernel_size = 3
        self.max_pool_kernel_size = 2

        # Encoder layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=self.conv_kernel_size, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.conv_kernel_size, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=self.max_pool_kernel_size)
        self.drop1 = nn.Dropout(0.25)

        # TODO it also depends on stride
        fc_size = self.calculate_fc_size_encoder(
            self.img_h, self.img_w, self.conv_kernel_size, self.conv2_n_fil, self.max_pool_kernel_size
        )
        self.lin1 = nn.Linear(fc_size, 128)
        self.drop2 = nn.Dropout(0.5)
        self.lin2_1 = nn.Linear(128, latent_dim)
        self.lin2_2 = nn.Linear(128, latent_dim)

        # Decoder layers
        self.lin3 = nn.Linear(latent_dim, 128)
        self.drop3 = nn.Dropout(0.5)
        fc_size = self.calculate_fc_size_decoder(self.img_h, self.img_w, self.conv_kernel_size, self.conv2_n_fil)
        self.lin4 = nn.Linear(128, fc_size)
        self.drop4 = nn.Dropout(0.25)
        self.dconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.conv_kernel_size, stride=1)
        self.dconv2 = nn.ConvTranspose2d(
            in_channels=32, out_channels=in_channels, kernel_size=self.conv_kernel_size, stride=1
        )

    @classmethod
    def calculate_fc_size_encoder(cls, img_h, img_w, conv_kernel_size, conv2_filters, max_pool_kernel):
        return cls.calculate_fc_size_decoder(img_h, img_w, conv_kernel_size, conv2_filters) // (2 * max_pool_kernel)

    @staticmethod
    def calculate_fc_size_decoder(img_h, img_w, conv_kernel_size, conv2_filters):
        conv1_size_reduction = conv_kernel_size - 1
        conv2_size_reduction = conv_kernel_size - 1
        total_conv_reduction = conv1_size_reduction + conv2_size_reduction

        h = img_h - total_conv_reduction
        w = img_w - total_conv_reduction

        return conv2_filters * h * w

    def forward(self, x):
        mu, logvar = self.encode(x)

        z, std = self.reparameterize(mu, logvar)

        x_hat = self.decode(z)

        return x_hat, mu, std, z

    def encode(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.drop1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.drop2(x)
        mu = self.lin2_1(x)
        logvar = self.lin2_2(x)
        return mu, logvar

    def decode(self, x):
        x = self.lin3(x)
        x = F.relu(x)
        x = self.drop3(x)
        x = self.lin4(x)
        x = F.relu(x)
        x = self.drop4(x)

        conv1_size_reduction = self.conv_kernel_size - 1
        conv2_size_reduction = self.conv_kernel_size - 1
        total_reduction = conv1_size_reduction + conv2_size_reduction

        x = x.view(-1, self.conv2_n_fil, (self.img_h - total_reduction), (self.img_w - total_reduction))
        x = self.dconv1(x)
        x = F.relu(x)
        x = self.dconv2(x)
        return x

    @staticmethod
    def reparameterize(mu, logvar) -> Tuple[torch.Tensor, torch.Tensor]:
        std = torch.exp(logvar)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z, std

    @torch.no_grad()
    def sample(self, n):
        device = next(self.parameters()).device

        mu = torch.randn(n, self.latent_dim).to(device)
        logvar = torch.randn(n, self.latent_dim).to(device)
        z, _ = self.reparameterize(mu, logvar)
        return self.decode(z)


class VAEGaussSigm(VAEGauss):
    def decode(self, x):
        x = super().decode(x)
        x = torch.sigmoid(x)
        return x


class VAEGaussBig(VAEGauss):
    def __init__(self, in_channels: int, latent_dim: int, img_h: int, img_w: int) -> None:
        super().__init__(in_channels, latent_dim, img_h, img_w)
        self.latent_dim = latent_dim

        # Image size
        self.img_h = img_h
        self.img_w = img_w

        # Number of filters in conv layers
        self.conv1_n_fil = 128
        self.conv2_n_fil = 256

        self.conv_kernel_size = 3
        self.max_pool_kernel_size = 2

        # Encoder layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=self.conv1_n_fil, kernel_size=self.conv_kernel_size, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1_n_fil, out_channels=self.conv2_n_fil, kernel_size=self.conv_kernel_size, stride=1
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=self.max_pool_kernel_size)
        self.drop1 = nn.Dropout(0.25)

        # TODO it also depends on stride
        fc_size = self.calculate_fc_size_encoder(
            self.img_h, self.img_w, self.conv_kernel_size, self.conv2_n_fil, self.max_pool_kernel_size
        )
        self.lin1 = nn.Linear(fc_size, 512)
        self.drop2 = nn.Dropout(0.5)
        self.lin2_1 = nn.Linear(512, latent_dim)
        self.lin2_2 = nn.Linear(512, latent_dim)

        # Decoder layers
        self.lin3 = nn.Linear(latent_dim, 512)
        self.drop3 = nn.Dropout(0.5)
        fc_size = self.calculate_fc_size_decoder(self.img_h, self.img_w, self.conv_kernel_size, self.conv2_n_fil)
        self.lin4 = nn.Linear(512, fc_size)
        self.drop4 = nn.Dropout(0.25)
        self.dconv1 = nn.ConvTranspose2d(
            in_channels=self.conv2_n_fil, out_channels=self.conv1_n_fil, kernel_size=self.conv_kernel_size, stride=1
        )
        self.dconv2 = nn.ConvTranspose2d(
            in_channels=self.conv1_n_fil, out_channels=in_channels, kernel_size=self.conv_kernel_size, stride=1
        )


class VAEGaussSigmBig(VAEGaussBig):
    def decode(self, x):
        x = super().decode(x)
        x = torch.sigmoid(x)
        return x


def models_factory(name):
    return {
        "vae-gauss": VAEGauss,
        "vae-gauss-big": VAEGaussBig,
        "vae-gauss-sigm": VAEGaussSigm,
        "vae-gauss-sigm-big": VAEGaussSigmBig,
    }[name]
