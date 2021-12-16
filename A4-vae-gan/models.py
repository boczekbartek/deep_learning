import logging
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty


class AbstractSamplingModel(nn.Module):
    def __init__(self) -> None:
        super(AbstractSamplingModel, self).__init__()

    @abstractclassmethod
    def sample(self, n: int) -> torch.Tensor:
        """ Sample n samples from the model """
        pass


class AbstractVAE(AbstractSamplingModel):
    def __init__(self) -> None:
        super(AbstractVAE, self).__init__()

    @abstractproperty
    def latent_dim(cls):
        pass

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    @abstractmethod
    def reparameterize(self, x):
        pass

    @torch.no_grad()
    def sample(self, n):
        device = next(self.parameters()).device

        mu = torch.randn(n, self.latent_dim).to(device)
        logvar = torch.randn(n, self.latent_dim).to(device)
        z, _ = self.reparameterize(mu, logvar)
        return self.decode(z)


def calculate_fc_size_decoder(img_h, img_w, conv_kernel_size, conv2_filters):
    # TODO it also depends on stride
    conv1_size_reduction = conv_kernel_size - 1
    conv2_size_reduction = conv_kernel_size - 1
    total_conv_reduction = conv1_size_reduction + conv2_size_reduction

    h = img_h - total_conv_reduction
    w = img_w - total_conv_reduction

    return conv2_filters * h * w


def calculate_fc_size_encoder(img_h, img_w, conv_kernel_size, conv2_filters, max_pool_kernel):
    return calculate_fc_size_decoder(img_h, img_w, conv_kernel_size, conv2_filters) // (2 * max_pool_kernel)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_h: int,
        img_w: int,
        out_size: int = 128,
        n_fil_1: int = 64,
        n_fil_2: int = 32,
        kernel_size: int = 3,
        mp_size: int = 2,
    ) -> None:
        super().__init__()
        # Image size
        self.img_h = img_h
        self.img_w = img_w

        # Number of filters in conv layers
        self.conv1_n_fil = n_fil_1
        self.conv2_n_fil = n_fil_2

        self.conv_kernel_size = kernel_size
        self.max_pool_kernel_size = mp_size

        # Encoder layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=self.conv1_n_fil, kernel_size=self.conv_kernel_size, stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1_n_fil, out_channels=self.conv2_n_fil, kernel_size=self.conv_kernel_size, stride=1
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=self.max_pool_kernel_size)
        self.drop1 = nn.Dropout(0.25)

        fc_size = calculate_fc_size_encoder(
            self.img_h, self.img_w, self.conv_kernel_size, self.conv2_n_fil, self.max_pool_kernel_size
        )
        self.lin1 = nn.Linear(fc_size, out_size)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x):
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
        return x


class ConvDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        img_h: int,
        img_w: int,
        n_fil_1: int = 32,
        n_fil_2: int = 64,
        kernel_size: int = 3,
        fc_size: int = 128,
    ) -> None:
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w

        self.conv1_n_fil = n_fil_1
        self.conv2_n_fil = n_fil_2

        self.conv_kernel_size = kernel_size

        self.lin1 = nn.Linear(latent_dim, fc_size)
        self.drop1 = nn.Dropout(0.5)
        fc_size_2 = calculate_fc_size_decoder(self.img_h, self.img_w, self.conv_kernel_size, self.conv2_n_fil)
        self.lin2 = nn.Linear(fc_size, fc_size_2)
        self.drop2 = nn.Dropout(0.25)
        self.dconv1 = nn.ConvTranspose2d(
            in_channels=self.conv2_n_fil, out_channels=self.conv1_n_fil, kernel_size=self.conv_kernel_size, stride=1
        )
        self.dconv2 = nn.ConvTranspose2d(
            in_channels=self.conv1_n_fil, out_channels=in_channels, kernel_size=self.conv_kernel_size, stride=1
        )

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.drop2(x)

        conv1_size_reduction = self.conv_kernel_size - 1
        conv2_size_reduction = self.conv_kernel_size - 1
        total_reduction = conv1_size_reduction + conv2_size_reduction

        x = x.view(-1, self.conv2_n_fil, (self.img_h - total_reduction), (self.img_w - total_reduction))
        x = self.dconv1(x)
        x = F.relu(x)
        x = self.dconv2(x)
        return x


class VAEGaussBase(AbstractVAE):
    latent_dim = 128
    conv1_n_fil = 32
    conv2_n_fil = 64
    conv_kernel_size = 3
    max_pool_kernel_size = 2
    fc_size = 128

    def __init__(self, in_channels: int, img_h: int, img_w: int) -> None:
        """Convolutional Variational Autoencoder with Gaussian prior

        Args:
            in_channels (int): Number of channels in the input image
            img_h (int): Height of input image
            img_w (int): Width of input image
        """
        super(VAEGaussBase, self).__init__()

        self.encoder = ConvEncoder(
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w,
            out_size=self.fc_size,
            n_fil_1=self.conv1_n_fil,
            n_fil_2=self.conv2_n_fil,
            kernel_size=self.conv_kernel_size,
            mp_size=self.max_pool_kernel_size,
        )

        self.lin_mu = nn.Linear(self.fc_size, self.latent_dim)
        self.lin_logvar = nn.Linear(self.fc_size, self.latent_dim)

        self.decoder = ConvDecoder(
            in_channels=in_channels,
            latent_dim=self.latent_dim,
            img_h=img_h,
            img_w=img_w,
            n_fil_1=self.conv1_n_fil,
            n_fil_2=self.conv2_n_fil,
            kernel_size=self.conv_kernel_size,
            fc_size=self.fc_size,
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.lin_mu(x)
        logvar = self.lin_logvar(x)
        return mu, logvar

    def decode(self, x):
        return self.decoder(x)

    @staticmethod
    def reparameterize(mu, logvar) -> Tuple[torch.Tensor, torch.Tensor]:
        std = torch.exp(logvar)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z, std

    def forward(self, x):
        mu, logvar = self.encode(x)

        z, std = self.reparameterize(mu, logvar)

        x_hat = self.decode(z)

        return x_hat, mu, std, z


class VAEGaussBaseSigm(VAEGaussBase):
    def decode(self, x):
        x = super().decode(x)
        x = torch.sigmoid(x)
        return x


class VAEGaussBig(VAEGaussBase):
    conv1_n_fil = 128
    conv2_n_fil = 256
    fc_size = 512


class VAEGaussBigSigm(VAEGaussBig):
    def decode(self, x):
        x = super().decode(x)
        x = torch.sigmoid(x)
        return x


def models_factory(name):
    return {
        "vae-gauss-base": VAEGaussBase,
        "vae-gauss-big": VAEGaussBig,
        "vae-gauss-base-sigm": VAEGaussBaseSigm,
        "vae-gauss-sigm-big": VAEGaussBigSigm,
    }[name]
