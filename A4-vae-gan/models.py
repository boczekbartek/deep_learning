import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEGauss(nn.Module):
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

        self.img_h = img_h
        self.img_w = img_w

        self.conv1_n_fil = 32
        self.conv2_n_fil = 64

        self.conv_kernel_size = 3
        self.max_pool_kernel_size = 2

        # Encoder layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=self.conv_kernel_size, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.conv_kernel_size, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=self.max_pool_kernel_size)
        self.drop1 = nn.Dropout(0.25)
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

        z = self.reparameterize(mu, logvar)

        x_hat = self.decode(z)

        return x_hat, mu, logvar

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
    def reparameterize(mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def activate_output(self, logits):
        return F.softmax(logits)

    def predict(self, x):
        logits = self.forward(x)
        y = self.activate_output(logits)
        return y
