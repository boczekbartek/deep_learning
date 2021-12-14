import torch
import torch.nn as nn
import torch.nn.functional as F


class View(nn.Module):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class VAEGauss(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, img_h: int, img_w: int) -> None:
        """[summary]

        Args:
            in_channels (int): Number of channels in the input image
            latent_dim (int): Size of the latent space
            img_h (int): Height of input image
            img_w (int): Width of input image
        """
        super().__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(0.25)
        self.fully_connected_size = (
            64 * (img_h - 4) // 2 * (img_w - 4) // 2
        )  # TODO calculate in code, this was read experimentically and works only for 28x28 images
        self.lin1 = nn.Linear(self.fully_connected_size, 128)
        self.drop2 = nn.Dropout(0.5)
        self.lin2 = nn.Linear(128, latent_dim)

        # Decoder layers
        self.lin3 = nn.Linear(latent_dim, 128)
        self.drop3 = nn.Dropout(0.5)
        self.lin4 = nn.Linear(128, 64 * 32)
        self.drop4 = nn.Dropout(0.25)
        self.dconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.dconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=in_channels, kernel_size=3, stride=1)

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
        x = self.lin2(x)
        return x

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def activate_output(self, logits):
        return F.softmax(logits)

    def predict(self, x):
        logits = self.forward(x)
        y = self.activate_output(logits)
        return y
