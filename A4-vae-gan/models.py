import torch.nn as nn
import torch.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear


class View(nn.Module):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class VAEGauss(nn.Module):
    def __init__(self, in_channels, latent_dim) -> None:
        super().__init__()
        self.fully_connected_size = 9216
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(self.fully_connected_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.fully_connected_size),
            View((-1, 64, 64)),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=in_channels, kernel_size=3, stride=1),
        )

    def encode(self, x):
        pass

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def activate_output(self, logits):
        return F.softmax(logits)

    def predict(self, x):
        logits = self.forward(x)
        y = self.activate_output(logits)
        return y
