import pathlib
from typing import Tuple, Union

import numpy as np
import torch
import torch.utils.data
import skimage.transform
from torchvision import datasets, transforms
from PIL.Image import Image
from sklearn.model_selection import train_test_split

here = pathlib.Path(__file__).parent


class Rescale(object):
    """Rescale the image from 2 to 3 dimensions. All layers in 3rd dimension will be the same.

    Args:
        output_size (Tuple[int,int,int]): Desired output shape - in 3 channels
    """

    def __init__(self, output_size: Tuple[int, int, int]):
        self.c, self.h, self.w = output_size

    def __call__(self, image: Union[Image, list, np.ndarray]) -> np.ndarray:
        """Rescale

        Args:
            image (anything transformable to np.array): 2-channels image: (h,w).

        Returns:
            np.ndarray: resized, 3-channels image
        """
        np_image = np.array(image)
        res_image_1ch = skimage.transform.resize(np_image, output_shape=(self.h, self.w))

        # Make all channels equal
        return np.stack([res_image_1ch] * self.c)


def load_mnist(batch_size: int, cuda: bool) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Return train and test loaders for mnist. """

    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(here / "../data", train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(here / "../data", train=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


def load_inceptionv3_mnist(
    batch_size: int, cuda: bool
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Rescale MNIST to fix inception v3 input. Return train and test loaders """
    inceptionv3_input_shape = (3, 32, 32)

    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}

    transforms_pipeline = transforms.Compose([Rescale(inceptionv3_input_shape), torch.FloatTensor])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(here / "../data", train=True, download=True, transform=transforms_pipeline),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(here / "../data", train=False, transform=transforms_pipeline),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader


def load_svhn(
    batch_size: int, cuda: bool, test_size=0.1, random_state=42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Return train and test loaders for SVHN """

    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
    dataset = datasets.SVHN(here / "../data", download=True, transform=transforms.ToTensor())

    train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs,)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, **kwargs,)

    return train_loader, test_loader


def data_factory(data_name: str):
    factory = {"mnist": load_mnist, "svhn": load_svhn, "mnist-inceptionv3": load_inceptionv3_mnist}
    return factory[data_name]
