import torch.nn as nn
import pytorch_lightning as pl
from typing import List


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        kernel_size: int,
        pool_size: int,
    ):
        """
        This class represents a CNN block which projects
        `in_channels' to a usually higher amount of channels
        `out_channels'. It runs the image through `depth' layers
        with a kernel of size `kernel_size'. In order to keep
        the resolution of the image the same, it must be padded
        by with `kernel_size//2' zeroes around the image.

        The image is then pooled with a MaxPool2d operation of size
        `pool_size`, the resulting is a model that has input of shape
        (batch, in_channels, width, height), and outputs an image of
        shape (batch, out_channels, width//pool_size, height//pool_size)
        Inputs:
            in_channels: number of input channels
            out_channels: number of output channels
            depth: number of convolutional layers in encoder block
            kernel_size: size of the kernel of the convolutional layers
            pool_size: size of the kernel of the pooling layer
        """
        super(Encoder, self).__init__()

        layers = list()
        for i in range(depth):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())

        self.sequence = nn.Sequential(*layers, nn.MaxPool2d(kernel_size=pool_size))

    def forward(self, img):
        """
        runs the partial prediction in the encoder block

        Inputs:
            img: input image of shape
            (batch, in_channels, width, height)

        Outputs:
            img: output image of shape
            (batch, out_channels, width//pool_size, height//pool_size)
        """
        return self.sequence(img)


class SegmentationCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        embedding_size: int = 64,
        pool_sizes: List[int] = [5, 5, 2],
        kernel_size: int = 3,
        **kwargs
    ):
        """
        Basic CNN that performs segmentation. This model takes
        an image of shape (batch, in_channels, width, height)
        and outputs an image of shape
        (batch, out_channels, width//prod(pool_sizes), height//prod(pool_sizes)),
        where prod() is the product of all the values in pool_sizes.

        This is done by using len(pool_sizes) Encoder layers, each of which
        pools the resolution down by a factor of pool_sizes[i].

        The first encoder must project the in_channels to embedding_size.
        Each subsequent layer must double the number of channels in its input,
        for example, the second layer must go from embedding_size to 2*embedding_size,
        the third layer from 2*embedding_size to 4*embedding_size and so on, until
        the pool_sizes list has been depleted.

        The final layer (decoder) must project the (2**len(pool_sizes))*embedding_size
        channels to output_channels channels. In order to keep the resolution, you
        may use a 1x1 kernel.

        Hint: If you use a regular list to save your Encoders, these
        will not register with the pytorch module and will not
        have their parameters added to the optimizer. Use nn.ModuleList to avoid this.

        Note: **kwargs is used to capture any additional arguments that are not
        explicitly defined in the function signature. This is so that we can
        construct the object from the same dictionary--certain models don't have addiitonal
        parameters.

        Inputs:
            in_channels: number of input channels
            out_channels: number of output channels
            depth: number of convolutional layers in encoder block
            embedding_size: number of channels in the first encoder
            pool_sizes: list of pool sizes for each encoder
            kernel_size: size of the kernel of the convolutional layers
        """
        super(SegmentationCNN, self).__init__()

        modules = list()
        for i in range(len(pool_sizes)):
            if i == 0:
                modules.append(
                    Encoder(
                        in_channels, embedding_size, depth, kernel_size, pool_sizes[i]
                    )
                )
            else:
                modules.append(
                    Encoder(
                        2 ** (i - 1) * embedding_size,
                        2**i * embedding_size,
                        depth,
                        kernel_size,
                        pool_sizes[i],
                    )
                )

        self.modules_list = nn.ModuleList(modules)
        self.decoder = nn.ConvTranspose2d(
            in_channels=2 ** (len(pool_sizes) - 1) * embedding_size,
            out_channels=out_channels,
            kernel_size=1,
            **kwargs
        )

    def forward(self, X):
        """
        Runs the input X through the encoders and decoders.
        Inputs:
            X: image of shape
            (batch, in_channels, width, height)
        Outputs:
            y_pred: image of shape
            (batch, out_channels, width//prod(pool_sizes), height//prod(pool_sizes))
        """
        for encoder in self.modules_list:
            X = encoder.forward(X)

        X = self.decoder(X)

        return X
