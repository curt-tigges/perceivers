import torch
import math
from torch import nn


class PositionalImageEmbedding(nn.Module):
    """Reshapes images and concatenates position encoding

    Initializes position encoding,

    Args:

    Returns:

    """

    def __init__(self, input_shape, input_channels, embed_dim, bands=4):
        super().__init__()

        self.ff = self.fourier_features(shape=input_shape, bands=bands)
        self.conv = nn.Conv1d(input_channels + self.ff.shape[0], embed_dim, 1)

    def forward(self, x):
        # initial x of shape [BATCH_SIZE x CHANNELS x HEIGHT x WIDTH]

        # create position encoding of the same shape as x
        enc = self.ff.unsqueeze(0).expand((x.shape[0],) + self.ff.shape)
        enc = enc.type_as(x)
        # print(enc.shape)
        x = torch.cat([x, enc], dim=1)
        # concatenate position encoding along the channel dimension
        # shape is now [BATCH_SIZE x COLOR_CHANNELS + POS_ENC_CHANNELS x HEIGHT x WIDTH]

        x = x.flatten(2)
        # reshape to [BATCH_SIZE x CHANNELS x HEIGHT*WIDTH]

        x = self.conv(x)
        # shape is now [BATCH_SIZE x EMBED_DIM x HEIGHT*WIDTH]

        x = x.permute(2, 0, 1)
        # shape is now [HEIGHT*WIDTH x BATCH_SIZE x EMBED_DIM]

        return x

    # Function adapted from Louis Arge's implementation: https://github.com/louislva/deepmind-perceiver
    def fourier_features(self, shape, bands):
        # This first "shape" refers to the shape of the input data, not the output of this function
        dims = len(shape)

        # Every tensor we make has shape: (bands, dimension, x, y, etc...)

        # Pos is computed for the second tensor dimension
        # (aptly named "dimension"), with respect to all
        # following tensor-dimensions ("x", "y", "z", etc.)
        pos = torch.stack(
            list(
                torch.meshgrid(
                    *(torch.linspace(-1.0, 1.0, steps=n) for n in list(shape))
                )
            )
        )
        pos = pos.unsqueeze(0).expand((bands,) + pos.shape)

        # Band frequencies are computed for the first
        # tensor-dimension (aptly named "bands") with
        # respect to the index in that dimension
        band_frequencies = (
            (
                torch.logspace(
                    math.log(1.0), math.log(shape[0] / 2), steps=bands, base=math.e
                )
            )
            .view((bands,) + tuple(1 for _ in pos.shape[1:]))
            .expand(pos.shape)
        )

        # For every single value in the tensor, let's compute:
        #             freq[band] * pi * pos[d]

        # We can easily do that because every tensor is the
        # same shape, and repeated in the dimensions where
        # it's not relevant (e.g. "bands" dimension for the "pos" tensor)
        result = (band_frequencies * math.pi * pos).view((dims * bands,) + shape)

        # Use both sin & cos for each band, and then add raw position as well
        # TODO: raw position
        result = torch.cat(
            [
                torch.sin(result),
                torch.cos(result),
            ],
            dim=0,
        )

        return result
