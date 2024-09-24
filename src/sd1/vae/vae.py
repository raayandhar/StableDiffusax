import jax
import jax.numpy as jnp
import flax.nnx as nnx
import einops
from common import AttentionBlock, ResidualBlock

"""
This repository adopts the Shazeer (Google, AIAYN) variable naming convention.

Dimension  key:

B: batch size
C: number of channels
H: height (image)
W: width (image)
L: sequence length
M: memory length (length of sequence being attended to)
D: model/embedding dimension (d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (d_kv)

When known, the name of a tensor should end in a dimension-suffix composed of those letters.
Recall that JAX uses (B, H, W, C) instead of (B, C, H, W) like in PyTorch.

In this case, we elect to let the name also include the channel/feature dimensions we are expanding or contracting, since 'C' by itself is not very descriptive.
"""

class Encoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):

        # 3 -> 128 -> (6 x 128)
        self.conv1 = nnx.Conv(in_features=3, out_features=128, kernel_size=(3,3), padding=1, rngs=rngs)

        self.res1 = ResidualBlock(in_features=128, out_features=128, rngs=rngs)
        self.res2 = ResidualBlock(in_features=128, out_features=128, rngs=rngs)

        self.conv2 = nnx.Conv(in_features=128, out_features=128, kernel_size=(3,3), strides=(2,2), padding=0, rngs=rngs)

        # 128 -> 256 -> (4 x 256)
        self.res3 = ResidualBlock(in_features=128, out_features=256, rngs=rngs)
        self.res4 = ResidualBlock(in_features=256, out_features=256, rngs=rngs)

        self.conv3 = ResidualBlock(in_features=256, out_features=256, kernel_size=(3,3), strides=(2,2), padding=0, rngs=rngs)

        # 256 -> 512 -> (10 x 512)
        self.res5 = ResidualBlock(in_features=256, out_features=512, rngs=rngs)
        self.res6 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)

        self.conv4 = nnx.Conv(in_features=512, out_features=512, kernel_size=(3,3), strides=(2,2), padding=0, rngs=rngs)

        self.res7 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)
        self.res8 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)
        self.res9 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)

        self.attention = AttentionBlock(512, rngs=rngs)

        # 512 -> 512
        self.res10 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)

        self.groupnorm = nnx.GroupNorm(num_features=512, num_groups=32, rngs=rngs)

        # SiLU
        # 512 -> 8 (2 x 8)
        self.conv5 = nnx.Conv(in_features=512, out_features=8, kernel_size=(3,3), padding=1, rngs=rngs)
        self.conv6 = nnx.Conv(in_features=8, out_features=8, kernel_size=(1,1), padding=0, rngs=rngs)

    def __call__(self, x, noise):
        """
        Args:
        x: (B, H, W, C=3)
        noise: (B, H/8, W/8, C=4)

        Returns:
        latent (z): (B, H/8, W/8, 4)
        """

        x = self






