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

In this case, we elect to also include the channel/feature dimensions we are expanding or contracing, since 'C' by itself is not very descriptive.
"""

class Encoder(nnx.Module):
    def __init__(self):
        self.conv1_C3_C128 = nnx.Conv(in_features=3, out_features=128, kernel_size=(3,3), padding=1)

        self.res1_C128_C128 = ResidualBlock(in_features=128, out_features=128)
        self.res2_C128_C128 = ResidualBlock(in_features=128, out_features=128)

        self.conv2_C128_C128 = nnx.Conv(in_features=128, out_features=128, kernel_size=(3,3), strides=(2,2), padding=0)

        self.res3_C128_C256 = ResidualBlock(in_features=128, out_features=256)
        self.res4_C256_C256 = ResidualBlock(in_features=256, out_features=256)

        self.conv3_C256_C256 = ResidualBlock(in_features=256, out_features=256, kernel_size=(3,3), strides=(2,2), padding=0)

        self.res5_C256_C256 = ResidualBlock(in_features=256, out_features=256)









