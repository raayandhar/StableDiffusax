import jax
import jax.numpy as jnp
import flax.nnx as nnx
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

TODO:
- Comments on shape outputs; go back to using dimension key (trim the key above)
- Better way to manage getting through the modules
- Better forward calls; implicit upsampling (in residual block)?
"""

class Encoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        # 3 -> 128 -> (6 x 128)
        self.conv1 = nnx.Conv(in_features=3, out_features=128, kernel_size=(3, 3), padding=1, rngs=rngs)

        self.res1 = ResidualBlock(in_features=128, out_features=128, rngs=rngs)
        self.res2 = ResidualBlock(in_features=128, out_features=128, rngs=rngs)

        self.conv2 = nnx.Conv(in_features=128, out_features=128, kernel_size=(3, 3), strides=(2, 2), padding=0, rngs=rngs)

        # 128 -> 256 -> (4 x 256)
        self.res3 = ResidualBlock(in_features=128, out_features=256, rngs=rngs)
        self.res4 = ResidualBlock(in_features=256, out_features=256, rngs=rngs)

        self.conv3 = nnx.Conv(in_features=256, out_features=256, kernel_size=(3, 3), strides=(2, 2), padding=0, rngs=rngs)

        # 256 -> 512 -> (10 x 512)
        self.res5 = ResidualBlock(in_features=256, out_features=512, rngs=rngs)
        self.res6 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)

        self.conv4 = nnx.Conv(in_features=512, out_features=512, kernel_size=(3, 3), strides=(2, 2), padding=0, rngs=rngs)

        self.res7 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)
        self.res8 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)
        self.res9 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)

        self.attention = AttentionBlock(num_features=512, rngs=rngs)

        # 512 -> 512
        self.res10 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)

        self.groupnorm = nnx.GroupNorm(num_features=512, num_groups=32, rngs=rngs)

        # 512 -> 8 -> (2 x 8)
        self.conv5 = nnx.Conv(in_features=512, out_features=8, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.conv6 = nnx.Conv(in_features=8, out_features=8, kernel_size=(1, 1), padding=0, rngs=rngs)

    def __call__(self, x, noise):
        """
        Args:
            x: (B, H, W, C=3)
            noise: (B, H/8, W/8, C=4)

        Returns:
            latent (z): (B, H/8, W/8, C=4)
        """

        # Q: is there a better way to do this?
        for module in [self.conv1, self.res1, self.res2, self.conv2, self.res3, self.res4, self.conv3,
                       self.res5, self.res6, self.conv4, self.res7, self.res8, self.res9, self.attention,
                       self.res10, self.groupnorm]:

            if isinstance(module, nnx.Conv) and module.strides == (2, 2):
                x = pad_asymmetric(x,  padding=(1, 1))
            x = module(x)

        x = nnx.silu(x)
        x = self.conv5(x)
        x = self.conv6(x)

        if x.shape[-1] % 2 != 0:
            raise ValueError("Final output channels must be divisible by 2 to split mean and log_var")

        mean, log_var = jnp.split(x, 2, axis=-1)
        log_var = jnp.clip(log_var, -30, 20)
        variance = jnp.exp(log_var)
        stdev = jnp.sqrt(variance)

        x = mean + stdev * noise
        x *= 0.18215

        return x

class Decoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):

        self.conv1 = nnx.Conv(in_features=4, out_features=4, kernel_size=(1, 1), padding=0, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=4, out_features=512, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.res1 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)
        self.attention = AttentionBlock(num_features=512, rngs=rngs)
        self.res2 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)
        self.res3 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)
        self.res4 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)
        self.res5 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)


        self.conv3 = nnx.Conv(in_features=512, out_features=512, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.res6 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)
        self.res7 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)
        self.res8 = ResidualBlock(in_features=512, out_features=512, rngs=rngs)

        self.conv4 = nnx.Conv(in_features=512, out_features=256, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.res9 = ResidualBlock(in_features=256, out_features=256, rngs=rngs)
        self.res10 = ResidualBlock(in_features=256, out_features=256, rngs=rngs)
        self.res11 = ResidualBlock(in_features=256, out_features=256, rngs=rngs)

        self.conv5 = nnx.Conv(in_features=256, out_features=128, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.res12 = ResidualBlock(in_features=128, out_features=128, rngs=rngs)
        self.res13 = ResidualBlock(in_features=128, out_features=128, rngs=rngs)
        self.res14 = ResidualBlock(in_features=128, out_features=128, rngs=rngs)

        self.groupnorm = nnx.GroupNorm(num_features=128, num_groups=32, rngs=rngs)
        self.conv_out = nnx.Conv(in_features=128, out_features=3, kernel_size=(3, 3), padding=1, rngs=rngs)

    def __call__(self, x):
        """
        Args:
            x: (B, H/8, W/8, C=4)

        Returns:
            decoded latent (x): (B, H, W, C=3)
        """
        x /= 0.18215  # Unscale

        upsamplings = {7, 11, 15}
        modules = [self.conv1, self.conv2, self.res1, self.attention,
                   self.res2, self.res3, self.res4, self.res5,
                   self.conv3, self.res6, self.res7, self.res8,
                   self.conv4, self.res9, self.res10, self.res11,
                   self.conv5, self.res12, self.res13, self.res14]

        for i, module in enumerate(modules):
            x = module(x)
            if i in upsamplings:
                x = upsample(x, scale_factor=2)

        x = self.groupnorm(x)
        x = nnx.silu(x)
        x = self.conv_out(x)

        return x

def pad_asymmetric(x, padding):
    pad_height = padding[0]
    pad_width = padding[1]
    x = jnp.pad(x, ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
    return x

def upsample(x, scale_factor):
    B, H, W, C = jnp.shape(x)
    out_shape = (B, H*scale_factor, W*scale_factor, C)
    x = jax.image.resize(x, out_shape, method="nearest")
    return x


