import flax.nnx as nnx
import jax
import jax.numpy as jnp
import einops
from attention import SelfAttention

class ResidualBlock:
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.gn1 = nnx.GroupNorm(num_features=in_features, num_groups=32, rngs=rngs)
        self.conv1 = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=(3,3), padding=1, rngs=rngs)

        self.gn2 = nnx.GroupNorm(num_features=out_features, num_groups=32, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=(3,3), padding=1, rngs=rngs)

        if in_features != out_features:
            self.shortcut = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=(1,1), padding=0, rngs=rngs)
        else:
            self.shortcut = lambda x: x

    def __call__(self, x):
        residual = x

        x = self.gn1(x)
        x = nnx.silu(x)
        x = self.conv1(x)
        x = self.gn2(x)
        x = nnx.silu(x)
        x = self.conv2(x)

        return x + self.shortcut(residual)

class AttentionBlock:
    def __init__(self, num_features: int, rngs: nnx.Rngs):
        self.gn = nnx.GroupNorm(num_features=num_features, num_groups=32, rngs=rngs)
        self.attention = SelfAttention(n_heads=1, d_embed=num_features, rngs=rngs)

    def __call__(self, x):
        # recall that JAX expects (B, H, W, C)
        residual = x

        x = self.gn(x)

        B, H, W, C = jnp.shape(x)

        x = einops.rearrange(x, 'B H W C -> B (H W) C')
        x = self.attention(x)
        x = einops.rearrange(x, 'B (H W) C -> B H W C', H=H, W=W)
        x += residual

        return x
