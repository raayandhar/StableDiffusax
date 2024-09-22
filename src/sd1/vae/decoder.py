import jax
import jax.numpy as jnp
import flax.nnx as nnx
import einops

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
F: feed-forward subnetwork hiddeen size
H: number of attention heads in a layer
K: size of each attention key or value (d_kv)

When known, the name of a tensor should end in a dimension-suffix composed of those letters
"""


