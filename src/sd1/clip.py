import jax
import jax.numpy as jnp
import flax.nnx as nnx
from attention import SelfAttention

"""
Dimension key:
V: vocabulary size (n_vocab)
D: model/embedding dimension (d_model or embedding_dim)
L: sequence length / max num of tokens
"""

class CLIPEmbedding(nnx.Module):
    def __init__(self, n_vocab: int, d_model: int, seq_len: int, rngs: nnx.Rngs):
        self.tok_embedding_VD = nnx.Embed(num_embeddings=n_vocab, features=d_model, rngs=rngs)
        self.pos_embedding_TD
