import jax
import jax.numpy as jnp
import flax.nnx as nnx
import einops

"""
Dimension key:
B: batch size
L: sequence length
D: embedding dimension
H: number of attention heads
K: size of each attention head (D // H)
"""

class SelfAttention(nnx.Module):
    def __init__(self, n_heads: int, d_embed: int, rngs: nnx.Rngs, in_proj_bias: bool = True, out_proj_bias: bool = True):
        self.n_heads = n_heads
        self.d_embed = d_embed
        self.d_head = d_embed // n_heads
        self.in_proj = nnx.Linear(in_features=d_embed, out_features=3*d_embed, use_bias=in_proj_bias, rngs=rngs)
        self.out_proj = nnx.Linear(in_features=d_embed, out_features=d_embed, use_bias=out_proj_bias, rngs=rngs)

    def __call__(self, x_BLD, *, causal_mask=False):
        B, L, D = jnp.shape(x_BLD)
        H = self.n_heads
        K = self.d_head # K = D // H

        qkv_BL3D = self.in_proj(x_BLD) # (B, L, 3D)
        q_BLD, k_BLD, v_BLD = jnp.split(qkv_BL3D, 3, axis=-1)

        q_BHLK = einops.rearrange(q_BLD, 'B L (H K) -> B H L K', H=H)
        k_BHLK = einops.rearrange(k_BLD, 'B L (H K) -> B H L K', H=H)
        v_BHLK = einops.rearrange(v_BLD, 'B L (H K) -> B H L K', H=H)

        attn_logits_BHLL = jnp.einsum('BHLK,BHMK->BHLM', q_BHLK, k_BHLK) / jnp.sqrt(K)

        if causal_mask:
            mask_LL = jnp.triu(jnp.ones((L, L)), k=1).astype(bool)
            attn_logits_BHLL = jnp.where(mask_LL, -jnp.inf, attn_logits_BHLL)

        attn_weights_BHLL = nnx.softmax(attn_logits_BHLL, axis=-1)

        out_BHLK = jnp.einsum('BHLM,BHMK->BHLK', attn_logits_BHLL, v_BHLK)

        out_BLD = einops.rearrange(out_BHLK, 'B H L K -> B L (H K)')

        out_BLD = self.out_proj(out_BLD)

        return out_BLD
