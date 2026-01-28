import torch
from torch import nn
from einops import einsum, rearrange
from .linear import linear
from .RotaryPositionalEmbedding import RoPE
from cs336_basics.func.dotAttention import dotAttention

class multiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_s = d_model // num_heads
        self.Wq = linear(d_model, d_model)
        self.Wk = linear(d_model, d_model)
        self.Wv = linear(d_model, d_model)
        self.Wo = linear(d_model, d_model)
    
    def _multihead(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                   RoPE: RoPE | None = None,
                   token_position: torch.Tensor | None = None):
        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)

        seq_len = Q.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device)).to(torch.bool)
        if RoPE is not None and token_position is not None:
            Q = RoPE.forward(Q, token_position)
            K = RoPE.forward(K, token_position)
        
        result = dotAttention(Q, K, V, mask)
        return rearrange(result, "... h s d -> ... s (h d)")
    
    def forward(self, x: torch.Tensor, RoPE: RoPE | None = None, token_position: torch.Tensor | None = None):
        Q = self.Wq.forward(x)
        K = self.Wk.forward(x)
        V = self.Wv.forward(x)
        result = self._multihead(Q, K, V, RoPE, token_position)
        return self.Wo.forward(result)