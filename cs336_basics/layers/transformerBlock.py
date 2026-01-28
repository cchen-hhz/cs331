import torch
from torch import nn
from typing import Dict

from .RMSnorm import RMSnorm
from .SwiGLU import SwiGLU
from .multihead import multiheadAttention
from .RotaryPositionalEmbedding import RoPE

class transformer(nn.Module):
    def __init__(self, 
                 d_model: int, num_heads: int,
                 d_ff: int):
        super().__init__()
        self.d_s = d_model // num_heads
        self.multihead = multiheadAttention(d_model, num_heads)
        self.swiglu = SwiGLU(d_model, d_ff)
        self.norm1 = RMSnorm(d_model)
        self.norm2 = RMSnorm(d_model)
        self.RoPE = None

    def gen_rope(self, theta: float, max_seq_len: int):
        self.RoPE = RoPE(theta,self.d_s, max_seq_len)

    def with_rope(self, RoPE: RoPE):
        self.RoPE = RoPE

    def load_with_dict(self, weights: Dict):  
        self.multihead.Wq.load_state_dict({"W": weights['attn.q_proj.weight']})
        self.multihead.Wk.load_state_dict({"W": weights['attn.k_proj.weight']})
        self.multihead.Wv.load_state_dict({"W": weights['attn.v_proj.weight']})
        self.multihead.Wo.load_state_dict({"W": weights['attn.output_proj.weight']})

        self.norm1.load_state_dict({"weight": weights['ln1.weight']})
        self.norm2.load_state_dict({"weight": weights['ln2.weight']})

        self.swiglu.W1.load_state_dict({"W": weights['ffn.w1.weight']})
        self.swiglu.W2.load_state_dict({"W": weights['ffn.w2.weight']})
        self.swiglu.W3.load_state_dict({"W": weights['ffn.w3.weight']})


    def forward(self, x: torch.Tensor, token_position: torch.Tensor | None = None):
        if token_position is None and self.RoPE is not None:
            seq_len = x.shape[-2]
            token_position = torch.arange(0, seq_len, device=x.device)
        
        x = x + self.multihead.forward(self.norm1.forward(x), self.RoPE, token_position)
        x = x + self.swiglu.forward(self.norm2.forward(x))
        return x

