import torch
from torch import nn
from typing import Dict

from .transformerBlock import transformer as TransformerBlock
from .embedidng import embedding
from .RotaryPositionalEmbedding import RoPE
from .RMSnorm import RMSnorm
from .linear import linear
from cs336_basics.func.softmax import softmax

class transformerLM(nn.Module):
    def __init__(self, vocab_size: int, 
                 context_length: int, 
                 theta: float,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 weights: Dict | None = None):
        super().__init__()
        
        self.RoPE = RoPE(theta, d_model // num_heads, context_length)
        self.embedding = embedding(vocab_size, d_model)
        self.num_layers = num_layers

        self.block = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        for block in self.block:
            block.with_rope(self.RoPE)
        
        self.normfinal = RMSnorm(d_model)
        self.linear = linear(d_model, vocab_size)

        if weights is not None:
            self._load_weights(weights)
    
    def _load_weights(self, weights: Dict):
        self.embedding.load_state_dict({"weight": weights['token_embeddings.weight']})
        for i in range(self.num_layers):
            prefix = f"layers.{i}."
            layer_dict = {
                key[len(prefix):]: value
                for key, value in weights.items()
                if key.startswith(prefix)
            }
            self.block[i].load_with_dict(layer_dict)
        self.normfinal.load_state_dict({"weight": weights['ln_final.weight']})
        self.linear.load_state_dict({"W": weights['lm_head.weight']})

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        for block in self.block:
            x = block(x)
        x = self.normfinal(x)
        x = self.linear(x)

        return x
