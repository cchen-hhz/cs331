import torch
from torch import nn
from .linear import linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff = None, device = None, dtype=None):
        super().__init__()
        self.d_model = d_model
        
        if d_ff is None:
            d_ff = d_model * 8 // 3
            d_ff = 64 * ((d_ff + 63) // 64)
            self.d_ff = d_ff
        else:
            self.d_ff = d_ff

        self.W1 = linear(d_model, self.d_ff, device, dtype)
        self.W3 = linear(d_model, self.d_ff, device, dtype)
        self.W2 = linear(self.d_ff, d_model, device, dtype)

    def _silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        result = self._silu(self.W1.forward(x)) * self.W3.forward(x)
        return self.W2.forward(result)
