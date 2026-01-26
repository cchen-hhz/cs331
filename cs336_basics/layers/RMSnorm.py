import torch
from torch import nn
from einops import einsum, reduce

class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps
    
    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = reduce(torch.pow(x, 2), "... d -> ... 1", "mean")
        rms = torch.rsqrt(rms + self.eps)
        x = x * rms
        return x.to(in_dtype) * self.weight