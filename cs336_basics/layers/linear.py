import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

class linear(nn.Module):
    def __init__(self, 
                 in_feature: int,
                 out_feature: int,
                 device = None,
                 dtype = None):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        W = torch.empty(out_feature, in_feature, device=device, dtype=dtype)
        std = math.sqrt(2.0 / (in_feature + out_feature))
        length = std * 3.0
        torch.nn.init.trunc_normal_(W, 0, std, -length, length)
        self.W = nn.Parameter(W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
    
        