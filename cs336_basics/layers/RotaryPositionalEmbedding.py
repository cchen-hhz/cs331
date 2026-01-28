import torch
from torch import nn
from einops import rearrange, einsum, repeat

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        d2 = d_k // 2
        self.max_len = max_len

        exp_d = -2. * torch.arange(0, d2, dtype=torch.float32, device=device) / d_k
        exp_d = torch.pow(theta, exp_d)
        i_base = torch.arange(0, max_len, dtype=torch.float32, device=device)
        theta_tensor = torch.outer(i_base, exp_d)

        #self.cos = torch.cos(theta_tensor)
        #self.sin = torch.sin(theta_tensor)
        self.register_buffer('cos', torch.cos(theta_tensor), persistent=False)
        self.register_buffer('sin', torch.sin(theta_tensor), persistent=False)
        
    
    def forward(self, x: torch.Tensor, token_position: torch.Tensor):
        cos_cut = self.cos[token_position]
        sin_cut = self.sin[token_position]
        x = rearrange(x, "... (d d2) -> ... d d2", d2=2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        x1_r = x1 * cos_cut - x2 * sin_cut
        x2_r = x2 * cos_cut + x1 * sin_cut

        result = torch.stack([x1_r, x2_r], dim=0) 
        return rearrange(result, "t ... d2 -> ... (d2 t)")