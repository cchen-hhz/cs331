import torch
from einops import reduce

def softmax(x: torch.Tensor, dim: int = -1):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x = x - x_max
    x = torch.exp(x)
    x_sum = torch.sum(x, dim=dim, keepdim=True)
    return x / x_sum