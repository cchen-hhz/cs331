import torch
from cs336_basics.func.softmax import softmax
from einops import einsum

def dotAttention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask):
    d_k = Q.shape[-1]
    result = einsum(Q, K, "... q d, ... k d -> ... q k")
    result = result / torch.sqrt(torch.tensor(d_k, device=Q.device, dtype=Q.dtype))
    zero = torch.tensor(0.0, device=result.device, dtype=result.dtype)
    neg_inf = torch.tensor(float('-inf'), device=result.device, dtype=result.dtype)
    maskp = torch.where(mask, zero, neg_inf)
    result += maskp
    return softmax(result) @ V
