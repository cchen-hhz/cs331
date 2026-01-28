import torch
import torch.nn as nn

class embedding(nn.Module):
    def __init__(self, num_embedding: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        weight = torch.randn(num_embedding, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weight, 0, 1, -3, 3)
        self.weight = nn.Parameter(weight)
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]
