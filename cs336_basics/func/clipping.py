import torch
import math
from typing import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    norm = 0.0
    eps = 1e-6
    for param in parameters:
        if param.grad is None:
            continue
        norm += param.grad.data.norm(2).item() ** 2

    norm = norm ** 0.5
    if norm > max_l2_norm:
        clip = max_l2_norm / (norm + eps)
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip)
