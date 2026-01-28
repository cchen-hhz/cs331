import torch
from typing import Optional
from collections.abc import Callable
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3,
                 weight_decay = 0.01, 
                 betas = (0.9, 0.999),
                 eps = 1e-8):
        assert lr >= 0
        defaults = {
            "lr": lr,
            "beta": betas,
            "eps": eps,
            "decay": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['beta']
            eps = group['eps']
            decay = group['decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', 0)
                v = state.get('v', 0)
                grad = p.grad.data
                
                m = beta1 * m + (1. - beta1) * grad
                v = beta2 * v + (1. - beta2) * grad * grad
                lr_t = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * decay * p.data 

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
        return loss


                