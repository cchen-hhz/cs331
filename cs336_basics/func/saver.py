import torch
import os
import typing
from typing import Optional

def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(obj, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: Optional[torch.nn.Module],
                    optim: Optional[torch.optim.Optimizer]):
    obj = torch.load(src, weights_only=False)
    if model is not None:
        model.load_state_dict(obj['model'])
    if optim is not None:
        optim.load_state_dict(obj['optimizer'])
    it = obj['iteration']
    return it