import torch
import os
import typing

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
                    model: torch.nn.Module,
                    optim: torch.optim.Optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optim.load_state_dict(obj['optimizer'])
    it = obj['iteration']
    return it