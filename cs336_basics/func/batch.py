import numpy as np
import numpy.typing as npt
import torch

def get_batch(dataset: npt.NDArray, batch_size: int, context_length:int, device: str):
    positions = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(
        dataset[i : i + context_length]
    ) for i in positions])
    y = torch.stack([torch.from_numpy(
        dataset[i + 1 : i + context_length + 1]
    ) for i in positions])
    x = x.to(device)
    y = y.to(device)
    return (x, y)

