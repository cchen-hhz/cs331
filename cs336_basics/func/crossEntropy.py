import torch
from einops import rearrange, reduce

def crossEntropyLoss(input: torch.Tensor, answer: torch.Tensor):
    maxp = torch.max(input, dim=-1, keepdim=True).values
    shift_input = input - maxp
    exp_sum = torch.sum(torch.exp(shift_input), dim=-1, keepdim=True)
    exp_sum_log = torch.log(exp_sum)
    result = exp_sum_log - torch.gather(input, -1, answer.unsqueeze(-1)) + maxp
    return reduce(result, "... a 1 -> ...", 'mean')


