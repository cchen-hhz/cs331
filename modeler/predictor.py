import torch
import yaml
import pickle
import time

from cs336_basics.layers.transformer import transformerLM
from cs336_basics.tokenizer import tokenizer
from cs336_basics.func.saver import load_checkpoint
from cs336_basics.func.softmax import softmax

# hyper params
p_config = {
    "model_name": "200.pth",
    "temp": 1,
    "top_p": 0.5,
    "input": "Once upon a time, there was a little boy, he fell in love with a little girl,"
}

root_dir = 'modeler'
conf_dir = f'{root_dir}/conf'
saver_dir = f'{root_dir}/dist/models'
bpe_dir = f'{root_dir}/dist/bpe'

with open(f'{conf_dir}/config.yaml') as f:
    config = yaml.safe_load(f)

# Load models and tokenizers
with open(f"{bpe_dir}/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
vocab_from = {token: i for i, token in vocab.items()}
end_token = vocab_from[b"<|endoftext|>"]

enc = tokenizer.fromfiles(f"{bpe_dir}/vocab.pkl",
                          f"{bpe_dir}/merges.pkl")

transformer = transformerLM(
    config['vocab_size'], 
    config['context_length'],
    config['rope_theta'],
    config['num_layers'],
    config['d_model'],
    config['num_heads'],
    config['d_ff']
)

print(f"Load model from {saver_dir}/{p_config['model_name']}...")
load_checkpoint(f"{saver_dir}/{p_config['model_name']}", transformer, None)

# Start predictor
def choose_next_token(pred: torch.Tensor, temp: float, top_p: float):
    eps = 1e-6
    pred = pred / (temp + eps)
    pred = softmax(pred)

    # top-p
    sorted_pred, sorted_ind = torch.sort(pred, descending=True, dim=-1)
    presum = torch.cumsum(sorted_pred, dim=-1)
    remove_mask = presum > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    sorted_pred[remove_mask] = 0.0
    sorted_pred = sorted_pred / torch.sum(sorted_pred, dim=-1, keepdim=True)

    sample_chosen = torch.multinomial(sorted_pred, num_samples=1)
    result = torch.gather(sorted_ind, -1, sample_chosen)
    return result
print(f"Input: {p_config['input']}", flush=True)
print("------")
cur_tokenlist = enc.encode(p_config['input'])
while True:
    output = transformer(torch.tensor(cur_tokenlist))
    next_token_logits = output[-1, :]
    next_token = choose_next_token(next_token_logits, p_config['temp'], p_config['top_p'])
    next_token_val = next_token.item()

    if int(next_token_val) == end_token:
        print(f"-----Got end token, generating ended!-----")
        break
    if len(cur_tokenlist) + 1 > config['context_length']:
        print(f"-----context length exceed {config['context_length']}, generating ended!-----")
        break
    print(f"{enc.decode([int(next_token_val)])}",end="" ,flush=True)
    cur_tokenlist.append(int(next_token_val))


