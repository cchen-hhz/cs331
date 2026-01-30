import wandb
import os
import yaml
import numpy as np
import torch
import math

from cs336_basics.layers.transformer import transformerLM
from cs336_basics.optim.lr_schedule import cos_schedule
from cs336_basics.optim.adamw import AdamW
from cs336_basics.func.batch import get_batch
from cs336_basics.func.clipping import gradient_clipping
from cs336_basics.func.crossEntropy import crossEntropyLoss
from cs336_basics.func.saver import save_checkpoint, load_checkpoint

# TODO: AdamW 没整合动态学习率调整，需要修改
device = "cuda"

root_dir = 'modeler'
conf_dir = f'{root_dir}/conf'
saver_dir = f'{root_dir}/dist/models'
bpe_dir = f'{root_dir}/dist/bpe'
data_dir = f'{root_dir}/dist/data'

# Load config hyper parameters
with open(f'{conf_dir}/config.yaml') as f:
    config = yaml.safe_load(f)

with open(f'{conf_dir}/secret.yaml') as f:
    secret_config = yaml.safe_load(f)
os.environ['WANDB_API_KEY']=secret_config['WANDB_API_KEY']

# Wandb
print(f"Login into wendb")
wandb.login()
run = wandb.init(
    entity="cchen-hhz-company",
    project="cs336",
    config=config
)

# Load textdata
train_data = np.memmap(f"{data_dir}/train.bin", np.uint16)
val_data = np.memmap(f"{data_dir}/val.bin", np.uint16)

# transformer and optim
transformer = transformerLM(
    config['vocab_size'], 
    config['context_length'],
    config['rope_theta'],
    config['num_layers'],
    config['d_model'],
    config['num_heads'],
    config['d_ff']
)

scheduler = cos_schedule(
    config['lr_max'],
    config['lr_min'],
    config['Tw'],
    config['Tc']
)
adamW = AdamW(
    transformer.parameters(), 
    config['lr_min'],
    config['weight_decay'],
    (config['beta1'], config['beta2']),
    scheduler,
    1e-8)


transformer.to(device)

print(f"------------Training up---------------")
for epoch in range(1, config['epochs'] + 1):
    x, y = get_batch(
        train_data, 
        config['batch_size'],
        config['context_length'],
        device)

    result = transformer(x) 
    loss_batch = crossEntropyLoss(result, y)
    loss = torch.mean(loss_batch, dim=0)
    print(f"epoch {epoch} with loss {loss.item()}, perplexity {math.exp(loss.item())}")
    run.log({"loss": loss.item(), "perplexity": math.exp(loss.item())})

    adamW.zero_grad()
    loss.backward()
    gradient_clipping(transformer.parameters(), config['grad_clip'])
    adamW.step()

    if epoch % 5 == 0:
        with torch.no_grad():
            val_loss = 0.
            for _ in range(3):
                x, y = get_batch(val_data, 
                                    config['batch_size'],
                                    config['context_length'],
                                    device)
                result = transformer(x)
                loss = crossEntropyLoss(result, y).mean()
                val_loss += loss
            
        val_loss /= 3.
        print(f"val loop: {val_loss} perplexity {math.exp(val_loss)}")
    if epoch % 100 == 0:
        print(f"save model and optim")
        save_checkpoint(transformer, adamW, epoch, f"{saver_dir}/{epoch}.pth")





