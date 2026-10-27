from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import click

VALID_TASKS = ['classification', 'regression']

@click.command()
@click.option(
  '--task',
  type=click.Choice(VALID_TASKS, case_sensitive=False),
  required=True,
  help="The task"
)

def main(task: str):
  train = pd.read_csv(f'../dataset/train_{task}.csv')
  val = pd.read_csv(f'../dataset/val_{task}.csv')
  test = pd.read_csv(f'../dataset/test_{task}.csv')
  model_path = '/root/project/titanBBB/TITAN-BBB-main/pretrained/ChemBERTa-100M-MLM'

  device = torch.device("cuda:0")

  tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
  model = AutoModel.from_pretrained(model_path, output_hidden_states=True, local_files_only=True).to(device)

  model.eval()

  process_data(train, tokenizer, model, device, "train", task)
  process_data(val, tokenizer, model, device, "val", task)
  process_data(test, tokenizer, model, device, "test", task)

def masked_mean_pooling(last_hidden_state, attention_mask):
  mask = attention_mask.unsqueeze(-1).float()
  masked = last_hidden_state * mask
  summed = masked.sum(dim=1)
  counts = mask.sum(dim=1).clamp(min=1e-9)
  return summed / counts

def process_data(df, tokenizer, model, device, set_name, task, batch_size=64):
  n = len(df)
  out = np.zeros((n, 768))

  smiles_list = df.SMILES.values.tolist()

  for start in tqdm(range(0, n, batch_size)):
    end = min(start + batch_size, n)
    batch_smiles = smiles_list[start:end]

    tok = tokenizer(batch_smiles, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    last_hidden = outputs.hidden_states[-1]
    pooled = masked_mean_pooling(last_hidden, attention_mask)
    out[start:end] = pooled.cpu().numpy()

  np.save(f'../features/{set_name}-text-{task}.npy', out)

if __name__ == '__main__':
  main()
