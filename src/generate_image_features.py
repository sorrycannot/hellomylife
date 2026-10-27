import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
import click

VALID_TASKS = ['classification', 'regression']

preprocess = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
  )
])

@click.command()
@click.option(
  '--task',
  type=click.Choice(VALID_TASKS, case_sensitive=False),
  required=True,
  help="The task"
)

def main(task):
  train = pd.read_csv(f'../dataset/train_{task}.csv')
  val = pd.read_csv(f'../dataset/val_{task}.csv')
  test = pd.read_csv(f'../dataset/test_{task}.csv')

  device = torch.device("cuda:0")

  encoder = get_resnet50_encoder(device)

  process_data(train, encoder, device, "train", task)
  process_data(val,   encoder, device, "val", task)
  process_data(test,  encoder, device, "test", task)

def get_resnet50_encoder(device):
    model = models.resnet50(weights="IMAGENET1K_V1")
    encoder = nn.Sequential(*list(model.children())[:-1])
    encoder.eval().to(device)
    return encoder

def smiles_to_image(smiles):
  mol = Chem.MolFromSmiles(smiles)
  if mol is None:
    return Image.new("RGB", (300,300), color=(0,0,0))

  img = Draw.MolToImage(mol, size=(300, 300))
  return img

def process_data(df, encoder, device, set_name, task, batch_size=32):
  n = len(df)
  out = np.zeros((n, 2048), dtype=np.float32)

  smiles_list = df.SMILES.values.tolist()

  for start in tqdm(range(0, n, batch_size)):
    end = min(start + batch_size, n)
    batch = smiles_list[start:end]
 
    imgs = []
    for s in batch:
      img = smiles_to_image(s)
      tensor = preprocess(img)
      imgs.append(tensor)

    x = torch.stack(imgs).to(device)

    with torch.no_grad():
      feats = encoder(x).squeeze(-1).squeeze(-1)

    out[start:end] = feats.cpu().numpy()

  np.save(f'../features/{set_name}-image-{task}.npy', out)

if __name__ == "__main__":
    main()
