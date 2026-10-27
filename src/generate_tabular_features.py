import click
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

BASE_DATA_PATH = '../dataset/'
BASE_FEATURES_PATH = '../features/{}-tabular-{}.npy'
VALID_TASKS = ['classification', 'regression']

@click.command()
@click.option(
  '--task',
  type=click.Choice(VALID_TASKS, case_sensitive=False),
  required=True,
  help="The task"
)
def main(task):
  for set_name in ['train', 'val', 'test']:
    input_path = f"{BASE_DATA_PATH}{set_name}_{task}.csv"
    df = pd.read_csv(input_path)
    smiles_list = df.SMILES.values.tolist()

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    calc = MoleculeDescriptors.MolecularDescriptorCalculator([i[0] for i in Descriptors.descList])
    desc_list = [calc.CalcDescriptors(mol) for mol in tqdm(mols)]
    desc_2d = np.array(desc_list, dtype=np.float32)
    desc_2d = np.nan_to_num(desc_2d, nan=0.0, posinf=0.0, neginf=0.0)

    desc_list = [MACCSkeys.GenMACCSKeys(mol) for mol in tqdm(mols)]
    desc_bitvects = [np.array(list(maccs.ToBitString()), dtype=int) for maccs in desc_list]
    desc_maccs = np.array(desc_bitvects, dtype=np.int8)

    combined = np.concatenate([desc_2d, desc_maccs], axis=1)
    print(f'{set_name} final shape: {combined.shape}')

    np.save(BASE_FEATURES_PATH.format(set_name, task), combined)

if __name__ == "__main__":
  main()
