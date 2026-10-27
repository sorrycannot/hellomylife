"""Chemical standardization matching revision utils_chem."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import inchi

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _HAS_STANDARDIZE = True
except Exception:
    _HAS_STANDARDIZE = False


@dataclass
class MolId:
    canonical_smiles: str
    inchikey: str


def _cleanup_mol(mol: Chem.Mol) -> Chem.Mol:
    if not _HAS_STANDARDIZE:
        return mol
    mol = rdMolStandardize.Cleanup(mol)
    parent = rdMolStandardize.FragmentParent(mol)
    uncharger = rdMolStandardize.Uncharger()
    return uncharger.uncharge(parent)


def mol_from_smiles_or_inchi(smiles: Optional[str], inchi_str: Optional[str] = None):
    mol = None
    if smiles and isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
    if mol is None and inchi_str and isinstance(inchi_str, str) and inchi_str.startswith("InChI="):
        try:
            mol = inchi.MolFromInchi(inchi_str, sanitize=True, removeHs=True)
        except Exception:
            mol = None
    return mol


def standardize_and_identify(smiles: Optional[str], inchi_str: Optional[str] = None) -> Optional[MolId]:
    mol = mol_from_smiles_or_inchi(smiles, inchi_str)
    if mol is None:
        return None
    try:
        mol = _cleanup_mol(mol)
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        ik = inchi.MolToInchiKey(mol)
        if not ik:
            return None
        return MolId(canonical_smiles=can, inchikey=ik)
    except Exception:
        return None
