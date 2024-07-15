import sys
import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Descriptors import MolWt
from standardiser import standardise

def read_smiles(smi, standardize, min_mw=100, max_mw=1000):
    try:
        mol = Chem.MolFromSmiles(smi)
        if standardize:
            mol = standardise.run(mol)
    except Exception:
        return None
    if not mol:
        return None
    mw = MolWt(mol)
    if mw < min_mw or mw > max_mw:
        return None
    ik = Chem.rdinchi.InchiToInchiKey(Chem.rdinchi.MolToInchi(mol)[0])
    smi = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None
    return ik, smi

def map_smiles(smiles, n_cpus=None, standardize=True, min_mw=100, max_mw=1000):

    if n_cpus is None:
        res = []
        for smile in tqdm(smiles):
            res.append(read_smiles(smile, standardize=standardize, min_mw=min_mw, max_mw=max_mw))
    else:
        smile_mapping = partial(read_smiles, standardize=standardize, min_mw=min_mw, max_mw=max_mw)
        res = Pool(n_cpus).map(smile_mapping, smiles)

    return res
