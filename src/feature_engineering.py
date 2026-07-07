import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from functools import lru_cache
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch_geometric.data import Data
from multiprocessing import Pool

from config import ProjectConfig

_cfg = ProjectConfig()
PROCESSED_DATA = _cfg.paths.processed

HETERO_NUMS = frozenset({7, 8, 9, 15, 16, 17, 35, 53})


# Atom / bond feature vectors

# Atom (12): atomic_num, degree, formal_charge, hybridisation, aromatic,
#            num_Hs, in_ring, min_ring_size/8, mass/100, radical_e,
#            explicit_valence, is_heteroatom
# Edge  (4): bond_type, conjugated, in_ring, stereo


def atom_vec(atom, ring_info) -> list:
    """12-dim feature vector for one atom."""
    idx = atom.GetIdx()
    in_ring = ring_info.NumAtomRings(idx) > 0
    min_rs = 0
    if in_ring:
        for sz in range(3, 9):
            if ring_info.IsAtomInRingOfSize(idx, sz):
                min_rs = sz
                break
    return [
        atom.GetAtomicNum(),       atom.GetTotalDegree(),
        atom.GetFormalCharge(),     int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),  atom.GetTotalNumHs(),
        int(in_ring),              min_rs / 8.0,
        atom.GetMass() / 100.0,    atom.GetNumRadicalElectrons(),
        atom.GetExplicitValence(),  int(atom.GetAtomicNum() in HETERO_NUMS),
    ]


def bond_vec(bond) -> list:
    """4-dim feature vector for one bond."""
    return [int(bond.GetBondType()),  int(bond.GetIsConjugated()),
            int(bond.IsInRing()),     int(bond.GetStereo())]


# Molecular Graph

def smiles_to_graph(smiles):
    """SMILES - PyG Data with 12 atom features and 4 edge features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    ri = mol.GetRingInfo()
    n_bonds = mol.GetNumBonds()

    # Single tensor from list — no per-atom tensor allocation
    x = torch.tensor([atom_vec(a, ri) for a in mol.GetAtoms()], dtype=torch.float)

    if n_bonds == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float)
    else:
        src, dst, attrs = [], [], []
        for b in mol.GetBonds():
            i, j, bv = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_vec(b)
            src += [i, j]
            dst += [j, i]
            attrs += [bv, bv]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@lru_cache(maxsize=8)
def morgan_generator(radius, n_bits):
    """Cached Morgan fingerprint generator. Reused across calls."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Morgan fingerprint as a uint8 numpy array, or None on parse failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = morgan_generator(radius, n_bits).GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# Lipinski descriptors - used by EDA notebook

def lipinski_descriptors(smiles):
    """Lipinski Rule of 5 descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'h_bond_donors': Lipinski.NumHDonors(mol),
        'h_bond_acceptors': Lipinski.NumHAcceptors(mol),
        'rotatable_bonds': Lipinski.NumRotatableBonds(mol),
        'tpsa': Descriptors.TPSA(mol),
    }


# Scaffold Analysis

def bemis_murcko_scaffold(smiles):
    """Extract Bemis-Murcko scaffold."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


# Parallel graph conversion

def smiles_to_graph_safe(smiles):
    """Worker-safe wrapper."""
    try:
        return smiles_to_graph(smiles)
    except Exception:
        return None


def pool_map(fn, items, n_workers):
    """Parallel map with sequential fallback."""
    if n_workers <= 1 or len(items) < 200:
        return list(map(fn, items))
    try:
        with Pool(n_workers) as p:
            return p.map(fn, items, chunksize=256)
    except Exception:
        return list(map(fn, items))


def parallel_smiles_to_graphs_ordered(smiles_list, n_workers=None):
    """SMILES to PyG graphs in parallel. Preserves None for alignment."""
    return pool_map(smiles_to_graph_safe, smiles_list,
                    n_workers or _cfg.train.num_workers)


# Model Splits

def split_dataset(df, train_frac=0.8, val_frac=0.1, seed=42):
    """Random split into train/val/test."""
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    t1, t2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return shuffled[:t1], shuffled[t1:t2], shuffled[t2:]


def scaffold_fold_labels(smiles, train_frac=0.8, val_frac=0.1, seed=42):
    """One fold per unique SMILES, shared across inputs so a scaffold never
    crosses folds between organisms."""
    uniq = pd.Series(pd.unique(pd.Series(smiles)))
    scaff = uniq.map(bemis_murcko_scaffold)
    solo = scaff.isna() | (scaff == "")
    key = scaff.mask(solo, "solo:" + uniq.astype(str))
    groups = [sub.index.to_numpy() for _, sub in key.groupby(key)]
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    sizes = np.array([len(g) for g in groups], dtype=np.int64)
    prefix = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    n = len(uniq)
    train_end, val_end = int(n * train_frac), int(n * (train_frac + val_frac))
    labels = np.empty(n, dtype=object)
    for members, start in zip(groups, prefix):
        labels[members] = ("train" if start < train_end
                           else "val" if start < val_end else "test")
    return pd.Series(labels, index=uniq.values)