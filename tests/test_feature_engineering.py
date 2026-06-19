"""Unit tests for graph featurization and molecular descriptors.
"""
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from src.feature_engineering import (
    smiles_to_graph,
    morgan_fingerprint,
    lipinski_descriptors,
    bemis_murcko_scaffold,
    atom_vec,
    bond_vec,
    split_dataset,
)

BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
ETHANOL = "CCO"
INVALID = "this_is_not_a_smiles"


def test_smiles_to_graph_benzene_shapes():
    """Benzene: 6 atoms, 6 bonds. Edges are stored both directions (12),
    atom features are 12-dim, edge features are 4-dim."""
    g = smiles_to_graph(BENZENE)
    assert g is not None
    assert g.x.shape == (6, 12)
    assert g.edge_index.shape == (2, 12)
    assert g.edge_attr.shape == (12, 4)


def test_smiles_to_graph_invalid_returns_none():
    assert smiles_to_graph(INVALID) is None


def test_atom_and_bond_vector_dims():
    """Feature widths must stay 12 (atom) and 4 (bond) — the GNN's input
    dims are hard-coded to these in config."""
    mol = Chem.MolFromSmiles(ETHANOL)
    ri = mol.GetRingInfo()
    assert len(atom_vec(mol.GetAtomWithIdx(0), ri)) == 12
    assert len(bond_vec(mol.GetBondWithIdx(0))) == 4


def test_morgan_fingerprint_shape_and_determinism():
    fp1 = morgan_fingerprint(ETHANOL)
    fp2 = morgan_fingerprint(ETHANOL)
    assert fp1.shape == (2048,)
    assert fp1.dtype == np.uint8
    assert np.array_equal(fp1, fp2)          # deterministic
    assert fp1.sum() > 0                      # not all-zero
    assert morgan_fingerprint(INVALID) is None


def test_bemis_murcko_scaffold_strips_substituent():
    """Toluene's scaffold is benzene — the methyl is removed."""
    scaffold = bemis_murcko_scaffold(TOLUENE)
    assert scaffold == Chem.CanonSmiles(BENZENE)
    assert bemis_murcko_scaffold(INVALID) is None


def test_lipinski_descriptors_keys_and_invalid():
    d = lipinski_descriptors(ETHANOL)
    assert set(d) == {
        "molecular_weight", "logp", "h_bond_donors",
        "h_bond_acceptors", "rotatable_bonds", "tpsa",
    }
    assert d["molecular_weight"] == pytest.approx(46.07, abs=0.1)
    assert d["h_bond_donors"] == 1                 # the -OH
    assert lipinski_descriptors(INVALID) is None


def test_split_dataset_is_a_partition():
    """train/val/test must be disjoint and cover every row exactly once."""
    df = pd.DataFrame({"id": range(100)})
    train, val, test = split_dataset(df, train_frac=0.8, val_frac=0.1, seed=42)
    assert len(train) + len(val) + len(test) == 100
    ids = set(train["id"]) | set(val["id"]) | set(test["id"])
    assert ids == set(range(100))                  # nothing lost or duplicated


def test_split_dataset_is_seed_deterministic():
    df = pd.DataFrame({"id": range(50)})
    a = split_dataset(df, seed=7)[0]["id"].tolist()
    b = split_dataset(df, seed=7)[0]["id"].tolist()
    assert a == b
