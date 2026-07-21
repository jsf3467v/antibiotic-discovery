"""Unit tests for the reward component functions.

Covers the deterministic scoring math — size gate, fingerprint similarity
index, novelty/resistance sigmoids, QED, atom composition — none of which
need the trained GNN, surrogate, or external reference datasets.
"""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from src.rewards import (
    FingerprintIndex,
    size_gate,
    novelty,
    resistance,
    qed_score,
    atom_fractions,
    composition_penalty,
    composition_arrays,
)
from config import CompositionConfig

OCTANE = "CCCCCCCC"   # 8 heavy atoms
BENZENE = "c1ccccc1"
ETHANOL = "CCO"


def _fp(smiles):
    return AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smiles), 2, nBits=2048)


def test_size_gate_peaks_at_center():
    """The band gate peaks at 1.0 when heavy-atom count equals the center."""
    mol = Chem.MolFromSmiles(OCTANE)            # 8 heavy atoms
    assert size_gate(mol, center=8.0) == pytest.approx(1.0, abs=1e-6)


def test_size_gate_falls_off_both_sides():
    """Two-sided band, the gate is highest at the center and lower for
    molecules both smaller and larger than it."""
    center = 8.0
    at_center = size_gate(Chem.MolFromSmiles(OCTANE), center=center)        # 8
    below = size_gate(Chem.MolFromSmiles("CCCC"), center=center)            # 4
    above = size_gate(Chem.MolFromSmiles("CCCCCCCCCCCC"), center=center)    # 12
    assert below < at_center
    assert above < at_center


def test_size_gate_respects_floor():
    tiny = Chem.MolFromSmiles("C")
    assert size_gate(tiny, center=8.0, floor=0.2) >= 0.2


def test_fingerprint_index_identity_and_empty():
    idx = FingerprintIndex([ETHANOL, BENZENE])
    assert len(idx) == 2
    # Querying with a molecule that's in the store gives perfect similarity.
    assert idx.max_tanimoto(_fp(ETHANOL)) == pytest.approx(1.0)
    # An empty store reports zero similarity rather than raising.
    assert FingerprintIndex([]).max_tanimoto(_fp(ETHANOL)) == 0.0


def test_novelty_low_for_known_high_for_empty():
    db = FingerprintIndex([ETHANOL])
    # Identical to a known compound -> not novel -> well below 0.5.
    assert novelty(_fp(ETHANOL), db) < 0.5
    # No reference set -> everything counts as novel.
    assert novelty(_fp(ETHANOL), FingerprintIndex([])) == 1.0


def test_resistance_low_for_known_substrate():
    card = FingerprintIndex([BENZENE])
    assert resistance(_fp(BENZENE), card) < 0.5


def test_qed_in_unit_interval():
    q = qed_score(Chem.MolFromSmiles(BENZENE))
    assert 0.0 <= q <= 1.0


def test_atom_fractions_all_carbon():
    """Benzene is pure carbon: the C fraction is 1.0 and fractions sum to 1."""
    symbols = ["C", "N", "O"]
    fr = atom_fractions(Chem.MolFromSmiles(BENZENE), symbols)
    assert fr[0] == pytest.approx(1.0)
    assert fr.sum() == pytest.approx(1.0)


def test_composition_penalty_self_is_one():
    """Observed == reference gives exp(0) == 1.0."""
    symbols, ref, tau, scale = composition_arrays(CompositionConfig())
    # Build a molecule and score it against its OWN fractions.
    mol = Chem.MolFromSmiles(BENZENE)
    own = atom_fractions(mol, symbols)
    assert composition_penalty(mol, own, symbols, tau) == pytest.approx(1.0)


def test_composition_penalty_disabled_for_nonpositive_tau():
    mol = Chem.MolFromSmiles(BENZENE)
    symbols, ref, _, _ = composition_arrays(CompositionConfig())
    assert composition_penalty(mol, ref, symbols, tau=0.0) == 1.0