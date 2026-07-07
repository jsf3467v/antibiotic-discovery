"""
Multi-objective reward for molecular generation. Potency surrogate and
GNN both produce per-organism log10(MIC), routed through the same
sigmoid-then-mean wrapper.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import QED
from torch_geometric.data import Batch
from typing import List, Optional

from config import RewardWeights, CompositionConfig, ORGANISM_KEYS, ProjectConfig
from src.gnn import log_mic_to_prob_torch
from src.feature_engineering import smiles_to_graph, morgan_generator

try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except Exception:
    sascorer = None


# Fingerprint store


class FingerprintIndex:
    """Bulk Morgan FP store with vectorized max-Tanimoto query."""

    def __init__(self, smiles_list, radius: int = 2, n_bits: int = 2048):
        self.fps = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                self.fps.append(morgan_generator(radius, n_bits)
                                .GetFingerprint(mol))
        self.radius = radius
        self.n_bits = n_bits

    def max_tanimoto(self, query_fp) -> float:
        if not self.fps:
            return 0.0
        return max(DataStructs.BulkTanimotoSimilarity(query_fp, self.fps))

    def __len__(self) -> int:
        return len(self.fps)


def fp_array(smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Morgan fingerprint as uint8 numpy array, or None on parse failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = morgan_generator(2, n_bits).GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_pair(smiles: str, radius: int = 2, n_bits: int = 2048):
    """(mol, BitVect) for one SMILES, or (None, None) on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    return mol, morgan_generator(radius, n_bits).GetFingerprint(mol)


# Per-organism potency surrogate

class PotencySurrogate(nn.Module):
    """Fingerprint - per-organism log10(MIC); .probability matches gnn_probability's wrapper."""

    def __init__(self, fp_dim: int = 2048, hidden: int = 256,
                 n_organisms: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_organisms),
        )

    def forward(self, x):
        return self.net(x)

    def probability(self, fp_arr: np.ndarray, device,
                    threshold: float = 10.0) -> float:
        with torch.no_grad():
            t = torch.as_tensor(fp_arr, dtype=torch.float32,
                                device=device).unsqueeze(0)
            log_mic = self.forward(t)
            return float(log_mic_to_prob_torch(
                log_mic, threshold).mean().item())

    def batch_probability(self, fp_matrix: np.ndarray, device,
                          threshold: float = 10.0) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(fp_matrix, dtype=torch.float32, device=device)
            log_mic = self.forward(t)
            return log_mic_to_prob_torch(
                log_mic, threshold).mean(dim=1).cpu().numpy()


# GNN-based potency


def gnn_probability(smiles: str, gnn_model, device,
                    threshold: float = 10.0) -> float:
    """Mean per-organism active probability from the regression GNN."""
    graph = smiles_to_graph(smiles)
    if graph is None:
        return 0.0
    graph = graph.to(device)
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        log_mic = gnn_model(graph.x, graph.edge_index, batch,
                            edge_attr=graph.edge_attr)
        probs = torch.stack([log_mic_to_prob_torch(v, threshold)
                             for v in log_mic.values()])
    return float(probs.mean().item())


def gnn_batch_log_mic(smiles_list, gnn_model, device,
                      chunk: int = 128) -> np.ndarray:
    """Per-organism log10(MIC) for a batch; shape  -n_smiles, n_organisms, NaN for parse failures."""
    graphs = [smiles_to_graph(s) for s in smiles_list]
    keep = np.array([g is not None for g in graphs])
    valid = [g for g in graphs if g is not None]
    organisms = list(ORGANISM_KEYS)
    out = np.full((len(smiles_list), len(organisms)),
                  np.nan, dtype=np.float32)
    if not valid:
        return out
    is_mps = device.type == "mps"
    parts = {k: [] for k in organisms}
    gnn_model.eval()
    with torch.no_grad():
        for i, start in enumerate(range(0, len(valid), chunk)):
            sub = Batch.from_data_list(valid[start:start + chunk]).to(device)
            ea = sub.edge_attr if hasattr(sub, "edge_attr") else None
            log_mic = gnn_model(sub.x, sub.edge_index, sub.batch,
                                edge_attr=ea)
            for k in organisms:
                parts[k].append(log_mic[k].cpu().numpy())
            if is_mps and (i + 1) % 10 == 0:
                torch.mps.empty_cache()
    out[keep] = np.stack(
        [np.concatenate(parts[k]) for k in organisms], axis=1)
    return out


# Component scores

def size_gate(mol, center: float = 8.0, steepness: float = 0.5,
              floor: float = 0.0) -> float:
    n = mol.GetNumHeavyAtoms()
    raw = 1.0 / (1.0 + np.exp(-steepness * (n - center)))
    return floor + (1.0 - floor) * raw


def applicability_gate(query_fp, training_index, lo: float = 0.3,
                       hi: float = 0.5) -> float:
    if training_index is None or not training_index.fps:
        return 1.0
    sim = training_index.max_tanimoto(query_fp)
    if sim >= hi:
        return 1.0
    if sim <= lo or hi <= lo:
        return 0.0
    return (sim - lo) / (hi - lo)


def novelty(query_fp, drugbank_index, midpoint: float = 0.6,
            steepness: float = 10.0) -> float:
    if not drugbank_index.fps:
        return 1.0
    sim = drugbank_index.max_tanimoto(query_fp)
    return float(1.0 / (1.0 + np.exp(steepness * (sim - midpoint))))


def resistance(query_fp, card_index, midpoint: float = 0.5,
               steepness: float = 10.0) -> float:
    if not card_index.fps:
        return 1.0
    sim = card_index.max_tanimoto(query_fp)
    return float(1.0 / (1.0 + np.exp(steepness * (sim - midpoint))))


def qed_score(mol) -> float:
    return float(QED.qed(mol))


def sa_score(mol) -> float:
    if sascorer is None:
        return 0.5
    return (10.0 - sascorer.calculateScore(mol)) / 9.0


# Atom-composition penalty

def atom_fractions(mol, symbols: List[str]) -> np.ndarray:
    """Per-symbol atom fraction over heavy atoms."""
    n = mol.GetNumHeavyAtoms()
    if n == 0:
        return np.zeros(len(symbols), dtype=np.float32)
    counts = np.zeros(len(symbols), dtype=np.float32)
    sym_index = {s: i for i, s in enumerate(symbols)}
    for atom in mol.GetAtoms():
        idx = sym_index.get(atom.GetSymbol())
        if idx is not None:
            counts[idx] += 1.0
    return counts / float(n)


def composition_penalty(mol, ref_fractions: np.ndarray,
                        symbols: List[str], tau: float) -> float:
    """exp(-||obs - ref||^2 / tau^2). 1.0 when fractions match the
    reference, decays toward 0 as squared deviation grows."""
    if mol is None or tau <= 0:
        return 1.0
    obs = atom_fractions(mol, symbols)
    sq = float(np.sum((obs - ref_fractions) ** 2))
    return float(np.exp(-sq / (tau ** 2)))


def composition_arrays(cfg: Optional[CompositionConfig]):
    """Unpack a CompositionConfig to - symbols, ref_fractions, tau."""
    if cfg is None:
        return [], np.zeros(0, dtype=np.float32), 0.0
    symbols = list(cfg.reference.keys())
    ref = np.array([cfg.reference[s] for s in symbols], dtype=np.float32)
    return symbols, ref, cfg.tau


# Composite reward

class RewardFunction:
    """Weighted sum of potency, novelty, resistance, QED, SA, gated by molecule size and composition."""

    def __init__(self, gnn_model, device,
                 drugbank_index: FingerprintIndex,
                 card_index: FingerprintIndex,
                 training_index: Optional[FingerprintIndex] = None,
                 weights: Optional[RewardWeights] = None,
                 size_center: float = 8.0,
                 size_steepness: float = 0.5,
                 gate_floor: float = 0.0,
                 potency_floor: float = 0.0,
                 surrogate: Optional[PotencySurrogate] = None,
                 mic_threshold: float = 10.0,
                 composition: Optional[CompositionConfig] = None):
        self.gnn = gnn_model
        self.device = device
        self.drugbank = drugbank_index
        self.card = card_index
        self.training = training_index
        self.w = weights if weights is not None else RewardWeights()
        self.size_center = size_center
        self.size_steepness = size_steepness
        self.gate_floor = gate_floor
        self.potency_floor = potency_floor
        self.surrogate = surrogate
        self.mic_threshold = mic_threshold
        self.atom_symbols, self.atom_ref, self.atom_tau = (
            composition_arrays(composition))

    def potency(self, smiles: str, fp) -> float:
        if self.surrogate is not None:
            arr = np.zeros(fp.GetNumBits(), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return self.surrogate.probability(
                arr, self.device, self.mic_threshold)
        return gnn_probability(
            smiles, self.gnn, self.device, self.mic_threshold)

    def __call__(self, smiles) -> float:
        if smiles is None:
            return 0.0
        mol, fp = morgan_pair(smiles)
        if mol is None or fp is None:
            return 0.0
        return self.weighted_total(self.components(mol, fp, smiles))

    def components(self, mol, fp, smiles) -> dict:
        return {
            "potency": self.potency(smiles, fp),
            "novelty": novelty(fp, self.drugbank),
            "resistance": resistance(fp, self.card),
            "qed": qed_score(mol),
            "sa": sa_score(mol),
            "size_gate": size_gate(mol, self.size_center,
                                   self.size_steepness, self.gate_floor),
            "ad_gate": applicability_gate(fp, self.training),
            "composition": composition_penalty(
                mol, self.atom_ref, self.atom_symbols, self.atom_tau),
        }

    def weighted_total(self, c: dict) -> float:
        if self.potency_floor > 0 and c["potency"] < self.potency_floor:
            return 0.0
        eff_comp = 1.0 - c["size_gate"] * (1.0 - c["composition"])
        raw = (self.w.potency * c["potency"] * c["ad_gate"]
               + self.w.novelty * c["novelty"]
               + self.w.resistance * c["resistance"]
               + self.w.qed * c["qed"]
               + self.w.sa_score * c["sa"])
        return float(c["size_gate"] * eff_comp * raw)

    def detailed(self, smiles) -> dict:
        """Per-component breakdown plus total."""
        mol, fp = morgan_pair(smiles)
        if mol is None or fp is None:
            return {k: 0.0 for k in (
                "total", "potency", "novelty", "resistance", "qed",
                "sa", "size_gate", "ad_gate", "composition")}
        c = self.components(mol, fp, smiles)
        c["total"] = self.weighted_total(c)
        return c

cfg = ProjectConfig()


def drugbank_index() -> "FingerprintIndex":
    """DrugBank antibiotics fingerprint index for novelty scoring."""
    df = pd.read_csv(cfg.paths.processed / "drugbank_antibiotics.csv")
    return FingerprintIndex(df["smiles"].dropna().tolist())


def card_index() -> "FingerprintIndex":
    """CARD substrates fingerprint index for resistance scoring. Empty
    when the file is missing or malformed.
    """
    path = cfg.paths.processed / "card_substrates.csv"
    if not path.exists() or path.stat().st_size < 10:
        return FingerprintIndex([])
    df = pd.read_csv(path)
    if df.empty or "smiles" not in df.columns:
        return FingerprintIndex([])
    return FingerprintIndex(df["smiles"].dropna().tolist())


def training_reward(gnn, device,
                    surrogate: "PotencySurrogate") -> "RewardFunction":
    """RL training reward; surrogate-based potency, no AD gate, size center mutated by the phase loop."""
    return RewardFunction(
        gnn_model=gnn, device=device,
        drugbank_index=drugbank_index(),
        card_index=card_index(),
        weights=cfg.rewards,
        size_center=cfg.rl.size_center_phase1,
        size_steepness=cfg.rl.size_steepness,
        gate_floor=cfg.rl.gate_floor,
        potency_floor=0.0,
        surrogate=surrogate,
        mic_threshold=cfg.data.mic_threshold,
        composition=cfg.composition,
    )


def evaluation_reward(gnn, device) -> "RewardFunction":
    """Canonical reward for RL/baseline comparison; GNN-direct potency, no AD gate, size center fixed."""
    return RewardFunction(
        gnn_model=gnn, device=device,
        drugbank_index=drugbank_index(),
        card_index=card_index(),
        weights=cfg.rewards,
        size_center=cfg.rl.size_center_phase2_end,
        size_steepness=cfg.rl.size_steepness,
        gate_floor=cfg.rl.gate_floor,
        potency_floor=0.0,
        mic_threshold=cfg.data.mic_threshold,
        composition=cfg.composition,
    )