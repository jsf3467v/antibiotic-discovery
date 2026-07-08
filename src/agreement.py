"""
Surrogate-to-GNN agreement on the generated pool, the molecules the surrogate
actually stands in for during rollouts. This file reports the Pearson correlation and the
binary-call agreement between the two potency signals under the same
sigmoid-then-mean wrapper the reward uses. Both signals come from the same
SMILES, so the pool is held out from neither yet biased toward neither.
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from config import ProjectConfig, pick_device, release_cache, ORGANISM_KEYS
from src.gnn import log_mic_to_prob_torch
from src.feature_engineering import morgan_fingerprint
from src.rewards import PotencySurrogate, gnn_batch_log_mic
from src.evaluate import trained_gnn, unique_canonical

cfg = ProjectConfig()
logger = logging.getLogger(__name__)


def pool_smiles():
    """Unique canonical SMILES from the generated RL pool."""
    path = cfg.paths.results / "generated_molecules.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        return []
    return unique_canonical(df["smiles"].dropna().astype(str).tolist())


def surrogate_model(device):
    """Surrogate weights from surrogate.pt, in eval mode."""
    sc = cfg.surrogate
    model = PotencySurrogate(fp_dim=sc.fp_dim, hidden=sc.hidden,
                             n_organisms=len(ORGANISM_KEYS)).to(device)
    model.load_state_dict(torch.load(cfg.paths.models / "surrogate.pt",
                                     map_location=device, weights_only=True))
    return model.eval()


def fingerprints(smiles):
    """(matrix, mask): Morgan fingerprint rows; mask marks parseable SMILES."""
    sc = cfg.surrogate
    rows = [morgan_fingerprint(s, sc.fp_radius, sc.fp_dim) for s in smiles]
    mask = np.array([r is not None for r in rows])
    matrix = np.zeros((len(smiles), sc.fp_dim), dtype=np.float32)
    valid = [r for r in rows if r is not None]
    if valid:
        matrix[mask] = np.asarray(valid, dtype=np.float32)
    return matrix, mask


def gnn_potency(smiles, gnn, device):
    """(prob, mask): mean-organism active probability from the GNN."""
    log_mic = gnn_batch_log_mic(smiles, gnn, device)
    mask = ~np.isnan(log_mic).any(axis=1)
    prob = log_mic_to_prob_torch(
        torch.from_numpy(log_mic), cfg.data.mic_threshold).mean(dim=1).numpy()
    return prob, mask


def surrogate_potency(matrix, surrogate, device):
    """Mean-organism active probability from the surrogate."""
    return surrogate.batch_probability(matrix, device, cfg.data.mic_threshold)


def pearson(a, b):
    """Pearson correlation; nan when either vector has no variance."""
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def binary_agreement(a, b, thr):
    """Fraction of molecules whose two calls fall on the same side of thr."""
    return float(((a >= thr) == (b >= thr)).mean())


def summary(surr_prob, gnn_prob):
    """Agreement metrics between the two potency vectors."""
    thr = cfg.surrogate.agreement_threshold
    return {"n": int(surr_prob.size),
            "pearson_r": pearson(surr_prob, gnn_prob),
            "binary_agreement": binary_agreement(surr_prob, gnn_prob, thr),
            "gnn_active_fraction": float((gnn_prob >= thr).mean()),
            "mic_threshold": cfg.data.mic_threshold,
            "agreement_threshold": thr}


def potencies(smiles, device):
    """Aligned surrogate and GNN potency vectors over molecules both can score."""
    gnn = trained_gnn(device)
    surrogate = surrogate_model(device)
    matrix, fp_mask = fingerprints(smiles)
    gnn_prob, gnn_mask = gnn_potency(smiles, gnn, device)
    release_cache(device)
    keep = fp_mask & gnn_mask
    if not keep.any():
        return None, None
    surr = surrogate_potency(matrix[keep], surrogate, device)
    release_cache(device)
    return surr, gnn_prob[keep]


def main():
    device = pick_device()
    cfg.ensure_dirs()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    logger.info(f"Device: {device}")

    smiles = pool_smiles()
    if not smiles:
        logger.warning("No generated pool found. Run train_rl.py first.")
        return
    logger.info(f"Generated pool: {len(smiles):,} unique mols")

    surr_prob, gnn_prob = potencies(smiles, device)
    if surr_prob is None:
        logger.warning("No molecules scored by both models.")
        return
    result = summary(surr_prob, gnn_prob)
    logger.info(f"Scored: {result['n']:,}   Pearson r: {result['pearson_r']:.4f}   "
                f"Binary agreement: {result['binary_agreement']:.4f}")
    out = cfg.paths.metrics / "surrogate_agreement.csv"
    pd.DataFrame([result]).to_csv(out, index=False)
    logger.info(f"Saved {out.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()