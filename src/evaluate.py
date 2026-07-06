"""
Evaluation shared across GNN test-set scoring, RL generation
quality, baselines, and statistical tests. Probability computation
routes through gnn.log_mic_to_prob_torch; pool scoring routes
through evaluation_reward.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

from config import ProjectConfig, pick_device, release_cache
from src.gnn import MultiTaskGNN, log_mic_to_prob_torch
from src.feature_engineering import morgan_generator
from src.train_gnn import mic_splits

_cfg = ProjectConfig()
_MPS_FLUSH_EVERY = 10
_MPS_FLUSH_SCORING = 256

# Public constants imported by eval_rl.py and eval_baselines.py
TOP_N = 10
COMPONENT_KEYS = ("total", "potency", "novelty", "resistance", "qed", "sa")


# Trained checkpoint

def trained_gnn(device) -> MultiTaskGNN:
    """Load gnn_best.pt into a fresh MultiTaskGNN, return in eval mode."""
    model = MultiTaskGNN(_cfg.atom, _cfg.gnn).to(device)
    ckpt = _cfg.paths.models / "gnn_best.pt"
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


# Classification and regression metrics

def auroc(labels, probs):
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, probs))


def auprc(labels, probs):
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, probs))


def classifier_metrics(labels, probs):
    """AUROC, AUPRC, precision, recall at 0.5 threshold."""
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "auroc": auroc(labels, probs),
        "auprc": auprc(labels, probs),
        "precision": float(precision),
        "recall": float(recall),
    }


def regression_metrics(targets, preds):
    """RMSE and MAE on log10(MIC)."""
    diff = preds - targets
    return {
        "rmse_logmic": float(np.sqrt(np.mean(diff ** 2))),
        "mae_logmic": float(np.mean(np.abs(diff))),
    }


# GNN forward pass and report

def gnn_predictions(model, dataloader, device):
    """Per-organism log_mic predictions, log_mic targets, binary labels, masks."""
    log_mic_per = {"saureus": [], "ecoli": []}
    target_per = {"saureus": [], "ecoli": []}
    label_per = {"saureus": [], "ecoli": []}
    mask_per = {"saureus": [], "ecoli": []}
    model.eval()
    is_mps = device.type == "mps"
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            ea = batch.edge_attr if hasattr(batch, "edge_attr") else None
            preds = model(batch.x, batch.edge_index, batch.batch, edge_attr=ea)
            for key in preds:
                log_mic_per[key].append(preds[key].cpu().numpy())
                target_per[key].append(
                    getattr(batch, f"logmic_{key}").cpu().numpy())
                label_per[key].append(
                    getattr(batch, f"y_{key}").cpu().numpy())
                mask_per[key].append(
                    getattr(batch, f"mask_{key}").cpu().numpy())
            if is_mps and (i + 1) % _MPS_FLUSH_EVERY == 0:
                torch.mps.empty_cache()
    return log_mic_per, target_per, label_per, mask_per


def gnn_report(model, dataloader, device, threshold=10.0) -> Dict[str, dict]:
    """Per-organism regression and derived classification metrics."""
    log_mic_per, target_per, label_per, mask_per = gnn_predictions(
        model, dataloader, device)
    report = {}
    for key in log_mic_per:
        m = np.concatenate(mask_per[key]).astype(bool)
        lm = np.concatenate(log_mic_per[key])[m]
        tg = np.concatenate(target_per[key])[m]
        lb = np.concatenate(label_per[key])[m]
        pr = log_mic_to_prob_torch(torch.tensor(lm), threshold).numpy()
        out = classifier_metrics(lb, pr)
        out.update(regression_metrics(tg, lm))
        report[key] = out
    return report


# Pool scoring shared by eval_rl.py and eval_baselines.py

def scored_descending(smiles: List[str],
                      reward_fn) -> Tuple[List[str], np.ndarray]:
    """Score under reward_fn; return SMILES and scores sorted descending.
    Periodic MPS cache flush keeps the allocator pool bounded across
    20k+-molecule scoring runs."""
    scores = np.zeros(len(smiles), dtype=np.float64)
    is_mps = torch.backends.mps.is_available()
    for i, s in enumerate(smiles):
        scores[i] = reward_fn(s)
        if is_mps and (i + 1) % _MPS_FLUSH_SCORING == 0:
            torch.mps.empty_cache()
    order = np.argsort(-scores)
    return [smiles[i] for i in order], scores[order]


def pool_summary(scores: np.ndarray) -> dict:
    """Table 2 row: count, mean, max, top-N mean, top-100 mean."""
    if scores.size == 0:
        return {"n": 0, "mean": 0.0, "max": 0.0,
                f"top{TOP_N}": 0.0, "top100": 0.0}
    n_top = min(TOP_N, scores.size)
    n_100 = min(100, scores.size)
    return {"n": int(scores.size),
            "mean": float(scores.mean()),
            "max": float(scores.max()),
            f"top{TOP_N}": float(scores[:n_top].mean()),
            "top100": float(scores[:n_100].mean())}


def component_breakdown(top_smiles: List[str], reward_fn) -> dict:
    """Table 3 row: top-K component means via RewardFunction.detailed."""
    if not top_smiles:
        return {k: 0.0 for k in COMPONENT_KEYS}
    rows = [reward_fn.detailed(s) for s in top_smiles]
    return {k: float(np.mean([r[k] for r in rows]))
            for k in COMPONENT_KEYS}


# Generation-set quality

def unique_canonical(smiles_list):
    """Canonical SMILES with parse failures and duplicates removed, order kept.
    Applied to every pool before scoring so methods compare on one basis."""
    seen, out = set(), []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def validity_rate(smiles_list):
    if not smiles_list:
        return 0.0
    valid = np.fromiter(
        (Chem.MolFromSmiles(s) is not None for s in smiles_list),
        dtype=bool, count=len(smiles_list))
    return float(valid.mean())


def tanimoto_to_reference(smiles, ref_fps, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = morgan_generator(radius, n_bits).GetFingerprint(mol)
    return max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps))


def novelty_fraction(smiles_list, ref_fps, threshold=0.6):
    """Fraction with max Tanimoto < threshold to reference set."""
    sims = [tanimoto_to_reference(s, ref_fps) for s in smiles_list]
    valid = np.array([s for s in sims if s is not None], dtype=np.float32)
    if valid.size == 0:
        return 0.0
    return float((valid < threshold).mean())


def scaffold_set(smiles_list):
    """Unique Bemis-Murcko scaffolds."""
    out = set()
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            out.add(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)))
        except Exception:
            continue
    return out


def scaffold_diversity(smiles_list):
    valid = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    if not valid:
        return 0.0
    return len(scaffold_set(valid)) / len(valid)


def internal_diversity(smiles_list, radius=2, n_bits=2048,
                       max_pairs=5000, seed=42):
    """1 - mean pairwise Tanimoto similarity among valid molecules."""
    gen = morgan_generator(radius, n_bits)
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fps.append(gen.GetFingerprint(mol))
    n = len(fps)
    if n < 2:
        return 0.0
    if n * (n - 1) // 2 > max_pairs:
        rng = np.random.default_rng(seed)
        keep = min(n, int(np.sqrt(2 * max_pairs)))
        idx = rng.choice(n, size=keep, replace=False)
        fps = [fps[i] for i in idx]
        n = len(fps)
    total_sim, pairs = 0.0, 0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total_sim += sum(sims)
        pairs += len(sims)
    return 1.0 - total_sim / max(pairs, 1)


def lipinski_pass_rate(smiles_list):
    passes, valid = 0, 0
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        valid += 1
        rules = (
            (Descriptors.MolWt(mol) <= 500)
            + (Descriptors.MolLogP(mol) <= 5)
            + (Lipinski.NumHDonors(mol) <= 5)
            + (Lipinski.NumHAcceptors(mol) <= 10)
        )
        passes += int(rules >= 3)
    return passes / max(valid, 1)


# CSV output

def metrics_csv(report, path):
    """Flatten a (possibly nested) metrics dict and write a one-row CSV."""
    flat = {}
    for k, v in report.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat[f"{k}_{k2}"] = v2
        else:
            flat[k] = v
    pd.DataFrame([flat]).to_csv(path, index=False)


# GNN test-set evaluation from gnn_best.pt

_ORG_NAMES = {"saureus": "S. aureus", "ecoli": "E. coli"}
_METRIC_ORDER = ("auroc", "auprc", "precision", "recall",
                 "rmse_logmic", "mae_logmic")


def main():
    device = pick_device()
    _cfg.ensure_dirs()
    torch.manual_seed(_cfg.train.seed)
    np.random.seed(_cfg.train.seed)
    print(f"Device: {device}")

    _, _, test_loader, _ = mic_splits(
        _cfg.train.batch_size, _cfg.train.seed, device)
    print(f"Test: {len(test_loader.dataset):,}")

    model = trained_gnn(device)
    print(f"Loaded {(_cfg.paths.models / 'gnn_best.pt').name}")

    report = gnn_report(model, test_loader, device,
                        threshold=_cfg.data.mic_threshold)
    release_cache(device)

    for key, metrics in report.items():
        print(f"\n  {_ORG_NAMES.get(key, key)}:")
        for name in _METRIC_ORDER:
            print(f"    {name}: {metrics[name]:.4f}")

    out_path = _cfg.paths.metrics / "gnn_test_metrics.csv"
    pd.DataFrame(report).T.to_csv(out_path)
    print(f"\nSaved {out_path.name}")


if __name__ == "__main__":
    main()