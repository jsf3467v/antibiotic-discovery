"""
Final-report evaluation for the RL pool.
"""

import sys
sys.path.insert(0,
    str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore", message=".*MorganGenerator.*")

from rdkit import RDLogger, rdBase
RDLogger.DisableLog('rdApp.*')
rdBase.DisableLog('rdApp.*')

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import ProjectConfig, pick_device, release_cache
from src.rewards import evaluation_reward
from src.evaluate import (TOP_N, COMPONENT_KEYS,
                          trained_gnn, scored_descending,
                          pool_summary, component_breakdown,
                          validity_rate, scaffold_diversity,
                          internal_diversity, lipinski_pass_rate,
                          novelty_fraction, metrics_csv)

cfg = ProjectConfig()

TOP_CANDIDATES = 20


def pool_csv() -> Path:
    return cfg.paths.results / "generated_molecules.csv"


def scored_csv() -> Path:
    return cfg.paths.results / "rl_pool_scored.csv"


def rl_pool_smiles() -> List[str]:
    """Valid SMILES from generated_molecules.csv."""
    path = pool_csv()
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        return []
    return df["smiles"].dropna().astype(str).tolist()


def cached_scores() -> Optional[Tuple[List[str], np.ndarray]]:
    """Return (sorted_smiles, sorted_scores) from cache, or None when
    absent or stale relative to the source pool."""
    cache = scored_csv()
    if not cache.exists():
        return None
    src = pool_csv()
    if src.exists() and src.stat().st_mtime > cache.stat().st_mtime:
        return None
    df = pd.read_csv(cache)
    return (df["smiles"].astype(str).tolist(),
            df["score"].to_numpy(dtype=np.float64))


def fresh_scores(smiles: List[str],
                 reward_fn) -> Tuple[List[str], np.ndarray]:
    """Score under reward_fn, sort descending, persist to cache."""
    sorted_smi, sorted_scores = scored_descending(smiles, reward_fn)
    pd.DataFrame({"smiles": sorted_smi, "score": sorted_scores}).to_csv(
        scored_csv(), index=False)
    return sorted_smi, sorted_scores


def quality_metrics(smiles: List[str], drugbank_fps) -> dict:
    """Validity, scaffold/internal diversity, Lipinski, novelty vs
    DrugBank. Internal diversity caps pairs at 5,000 internally so
    the 20k+-molecule pool stays bounded."""
    return {"validity": validity_rate(smiles),
            "scaffold_diversity": scaffold_diversity(smiles),
            "internal_diversity": internal_diversity(smiles),
            "lipinski_pass": lipinski_pass_rate(smiles),
            "novelty_vs_drugbank": novelty_fraction(smiles, drugbank_fps)}


def top_candidates(top_smiles, top_scores, reward_fn) -> pd.DataFrame:
    """Top-K candidate frame with per-molecule components for the
    manual-inspection step described in the future-work plan."""
    k = min(TOP_CANDIDATES, len(top_smiles))
    rows = []
    for i in range(k):
        c = reward_fn.detailed(top_smiles[i])
        rows.append({"rank": i + 1, "smiles": top_smiles[i],
                     "score": float(top_scores[i]),
                     **{kk: c[kk] for kk in COMPONENT_KEYS[1:]}})
    return pd.DataFrame(rows)


def print_report(summary: dict, components: dict, quality: dict):
    """Print Table 2 row, Table 3 row, and full-pool quality."""
    print(f"\nTable 2 row (RL):  N={summary['n']:,}  "
          f"mean={summary['mean']:.4f}  max={summary['max']:.4f}  "
          f"top{TOP_N}={summary[f'top{TOP_N}']:.4f}  "
          f"top100={summary['top100']:.4f}")
    print(f"\nTable 3 row (RL): top-{TOP_N} component means")
    for k in COMPONENT_KEYS:
        print(f"  {k:>10}: {components[k]:.4f}")
    print("\nFull-pool quality:")
    for k, v in quality.items():
        print(f"  {k:>22}: {v:.4f}")


def write_artifacts(summary, components, quality, candidates):
    """Persist metrics and top-candidates for the report."""
    out = cfg.paths.metrics
    metrics_csv({"summary": summary, "components": components,
                 "quality": quality}, out / "rl_pool_metrics.csv")
    candidates.to_csv(out / "rl_top_candidates.csv", index=False)


def main():
    cfg.ensure_dirs()
    device = pick_device()
    print(f"Device: {device}")
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    smiles = rl_pool_smiles()
    if not smiles:
        print("No RL pool found at "
              f"{pool_csv()}. Run train_rl.py first.")
        return
    print(f"RL pool: {len(smiles):,} mols")
    gnn = trained_gnn(device)
    reward_fn = evaluation_reward(gnn, device)
    hit = cached_scores()
    sorted_smi, sorted_scores = (
        hit if hit is not None
        else fresh_scores(smiles, reward_fn))
    release_cache(device)
    summary = pool_summary(sorted_scores)
    components = component_breakdown(sorted_smi[:TOP_N], reward_fn)
    quality = quality_metrics(smiles, reward_fn.drugbank.fps)
    candidates = top_candidates(sorted_smi, sorted_scores, reward_fn)
    print_report(summary, components, quality)
    write_artifacts(summary, components, quality, candidates)
    print(f"\nSaved to {cfg.paths.metrics}")


if __name__ == "__main__":
    main()