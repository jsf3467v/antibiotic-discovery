"""
Evaluate the four baseline pools written by baselines.py. Reads
baseline_{name}.csv from paths.results, scores under the canonical
evaluation reward
"""

import sys
sys.path.insert(0,
    str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore", message=".*MorganGenerator.*")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import ProjectConfig, pick_device, release_cache
from src.rewards import evaluation_reward
from src.evaluate import (TOP_N, COMPONENT_KEYS,
                          trained_gnn, scored_descending, unique_canonical,
                          pool_summary, component_breakdown,
                          scaffold_diversity,
                          internal_diversity, lipinski_pass_rate)

cfg = ProjectConfig()

BASELINES = ("random", "genetic_algorithm", "hill_climbing", "smiles_rnn")


def pool_csv(name: str) -> Path:
    return cfg.run / f"baseline_{name}.csv"


def scored_csv(name: str) -> Path:
    return cfg.run / f"baseline_{name}_scored.csv"


def csv_pool(name: str) -> List[str]:
    """Read smiles column from baseline_{name}.csv. Empty if missing."""
    path = pool_csv(name)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        return []
    return df["smiles"].dropna().astype(str).tolist()


def cached_scores(name: str) -> Optional[Tuple[List[str], np.ndarray]]:
    """Return (sorted_smiles, sorted_scores) from cache, or None when
    absent or stale relative to the source pool."""
    cache = scored_csv(name)
    if not cache.exists():
        return None
    src = pool_csv(name)
    if src.exists() and src.stat().st_mtime > cache.stat().st_mtime:
        return None
    df = pd.read_csv(cache)
    return (df["smiles"].astype(str).tolist(),
            df["score"].to_numpy(dtype=np.float64))


def fresh_scores(name: str, smiles: List[str],
                 reward_fn) -> Tuple[List[str], np.ndarray]:
    """Score under reward_fn, sort descending, persist to cache."""
    sorted_smi, sorted_scores = scored_descending(smiles, reward_fn)
    pd.DataFrame({"smiles": sorted_smi, "score": sorted_scores}).to_csv(
        scored_csv(name), index=False)
    return sorted_smi, sorted_scores


def quality_metrics(smiles: List[str]) -> dict:
    """Scaffold/internal diversity and Lipinski pass rate on the unique pool.
    Novelty against DrugBank is in Table 3 components, not duplicated."""
    return {"scaffold_diversity": scaffold_diversity(smiles),
            "internal_diversity": internal_diversity(smiles),
            "lipinski_pass": lipinski_pass_rate(smiles)}


def evaluate_one(name: str, smiles: List[str],
                 reward_fn) -> Tuple[dict, dict, dict]:
    """Per-baseline (table2_row, table3_row, quality_row). Uses cache
    when present and current; populates it on first run."""
    hit = cached_scores(name)
    sorted_smi, sorted_scores = (
        hit if hit is not None
        else fresh_scores(name, smiles, reward_fn))
    t2 = {"method": name, **pool_summary(sorted_scores)}
    t3 = {"method": name,
          **component_breakdown(sorted_smi[:TOP_N], reward_fn)}
    qm = {"method": name, **quality_metrics(smiles)}
    return t2, t3, qm


def print_table_2(rows: List[dict]):
    print("\nTable 2 (reward statistics):")
    print(f"  {'method':<20} {'n':>6} {'mean':>7} {'max':>7} "
          f"{'top10':>7} {'top100':>7}")
    for r in rows:
        print(f"  {r['method']:<20} {r['n']:>6,} {r['mean']:>7.4f} "
              f"{r['max']:>7.4f} {r[f'top{TOP_N}']:>7.4f} "
              f"{r['top100']:>7.4f}")


def print_table_3(rows: List[dict]):
    print("\nTable 3 - top-10 component means :")
    head = f"  {'method':<20}" + "".join(
        f" {k:>10}" for k in COMPONENT_KEYS)
    print(head)
    for r in rows:
        line = f"  {r['method']:<20}" + "".join(
            f" {r[k]:>10.4f}" for k in COMPONENT_KEYS)
        print(line)


def print_quality(rows: List[dict]):
    print("\nQuality (unique pool):")
    print(f"  {'method':<20} {'scaff':>6} {'intdiv':>7} {'lipinski':>9}")
    for r in rows:
        print(f"  {r['method']:<20} {r['scaffold_diversity']:>6.4f} "
              f"{r['internal_diversity']:>7.4f} "
              f"{r['lipinski_pass']:>9.4f}")


def write_csvs(t2_rows, t3_rows, qm_rows):
    """Three per-table CSVs into the per-seed metrics dir, seed-labeled."""
    out = cfg.run_metrics
    seed = cfg.rl.seed
    for rows, name in ((t2_rows, "baselines_table2.csv"),
                       (t3_rows, "baselines_table3.csv"),
                       (qm_rows, "baselines_quality.csv")):
        frame = pd.DataFrame(rows)
        frame.insert(0, "seed", seed)
        frame.to_csv(out / name, index=False)


def main():
    cfg.ensure_dirs()
    cfg.ensure_seed_dirs(cfg.rl.seed)
    device = pick_device()
    print(f"Device: {device}  seed: {cfg.rl.seed}")
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    gnn = trained_gnn(device)
    reward_fn = evaluation_reward(gnn, device)
    t2_rows, t3_rows, qm_rows = [], [], []
    for name in BASELINES:
        smiles = unique_canonical(csv_pool(name))
        if not smiles:
            print(f"  {name}: missing or empty, skipping")
            continue
        print(f"  {name}: {len(smiles):,} unique mols")
        t2, t3, qm = evaluate_one(name, smiles, reward_fn)
        t2_rows.append(t2)
        t3_rows.append(t3)
        qm_rows.append(qm)
        release_cache(device)
    if not t2_rows:
        print("\nNo baseline pools found. Run baselines.py first.")
        return
    print_table_2(t2_rows)
    print_table_3(t3_rows)
    print_quality(qm_rows)
    write_csvs(t2_rows, t3_rows, qm_rows)
    print(f"\nSaved to {cfg.run_metrics}")


if __name__ == "__main__":
    main()