"""Aggregate per-seed metric tables into cross-seed summaries. Reads
runs/seed*/metrics, writes results/summary. Derived data only, no rescoring."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Group keys per table; value columns are the remaining numeric fields.
KEYS = {
    "rl_pool_metrics": [],
    "baselines_table2": ["method"],
    "baselines_table3": ["method"],
    "baselines_quality": ["method"],
    "stat_tests_significance": ["baseline"],
    "stat_tests_distribution": ["pool"],
    "stat_tests_phase": ["phase"],
    "stat_tests_quintile": ["phase", "quintile"],
    "surrogate_agreement": [],
    "dynamics_phase": ["phase"],
    "dynamics_contrast": ["phase_a", "phase_b"],
}


def seed_dirs(root: Path) -> list:
    """Existing per-seed run dirs, in seed order."""
    return sorted((root / "runs").glob("seed*"))


def stacked(dirs: list, name: str) -> pd.DataFrame:
    """Stack one metric table across every seed that has it."""
    paths = [d / "metrics" / f"{name}.csv" for d in dirs]
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summary(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Per-key mean, std, min, max, and range across seeds, in long form."""
    if df.empty:
        return df
    values = [c for c in df.columns if c not in keys + ["seed"]
              and pd.api.types.is_numeric_dtype(df[c])]
    gk = keys or ["all"]
    work = df if keys else df.assign(all=0)
    agg = work.groupby(gk, dropna=False)[values].agg(["mean", "std", "min", "max"])
    agg.columns = agg.columns.set_names(["metric", "stat"])
    tidy = agg.stack("metric", future_stack=True).reset_index()
    tidy["range"] = tidy["max"] - tidy["min"]
    sizes = work.groupby(gk, dropna=False).size().rename("n_seeds").reset_index()
    tidy = tidy.merge(sizes, on=gk)
    return tidy.drop(columns="all") if not keys else tidy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()

    dirs = seed_dirs(args.root)
    if not dirs:
        raise SystemExit("no runs/seed* directories found")
    out = args.root / "results" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    print(f"seeds: {', '.join(d.name for d in dirs)}")

    for name, keys in KEYS.items():
        tidy = summary(stacked(dirs, name), keys)
        if tidy.empty:
            print(f"  {name}: missing")
            continue
        tidy.round(6).to_csv(out / f"{name}.csv", index=False)
        print(f"  {name}: {len(tidy)} rows, {int(tidy['n_seeds'].max())} seeds")
        if not keys:
            print(tidy.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()