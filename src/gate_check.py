"""Pre-run check on the size gate and reward weights. Reads existing artifacts,
scores nothing with the GNN, runs in seconds."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from config import ProjectConfig

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
KEYS = ["potency", "novelty", "resistance", "qed", "sa"]


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def ramp(n: np.ndarray, center: float, steepness: float) -> np.ndarray:
    """One-sided gate as originally shipped."""
    return sigmoid(steepness * (n - center))


def band(n: np.ndarray, center: float, window: float,
         steepness: float) -> np.ndarray:
    """Two-sided gate, normalized to 1.0 at the center."""
    half = 0.5 * max(window, 1e-6)
    peak = sigmoid(steepness * half) ** 2
    return sigmoid(steepness * (n - center + half)) * \
        sigmoid(steepness * (center + half - n)) / peak


def totals(gate: np.ndarray, comp: np.ndarray, parts: np.ndarray,
           w: np.ndarray, ad: np.ndarray, scale) -> np.ndarray:
    """Weighted total under a given gate, mirroring RewardFunction;
    scale=None uses the original size-gate-scaled composition penalty."""
    weighted = parts.copy()
    weighted[:, 0] = weighted[:, 0] * ad
    eff = 1.0 - (gate if scale is None else scale) * (1.0 - comp)
    return gate * eff * (weighted @ w)


def recovered(p: pd.DataFrame, scale: float) -> np.ndarray:
    """Least-squares recovery of the weights behind a probe file, using the
    current composition scale. Raises when the residual is non-trivial,
    meaning the probe file is stale relative to the config."""
    eff = 1.0 - scale * (1.0 - p["composition"])
    denom = p["size_gate"] * eff
    ok = denom.abs() > 1e-12
    x = p.loc[ok, KEYS].to_numpy(dtype=np.float64)
    x[:, 0] = x[:, 0] * p.loc[ok, "ad_gate"].to_numpy()
    y = (p.loc[ok, "total"] / denom[ok]).to_numpy(dtype=np.float64)
    w = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = float(np.abs(x @ w - y).max())
    if resid > 1e-8:
        raise ValueError(
            f"weight recovery residual {resid:.2e}: the probe file "
            "results/metrics/reward_probes.csv is stale relative to the "
            "current reward config. Regenerate it with "
            "`python -m src.diagnose_rewards`.")
    return w


def bench(root: Path, run: Path, probes: Path | None = None) -> pd.DataFrame:
    """Probe molecules plus the run's top RL candidates, joined with the
    fields the checks need."""
    met = root / "results" / "metrics"
    p = pd.read_csv(probes or met / "reward_probes.csv")
    p = p[p["phase"] == p["phase"].max()]
    t = pd.read_csv(run / "metrics" / "rl_top_candidates.csv")
    props = pd.read_csv(run / "rl_episode_props.csv")
    t = t.merge(props.drop_duplicates("canonical")[["canonical", "composition"]],
                left_on="smiles", right_on="canonical")
    frame = pd.DataFrame({
        "name": list(p["probe"]) + [f"generated #{r}" for r in t["rank"]],
        "smiles": list(p["smiles"]) + list(t["smiles"]),
        "composition": list(p["composition"]) + list(t["composition"]),
        "ad_gate": list(p["ad_gate"]) + [1.0] * len(t),
    })
    for k in KEYS:
        frame[k] = list(p[k]) + list(t[k])
    frame["n"] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in frame["smiles"]]
    return frame


def report(d: pd.DataFrame, w: np.ndarray, cfg_w: np.ndarray,
           args, quiet: bool = False) -> None:
    print(f"gate: center={args.center} window={args.window} "
          f"steepness={args.steepness} comp_scale={args.scale}")
    print("weights recovered from probes:", np.round(w, 4))
    print("weights in config.py        :", np.round(cfg_w, 4))
    if not np.allclose(w, cfg_w, atol=1e-6):
        print("MISMATCH: the recorded run did not use the current config\n")
    if not quiet:
        cols = ["name", "n", "qed", "old", "rank_old", "new", "rank_new"]
        print(d.sort_values("old", ascending=False)[cols].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
    drugs = d[d["name"].str.startswith("drug")]
    gen = d[d["name"].str.startswith("generated")]
    novel = d[d["name"].str.startswith("novel")]
    hack = d[d["name"].str.startswith("hack")]
    for col in ("old", "new"):
        print(f"\n{col}: bestgen/bestdrug "
              f"{gen[col].max() / max(drugs[col].max(), 1e-12):.2f}x   "
              f"novel.min/hack.max "
              f"{novel[col].min() / max(hack[col].max(), 1e-12):.2f}x "
              f"(diagnose_rewards threshold 3.0)")


def main() -> None:
    cfg = ProjectConfig()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--seed", type=int, default=cfg.rl.seed,
                    help="RL run to read, selects runs/seed{seed}/")
    ap.add_argument("--center", type=float, default=cfg.rl.size_center_phase2_end)
    ap.add_argument("--window", type=float, default=cfg.rl.size_window)
    ap.add_argument("--steepness", type=float, default=cfg.rl.size_steepness)
    ap.add_argument("--scale", type=float, default=cfg.composition.scale)
    ap.add_argument("--probes", type=Path, default=None,
                    help="probe file; defaults to results/metrics/reward_probes.csv")
    ap.add_argument("--quiet", action="store_true",
                    help="print the weight check and the hackability ratios only; "
                         "the full ranking table is omitted from the console")
    args = ap.parse_args()

    probes_path = args.probes or (
        args.root / "results" / "metrics" / "reward_probes.csv")

    run = args.root / "runs" / f"seed{args.seed}"
    print(f"seed: {args.seed}")
    d = bench(args.root, run, probes_path)
    w = recovered(pd.read_csv(probes_path), cfg.composition.scale)
    cfg_w = np.array([getattr(cfg.rewards, k if k != "sa" else "sa_score")
                      for k in KEYS])
    parts = d[KEYS].to_numpy(dtype=np.float64)
    comp = d["composition"].to_numpy(dtype=np.float64)
    ad = d["ad_gate"].to_numpy(dtype=np.float64)
    n = d["n"].to_numpy(dtype=np.float64)
    d["old"] = totals(ramp(n, args.center, args.steepness),
                      comp, parts, w, ad, None)
    d["new"] = totals(band(n, args.center, args.window, args.steepness),
                      comp, parts, w, ad, args.scale)
    d["rank_old"] = d["old"].rank(ascending=False).astype(int)
    d["rank_new"] = d["new"].rank(ascending=False).astype(int)
    report(d, w, cfg_w, args, quiet=args.quiet)


if __name__ == "__main__":
    main()