"""Post-hoc RL training dynamics from existing logs. No retraining, no rescoring."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import pandas as pd

from config import ProjectConfig

COLS = ["score", "qed", "lipinski", "n_atoms", "composition"]
ROOT = Path(__file__).resolve().parents[1]
cfg = ProjectConfig()


def frames(run: Path):
    log = pd.read_csv(run / "rl_episode_log.csv")
    scored = pd.read_csv(run / "rl_pool_scored.csv")
    props = pd.read_csv(run / "rl_episode_props.csv")
    return log, scored, props


def sequence(log: pd.DataFrame) -> pd.DataFrame:
    """Episode counter restarts each phase; impose one global training order."""
    out = log.sort_values(["phase", "episode"], kind="stable").reset_index(drop=True)
    out["order"] = np.arange(len(out), dtype=np.int32)
    return out


def table(log: pd.DataFrame, scored: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    keys = props.rename(columns={"canonical": "smiles"}).drop_duplicates("smiles")
    df = sequence(log).merge(scored, on="smiles", how="left").merge(keys, on="smiles", how="left")
    out = df.dropna(subset=["score"]).reset_index(drop=True)
    for name, col in (("scored", "score"), ("props", "n_atoms")):
        hit = out[col].notna().mean() if len(out) else 0.0
        if hit < 0.5:
            raise ValueError(
                f"{name} matched {hit:.1%} of episode rows; the file is stale "
                "or its keys differ from the episode log")
    return out


def by_phase(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("phase", sort=True)
    out = grp[COLS].mean()
    out.insert(0, "n", grp.size())
    out.insert(1, "train_reward", grp["reward"].mean())
    return out.reset_index()


def by_decile(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    edges = pd.qcut(df["order"], bins, labels=False, duplicates="drop")
    out = df.groupby(edges, sort=True)[COLS].mean()
    out.index.name = "decile"
    return out.reset_index()


def provenance(df: pd.DataFrame, k: int) -> pd.DataFrame:
    top = df.nlargest(k, "score")
    out = pd.DataFrame(
        {"top_n": top["phase"].value_counts(), "pool_n": df["phase"].value_counts()}
    ).fillna(0.0)
    out["top_share"] = out["top_n"] / len(top)
    out["pool_share"] = out["pool_n"] / len(df)
    out["lift"] = np.where(out["pool_share"] > 0, out["top_share"] / out["pool_share"], np.nan)
    return out.rename_axis("phase").reset_index().sort_values("phase", ignore_index=True)


def quarters(df: pd.DataFrame, phase: int, bins: int, k: int) -> pd.DataFrame:
    sub = df[df["phase"] == phase]
    edges = pd.qcut(sub["order"], bins, labels=False, duplicates="drop")
    out = sub.groupby(edges, sort=True).agg(
        n=("score", "size"),
        mean=("score", "mean"),
        top=("score", lambda s: s.nlargest(k).mean()),
        qed=("qed", "mean"),
        n_atoms=("n_atoms", "mean"),
    )
    out.index.name = "quarter"
    return out.reset_index()


def contrast(df: pd.DataFrame, a: int, b: int, iters: int, seed: int) -> pd.DataFrame:
    """Bootstrap CI on the canonical-reward mean difference between two phases."""
    xa = df.loc[df["phase"] == a, "score"].to_numpy()
    xb = df.loc[df["phase"] == b, "score"].to_numpy()
    if xa.size == 0 or xb.size == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    ma = xa[rng.integers(0, xa.size, (iters, xa.size), dtype=np.int32)].mean(axis=1)
    mb = xb[rng.integers(0, xb.size, (iters, xb.size), dtype=np.int32)].mean(axis=1)
    lo, hi = np.percentile(mb - ma, [2.5, 97.5])
    return pd.DataFrame(
        [{"phase_a": a, "phase_b": b, "delta": xb.mean() - xa.mean(), "lo": lo, "hi": hi}]
    )


def stage(path: Path, build, seed: int) -> pd.DataFrame:
    """Resume point: reuse the cached table if present, otherwise build,
    seed-label, and cache it."""
    if path.exists():
        return pd.read_csv(path)
    out = build()
    if not out.empty:
        out.insert(0, "seed", seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def panels(decile: pd.DataFrame, tail: pd.DataFrame, prov: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.6), constrained_layout=True)
    ax[0].plot(decile["decile"], decile["score"], marker="o")
    ax[0].set(xlabel="Decile of training order", ylabel="Canonical reward",
              title="(a) Canonical reward over training")
    ax[1].plot(tail["quarter"], tail["top"], marker="o")
    ax[1].set(xlabel="Phase 3 quarter", ylabel="Top-K mean", title="(b) Phase 3 top-K")
    ax[2].bar(prov["phase"].astype(str), prov["lift"])
    ax[2].axhline(1.0, linestyle="--", linewidth=1.0, color="black")
    ax[2].set(xlabel="Phase", ylabel="Top-K share / pool share",
              title="(c) Provenance of best molecules")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def report(phase: pd.DataFrame, decile: pd.DataFrame, prov: pd.DataFrame,
           tail: pd.DataFrame, boot: pd.DataFrame) -> None:
    fmt = dict(index=False, float_format=lambda v: f"{v:.4f}")
    print("Phase means (canonical score vs phase-local train_reward)")
    print(phase.to_string(**fmt), end="\n\n")
    print("Canonical reward by decile of training order")
    print(decile[["decile", "score", "qed", "lipinski", "n_atoms"]].to_string(**fmt), end="\n\n")
    print("Provenance of top molecules")
    print(prov.to_string(**fmt), end="\n\n")
    print("Phase 3 quarters")
    print(tail.to_string(**fmt), end="\n\n")
    if not boot.empty:
        r = boot.iloc[0]
        print(f"Phase {int(r.phase_a)} to {int(r.phase_b)} canonical delta "
              f"{r.delta:+.4f}  95% CI [{r.lo:+.4f}, {r.hi:+.4f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--quarters", type=int, default=4)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0,
                    help="bootstrap RNG seed for the contrast CI")
    ap.add_argument("--run-seed", type=int, default=cfg.rl.seed,
                    help="RL run to read, selects runs/seed{run_seed}/")
    ap.add_argument("--fresh", action="store_true", help="clear cached tables first")
    args = ap.parse_args()

    run = args.root / "runs" / f"seed{args.run_seed}"
    out = run / "metrics"
    print(f"seed: {args.run_seed}")
    names = ["dynamics_phase", "dynamics_decile", "dynamics_provenance",
             "dynamics_phase3", "dynamics_contrast"]
    if args.fresh:
        for n in names:
            (out / f"{n}.csv").unlink(missing_ok=True)

    df = table(*frames(run))
    last = int(df["phase"].max())
    first = int(df["phase"].min())

    phase = stage(out / "dynamics_phase.csv", lambda: by_phase(df), args.run_seed)
    decile = stage(out / "dynamics_decile.csv", lambda: by_decile(df, args.bins), args.run_seed)
    prov = stage(out / "dynamics_provenance.csv", lambda: provenance(df, args.top), args.run_seed)
    tail = stage(out / "dynamics_phase3.csv", lambda: quarters(df, last, args.quarters, args.top), args.run_seed)
    boot = stage(out / "dynamics_contrast.csv",
                 lambda: contrast(df, first, last, args.iters, args.seed), args.run_seed)

    panels(decile, tail, prov, run / "plots" / "dynamics.png")
    report(phase, decile, prov, tail, boot)
    print(f"\nmatched {len(df)} episode rows to canonical scores")


if __name__ == "__main__":
    main()