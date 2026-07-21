"""
Figures for the final paper. Single-seed figures read runs/seed{seed}/;
cross-seed figures read results/summary/ written by collect.py. Run after
run.sh and collect have populated the per-seed metrics and the summary.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import warnings
warnings.filterwarnings("ignore", message=".*MorganGenerator.*")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from rdkit import Chem
from rdkit.Chem import Draw

from config import ProjectConfig

cfg = ProjectConfig()

DISPLAY = {
    "random": "Random",
    "genetic_algorithm": "Genetic Alg.",
    "hill_climbing": "Hill Climbing",
    "smiles_rnn": "SMILES-RNN",
    "rl": "RL Agent",
}
PANEL_ORDER = ("random", "hill_climbing", "smiles_rnn",
               "genetic_algorithm", "rl")
REP_SEED = 42


def style_defaults():
    """Common matplotlib rcParams across all figures."""
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })


# --- data access -------------------------------------------------------------

def run_dir(seed: int) -> Path:
    return cfg.paths.seed_dir(seed)


def run_metrics(seed: int) -> Path:
    return cfg.paths.seed_metrics(seed)


def summary_dir() -> Path:
    return cfg.paths.results / "summary"


def episode_log(seed: int) -> pd.DataFrame:
    """Per-episode RL training log for one seed."""
    return pd.read_csv(run_dir(seed) / "rl_episode_log.csv")


def pool_scores(name: str, seed: int) -> Tuple[List[str], np.ndarray]:
    """Canonical-evaluation scores for a method's pool in one seed."""
    fname = "rl_pool_scored.csv" if name == "rl" else f"baseline_{name}_scored.csv"
    df = pd.read_csv(run_dir(seed) / fname)
    return df["smiles"].tolist(), df["score"].to_numpy()


def candidates_table(seed: int) -> pd.DataFrame:
    """Top-K RL candidate table for one seed."""
    return pd.read_csv(run_metrics(seed) / "rl_top_candidates.csv")


def summary_table(name: str) -> pd.DataFrame:
    """A cross-seed summary table from results/summary/, long form."""
    return pd.read_csv(summary_dir() / f"{name}.csv")


def keyed_stat(name: str, key_col: str, key_val: str,
               metric: str) -> Tuple[float, float, float]:
    """(mean, min, max) for one metric of one keyed row in a summary table."""
    df = summary_table(name)
    row = df[(df[key_col] == key_val) & (df["metric"] == metric)]
    r = row.iloc[0]
    return float(r["mean"]), float(r["min"]), float(r["max"])


def rl_stat(metric: str) -> Tuple[float, float, float]:
    """(mean, min, max) for one metric from the keyless RL summary."""
    df = summary_table("rl_pool_metrics")
    r = df[df["metric"] == metric].iloc[0]
    return float(r["mean"]), float(r["min"]), float(r["max"])


def seed_dirs() -> list:
    """Existing per-seed run directories, in seed order."""
    return sorted((cfg.paths.root / "runs").glob("seed*"))


# --- training-log geometry ---------------------------------------------------

def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Edge-padded rolling mean over a 1D array."""
    return pd.Series(arr).rolling(window=window, min_periods=1).mean().to_numpy()


def phase_lengths(log_df: pd.DataFrame) -> Dict[int, int]:
    """Completed episode count per phase, read from the log rather than the
    configured caps, since a phase can overshoot its cap by a few episodes."""
    counts = log_df.groupby("phase")["episode"].max()
    return {int(p): int(n) for p, n in counts.items()}


def phase_offsets(log_df: pd.DataFrame) -> Dict[int, int]:
    """Cumulative episode offsets so phases lay out on a continuous axis."""
    lengths = phase_lengths(log_df)
    return {1: 0, 2: lengths.get(1, 0), 3: lengths.get(1, 0) + lengths.get(2, 0)}


def phase_boundaries(log_df: pd.DataFrame) -> Tuple[int, int]:
    """Global-episode positions where phase 1 and phase 2 ended."""
    lengths = phase_lengths(log_df)
    first = lengths.get(1, 0)
    return first, first + lengths.get(2, 0)


def episode_axis(log_df: pd.DataFrame) -> pd.DataFrame:
    """Append global_ep so phases lay out continuously on one axis."""
    offsets = phase_offsets(log_df)
    out = log_df.copy()
    out["global_ep"] = out["episode"] + out["phase"].map(offsets)
    return out


def phase3_top100(log_df: pd.DataFrame, score_map: dict,
                  every: int) -> Tuple[np.ndarray, np.ndarray]:
    """Running top-100 mean over phase 3 evaluation ticks."""
    log_df = log_df.sort_values(["phase", "episode"]).reset_index(drop=True)
    seen, pool = set(), []
    eps, tops, last_tick = [], [], -1
    for _, row in log_df.iterrows():
        smi = row.get("smiles")
        if isinstance(smi, str) and smi not in seen:
            seen.add(smi)
            if smi in score_map:
                pool.append(score_map[smi])
        if row["phase"] == 3 and pool:
            tick = int(row["episode"]) // every
            if tick != last_tick:
                top = sorted(pool, reverse=True)[:100]
                eps.append(int(row["episode"]))
                tops.append(float(np.mean(top)))
                last_tick = tick
    return np.array(eps), np.array(tops)


# --- figures -----------------------------------------------------------------

def plot_pipeline(out_path: Path):
    """Figure 1. Pipeline schematic across the five components."""
    fig, ax = plt.subplots(figsize=(6.5, 1.8))
    boxes = ["GNN", "Surrogate", "Reward\nFunction",
             "PPO Agent", "Generated\nMolecules"]
    width, height, gap = 1.4, 0.9, 0.45
    n = len(boxes)
    x0 = -(n * width + (n - 1) * gap) / 2
    for i, label in enumerate(boxes):
        x = x0 + i * (width + gap)
        ax.add_patch(FancyBboxPatch(
            (x, -height / 2), width, height,
            boxstyle="round,pad=0.04", linewidth=1.2,
            edgecolor="black", facecolor="white"))
        ax.text(x + width / 2, 0, label, ha="center", va="center", fontsize=9)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + width, 0), (x + width + gap, 0),
                arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="black"))
    ax.set_xlim(x0 - 0.2, -x0 + 0.2)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_avg_reward(ax, log_df: pd.DataFrame):
    """Rolling-window average reward by global episode."""
    df = episode_axis(log_df).sort_values("global_ep")
    smooth = rolling_mean(df["reward"].to_numpy(), window=128)
    ax.plot(df["global_ep"], smooth, color="tab:blue", linewidth=1.0)
    for boundary in phase_boundaries(log_df):
        ax.axvline(boundary, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg. reward (rolling 128)")
    ax.set_title("(a) Training reward per episode", loc="left")
    ax.grid(True, alpha=0.3)


def panel_size_center(ax, log_df: pd.DataFrame):
    """Size-gate center across phases."""
    df = episode_axis(log_df).sort_values("global_ep")
    ax.plot(df["global_ep"], df["size_center"], color="tab:orange", linewidth=1.0)
    for boundary in phase_boundaries(log_df):
        ax.axvline(boundary, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Size center (heavy atoms)")
    ax.set_title("(b) Size-gate center over training", loc="left")
    ax.grid(True, alpha=0.3)


def panel_top100(ax, log_df: pd.DataFrame, score_map: dict, every: int):
    """Phase-3 running top-100 trajectory."""
    eps, tops = phase3_top100(log_df, score_map, every)
    if len(eps) == 0:
        ax.text(0.5, 0.5, "No phase-3 data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("(c) Phase-3 top-100 trajectory", loc="left")
        return
    ax.plot(eps, tops, marker="o", markersize=4, color="tab:green", linewidth=1.2)
    best = -np.inf
    for x, y in zip(eps, tops):
        if y > best:
            ax.plot(x, y, marker="*", markersize=10, color="darkgreen", zorder=5)
            best = y
    ax.set_xlabel("Phase-3 episode")
    ax.set_ylabel("Top-100 mean (canonical reward)")
    ax.set_title("(c) Phase-3 top-100 trajectory", loc="left")
    ax.grid(True, alpha=0.3)


def plot_training(out_path: Path, seed: int):
    """Figure 2. Three-panel RL training curves for the representative seed."""
    log_df = episode_log(seed).dropna(subset=["episode", "phase"]).copy()
    log_df["episode"] = log_df["episode"].astype(int)
    log_df["phase"] = log_df["phase"].astype(int)
    rl_smi, rl_scores = pool_scores("rl", seed)
    score_map = dict(zip(rl_smi, rl_scores))
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5))
    panel_avg_reward(axes[0], log_df)
    panel_size_center(axes[1], log_df)
    panel_top100(axes[2], log_df, score_map, cfg.rl.phase3_eval_every)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_violins(out_path: Path, seed: int):
    """Figure 3. Full-pool reward distributions per method, representative seed."""
    data, labels = [], []
    for m in PANEL_ORDER:
        try:
            _, scores = pool_scores(m, seed)
        except FileNotFoundError:
            continue
        data.append(scores)
        labels.append(DISPLAY[m])
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    parts = ax.violinplot(data, showmeans=True, showmedians=False, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("tab:blue")
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Canonical reward")
    ax.set_title("Reward distribution by method", loc="left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def distribution_panel(ax, labels, values, yerr, ylabel, title, highlight):
    """One bar panel for a distribution metric, with cross-seed range bars."""
    colors = ["tab:blue" if lab == highlight else "lightgrey" for lab in labels]
    ax.bar(labels, values, yerr=yerr, capsize=3, color=colors,
           edgecolor="black", linewidth=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)


def plot_distribution(out_path: Path):
    """Figure 4. Property KL and FCD vs active antibiotics, cross-seed range bars."""
    df = summary_table("stat_tests_distribution")
    order = [p for p in ("rl",) + PANEL_ORDER if p in set(df["pool"])]
    seen = []
    order = [p for p in order if not (p in seen or seen.append(p))]
    labels = [DISPLAY[p] for p in order]

    def series(metric):
        means, lo, hi = [], [], []
        for p in order:
            r = df[(df["pool"] == p) & (df["metric"] == metric)].iloc[0]
            means.append(r["mean"])
            lo.append(r["mean"] - r["min"])
            hi.append(r["max"] - r["mean"])
        return np.array(means), np.array([lo, hi])

    kl_m, kl_e = series("property_kl")
    fcd_m, fcd_e = series("fcd")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    distribution_panel(axes[0], labels, kl_m, kl_e, "Property KL",
                       "(a) Property distribution KL", DISPLAY["rl"])
    distribution_panel(axes[1], labels, fcd_m, fcd_e, "FCD",
                       "(b) Frechet ChemNet Distance", DISPLAY["rl"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def candidate_legend(rec) -> str:
    """Two-line legend under each rendered molecule."""
    return (f"#{int(rec['rank'])}  R={rec['score']:.2f}\n"
            f"pot={rec['potency']:.2f}  qed={rec['qed']:.2f}  sa={rec['sa']:.2f}")


def plot_candidates(out_path: Path, seed: int, k: int = 8):
    """Figure 5. Top-K RL molecule grid with score breakdowns, representative seed."""
    df = candidates_table(seed).head(k)
    cols = 4
    rows = max(1, (len(df) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.2 * rows))
    axes = np.atleast_2d(axes)
    flat = axes.flatten()
    for i, (_, rec) in enumerate(df.iterrows()):
        mol = Chem.MolFromSmiles(rec["smiles"])
        if mol is None:
            flat[i].axis("off")
            continue
        flat[i].imshow(Draw.MolToImage(mol, size=(300, 300)))
        flat[i].set_title(candidate_legend(rec), fontsize=8)
        flat[i].axis("off")
    for j in range(len(df), len(flat)):
        flat[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(out_path: Path):
    """Figure 6. Top-10 reward against internal diversity per method, with
    cross-seed range bars on both axes."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for m in PANEL_ORDER:
        try:
            if m == "rl":
                dm, dlo, dhi = rl_stat("quality_internal_diversity")
                tm, tlo, thi = rl_stat("summary_top10")
            else:
                dm, dlo, dhi = keyed_stat("baselines_quality", "method", m,
                                          "internal_diversity")
                tm, tlo, thi = keyed_stat("baselines_table2", "method", m, "top10")
        except (FileNotFoundError, IndexError):
            continue
        is_rl = (m == "rl")
        ax.errorbar(dm, tm, xerr=[[dm - dlo], [dhi - dm]],
                    yerr=[[tm - tlo], [thi - tm]], fmt="o",
                    ms=11 if is_rl else 7,
                    color="tab:blue" if is_rl else "lightgrey",
                    ecolor="gray", elinewidth=0.8, capsize=2,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=3)
        ax.annotate(DISPLAY[m], (dm, tm), textcoords="offset points",
                    xytext=(6, 5), fontsize=9)
    ax.set_xlabel("Internal diversity")
    ax.set_ylabel("Top-10 mean reward")
    ax.set_title("Reward against structural diversity", loc="left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_seed_stability(out_path: Path):
    """Figure 7. RL pool metrics across every seed, one panel per metric."""
    dirs = seed_dirs()
    if not dirs:
        raise FileNotFoundError("no runs/seed* directories")
    seeds, pool_rows, dist_rows = [], [], []
    for d in dirs:
        seeds.append(d.name.replace("seed", ""))
        pool_rows.append(pd.read_csv(d / "metrics" / "rl_pool_metrics.csv").iloc[0])
        dd = pd.read_csv(d / "metrics" / "stat_tests_distribution.csv")
        dist_rows.append(dd[dd["pool"] == "rl"].iloc[0])
    pool = pd.DataFrame(pool_rows)
    dist = pd.DataFrame(dist_rows)
    panels = [
        ("Pool reward mean", pool["summary_mean"].to_numpy(), "%.3f"),
        ("Top-100 reward mean", pool["summary_top100"].to_numpy(), "%.3f"),
        ("Scaffold diversity", pool["quality_scaffold_diversity"].to_numpy(), "%.3f"),
        ("Internal diversity", pool["quality_internal_diversity"].to_numpy(), "%.3f"),
        ("Lipinski pass rate", pool["quality_lipinski_pass"].to_numpy(), "%.3f"),
        ("Novelty vs DrugBank", pool["quality_novelty_vs_drugbank"].to_numpy(), "%.3f"),
        ("Frechet ChemNet Distance", dist["fcd"].to_numpy(), "%.1f"),
        ("Property KL divergence", dist["property_kl"].to_numpy(), "%.2f"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.2))
    for ax, (name, vals, fmt) in zip(axes.flatten(), panels):
        vals = np.asarray(vals, dtype=float)
        bars = ax.bar(seeds, vals, color="tab:blue", alpha=0.85,
                      edgecolor="black", linewidth=0.5, width=0.6)
        ax.set_ylim(0, vals.max() * 1.22)
        ax.set_title(name, fontsize=9.5)
        ax.set_xlabel("seed", fontsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt % v,
                    ha="center", va="bottom", fontsize=7.5)
        rng = vals.max() - vals.min()
        ax.text(0.5, 0.90, f"range {rng:.3f}", transform=ax.transAxes,
                ha="center", fontsize=7.5, color="dimgray")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("RL agent pool metrics across seeds", fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Paper figures")
    ap.add_argument("--seed", type=int, default=REP_SEED,
                    help="representative seed for the single-seed figures")
    args = ap.parse_args()
    seed = args.seed
    cfg.ensure_dirs()
    style_defaults()
    out = cfg.paths.plots
    print(f"representative seed: {seed}")
    figures = (
        ("fig1_pipeline.pdf", lambda p: plot_pipeline(p)),
        ("fig2_training.pdf", lambda p: plot_training(p, seed)),
        ("fig3_violins.pdf", lambda p: plot_violins(p, seed)),
        ("fig4_distribution.pdf", lambda p: plot_distribution(p)),
        ("fig5_candidates.pdf", lambda p: plot_candidates(p, seed)),
        ("fig6_tradeoff.pdf", lambda p: plot_tradeoff(p)),
        ("fig7_seed_stability.pdf", lambda p: plot_seed_stability(p)),
    )
    for name, fn in figures:
        try:
            fn(out / name)
            print(f"  wrote {name}")
        except FileNotFoundError as e:
            print(f"  skipped {name}: {e}")
        except Exception as e:
            print(f"  failed {name}: {e}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()