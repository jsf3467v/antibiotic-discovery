"""
Figures for the final paper. Reads cached results from paths.results
and paths.metrics; writes PDFs to paths.plots. Run after eval_rl.py,
eval_baselines.py, and stat_tests.py have populated their caches.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def style_defaults():
    """Common matplotlib rcParams across all figures."""
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def episode_log() -> pd.DataFrame:
    """Per-episode RL training log."""
    return pd.read_csv(cfg.paths.results / "rl_episode_log.csv")


def pool_scores(name: str) -> Tuple[List[str], np.ndarray]:
    """Canonical-evaluation scores for a method's pool."""
    if name == "rl":
        path = cfg.paths.results / "rl_pool_scored.csv"
    else:
        path = cfg.paths.results / f"baseline_{name}_scored.csv"
    df = pd.read_csv(path)
    return df["smiles"].tolist(), df["score"].to_numpy()


def dist_table() -> pd.DataFrame:
    """Distribution metrics from stat_tests.py."""
    return pd.read_csv(cfg.paths.metrics / "stat_tests_distribution.csv")


def candidates_table() -> pd.DataFrame:
    """Top-K RL candidate table from eval_rl.py."""
    return pd.read_csv(cfg.paths.metrics / "rl_top_candidates.csv")


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Edge-padded rolling mean over a 1D array."""
    return pd.Series(arr).rolling(
        window=window, min_periods=1).mean().to_numpy()


def phase_lengths(log_df: pd.DataFrame) -> Dict[int, int]:
    """Actual completed episode count per phase, taken from the log.

    Each phase can overshoot its configured cap by a few episodes because
    termination is checked at episode boundaries, so the true lengths come
    from the log rather than from the configured caps in config.py.
    """
    counts = log_df.groupby("phase")["episode"].max()
    return {int(p): int(n) for p, n in counts.items()}


def phase_offsets(log_df: pd.DataFrame) -> Dict[int, int]:
    """Cumulative episode offsets so phases plot on a continuous x-axis.

    Offsets are built from the true per-phase lengths, so phase boundaries
    fall exactly where each phase ended in the run.
    """
    lengths = phase_lengths(log_df)
    return {1: 0,
            2: lengths.get(1, 0),
            3: lengths.get(1, 0) + lengths.get(2, 0)}


def phase_boundaries(log_df: pd.DataFrame) -> Tuple[int, int]:
    """The two global-episode positions where phase 1 and phase 2 ended."""
    lengths = phase_lengths(log_df)
    first = lengths.get(1, 0)
    second = first + lengths.get(2, 0)
    return first, second


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
        ax.text(x + width / 2, 0, label, ha="center", va="center",
                fontsize=9)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + width, 0), (x + width + gap, 0),
                arrowstyle="-|>", mutation_scale=12,
                linewidth=1.0, color="black"))
    ax.set_xlim(x0 - 0.2, -x0 + 0.2)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_avg_reward(ax, log_df: pd.DataFrame, rl):
    """Top panel: rolling-window average reward by global episode."""
    df = episode_axis(log_df).sort_values("global_ep")
    smooth = rolling_mean(df["reward"].to_numpy(), window=128)
    ax.plot(df["global_ep"], smooth, color="tab:blue", linewidth=1.0)
    for boundary in phase_boundaries(log_df):
        ax.axvline(boundary, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg. reward (rolling 128)")
    ax.set_title("(a) Training reward per episode", loc="left")
    ax.grid(True, alpha=0.3)


def panel_size_center(ax, log_df: pd.DataFrame, rl):
    """Middle panel: size-gate center across phases."""
    df = episode_axis(log_df).sort_values("global_ep")
    ax.plot(df["global_ep"], df["size_center"],
            color="tab:orange", linewidth=1.0)
    for boundary in phase_boundaries(log_df):
        ax.axvline(boundary, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Size center (heavy atoms)")
    ax.set_title("(b) Size-gate center over training", loc="left")
    ax.grid(True, alpha=0.3)


def panel_top100(ax, log_df: pd.DataFrame, score_map: dict, rl):
    """Bottom panel: phase-3 running top-100 trajectory."""
    eps, tops = phase3_top100(log_df, score_map, rl.phase3_eval_every)
    if len(eps) == 0:
        ax.text(0.5, 0.5, "No phase-3 data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(c) Phase-3 top-100 trajectory", loc="left")
        return
    ax.plot(eps, tops, marker="o", markersize=4,
            color="tab:green", linewidth=1.2)
    best = -np.inf
    for x, y in zip(eps, tops):
        if y > best:
            ax.plot(x, y, marker="*", markersize=10,
                    color="darkgreen", zorder=5)
            best = y
    ax.set_xlabel("Phase-3 episode")
    ax.set_ylabel("Top-100 mean (canonical reward)")
    ax.set_title("(c) Phase-3 top-100 trajectory", loc="left")
    ax.grid(True, alpha=0.3)


def plot_training(out_path: Path):
    """Figure 2. Three-panel RL training curves."""
    log_df = episode_log().dropna(subset=["episode", "phase"]).copy()
    log_df["episode"] = log_df["episode"].astype(int)
    log_df["phase"] = log_df["phase"].astype(int)
    rl_smi, rl_scores = pool_scores("rl")
    score_map = dict(zip(rl_smi, rl_scores))
    rl = cfg.rl
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5))
    panel_avg_reward(axes[0], log_df, rl)
    panel_size_center(axes[1], log_df, rl)
    panel_top100(axes[2], log_df, score_map, rl)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_violins(out_path: Path):
    """Figure 3. Full-pool reward distributions per method."""
    data, labels = [], []
    for m in PANEL_ORDER:
        try:
            _, scores = pool_scores(m)
        except FileNotFoundError:
            continue
        data.append(scores)
        labels.append(DISPLAY[m])
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    parts = ax.violinplot(data, showmeans=True, showmedians=False,
                          widths=0.7)
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


def distribution_panel(ax, labels, values, ylabel, title, highlight):
    """Single bar panel for KL or FCD against the active reference."""
    colors = ["tab:blue" if lab == highlight else "lightgrey"
              for lab in labels]
    ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)


def plot_distribution(out_path: Path):
    """Figure 4. Property KL and FCD against active antibiotics."""
    df = dist_table().set_index("pool")
    order = [p for p in ("rl",) + tuple(
        x for x in PANEL_ORDER if x != "rl") if p in df.index]
    labels = [DISPLAY[p] for p in order]
    kl_vals = [float(df.loc[p, "property_kl"]) for p in order]
    fcd_vals = [float(df.loc[p, "fcd"]) for p in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    distribution_panel(axes[0], labels, kl_vals, "Property KL",
                       "(a) Property distribution KL", DISPLAY["rl"])
    distribution_panel(axes[1], labels, fcd_vals, "FCD",
                       "(b) Frechet ChemNet Distance", DISPLAY["rl"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def candidate_legend(rec) -> str:
    """Two-line legend under each rendered molecule."""
    return (f"#{int(rec['rank'])}  R={rec['score']:.2f}\n"
            f"pot={rec['potency']:.2f}  "
            f"qed={rec['qed']:.2f}  sa={rec['sa']:.2f}")


def plot_candidates(out_path: Path, k: int = 8):
    """Figure 5. Top-K RL molecule grid with score breakdowns."""
    df = candidates_table().head(k)
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


def quality_metrics() -> Dict[str, float]:
    """Internal diversity per method, keyed by method name.

    Baseline values come from baselines_quality.csv; the RL value comes
    from the pooled RL metrics written by eval_rl.py.
    """
    bq = pd.read_csv(
        cfg.paths.metrics / "baselines_quality.csv").set_index("method")
    out = {m: float(bq.loc[m, "internal_diversity"]) for m in bq.index}
    rl = pd.read_csv(cfg.paths.metrics / "rl_pool_metrics.csv").iloc[0]
    out["rl"] = float(rl["quality_internal_diversity"])
    return out


def plot_tradeoff(out_path: Path):
    """Figure 6. Top-10 reward against internal diversity per method.

    The productive region is the upper right, high reward held together
    with high diversity. Pools that reach high reward by collapsing onto
    a narrow region of chemical space fall to the left, and reward-free
    pools fall to the bottom.
    """
    div = quality_metrics()
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for m in PANEL_ORDER:
        try:
            _, scores = pool_scores(m)
        except FileNotFoundError:
            continue
        top10 = float(np.mean(np.sort(scores)[::-1][:10]))
        is_rl = (m == "rl")
        ax.scatter(div[m], top10, s=110 if is_rl else 70,
                   color="tab:blue" if is_rl else "lightgrey",
                   edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(DISPLAY[m], (div[m], top10),
                    textcoords="offset points", xytext=(6, 5), fontsize=9)
    ax.set_xlabel("Internal diversity")
    ax.set_ylabel("Top-10 mean reward")
    ax.set_title("Reward against structural diversity", loc="left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg.ensure_dirs()
    style_defaults()
    out = cfg.paths.plots
    figures = (
        ("fig1_pipeline.pdf", plot_pipeline),
        ("fig2_training.pdf", plot_training),
        ("fig3_violins.pdf", plot_violins),
        ("fig4_distribution.pdf", plot_distribution),
        ("fig5_candidates.pdf", plot_candidates),
        ("fig6_tradeoff.pdf", plot_tradeoff),
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