"""
Project configuration. Single source of truth for paths, hyperparameters,
and the canonical organism key/display-name/source-stem mapping. All paths
are relative to the project root.
"""

import gc
import os
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent

# Canonical organism mapping: internal key, display name, source-file stem.
ORGANISMS: Tuple[Tuple[str, str, str], ...] = (
    ("saureus", "S. aureus", "staphylococcus_aureus"),
    ("ecoli",   "E. coli",   "escherichia_coli"),
)
ORGANISM_KEYS: Tuple[str, ...] = tuple(k for k, _, _ in ORGANISMS)
ORGANISM_DISPLAY: Dict[str, str] = {k: d for k, d, _ in ORGANISMS}
ORGANISM_SOURCE: Dict[str, str] = {k: s for k, _, s in ORGANISMS}


def cpu_workers() -> int:
    return max(1, min((os.cpu_count() or 4) // 2, 8))


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def release_cache(device: torch.device) -> None:
    """Free Python and accelerator workspace memory."""
    gc.collect()
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


@dataclass
class Paths:
    root: Path = ROOT

    @property
    def raw(self) -> Path:
        return self.root / "Datasets" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "Datasets" / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def metrics(self) -> Path:
        return self.root / "results" / "metrics"

    @property
    def plots(self) -> Path:
        return self.root / "results" / "plots"

    @property
    def eda_plots(self) -> Path:
        return self.root / "EDA" / "plots"


@dataclass
class AtomConfig:
    features: int = 12


@dataclass
class GNNConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.2
    edge_dropout: float = 0.15
    pool: str = "mean_max"
    readout_dim: int = 64
    edge_dim: int = 4


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 80
    patience: int = 20
    cosine_eta_min: float = 1e-6
    grad_clip: float = 1.0
    seed: int = 42
    huber_delta: float = 0.5
    num_workers: int = field(default_factory=cpu_workers)
    checkpoint_every: int = 5
    mps_cache_interval: int = 10


@dataclass
class DataConfig:
    mic_threshold: float = 10.0
    train_frac: float = 0.8
    val_frac: float = 0.1
    scaffold_split: bool = True
    organisms: List[str] = field(default_factory=lambda: list(
        ORGANISM_SOURCE[k] for k in ORGANISM_KEYS))


@dataclass
class CompositionConfig:
    """Reference atom-fraction distribution from active-subset.
    """
    reference: Dict[str, float] = field(default_factory=lambda: {
        "C": 0.70, "N": 0.12, "O": 0.14,
        "F": 0.017, "S": 0.013, "Cl": 0.008,
    })
    tau: float = 0.20


@dataclass
class SurrogateConfig:
    """Per-organism log10(MIC) regressor over Morgan fingerprints.
    """
    n_active: int = 4_000
    n_inactive: int = 4_000
    epochs: int = 30
    batch_size: int = 512
    hidden: int = 256
    fp_dim: int = 2048
    fp_radius: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    agreement_threshold: float = 0.5
    retrain_interval: int = 1_000_000
    retrain_min_unique: int = 500


@dataclass
class RLConfig:
    atom_actions: List[str] = field(default_factory=lambda: [
        "C", "N", "O", "S", "F", "Cl",
    ])
    gamma: float = 0.97
    gae_lambda: float = 0.92
    clip_eps: float = 0.2
    max_steps: int = 60
    n_envs: int = 32

    # Three-phase episode budget.
    phase1_episodes: int = 6_000
    phase2_episodes: int = 10_000
    phase3_episodes: int = 4_000

    entropy_phase1: float = 0.10
    entropy_phase2: float = 0.10

    # Size gate: small ramp inside the active size distribution
    size_center_phase1: float = 25.0
    size_center_phase2_end: float = 30.0
    size_steepness: float = 0.20
    gate_floor: float = 0.0
    potency_floor: float = 0.0

    # PPO
    policy_hidden: int = 128
    policy_layers: int = 3
    shaping_scale: float = 0.25
    update_every: int = 64
    ppo_epochs: int = 4
    minibatch: int = 128
    grad_clip: float = 0.5

    # Replay buffer with per-canonical dedup cap
    replay_capacity: int = 1_000
    replay_frac: float = 0.20
    replay_per_canonical: int = 3

    # KL anchor schedule. Phase 3 matches phase 2 under Option D
    # (kl=0.5) since the lower 0.2 / 0.05 settings repeatedly let
    # avg_r drift in earlier attempts.
    kl_phase1: float = 1.0
    kl_phase2: float = 0.5
    kl_phase3: float = 0.5

    # Phase 3 gate. Top-N is computed under the canonical
    # evaluation_reward path (rule 16), so the improvement signal
    # matches what stat_tests.py reports. .
    phase3_top_n: int = 100
    phase3_patience: int = 2
    phase3_eval_every: int = 512

    # Behavior cloning pretrain on active expert trajectories
    pretrain_epochs: int = 8
    pretrain_lr: float = 1e-3
    expert_max_mols: int = 3_000
    expert_min_atoms: int = 10
    expert_max_atoms: int = 40

    log_every: int = 512
    checkpoint_every: int = 2_000


@dataclass
class RewardWeights:
    potency: float = 0.30
    novelty: float = 0.15
    resistance: float = 0.10
    qed: float = 0.20
    sa_score: float = 0.25


@dataclass
class ProjectConfig:
    paths: Paths = field(default_factory=Paths)
    atom: AtomConfig = field(default_factory=AtomConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    rewards: RewardWeights = field(default_factory=RewardWeights)
    composition: CompositionConfig = field(default_factory=CompositionConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)

    def ensure_dirs(self) -> None:
        for d in (self.paths.processed, self.paths.models,
                  self.paths.metrics, self.paths.plots, self.paths.eda_plots):
            d.mkdir(parents=True, exist_ok=True)