"""
Three-phase PPO training for molecular generation. Phase 1 anchors
the policy to a BC prior at small sizes. Phase 2 ramps the size
center and phase 3 freezes size and uses a top-N evaluation-reward gate
for early stopping. BC prior and initial surrogate are disk-cached.
"""

import sys
sys.path.insert(0,
    str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore", message=".*MorganGenerator.*")
warnings.filterwarnings("ignore", message=".*target size.*input size.*")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import copy
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from rdkit import Chem

from config import ProjectConfig, ORGANISM_KEYS, pick_device, release_cache
from src.gnn import MultiTaskGNN
from src.feature_engineering import smiles_to_graph
from src.rewards import (PotencySurrogate, fp_array,
                         training_reward, evaluation_reward,
                         aggregated_organism, active_smiles)
from src.rl import (MolPolicy, VecMolEnv, PPOTrainer, ExpertStep,
                    expert_trajectory,
                    node_kind_tensors, node_action_stats)

cfg = ProjectConfig()


# SMILES sources for surrogate training and BC pretraining

def stratified_smiles(n_active: int, n_inactive: int,
                      seed: int) -> List[str]:
    """Active and inactive sample, balanced per organism, deduplicated."""
    log_thr = float(np.log10(cfg.data.mic_threshold))
    rng = np.random.default_rng(seed)
    n_orgs = max(len(cfg.data.organisms), 1)
    per_org_a = max(n_active // n_orgs, 1)
    per_org_i = max(n_inactive // n_orgs, 1)
    seen, out = set(), []
    for stem in cfg.data.organisms:
        agg = aggregated_organism(stem)
        if agg is None:
            continue
        actives = agg.loc[agg["log_mic"] < log_thr, "canonical_smiles"].tolist()
        inactives = agg.loc[agg["log_mic"] >= log_thr, "canonical_smiles"].tolist()
        rng.shuffle(actives)
        rng.shuffle(inactives)
        for s in actives[:per_org_a] + inactives[:per_org_i]:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


# Trained GNN


def trained_gnn(device) -> MultiTaskGNN:
    model = MultiTaskGNN(cfg.atom, cfg.gnn).to(device)
    ckpt = cfg.paths.models / "gnn_best.pt"
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def transfer_encoder(policy: MolPolicy, gnn: MultiTaskGNN) -> int:
    """Copy GNN conv and projection weights into the policy encoder.
    Norm layers differ (BatchNorm vs LayerNorm), so their affines are
    left at the policy's own initialization."""
    src = gnn.encoder.state_dict()
    dst = policy.encoder.state_dict()
    moved = 0
    for key in dst:
        if key.startswith("norms."):
            continue
        if key in src and src[key].shape == dst[key].shape:
            dst[key] = src[key].clone()
            moved += 1
    policy.encoder.load_state_dict(dst)
    return moved


# Surrogate training - per-organism log10(MIC))

def chunk_fps_graphs(chunk):
    """Aligned (fp_arrays, graphs) for one SMILES chunk; drops parse failures."""
    arrs, graphs = [], []
    for smi in chunk:
        arr = fp_array(smi)
        g = smiles_to_graph(smi)
        if arr is not None and g is not None:
            arrs.append(arr)
            graphs.append(g)
    return arrs, graphs


def gnn_log_mic_targets(gnn, device, smiles_list) -> Tuple[list, np.ndarray]:
    """Per-SMILES - fingerprint, per-organism log_mic. Column order: ORGANISM_KEYS."""
    organisms = list(ORGANISM_KEYS)
    fps, target_chunks = [], []
    is_mps = device.type == "mps"
    for start in range(0, len(smiles_list), 512):
        arrs, graphs = chunk_fps_graphs(smiles_list[start:start + 512])
        if not graphs:
            continue
        batch = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            log_mic = gnn(batch.x, batch.edge_index, batch.batch,
                          edge_attr=batch.edge_attr)
        per_org = torch.stack(
            [log_mic[k] for k in organisms], dim=1).cpu().numpy()
        fps.extend(arrs)
        target_chunks.append(per_org)
        if is_mps:
            torch.mps.empty_cache()
    if not target_chunks:
        return [], np.zeros((0, len(organisms)), dtype=np.float32)
    return fps, np.concatenate(target_chunks, axis=0)


def fit_surrogate(gnn, device, smiles_list,
                  epochs: Optional[int] = None) -> PotencySurrogate:
    """Train surrogate on per-organism log10(MIC) MSE."""
    sc = cfg.surrogate
    n_epochs = epochs if epochs is not None else sc.epochs
    fps, targets = gnn_log_mic_targets(gnn, device, smiles_list)
    if not fps:
        raise RuntimeError("no valid SMILES for surrogate training")
    fps_t = torch.tensor(np.array(fps), dtype=torch.float32, device=device)
    tgt_t = torch.tensor(targets, dtype=torch.float32, device=device)
    model = PotencySurrogate(
        fp_dim=sc.fp_dim, hidden=sc.hidden,
        n_organisms=tgt_t.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=sc.lr,
                           weight_decay=sc.weight_decay)
    n = len(fps_t)
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, sc.batch_size):
            bi = perm[i:i + sc.batch_size]
            loss = F.mse_loss(model(fps_t[bi]), tgt_t[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


# Behavior cloning

def safe_expert_trajectory(smi):
    try:
        return expert_trajectory(smi)
    except Exception:
        return None


def has_ring(smi) -> bool:
    mol = Chem.MolFromSmiles(smi)
    return mol is not None and mol.GetRingInfo().NumRings() > 0


def atom_count_in_range(smi, lo: int, hi: int) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    n = mol.GetNumHeavyAtoms()
    return lo <= n <= hi


def expert_dataset(smiles_list,
                   max_mols: Optional[int] = None,
                   min_atoms: Optional[int] = None,
                   max_atoms: Optional[int] = None) -> List[ExpertStep]:
    """Expert build trajectories filtered by heavy-atom range, prioritizing rings."""
    rl = cfg.rl
    max_mols = max_mols or rl.expert_max_mols
    min_atoms = min_atoms or rl.expert_min_atoms
    max_atoms = max_atoms or rl.expert_max_atoms
    in_range = [s for s in smiles_list
                if atom_count_in_range(s, min_atoms, max_atoms)]
    rings = [s for s in in_range if has_ring(s)]
    chains = [s for s in in_range if not has_ring(s)]
    rng = np.random.default_rng(cfg.train.seed)
    rng.shuffle(rings)
    rng.shuffle(chains)
    candidates = (rings + chains)[:max_mols * 2]
    steps, converted = [], 0
    for smi in candidates:
        if converted >= max_mols:
            break
        traj = safe_expert_trajectory(smi)
        if traj:
            steps.extend(traj)
            converted += 1
    return steps


def expert_loss(policy, batch_steps, batch, all_node, all_graph, device):
    """Vectorized type/anchor/target cross-entropy for an expert minibatch."""
    type_mask_np = np.stack([s.type_mask for s in batch_steps])
    type_masks = torch.from_numpy(type_mask_np).to(device)
    type_logits = policy.type_head(all_graph).masked_fill(
        ~type_masks, float("-inf"))
    type_targets = torch.tensor(
        [s.action_type for s in batch_steps], device=device)
    t_loss = F.cross_entropy(type_logits, type_targets)
    ptr_cpu = batch.ptr.cpu().numpy()
    a_mask, a_chosen, a_valid = node_kind_tensors(
        batch_steps, ptr_cpu, "anchor", device)
    tg_mask, tg_chosen, tg_valid = node_kind_tensors(
        batch_steps, ptr_cpu, "target", device)
    a_lp, _ = node_action_stats(
        policy.anchor_proj, all_node, all_graph, batch.batch,
        a_mask, a_chosen, a_valid, batch.num_graphs)
    tg_lp, _ = node_action_stats(
        policy.target_proj, all_node, all_graph, batch.batch,
        tg_mask, tg_chosen, tg_valid, batch.num_graphs)
    a_loss = (-a_lp[a_valid].mean() if a_valid.any()
              else torch.zeros((), device=device))
    tg_loss = (-tg_lp[tg_valid].mean() if tg_valid.any()
               else torch.zeros((), device=device))
    return t_loss + a_loss + tg_loss


def pretrain_batch(policy, batch_steps, device):
    """Single behavioral-cloning forward pass."""
    pyg = Batch.from_data_list(
        [s.graph for s in batch_steps]).to(device)
    ea = pyg.edge_attr if hasattr(pyg, "edge_attr") else None
    all_node, all_graph = policy.encoder(
        pyg.x, pyg.edge_index, pyg.batch, edge_attr=ea)
    return expert_loss(policy, batch_steps, pyg, all_node, all_graph, device)


def pretrain_policy(policy, steps, device,
                    epochs: int, batch_size: int = 256):
    """Behavior clone the policy on expert trajectories."""
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.rl.pretrain_lr)
    n = len(steps)
    rng = np.random.default_rng(cfg.train.seed)
    for epoch in range(epochs):
        policy.train()
        order = rng.permutation(n)
        total, batches = 0.0, 0
        for start in range(0, n, batch_size):
            bi = order[start:start + batch_size]
            batch = [steps[i] for i in bi]
            loss = pretrain_batch(policy, batch, device)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            total += loss.item()
            batches += 1
        print(f"  Pretrain epoch {epoch + 1}/{epochs}  "
              f"loss={total / max(batches, 1):.4f}")


# Phase 3 top-N gate - early-stop

class TopNGate:
    """Top-N early-stop gate. Each SMILES is scored once and cached by canonical key."""

    def __init__(self, n: int, patience: int, reward_fn,
                 device=None, prior_state: Optional[dict] = None):
        self.n = n
        self.patience = patience
        self.reward_fn = reward_fn
        self.device = device
        if prior_state is None:
            prior_state = {}
        self.cache: dict = dict(prior_state.get("cache", {}))
        self.best = float(prior_state.get("best", float("-inf")))
        self.misses = int(prior_state.get("misses", 0))
        self.scored_through = int(prior_state.get("scored_through", 0))

    @property
    def state(self) -> dict:
        return {"cache": self.cache, "best": self.best,
                "misses": self.misses, "scored_through": self.scored_through}

    def score_pending(self, smiles_list):
        """Score canonical SMILES not yet in the cache."""
        is_mps = self.device is not None and self.device.type == "mps"
        new_slice = smiles_list[self.scored_through:]
        self.scored_through = len(smiles_list)
        for i, s in enumerate(new_slice):
            if s is not None and s not in self.cache:
                self.cache[s] = float(self.reward_fn(s))
            if is_mps and (i + 1) % 256 == 0:
                torch.mps.empty_cache()

    def top_n_mean(self) -> float:
        if not self.cache:
            return 0.0
        scores = sorted(self.cache.values(), reverse=True)
        n = min(self.n, len(scores))
        return float(np.mean(scores[:n]))

    def step(self, smiles_list) -> Tuple[float, bool, bool]:
        """Score new SMILES, update best, return - top_mean, improved, stop."""
        self.score_pending(smiles_list)
        top = self.top_n_mean()
        improved = top > self.best
        if improved:
            self.best = top
            self.misses = 0
        else:
            self.misses += 1
        return top, improved, self.misses > self.patience


# Schedules and accumulation

def size_center_phase2(phase2_ep: int, rl_cfg) -> float:
    """Linear ramp from phase-1 center to phase-2 end center."""
    frac = min(phase2_ep / max(rl_cfg.phase2_episodes, 1), 1.0)
    return rl_cfg.size_center_phase1 + frac * (
        rl_cfg.size_center_phase2_end - rl_cfg.size_center_phase1)


def deduplicated_valid(raw: List[str]) -> List[str]:
    """Canonicalize, drop None, drop duplicates."""
    seen, out = set(), []
    for s in raw:
        if s is None:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        canon = Chem.MolToSmiles(m)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
    return out


# Episode collection / log row

def episode_log_row(episode: int, smi: str, raw: float,
                    phase: int, size_center: float,
                    entropy: float, kl_coef: float,
                    max_steps: int, replay_size: int) -> dict:
    return {"episode": episode, "phase": phase, "smiles": smi,
            "reward": raw, "size_center": size_center,
            "entropy": entropy, "kl_coef": kl_coef,
            "max_steps": max_steps, "replay": replay_size}


def rollout_batch(trainer, vec_env, all_generated, recent_rewards,
                  phase, size_center, max_steps, episode):
    """One batched rollout. Mutates the generation list and rolling
    reward window; returns log rows for the episode log.
    """
    rows = []
    for smi, raw in trainer.batched_episodes(vec_env):
        all_generated.append(smi)
        recent_rewards.append(raw)
        rows.append(episode_log_row(
            episode, smi, raw, phase, size_center,
            trainer.entropy_coeff, trainer.kl_coef,
            max_steps, len(trainer.replay)))
    return rows


# Atomic checkpointing

def atomic_save(obj, path: Path):
    """Sync MPS workspace, write to a temp file, atomic-rename."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def policy_checkpoint(path: Path, policy, optimizer, episode: int,
                      phase: int, gate: Optional[TopNGate] = None):
    """Save policy, optimizer, resume position, and gate state (phase 3)."""
    payload = {
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
        "phase": phase,
    }
    if gate is not None:
        payload["gate"] = gate.state
    atomic_save(payload, path)


def latest_checkpoint() -> Optional[Tuple[int, Path]]:
    """Return (phase, path) of the most advanced checkpoint, or None."""
    for phase in (3, 2, 1):
        path = cfg.run / f"policy_phase{phase}.pt"
        if path.exists():
            return phase, path
    return None


def resume_state(policy, optimizer, device):
    """Restore from the most advanced checkpoint and the episode log
    """
    found = latest_checkpoint()
    if found is None:
        return None
    _, path = found
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    optimizer.load_state_dict(ckpt["optimizer"])
    log_path = cfg.run / "rl_episode_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        log_rows = log_df.to_dict("records")
        all_gen = [s if isinstance(s, str) else None
                   for s in log_df["smiles"]]
    else:
        log_rows, all_gen = [], []
    return (ckpt["phase"], ckpt["episode"] + cfg.rl.n_envs,
            all_gen, log_rows, ckpt.get("gate"))


# Phase loops


def fires_now(ep: int, interval: int, n_envs: int) -> bool:
    """True when the outer iteration aligns with an interval-in-episodes."""
    period = max(1, interval // n_envs)
    return (ep // n_envs) % period == 0


def update_step(trainer, episode: int) -> dict:
    """Trigger a PPO update at the configured cadence and release the device cache."""
    rl = trainer.cfg
    if not fires_now(episode, rl.update_every, rl.n_envs):
        return {}
    metrics = trainer.ppo_update() or {}
    release_cache(trainer.device)
    return metrics


def maybe_log(episode, recent_rewards, size_center, trainer,
              last_metrics, log_every):
    if not fires_now(episode, log_every, trainer.cfg.n_envs):
        return
    avg = np.mean(recent_rewards) if recent_rewards else 0.0
    nan_frac = last_metrics.get("nan_skip", 0.0) if last_metrics else 0.0
    nan_str = f"  nan={nan_frac:.1%}" if nan_frac > 0 else ""
    print(f"  ep {episode:>6}  avg_r={avg:.3f}  "
          f"size_c={size_center:>4.1f}  ent={trainer.entropy_coeff:.4f}  "
          f"kl={trainer.kl_coef:.3f}  replay={len(trainer.replay)}{nan_str}")


def maybe_checkpoint(trainer, all_generated, log_rows, ep, resume_at,
                     phase: int, ckpt_path: Path,
                     gate: Optional[TopNGate] = None):
    """Persist policy state and the molecules CSV at checkpoint cadence.
    Phase 3 forwards the gate so its cache and miss count survive resume.
    """
    if ep <= resume_at or not fires_now(
            ep, trainer.cfg.checkpoint_every, trainer.cfg.n_envs):
        return
    policy_checkpoint(ckpt_path, trainer.policy, trainer.optimizer,
                      ep, phase, gate=gate)
    save_outputs(deduplicated_valid(all_generated), log_rows)
    release_cache(trainer.device)


def gate_tick(ep, gate, all_generated, trainer, ckpt_best):
    """One eval-cadence tick. Scores new molecules, snapshots the
    policy on improvement, and returns True when patience is exceeded.
    """
    rl = trainer.cfg
    if not fires_now(ep, rl.phase3_eval_every, rl.n_envs):
        return False
    _, improved, stop = gate.step(all_generated)
    if improved:
        atomic_save(trainer.policy.state_dict(), ckpt_best)
    if stop:
        print(f"  early stop ep {ep}: top{gate.n} not improving")
    release_cache(trainer.device)
    return stop


def phase3_gate_setup(trainer, reward_fn, gate_state):
    """Construct the phase-3 gate from a resumed state."""
    rl = trainer.cfg
    eval_reward = evaluation_reward(reward_fn.gnn, trainer.device)
    return TopNGate(rl.phase3_top_n, rl.phase3_patience, eval_reward,
                    device=trainer.device, prior_state=gate_state)


def phase3_log(ep, recent, trainer, last_metrics, gate):
    """Phase 3 log line. Combines rollout avg with gate state so the
    operator sees both signals on one line at log_every cadence.
    """
    rl = trainer.cfg
    if not fires_now(ep, rl.log_every, rl.n_envs):
        return
    avg = float(np.mean(recent)) if recent else 0.0
    nan_frac = last_metrics.get("nan_skip", 0.0) if last_metrics else 0.0
    nan_str = f"  nan={nan_frac:.1%}" if nan_frac > 0 else ""
    top_disp = (f"{gate.best:.4f}"
                if gate.best > float("-inf") else "-")
    print(f"  ep {ep:>6}  avg_r={avg:.3f}  top{gate.n}={top_disp}  "
          f"miss={gate.misses}/{gate.patience}  "
          f"kl={trainer.kl_coef:.3f}{nan_str}")


def phase1_loop(trainer, vec_env, all_generated,
                resume_at: int = 0, log_rows: Optional[list] = None):
    """Structure exploration. KL=phase1 holds policy near BC prior."""
    rl = trainer.cfg
    if log_rows is None:
        log_rows = []
    recent: deque = deque(maxlen=rl.log_every)
    last_metrics: dict = {}
    print("\nPhase 1: structure exploration - KL-anchored to BC prior")
    vec_env.set_max_steps(rl.max_steps)
    trainer.entropy_coeff = rl.entropy_phase1
    trainer.kl_coef = rl.kl_phase1
    ckpt_path = cfg.run / "policy_phase1.pt"
    size_c = rl.size_center_phase1
    for ep in range(resume_at, rl.phase1_episodes, rl.n_envs):
        log_rows.extend(rollout_batch(
            trainer, vec_env, all_generated, recent,
            phase=1, size_center=size_c,
            max_steps=rl.max_steps, episode=ep))
        m = update_step(trainer, ep)
        if m:
            last_metrics = m
        maybe_log(ep, recent, size_c, trainer, last_metrics, rl.log_every)
        maybe_checkpoint(trainer, all_generated, log_rows, ep, resume_at,
                         1, ckpt_path)
    return log_rows


def phase2_loop(trainer, vec_env, reward_fn, all_generated,
                resume_at: int = 0, log_rows: Optional[list] = None):
    """Size-center ramp. KL relaxed to phase2."""
    rl = trainer.cfg
    if log_rows is None:
        log_rows = []
    recent: deque = deque(maxlen=rl.log_every)
    last_metrics: dict = {}
    print("\nPhase 2: size ramp - KL relaxed")
    trainer.entropy_coeff = rl.entropy_phase2
    trainer.kl_coef = rl.kl_phase2
    ckpt_path = cfg.run / "policy_phase2.pt"
    for ep in range(resume_at, rl.phase2_episodes, rl.n_envs):
        size_c = size_center_phase2(ep, rl)
        reward_fn.size_center = size_c
        log_rows.extend(rollout_batch(
            trainer, vec_env, all_generated, recent,
            phase=2, size_center=size_c,
            max_steps=rl.max_steps, episode=ep))
        m = update_step(trainer, ep)
        if m:
            last_metrics = m
        maybe_log(ep, recent, size_c, trainer, last_metrics, rl.log_every)
        maybe_checkpoint(trainer, all_generated, log_rows, ep, resume_at,
                         2, ckpt_path)
    return log_rows


def phase3_transition(trainer, reward_fn):
    """Phase 3 settings. Freeze size at the phase-2 endpoint, reuse phase-2 KL and entropy."""
    rl = trainer.cfg
    reward_fn.size_center = rl.size_center_phase2_end
    reward_fn.potency_floor = rl.potency_floor
    trainer.entropy_coeff = rl.entropy_phase2
    trainer.kl_coef = rl.kl_phase3


def phase3_loop(trainer, vec_env, reward_fn, all_generated,
                resume_at: int = 0, log_rows: Optional[list] = None,
                gate_state: Optional[dict] = None):
    """Top-N gated extension of phase 2. Stops when top-N stalls
    """
    rl = trainer.cfg
    if log_rows is None:
        log_rows = []
    recent: deque = deque(maxlen=rl.log_every)
    last_metrics: dict = {}
    print("\nPhase 3: top-N gated extension")
    phase3_transition(trainer, reward_fn)
    gate = phase3_gate_setup(trainer, reward_fn, gate_state)
    ckpt_path = cfg.run / "policy_phase3.pt"
    ckpt_best = cfg.run / "policy_phase3_best.pt"
    size_c = rl.size_center_phase2_end
    for ep in range(resume_at, rl.phase3_episodes, rl.n_envs):
        log_rows.extend(rollout_batch(trainer, vec_env, all_generated,
                                      recent, 3, size_c, rl.max_steps, ep))
        m = update_step(trainer, ep)
        if m:
            last_metrics = m
        if gate_tick(ep, gate, all_generated, trainer, ckpt_best):
            break
        phase3_log(ep, recent, trainer, last_metrics, gate)
        maybe_checkpoint(trainer, all_generated, log_rows, ep, resume_at,
                         3, ckpt_path, gate=gate)
    return log_rows


# Output


def save_outputs(molecules: List[str], log_rows: List[dict]):
    """Write deduplicated valid molecules and the per-episode log."""
    cfg.ensure_dirs()
    cfg.ensure_seed_dirs(cfg.rl.seed)
    pd.DataFrame({"smiles": molecules}).to_csv(
        cfg.run / "generated_molecules.csv", index=False)
    pd.DataFrame(log_rows).to_csv(
        cfg.run / "rl_episode_log.csv", index=False)


def final_snapshot(trainer):
    """Persist the gate-selected best phase-3 weights as policy_final.pt.
    A fall back to the in-memory policy when no phase-3 best exists.
    """
    src = cfg.run / "policy_phase3_best.pt"
    dst = cfg.run / "policy_final.pt"
    if src.exists():
        weights = torch.load(src, map_location="cpu", weights_only=True)
        atomic_save(weights, dst)
        return
    atomic_save(trainer.policy.state_dict(), dst)


# Setup with on-disk caching f

def seeded_policy(device, gnn) -> MolPolicy:
    """Construct a policy and seed its encoder from the trained GNN."""
    policy = MolPolicy(hidden=cfg.gnn.hidden_dim,
                       layers=cfg.gnn.num_layers,
                       heads=cfg.gnn.heads,
                       edge_dim=cfg.gnn.edge_dim).to(device)
    n = transfer_encoder(policy, gnn)
    print(f"Transferred {n} parameter tensors from GNN encoder")
    return policy


def pretrain_or_cached_prior(policy, anchor_smiles, device):
    """Pretrain via BC, or restore cached prior weights from policy_prior.pt."""
    path = cfg.paths.models / "policy_prior.pt"
    if path.exists():
        policy.load_state_dict(
            torch.load(path, map_location=device, weights_only=True))
        print(f"\nLoaded cached BC prior from {path.name}")
        return
    print("\nBehavior cloning pretrain")
    expert = expert_dataset(anchor_smiles)
    print(f"  expert steps: {len(expert)}")
    pretrain_policy(policy, expert, device, epochs=cfg.rl.pretrain_epochs)
    atomic_save(policy.state_dict(), path)
    print(f"  cached BC prior to {path.name}")
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    release_cache(device)


def freeze_prior(policy: MolPolicy) -> MolPolicy:
    """Frozen deep copy of the policy used as the KL-prior anchor."""
    prior = copy.deepcopy(policy).eval()
    for p in prior.parameters():
        p.requires_grad_(False)
    return prior


def fit_initial_surrogate(gnn, device) -> PotencySurrogate:
    """Surrogate fit on a stratified sample, cached at surrogate.pt."""
    sc = cfg.surrogate
    path = cfg.paths.models / "surrogate.pt"
    if path.exists():
        model = PotencySurrogate(
            fp_dim=sc.fp_dim, hidden=sc.hidden,
            n_organisms=len(ORGANISM_KEYS)).to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True))
        return model.eval()
    train_smiles = stratified_smiles(
        sc.n_active, sc.n_inactive, cfg.train.seed)
    print(f"  surrogate fit on {len(train_smiles)} SMILES")
    surrogate = fit_surrogate(gnn, device, train_smiles)
    atomic_save(surrogate.state_dict(), path)
    release_cache(device)
    return surrogate


def prepared_pipeline(device):
    """GNN, surrogate, reward, BC-pretrained policy, frozen prior, PPO trainer, and vec env."""
    gnn = trained_gnn(device)
    anchor_smiles = active_smiles()
    print(f"  active SMILES: {len(anchor_smiles)}")
    surrogate = fit_initial_surrogate(gnn, device)
    reward_fn = training_reward(gnn, device, surrogate)
    policy = seeded_policy(device, gnn)
    pretrain_or_cached_prior(policy, anchor_smiles, device)
    prior = freeze_prior(policy)
    trainer = PPOTrainer(policy, reward_fn, cfg.rl, device, prior=prior)
    vec_env = VecMolEnv(cfg.rl.n_envs, cfg.rl.max_steps, max_atoms=60)
    return trainer, vec_env, reward_fn



def initial_state(resumed):
    """Unpack resumed tuple, or return fresh state when None."""
    if resumed is None:
        return 1, 0, [], [], None
    phase, start_ep, all_generated, log_rows, gate_state = resumed
    print(f"Resumed phase {phase} ep {start_ep}, "
          f"{len(all_generated)} mols so far")
    return phase, start_ep, all_generated, log_rows, gate_state


def run_phases(trainer, vec_env, reward_fn, resumed
               ) -> Tuple[List[str], List[dict]]:
    """Dispatch phases 1, 2, 3. Replay flushed at boundaries; gate_state
    is reset on cross-phase transition and preserved on phase-3 resume.
    """
    phase, start_ep, all_generated, log_rows, gate_state = initial_state(resumed)
    if phase == 1:
        log_rows = phase1_loop(trainer, vec_env, all_generated,
                               resume_at=start_ep, log_rows=log_rows)
        trainer.replay.flush()
        release_cache(trainer.device)
        phase, start_ep, gate_state = 2, 0, None
    if phase == 2:
        log_rows = phase2_loop(trainer, vec_env, reward_fn, all_generated,
                               resume_at=start_ep, log_rows=log_rows)
        trainer.replay.flush()
        release_cache(trainer.device)
        phase, start_ep, gate_state = 3, 0, None
    if phase == 3:
        log_rows = phase3_loop(trainer, vec_env, reward_fn, all_generated,
                               resume_at=start_ep, log_rows=log_rows,
                               gate_state=gate_state)
        release_cache(trainer.device)
    return all_generated, log_rows


def run():
    cfg.ensure_dirs()
    cfg.ensure_seed_dirs(cfg.rl.seed)
    device = pick_device()
    print(f"Device: {device}  seed: {cfg.rl.seed}")
    torch.manual_seed(cfg.rl.seed)
    np.random.seed(cfg.rl.seed)
    trainer, vec_env, reward_fn = prepared_pipeline(device)
    resumed = resume_state(trainer.policy, trainer.optimizer, device)
    all_generated, log_rows = run_phases(
        trainer, vec_env, reward_fn, resumed)
    molecules = deduplicated_valid(all_generated)
    print(f"\nTotal unique valid: {len(molecules)}")
    save_outputs(molecules, log_rows)
    final_snapshot(trainer)
    release_cache(device)
    print(f"Saved to {cfg.run}")


if __name__ == "__main__":
    run()