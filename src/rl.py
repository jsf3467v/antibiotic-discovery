"""
RL for molecular generation. MDP environment, autoregressive GNN
policy, PPO with batched rollouts, KL-prior anchor, top-k replay,
and BC expert-trajectory pretraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import RWMol
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional

from config import RLConfig
from src.feature_engineering import atom_vec, bond_vec


ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl"]
NUM_ATOM_TYPES = len(ATOM_SYMBOLS)
ACT_BOND = NUM_ATOM_TYPES
ACT_UPGRADE = NUM_ATOM_TYPES + 1
ACT_STOP = NUM_ATOM_TYPES + 2
NUM_ACT_TYPES = NUM_ATOM_TYPES + 3

MAX_VALENCE = {6: 4, 7: 3, 8: 2, 16: 2, 9: 1, 17: 1}
ATOM_FEAT_DIM = 12
EDGE_FEAT_DIM = 4
HETEROATOMS = {7, 8, 16}
SHAPING_CLIP = 0.5



# Potential shaping

def mol_potential(mol) -> float:
    """Potential function on a partial molecule for dense shaping."""
    if mol is None or mol.GetNumAtoms() == 0:
        return 0.0
    n = mol.GetNumAtoms()
    atoms = {a.GetAtomicNum() for a in mol.GetAtoms()}
    size = min(n, 30) / 30.0
    hetero = len(atoms & HETEROATOMS) / 3.0
    bonds = mol.GetNumBonds()
    cycles = max(min(bonds - n + 1, 3), 0) / 3.0 if bonds > 0 else 0.0
    return 0.20 * size + 0.15 * hetero + 0.25 * cycles


# Environment

class MolEnv:
    """Molecular MDP. State is an RWMol edited in place."""

    def __init__(self, max_steps: int = 60, max_atoms: int = 60):
        self.max_steps = max_steps
        self.max_atoms = max_atoms
        self.mol: Optional[RWMol] = None
        self.steps = 0
        self.cached_fv: Optional[np.ndarray] = None
        self.cached_bam: Optional[np.ndarray] = None

    def reset(self) -> str:
        self.mol = RWMol()
        self.mol.AddAtom(Chem.Atom(6))
        self.steps = 0
        self.cached_fv = None
        self.cached_bam = None
        return self.smiles()

    def smiles(self) -> Optional[str]:
        try:
            return Chem.MolToSmiles(self.mol)
        except Exception:
            return None

    def free_valence(self) -> np.ndarray:
        if self.cached_fv is not None:
            return self.cached_fv
        n = self.mol.GetNumAtoms()
        free = np.empty(n, dtype=np.int32)
        for i in range(n):
            atom = self.mol.GetAtomWithIdx(i)
            cap = MAX_VALENCE.get(atom.GetAtomicNum(), 4)
            used = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
            free[i] = max(cap - used, 0)
        self.cached_fv = free
        return free

    def type_mask(self) -> np.ndarray:
        n = self.mol.GetNumAtoms()
        fv = self.free_valence()
        has_free = fv.any()
        mask = np.zeros(NUM_ACT_TYPES, dtype=bool)
        if has_free and n < self.max_atoms:
            mask[:NUM_ATOM_TYPES] = True
        if n >= 3 and self.bond_anchor_mask().any():
            mask[ACT_BOND] = True
        if self.has_upgradable_bond(fv):
            mask[ACT_UPGRADE] = True
        if n >= 2:
            mask[ACT_STOP] = True
        return mask

    def bond_anchor_mask(self) -> np.ndarray:
        """Anchors with at least one non-bonded peer that has free valence."""
        if self.cached_bam is not None:
            return self.cached_bam
        n = self.mol.GetNumAtoms()
        fv = self.free_valence()
        mask = np.zeros(n, dtype=bool)
        free_idx = np.flatnonzero(fv > 0)
        if len(free_idx) < 2:
            self.cached_bam = mask
            return mask
        free_set = set(int(j) for j in free_idx)
        for i in free_idx:
            ii = int(i)
            bonded = {b.GetOtherAtomIdx(ii)
                      for b in self.mol.GetAtomWithIdx(ii).GetBonds()}
            if free_set - bonded - {ii}:
                mask[ii] = True
        self.cached_bam = mask
        return mask

    def has_upgradable_bond(self, fv: np.ndarray) -> bool:
        for bond in self.mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if fv[bond.GetBeginAtomIdx()] > 0 and fv[bond.GetEndAtomIdx()] > 0:
                return True
        return False

    def anchor_mask(self) -> np.ndarray:
        return self.free_valence() > 0

    def upgrade_anchor_mask(self) -> np.ndarray:
        n = self.mol.GetNumAtoms()
        fv = self.free_valence()
        mask = np.zeros(n, dtype=bool)
        for bond in self.mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if fv[i] > 0 and fv[j] > 0:
                mask[i] = mask[j] = True
        return mask

    def target_mask(self, anchor: int) -> np.ndarray:
        n = self.mol.GetNumAtoms()
        fv = self.free_valence()
        bonded = np.zeros(n, dtype=bool)
        for b in self.mol.GetAtomWithIdx(anchor).GetBonds():
            bonded[b.GetOtherAtomIdx(anchor)] = True
        mask = (fv > 0) & ~bonded
        mask[anchor] = False
        return mask

    def upgrade_target_mask(self, anchor: int) -> np.ndarray:
        fv = self.free_valence()
        n = self.mol.GetNumAtoms()
        mask = np.zeros(n, dtype=bool)
        if fv[anchor] <= 0:
            return mask
        for bond in self.mol.GetAtomWithIdx(anchor).GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                other = bond.GetOtherAtomIdx(anchor)
                if fv[other] > 0:
                    mask[other] = True
        return mask

    def step(self, action_type: int, anchor: int = -1, target: int = -1) -> bool:
        """One MDP action; return done flag. Caller fetches smiles() at termination."""
        self.steps += 1
        self.cached_fv = None
        self.cached_bam = None
        if action_type == ACT_STOP:
            return True
        if action_type < NUM_ATOM_TYPES and anchor >= 0:
            self.place_atom(action_type, anchor)
        elif action_type == ACT_BOND and anchor >= 0 and target >= 0:
            self.place_bond(anchor, target)
        elif action_type == ACT_UPGRADE and anchor >= 0 and target >= 0:
            self.upgrade_bond(anchor, target)
        return self.steps >= self.max_steps

    def place_atom(self, type_idx: int, anchor: int):
        new_idx = self.mol.AddAtom(Chem.Atom(ATOM_SYMBOLS[type_idx]))
        self.mol.AddBond(anchor, new_idx, Chem.BondType.SINGLE)

    def place_bond(self, i: int, j: int):
        if self.mol.GetBondBetweenAtoms(i, j) is None:
            self.mol.AddBond(i, j, Chem.BondType.SINGLE)

    def upgrade_bond(self, i: int, j: int):
        bond = self.mol.GetBondBetweenAtoms(i, j)
        if bond and bond.GetBondType() == Chem.BondType.SINGLE:
            bond.SetBondType(Chem.BondType.DOUBLE)

    def mol_graph(self) -> Optional[Data]:
        n = self.mol.GetNumAtoms()
        if n == 0:
            return None
        self.mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(self.mol)
        ri = self.mol.GetRingInfo()
        x = torch.tensor(
            [atom_vec(a, ri) for a in self.mol.GetAtoms()], dtype=torch.float)
        if self.mol.GetNumBonds() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float)
        else:
            src, dst, attrs = [], [], []
            for b in self.mol.GetBonds():
                i, j, bv = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), bond_vec(b)
                src += [i, j]
                dst += [j, i]
                attrs += [bv, bv]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(attrs, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class VecMolEnv:
    """N parallel MolEnvs for batched rollouts."""

    def __init__(self, n_envs: int, max_steps: int = 60, max_atoms: int = 60):
        self.envs = [MolEnv(max_steps, max_atoms) for _ in range(n_envs)]
        self.n = n_envs

    def reset_all(self):
        for env in self.envs:
            env.reset()

    def set_max_steps(self, val: int):
        for env in self.envs:
            env.max_steps = val



# Policy: GATv2 encoder + autoregressive type/anchor/target heads + value

class PolicyEncoder(nn.Module):
    """GATv2 encoder with LayerNorm; mean and max pool over nodes."""

    def __init__(self, hidden: int = 128, layers: int = 3,
                 dropout: float = 0.1, heads: int = 4, edge_dim: int = 4):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Linear(ATOM_FEAT_DIM, hidden)
        self.convs = nn.ModuleList([
            GATv2Conv(hidden, hidden, heads=heads,
                      edge_dim=edge_dim, concat=False)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    @property
    def out_dim(self) -> int:
        return self.hidden * 2

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = self.drop(F.relu(norm(
                conv(x, edge_index, edge_attr=edge_attr)))) + x
        graph = torch.cat([global_mean_pool(x, batch),
                           global_max_pool(x, batch)], dim=-1)
        return x, graph


class MolPolicy(nn.Module):
    """Autoregressive policy: type -> anchor -> target. Plus a value head
    sharing the encoder.
    """

    def __init__(self, hidden: int = 128, layers: int = 3,
                 heads: int = 4, edge_dim: int = 4):
        super().__init__()
        self.encoder = PolicyEncoder(
            hidden=hidden, layers=layers, heads=heads, edge_dim=edge_dim)
        gdim = self.encoder.out_dim
        self.type_head = nn.Linear(gdim, NUM_ACT_TYPES)
        self.anchor_proj = nn.Linear(hidden + gdim, 1)
        self.target_proj = nn.Linear(hidden + gdim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(gdim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, graph: Data):
        batch = torch.zeros(
            graph.x.size(0), dtype=torch.long, device=graph.x.device)
        ea = graph.edge_attr if hasattr(graph, "edge_attr") else None
        return self.encoder(graph.x, graph.edge_index, batch, edge_attr=ea)

    def type_dist(self, graph_emb, mask):
        logits = self.type_head(graph_emb).clamp(min=-30.0, max=30.0)
        logits = logits.masked_fill(~mask, float("-inf"))
        return torch.distributions.Categorical(logits=logits)

    def anchor_dist(self, node_emb, mask, graph_emb):
        expanded = graph_emb.expand(node_emb.size(0), -1)
        scores = self.anchor_proj(
            torch.cat([node_emb, expanded], dim=-1)).squeeze(-1)
        scores = scores.clamp(min=-30.0, max=30.0).masked_fill(
            ~mask, float("-inf"))
        return torch.distributions.Categorical(logits=scores)

    def target_dist(self, node_emb, mask, graph_emb):
        expanded = graph_emb.expand(node_emb.size(0), -1)
        scores = self.target_proj(
            torch.cat([node_emb, expanded], dim=-1)).squeeze(-1)
        scores = scores.clamp(min=-30.0, max=30.0).masked_fill(
            ~mask, float("-inf"))
        return torch.distributions.Categorical(logits=scores)

    def state_value(self, graph_emb):
        return self.value_head(graph_emb).squeeze(-1)


# Transition / expert step

@dataclass
class Transition:
    graph: Data
    action_type: int
    anchor: int
    target: int
    type_mask: np.ndarray
    anchor_mask: Optional[np.ndarray]
    target_mask: Optional[np.ndarray]
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass
class ExpertStep:
    """One step of an expert build trajectory for behavioral cloning."""
    graph: Data
    action_type: int
    anchor: int
    target: int
    type_mask: np.ndarray
    anchor_mask: Optional[np.ndarray]
    target_mask: Optional[np.ndarray]

# Buffers

class RolloutBuffer:
    """Append-only transition list with flush-and-reset."""

    def __init__(self):
        self.data: List[Transition] = []

    def extend(self, ts: List[Transition]):
        self.data.extend(ts)

    def flush(self) -> List[Transition]:
        out, self.data = self.data, []
        return out

    def __len__(self) -> int:
        return len(self.data)


class ReplayBuffer:
    """Top-k episode replay with a per-canonical-SMILES cap."""

    def __init__(self, capacity: int = 1000, per_canonical: int = 3):
        self.capacity = capacity
        self.per_canonical = per_canonical
        self.entries: list = []
        self.canonical_counts: dict = {}
        self.counter = 0

    def retain_if_top(self, transitions: List[Transition],
                      reward: float, canonical: Optional[str]):
        if canonical is None or not transitions:
            return
        if self.canonical_counts.get(canonical, 0) >= self.per_canonical:
            same = [(i, e[0]) for i, e in enumerate(self.entries)
                    if e[2] == canonical]
            min_idx, min_r = min(same, key=lambda x: x[1])
            if reward <= min_r:
                return
            del self.entries[min_idx]
            self.canonical_counts[canonical] -= 1
        copies = [Transition(**t.__dict__) for t in transitions]
        self.entries.append((reward, self.counter, canonical, copies))
        self.canonical_counts[canonical] = (
            self.canonical_counts.get(canonical, 0) + 1)
        self.counter += 1
        if len(self.entries) > self.capacity:
            self.evict_to_capacity()

    def evict_to_capacity(self):
        self.entries.sort(key=lambda e: e[0], reverse=True)
        for _, _, c, _ in self.entries[self.capacity:]:
            self.canonical_counts[c] -= 1
            if self.canonical_counts[c] <= 0:
                del self.canonical_counts[c]
        self.entries = self.entries[:self.capacity]

    def draw(self, n: int) -> List[Transition]:
        """Sample whole episodes to preserve temporal structure for GAE."""
        if not self.entries:
            return []
        order = np.random.permutation(len(self.entries))
        result, count = [], 0
        for i in order:
            if count >= n:
                break
            episode = self.entries[i][3]
            result.extend(episode)
            count += len(episode)
        return result

    def flush(self):
        """Empty the buffer"""
        self.entries.clear()
        self.canonical_counts.clear()
        self.counter = 0

    def __len__(self) -> int:
        return len(self.entries)


# Expert trajectory decomposition

def bfs_atom_order(mol):
    """BFS from first carbon. Returns (orig_to_env, parents, tree_bonds)."""
    start = next((i for i in range(mol.GetNumAtoms())
                  if mol.GetAtomWithIdx(i).GetSymbol() == "C"), None)
    if start is None:
        return None, None, None
    visited, queue = {start}, [start]
    orig_to_env, parents, tree_bonds = {start: 0}, {}, set()
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for bond in mol.GetAtomWithIdx(u).GetBonds():
            v = bond.GetOtherAtomIdx(u)
            if v not in visited:
                visited.add(v)
                queue.append(v)
                orig_to_env[v] = len(orig_to_env)
                parents[v] = u
                tree_bonds.add((min(u, v), max(u, v)))
    if len(visited) != mol.GetNumAtoms():
        return None, None, None
    return orig_to_env, parents, tree_bonds


def mol_build_order(smiles):
    """Decompose a molecule into MolEnv build actions via BFS."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 3:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None
    if any(a.GetSymbol() not in ATOM_SYMBOLS for a in mol.GetAtoms()):
        return None
    if any(b.GetBondTypeAsDouble() > 2.0 for b in mol.GetBonds()):
        return None
    o2e, parents, tree_bonds = bfs_atom_order(mol)
    if o2e is None:
        return None
    actions = []
    for orig, env_idx in sorted(o2e.items(), key=lambda x: x[1]):
        if env_idx == 0:
            continue
        sym = mol.GetAtomWithIdx(orig).GetSymbol()
        actions.append((ATOM_SYMBOLS.index(sym), o2e[parents[orig]], -1))
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (min(u, v), max(u, v)) not in tree_bonds:
            actions.append((ACT_BOND, o2e[u], o2e[v]))
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            actions.append((ACT_UPGRADE, o2e[u], o2e[v]))
    actions.append((ACT_STOP, -1, -1))
    return actions


def anchor_mask_by_action(env, action_type, anchor):
    if anchor < 0:
        return None
    if action_type == ACT_UPGRADE:
        return env.upgrade_anchor_mask()
    if action_type == ACT_BOND:
        return env.bond_anchor_mask()
    return env.anchor_mask()


def target_mask_by_action(env, action_type, anchor, target):
    if target < 0 or anchor < 0:
        return None
    if action_type == ACT_UPGRADE:
        return env.upgrade_target_mask(anchor)
    if action_type == ACT_BOND:
        return env.target_mask(anchor)
    return None


def expert_trajectory(smiles):
    """Replay a SMILES build sequence in MolEnv, recording per-step state."""
    actions = mol_build_order(smiles)
    if actions is None:
        return None
    env = MolEnv(max_steps=len(actions) + 5, max_atoms=120)
    env.reset()
    steps = []
    for at, anchor, target in actions:
        graph = env.mol_graph()
        if graph is None:
            break
        tm = env.type_mask()
        if not tm[at]:
            break
        am = anchor_mask_by_action(env, at, anchor)
        if am is not None and anchor >= 0 and not am[anchor]:
            break
        tgt_m = target_mask_by_action(env, at, anchor, target)
        if tgt_m is not None and target >= 0 and not tgt_m[target]:
            break
        steps.append(ExpertStep(graph, at, anchor, target, tm, am, tgt_m))
        if at == ACT_STOP:
            break
        env.step(at, anchor, target)
    return steps if len(steps) >= 3 else None


# Single-graph rollout path


def bool_mask(arr: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.bool, device=device)


def sample_node(policy_fn, mask_np, node_emb, device, graph_emb):
    dist = policy_fn(node_emb, bool_mask(mask_np, device), graph_emb)
    idx = dist.sample().item()
    lp = dist.log_prob(torch.tensor(idx, device=device))
    return idx, lp


def anchor_and_target(policy, env, at, node_emb, device, graph_emb):
    """Resolve anchor and optional target for the chosen action type, or None if no valid pair."""
    anchor, target, am_np, tgt_np, lp = -1, -1, None, None, 0.0
    needs_anchor = at < NUM_ATOM_TYPES or at in (ACT_BOND, ACT_UPGRADE)
    if needs_anchor:
        if at == ACT_UPGRADE:
            am_np = env.upgrade_anchor_mask()
        elif at == ACT_BOND:
            am_np = env.bond_anchor_mask()
        else:
            am_np = env.anchor_mask()
        if not am_np.any():
            return None
        anchor, alp = sample_node(
            policy.anchor_dist, am_np, node_emb, device, graph_emb)
        lp = alp
    if at in (ACT_BOND, ACT_UPGRADE):
        tgt_np = (env.upgrade_target_mask(anchor) if at == ACT_UPGRADE
                  else env.target_mask(anchor))
        if not tgt_np.any():
            return None
        target, tlp = sample_node(
            policy.target_dist, tgt_np, node_emb, device, graph_emb)
        lp = lp + tlp
    return at, anchor, target, am_np, tgt_np, lp


def sample_action(policy, env, node_emb, graph_emb, device, cpu_graph):
    """Full autoregressive action from pre-encoded embeddings."""
    tm = env.type_mask()
    if not tm.any():
        return None
    td = policy.type_dist(graph_emb, bool_mask(tm, device).unsqueeze(0))
    at = td.sample().item()
    lp = td.log_prob(torch.tensor(at, device=device))
    resolved = anchor_and_target(
        policy, env, at, node_emb, device, graph_emb)
    if resolved is None:
        return None
    at, anchor, target, am_np, tgt_np, sub_lp = resolved
    lp = lp + sub_lp
    return Transition(
        graph=cpu_graph, action_type=at, anchor=anchor,
        target=target, type_mask=tm, anchor_mask=am_np,
        target_mask=tgt_np, log_prob=lp.item(),
        value=policy.state_value(graph_emb).item(),
        reward=0.0, done=False)


# Batched encoding across parallel envs

def mol_embeddings(policy, envs, device):
    """Single GNN forward across all envs. Returns per-env (node, graph) pairs."""
    graphs = [env.mol_graph() for env in envs]
    valid = [(i, g) for i, g in enumerate(graphs) if g is not None]
    if not valid:
        return None, None, graphs
    indices, raw = zip(*valid)
    batch = Batch.from_data_list(list(raw)).to(device)
    ea = batch.edge_attr if hasattr(batch, "edge_attr") else None
    node_emb, graph_emb = policy.encoder(
        batch.x, batch.edge_index, batch.batch, edge_attr=ea)
    splits = [g.x.size(0) for g in raw]
    node_per = torch.split(node_emb, splits)
    all_nodes: List[Optional[torch.Tensor]] = [None] * len(envs)
    all_graphs_emb: List[Optional[torch.Tensor]] = [None] * len(envs)
    for k, i in enumerate(indices):
        all_nodes[i] = node_per[k]
        all_graphs_emb[i] = graph_emb[k:k + 1]
    return all_nodes, all_graphs_emb, graphs


# Parallel rollout (eval mode and no_grad)

def advance_env(env, policy, node_emb, graph_emb, device, cpu_g,
                prev_phi, scale, gamma):
    """Sample one action, apply it; return (transition, next_phi) or None."""
    t = sample_action(policy, env, node_emb, graph_emb, device, cpu_g)
    if t is None:
        return None
    done = env.step(t.action_type, t.anchor, t.target)
    next_phi = 0.0 if done else mol_potential(env.mol)  # zero at terminal
    t.done = done
    t.reward = float(np.clip(scale * (gamma * next_phi - prev_phi),
                             -SHAPING_CLIP, SHAPING_CLIP))
    return t, next_phi


def parallel_step(policy, vec_env, active, potentials, buffers,
                  device, scale, gamma):
    """One timestep across all active envs with batched encoding."""
    alive = np.where(active)[0]
    if len(alive) == 0:
        return
    nodes, graphs_emb, raw_gs = mol_embeddings(
        policy, [vec_env.envs[i] for i in alive], device)
    if nodes is None:
        active[alive] = False
        return
    for k, i in enumerate(alive):
        if nodes[k] is None:
            active[i] = False
            continue
        cpu_g = raw_gs[k].cpu() if raw_gs[k] is not None else None
        result = advance_env(vec_env.envs[i], policy, nodes[k],
                             graphs_emb[k], device, cpu_g,
                             potentials[i], scale, gamma)
        if result is None:
            active[i] = False
            continue
        t, potentials[i] = result
        buffers[i].append(t)
        if t.done:
            active[i] = False


def parallel_rollout(policy, vec_env, reward_fn, shaping_scale, gamma, device):
    """Collect one episode per env under eval mode and no_grad."""
    n = vec_env.n
    max_steps = vec_env.envs[0].max_steps
    vec_env.reset_all()
    active = np.ones(n, dtype=bool)
    potentials = np.array([mol_potential(e.mol) for e in vec_env.envs])
    buffers: List[list] = [[] for _ in range(n)]
    policy.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            if not active.any():
                break
            parallel_step(policy, vec_env, active, potentials, buffers,
                          device, shaping_scale, gamma)
    return finalize_episodes(vec_env, buffers, reward_fn)


def finalize_episodes(vec_env, buffers, reward_fn):
    """Assign terminal rewards and package results."""
    results = []
    for i, env in enumerate(vec_env.envs):
        smi = env.smiles()
        if buffers[i] and not buffers[i][-1].done:
            buffers[i][-1].done = True
        raw = reward_fn(smi) if smi else 0.0
        if buffers[i]:
            buffers[i][-1].reward += raw
        results.append((smi, raw, buffers[i]))
    return results


# GAE

def advantages(rewards: np.ndarray, values: np.ndarray,
               dones: np.ndarray, gamma: float = 0.97,
               lam: float = 0.92) -> tuple:
    """Vectorized GAE-lambda. Returns -advantages, returns."""
    T = len(rewards)
    nextnonterminal = 1.0 - dones
    nextvalues = np.append(values[1:], 0.0)
    deltas = rewards + gamma * nextvalues * nextnonterminal - values
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        adv[t] = last = deltas[t] + gamma * lam * nextnonterminal[t] * last
    return adv, adv + values


# Batched scoring

def node_kind_tensors(steps_or_trans, ptr_cpu, kind: str, device):
    """Mmask_flat, chosen_global, valid - tensors for one node-action kind across a batch."""
    masks, chosen, valid = [], [], []
    for i, t in enumerate(steps_or_trans):
        n_i = int(ptr_cpu[i + 1] - ptr_cpu[i])
        m = getattr(t, f"{kind}_mask")
        idx = getattr(t, kind)
        if m is not None and idx >= 0:
            masks.append(np.asarray(m, dtype=bool))
            chosen.append(int(ptr_cpu[i]) + int(idx))
            valid.append(True)
        else:
            masks.append(np.zeros(n_i, dtype=bool))
            chosen.append(int(ptr_cpu[i]))
            valid.append(False)
    mask_t = torch.from_numpy(np.concatenate(masks)).to(device)
    chosen_t = torch.from_numpy(np.asarray(chosen, dtype=np.int64)).to(device)
    valid_t = torch.from_numpy(np.asarray(valid, dtype=bool)).to(device)
    return mask_t, chosen_t, valid_t


def seg_log_softmax(scores, batch_idx, mask_flat, num_graphs):
    """Per-graph log-softmax over masked node scores; returns  - log_probs, exp_shifted, sum_exp."""
    max_per_g = torch.full(
        (num_graphs,), float("-inf"),
        device=scores.device, dtype=scores.dtype)
    max_per_g.scatter_reduce_(
        0, batch_idx, scores, reduce="amax", include_self=True)
    max_per_g = max_per_g.clamp(min=-1e6)
    shifted = scores - max_per_g.index_select(0, batch_idx)
    exp_shifted = torch.where(
        mask_flat, shifted.exp(), torch.zeros_like(shifted))
    sum_exp = torch.zeros(
        num_graphs, device=scores.device, dtype=scores.dtype)
    sum_exp.scatter_add_(0, batch_idx, exp_shifted)
    sum_exp = sum_exp.clamp(min=1e-20)
    log_probs = shifted - sum_exp.log().index_select(0, batch_idx)
    return log_probs, exp_shifted, sum_exp


def node_action_stats(proj, all_node, all_graph, batch_idx,
                      mask_flat, chosen_global, valid, num_graphs):
    """Vectorized log-prob at chosen index and entropy per graph for one node-action head."""
    graph_per_node = all_graph.index_select(0, batch_idx)
    raw = proj(torch.cat([all_node, graph_per_node], dim=-1)).squeeze(-1)
    raw = raw.clamp(min=-30.0, max=30.0)
    scores = raw.masked_fill(~mask_flat, -1e9)
    log_probs, exp_shifted, sum_exp = seg_log_softmax(
        scores, batch_idx, mask_flat, num_graphs)
    probs = torch.where(
        mask_flat,
        exp_shifted / sum_exp.index_select(0, batch_idx),
        torch.zeros_like(scores))
    ent_terms = torch.where(
        mask_flat, -probs * log_probs, torch.zeros_like(scores))
    ent_per_g = torch.zeros(
        num_graphs, device=scores.device, dtype=scores.dtype)
    ent_per_g.scatter_add_(0, batch_idx, ent_terms)
    lp_at_chosen = log_probs.index_select(0, chosen_global)
    zero = torch.zeros(num_graphs, device=scores.device, dtype=scores.dtype)
    return torch.where(valid, lp_at_chosen, zero), torch.where(valid, ent_per_g, zero)


# PPO trainer

class PPOTrainer:
    """PPO with batched rollouts, KL-prior anchor, and top-k replay. NaN losses are skipped."""

    def __init__(self, policy: MolPolicy, reward_fn: Callable,
                 cfg: RLConfig, device: torch.device,
                 prior: Optional[MolPolicy] = None):
        self.policy = policy
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=3e-4, eps=1e-5)
        self.buffer = RolloutBuffer()
        self.replay = ReplayBuffer(cfg.replay_capacity, cfg.replay_per_canonical)
        self.shaping_scale = cfg.shaping_scale
        self.replay_frac = cfg.replay_frac
        self.entropy_coeff = cfg.entropy_phase1
        self.kl_coef = cfg.kl_phase1
        self.prior = prior

    def batched_episodes(self, vec_env: VecMolEnv) -> List[tuple]:
        """N parallel episodes. Returns [(smiles, raw_reward), ...]."""
        results = parallel_rollout(
            self.policy, vec_env, self.reward_fn,
            self.shaping_scale, self.cfg.gamma, self.device)
        episode_returns = []
        for smi, raw, transitions in results:
            self.buffer.extend(transitions)
            self.replay.retain_if_top(transitions, raw, smi)
            episode_returns.append((smi, raw))
        return episode_returns

    @torch.no_grad()
    def reeval_replay(self, transitions: List[Transition]):
        """Recompute log_probs and values for replay under the current policy."""
        batch, all_node, all_graph = self.batch_encode(transitions, self.policy)
        lps, vals, _ = self.batched_scores(
            transitions, batch, all_node, all_graph, self.policy)
        for i, t in enumerate(transitions):
            t.log_prob = lps[i].item()
            t.value = vals[i].item()

    def batch_encode(self, transitions: List[Transition], model: MolPolicy):
        """Encoder forward; return Batch, all_node, all_graph"""
        batch = Batch.from_data_list(
            [t.graph for t in transitions]).to(self.device)
        ea = batch.edge_attr if hasattr(batch, "edge_attr") else None
        all_node, all_graph = model.encoder(
            batch.x, batch.edge_index, batch.batch, edge_attr=ea)
        return batch, all_node, all_graph

    def batched_scores(self, trans, batch, all_node, all_graph, model):
        """Vectorized type/value, anchor/target log-probs and entropy."""
        dev = self.device
        type_mask_np = np.stack([t.type_mask for t in trans])
        type_masks = torch.from_numpy(type_mask_np).to(dev)
        type_logits = model.type_head(all_graph).clamp(min=-30.0, max=30.0)
        type_logits = type_logits.masked_fill(~type_masks, float("-inf"))
        type_actions = torch.tensor(
            [t.action_type for t in trans], device=dev)
        type_dist = torch.distributions.Categorical(logits=type_logits)
        type_lps = type_dist.log_prob(type_actions)
        type_ents = type_dist.entropy()
        vals = model.value_head(all_graph).squeeze(-1)
        ptr_cpu = batch.ptr.cpu().numpy()
        a_mask, a_chosen, a_valid = node_kind_tensors(
            trans, ptr_cpu, "anchor", dev)
        t_mask, t_chosen, t_valid = node_kind_tensors(
            trans, ptr_cpu, "target", dev)
        a_lps, a_ents = node_action_stats(
            model.anchor_proj, all_node, all_graph, batch.batch,
            a_mask, a_chosen, a_valid, batch.num_graphs)
        t_lps, t_ents = node_action_stats(
            model.target_proj, all_node, all_graph, batch.batch,
            t_mask, t_chosen, t_valid, batch.num_graphs)
        return type_lps + a_lps + t_lps, vals, type_ents + a_ents + t_ents

    def rollout_tensors(self, transitions):
        """Rewards, advantages."""
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        values = np.array([t.value for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        adv, returns = advantages(
            rewards, values, dones, self.cfg.gamma, self.cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return (
            torch.tensor(returns, device=self.device),
            torch.tensor(adv, device=self.device),
            torch.tensor([t.log_prob for t in transitions],
                         dtype=torch.float32, device=self.device))

    def ppo_minibatches(self, transitions, old_lps_t, adv_t, returns_t,
                        metrics: dict) -> int:
        """All PPO minibatch updates over the rollout. Returns batch count."""
        n = 0
        for _ in range(self.cfg.ppo_epochs):
            idx = np.random.permutation(len(transitions))
            for start in range(0, len(idx), self.cfg.minibatch):
                bi = idx[start:start + self.cfg.minibatch]
                if len(bi) < 2:
                    continue
                step = self.ppo_grad_step(
                    [transitions[i] for i in bi],
                    old_lps_t[bi], adv_t[bi], returns_t[bi])
                for k, v in step.items():
                    metrics[k] += v
                n += 1
        return n

    def ppo_update(self) -> dict:
        """PPO update with replay mixing and KL-prior anchor."""
        transitions = self.buffer.flush()
        if len(transitions) < 2:
            return {}
        n_replay = int(len(transitions) * self.replay_frac)
        replay_trans = self.replay.draw(n_replay)
        if replay_trans:
            self.reeval_replay(replay_trans)
            transitions = transitions + replay_trans
        returns_t, adv_t, old_lps_t = self.rollout_tensors(transitions)
        self.policy.train()
        metrics = {"policy_loss": 0.0, "value_loss": 0.0,
                   "entropy": 0.0, "kl_prior": 0.0, "nan_skip": 0.0}
        n = self.ppo_minibatches(
            transitions, old_lps_t, adv_t, returns_t, metrics)
        return {k: v / max(n, 1) for k, v in metrics.items()}

    def ppo_grad_step(self, batch_trans, old_lp, adv, ret) -> dict:
        """Single PPO gradient step on a minibatch with KL-prior penalty."""
        batch, all_node, all_graph = self.batch_encode(batch_trans, self.policy)
        new_lps, new_vals, ents = self.batched_scores(
            batch_trans, batch, all_node, all_graph, self.policy)
        ent = ents.mean()
        ratio = (new_lps - old_lp).exp()
        clipped = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
        policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
        value_loss = F.mse_loss(new_vals, ret)
        kl = self.kl_to_prior(batch_trans, batch, new_lps)
        loss = (policy_loss + 0.5 * value_loss
                - self.entropy_coeff * ent
                + self.kl_coef * kl)
        if torch.isnan(loss) or torch.isinf(loss):
            return {"policy_loss": 0.0, "value_loss": 0.0,
                    "entropy": 0.0, "kl_prior": 0.0, "nan_skip": 1.0}
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return {"policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": ent.item(),
                "kl_prior": float(kl.item()) if isinstance(kl, torch.Tensor) else 0.0,
                "nan_skip": 0.0}

    def kl_to_prior(self, batch_trans, batch, new_lps):
        """KL(new || prior) through the Schulman k3 estimator -non-negative."""
        if self.prior is None or self.kl_coef <= 0:
            return torch.zeros(1, device=self.device).squeeze()
        with torch.no_grad():
            ea = batch.edge_attr if hasattr(batch, "edge_attr") else None
            all_node_p, all_graph_p = self.prior.encoder(
                batch.x, batch.edge_index, batch.batch, edge_attr=ea)
            prior_lps, _, _ = self.batched_scores(
                batch_trans, batch, all_node_p, all_graph_p, self.prior)
        log_r = (prior_lps - new_lps).clamp(min=-10.0, max=10.0)
        return (log_r.exp() - 1 - log_r).mean()