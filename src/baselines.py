"""
Train the four generation baselines. Random construction, genetic
algorithm, hill climbing, SMILES.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore", message=".*MorganGenerator.*")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import RWMol, BRICS

from config import ProjectConfig, pick_device, release_cache
from src.gnn import MultiTaskGNN, log_mic_to_prob_torch
from src.rewards import (evaluation_reward, gnn_batch_log_mic,
                         morgan_pair, novelty, resistance,
                         qed_score, sa_score, size_gate,
                         applicability_gate, composition_penalty)
from src.rl import ATOM_SYMBOLS, MAX_VALENCE
from src.train_rl import active_smiles

cfg = ProjectConfig()


# Search hyperparameters; protocol documented in the paper.
RANDOM_N = 20000
GA_POP = 200
GA_GENERATIONS = 50
GA_ELITE_FRAC = 0.2
GA_MUTATE_RATE = 0.5
HC_RESTARTS = 1000
HC_STEPS = 50
RNN_SAMPLE_N = 20000
RNN_FINETUNE_ROUNDS = 20
RNN_FINETUNE_BATCH = 200
RNN_PRETRAIN_EPOCHS = 10
RNN_BATCH = 128
RNN_MAX_LEN = 80


# Molecular operators

def random_smiles(max_atoms: int = 30) -> Optional[str]:
    """One molecule via valence-respecting atom-by-atom random walk."""
    mol = RWMol()
    mol.AddAtom(Chem.Atom("C"))
    n_target = np.random.randint(3, max_atoms)
    for _ in range(n_target):
        sym = np.random.choice(ATOM_SYMBOLS)
        anchor = np.random.randint(0, mol.GetNumAtoms())
        atom = mol.GetAtomWithIdx(anchor)
        cap = MAX_VALENCE.get(atom.GetAtomicNum(), 4)
        used = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
        if used >= cap:
            continue
        new_idx = mol.AddAtom(Chem.Atom(sym))
        mol.AddBond(anchor, new_idx, Chem.BondType.SINGLE)
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def mutate(smiles: str) -> Optional[str]:
    """Single-point mutation. Atom-swap or atom-add. None on failure.
    Atom-swap requires n>1 so the seed atom can't be silently dropped."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    rw = RWMol(mol)
    n = rw.GetNumAtoms()
    if np.random.rand() < 0.5 and n > 1:
        idx = np.random.randint(0, n)
        sym = np.random.choice(ATOM_SYMBOLS)
        rw.GetAtomWithIdx(idx).SetAtomicNum(Chem.Atom(sym).GetAtomicNum())
    else:
        anchor = np.random.randint(0, n)
        atom = rw.GetAtomWithIdx(anchor)
        cap = MAX_VALENCE.get(atom.GetAtomicNum(), 4)
        used = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
        if used < cap:
            sym = np.random.choice(ATOM_SYMBOLS)
            new_idx = rw.AddAtom(Chem.Atom(sym))
            rw.AddBond(anchor, new_idx, Chem.BondType.SINGLE)
    try:
        return Chem.MolToSmiles(rw)
    except Exception:
        return None


def crossover(s1: str, s2: str) -> Optional[str]:
    """BRICS fragment swap. Falls back to mutation when either parent
    decomposes into fewer than two fragments."""
    m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        return None
    f1 = list(BRICS.BRICSDecompose(m1))
    f2 = list(BRICS.BRICSDecompose(m2))
    if len(f1) < 2 or len(f2) < 2:
        return mutate(s1 if m1.GetNumAtoms() >= m2.GetNumAtoms() else s2)
    hybrid = list(f1)
    hybrid[np.random.randint(0, len(f1))] = f2[np.random.randint(0, len(f2))]
    try:
        built = list(BRICS.BRICSBuild(
            [Chem.MolFromSmiles(f) for f in hybrid]))
    except Exception:
        return mutate(s1)
    return Chem.MolToSmiles(built[0]) if built else mutate(s1)


def valid_smiles(pool: List[Optional[str]]) -> List[str]:
    """Canonicalize, drop None and parse failures. Duplicates kept so
    convergence behavior (e.g. GA collapsing onto one structure) stays
    visible to downstream evaluation."""
    out = []
    for s in pool:
        if s is None:
            continue
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append(Chem.MolToSmiles(m))
    return out


# Random construction baseline

def random_pool(n: int = RANDOM_N) -> List[str]:
    """Reward-agnostic atom-by-atom construction."""
    return valid_smiles([random_smiles() for _ in range(n)])


# Search-time scoring batches the GNN potency forward (the dominant cost)
# across the pool. Per-call-equivalent because eval-mode BatchNorm is
# batch-size invariant. eval_baselines.py uses the canonical per-call path.

def batch_potency(smiles_list: List[str], gnn,
                  device, threshold: float) -> np.ndarray:
    """Mean-over-organisms active probability for each SMILES.
    NaN for unparseable rows (caller should treat as 0)."""
    log_mic = gnn_batch_log_mic(smiles_list, gnn, device)
    probs = log_mic_to_prob_torch(torch.from_numpy(log_mic), threshold)
    return probs.mean(dim=1).numpy()


def batch_score(smiles_list: List[str], reward_fn,
                gnn, device) -> np.ndarray:
    """One batched GNN forward for potency; remaining components per
    molecule; combine via reward_fn.weighted_total."""
    potencies = batch_potency(
        smiles_list, gnn, device, reward_fn.mic_threshold)
    scores = np.zeros(len(smiles_list), dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        if np.isnan(potencies[i]):
            continue
        mol, fp = morgan_pair(smi)
        if mol is None or fp is None:
            continue
        c = {"potency": float(potencies[i]),
             "novelty": novelty(fp, reward_fn.drugbank),
             "resistance": resistance(fp, reward_fn.card),
             "qed": qed_score(mol),
             "sa": sa_score(mol),
             "size_gate": size_gate(mol, reward_fn.size_center,
                                    reward_fn.size_steepness,
                                    reward_fn.gate_floor),
             "ad_gate": applicability_gate(fp, reward_fn.training),
             "composition": composition_penalty(
                 mol, reward_fn.atom_ref, reward_fn.atom_symbols,
                 reward_fn.atom_tau)}
        scores[i] = reward_fn.weighted_total(c)
    return scores


# Genetic algorithm

def ga_step(pop: List[str], reward_fn, gnn, device,
            pop_size: int, elite_frac: float,
            mutate_rate: float) -> List[str]:
    """One GA generation. Batched scoring, take elites, breed children.
    Capped at 4*pop_size breeding attempts so a stalled crossover/mutate
    cannot spin forever on pathological populations."""
    scores = batch_score(pop, reward_fn, gnn, device)
    order = np.argsort(-scores)
    n_elite = max(int(len(pop) * elite_frac), 2)
    elites = [pop[i] for i in order[:n_elite]]
    if len(elites) < 2:
        return list(pop)
    children = list(elites)
    attempts, attempt_cap = 0, pop_size * 4
    while len(children) < pop_size and attempts < attempt_cap:
        attempts += 1
        p1, p2 = np.random.choice(elites, 2, replace=False)
        child = crossover(p1, p2)
        if child is not None and np.random.rand() < mutate_rate:
            child = mutate(child)
        if child is not None:
            children.append(child)
    return children


def genetic_pool(reward_fn, gnn, device,
                 pop_size: int = GA_POP,
                 generations: int = GA_GENERATIONS,
                 elite_frac: float = GA_ELITE_FRAC,
                 mutate_rate: float = GA_MUTATE_RATE) -> List[str]:
    """Genetic algorithm baseline. Returns the final-generation pool."""
    pop = valid_smiles([random_smiles() for _ in range(pop_size)])
    for _ in range(generations):
        pop = ga_step(pop, reward_fn, gnn, device,
                      pop_size, elite_frac, mutate_rate)
    return valid_smiles(pop)


# Hill climbing

def hill_pool(reward_fn, gnn, device,
              restarts: int = HC_RESTARTS,
              steps: int = HC_STEPS) -> List[str]:
    """Hill-climbing baseline. All restarts step in lockstep. Per-step
    scoring is batched across the pool. One molecule per restart."""
    current = [s for s in (random_smiles() for _ in range(restarts))
               if s is not None]
    if not current:
        return []
    scores = batch_score(current, reward_fn, gnn, device)
    for _ in range(steps):
        proposals = [(mutate(s) or s) for s in current]
        prop_scores = batch_score(proposals, reward_fn, gnn, device)
        better = prop_scores > scores
        for i in np.where(better)[0]:
            current[i] = proposals[i]
            scores[i] = prop_scores[i]
    return valid_smiles(current)


# SMILES-RNN

class CharRNN(nn.Module):
    """Character-level GRU decoder."""

    def __init__(self, vocab_size: int, embed: int = 64, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        e = self.embed(x)
        o, h = self.rnn(e, h)
        return self.out(o), h


class SmilesRNN:
    """Character-level SMILES generator. Sampling is fully batched on
    device; decoding to Python strings happens once on CPU at the end."""

    PAD = " "

    def __init__(self, device: torch.device, corpus: List[str],
                 max_len: int = RNN_MAX_LEN):
        self.device = device
        self.max_len = max_len
        self.vocab = sorted({c for s in corpus for c in s} | {self.PAD})
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.pad_idx = self.char2idx[self.PAD]
        self.model = CharRNN(len(self.vocab)).to(device)

    def encode_smiles(self, smiles: str) -> List[int]:
        return [self.char2idx.get(c, self.pad_idx) for c in smiles]

    def padded_batch(self, smiles_list: List[str]):
        """Pack one batch into (input, target, mask) tensors. Sequences
        shorter than 2 tokens after PAD-append are dropped."""
        encoded = [self.encode_smiles(s + self.PAD) for s in smiles_list]
        encoded = [e for e in encoded if len(e) >= 2]
        if not encoded:
            return None, None, None
        max_t = max(len(e) for e in encoded) - 1
        x = torch.full((len(encoded), max_t), self.pad_idx,
                       dtype=torch.long, device=self.device)
        y = torch.full((len(encoded), max_t), self.pad_idx,
                       dtype=torch.long, device=self.device)
        mask = torch.zeros(len(encoded), max_t,
                           dtype=torch.bool, device=self.device)
        for i, e in enumerate(encoded):
            seq_len = len(e) - 1
            x[i, :seq_len] = torch.tensor(e[:-1], device=self.device)
            y[i, :seq_len] = torch.tensor(e[1:], device=self.device)
            mask[i, :seq_len] = True
        return x, y, mask

    def pretrain(self, smiles_list: List[str],
                 epochs: int = RNN_PRETRAIN_EPOCHS,
                 lr: float = 1e-3,
                 batch: int = RNN_BATCH):
        """Maximum-likelihood pretrain on a known SMILES set."""
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        v = len(self.vocab)
        for _ in range(epochs):
            order = np.random.permutation(len(smiles_list))
            for start in range(0, len(order), batch):
                bs = [smiles_list[i] for i in order[start:start + batch]]
                x, y, mask = self.padded_batch(bs)
                if x is None:
                    continue
                logits, _ = self.model(x)
                tok = F.cross_entropy(
                    logits.view(-1, v), y.view(-1), reduction="none")
                loss = (tok * mask.view(-1).float()).sum() / mask.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def reinforce_step(self, batch_smiles: List[str],
                       batch_adv: torch.Tensor) -> Optional[torch.Tensor]:
        """One REINFORCE minibatch. Returns scalar loss or None when
        the batch contains no usable sequence."""
        x, y, mask = self.padded_batch(batch_smiles)
        if x is None:
            return None
        logits, _ = self.model(x)
        tok = F.cross_entropy(
            logits.view(-1, len(self.vocab)), y.view(-1), reduction="none")
        per_seq = (tok.view(x.size(0), -1) * mask.float()).sum(dim=1)
        per_seq = per_seq / mask.sum(dim=1).clamp(min=1)
        return (per_seq * batch_adv[:per_seq.size(0)]).mean()

    def finetune(self, scorer: Callable[[List[str]], np.ndarray],
                 rounds: int = RNN_FINETUNE_ROUNDS,
                 n: int = RNN_FINETUNE_BATCH,
                 lr: float = 5e-4, batch: int = RNN_BATCH):
        """REINFORCE fine-tune with batched per-round scoring.
        `scorer` takes a list of SMILES and returns ndarray of rewards."""
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(rounds):
            mols = self.sample(n)
            rewards = scorer(mols).astype(np.float32)
            adv = torch.tensor(rewards - rewards.mean(),
                               dtype=torch.float32, device=self.device)
            for start in range(0, len(mols), batch):
                loss = self.reinforce_step(
                    mols[start:start + batch], adv[start:start + batch])
                if loss is None:
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()

    def sample(self, n: int, temperature: float = 0.8) -> List[str]:
        """Batched autoregressive sampling. Decoding stops at the first PAD and recovers the right
        string regardless."""
        self.model.eval()
        start = self.char2idx.get("C", 0)
        active = torch.full((n, 1), start, dtype=torch.long, device=self.device)
        out = torch.full((n, self.max_len), self.pad_idx,
                         dtype=torch.long, device=self.device)
        alive = torch.ones(n, dtype=torch.bool, device=self.device)
        h = None
        with torch.no_grad():
            for t in range(self.max_len):
                logits, h = self.model(active, h)
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                sampled = torch.multinomial(probs, 1)
                flat = sampled.squeeze(-1)
                out[alive, t] = flat[alive]
                alive = alive & (flat != self.pad_idx)
                active = sampled
                if not alive.any():
                    break
        return self.decode_batch(out.cpu().numpy())

    def decode_batch(self, ids: np.ndarray) -> List[str]:
        """Decode integer rows into SMILES strings, terminating at PAD."""
        out = []
        for row in ids:
            chars = []
            for idx in row:
                if idx == self.pad_idx:
                    break
                chars.append(self.idx2char[int(idx)])
            out.append("".join(chars))
        return out


def rnn_pool(reward_fn, gnn, device: torch.device,
             pretrain_smiles: List[str],
             n: int = RNN_SAMPLE_N,
             rounds: int = RNN_FINETUNE_ROUNDS,
             finetune_n: int = RNN_FINETUNE_BATCH) -> List[str]:
    """Pretrain on actives, REINFORCE fine-tune against batched reward,
    sample n. The fine-tune scorer batches the GNN forward across the
    per-round draw."""
    rnn = SmilesRNN(device, pretrain_smiles)
    rnn.pretrain(pretrain_smiles)
    scorer = lambda smis: batch_score(smis, reward_fn, gnn, device)
    rnn.finetune(scorer, rounds=rounds, n=finetune_n)
    return valid_smiles(rnn.sample(n))


# Runner: trains all four baselines, saves baseline_{name}.csv per run.
# Per-baseline resume via existence check on the CSV; delete the file
# to force re-run of one baseline. The reward seen by the search is the
# same reward eval_baselines.py and stat_tests.py use to score the pools.

def trained_gnn(device: torch.device) -> MultiTaskGNN:
    """Locked GNN from gnn_best.pt for canonical scoring."""
    model = MultiTaskGNN(cfg.atom, cfg.gnn).to(device)
    ckpt = cfg.paths.models / "gnn_best.pt"
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def pool_path(name: str) -> Path:
    return cfg.paths.results / f"baseline_{name}.csv"


def run_pool(name: str, generator: Callable[[], List[str]],
             device: torch.device):
    """Per-baseline resume via existence check. Release device cache after every run because the
    RNN baseline allocates a model on device and MPS does not shrink
    its allocator pool proactively."""
    path = pool_path(name)
    if path.exists():
        print(f"  {name}: skipping (already at {path.name})")
        return
    print(f"  {name}: running...")
    pool = generator()
    pd.DataFrame({"smiles": pool}).to_csv(path, index=False)
    print(f"  {name}: saved {len(pool):,} molecules to {path.name}")
    release_cache(device)


def main():
    cfg.ensure_dirs()
    device = pick_device()
    print(f"Device: {device}")
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    gnn = trained_gnn(device)
    reward_fn = evaluation_reward(gnn, device)
    corpus = active_smiles()
    print(f"RNN pretrain corpus: {len(corpus):,} active SMILES")
    run_pool("random", random_pool, device)
    run_pool("genetic_algorithm",
             lambda: genetic_pool(reward_fn, gnn, device), device)
    run_pool("hill_climbing",
             lambda: hill_pool(reward_fn, gnn, device), device)
    run_pool("smiles_rnn",
             lambda: rnn_pool(reward_fn, gnn, device, corpus), device)
    print(f"\nBaseline pools in {cfg.paths.results}")


if __name__ == "__main__":
    main()