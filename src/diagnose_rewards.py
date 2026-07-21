"""
Pre-RL diagnostic suite
"""

import sys
sys.path.insert(0,
    str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore", message=".*MorganGenerator.*")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import argparse
import contextlib
import io

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import RWMol

from config import ProjectConfig, pick_device, release_cache
from src.gnn import MultiTaskGNN
from src.rewards import fp_array, training_reward
from src.rl import (MolPolicy, VecMolEnv, ATOM_SYMBOLS, MAX_VALENCE,
                    parallel_step, mol_potential)
from src.train_rl import (active_smiles, fit_surrogate,
                          seeded_policy, expert_dataset, pretrain_policy,
                          atomic_save, gnn_log_mic_targets,
                          stratified_smiles)

cfg = ProjectConfig()

DIVIDER = "-" * 110
SIZE_CENTERS = [20.0, 25.0, 30.0, 35.0]
LANDSCAPE_SAMPLES = 500
POLICY_SAMPLES = 200
AGREEMENT_PER_BUCKET = 30
GROWTH_THRESHOLD = 0.005
LANDSCAPE_FLAT_CV = 0.10
LANDSCAPE_LOW_CV = 0.20


PROBES = [
    ("phase1 tiny", "CC"),
    ("phase1 small", "CCC"),
    ("phase1 small N", "CCCN"),
    ("phase1 5atom", "CCCCC"),
    ("phase1 5atom CN", "CCCCN"),
    ("mid 14atom", "CCCCN1CCCCC1CC"),
    ("mid 14 ringy", "c1ccc(CCN)cc1CCN"),
    ("novel 21atom drug-like",
     "CCC1=CC2=C(N=C1)N=C(NC3CCCCC3O)N=C2N"),
    ("novel 32atom drug-like",
     "CN1CCN(CC1)c2ccc(cc2)C(=O)Nc3ccc(cc3)C(=O)NCc4ccncc4"),
    ("drug ampicillin",
     "CC1(C)SC2C(NC(=O)C(N)c3ccccc3)C(=O)N2C1C(=O)O"),
    ("drug ciprofloxacin",
     "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"),
    ("drug halicin", "Nc1nnc(SC2=NN=C(N)S2)s1"),
    ("hack polysulfide", "FN=C1N(SSOF)C1(F)n1[nH]s1"),
    ("hack S-O chain", "FOC12n3[nH]on4n3C13OSSC23OS4"),
    ("hack pure S", "S1SSSSSSSSSSSSS1"),
]

# Growth parents span the whole size range so that at every size center some
# parents sit below the band and some above it. The band is two-sided, so the
# expected gradient sign differs by side. Growth pays below the center and
# costs above it.
GROWTH_PARENTS = [
    "CCC1=NC(=NC=C1)N(C)CC",
    "CC(C)NCCN1CCN(CC1)C",
    "Nc1ccc(NC(=O)CCC)cc1",
    "Cc1ccc(NCCN(C)C)cc1O",
    "CCN(CC)CCNc1ccccc1",
    "CCNc1nc(N)nc2c1ncn2CC",
    "CN1CCN(CC1)c2ccccc2NCCO",
    "Cc1ccc(N2CCN(CCO)CC2)cc1Cl",
    "Nc1ccc(C(=O)N2CCN(CC)CC2)cc1",
    "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",
    "OCC(NC(=O)C(Cl)Cl)C(O)c1ccc([N+](=O)[O-])cc1",
    "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC",
    "CCC1=CC2=C(N=C1)N=C(NC3CCCCC3O)N=C2N",
    "CC(C)Cc1ccc(C(C)C(=O)NC2CCCCC2)cc1",
    "CC1(C)SC2C(NC(=O)C(N)c3ccccc3)C(=O)N2C1C(=O)O",
    "CC(=O)NCC1CN(c2ccc(N3CCOCC3)c(F)c2)C(=O)O1",
    "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
    "CCCC1CC(N(C)C1)C(=O)NC(C(C)Cl)C1OC(SC)C(O)C(O)C1O",
    "CN(C)C1C2CC3C(=O)c4c(O)cccc4C(C)(O)C3CC2C(O)=C(C(N)=O)C1=O",
    "CN1CCN(CC1)c2ccc(cc2)C(=O)Nc3ccc(cc3)C(=O)NCc4ccncc4",
    "CC1(O)C2CC3C(N(C)C)C(O)=C(C(N)=O)C(=O)C3(O)C(O)=C2C(=O)c2c(O)cccc21",
    "CC1C=CC=C(C)C(=O)NC2=C(O)C3=C(O)C(C)=C(O)C(=C3C(=O)C2=O)C(C)C(O)C(C)C1O",
    "CN1CCN(CC1)c1ccc(cc1)C(=O)Nc1ccc(cc1)C(=O)Nc1ccc(cc1)C(=O)NC",
    "CCC1OC(=O)C(C)C(O)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O",
    "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)"
    "C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O",
]


# Setup


def trained_gnn(device) -> MultiTaskGNN:
    model = MultiTaskGNN(cfg.atom, cfg.gnn).to(device)
    ckpt = cfg.paths.models / "gnn_best.pt"
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def phase_gates(reward_fn, size_center, potency_floor):
    """Reset reward gate parameters in place."""
    reward_fn.size_center = size_center
    reward_fn.potency_floor = potency_floor



# Section 1: reward sanity probes


def probe_table(reward_fn, label) -> pd.DataFrame:
    rows = []
    for name, smi in PROBES:
        d = reward_fn.detailed(smi)
        d["probe"] = name
        d["smiles"] = smi
        d["phase"] = label
        rows.append(d)
    return pd.DataFrame(rows)


def print_probes(df: pd.DataFrame, header: str):
    print(f"\n{DIVIDER}\n  {header}\n{DIVIDER}")
    print(f"  {'probe':<22} {'total':>6} {'pot':>5} {'nov':>5} "
          f"{'res':>5} {'qed':>5} {'sa':>5} {'gate':>5} {'comp':>5} smiles")
    for _, r in df.iterrows():
        print(f"  {r['probe']:<22} {r['total']:>6.3f} {r['potency']:>5.2f} "
              f"{r['novelty']:>5.2f} {r['resistance']:>5.2f} "
              f"{r['qed']:>5.2f} {r['sa']:>5.2f} {r['size_gate']:>5.2f} "
              f"{r['composition']:>5.2f}  {r['smiles'][:40]}")


def probes_two_phases(reward_fn) -> tuple:
    p1 = cfg.rl.size_center_phase1
    p3 = cfg.rl.size_center_phase2_end
    phase_gates(reward_fn, p1, 0.0)
    df_p1 = probe_table(reward_fn, "phase1")
    print_probes(df_p1, f"Section 1.a: Phase 1 reward probes (size_c={p1:.0f})")
    phase_gates(reward_fn, p3, cfg.rl.potency_floor)
    df_p3 = probe_table(reward_fn, "phase3")
    print_probes(df_p3, f"Section 1.b: Phase 3 reward probes (size_c={p3:.0f})")
    return df_p1, df_p3


# Section 2: reward landscape variance


def random_molecule(target_atoms: float, rng) -> str:
    """Random valid SMILES near target_atoms heavy atoms."""
    sigma = max(1.0, target_atoms / 5.0)
    n = max(2, int(rng.normal(target_atoms, sigma)))
    mol = RWMol()
    mol.AddAtom(Chem.Atom("C"))
    for _ in range(n):
        sym = ATOM_SYMBOLS[rng.integers(len(ATOM_SYMBOLS))]
        anchor = int(rng.integers(0, mol.GetNumAtoms()))
        atom = mol.GetAtomWithIdx(anchor)
        cap = MAX_VALENCE.get(atom.GetAtomicNum(), 4)
        used = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
        if used >= cap:
            continue
        new = mol.AddAtom(Chem.Atom(sym))
        mol.AddBond(anchor, new, Chem.BondType.SINGLE)
    if mol.GetNumAtoms() < 2:
        return None
    try:
        smi = Chem.MolToSmiles(mol)
        return smi if Chem.MolFromSmiles(smi) is not None else None
    except Exception:
        return None


def smiles_batch(target_atoms: float, n_samples: int, rng) -> list:
    """Up to n valid random SMILES near target_atoms; cap retries at n*10."""
    out, attempts, max_attempts = [], 0, n_samples * 10
    while len(out) < n_samples and attempts < max_attempts:
        attempts += 1
        s = random_molecule(target_atoms, rng)
        if s is not None:
            out.append(s)
    return out


def batch_potencies(reward_fn, smiles_list: list) -> dict:
    """Single MPS forward pass; smi -> active-class probability."""
    if reward_fn.surrogate is None:
        return {}
    fps, valid = [], []
    for s in smiles_list:
        arr = fp_array(s)
        if arr is not None:
            fps.append(arr)
            valid.append(s)
    if not fps:
        return {}
    matrix = np.stack(fps).astype(np.float32)
    probs = reward_fn.surrogate.batch_probability(
        matrix, reward_fn.device, reward_fn.mic_threshold)
    return dict(zip(valid, probs.tolist()))


def scored_rows(reward_fn, smiles_list: list,
                potency_cache: dict) -> list:
    """detailed() over smiles with potency overridden by cache."""
    original = reward_fn.potency
    reward_fn.potency = lambda s, fp: potency_cache.get(s, 0.0)
    try:
        rows = []
        for s in smiles_list:
            d = reward_fn.detailed(s)
            d["smiles"] = s
            rows.append(d)
        return rows
    finally:
        reward_fn.potency = original


def landscape_samples(reward_fn, target_atoms: float,
                      n_samples: int, rng) -> pd.DataFrame:
    """Reward components for n random molecules near target size."""
    smiles = smiles_batch(target_atoms, n_samples, rng)
    if not smiles:
        return pd.DataFrame()
    cache = batch_potencies(reward_fn, smiles)
    return pd.DataFrame(scored_rows(reward_fn, smiles, cache))


def landscape_row(df: pd.DataFrame, size_c: float) -> dict:
    """One landscape summary row."""
    return {"size_c": size_c, "n": len(df),
            "total_mean": float(df["total"].mean()),
            "total_std": float(df["total"].std()),
            "total_p5": float(df["total"].quantile(0.05)),
            "total_p95": float(df["total"].quantile(0.95)),
            "potency_mean": float(df["potency"].mean()),
            "potency_std": float(df["potency"].std()),
            "qed_std": float(df["qed"].std()),
            "sa_std": float(df["sa"].std())}


def landscape_table(reward_fn, size_centers, n_samples, rng) -> pd.DataFrame:
    rows = []
    for sc in size_centers:
        print(f"  scoring landscape at size_c={sc:.0f}")
        phase_gates(reward_fn, sc, 0.0)
        df = landscape_samples(reward_fn, sc, n_samples, rng)
        rows.append(landscape_row(df, sc))
    return pd.DataFrame(rows)


def print_landscape(df: pd.DataFrame):
    print(f"\n{DIVIDER}\n  Section 2: Reward landscape variance "
          f"(n={LANDSCAPE_SAMPLES} per size)\n{DIVIDER}")
    print(f"  {'size_c':>6} {'n':>4} {'mean':>6} {'std':>6} {'p5':>6} "
          f"{'p95':>6} {'pot_mu':>7} {'pot_sd':>7} {'qed_sd':>7} {'sa_sd':>7}")
    for _, r in df.iterrows():
        print(f"  {r['size_c']:>6.0f} {r['n']:>4} {r['total_mean']:>6.3f} "
              f"{r['total_std']:>6.3f} {r['total_p5']:>6.3f} "
              f"{r['total_p95']:>6.3f} {r['potency_mean']:>7.3f} "
              f"{r['potency_std']:>7.3f} {r['qed_std']:>7.3f} "
              f"{r['sa_std']:>7.3f}")


# Section 3: growth gradient across the curriculum

def grow_one_atom(smiles: str) -> list:
    """At most 3 +1-atom variants by appending C/N/O at first free anchor."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    out = []
    for sym in ["C", "N", "O"]:
        rw = RWMol(mol)
        for anchor in range(rw.GetNumAtoms()):
            atom = rw.GetAtomWithIdx(anchor)
            cap = MAX_VALENCE.get(atom.GetAtomicNum(), 4)
            used = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
            if used >= cap:
                continue
            new = RWMol(rw)
            idx = new.AddAtom(Chem.Atom(sym))
            new.AddBond(anchor, idx, Chem.BondType.SINGLE)
            try:
                out.append(Chem.MolToSmiles(new))
            except Exception:
                continue
            break
    return out


def growth_at_size(reward_fn, parents: list, size_c: float) -> pd.DataFrame:
    phase_gates(reward_fn, size_c, 0.0)
    rows = []
    for p in parents:
        children = grow_one_atom(p)
        if not children:
            continue
        pr = reward_fn(p)
        n = Chem.MolFromSmiles(p).GetNumHeavyAtoms()
        crs = np.array([reward_fn(c) for c in children], dtype=np.float32)
        rows.append({"size_c": size_c, "parent": p,
                     "n_atoms": n,
                     "side": "below" if n < size_c else "above",
                     "parent_r": float(pr),
                     "child_mean": float(crs.mean()),
                     "child_best": float(crs.max()),
                     "delta_mean": float(crs.mean() - pr),
                     "delta_best": float(crs.max() - pr)})
    return pd.DataFrame(rows)


def growth_table(reward_fn, size_centers) -> pd.DataFrame:
    frames = [growth_at_size(reward_fn, GROWTH_PARENTS, sc)
              for sc in size_centers]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def print_growth(df: pd.DataFrame):
    print(f"\n{DIVIDER}\n  Section 3: Growth gradient across curriculum"
          f"\n{DIVIDER}")
    if df.empty:
        print("  (no parents produced valid +1-atom variants)")
        return
    print(f"  {'size_c':>6} {'side':>6} {'n_par':>5} {'d_mean':>8} "
          f"{'d_best':>8}  sign")
    for (sc, side), sub in df.groupby(["size_c", "side"]):
        d_m = float(sub["delta_mean"].mean())
        d_b = float(sub["delta_best"].mean())
        sign = "rewards growth" if d_m > 0 else "punishes growth"
        print(f"  {sc:>6.0f} {side:>6} {len(sub):>5} {d_m:>+8.4f} "
              f"{d_b:>+8.4f}  {sign}")


# Section 4: surrogate vs GNN agreement

def agreement_test_smiles(anchors: list, n_per: int, rng) -> list:
    """Deduplicated mix of random-across-sizes molecules, sampled actives, and probes."""
    out = []
    for sc in SIZE_CENTERS:
        for _ in range(n_per):
            smi = random_molecule(sc, rng)
            if smi:
                out.append(smi)
    pool = list(anchors)
    rng.shuffle(pool)
    out.extend(pool[:n_per * len(SIZE_CENTERS)])
    out.extend(s for _, s in PROBES)
    return list(set(out))


def surrogate_agreement(gnn, surrogate, device, smiles_list) -> pd.DataFrame:
    """Per-SMILES (gnn_prob, surr_prob), both under the canonical
    mean-over-organisms sigmoid the reward uses."""
    fps, gnn_log = gnn_log_mic_targets(gnn, device, smiles_list)
    if not fps:
        return pd.DataFrame()
    fps_t = torch.tensor(np.array(fps), dtype=torch.float32, device=device)
    log_thr = float(np.log10(cfg.data.mic_threshold))
    with torch.no_grad():
        surr_log = surrogate(fps_t).cpu().numpy()
    gnn_prob = (1.0 / (1.0 + np.exp(-(log_thr - gnn_log)))).mean(axis=1)
    surr_prob = (1.0 / (1.0 + np.exp(-(log_thr - surr_log)))).mean(axis=1)
    return pd.DataFrame({"gnn_prob": gnn_prob.tolist(),
                         "surr_prob": surr_prob.tolist()})


def agreement_stats(df: pd.DataFrame, threshold: float) -> dict:
    """Pearson r, MAE, and active-class agreement in probability space.
    `threshold` is unused; the active-class cutoff is fixed at 0.5."""
    del threshold
    if df.empty or len(df) < 2:
        return {}
    g = df["gnn_prob"].values.astype(np.float32)
    s = df["surr_prob"].values.astype(np.float32)
    if g.std() < 1e-8 or s.std() < 1e-8:
        r = 0.0
    else:
        r = float(np.corrcoef(g, s)[0, 1])
    mae = float(np.abs(g - s).mean())
    agree = float(((g >= 0.5) == (s >= 0.5)).mean())
    return {"n": len(df), "pearson_r": r,
            "mae_prob": mae, "active_agreement": agree}


def surrogate_section(gnn, surrogate, device, anchors, rng) -> dict:
    print(f"\n{DIVIDER}\n  Section 4: Surrogate vs GNN agreement"
          f"\n{DIVIDER}")
    smiles = agreement_test_smiles(anchors, AGREEMENT_PER_BUCKET, rng)
    df = surrogate_agreement(gnn, surrogate, device, smiles)
    stats = agreement_stats(df, cfg.data.mic_threshold)
    if not stats:
        print("  (insufficient valid molecules for agreement test)")
        return stats
    print(f"  n={stats['n']}  pearson r={stats['pearson_r']:.3f}  "
          f"MAE={stats['mae_prob']:.3f} prob  "
          f"active agreement={stats['active_agreement']:.1%}")
    return stats


# Sections 5 and 6: policy population sampling


def diagnostic_rollout(policy, vec_env, device) -> list:
    """Eval-mode rollout. Returns [(smiles, transitions), ...]."""
    n = vec_env.n
    vec_env.reset_all()
    active = np.ones(n, dtype=bool)
    potentials = np.array([mol_potential(e.mol) for e in vec_env.envs])
    buffers = [[] for _ in range(n)]
    policy.eval()
    with torch.no_grad():
        for _ in range(vec_env.envs[0].max_steps):
            if not active.any():
                break
            parallel_step(policy, vec_env, active, potentials,
                          buffers, device, scale=0.0, gamma=cfg.rl.gamma)
    return [(e.smiles(), buffers[i]) for i, e in enumerate(vec_env.envs)]


def policy_samples(policy, reward_fn, device, n_samples) -> list:
    """[(smiles, reward, mean_log_prob), ...] from eval-mode rollouts."""
    n_envs = cfg.rl.n_envs
    rounds = (n_samples + n_envs - 1) // n_envs
    vec = VecMolEnv(n_envs, cfg.rl.max_steps, max_atoms=60)
    out = []
    for _ in range(rounds):
        for smi, transitions in diagnostic_rollout(policy, vec, device):
            raw = reward_fn(smi) if smi else 0.0
            mlp = (float(np.mean([t.log_prob for t in transitions]))
                   if transitions else 0.0)
            out.append((smi, raw, mlp))
    return out[:n_samples]


def parsed_samples(samples: list) -> list:
    """Drop unparseables; attach parsed mol object to each tuple."""
    out = []
    for s, r, lp in samples:
        if s is None:
            continue
        m = Chem.MolFromSmiles(s)
        if m is not None:
            out.append((s, r, lp, m))
    return out


def atom_distribution(parsed: list) -> dict:
    syms = [a.GetSymbol() for _, _, _, m in parsed for a in m.GetAtoms()]
    if not syms:
        return {}
    return {k: float(v) for k, v in
            pd.Series(syms).value_counts(normalize=True).items()}


def population_stats(samples: list) -> dict:
    parsed = parsed_samples(samples)
    n, nv = len(samples), len(parsed)
    if nv == 0:
        return {"n": n, "n_valid": 0, "validity": 0.0, "n_unique": 0,
                "uniqueness": 0.0, "mean_reward": 0.0,
                "realized_entropy": 0.0, "median_atoms": 0.0,
                "p5_atoms": 0.0, "p95_atoms": 0.0, "atom_dist": {}}
    canonical = {Chem.MolToSmiles(m) for _, _, _, m in parsed}
    rewards = np.array([r for _, r, _, _ in parsed])
    lps = np.array([lp for _, _, lp, _ in parsed])
    atoms = np.array([m.GetNumHeavyAtoms() for _, _, _, m in parsed])
    return {"n": n, "n_valid": nv, "validity": nv / n,
            "n_unique": len(canonical), "uniqueness": len(canonical) / nv,
            "mean_reward": float(rewards.mean()),
            "realized_entropy": float(-lps.mean()),
            "median_atoms": float(np.median(atoms)),
            "p5_atoms": float(np.percentile(atoms, 5)),
            "p95_atoms": float(np.percentile(atoms, 95)),
            "atom_dist": atom_distribution(parsed)}


def print_population(label: str, stats: dict):
    print(f"\n  {label}:")
    print(f"    validity:    {stats['validity']:.1%} "
          f"({stats['n_valid']}/{stats['n']})")
    print(f"    uniqueness:  {stats['uniqueness']:.1%} "
          f"({stats['n_unique']} unique)")
    print(f"    mean reward: {stats['mean_reward']:.3f}")
    print(f"    realized H:  {stats['realized_entropy']:.3f}")
    print(f"    heavy atoms: median {stats['median_atoms']:.0f}, "
          f"5th-95th {stats['p5_atoms']:.0f}-{stats['p95_atoms']:.0f}")
    if stats.get("atom_dist"):
        top = sorted(stats["atom_dist"].items(), key=lambda x: -x[1])[:6]
        print("    atom fracs:  " + ", ".join(
            f"{s}={v:.2f}" for s, v in top))


def bc_prior_policy(gnn, device, anchors) -> MolPolicy:
    """Cached BC prior. Trains and saves only if file is missing."""
    prior_path = cfg.paths.models / "policy_prior.pt"
    policy = seeded_policy(device, gnn)
    if prior_path.exists():
        policy.load_state_dict(
            torch.load(prior_path, map_location=device, weights_only=True))
        print(f"  loaded {prior_path.name}")
        return policy
    print(f"  {prior_path.name} not found, BC pretraining (one-time)")
    expert = expert_dataset(anchors, max_mols=3000)
    print(f"  expert steps: {len(expert)}")
    pretrain_policy(policy, expert, device, epochs=cfg.rl.pretrain_epochs)
    atomic_save(policy.state_dict(), prior_path)
    print(f"  saved {prior_path.name}")
    return policy


def phase1_policy(gnn, device):
    """Returns the post-phase-1 policy or None if checkpoint missing."""
    ckpt = cfg.run / "policy_phase1.pt"
    if not ckpt.exists():
        return None
    policy = seeded_policy(device, gnn)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    policy.load_state_dict(state["policy"])
    print(f"  loaded {ckpt.name} at episode {state.get('episode', '?')}")
    return policy


def bc_prior_section(gnn, device, anchors, reward_fn) -> dict:
    print(f"\n{DIVIDER}\n  Section 5: BC prior population\n{DIVIDER}")
    phase_gates(reward_fn, cfg.rl.size_center_phase2_end, cfg.rl.potency_floor)
    policy = bc_prior_policy(gnn, device, anchors)
    samples = policy_samples(policy, reward_fn, device, POLICY_SAMPLES)
    stats = population_stats(samples)
    print_population("BC prior outputs (phase 3 reward)", stats)
    release_cache(device)
    return stats


def phase1_section(gnn, device, reward_fn) -> dict:
    print(f"\n{DIVIDER}\n  Section 6: Phase 1 checkpoint autopsy\n{DIVIDER}")
    policy = phase1_policy(gnn, device)
    if policy is None:
        print("  policy_phase1.pt not found, skipping")
        return {}
    phase_gates(reward_fn, cfg.rl.size_center_phase1, 0.0)
    samples = policy_samples(policy, reward_fn, device, POLICY_SAMPLES)
    stats = population_stats(samples)
    print_population("Phase 1 outputs (phase 1 reward)", stats)
    release_cache(device)
    return stats


# Verdicts

def reward_verdicts(df_p1, df_p3) -> list:
    """Discrimination verdict: the worst novel drug-like probe must score
    at least 3x the best hack probe at the phase 3 size center."""
    out = []
    novel = df_p3[df_p3["probe"].str.startswith("novel")]["total"]
    hacks = df_p3[df_p3["probe"].str.startswith("hack")]["total"]
    if novel.empty or hacks.empty:
        return [("WARN", "novel or hack probes missing")]
    novel_min = float(novel.min())
    hack_max = float(hacks.max())
    ratio = novel_min / max(hack_max, 1e-6)
    status = "PASS" if ratio > 3.0 else "FAIL"
    out.append((status,
                f"novel/hack ratio = {ratio:.1f}x "
                f"(novel min {novel_min:.3f}, hack max {hack_max:.3f})"))
    return out


def landscape_verdicts(df) -> list:
    """Variance verdict on coefficient of variation, not absolute std,
    which the size gate collapses at higher size centers."""
    out = []
    for _, r in df.iterrows():
        sc = r["size_c"]
        std = r["total_std"]
        mean = r["total_mean"]
        cv = std / max(mean, 1e-6)
        msg_tail = f"(CV={cv:.2f}, std={std:.3f})"
        if cv < LANDSCAPE_FLAT_CV:
            out.append(("FAIL", f"size_c={sc:.0f}: landscape flat {msg_tail}"))
        elif cv < LANDSCAPE_LOW_CV:
            out.append(("WARN", f"size_c={sc:.0f}: low variance {msg_tail}"))
        else:
            out.append(("PASS", f"size_c={sc:.0f}: variance OK {msg_tail}"))
    return out


def growth_verdicts(df) -> list:
    """Growth verdict: a healthy two-sided gate rewards growth below the
    band center and penalizes it above; a positive upper-side delta fails."""
    if df.empty:
        return [("WARN", "growth section: no parents valid")]
    want = {"below": 1.0, "above": -1.0}
    out = []
    for (sc, side), sub in df.groupby(["size_c", "side"]):
        d_m = float(sub["delta_mean"].mean())
        signed = want[side] * d_m
        tag = f"size_c={sc:.0f} {side}: {d_m:+.4f} (n={len(sub)})"
        if signed > GROWTH_THRESHOLD:
            out.append(("PASS", f"{tag} gradient correct"))
        elif signed < -GROWTH_THRESHOLD:
            out.append(("FAIL", f"{tag} gradient inverted"))
        else:
            out.append(("WARN", f"{tag} gradient neutral"))
    return out


def surrogate_verdicts(stats) -> list:
    if not stats:
        return [("WARN", "surrogate: insufficient agreement data")]
    msg = (f"r={stats['pearson_r']:.2f}  MAE={stats['mae_prob']:.2f}  "
           f"agree={stats['active_agreement']:.1%}")
    threshold = cfg.surrogate.agreement_threshold
    status = "PASS" if stats["pearson_r"] >= threshold else "FAIL"
    return [(status, f"surrogate vs GNN: {msg}")]


def policy_verdicts(label: str, stats: dict) -> list:
    if not stats or stats.get("n_valid", 0) == 0:
        return [("FAIL", f"{label}: no valid molecules")]
    out = []
    if stats["validity"] < 0.5:
        out.append(("WARN",
                    f"{label}: validity {stats['validity']:.1%} below 50%"))
    uniq = stats["uniqueness"]
    if uniq < 0.3:
        out.append(("FAIL",
                    f"{label}: mode collapse ({uniq:.1%} unique)"))
    elif uniq < 0.6:
        out.append(("WARN",
                    f"{label}: low diversity ({uniq:.1%} unique)"))
    else:
        out.append(("PASS",
                    f"{label}: diversity OK ({uniq:.1%} unique)"))
    if stats["realized_entropy"] < 0.5:
        out.append(("FAIL",
                    f"{label}: realized H={stats['realized_entropy']:.2f} "
                    "indicates collapse"))
    return out


def all_verdicts(df_p1, df_p3, df_land, df_grow,
                 surr_stats, bc_stats, p1_stats) -> list:
    out = reward_verdicts(df_p1, df_p3)
    out.extend(landscape_verdicts(df_land))
    out.extend(growth_verdicts(df_grow))
    out.extend(surrogate_verdicts(surr_stats))
    if bc_stats:
        out.extend(policy_verdicts("BC prior", bc_stats))
    if p1_stats:
        out.extend(policy_verdicts("Phase 1", p1_stats))
    return out


def print_verdicts(verdicts: list):
    print(f"\n{DIVIDER}\n  VERDICTS\n{DIVIDER}")
    for status, msg in verdicts:
        print(f"  [{status}] {msg}")
    n_fail = sum(1 for s, _ in verdicts if s == "FAIL")
    n_warn = sum(1 for s, _ in verdicts if s == "WARN")
    print(DIVIDER)
    if n_fail == 0 and n_warn == 0:
        print("  All checks passed.")
    elif n_fail == 0:
        print(f"  {n_warn} warning(s).")
    else:
        print(f"  {n_fail} FAIL, {n_warn} WARN.")
    print(DIVIDER)


# Output

def save_diagnostics(df_p1, df_p3, df_land, df_grow):
    out = cfg.paths.metrics
    pd.concat([df_p1, df_p3], ignore_index=True).to_csv(
        out / "reward_probes.csv", index=False)
    df_land.to_csv(out / "reward_landscape.csv", index=False)
    df_grow.to_csv(out / "growth_gradient.csv", index=False)


# Main

def reward_pipeline(device) -> tuple:
    """GNN, anchor SMILES, fitted surrogate, and composite reward function."""
    print("Fitting surrogate")
    gnn = trained_gnn(device)
    anchors = active_smiles()
    print(f"  active SMILES: {len(anchors)}")
    sc = cfg.surrogate
    train_smiles = stratified_smiles(sc.n_active, sc.n_inactive, cfg.train.seed)
    print(f"  surrogate training set (stratified): {len(train_smiles)}")
    surrogate = fit_surrogate(gnn, device, train_smiles)
    release_cache(device)
    reward_fn = training_reward(gnn, device, surrogate)
    return gnn, anchors, surrogate, reward_fn


def run_sections(device, rng, quiet: bool) -> tuple:
    """Run all diagnostic sections and return the frames and stats the
    verdicts need. In quiet mode the section tables are hidden from the
    console but still written to CSV."""
    hush = (contextlib.redirect_stdout(io.StringIO()) if quiet
            else contextlib.nullcontext())
    with hush:
        gnn, anchors, surrogate, reward_fn = reward_pipeline(device)
        df_p1, df_p3 = probes_two_phases(reward_fn)
        df_land = landscape_table(reward_fn, SIZE_CENTERS,
                                  LANDSCAPE_SAMPLES, rng)
        print_landscape(df_land)
        df_grow = growth_table(reward_fn, SIZE_CENTERS)
        print_growth(df_grow)
        save_diagnostics(df_p1, df_p3, df_land, df_grow)
        surr_stats = surrogate_section(gnn, surrogate, device, anchors, rng)
        release_cache(device)
        bc_stats = bc_prior_section(gnn, device, anchors, reward_fn)
        p1_stats = phase1_section(gnn, device, reward_fn)
    return df_p1, df_p3, df_land, df_grow, surr_stats, bc_stats, p1_stats


def main():
    ap = argparse.ArgumentParser(description="Pre-RL reward diagnostics")
    ap.add_argument("--quiet", action="store_true",
                    help="print the verdict summary only; the full section tables "
                         "are still written to results/metrics/ as CSV")
    args = ap.parse_args()

    cfg.ensure_dirs()
    device = pick_device()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)

    if args.quiet:
        print(f"diagnose_rewards  device={device}  (quiet, full tables in CSV)")
    else:
        print(f"Device: {device}")

    df_p1, df_p3, df_land, df_grow, surr_stats, bc_stats, p1_stats = \
        run_sections(device, rng, args.quiet)

    print_verdicts(all_verdicts(df_p1, df_p3, df_land, df_grow,
                                surr_stats, bc_stats, p1_stats))
    print(f"\nSaved diagnostics to {cfg.paths.metrics}/")


if __name__ == "__main__":
    main()