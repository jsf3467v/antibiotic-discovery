"""Rebuild rl_episode_props.csv from the episode log. Derived data only, so no
retraining and no GNN. Composition matches src.rewards through config."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors, Lipinski, RDConfig

from config import ProjectConfig

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

cfg = ProjectConfig()
ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = list(cfg.composition.reference.keys())
SYMBOL_INDEX = {s: i for i, s in enumerate(SYMBOLS)}
REF = np.array([cfg.composition.reference[s] for s in SYMBOLS], dtype=np.float32)
COLUMNS = ["canonical", "n_atoms", "qed", "sa", "lipinski", "composition"]


def fractions(mol) -> np.ndarray:
    """Per-symbol atom fraction over heavy atoms."""
    counts = np.zeros(len(SYMBOLS), dtype=np.float32)
    for atom in mol.GetAtoms():
        k = SYMBOL_INDEX.get(atom.GetSymbol())
        if k is not None:
            counts[k] += 1.0
    return counts / np.float32(mol.GetNumHeavyAtoms())


def lipinski_hits(mol) -> int:
    """Rule-of-five criteria met, out of four."""
    return (int(Descriptors.MolWt(mol) <= 500)
            + int(Descriptors.MolLogP(mol) <= 5)
            + int(Lipinski.NumHDonors(mol) <= 5)
            + int(Lipinski.NumHAcceptors(mol) <= 10))


def row(smiles: str):
    """One props row, or None when the SMILES will not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    sq = float(np.sum((fractions(mol) - REF) ** 2))
    tau = cfg.composition.tau
    return (Chem.MolToSmiles(mol),
            mol.GetNumHeavyAtoms(),
            float(QED.qed(mol)),
            (10.0 - sascorer.calculateScore(mol)) / 9.0,
            int(lipinski_hits(mol) >= 3),
            1.0 if tau <= 0 else float(np.exp(-sq / tau ** 2)))


def frame(smiles: list, workers: int) -> pd.DataFrame:
    """Props for every log row, parsed in parallel."""
    if workers <= 1 or len(smiles) < 500:
        rows = [row(s) for s in smiles]
    else:
        with Pool(workers) as pool:
            rows = pool.map(row, smiles, chunksize=256)
    return pd.DataFrame([r for r in rows if r is not None], columns=COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--seed", type=int, default=cfg.rl.seed)
    ap.add_argument("--workers", type=int, default=cfg.train.num_workers)
    args = ap.parse_args()

    run = args.root / "runs" / f"seed{args.seed}"
    src = run / "rl_episode_log.csv"
    dst = run / "rl_episode_props.csv"
    print(f"seed: {args.seed}")
    log = pd.read_csv(src)
    out = frame(log["smiles"].astype(str).tolist(), args.workers)
    out.to_csv(dst, index=False)
    covered = log["smiles"].astype(str).isin(set(out["canonical"])).mean()
    print(f"{len(out)} rows from {len(log)} log rows, "
          f"{covered:.1%} of log SMILES already canonical")


if __name__ == "__main__":
    main()