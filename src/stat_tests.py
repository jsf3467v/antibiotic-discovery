"""
Statistical comparison of the RL pool against the four baseline
pools, distributional metrics versus the active-antibiotics
training set, and training-dynamics validation joined to the
canonical scored cache.
"""

import sys
sys.path.insert(0,
    str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore", message=".*MorganGenerator.*")

from rdkit import RDLogger, rdBase
RDLogger.DisableLog('rdApp.*')
rdBase.DisableLog('rdApp.*')

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import (Crippen, Descriptors, FilterCatalog, Lipinski,
                        QED, rdMolDescriptors)
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import mannwhitneyu

from config import ProjectConfig
from src.rewards import composition_arrays, composition_penalty, sa_score
from src.train_rl import active_smiles

cfg = ProjectConfig()

ALPHA = 0.05
TOP_K = 10
BOOT_N = 1000
BOOT_CI = 0.95
PROPERTY_BINS = 30
TRAINING_QUINTILES = 5
LIPINSKI_RULES_PASS = 3
MW_LIMIT = 500.0
LOGP_LIMIT = 5.0
HBD_LIMIT = 5
HBA_LIMIT = 10
BASELINES = ("random", "genetic_algorithm", "hill_climbing", "smiles_rnn")
PROPERTIES = ("mw", "logp", "hba", "hbd", "tpsa", "rotbonds")
FCD_DEVICE = "cpu"


def scored_csv(name: str) -> Path:
    return cfg.paths.results / f"baseline_{name}_scored.csv"


def rl_scored_csv() -> Path:
    return cfg.paths.results / "rl_pool_scored.csv"


def episode_log_path() -> Path:
    return cfg.paths.results / "rl_episode_log.csv"


def props_cache_path() -> Path:
    return cfg.paths.results / "rl_episode_props.csv"


def cached_pool(scored_path: Path
                ) -> Optional[Tuple[List[str], np.ndarray]]:
    if not scored_path.exists():
        return None
    df = pd.read_csv(scored_path)
    return (df["smiles"].astype(str).tolist(),
            df["score"].to_numpy(dtype=np.float64))


def cliffs_delta(u_stat: float, n_a: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return 0.0
    return 2.0 * u_stat / (n_a * n_b) - 1.0


def mannwhitney(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if a.size == 0 or b.size == 0:
        return 0.0, 1.0
    result = mannwhitneyu(a, b, alternative='greater')
    return float(result.statistic), float(result.pvalue)


def bootstrap_topk_mean(scores: np.ndarray, k: int = TOP_K,
                        n_boot: int = BOOT_N, ci: float = BOOT_CI,
                        seed: int = 0) -> Tuple[float, float, float]:
    if scores.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = scores.size
    samples = rng.choice(scores, size=(n_boot, n), replace=True)
    samples.sort(axis=1)
    kk = min(k, n)
    boots = samples[:, -kk:].mean(axis=1)
    lo, hi = np.percentile(boots,
                           [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(boots.mean()), float(lo), float(hi)


def molecular_properties(smi: str) -> Optional[dict]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {"mw": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "hbd": Lipinski.NumHDonors(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "rotbonds": Lipinski.NumRotatableBonds(mol)}


def property_array(smiles_list: List[str]) -> dict:
    cols = {p: [] for p in PROPERTIES}
    for s in smiles_list:
        props = molecular_properties(s)
        if props is None:
            continue
        for p in PROPERTIES:
            cols[p].append(props[p])
    return {p: np.array(cols[p], dtype=np.float64)
            for p in PROPERTIES}


def kl_two_histograms(pool: np.ndarray, reference: np.ndarray,
                      bins: int = PROPERTY_BINS) -> float:
    if pool.size == 0 or reference.size == 0:
        return 0.0
    lo = float(min(pool.min(), reference.min()))
    hi = float(max(pool.max(), reference.max()))
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(pool, bins=edges)
    q, _ = np.histogram(reference, bins=edges)
    p = (p + 1) / (p.sum() + bins)
    q = (q + 1) / (q.sum() + bins)
    return float((p * np.log(p / q)).sum())


def property_kl(pool_props: dict, reference_props: dict) -> float:
    return sum(kl_two_histograms(pool_props[p], reference_props[p])
               for p in PROPERTIES)


def scaffold_smiles(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def largest_scaffold_fraction(smiles_list: List[str]) -> float:
    scafs = [s for s in (scaffold_smiles(smi) for smi in smiles_list)
             if s is not None]
    if not scafs:
        return 0.0
    counts = pd.Series(scafs).value_counts()
    return float(counts.iloc[0]) / len(scafs)


def filter_pass_rate(smiles_list: List[str], catalog_name: str
                     ) -> float:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(getattr(
        FilterCatalog.FilterCatalogParams.FilterCatalogs, catalog_name))
    catalog = FilterCatalog.FilterCatalog(params)
    valid = passed = 0
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        valid += 1
        if not catalog.HasMatch(mol):
            passed += 1
    return passed / valid if valid > 0 else 0.0


def fcd_distance(pool_smiles: List[str],
                 reference_smiles: List[str], device: str) -> float:
    try:
        from fcd_torch import FCD
    except ImportError:
        return float('nan')
    try:
        fcd = FCD(device=device, n_jobs=1)
        return float(fcd(reference_smiles, pool_smiles))
    except Exception as exc:
        print(f"  FCD failed: {exc}")
        return float('nan')


def comparison_row(name: str, rl_scores: np.ndarray,
                   base_scores: np.ndarray, n_tests: int) -> dict:
    u_stat, p_raw = mannwhitney(rl_scores, base_scores)
    p_bonf = min(1.0, p_raw * n_tests)
    delta = cliffs_delta(u_stat, rl_scores.size, base_scores.size)
    rl_mean, rl_lo, rl_hi = bootstrap_topk_mean(rl_scores)
    bs_mean, bs_lo, bs_hi = bootstrap_topk_mean(base_scores)
    return {"baseline": name,
            "mwu_p_raw": p_raw, "mwu_p_bonf": p_bonf,
            "cliffs_delta": delta,
            "rl_top10_mean": rl_mean,
            "rl_top10_lo": rl_lo, "rl_top10_hi": rl_hi,
            "base_top10_mean": bs_mean,
            "base_top10_lo": bs_lo, "base_top10_hi": bs_hi}


def distribution_row(name: str, smiles_list: List[str],
                     reference_props: dict,
                     reference_smiles: List[str]) -> dict:
    pool_props = property_array(smiles_list)
    return {"pool": name,
            "property_kl": property_kl(pool_props, reference_props),
            "fcd": fcd_distance(smiles_list, reference_smiles,
                                FCD_DEVICE),
            "scaffold_dominance": largest_scaffold_fraction(smiles_list),
            "pains_pass": filter_pass_rate(smiles_list, "PAINS"),
            "brenk_pass": filter_pass_rate(smiles_list, "BRENK")}


def molecule_props(smi, ref, symbols, tau):
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    rules = ((Descriptors.MolWt(mol) <= MW_LIMIT)
             + (Descriptors.MolLogP(mol) <= LOGP_LIMIT)
             + (Lipinski.NumHDonors(mol) <= HBD_LIMIT)
             + (Lipinski.NumHAcceptors(mol) <= HBA_LIMIT))
    return {"canonical": Chem.MolToSmiles(mol),
            "n_atoms": mol.GetNumHeavyAtoms(),
            "qed": float(QED.qed(mol)),
            "sa": sa_score(mol),
            "lipinski": int(rules >= LIPINSKI_RULES_PASS),
            "composition": composition_penalty(mol, ref, symbols, tau)}


def episode_props(df_log: pd.DataFrame) -> pd.DataFrame:
    """Per-episode properties row-aligned to df_log."""
    symbols, ref, tau = composition_arrays(cfg.composition)
    nan_row = {"canonical": None, "n_atoms": np.nan, "qed": np.nan,
               "sa": np.nan, "lipinski": np.nan, "composition": np.nan}
    rows = [molecule_props(s, ref, symbols, tau) or nan_row
            for s in df_log["smiles"].astype(str)]
    return pd.DataFrame(rows)


def cached_episode_props(log_path: Path) -> Optional[pd.DataFrame]:
    cache = props_cache_path()
    if not cache.exists():
        return None
    if cache.stat().st_mtime < log_path.stat().st_mtime:
        return None
    return pd.read_csv(cache)


def joined_episodes(df_log, df_props, df_pool):
    side = pd.concat([df_log[["episode", "phase"]].reset_index(drop=True),
                      df_props.reset_index(drop=True)], axis=1)
    pool = df_pool[["smiles", "score"]].rename(columns={"smiles": "canonical"})
    return side.merge(pool, on="canonical", how="inner").reset_index(drop=True)


def phase_table(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("phase", as_index=False)
              .agg(n=("score", "size"),
                   reward=("score", "mean"),
                   qed=("qed", "mean"),
                   sa=("sa", "mean"),
                   lipinski=("lipinski", "mean"),
                   size=("n_atoms", "mean"),
                   composition=("composition", "mean")))


def phase_quintile_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.sort_values(["phase", "episode"]).copy()
    work["pos"] = work.groupby("phase").cumcount()
    work["quintile"] = (work.groupby("phase")["pos"]
                            .transform(lambda s: pd.qcut(
                                s, TRAINING_QUINTILES,
                                labels=False, duplicates="drop")))
    return (work.groupby(["phase", "quintile"], as_index=False)
                .agg(n=("score", "size"),
                     reward=("score", "mean"),
                     qed=("qed", "mean"),
                     sa=("sa", "mean"),
                     lipinski=("lipinski", "mean"),
                     size=("n_atoms", "mean"),
                     composition=("composition", "mean")))


def training_dynamics(df_pool: pd.DataFrame):
    log_path = episode_log_path()
    if not log_path.exists():
        print(f"\nMissing {log_path}, skipping training dynamics")
        return None, None
    df_log = pd.read_csv(log_path)
    cached = cached_episode_props(log_path)
    if cached is None:
        df_props = episode_props(df_log)
        df_props.to_csv(props_cache_path(), index=False)
    else:
        df_props = cached
    joined = joined_episodes(df_log, df_props, df_pool)
    return phase_table(joined), phase_quintile_table(joined)


def print_significance(rows: List[dict]):
    print("\nSignificance (RL > baseline, one-sided MWU):")
    print(f"  {'baseline':<20} {'p_raw':>10} {'p_bonf':>10} "
          f"{'cliff_d':>9} {'rl_top10':>10} {'base_top10':>12}")
    for r in rows:
        print(f"  {r['baseline']:<20} {r['mwu_p_raw']:>10.2e} "
              f"{r['mwu_p_bonf']:>10.2e} {r['cliffs_delta']:>9.3f} "
              f"{r['rl_top10_mean']:>10.4f} "
              f"{r['base_top10_mean']:>12.4f}")


def print_distribution(rows: List[dict]):
    print("\nDistribution metrics - vs active antibiotics reference:")
    print(f"  {'pool':<20} {'kl':>7} {'fcd':>8} {'scaff_dom':>10} "
          f"{'pains':>7} {'brenk':>7}")
    for r in rows:
        fcd_str = (f"{r['fcd']:>8.3f}" if not np.isnan(r['fcd'])
                   else f"{'n/a':>8}")
        print(f"  {r['pool']:<20} {r['property_kl']:>7.3f} {fcd_str} "
              f"{r['scaffold_dominance']:>10.4f} "
              f"{r['pains_pass']:>7.3f} {r['brenk_pass']:>7.3f}")


def print_phase_table(df: pd.DataFrame):
    print("\nDynamics by phase - canonical reward:")
    print(f"  {'phase':>5} {'n':>6} {'reward':>7} {'qed':>6} {'sa':>6} "
          f"{'lipinski':>9} {'size':>6} {'comp':>6}")
    for _, r in df.iterrows():
        print(f"  {int(r['phase']):>5d} {int(r['n']):>6,} "
              f"{r['reward']:>7.3f} {r['qed']:>6.3f} {r['sa']:>6.3f} "
              f"{r['lipinski']:>9.3f} {r['size']:>6.1f} "
              f"{r['composition']:>6.3f}")


def print_quintile_table(df: pd.DataFrame):
    print("\nWithin-phase quintile trajectories:")
    print(f"  {'phase':>5} {'q':>2} {'n':>6} {'reward':>7} {'qed':>6} "
          f"{'sa':>6} {'lipinski':>9} {'size':>6} {'comp':>6}")
    for _, r in df.iterrows():
        print(f"  {int(r['phase']):>5d} {int(r['quintile']):>2d} "
              f"{int(r['n']):>6,} {r['reward']:>7.3f} {r['qed']:>6.3f} "
              f"{r['sa']:>6.3f} {r['lipinski']:>9.3f} "
              f"{r['size']:>6.1f} {r['composition']:>6.3f}")


def write_csvs(sig_rows, dist_rows, dynamics_df):
    out = cfg.paths.metrics
    pd.DataFrame(sig_rows).to_csv(
        out / "stat_tests_significance.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(
        out / "stat_tests_distribution.csv", index=False)
    if dynamics_df is not None:
        dynamics_df.to_csv(
            out / "stat_tests_training_dynamics.csv", index=False)


def cached_pools() -> Optional[Tuple[List[str], np.ndarray, dict]]:
    rl = cached_pool(rl_scored_csv())
    if rl is None:
        print(f"Missing {rl_scored_csv()}. Run eval_rl.py first.")
        return None
    pools = {}
    for name in BASELINES:
        hit = cached_pool(scored_csv(name))
        if hit is None:
            print(f"Missing {scored_csv(name)}. "
                  "Run eval_baselines.py first.")
            return None
        pools[name] = hit
    return rl[0], rl[1], pools


def main():
    cfg.ensure_dirs()
    np.random.seed(cfg.train.seed)
    loaded = cached_pools()
    if loaded is None:
        return
    rl_smi, rl_scores, pools = loaded
    print(f"RL: {len(rl_smi):,} mols")
    for name, (smi, _) in pools.items():
        print(f"  {name}: {len(smi):,} mols")
    sig_rows = [comparison_row(n, rl_scores, sc, len(BASELINES))
                for n, (_, sc) in pools.items()]
    reference_smi = active_smiles()
    print(f"\nReference: {len(reference_smi):,} active antibiotics")
    reference_props = property_array(reference_smi)
    pool_specs = [("rl", rl_smi)] + [(n, smi) for n, (smi, _)
                                     in pools.items()]
    dist_rows = [distribution_row(n, smi, reference_props,
                                  reference_smi)
                 for n, smi in pool_specs]
    df_pool = pd.DataFrame({"smiles": rl_smi, "score": rl_scores})
    phase_df, quintile_df = training_dynamics(df_pool)
    print_significance(sig_rows)
    print_distribution(dist_rows)
    if phase_df is not None:
        print_phase_table(phase_df)
        print_quintile_table(quintile_df)
    write_csvs(sig_rows, dist_rows, quintile_df)
    print(f"\nSaved to {cfg.paths.metrics}")


if __name__ == "__main__":
    main()