[![CI](https://github.com/jsf3467v/antibiotic-discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/antibiotic-discovery/actions/workflows/ci.yml)

# Deep Reinforcement Learning with Graph Neural Networks for Antibiotic Discovery

> Final project for the Advanced Machine Learning course, JHU MS in AI program, May 2026.

This project employs machine learning to discover new antibiotic candidates. It combines a model that predicts the bacterial inhibition strength of molecules with a reinforcement learning agent that designs molecules to achieve high prediction scores. A multi-task GATv2 graph neural network predicts potency, measured as $\log_{10}(\mathrm{MIC})$, for both *S. aureus* and *E. coli* based on molecular structure. A three-phase PPO agent then generates novel candidates against these pathogens by using the prediction signal through a fingerprint surrogate. It is evaluated against four baseline generators, random construction, genetic algorithms, hill climbing, and a SMILES-RNN, which serve as comparison benchmarks.

![Top RL candidates by canonical reward](assets/top_candidates.png)

## Summary

The potency model is a three-layer GATv2 network with organism-specific heads, trained on 78,314 compound-organism measurements from ChEMBL under a masked Huber loss. On a held-out scaffold-split test set it reaches an AUROC of 0.84 for *S. aureus* and 0.86 for *E. coli*, with the *E. coli* result close to the noise ceiling estimated from replicate measurements.

The three-phase PPO agent generates close to 20,000 unique valid molecules per seed. Under the canonical reward it decisively exceeds the random and hill-climbing baselines, with Bonferroni-corrected $p$ values below $10^{-16}$. The genetic algorithm and the SMILES-RNN post higher pool-wide reward, but reach it only through low structural diversity, the genetic algorithm collapsing onto a single scaffold and the SMILES-RNN producing small and unstable pools. Judged on the properties that matter for a candidate set, which are structural diversity, drug-likeness, and similarity to known active antibiotics, the agent leads every method. It carries the highest scaffold and internal diversity, complete novelty against the DrugBank antibiotic reference, and the closest distributional match to active antibiotics by Frechet ChemNet Distance.

The pool metrics are stable across three random seeds, which is the main evidence that the behavior is not an artifact of a single run. Seed 42 is used as the representative run for the figures, and the paper holds the full per-method tables.

| Metric | Seed 42 | Seed 43 | Seed 44 |
| --- | --- | --- | --- |
| Unique molecules | 20,021 | 20,030 | 20,019 |
| Pool reward mean | 0.254 | 0.255 | 0.255 |
| Top-100 reward mean | 0.480 | 0.483 | 0.481 |
| Scaffold diversity | 0.607 | 0.675 | 0.605 |
| Internal diversity | 0.917 | 0.919 | 0.917 |
| Lipinski pass rate | 0.788 | 0.789 | 0.795 |
| Novelty against DrugBank | 1.000 | 1.000 | 1.000 |
| Frechet ChemNet Distance | 24.5 | 23.6 | 24.3 |
| Property KL divergence | 1.11 | 1.05 | 1.14 |

## Limitations

Several caveats should be considered. Approximately 88 percent of the generated molecules activate at least one Brenk structural alert, and the highest-scoring molecules tend to be large and to contain reactive groups, so a structural-alert and drug-likeness filter is recommended before any molecule is selected for synthesis or docking. Nearly 30 percent of the pool consists of acyclic molecules, a shape that known antibiotics rarely take, which is a further reason to filter before selection. The surrogate model that stands in for the graph network during sampling shows only weak agreement with it on the generated pool, with a Pearson correlation of $r = 0.18$ and 67 percent agreement on binary active-class calls. This indicates that the reward behavior of the agent depends more on the structural reward terms and the behavior-cloned prior than on surrogate-guided potency. The generation results come from three random seeds for the agent and the four baseline generators, so run-to-run variability of the generation is measured, while the graph network is trained once on a single scaffold split, so variability from the model and from the split remains unmeasured. The reward also applies an applicability-domain factor that reduces potency credit for molecules far from the training chemistry. In this run the factor was active, flooring at 0.25 for the most distant molecules and rising to 1.0 for those closest to the training set, so off-distribution potency was gated rather than left unchecked. The high Brenk-alert and acyclic fractions noted above show that structural quality still needs filtering despite this. The Limitations section of the paper discusses these points further.

## Paper

The full write-up is the [paper (PDF)](https://huggingface.co/jsf3467v/antibiotic-discovery/blob/main/paper.pdf), which holds the tables, figures, methodology, and discussion and is hosted on Hugging Face alongside the trained checkpoints.

## Setup

The pipeline was developed on macOS with Apple Silicon and on Linux, and it runs on Python 3.10 or newer. Every step runs on the CPU by default, and GPU acceleration through CUDA or Apple MPS can be enabled by setting the `PROJECT_DEVICE` variable.

```bash
pip install -r requirements.txt
```

The trained checkpoints can be pulled without retraining.

```bash
hf download jsf3467v/antibiotic-discovery --local-dir models
```

The graph network, the surrogate, and the behavior-cloned prior are seed-invariant and live under `models/` as `gnn_best.pt`, `surrogate.pt`, and `policy_prior.pt`. The final policy is produced once per seed and lives under `runs/seed{N}/` as `policy_final.pt`, alongside the per-phase snapshots `policy_phase1.pt` through `policy_phase3_best.pt`, which are kept for inspection but are not required. With the three global checkpoints in place and a seed's policy present, `eval_rl.py` and `stat_tests.py` regenerate that seed's result tables with no training run, and `evaluate.py` regenerates the graph network metrics for Table 1.

## Data

Three raw inputs are needed under `Datasets/raw/`. ChEMBL 33 supplies the MIC measurements as a SQLite snapshot at `chembl_33/chembl_33_sqlite/chembl_33.db`, available from the [EBI ChEMBL downloads page](https://chembl.gitbook.io/chembl-interface-documentation/downloads) under CC BY-SA 3.0. DrugBank's full 5.x XML export, free for academic use once an account is created, provides the antibiotic reference set as `full database.xml` from [drugbank.com](https://go.drugbank.com/releases/latest). The CARD ontology export, free with attribution requested, supplies the resistance substrates as `card.json` from the [CARD downloads page](https://card.mcmaster.ca/download). From these the EDA notebook writes processed CSVs into `Datasets/processed/`, and because the raw tree runs to about 26 GB the whole `Datasets/` directory is gitignored.

## Reproducing from scratch

Run from the project root in order. Each step caches its output, so re-running one skips work already done, and the wall-clock times are for a MacBook Pro M4 Max running on the CPU.

```bash
# 1. Extract and clean the data, then write processed CSVs and EDA plots
jupyter notebook EDA/EDA.ipynb

# 2. Train the multi-task graph network regressor                 (about 2 hours)
python src/train_gnn.py

# 3. Report the graph network test-set metrics for Table 1        (under 1 minute)
python src/evaluate.py

# 4. Run the reinforcement learning pipeline for all three seeds  (about 45 minutes
bash run.sh --fresh                                             # of agent training
                                                                # per seed, near
                                                                # 3 hours in total
```

The `run.sh` script drives the reinforcement learning pipeline across seeds 42, 43, and 44. For each seed it trains the agent, scores the pool under the canonical reward, trains and scores the four baseline generators, runs the statistical comparison, checks surrogate agreement, and summarizes the training dynamics. It runs a reward sanity check before the seeds and a weight-recovery check after each seed, and it writes a cross-seed summary into `results/summary/`. A single seed can be run with `bash run.sh --fresh 42`, and a run that stops partway can be resumed by omitting `--fresh`, since the agent training continues from its last checkpoint.

Scores, tables, and metrics are written under `runs/seed{N}/` for each seed and summarized under `results/summary/`, so the reported numbers can be inspected without rerunning the pipeline. The plot folders and the checkpoints in `models/` are created on the first run and are not committed. Download the checkpoints from Hugging Face as described under Setup.

## Project layout

```
.
├── config.py                  # All hyperparameters and paths
├── run.sh                     # Multi-seed pipeline driver
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── gnn.py                 # Multi-task GATv2 regressor
│   ├── rl.py                  # Environment, policy, PPO trainer
│   ├── rewards.py             # Composite reward and surrogate
│   ├── feature_engineering.py # Graph featurization
│   ├── train_gnn.py           # Graph network training
│   ├── train_rl.py            # PPO training across three phases
│   ├── baselines.py           # Random, genetic algorithm, hill climbing, SMILES-RNN
│   ├── evaluate.py            # Graph network test metrics and shared evaluation helpers
│   ├── eval_rl.py             # Reinforcement learning pool evaluation
│   ├── eval_baselines.py      # Baseline pool evaluation
│   ├── stat_tests.py          # Rank tests, property divergence, and Frechet distance
│   ├── agreement.py           # Surrogate against graph network agreement on the pool
│   ├── dynamics.py            # Post-hoc training dynamics across phases
│   ├── diagnose_rewards.py    # Reward sanity checks run before training
│   ├── gate_check.py          # Reward weight recovery and gate check run after training
│   └── collect.py             # Cross-seed aggregation of the metric tables
├── EDA/
│   ├── EDA.ipynb              # Data extraction and exploratory analysis
│   └── plots/                 # Exploratory figures, written by the notebook, not tracked
├── assets/                    # Figures used in this README
├── Datasets/{raw,processed}/  # Not tracked, about 26 GB, fetch from public sources, see Data
├── models/                    # Seed-invariant checkpoints, not tracked, download from Hugging Face
├── runs/                      # Per-seed outputs, created on run, not tracked
│   └── seed{42,43,44}/
│       ├── generated_molecules.csv    # Sampled molecules for this seed
│       ├── rl_pool_scored.csv         # Pool under the canonical reward
│       ├── rl_episode_log.csv         # Per-episode training log
│       ├── policy_final.pt            # Best phase-3 policy for this seed
│       └── metrics/                   # Table 2 through Table 5 metrics for this seed
└── results/
    ├── metrics/               # Seed-invariant reward diagnostics
    └── summary/               # Cross-seed mean and range for every table
```

The reward-diagnostic files `reward_probes.csv`, `reward_landscape.csv`, and `growth_gradient.csv` in `results/metrics/` come from `src/diagnose_rewards.py`, which `run.sh` runs as a pre-flight check, so they are regenerated by the pipeline rather than provided in fixed form.

## Citation

```
@misc{keith2026antibiotic,
  author       = {Keith, Arlene},
  title        = {Deep Reinforcement Learning with Graph Neural Networks for Antibiotic Discovery},
  year         = {2026},
  howpublished = {Final project, Advanced Machine Learning course, JHU MS in AI program},
  url          = {https://github.com/jsf3467v/antibiotic-discovery}
}
```
