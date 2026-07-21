[![CI](https://github.com/jsf3467v/antibiotic-discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/antibiotic-discovery/actions/workflows/ci.yml)

# Deep Reinforcement Learning with Graph Neural Networks for Antibiotic Discovery

> Final project for the Advanced Machine Learning course, JHU MS in AI program, May 2026.

This project employs machine learning to discover new antibiotic candidates. It combines a model that predicts the bacterial inhibition strength of molecules with a reinforcement learning agent that designs molecules to achieve high prediction scores. A multi-task GATv2 graph neural network predicts potency, measured as $\log_{10}(\mathrm{MIC})$, for both *S. aureus* and *E. coli* based on molecular structure. A three-phase PPO agent then generates novel candidates against these pathogens by using the prediction signal through a fingerprint surrogate. It is evaluated against four baseline generators, random construction, genetic algorithms, hill climbing, and a SMILES-RNN, which serve as comparison benchmarks.

![Top RL candidates by canonical reward](assets/top_candidates.png)

## Summary

The potency model is a three-layer GATv2 network with organism-specific heads, trained on 78,314 compound-organism measurements from ChEMBL using a masked Huber loss. It achieves an AUROC of 0.84 for *S. aureus* and 0.86 for *E. coli* on a hold-out scaffold-split test set, with the *E. coli* score nearing the estimated noise ceiling from replicate measurements.

The three-phase PPO agent generates nearly 20,000 unique valid molecules per seed. Under the standard reward, it significantly outperforms random and hill-climbing baselines, with Bonferroni-corrected $p$ values below $10^{-16}$. Although the genetic algorithm and the SMILES-RNN produce higher pooled rewards, they do so with low structural diversity— the genetic algorithm collapsing onto a single scaffold and the SMILES-RNN generating small, unstable pools. In terms of key properties like structural diversity, novelty, and similarity to known antibiotics, the agent surpasses all other methods. It exhibits the highest scaffold and internal diversity, is completely novel compared to DrugBank antibiotics, and has the closest distributional match to active antibiotics based on Frechet ChemNet Distance. Drug-likeness results are mixed: while the top candidates score well on QED, the overall Lipinski pass rate is below baseline levels, indicating the need for a drug-likeness filter before selection.

The pool metrics remain consistent across three random seeds, supporting that these results are not due to a single run. Seed 42 is used as the representative in figures, and the full per-method tables are included in the paper.


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

Several caveats should be considered. About 88% of the generated molecules activate at least one Brenk structural alert, and the top-scoring molecules tend to be large and contain reactive groups. Therefore, applying a structural-alert and drug-likeness filter is advised before selecting molecules for synthesis or docking. Nearly 30% of the pool are acyclic molecules, a shape rarely seen in known antibiotics, which further underscores the need for filtering before selection. The surrogate model used for sampling shows only weak agreement with the graph network, with a Pearson correlation of $r = 0.18$ and 67% agreement on binary active-class calls. This suggests that the agent's reward behavior relies more on the structural reward terms and behavior-cloned prior than on surrogate-guided potency. Results were based on three random seeds for the agent and four baseline generators, capturing run-to-run variability, while the graph network was trained once on a single scaffold split, leaving variability from the model and split unmeasured. An applicability-domain factor was included in the reward to reduce potency credit for molecules far from training chemistry. In this run, the factor was active, with a floor at 0.25 for the most distant molecules and rising to 1.0 for those closest to the training set, controlling off-distribution potency. The high fractions of Brenk-alerts and acyclic molecules highlight the continued need for filtering despite these measures. The paper's Limitations section discusses these issues further.

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

# 5. Verify reward integrity, reading existing artifacts only    (seconds)
bash verify.sh
```

The `run.sh` script drives the reinforcement learning pipeline across seeds 42, 43, and 44. For each seed it trains the agent, scores the pool under the canonical reward, trains and scores the four baseline generators, runs the statistical comparison, checks surrogate agreement, and summarizes the training dynamics. It then writes a cross-seed summary into `results/summary/`. A single seed can be run with `bash run.sh --fresh 42`, and a run that stops partway can be resumed by omitting `--fresh`, since the agent training continues from its last checkpoint.

The `verify.sh` script runs the reward-integrity checks after `run.sh` finishes, and it only reads existing artifacts rather than scoring new molecules with the graph network. It runs `diagnose_rewards` once at the start to write the seed-invariant probe files, then runs `gate_check` for each seed against that seed's run directory. A single seed can be checked with `bash verify.sh 42`.

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

The reward-diagnostic files `reward_probes.csv`, `reward_landscape.csv`, and `growth_gradient.csv` in `results/metrics/` come from `src/diagnose_rewards.py`, which `verify.sh` runs before the per-seed checks, so they are regenerated by the pipeline rather than provided in fixed form.

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
