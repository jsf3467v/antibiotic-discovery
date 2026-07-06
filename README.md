[![CI](https://github.com/jsf3467v/antibiotic-discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/antibiotic-discovery/actions/workflows/ci.yml)

# Deep Reinforcement Learning with Graph Neural Networks for Antibiotic Discovery

> Final project for the Advanced Machine Learning course, JHU MS in AI program, May 2026.

This project employs machine learning to discover new antibiotic candidates. It combines a model that predicts the bacterial 
inhibition strength of molecules with a reinforcement learning agent that designs molecules to achieve high prediction scores. 
A multi-task GATv2 graph neural network predicts potency, measured as $\log_{10}(\mathrm{MIC})$, for both *S. aureus* and *E. coli* 
based on molecular structure. Meanwhile, a three-phase PPO agent generates novel candidates against these pathogens by using the prediction 
signal through a fingerprint surrogate. It is evaluated against four baseline generators, random construction, genetic algorithms, 
hill climbing, and a SMILES-RNN, which serve as comparison benchmarks.

![Top RL candidates by canonical reward](assets/top_candidates.png)

## Summary

The potency model is trained using 78,314 compound-organism data points from ChEMBL, which are median-aggregated from 112,642 raw MIC 
readings and divided into training, validation, and test sets with an 80/10/10 split based on scaffold, with each scaffold placed in a 
single fold jointly across both organisms so that no scaffold crosses folds between them. Its encoder consists of a three-layer GATv2 
network with organism-specific regression heads, trained using a masked Huber loss function. On the held-out test set, it achieves an 
AUROC of 0.84 for *S. aureus* and 0.86 for *E. coli*. The *E. coli* performance sits just below the noise ceiling estimated from 
replicate measurements, close to what the labels' inherent variability allows.

The generative agent uses a PPO policy on a GATv2 graph, trained with autoregressive heads that decide what to add and where. 
It initially mimics behavior from known antibiotics and then is KL-anchored to this prior, enabling exploration without generating 
nonsensical results. The training follows a three-phase curriculum, starting with broad structural exploration, gradually increasing 
molecule size from 25 to 30 heavy atoms, and finally expanding the best candidates. A surrogate fingerprint network handles inner-loop 
reward calls, keeping the expensive GNN reward for final scoring. The reward combines predicted potency, drug-likeness, synthetic 
accessibility, novelty relative to DrugBank, and resistance evasion against CARD.

The run generated 20,032 unique valid molecules and significantly outperformed the random, hill-climbing, and genetic-algorithm baselines, 
with Bonferroni-corrected p-values below $10^{-16}$ against the first two and $1.2 \times 10^{-4}$ against the genetic algorithm. 
Effect sizes give a more realistic measure, with Cliff's $\delta$ of 0.98 against random, 0.82 against hill climbing, and 0.38 against 
the genetic algorithm. The character-level SMILES-RNN, once its vocabulary was corrected, reached a higher full-distribution reward than 
the agent, with Cliff's $\delta$ of $-0.55$, but it did so through synthetic accessibility rather than potency, and the agent kept the 
higher top-ten reward of 0.57 against 0.52. The genetic algorithm converges on a single Bemis-Murcko scaffold and yields no Lipinski-compliant 
molecules, so the agent now clearly exceeds it on reward as well. On resemblance to real antibiotics the agent leads every method, with a scaffold dominance of 0.01, the lowest Fréchet ChemNet Distance to the active reference at 24.1 against 44 or higher for every other pool, and the closest match to the reference on physicochemical properties. The central result is this separation between aggregate reward, where the SMILES-RNN leads, and resemblance to known antibiotics, where the agent leads.

The honest caveat is that about 92 percent of the generated molecules trigger at least one Brenk structural alert and require medicinal-chemistry cleanup before synthesis. Therefore, the pipeline mainly shows the search process rather than providing a ready-to-synthesize lead compound. A further caveat concerns the surrogate that stands in for the graph network during rollouts. It agrees with the graph network only weakly on the generated pool, with a Pearson correlation of $r = 0.22$ at 56 percent binary agreement, while agreeing more closely on in-distribution chemistry, at $r = 0.57$ with 68 percent agreement. The agent's performance under the full reward therefore rests more on the structural reward terms and the behavior-cloned prior than on surrogate-guided potency. The paper's Limitations section discusses these points in more detail.

## Paper

The full write-up is the [paper (PDF)](https://huggingface.co/jsf3467v/antibiotic-discovery/blob/main/paper.pdf), which holds the tables, figures, methodology, and discussion and is hosted on Hugging Face alongside the trained checkpoints.

## Setup

Tested on macOS (Apple Silicon, MPS) and Linux (CUDA), on Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

The trained checkpoints, about 25 MB in total, can be pulled without retraining.

```bash
hf download jsf3467v/antibiotic-discovery --local-dir models
```

That places the trained checkpoints under `models/`. The four used to reproduce the tables are 
`gnn_best.pt`, `policy_final.pt`, `surrogate.pt`, and `policy_prior.pt`. The four per-phase policy 
snapshots `policy_phase1.pt`, `policy_phase2.pt`, `policy_phase3.pt`, and `policy_phase3_best.pt` 
are included for inspection but are not required. With the first four in place, `evaluate.py`, 
`eval_rl.py`, and `stat_tests.py` regenerate the result tables with no training run.

## Data

Three raw inputs are needed under `Datasets/raw/`. ChEMBL 33 supplies the MIC measurements as a SQLite 
snapshot at `chembl_33/chembl_33_sqlite/chembl_33.db`, available from the 
[EBI ChEMBL downloads page](https://chembl.gitbook.io/chembl-interface-documentation/downloads) 
under CC BY-SA 3.0. DrugBank's full 5.x XML export, free for academic use once an account is created, 
provides the antibiotic reference set as `full database.xml` from [drugbank.com](https://go.drugbank.com/releases/latest). 
The CARD ontology export, free with attribution requested, supplies the resistance substrates as `card.json` from the 
[CARD downloads page](https://card.mcmaster.ca/download). From these the EDA notebook writes processed CSVs into `Datasets/processed/`, 
and because the raw tree runs to about 26 GB the whole `Datasets/` directory is gitignored.

## Reproducing from scratch

Run from the project root in order. Each step caches its output, so re-running one skips work already done, and the wall-clock times are 
for a MacBook Pro M4 Max.

```bash
# 1. Extract and clean data; generates processed CSVs and EDA plots
jupyter notebook EDA/EDA.ipynb

# 2. Train the multi-task GNN regressor              (~2 hours)
python src/train_gnn.py

# 3. GNN test-set metrics (Table 1)                  (<1 min)
python src/evaluate.py

# 4. PPO agent, three phases                         (~6-8 hours)
python src/train_rl.py

# 5. Score the RL pool under the canonical reward    (~5 min)
python src/eval_rl.py

# 6. Train the four baseline generators              (~1 hour)
python src/baselines.py

# 7. Score the baseline pools                        (~5 min)
python src/eval_baselines.py

# 8. Statistical comparison + distributional metrics (~10 min)
python src/stat_tests.py
```

Scores, tables, and metrics are written into `results/` and committed to the repository, so the reported numbers can be inspected 
without rerunning the pipeline. The plot folders `results/plots/` and `EDA/plots/`, along with the checkpoints in `models/`, 
are created on the first run and are not committed. Download the checkpoints from Hugging Face as described under Setup.

## Project layout

```
.
├── config.py                  # All hyperparameters and paths
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── gnn.py                 # Multi-task GATv2 regressor
│   ├── rl.py                  # MDP env, policy, PPO trainer
│   ├── rewards.py             # Composite reward + surrogate
│   ├── feature_engineering.py # Graph featurization
│   ├── train_gnn.py           # GNN training
│   ├── train_rl.py            # PPO training (three phases)
│   ├── baselines.py           # Random, GA, hill-climbing, SMILES-RNN
│   ├── evaluate.py            # GNN test metrics + shared eval utils
│   ├── eval_rl.py             # RL pool evaluation
│   ├── eval_baselines.py      # Baseline pool evaluation
│   └── stat_tests.py          # Mann-Whitney + KL + FCD
├── EDA/
│   ├── EDA.ipynb              # Data extraction + exploratory analysis
│   └── plots/                 # EDA figures; written by the notebook, not tracked
├── assets/                    # PNG figures used in this README
├── Datasets/{raw,processed}/  # Not tracked (~26 GB); fetch from public sources, see Data
├── models/                    # Trained checkpoints; not tracked, download from HF
└── results/                   # Scores, tables, and metrics; committed (plots created on run, not tracked)
    ├── *_scored.csv           #   RL + baseline pools under the canonical reward
    ├── rl_episode_*.csv       #   Per-episode training logs
    └── metrics/               #   Table 1 to 5 metrics + reward diagnostics
```

The three reward-diagnostic files `reward_probes.csv`, `reward_landscape.csv`, and `growth_gradient.csv` in `results/metrics/` come from `src/diagnose_rewards.py`, 
which is not included in the public repository, so they are provided in their current form rather than regenerated by the pipeline steps above.

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

## License

The code is released under the MIT License, and the paper PDF under CC BY 4.0. See `LICENSE` for the code terms.
