[![CI](https://github.com/jsf3467v/antibiotic-discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/antibiotic-discovery/actions/workflows/ci.yml)

# Deep Reinforcement Learning with Graph Neural Networks for Antibiotic Discovery

> Final project for the Advanced Machine Learning course, JHU MS in AI program, May 2026.

This project employs machine learning to discover new antibiotic candidates. It combines a model that predicts the bacterial inhibition strength of molecules with a reinforcement learning agent that designs molecules to achieve high prediction scores. A multi-task GATv2 graph neural network predicts potency, measured as $\log_{10}(\mathrm{MIC})$, for both *S. aureus* and *E. coli* based on molecular structure. Meanwhile, a three-phase PPO agent generates novel candidates against these pathogens by using the prediction signal through a fingerprint surrogate. It is evaluated against four baseline generators, random construction, genetic algorithms, hill climbing, and a SMILES-RNN, which serve as comparison benchmarks.

![Top RL candidates by canonical reward](assets/top_candidates.png)

## Summary

The potency model is trained using 78,314 compound-organism data points from ChEMBL, which are median-aggregated from 112,642 raw MIC readings and divided into training, validation, and test sets with an 80/10/10 split based on scaffold. Its encoder consists of a three-layer GATv2 network with organism-specific regression heads, trained using a masked Huber loss function. On the held-out test set, it achieves an AUROC of 0.83 for *S. aureus* and 0.87 for *E. coli*. The *E. coli* performance is near the noise ceiling estimated from replicate measurements, indicating it performs as well as the labels' inherent variability allows.

The generative agent uses a PPO policy on a GATv2 graph, trained with autoregressive heads that decide what to add and where. It initially mimics behavior from known antibiotics and then is KL-anchored to this prior, enabling exploration without generating nonsensical results. The training follows a three-phase curriculum, starting with broad structural exploration, gradually increasing molecule size from 25 to 30 heavy atoms, and finally expanding the best candidates. A surrogate fingerprint network handles inner-loop reward calls, keeping the expensive GNN reward for final scoring. The reward combines predicted potency, drug-likeness, synthetic accessibility, novelty relative to DrugBank, and resistance evasion against CARD.

The run generated 20,031 unique valid molecules and significantly outperformed the random, hill-climbing, and SMILES-RNN baselines with a p-value less than $10^{-16}$ after Bonferroni correction. Effect sizes provide a more realistic measure, with Cliff's $\delta$ of 0.97 and 0.73 compared to the first two baselines, but only 0.05 against the SMILES-RNN, which is statistically significant yet practically negligible. The genetic algorithm achieves higher raw top-ten rewards but tends to converge on a single Bemis-Murcko scaffold, while the RL pool remains entirely diverse, with a scaffold dominance of 0.003. It also attains the lowest Fréchet ChemNet Distance to the active reference among all tested methods, measuring 26.1 versus 43.8 or higher elsewhere.

The honest caveat is that about 95 percent of the generated molecules trigger at least one Brenk structural alert and require medicinal-chemistry cleanup before synthesis. Therefore, the pipeline mainly shows the search process rather than providing a ready-to-synthesize lead compound. The paper's Limitations section also discusses other issues, such as the soft cross-task scaffold leakage and the surrogate-to-GNN agreement of $r = 0.52$ at 63 percent binary agreement.

## Paper

The full write-up is the [paper (PDF)](https://huggingface.co/jsf3467v/antibiotic-discovery/blob/main/paper.pdf), which holds the tables, figures, methodology, and discussion and is hosted on Hugging Face alongside the trained checkpoints.

## Setup

Tested on macOS (Apple Silicon, MPS) and Linux (CUDA), on Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

The trained checkpoints, about 30 MB in total, can be pulled without retraining.

```bash
hf download jsf3467v/antibiotic-discovery --local-dir models
```

That places `gnn_best.pt`, `policy_final.pt`, `surrogate.pt`, and `policy_prior.pt` under `models/`, after which `evaluate.py`, `eval_rl.py`, and `stat_tests.py` reproduce the paper's tables with no training run.

## Data

Three raw inputs are needed under `Datasets/raw/`. ChEMBL 33 supplies the MIC measurements as a SQLite snapshot at `chembl_33/chembl_33_sqlite/chembl_33.db`, available from the [EBI ChEMBL downloads page](https://chembl.gitbook.io/chembl-interface-documentation/downloads) under CC BY-SA 3.0. DrugBank's full 5.x XML export, free for academic use once an account is created, provides the antibiotic reference set as `full database.xml` from [drugbank.com](https://go.drugbank.com/releases/latest). The CARD ontology export, free with attribution requested, supplies the resistance substrates as `card.json` from the [CARD downloads page](https://card.mcmaster.ca/download). From these the EDA notebook writes processed CSVs into `Datasets/processed/`, and because the raw tree runs to about 26 GB the whole `Datasets/` directory is gitignored.

## Reproducing from scratch

Run from the project root in order. Each step caches its output, so re-running one skips work already done, and the wall-clock times are for a MacBook Pro M4 Max.

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

Outputs land in `results/metrics/` and `results/plots/`, and checkpoints land in `models/`. These directories are created on first run and are not part of the repository.

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
└── results/{metrics,plots}/   # Created when the pipeline runs; not tracked
```

The `results/` tree and `EDA/plots/` are not committed; both are produced by the steps in **Reproducing from scratch** (the EDA notebook writes its figures into `EDA/plots/`, and the pipeline writes scores and tables into `results/`). The trained checkpoints under `models/` regenerate the result tables without retraining, and the paper PDF on Hugging Face holds the final tables and figures.

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
