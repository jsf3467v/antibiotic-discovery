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

The potency model is developed with 78,314 compound-organism data points from ChEMBL, derived from 112,642 raw MIC readings, and split into training, 
validation, and test sets in an 80/10/10 ratio based on scaffold. Each scaffold is assigned to a single fold across both organisms to prevent crossing. 
Its encoder features a three-layer GATv2 network with organism-specific regression heads, trained using a masked Huber loss. On the test set, the model 
achieves an AUROC of 0.84 for *S. aureus* and 0.86 for *E. coli*. The *E. coli* performance is just below the estimated noise ceiling from replicate measurements. 
It is nearly matching the limits set by the labels' inherent variability.

The generative agent employs a PPO policy on a GATv2 graph, trained with autoregressive heads that determine what to add and where. It initially replicates the 
behavior of known antibiotics and then is anchored to this prior using KL divergence, facilitating exploration while avoiding nonsensical outputs. The training 
proceeds through three phases, starting with broad structural exploration, then gradually increasing molecule size from 25 to 30 heavy atoms, and finally expanding 
the top candidates. An inner-loop surrogate fingerprint network manages reward evaluations, reserving the computationally intensive GNN reward for final scoring. 
The reward combines predicted potency, drug-likeness, synthetic accessibility, novelty compared to DrugBank, and resistance evasion against CARD. The predicted potency 
is multiplied by an applicability-domain factor, ensuring the reward emphasizes potency only when molecules are close to the training chemistry, reducing the reward for 
molecules far from this chemical space.

The run generated 20,030 unique valid molecules. Under the canonical reward, the agent significantly exceeded the random and 
hill-climbing baselines, with Bonferroni-corrected $p$ values below $10^{-16}$. This includes both comparisons; Cliff's $\delta$ is 0.98 and 0.83. 
On the full reward distribution, the genetic algorithm and the character-level SMILES-RNN scored higher than the agent, with Cliff's 
$\delta$ of $-0.92$ and $-0.98$. However, each reached that score by collapsing onto a very small set of structures. The genetic algorithm 
converged to a single Bemis-Murcko scaffold across 100 molecules, and the SMILES-RNN covered only four scaffolds across roughly 13,000 
molecules. The Mann-Whitney test favors this concentration, rendering it an inappropriate metric for assessing generator quality in these two pools. 
The number of distinct scaffolds produced by a method is a design goal-relevant metric. Moreover, the agent produced 13,925 distinct scaffolds, 
and its 100 highest-scoring molecules span 100 distinct scaffolds, while the same counts for the genetic algorithm and the SMILES-RNN are 
1 and 2. The average reward of the agent's 100 best molecules also exceeds the single highest reward the genetic algorithm reached.

Regarding structural quality, the agent outperforms all methods, with scaffold diversity of 0.695 and internal diversity of 0.910. Every molecule 
it generates is novel compared to the DrugBank antibiotic reference. Over half of the molecules meet at least three of the four Lipinski criteria. 
Compared with known active antibiotics, the agent's pool is closest, with a Fréchet ChemNet Distance of 24.5, which is lower than the 44+ values 
reported by other methods. This exhibits the lowest divergence in physicochemical properties. Overall, the key insight is that the combined reward 
favors the collapsed pools, while the agent excels in maintaining structural diversity and similarity to real antibiotics.


Several caveats should be considered. Approximately 94% of the generated molecules activate at least one Brenk structural alert, especially the 
highest-scoring ones, which tend to be large and contain reactive groups. Therefore, applying a structural alert and drug-likeness filter before 
selecting molecules for synthesis or docking is recommended. The surrogate model used during rollouts, which replaces the graph network, shows only 
weak agreement with it on the generated pool, with a Pearson correlation of r=0.23 and 63% binary agreement. This indicates that the agent's full reward 
behavior relies more on structural reward terms and the behavior-cloned prior than on surrogate-guided potency. These results are based on a single random 
seed, one data split, one training run for the graph network, and one run per generator, so variability across seeds and splits remains unassessed. The paper's 
Limitations section discusses these issues further.


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

# 9. Surrogate-vs-GNN agreement on the pool (diagnostic) (~2 min)
python src/agreement.py
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
│   ├── stat_tests.py          # Mann-Whitney + KL + FCD
│   └── agreement.py           # Surrogate-vs-GNN agreement on the pool
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
