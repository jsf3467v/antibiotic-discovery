---
tags:
- reinforcement-learning
- graph-neural-networks
- molecular-generation
- drug-discovery
- antibiotics
library_name: pytorch
---

# Deep Reinforcement Learning with Graph Neural Networks for Antibiotic Discovery

This repository hosts the trained checkpoints for a machine learning pipeline that designs new antibiotic candidates. A multi-task GATv2 graph neural network predicts antibacterial potency, measured as $\log_{10}(\mathrm{MIC})$, for *S. aureus* and *E. coli*, and a three-phase PPO agent generates novel molecules that score highly under a composite reward built on that prediction. The work is the final project for the Advanced Machine Learning course in the JHU MS in AI program, and the full write-up, tables, and figures are in the [paper](https://huggingface.co/jsf3467v/antibiotic-discovery/blob/main/paper.pdf). The code is on [GitHub](https://github.com/jsf3467v/antibiotic-discovery).

## Model description

The pipeline has two trained parts. The potency model is a three-layer GATv2 encoder with organism-specific regression heads that outputs $\log_{10}(\mathrm{MIC})$ for each organism, wrapped by a sigmoid into an active-class probability. The generative agent is a PPO policy on a GATv2 graph with autoregressive heads that choose an action type, an anchor atom, and a target atom. During sampling the agent scores candidates through a fingerprint surrogate that stands in for the graph network, and the graph network reward is reserved for final scoring. A fingerprint surrogate and a behavior-cloned prior support the agent and are provided as checkpoints.

The composite reward combines predicted potency, synthetic accessibility, drug-likeness, novelty against DrugBank, and resistance evasion against CARD, with weights of 0.30, 0.25, 0.20, 0.15, and 0.10 in that order. A two-sided size band centered at 30 heavy atoms keeps molecules in the active size range, a composition penalty discourages heteroatom-rich exploits, and an applicability-domain factor reduces potency credit for molecules far from the training chemistry. In the reported run the applicability-domain factor was active, flooring at 0.25 for the most distant molecules and rising to 1.0 for those closest to the training set.

## Hosted checkpoints

The graph network, the surrogate, and the behavior-cloned prior are seed-invariant and are provided as one file each. The final policy is produced once per seed, so three copies are provided, one for each of seeds 42, 43, and 44.

| Checkpoint | Role | Scope | Size |
| --- | --- | --- | --- |
| `gnn_best.pt` | Multi-task GATv2 potency regressor | Global | 1.7 MB |
| `surrogate.pt` | Fingerprint potency surrogate used during sampling | Global | 2.2 MB |
| `policy_prior.pt` | Behavior-cloned prior and KL anchor | Global | 1.8 MB |
| `policy_final.pt` (seed 42, 43, 44) | Final PPO policy for one seed | Per seed | 1.8 MB each |

The per-phase policy snapshots `policy_phase1.pt` through `policy_phase3_best.pt` may also be present for inspection but are not required to reproduce the results.

## Training data

The potency model was trained on 78,314 compound-organism measurements from ChEMBL, derived from 112,642 raw minimum inhibitory concentration readings and split into training, validation, and test sets in an 80/10/10 ratio by scaffold. Novelty scoring uses 458 DrugBank antibiotics and resistance scoring uses 457 substrate SMILES from the Comprehensive Antibiotic Resistance Database. The raw data is not redistributed here and is available from the public sources described in the GitHub repository.

## Training procedure

The graph network was trained for 61 epochs under a masked Huber loss with an Adam optimizer and a cosine decay schedule, with early stopping. The agent was first pretrained by behavior cloning on expert build trajectories from active compounds, then optimized with Proximal Policy Optimization across three phases. Phase 1 holds a strong KL anchor to the prior at a size center of 25 heavy atoms, phase 2 relaxes the anchor while raising the size center to 30, and phase 3 holds at 30 and expands the top candidates. The surrogate was trained once before reinforcement learning on a stratified sample of active and inactive compounds and was held fixed during training.

## Evaluation

On a held-out scaffold-split test set the graph network reaches an AUROC of 0.84 for *S. aureus* and 0.86 for *E. coli*, with the *E. coli* result close to the noise ceiling estimated from replicate measurements. The agent generates close to 20,000 unique valid molecules per seed and, under the canonical reward, decisively exceeds the random and hill-climbing baselines with Bonferroni-corrected $p$ values below $10^{-16}$. The genetic algorithm and the SMILES-RNN post higher pool-wide reward but reach it only through low structural diversity, so on structural diversity, drug-likeness, and distributional similarity to known active antibiotics the agent leads every method.

The pool metrics are stable across the three seeds.

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

## Intended use and limitations

These checkpoints are a research artifact for de novo antibiotic candidate generation and potency prediction on *S. aureus* and *E. coli*. They are not validated for clinical or laboratory use, and generated molecules require medicinal chemistry review before any synthesis or assay.

Several limitations apply. Approximately 88 percent of the generated molecules activate at least one Brenk structural alert, and nearly 30 percent of the pool consists of acyclic molecules, so structural-alert and drug-likeness filtering is needed before selection. The surrogate that stands in for the graph network during sampling shows only weak agreement with it on the generated pool, with a Pearson correlation of $r = 0.18$ and 67 percent agreement on binary active-class calls, so the reward behavior depends more on the structural reward terms and the behavior-cloned prior than on surrogate-guided potency. The generation results come from three seeds for the agent and the baselines, so run-to-run variability of the generation is measured, while the graph network is trained once on a single scaffold split, so variability from the model and the split remains unmeasured. The potency model was trained on whole-cell measurements, so a high predicted score does not guarantee activity in vitro.

## How to use

Download the checkpoints into `models/`, then place each seed's policy under `runs/seed{N}/`. With the three global checkpoints present and a seed's policy in place, `eval_rl.py` and `stat_tests.py` regenerate that seed's result tables with no training run, and `evaluate.py` regenerates the graph network metrics. Full setup and reproduction steps are in the GitHub repository.

```bash
hf download jsf3467v/antibiotic-discovery --local-dir models
```

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
