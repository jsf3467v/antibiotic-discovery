---
license: mit
tags:
  - reinforcement-learning
  - graph-neural-networks
  - drug-discovery
  - antibiotic-discovery
  - cheminformatics
library_name: pytorch
---

# Antibiotic Discovery with Deep Reinforcement Learning and Graph Neural Networks

This model is a multi-task GATv2 regressor and a three-phase PPO agent that together generate novel antibiotic candidates against *S. aureus* and *E. coli*. This was a final project for an Advanced Machine Learning course, JHU MS in AI program (May 2026).

**Code, README, and full reproducibility instructions** are at [github.com/jsf3467v/antibiotic-discovery](https://github.com/jsf3467v/antibiotic-discovery).

## Files in this repository

- **`gnn_best.pt`**, the trained multi-task GATv2 regressor. Test AUROC is 0.84 (*S. aureus*) and 0.86 (*E. coli*) on a held-out scaffold-split test set. Scaffolds do not cross folds within each organism, and the *E. coli* result sits just below the empirical replicate-noise ceiling.
- **`policy_final.pt`**, the final reinforcement learning policy from phase 3, used to generate the 20,030-molecule candidate pool.
- **`policy_prior.pt`**, the behavior-cloned policy used as the KL anchor during PPO training.
- **`surrogate.pt`**, the fingerprint multilayer perceptron surrogate, used for fast inner-loop reward calls during sampling.
- **`policy_phase1.pt`, `policy_phase2.pt`, `policy_phase3.pt`, `policy_phase3_best.pt`**, the per-phase policy snapshots, provided for inspection and not required to reproduce the tables.
- **`paper.pdf`**, the full paper with tables, figures, methodology, results, and discussion.

## Results

The reinforcement learning agent produced 20,030 unique and valid molecules. Under the canonical reward it significantly outperformed the random and hill-climbing baselines, with Bonferroni-corrected \\(p\\) values below \\(10^{-16}\\) in both comparisons and Cliff's \\(\delta\\) of 0.98 and 0.83. The genetic algorithm and the character-level SMILES-RNN reached a higher full-distribution reward than the agent, with Cliff's \\(\delta\\) of \\(-0.92\\) and \\(-0.98\\). However, each collapsed onto a very small set of molecule structures. The genetic algorithm converged to a single Bemis-Murcko scaffold across 100 molecules, and the SMILES-RNN collapsed onto a single dominant scaffold across approximately 13,000 molecules. Because the Mann-Whitney test rewards this concentration, it is not the right measure of generator quality for those two pools. On the count of distinct scaffolds, which matches the design goal, the agent produced 13,925 scaffolds, and its 100 best molecules span 100 distinct scaffolds, far more than the collapsed baselines. The average reward of the agent's 100 best molecules also exceeds the single highest reward the genetic algorithm reached.

The agent's pool closely resembles known active antibiotics, with a Fréchet ChemNet Distance of 24.5 to the reference, the lowest among all pools and well below the 44+ values seen in others. Its physicochemical property divergence is also minimal. It exhibits a scaffold diversity of 0.695 and an internal diversity of 0.910, with every molecule being novel compared to the DrugBank antibiotic reference. Approximately 94% of generated molecules trigger at least one Brenk structural alert, and the most highly scored ones are large and contain reactive groups, suggesting the need for structural-alert and drug-likeness filters before synthesis or docking. The fingerprint surrogate used in sampling correlates weakly with the regressor on the generated set, with a Pearson correlation of 0.23 and 63% binary agreement. Thus, the agent's behavior relies more on structural reward terms and behavior-cloned priors than on the surrogate-guided potency. These results are from a single seed, data split, and run, so variance is not fully characterized. Additional details, limitations, and caveats are in the paper.

## Loading checkpoints

```python
import torch

# Example: load the trained GNN regressor
state_dict = torch.load("gnn_best.pt", weights_only=True)
# Then instantiate MultiTaskGNN from src/gnn.py in the GitHub repo
# and call .load_state_dict(state_dict)
```

See `src/evaluate.py` in the GitHub repository for the full loading and evaluation pattern.

## Citation

```bibtex
@misc{keith2026antibiotic,
  author       = {Keith, Arlene},
  title        = {Deep Reinforcement Learning with Graph Neural Networks for Antibiotic Discovery},
  year         = {2026},
  howpublished = {Final project, Advanced Machine Learning course, JHU MS in AI program},
  url          = {https://github.com/jsf3467v/antibiotic-discovery}
}
```

## License

MIT. See LICENSE in the [GitHub repository](https://github.com/jsf3467v/antibiotic-discovery).
