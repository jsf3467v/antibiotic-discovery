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

# Antibiotic Discovery with Deep RL + Graph Neural Networks

Multi-task GATv2 regressor and three-phase PPO agent that together generate novel antibiotic candidates against *S. aureus* and *E. coli*. Final project for the Advanced Machine Learning course, JHU MS in AI program (May 2026).

**Code, README, and full reproducibility instructions:** [github.com/jsf3467v/antibiotic-discovery](https://github.com/jsf3467v/antibiotic-discovery)

## Files in this repository

- **`gnn_best.pt`** — trained multi-task GATv2 regressor. Test AUROC 0.84 (*S. aureus*) and 0.86 (*E. coli*) on a scaffold-split test set of 7,874 compounds. The scaffold assignment spans both organisms jointly, so no scaffold crosses folds between them, and the *E. coli* result sits just below the empirical replicate-noise ceiling.
- **`policy_final.pt`** — final RL policy from phase 3, used to generate the 20,032-molecule candidate pool.
- **`policy_prior.pt`** — behavior-cloned policy used as the KL anchor during PPO training.
- **`surrogate.pt`** — fingerprint MLP surrogate, used for fast inner-loop reward calls during rollouts.
- **`policy_phase1.pt`, `policy_phase2.pt`, `policy_phase3.pt`, `policy_phase3_best.pt`** — per-phase policy snapshots, provided for inspection and not required to reproduce the tables.
- **`paper.pdf`** — full paper with tables, figures, methodology, results, and discussion.

## Headline results

The RL agent produces 20,032 unique valid molecules and significantly outperforms the random, hill-climbing, and genetic-algorithm baselines, with Bonferroni-corrected *p* below 10^-16 against the first two and 1.2 \times 10^-4 against the genetic algorithm. The character-level SMILES-RNN, once its vocabulary was corrected, reaches a higher full-distribution reward than the agent, though it does so through synthetic accessibility rather than potency, and the agent keeps the higher top-ten reward. On resemblance to known active antibiotics the agent leads every method, with the lowest Fréchet ChemNet Distance of 24.1 against 44 or higher for every other pool. About 92% of generated molecules trigger at least one Brenk structural alert and would require medicinal-chemistry refinement before any synthesis. A fingerprint surrogate stands in for the regressor during rollouts and agrees with it only weakly on the generated pool, at a Pearson correlation of 0.22, while agreeing more closely on in-distribution chemistry at 0.57. The full breakdown, including caveats and limitations, is in the paper.

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
