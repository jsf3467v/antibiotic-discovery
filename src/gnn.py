"""
Multi-task GATv2 regressor for log10(MIC). Edge dropout, mean and max pooling,
organism-specific heads. Heads predict log10(MIC) directly; a sigmoid
wrapper produces active-class probabilities for downstream consumers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import dropout_edge

from config import GNNConfig, AtomConfig


def log_mic_to_prob_torch(log_mic, threshold, scale=1.0):
    """Sigmoid wrapper from log10(MIC) prediction to active-class probability."""
    log_thr = float(np.log10(threshold))
    return torch.sigmoid((log_thr - log_mic) / scale)


class SharedEncoder(nn.Module):

    def __init__(self, atom_cfg: AtomConfig, gnn_cfg: GNNConfig):
        super().__init__()
        self.edge_dropout = gnn_cfg.edge_dropout
        self.input_proj = nn.Linear(atom_cfg.features, gnn_cfg.hidden_dim)
        self.convs = nn.ModuleList([
            GATv2Conv(gnn_cfg.hidden_dim, gnn_cfg.hidden_dim,
                      heads=gnn_cfg.heads, edge_dim=gnn_cfg.edge_dim,
                      concat=False)
            for _ in range(gnn_cfg.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(gnn_cfg.hidden_dim)
            for _ in range(gnn_cfg.num_layers)
        ])
        self.drop = nn.Dropout(gnn_cfg.dropout)
        self.use_mean_max = gnn_cfg.pool == "mean_max"

    @property
    def out_dim(self):
        hd = self.convs[0].out_channels
        return hd * 2 if self.use_mean_max else hd

    def drop_edges(self, edge_index, edge_attr):
        if not self.training or self.edge_dropout <= 0:
            return edge_index, edge_attr
        ei, mask = dropout_edge(edge_index, p=self.edge_dropout, training=True)
        ea = edge_attr[mask] if edge_attr is not None else None
        return ei, ea

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.input_proj(x)
        ei, ea = self.drop_edges(edge_index, edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            x = self.drop(F.relu(norm(conv(x, ei, edge_attr=ea)))) + x
        if self.use_mean_max:
            return torch.cat([global_mean_pool(x, batch),
                              global_max_pool(x, batch)], dim=-1)
        return global_mean_pool(x, batch)


class OrganismHead(nn.Module):
    """Two-layer MLP that predicts log10(MIC) from a graph embedding."""

    def __init__(self, in_dim, mid_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)


class MultiTaskGNN(nn.Module):

    def __init__(self, atom_cfg: AtomConfig, gnn_cfg: GNNConfig):
        super().__init__()
        self.encoder = SharedEncoder(atom_cfg, gnn_cfg)
        rd = self.encoder.out_dim
        self.head_saureus = OrganismHead(
            rd, gnn_cfg.readout_dim, gnn_cfg.dropout)
        self.head_ecoli = OrganismHead(
            rd, gnn_cfg.readout_dim, gnn_cfg.dropout)

    def forward(self, x, edge_index, batch, edge_attr=None):
        """Per-organism predicted log10(MIC)."""
        h = self.encoder(x, edge_index, batch, edge_attr=edge_attr)
        return {
            "saureus": self.head_saureus(h),
            "ecoli": self.head_ecoli(h),
        }


def multitask_huber_loss(preds, targets, masks, delta=1.0):
    """Masked Huber loss across organism heads, averaged over active tasks."""
    loss = torch.tensor(0.0, device=next(iter(preds.values())).device)
    n_tasks = 0
    for key in preds:
        m = masks[key].bool()
        if not m.any():
            continue
        loss = loss + F.huber_loss(
            preds[key][m], targets[key][m], delta=delta)
        n_tasks += 1
    return loss / max(n_tasks, 1)