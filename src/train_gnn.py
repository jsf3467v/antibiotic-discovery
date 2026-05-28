"""
Multi-task GNN training on log10(MIC). Aggregates replicates by median
log-MIC per compound, masked Huber loss, scaffold split, atomic checkpointing.
Evaluation lives in evaluate.py.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from config import ProjectConfig, pick_device, release_cache
from src.gnn import MultiTaskGNN, multitask_huber_loss
from src.feature_engineering import (
    parallel_smiles_to_graphs_ordered,
    scaffold_split_dataset,
    split_dataset,
)

cfg = ProjectConfig()
CACHE_DIR = cfg.paths.processed / "graph_cache"


def aggregated_organism(name, threshold):
    """Median log-MIC per canonical SMILES; binary label derived from it."""
    path = cfg.paths.processed / f"{name}_mic_data.csv"
    df = pd.read_csv(path)
    clean = df.dropna(subset=["canonical_smiles", "mic_value"])
    clean = clean[clean["mic_value"] > 0].copy()
    clean["log_mic"] = np.log10(clean["mic_value"])
    agg = (clean.groupby("canonical_smiles")["log_mic"]
           .median().reset_index())
    agg["label"] = (agg["log_mic"] < np.log10(threshold)).astype(np.float32)
    return agg


def stamp_labels(graph, organism_key, label, log_mic):
    """Write binary label and continuous log-MIC target for one organism."""
    graph.y_saureus = torch.tensor([0.0])
    graph.y_ecoli = torch.tensor([0.0])
    graph.mask_saureus = torch.tensor([0.0])
    graph.mask_ecoli = torch.tensor([0.0])
    graph.logmic_saureus = torch.tensor([0.0])
    graph.logmic_ecoli = torch.tensor([0.0])
    setattr(graph, f"y_{organism_key}", torch.tensor([float(label)]))
    setattr(graph, f"mask_{organism_key}", torch.tensor([1.0]))
    setattr(graph, f"logmic_{organism_key}", torch.tensor([float(log_mic)]))
    return graph


def cached_graphs(df, organism_key, split_name):
    """Cached PyG graphs for one organism and split, suffixed _logmic."""
    cp = CACHE_DIR / f"{organism_key}_{split_name}_logmic.pt"
    if cp.exists():
        return torch.load(cp, weights_only=False)
    smiles = df["canonical_smiles"].tolist()
    labels = df["label"].values
    log_mics = df["log_mic"].values
    raw = parallel_smiles_to_graphs_ordered(smiles)
    graphs = [stamp_labels(g, organism_key, labels[i], log_mics[i])
              for i, g in enumerate(raw) if g is not None]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, cp)
    return graphs


def mic_splits(batch_size, seed, device):
    """Train/val/test DataLoaders plus aggregated-train class balance."""
    splits = {"train": [], "val": [], "test": []}
    balance = {}
    for org in cfg.data.organisms:
        tag = "saureus" if "aureus" in org else "ecoli"
        df = aggregated_organism(org, cfg.data.mic_threshold)
        if cfg.data.scaffold_split:
            trn, val, tst = scaffold_split_dataset(df, seed=seed)
        else:
            trn, val, tst = split_dataset(df, seed=seed)
        balance[tag] = float(trn["label"].mean())
        splits["train"].extend(cached_graphs(trn, tag, "train"))
        splits["val"].extend(cached_graphs(val, tag, "val"))
        splits["test"].extend(cached_graphs(tst, tag, "test"))

    if device.type == "cuda":
        kw = {"num_workers": cfg.train.num_workers, "pin_memory": True,
              "persistent_workers": cfg.train.num_workers > 0}
    else:
        kw = {"num_workers": 0, "pin_memory": False}
    loaders = {}
    rng = np.random.default_rng(seed)
    for name, data in splits.items():
        rng.shuffle(data)
        loaders[name] = DataLoader(data, batch_size=batch_size,
                                   shuffle=(name == "train"), **kw)
    return loaders["train"], loaders["val"], loaders["test"], balance


def forward_loss(model, batch, device):
    """Forward pass and  masked Huber loss on log-MIC targets."""
    batch = batch.to(device)
    edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None
    preds = model(batch.x, batch.edge_index, batch.batch,
                  edge_attr=edge_attr)
    targets = {"saureus": batch.logmic_saureus,
               "ecoli": batch.logmic_ecoli}
    masks = {"saureus": batch.mask_saureus,
             "ecoli": batch.mask_ecoli}
    return multitask_huber_loss(preds, targets, masks,
                                delta=cfg.train.huber_delta)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = torch.zeros(1, device=device)
    n = 0
    interval = cfg.train.mps_cache_interval
    is_mps = device.type == "mps"
    for i, batch in enumerate(loader):
        loss = forward_loss(model, batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        total.add_(loss.detach() * batch.num_graphs)
        n += batch.num_graphs
        if is_mps and (i + 1) % interval == 0:
            torch.mps.empty_cache()
    return (total / max(n, 1)).item()


def val_loss(model, loader, device):
    model.eval()
    total = torch.zeros(1, device=device)
    n = 0
    is_mps = device.type == "mps"
    interval = cfg.train.mps_cache_interval
    with torch.no_grad():
        for i, batch in enumerate(loader):
            loss = forward_loss(model, batch, device)
            total.add_(loss * batch.num_graphs)
            n += batch.num_graphs
            if is_mps and (i + 1) % interval == 0:
                torch.mps.empty_cache()
    return (total / max(n, 1)).item()


def atomic_write(obj, path):
    """Write to a temp file and rename. Survives mid-write crashes."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def snapshot(path, epoch, model, optimizer, scheduler, best_val, wait):
    atomic_write({"epoch": epoch, "best_val": best_val, "wait": wait,
                  "model": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict()}, path)


def try_resume(path, model, optimizer, scheduler, device):
    if not path.exists():
        return 1, float("inf"), 0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed from epoch {ckpt['epoch']}  "
          f"best_val={ckpt['best_val']:.4f}")
    return ckpt["epoch"] + 1, ckpt["best_val"], ckpt["wait"]


def epoch_step(model, train_loader, val_loader, optimizer, scheduler,
               device, epoch):
    """Run one training and validation epoch and step the scheduler."""
    t = train_epoch(model, train_loader, optimizer, device)
    v = val_loss(model, val_loader, device)
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:3d}  train={t:.4f}  val={v:.4f}  lr={lr:.1e}")
    return v


def best_or_wait(v, best_val, wait, model, ckpt_best, device):
    """Save best checkpoint when val improves, else increment patience."""
    if v < best_val:
        if device.type == "mps":
            torch.mps.synchronize()
        atomic_write(model.state_dict(), ckpt_best)
        return v, 0
    return best_val, wait + 1


def fit(model, optimizer, scheduler, train_loader, val_loader, device):
    ckpt_best = cfg.paths.models / "gnn_best.pt"
    ckpt_resume = cfg.paths.models / "gnn_resume.pt"
    start, best_val, wait = try_resume(
        ckpt_resume, model, optimizer, scheduler, device)
    for epoch in range(start, cfg.train.epochs + 1):
        v = epoch_step(model, train_loader, val_loader,
                       optimizer, scheduler, device, epoch)
        best_val, wait = best_or_wait(
            v, best_val, wait, model, ckpt_best, device)
        if wait >= cfg.train.patience:
            print(f"Early stop at epoch {epoch}")
            break
        if epoch % cfg.train.checkpoint_every == 0:
            if device.type == "mps":
                torch.mps.synchronize()
            snapshot(ckpt_resume, epoch, model, optimizer, scheduler,
                     best_val, wait)
        release_cache(device)
    ckpt_resume.unlink(missing_ok=True)
    return ckpt_best


def train():
    device = pick_device()
    print(f"Device: {device}")
    cfg.ensure_dirs()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    split_type = "scaffold" if cfg.data.scaffold_split else "random"
    print(f"Loading data {split_type} split, log-MIC regressio")
    trn, val, tst, balance = mic_splits(
        cfg.train.batch_size, cfg.train.seed, device)
    print(f"Train: {len(trn.dataset)}, Val: {len(val.dataset)}, "
          f"Test: {len(tst.dataset)}")
    print(f"Active rate (train): "
          f"saureus={balance.get('saureus', 0):.1%}, "
          f"ecoli={balance.get('ecoli', 0):.1%}")
    model = MultiTaskGNN(cfg.atom, cfg.gnn).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs,
        eta_min=cfg.train.cosine_eta_min)
    ckpt = fit(model, optimizer, scheduler, trn, val, device)
    release_cache(device)
    print(f"\nTraining complete. Best checkpoint: {ckpt.name}")


if __name__ == "__main__":
    train()