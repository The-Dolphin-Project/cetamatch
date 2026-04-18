"""
Training script for the FinEmbedder model.

Usage:
    python -m app.training.train \
        --data-dir ./data \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-4 \
        --output ./models/fin_embedder.pt

Data layout:
    data/
      individual_A/   (use the individual's catalogue ID as the folder name)
        photo1.jpg
        photo2.jpg
      individual_B/
        ...

At least 2 photos per individual are required to form triplets.
More photos per individual = better model. Even 3–5 photos per individual
will produce useful embeddings thanks to the ImageNet pretrained backbone.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from ..core.model import FinEmbedder, EMBEDDING_DIM
from .dataset import FinTripletDataset


# ---------------------------------------------------------------------------
# Triplet loss with online hard mining
# ---------------------------------------------------------------------------

def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Batch hard triplet loss.
    d(a,p) and d(a,n) are Euclidean distances of L2-normalised embeddings,
    which equals sqrt(2 - 2*cosine_similarity) for unit vectors.
    """
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    val_split: float = 0.15,
    margin: float = 0.3,
    num_workers: int = 2,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}")

    # Dataset
    full_dataset = FinTripletDataset(data_dir, train=True)
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Use val transform for the validation split
    val_ds.dataset = FinTripletDataset(data_dir, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    # Model
    model = FinEmbedder(embedding_dim=EMBEDDING_DIM, pretrained=True).to(device)

    # Fine-tune: use a lower LR for the backbone, higher for the projection head
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.projector.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params,     "lr": lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for anchor, positive, negative in train_loader:
            anchor   = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = triplet_loss(emb_a, emb_p, emb_n, margin=margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor   = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)
                val_loss += triplet_loss(emb_a, emb_p, emb_n, margin=margin).item()
        val_loss /= max(1, len(val_loader))

        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"  → Saved best model to {output_path}")

    print(f"\n[train] Done. Best val loss: {best_val_loss:.4f}")
    print(f"[train] Model saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the FinEmbedder triplet network")
    parser.add_argument("--data-dir",   required=True,             help="Root directory of labelled fin images")
    parser.add_argument("--output",     default="models/fin_embedder.pt")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--margin",     type=float, default=0.3)
    parser.add_argument("--val-split",  type=float, default=0.15)
    parser.add_argument("--workers",    type=int,   default=2)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        val_split=args.val_split,
        num_workers=args.workers,
    )
