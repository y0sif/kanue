"""Shared training and evaluation loops.

Used by both baseline and KAN notebooks to ensure identical
training conditions for fair comparison.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from kanue.utils.drive import DriveCheckpointer


def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> float:
    """Train for one epoch, return average loss."""
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    model.train()
    total_loss = 0.0
    n_batches = 0

    for stm, nstm, targets in tqdm(loader, desc="Training", leave=False):
        stm = stm.to(device)
        nstm = nstm.to(device)
        targets = targets.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        pred = model(stm, nstm)
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
) -> dict:
    """Evaluate model, return loss and accuracy metrics."""
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    with torch.no_grad():
        for stm, nstm, targets in loader:
            stm = stm.to(device)
            nstm = nstm.to(device)
            targets = targets.to(device, dtype=torch.float32)

            pred = model(stm, nstm)
            total_loss += loss_fn(pred, targets).item()

            # "Correct" = predicted winner matches actual winner (threshold 0.5)
            pred_winner = (pred > 0.5).float()
            target_winner = (targets > 0.5).float()
            total_correct += (pred_winner == target_winner).sum().item()
            total_samples += targets.size(0)
            n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpointer: DriveCheckpointer,
    checkpoint_every: int = 5,
    lr_drop_epoch: int | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    """Full training loop with checkpointing and logging.

    Returns training log dict with per-epoch metrics.
    """
    loss_fn = nn.MSELoss()
    log = {"train_loss": [], "val_loss": [], "val_accuracy": [], "epoch_time": []}

    # Resume from checkpoint if available
    start_epoch = 0
    existing = checkpointer.latest_checkpoint()
    if existing is not None:
        print(f"Resuming from epoch {existing}")
        checkpointer.load_checkpoint(existing, model, optimizer)
        start_epoch = existing + 1
        prev_log = checkpointer.load_log()
        if prev_log:
            log = prev_log

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # LR drop
        if lr_drop_epoch and epoch == lr_drop_epoch:
            for pg in optimizer.param_groups:
                pg["lr"] /= 10
            print(f"LR dropped to {optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        elapsed = time.time() - t0

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_metrics["loss"])
        log["val_accuracy"].append(val_metrics["accuracy"])
        log["epoch_time"].append(elapsed)

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if scheduler:
            scheduler.step()

        # Checkpoint
        if (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1:
            checkpointer.save_checkpoint(model, optimizer, epoch, train_loss)
            checkpointer.save_log(log)

    return log
