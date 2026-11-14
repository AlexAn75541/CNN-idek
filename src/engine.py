"""CPU-optimized training and evaluation loops, Phase 1/2 scheduler, and checkpointing."""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.cpu_runtime import get_device, seed_everything
from src.dataset import create_dataloaders, load_dataset, stratified_split
from src.models import (
    MajorityClassBaseline,
    ScratchCNN,
    build_resnet18_transfer,
    unfreeze_layer4,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Train for one epoch with channels_last tensor layout."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        if x.dim() == 4 and not x.is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += batch_size

    epoch_loss = total_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model under torch.inference_mode() returning loss, acc, preds, targets, and probs."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []
    total = 0

    with torch.inference_mode():
        for x, y in dataloader:
            if x.dim() == 4 and not x.is_contiguous(memory_format=torch.channels_last):
                x = x.to(memory_format=torch.channels_last)
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            batch_size = y.size(0)

            if criterion is not None:
                loss = criterion(outputs, y)
                total_loss += loss.item() * batch_size

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
            all_probs.append(probs.cpu().numpy())
            total += batch_size

    avg_loss = (total_loss / max(total, 1)) if criterion is not None else 0.0
    preds_arr = np.array(all_preds, dtype=int)
    targets_arr = np.array(all_targets, dtype=int)
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0))
    accuracy = float(np.mean(preds_arr == targets_arr)) if total > 0 else 0.0

    return avg_loss, accuracy, preds_arr, targets_arr, probs_arr


def save_checkpoint(
    state: dict,
    filepath: str,
) -> None:
    """Save model checkpoint to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load model checkpoint from disk."""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module = nn.CrossEntropyLoss(),
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
    desc: str = "Training",
    eval_every_epoch: bool = False,
) -> Dict[str, list]:
    """Train model over specified epochs, saving best validation checkpoint."""
    device = get_device()
    model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        should_eval = eval_every_epoch or (epoch == 1) or (epoch == epochs)
        if should_eval:
            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if verbose:
                print(
                    f"[{desc}] Epoch {epoch}/{epochs} | "
                    f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%"
                )
            if val_acc > best_val_acc and checkpoint_path is not None:
                best_val_acc = val_acc
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )
        else:
            if verbose:
                print(f"[{desc}] Epoch {epoch}/{epochs} | Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc * 100:.2f}%")

    return history


def run_all_experiments(
    data_dir: Optional[str] = None,
    seeds: List[int] = [42, 100, 2026],
    n_classes: int = 5,
    p1_epochs: int = 5,
    p2_epochs: int = 5,
    scratch_epochs: int = 5,
    batch_size: int = 90,
    checkpoint_dir: str = "./artifacts/checkpoints",
    verbose: bool = False,
) -> Dict[str, dict]:
    """Execute all 5 models across random seeds to compute mean ± std (Slide 62)."""
    criterion = nn.CrossEntropyLoss()
    os.makedirs(checkpoint_dir, exist_ok=True)

    results_by_model: Dict[str, List[float]] = {
        "Majority Baseline": [],
        "Scratch CNN": [],
        "Transfer Phase 1 (No Aug)": [],
        "Transfer Phase 1 (Aug)": [],
        "Transfer Phase 2 (layer4)": [],
    }

    first_seed_artifacts: dict = {}

    for s_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\n================ Running Seed {seed} ({s_idx + 1}/{len(seeds)}) ================")
        seed_everything(seed)
        samples, class_names = load_dataset(data_dir=data_dir, n_classes=n_classes, seed=seed)

        # 1. Augmented datasets
        tr_ds, val_ds, test_ds = stratified_split(
            samples, class_names, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=seed, augment_train=True
        )
        tr_loader, val_loader, test_loader = create_dataloaders(
            tr_ds, val_ds, test_ds, batch_size=batch_size, num_workers=0
        )

        # 2. Non-augmented dataset for ablation
        tr_ds_noaug, val_ds_noaug, _ = stratified_split(
            samples, class_names, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=seed, augment_train=False
        )
        tr_loader_noaug, val_loader_noaug, _ = create_dataloaders(
            tr_ds_noaug, val_ds_noaug, test_ds, batch_size=batch_size, num_workers=0
        )

        # Model 1: Majority Baseline
        baseline = MajorityClassBaseline(n_classes=n_classes)
        baseline.fit_loader(tr_loader)
        base_res = baseline.evaluate(val_loader)
        results_by_model["Majority Baseline"].append(base_res["accuracy"])

        # Model 2: Scratch CNN
        scratch_model = ScratchCNN(n_classes=n_classes, use_adaptive_pool=True)
        opt_scratch = torch.optim.Adam(scratch_model.parameters(), lr=1e-3)
        ckpt_scratch = os.path.join(checkpoint_dir, f"scratch_seed{seed}.pt")
        hist_scratch = train_model(
            scratch_model,
            tr_loader,
            val_loader,
            epochs=scratch_epochs,
            optimizer=opt_scratch,
            criterion=criterion,
            checkpoint_path=ckpt_scratch,
            verbose=verbose,
            desc="Scratch CNN",
            eval_every_epoch=False,
        )
        _, scratch_val_acc, _, _, _ = evaluate(scratch_model, val_loader, criterion)
        results_by_model["Scratch CNN"].append(scratch_val_acc)

        # Model 3: ResNet-18 Phase 1 No Augmentation (Ablation)
        r18_noaug = build_resnet18_transfer(n_classes=n_classes)
        # Fast feature caching for unaugmented split
        with torch.no_grad():
            all_feats = []
            all_targets = []
            for x, y in tr_loader_noaug:
                feat = r18_noaug.avgpool(
                    r18_noaug.layer4(
                        r18_noaug.layer3(
                            r18_noaug.layer2(
                                r18_noaug.layer1(
                                    r18_noaug.maxpool(
                                        r18_noaug.relu(r18_noaug.bn1(r18_noaug.conv1(x.to(memory_format=torch.channels_last))))
                                    )
                                )
                            )
                        )
                    )
                )
                all_feats.append(torch.flatten(feat, 1).clone())
                all_targets.append(y.clone())
            cached_feats = torch.cat(all_feats, dim=0)
            cached_targets = torch.cat(all_targets, dim=0)

        opt_noaug = torch.optim.Adam(r18_noaug.fc.parameters(), lr=1e-3)
        for _ in range(p1_epochs):
            opt_noaug.zero_grad(set_to_none=True)
            out = r18_noaug.fc(cached_feats)
            loss = criterion(out, cached_targets)
            loss.backward()
            opt_noaug.step()

        _, noaug_val_acc, _, _, _ = evaluate(r18_noaug, val_loader, criterion)
        results_by_model["Transfer Phase 1 (No Aug)"].append(noaug_val_acc)

        # Model 4: ResNet-18 Phase 1 With Augmentation (Frozen)
        r18_aug = build_resnet18_transfer(n_classes=n_classes)
        opt_p1 = torch.optim.Adam(r18_aug.fc.parameters(), lr=1e-3)
        ckpt_p1 = os.path.join(checkpoint_dir, f"r18_p1_seed{seed}.pt")
        hist_p1 = train_model(
            r18_aug,
            tr_loader,
            val_loader,
            epochs=p1_epochs,
            optimizer=opt_p1,
            criterion=criterion,
            checkpoint_path=ckpt_p1,
            verbose=verbose,
            desc="Transfer P1 (Aug)",
            eval_every_epoch=False,
        )
        _, p1_val_acc, _, _, _ = evaluate(r18_aug, val_loader, criterion)
        results_by_model["Transfer Phase 1 (Aug)"].append(p1_val_acc)

        # Model 5: ResNet-18 Phase 2 (Unfreeze layer4 at lr=1e-5)
        unfreeze_layer4(r18_aug)
        active_params = [p for p in r18_aug.parameters() if p.requires_grad]
        opt_p2 = torch.optim.Adam(active_params, lr=1e-5)
        ckpt_p2 = os.path.join(checkpoint_dir, f"r18_p2_seed{seed}.pt")
        hist_p2 = train_model(
            r18_aug,
            tr_loader,
            val_loader,
            epochs=p2_epochs,
            optimizer=opt_p2,
            criterion=criterion,
            checkpoint_path=ckpt_p2,
            verbose=verbose,
            desc="Transfer P2 (layer4)",
            eval_every_epoch=False,
        )
        _, p2_val_acc, _, _, _ = evaluate(r18_aug, val_loader, criterion)
        results_by_model["Transfer Phase 2 (layer4)"].append(p2_val_acc)

        if s_idx == 0:
            first_seed_artifacts = {
                "class_names": class_names,
                "train_loader": tr_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
                "scratch_model": scratch_model,
                "r18_p1_noaug": r18_noaug,
                "r18_p1_aug": r18_aug,
                "hist_scratch": hist_scratch,
                "hist_p1": hist_p1,
                "hist_p2": hist_p2,
                "p1_val_acc": p1_val_acc,
                "p2_val_acc": p2_val_acc,
                "phase_delta": p2_val_acc - p1_val_acc,
                "ablation_drop": p1_val_acc - noaug_val_acc,
            }

    summary: Dict[str, dict] = {}
    for model_name, accs in results_by_model.items():
        arr = np.array(accs)
        summary[model_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "runs": [float(a) for a in accs],
        }

    return {
        "summary": summary,
        "first_seed": first_seed_artifacts,
        "seeds": seeds,
    }
