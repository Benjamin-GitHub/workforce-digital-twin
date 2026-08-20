from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stgcn_model import STGCN

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
from split_utils import split_dataset


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def split_groups(y, groups, fractions, seed, attempts=500, strategy=None):
    return split_dataset(
        y, groups, fractions, seed, strategy=strategy, attempts=attempts
    )


def loader(x, y, indices, batch_size, shuffle, workers):
    dataset = TensorDataset(torch.from_numpy(x[indices]).float(), torch.from_numpy(y[indices]).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


@torch.no_grad()
def predict(model, data, device):
    model.eval(); truth, predictions = [], []
    for x, y in data:
        predictions.extend(model(x.to(device)).argmax(1).cpu().tolist()); truth.extend(y.tolist())
    return np.asarray(truth), np.asarray(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/train.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); seed_everything(cfg["seed"])
    archive = np.load(args.data); x, y, groups, sample_ids = archive["x"], archive["y"], archive["groups"], archive["sample_ids"]
    classes = archive["classes"].tolist()
    if x.ndim != 5 or x.shape[1] != 3 or x.shape[2] <= 0 or x.shape[3] != 17 or x.shape[4] != 1:
        raise ValueError(f"Expected archive x shaped (N,3,T,17,1) with T > 0, found {x.shape}")
    if not (len(x) == len(y) == len(groups) == len(sample_ids)):
        raise ValueError("x, y, groups, and sample_ids lengths differ")
    if not classes or np.any(y < 0) or np.any(y >= len(classes)):
        raise ValueError("y labels must index the classes array")
    train_i, val_i, test_i = split_groups(
        y, groups, cfg["split"], cfg["seed"], strategy=cfg.get("split_strategy")
    )
    if args.device == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name); print(f"device={device}; cuda_available={torch.cuda.is_available()}")
    train_loader = loader(x, y, train_i, cfg["batch_size"], True, cfg["num_workers"])
    val_loader = loader(x, y, val_i, cfg["batch_size"], False, cfg["num_workers"])
    test_loader = loader(x, y, test_i, cfg["batch_size"], False, cfg["num_workers"])
    model = STGCN(len(classes), cfg["hidden_channels"], cfg["dropout"]).to(device)
    counts = np.bincount(y[train_i], minlength=len(classes))
    if np.any(counts == 0):
        missing = [classes[index] for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"Training split has no samples for: {missing}")
    balanced = len(train_i) / (len(classes) * counts)
    if cfg["class_weights"] == "balanced":
        weights = balanced
    elif cfg["class_weights"] == "sqrt_balanced":
        weights = np.sqrt(balanced)
    elif cfg["class_weights"] in (None, "none"):
        weights = np.ones(len(classes))
    else:
        raise ValueError("class_weights must be balanced, sqrt_balanced, or none")
    print("class_weights=" + json.dumps(dict(zip(classes, weights.tolist()))))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler_cfg = cfg.get("lr_scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_cfg.get("factor", 0.5),
        patience=scheduler_cfg.get("patience", 5),
        min_lr=scheduler_cfg.get("minimum_lr", 1e-5),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "splits.npz", train=train_i, val=val_i, test=test_i)
    split_manifest = {
        name: sample_ids[index].tolist()
        for name, index in (("train", train_i), ("val", val_i), ("test", test_i))
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    best_f1 = -1.0
    history = []
    for epoch in range(1, (args.epochs or cfg["epochs"]) + 1):
        model.train(); losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True); logits = model(batch_x.to(device)); loss = criterion(logits, batch_y.to(device)); loss.backward(); optimizer.step(); losses.append(loss.item())
        val_y, val_pred = predict(model, val_loader, device); score = f1_score(val_y, val_pred, average="macro", zero_division=0)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_f1": float(score), "learning_rate": learning_rate})
        print(f"epoch={epoch} loss={np.mean(losses):.5f} val_macro_f1={score:.5f} lr={learning_rate:.7f}")
        if score > best_f1:
            best_f1 = score
            torch.save({"model_state": model.state_dict(), "classes": classes, "input_shape": [int(value) for value in x.shape[1:]], "epoch": epoch, "val_macro_f1": score, "config": cfg}, args.output / "best.pt")
        scheduler.step(score)
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state"])
    test_y, test_pred = predict(model, test_loader, device)
    report = classification_report(test_y, test_pred, labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0)
    metrics = {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"], "per_class": {name: report[name] for name in classes}, "split_samples": {"train": len(train_i), "val": len(val_i), "test": len(test_i)}, "split_groups": {"train": len(set(groups[train_i])), "val": len(set(groups[val_i])), "test": len(set(groups[test_i]))}}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "val_macro_f1", "learning_rate"])
        writer.writeheader(); writer.writerows(history)
    with (args.output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["actual/predicted", *classes]); writer.writerows([[classes[i], *row] for i, row in enumerate(confusion_matrix(test_y, test_pred, labels=range(len(classes))))])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
