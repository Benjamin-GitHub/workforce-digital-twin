from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gru_data import group_safe_split, load_pose_archive
from gru_model import StreamingGRU


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(x, y, indices, batch_size, shuffle, workers):
    dataset = TensorDataset(
        torch.from_numpy(x[indices]),
        torch.from_numpy(y[indices]),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def predict(model, data, device):
    model.eval()
    truth, predictions = [], []
    for x, y in data:
        logits, _ = model(x.to(device))
        predictions.extend(logits.argmax(1).cpu().tolist())
        truth.extend(y.tolist())
    return np.asarray(truth), np.asarray(predictions)


def class_weights(y, indices, classes, mode):
    counts = np.bincount(y[indices], minlength=len(classes))
    if np.any(counts == 0):
        missing = [classes[i] for i, count in enumerate(counts) if count == 0]
        raise ValueError(f"Training split has no samples for: {missing}")
    balanced = len(indices) / (len(classes) * counts)
    if mode == "balanced":
        return balanced
    if mode == "sqrt_balanced":
        return np.sqrt(balanced)
    if mode in (None, "none"):
        return np.ones(len(classes))
    raise ValueError("class_weights must be balanced, sqrt_balanced, or none")


def write_confusion(path, matrix, classes):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *classes])
        writer.writerows([[classes[i], *row] for i, row in enumerate(matrix)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the streaming COCO-17 activity GRU")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/train.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(cfg["seed"]))
    data = load_pose_archive(args.data, int(cfg["temporal_stride"]))
    train_i, val_i, test_i = group_safe_split(
        data.y, data.groups, cfg["split"], int(cfg["seed"])
    )
    device = choose_device(args.device)
    print(f"device={device}; input_shape={data.x.shape}; source={args.data}")

    train_loader = make_loader(data.x, data.y, train_i, cfg["batch_size"], True, cfg["num_workers"])
    val_loader = make_loader(data.x, data.y, val_i, cfg["batch_size"], False, cfg["num_workers"])
    test_loader = make_loader(data.x, data.y, test_i, cfg["batch_size"], False, cfg["num_workers"])

    model_config = {
        "num_classes": len(data.classes),
        "input_size": 51,
        "hidden_size": int(cfg["hidden_size"]),
        "num_layers": int(cfg["num_layers"]),
        "dropout": float(cfg["dropout"]),
    }
    model = StreamingGRU(**model_config).to(device)
    weights = class_weights(data.y, train_i, data.classes, cfg["class_weights"])
    print("class_weights=" + json.dumps(dict(zip(data.classes, weights.tolist()))))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler_cfg = cfg["lr_scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(scheduler_cfg["factor"]),
        patience=int(scheduler_cfg["patience"]),
        min_lr=float(scheduler_cfg["minimum_lr"]),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "splits.npz", train=train_i, val=val_i, test=test_i)
    split_manifest = {
        name: data.sample_ids[index].tolist()
        for name, index in (("train", train_i), ("val", val_i), ("test", test_i))
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []
    total_epochs = int(args.epochs or cfg["epochs"])
    for epoch in range(1, total_epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_x.to(device))
            loss = criterion(logits, batch_y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]))
            optimizer.step()
            losses.append(loss.item())

        val_y, val_pred = predict(model, val_loader, device)
        score = f1_score(val_y, val_pred, average="macro", zero_division=0)
        learning_rate = optimizer.param_groups[0]["lr"]
        mean_loss = float(np.mean(losses))
        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "val_macro_f1": float(score),
            "learning_rate": learning_rate,
        })
        print(
            f"epoch={epoch} loss={mean_loss:.5f} "
            f"val_macro_f1={score:.5f} lr={learning_rate:.7f}"
        )

        if score > best_f1 + float(cfg["early_stopping_min_delta"]):
            best_f1 = float(score)
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "classes": data.classes,
                    "input_shape": [None, int(data.x.shape[1]), 51],
                    "feature_layout": "channel-major: x[17], y[17], confidence[17]",
                    "temporal_stride": int(cfg["temporal_stride"]),
                    "source_pose_hz": float(cfg["source_pose_hz"]),
                    "effective_pose_hz": float(cfg["source_pose_hz"]) / int(cfg["temporal_stride"]),
                    "preprocessing": {
                        "mask_coco_indices": [1, 2, 3, 4],
                        "binary_confidence": True,
                        "centre": "hip midpoint with shoulder/centroid fallback",
                        "scale": "shoulder-to-hip torso with visible-extent fallback",
                    },
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "training_config": cfg,
                },
                args.output / "best.pt",
            )
        else:
            epochs_without_improvement += 1
        scheduler.step(score)
        if epochs_without_improvement >= int(cfg["early_stopping_patience"]):
            print(f"early_stopping epoch={epoch}")
            break

    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_y, test_pred = predict(model, test_loader, device)
    report = classification_report(
        test_y,
        test_pred,
        labels=range(len(data.classes)),
        target_names=data.classes,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "best_epoch": checkpoint["epoch"],
        "best_val_macro_f1": checkpoint["val_macro_f1"],
        "per_class": {name: report[name] for name in data.classes},
        "split_samples": {"train": len(train_i), "val": len(val_i), "test": len(test_i)},
        "split_groups": {
            "train": len(set(data.groups[train_i])),
            "val": len(set(data.groups[val_i])),
            "test": len(set(data.groups[test_i])),
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss", "val_macro_f1", "learning_rate"])
        writer.writeheader()
        writer.writerows(history)
    write_confusion(
        args.output / "confusion_matrix.csv",
        confusion_matrix(test_y, test_pred, labels=range(len(data.classes))),
        data.classes,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
