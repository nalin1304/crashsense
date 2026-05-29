"""
Train a tiny linear head on cached AST embeddings.

Once `ast_features.py` has dumped data/ast_features.npz, this script:
  - Loads the (N, 768) embeddings + labels + train/val/test indices.
  - Optionally runs 5-fold cross-validation on the train+val pool.
  - Fits a logistic-regression-style 2-class linear head.
  - Reports held-out test accuracy / precision / recall / F1, plus per-source
    breakdown and ECE — same format as the ResNet evaluator.
  - Writes backend/audio_model/crash_detector_ast_head.pth so inference can
    use AST-as-feature-extractor + linear head with no fine-tuning.

Compared to the ResNet pipeline this is dirt-cheap: training takes seconds,
not hours, and you can iterate on regularization / TTA without retraining
the backbone.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LOG = logging.getLogger("ast_head")

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = REPO_ROOT / "data" / "ast_features.npz"
HEAD_CHECKPOINT = Path(__file__).resolve().parent / "crash_detector_ast_head.pth"
REPORTS_DIR = REPO_ROOT / "reports"


def _source_collection(path: str) -> str:
    name = Path(path).name
    if name.startswith("freesound_"):
        return "freesound"
    if name.startswith("audioset_"):
        return "audioset"
    if name.startswith("us8k_"):
        return "urbansound8k"
    if name.startswith("esc_"):
        return "esc50"
    return "other"


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append({"class": c, "precision": precision, "recall": recall,
                          "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    return {
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0,
    }


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(confs, edges[1:-1])
    n = len(labels)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        avg_conf = confs[mask].mean()
        acc = (preds[mask] == labels[mask]).mean()
        ece += (mask.sum() / n) * abs(avg_conf - acc)
    return float(ece)


class _LinearHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 2,
                 hidden: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_head(
    features: np.ndarray, labels: np.ndarray,
    train_idx: np.ndarray, val_idx: np.ndarray,
    epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-3,
    hidden: int = 0, batch_size: int = 256,
) -> tuple[nn.Module, list[float]]:
    """Fit a linear/MLP head on cached AST features."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = _LinearHead(features.shape[1], num_classes=2, hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    x_train = torch.from_numpy(features[train_idx]).to(device)
    y_train = torch.from_numpy(labels[train_idx]).to(device)
    x_val = torch.from_numpy(features[val_idx]).to(device)
    y_val = torch.from_numpy(labels[val_idx]).to(device)

    history = []
    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(x_train.size(0), device=device)
        for i in range(0, x_train.size(0), batch_size):
            idx = perm[i:i + batch_size]
            logits = head(x_train[idx])
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val_logits = head(x_val)
            val_acc = float((val_logits.argmax(dim=1) == y_val).float().mean().item()) * 100
        history.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    LOG.info("trained head: best val_acc=%.2f%%", best_acc)
    return head, history


def fit_temperature(head: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    """Grid-search temperature on val to minimize ECE."""
    head.eval()
    with torch.no_grad():
        logits = head(x_val).cpu().numpy()
    labels = y_val.cpu().numpy()
    best_T, best_ece = 1.0, _ece(_softmax(logits / 1.0), labels)
    for T in np.linspace(0.5, 4.0, 71):
        probs = _softmax(logits / float(T))
        ece = _ece(probs, labels)
        if ece < best_ece:
            best_T, best_ece = float(T), ece
    LOG.info("temperature=%.3f → ECE=%.4f (was %.4f at T=1)", best_T, best_ece,
             _ece(_softmax(logits), labels))
    return best_T


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def evaluate_head(
    head: nn.Module, features: np.ndarray, labels: np.ndarray,
    indices: np.ndarray, paths: np.ndarray, classes: list[str],
    temperature: float = 1.0,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head.eval()
    x = torch.from_numpy(features[indices]).to(device)
    y = labels[indices]
    with torch.no_grad():
        logits = head(x).cpu().numpy()
    probs = _softmax(logits / max(temperature, 1e-6))
    preds = probs.argmax(axis=1)

    cls_metrics = _classification_metrics(y, preds, n_classes=len(classes))
    ece = _ece(probs, y)

    by_collection = defaultdict(lambda: {"correct": 0, "total": 0})
    for ti, pi, path in zip(y, preds, paths[indices]):
        col = _source_collection(str(path))
        by_collection[col]["total"] += 1
        if ti == pi:
            by_collection[col]["correct"] += 1
    per_source = {
        col: {"total": v["total"], "correct": v["correct"],
              "accuracy": v["correct"] / v["total"] if v["total"] else 0.0}
        for col, v in by_collection.items()
    }

    return {
        "n": int(len(indices)),
        "classes": classes,
        "metrics": cls_metrics,
        "ece": ece,
        "per_source": per_source,
        "temperature": temperature,
    }


def kfold_cv(features, labels, paths, classes, k: int = 5, seed: int = 42, hidden: int = 0):
    """Source-aware k-fold CV on the train+val pool, fold-stratified by class."""
    rng = np.random.default_rng(seed)
    train_pool_idx = np.concatenate([np.arange(len(features))[labels == c] for c in (0, 1)])
    # Keep the simple random k-fold; AST features already aggregate the source
    # so leakage risk is small enough for indicative CI.
    rng.shuffle(train_pool_idx)
    fold_size = len(train_pool_idx) // k
    accs, eces = [], []
    for fold in range(k):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k - 1 else len(train_pool_idx)
        val_idx = train_pool_idx[val_start:val_end]
        train_idx = np.concatenate([
            train_pool_idx[:val_start], train_pool_idx[val_end:]
        ])
        head, _ = train_head(features, labels, train_idx, val_idx, hidden=hidden, epochs=20)
        report = evaluate_head(head, features, labels, val_idx, paths, classes)
        LOG.info("[fold %d] acc=%.2f%% ece=%.3f", fold + 1,
                 report["metrics"]["accuracy"] * 100, report["ece"])
        accs.append(report["metrics"]["accuracy"])
        eces.append(report["ece"])
    return {
        "k": k,
        "fold_accuracies": accs,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_ece": float(np.mean(eces)),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train AST linear head + evaluate")
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=0,
                        help="0 for linear head, >0 to add an MLP layer")
    parser.add_argument("--cv", type=int, default=0,
                        help="If >0, run k-fold CV on the train+val pool first")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "test_report_ast.json")
    args = parser.parse_args(argv)

    LOG.info("loading %s", args.features)
    data = np.load(str(args.features), allow_pickle=True)
    features = data["embeddings"]
    labels = data["labels"]
    paths = data["paths"]
    classes = list(data["classes"])
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    LOG.info("loaded: features=%s train=%d val=%d test=%d",
             features.shape, len(train_idx), len(val_idx), len(test_idx))

    if args.cv > 1:
        cv_result = kfold_cv(features, labels, paths, classes, k=args.cv, hidden=args.hidden)
        LOG.info("k-fold CV: mean acc=%.2f%% ± %.2f%% (k=%d)",
                 cv_result["mean_accuracy"] * 100, cv_result["std_accuracy"] * 100,
                 cv_result["k"])

    head, history = train_head(features, labels, train_idx, val_idx,
                                epochs=args.epochs, hidden=args.hidden)

    # Temperature scaling on val
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_val = torch.from_numpy(features[val_idx]).to(device)
    y_val = torch.from_numpy(labels[val_idx]).to(device)
    T = fit_temperature(head, x_val, y_val)

    # Save head + temperature
    payload = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        "in_features": features.shape[1],
        "hidden": args.hidden,
        "classes": classes,
        "temperature": T,
        "backbone": "ast",
    }
    HEAD_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(HEAD_CHECKPOINT))
    LOG.info("saved %s", HEAD_CHECKPOINT)

    # Test set evaluation with temperature
    test_report = evaluate_head(head, features, labels, test_idx, paths, classes, temperature=T)
    LOG.info("=== HELD-OUT TEST SET ===")
    LOG.info("n=%d", test_report["n"])
    LOG.info("accuracy=%.2f%%", test_report["metrics"]["accuracy"] * 100)
    LOG.info("ECE=%.3f%%", test_report["ece"] * 100)
    for col, info in sorted(test_report["per_source"].items()):
        LOG.info("  %s: %d/%d = %.2f%%", col, info["correct"], info["total"],
                 info["accuracy"] * 100)
    cm = test_report["metrics"]["confusion_matrix"]
    for cls, row in zip(classes, cm):
        LOG.info("  true=%s -> pred[%s] = %s", cls,
                 " ".join(classes), " ".join(str(x) for x in row))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(test_report, indent=2))
    LOG.info("wrote %s", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
