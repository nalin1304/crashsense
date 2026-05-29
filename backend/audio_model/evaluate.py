"""
Held-out test set evaluation + calibration report.

Produces:
- Per-class accuracy / precision / recall / F1
- Confusion matrix
- Per-source-collection breakdown (ESC-50, UrbanSound8K, Freesound)
- Reliability diagram bins + Expected Calibration Error (ECE)
- Optional matplotlib plots saved to docs/figures/

Usage:
    python -m backend.audio_model.evaluate
    python -m backend.audio_model.evaluate --plots
    python -m backend.audio_model.evaluate --json reports/test_report.json
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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .data_split import load_split
from .model import build_model

LOG = logging.getLogger("evaluate")

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECTROGRAM_ROOT = REPO_ROOT / "data" / "spectrograms"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "crash_detector.pth"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_CALIBRATION_BINS = 10


def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class _EvalDataset(torch.utils.data.Dataset):
    """ImageFolder-style dataset for evaluation that exposes the source path."""

    def __init__(self, base: ImageFolder, indices: list[int], transform: transforms.Compose):
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        path, label = self.base.samples[self.indices[idx]]
        from PIL import Image
        with Image.open(path) as pil:
            pil = pil.convert("RGB")
            return self.transform(pil), label, path


def _source_collection(path: str) -> str:
    name = Path(path).name
    if name.startswith("freesound_"):
        return "freesound"
    if name.startswith("us8k_"):
        return "urbansound8k"
    if name.startswith("esc_"):
        return "esc50"
    return "other"


def _load_model_from_checkpoint(checkpoint: Path) -> tuple[torch.nn.Module, str, float]:
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    backbone = payload.get("backbone", "resnet18")
    temperature = float(payload.get("temperature", 1.0))
    model = build_model(backbone)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, backbone, temperature


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append({
            "class": c,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    return {
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "accuracy": accuracy,
    }


def _calibration(y_true: np.ndarray, confidences: np.ndarray, predictions: np.ndarray) -> dict:
    """Compute Expected Calibration Error (ECE) + per-bin reliability data.

    Bins predictions by their max-class confidence and measures the gap
    between average confidence and average accuracy in each bin.
    """
    bin_edges = np.linspace(0.0, 1.0, N_CALIBRATION_BINS + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1])

    bins = []
    n = len(y_true)
    ece = 0.0
    for b in range(N_CALIBRATION_BINS):
        mask = bin_indices == b
        count = int(mask.sum())
        if count == 0:
            bins.append({
                "bin_low": float(bin_edges[b]),
                "bin_high": float(bin_edges[b + 1]),
                "count": 0,
                "avg_confidence": None,
                "accuracy": None,
                "gap": 0.0,
            })
            continue
        avg_conf = float(confidences[mask].mean())
        accuracy = float((y_true[mask] == predictions[mask]).mean())
        gap = abs(avg_conf - accuracy)
        ece += (count / n) * gap
        bins.append({
            "bin_low": float(bin_edges[b]),
            "bin_high": float(bin_edges[b + 1]),
            "count": count,
            "avg_confidence": avg_conf,
            "accuracy": accuracy,
            "gap": gap,
        })
    return {"ece": float(ece), "bins": bins, "n_samples": n}


def _per_source_breakdown(y_true: np.ndarray, y_pred: np.ndarray,
                           paths: list[str], n_classes: int) -> dict:
    by_collection: dict[str, dict] = defaultdict(lambda: {"counts": np.zeros((n_classes, n_classes), dtype=int)})
    for t, p, path in zip(y_true, y_pred, paths):
        col = _source_collection(path)
        by_collection[col]["counts"][t, p] += 1

    out = {}
    for col, data in by_collection.items():
        cm = data["counts"]
        total = int(cm.sum())
        correct = int(np.trace(cm))
        out[col] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "confusion_matrix": cm.tolist(),
        }
    return out


def evaluate(checkpoint: Path = CHECKPOINT_PATH) -> dict:
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    model, backbone, temperature = _load_model_from_checkpoint(checkpoint)
    LOG.info("loaded checkpoint: backbone=%s temperature=%.3f", backbone, temperature)

    _train_idx, _val_idx, test_idx, samples, classes = load_split()
    base = ImageFolder(str(SPECTROGRAM_ROOT))
    test_ds = _EvalDataset(base, test_idx, _eval_transform())
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    LOG.info("evaluating on %d held-out spectrograms (%s)",
             len(test_ds), ", ".join(classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_y, all_pred, all_conf, all_logits, all_paths = [], [], [], [], []
    with torch.no_grad():
        for x, y, paths in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=1)
            confs, preds = probs.max(dim=1)
            all_y.append(y.numpy())
            all_pred.append(preds.cpu().numpy())
            all_conf.append(confs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_paths.extend(paths)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    confidences = np.concatenate(all_conf)
    logits = np.concatenate(all_logits)

    cls_metrics = _classification_metrics(y_true, y_pred, n_classes=len(classes))
    cal = _calibration(y_true, confidences, y_pred)
    per_source = _per_source_breakdown(y_true, y_pred, all_paths, n_classes=len(classes))

    return {
        "checkpoint": str(checkpoint),
        "backbone": backbone,
        "n_test": int(len(y_true)),
        "classes": classes,
        "metrics": cls_metrics,
        "calibration": cal,
        "per_source": per_source,
        # Keep raw arrays accessible to the calibration plot helper
        "_arrays": {
            "y_true": y_true,
            "y_pred": y_pred,
            "confidences": confidences,
            "logits": logits,
        },
    }


def _format_report(report: dict) -> str:
    classes = report["classes"]
    cm = report["metrics"]["confusion_matrix"]
    lines = [
        "============================================================",
        " CrashSense — Held-Out Test Set Evaluation",
        "============================================================",
        f" Checkpoint: {report['checkpoint']}",
        f" Backbone:   {report['backbone']}",
        f" Test set:   {report['n_test']} spectrograms (no source overlap with train/val)",
        "",
        f" Overall accuracy: {report['metrics']['accuracy']*100:6.2f}%",
        "",
        " Confusion matrix (rows=true, cols=pred):",
        "                  " + "".join(f"{c:>10s}" for c in classes),
    ]
    for i, c in enumerate(classes):
        row = "".join(f"{cm[i][j]:>10d}" for j in range(len(classes)))
        lines.append(f"   true={c:<8s}{row}")
    lines.append("")
    lines.append(" Per-class metrics:")
    lines.append(f"   {'class':<10s}{'precision':>12s}{'recall':>10s}{'f1':>8s}{'support':>10s}")
    for c, m in zip(classes, report["metrics"]["per_class"]):
        support = m["tp"] + m["fn"]
        lines.append(
            f"   {c:<10s}{m['precision']:>12.4f}{m['recall']:>10.4f}{m['f1']:>8.4f}{support:>10d}"
        )
    lines.append("")
    lines.append(" Per-source-collection accuracy:")
    for col, data in sorted(report["per_source"].items()):
        lines.append(f"   {col:<14s}{data['correct']}/{data['total']} = {data['accuracy']*100:.2f}%")
    lines.append("")
    cal = report["calibration"]
    lines.append(f" Expected Calibration Error: {cal['ece']*100:.2f}% over {cal['n_samples']} samples")
    lines.append(" Reliability diagram (10 bins):")
    lines.append(f"   {'bin':>14s}{'count':>8s}{'avg_conf':>12s}{'accuracy':>12s}{'gap':>10s}")
    for b in cal["bins"]:
        lo, hi = b["bin_low"], b["bin_high"]
        if b["count"] == 0:
            lines.append(f"   {lo:.2f}-{hi:.2f}     {0:>5d}     ---         ---       0.00")
            continue
        lines.append(
            f"   {lo:.2f}-{hi:.2f}     {b['count']:>5d}    {b['avg_confidence']:>9.4f}"
            f"   {b['accuracy']:>9.4f}  {b['gap']:>7.4f}"
        )
    lines.append("============================================================")
    return "\n".join(lines)


def _save_plots(report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Reliability diagram
    bins = report["calibration"]["bins"]
    centers = [(b["bin_low"] + b["bin_high"]) / 2 for b in bins]
    accs = [b["accuracy"] if b["accuracy"] is not None else 0 for b in bins]
    confs = [b["avg_confidence"] if b["avg_confidence"] is not None else 0 for b in bins]
    counts = [b["count"] for b in bins]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]})
    bar_w = 0.09
    ax1.bar(centers, accs, width=bar_w, label="actual accuracy",
            edgecolor="white", color="#22d3ee")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    for cx, conf in zip(centers, confs):
        if conf:
            ax1.plot([cx, cx], [0, conf], color="#fb7185", linewidth=2, alpha=0.7)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("predicted confidence")
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"Reliability diagram — ECE={report['calibration']['ece']*100:.2f}%")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.bar(centers, counts, width=bar_w, color="#475569", edgecolor="white")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("confidence bin")
    ax2.set_ylabel("# samples")
    ax2.set_title("Sample distribution per confidence bin")
    fig.tight_layout()
    out_rel = FIGURES_DIR / "reliability_diagram.png"
    fig.savefig(str(out_rel), dpi=120)
    plt.close(fig)
    LOG.info("wrote %s", out_rel)

    # Confusion matrix heatmap
    classes = report["classes"]
    cm = np.array(report["metrics"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="magma")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Held-out confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] < cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    out_cm = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(str(out_cm), dpi=120)
    plt.close(fig)
    LOG.info("wrote %s", out_cm)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate CrashSense audio model on held-out test set")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional JSON report path")
    parser.add_argument("--plots", action="store_true",
                        help="Save reliability diagram + confusion matrix PNGs to docs/figures/")
    args = parser.parse_args(argv)

    report = evaluate(args.checkpoint)
    print(_format_report(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        # Strip the numpy arrays before serializing
        serializable = {k: v for k, v in report.items() if k != "_arrays"}
        args.json.write_text(json.dumps(serializable, indent=2))
        LOG.info("wrote %s", args.json)

    if args.plots:
        _save_plots(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
