"""
Fine-tune Audio Spectrogram Transformer for CrashSense binary classification.

Strategy: freeze the AST backbone for a few warmup epochs (head-only) so the
fresh 2-class head doesn't blow up the pretrained features, then optionally
unfreeze for a small number of full fine-tune epochs.

Saves to backend/audio_model/crash_detector_ast.pth so the ResNet checkpoint
remains intact and switchable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

from .ast_dataset import build_loaders
from .ast_model import freeze_backbone, load_ast_model

LOG = logging.getLogger("train_ast")

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = Path(__file__).resolve().parent / "crash_detector_ast.pth"

LR_HEAD = 5e-4         # high LR for fresh head
LR_FULL = 5e-5         # gentle LR once the whole model is unlocked
WEIGHT_DECAY = 1e-4
HEAD_EPOCHS = 1        # warmup with backbone frozen
FULL_EPOCHS = 2        # full fine-tune
BATCH_SIZE = 8


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(input_values=x).logits
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += y.numel()
    return 0.0 if total == 0 else 100.0 * correct / total


def save_checkpoint(model, val_acc: float, classes: list[str], frozen: bool, temperature: float = 1.0) -> None:
    payload = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "backbone": "ast",
        "val_acc": val_acc,
        "classes": classes,
        "temperature": temperature,
        "head_only": frozen,
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(CHECKPOINT_PATH))
    LOG.info("saved %s (val_acc=%.2f%%, head_only=%s)", CHECKPOINT_PATH, val_acc, frozen)


def train(head_epochs: int = HEAD_EPOCHS, full_epochs: int = FULL_EPOCHS,
          batch_size: int = BATCH_SIZE) -> tuple[float, list[str]]:
    LOG.info("loading AST...")
    model, feat, _ = load_ast_model(num_classes=2)

    LOG.info("building data loaders...")
    train_loader, val_loader, _test_loader, classes = build_loaders(
        feat, batch_size=batch_size, augment_train=True,
    )
    LOG.info("classes=%s | train=%d val=%d", classes, len(train_loader.dataset), len(val_loader.dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    # ---------------------- Phase 1: head-only warmup --------------------
    if head_epochs > 0:
        LOG.info("[phase 1] freezing backbone, training head only at lr=%g", LR_HEAD)
        freeze_backbone(model, freeze=True)
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
        for epoch in range(1, head_epochs + 1):
            model.train()
            for batch_idx, (x, y, _) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(input_values=x).logits
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                if (batch_idx + 1) % 25 == 0:
                    LOG.info("  head ep %d batch %d/%d loss=%.4f",
                             epoch, batch_idx + 1, len(train_loader), float(loss))
            val = _evaluate(model, val_loader, device)
            LOG.info("[phase 1] epoch %d val_acc=%.2f%%", epoch, val)
            if val > best_acc:
                best_acc = val
                save_checkpoint(model, val, classes, frozen=True)

    # ---------------------- Phase 2: full fine-tune ----------------------
    if full_epochs > 0:
        LOG.info("[phase 2] unfreezing backbone, fine-tuning at lr=%g", LR_FULL)
        freeze_backbone(model, freeze=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY)
        for epoch in range(1, full_epochs + 1):
            model.train()
            for batch_idx, (x, y, _) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(input_values=x).logits
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                if (batch_idx + 1) % 25 == 0:
                    LOG.info("  full ep %d batch %d/%d loss=%.4f",
                             epoch, batch_idx + 1, len(train_loader), float(loss))
            val = _evaluate(model, val_loader, device)
            LOG.info("[phase 2] epoch %d val_acc=%.2f%%", epoch, val)
            if val > best_acc:
                best_acc = val
                save_checkpoint(model, val, classes, frozen=False)

    LOG.info("done. best val_acc=%.2f%%", best_acc)
    return best_acc, classes


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Fine-tune AST on CrashSense data")
    parser.add_argument("--head-epochs", type=int, default=HEAD_EPOCHS)
    parser.add_argument("--full-epochs", type=int, default=FULL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    train(args.head_epochs, args.full_epochs, args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
