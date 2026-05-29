"""
Cache AST embeddings for every clip in the dataset.

AST is too big to fine-tune on CPU. The standard alternative is to use it as
a frozen feature extractor: run each clip through AST once, save the
768-dim pooled embedding, then train a tiny linear or MLP head on the
cached embeddings. The head trains in seconds and can be retrained as
hyper-params change.

Output: data/ast_features.npz with arrays:
    embeddings: (N, 768) float32     — AST pooler output per clip
    labels:     (N,) int64           — class id (0=crash, 1=noise)
    paths:      (N,) str             — original wav path
    classes:    list[str]            — class names in order
    train_idx:  (n_train,) int64     — indices into embeddings for train split
    val_idx:    (n_val,) int64
    test_idx:   (n_test,) int64

Incremental mode: if --reuse-existing points at a previous .npz, every clip
already cached is reused as-is and only new clips run through AST.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .ast_dataset import build_loaders
from .ast_model import load_ast_model

LOG = logging.getLogger("ast_features")

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = REPO_ROOT / "data" / "ast_features.npz"


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    old = np.load(str(path), allow_pickle=True)
    cache: dict[str, np.ndarray] = {}
    for path_str, emb in zip(old["paths"], old["embeddings"]):
        cache[str(path_str)] = np.asarray(emb, dtype=np.float32)
    LOG.info("loaded %d cached embeddings from %s", len(cache), path)
    return cache


@torch.no_grad()
def extract_features(batch_size: int = 4, reuse_from: Path | None = None) -> dict:
    cache = _load_cache(reuse_from) if reuse_from else {}

    LOG.info("loading AST...")
    model, feat, _ = load_ast_model(num_classes=2)
    backbone = model.audio_spectrogram_transformer
    backbone.eval()

    train_loader, val_loader, test_loader, classes = build_loaders(
        feat, batch_size=batch_size, augment_train=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)

    embeddings_per_split: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    labels_per_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    paths_per_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for split_name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        n = len(loader.dataset)
        LOG.info("[%s] processing %d clips (cache hits possible: %d)", split_name, n, len(cache))
        t0 = time.monotonic()
        seen = 0
        new_count = 0

        for x, y, paths in loader:
            # Per-row decision: cache or forward.
            row_emb: list[np.ndarray] = [None] * len(paths)
            need_idx: list[int] = []
            need_x: list[torch.Tensor] = []

            for i, p in enumerate(paths):
                p_str = str(p)
                if p_str in cache:
                    row_emb[i] = cache[p_str]
                else:
                    need_idx.append(i)
                    need_x.append(x[i])

            if need_x:
                stack = torch.stack(need_x).to(device)
                outputs = backbone(input_values=stack)
                pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
                for slot, vec in zip(need_idx, pooled):
                    row_emb[slot] = vec
                    new_count += 1

            for i in range(len(paths)):
                embeddings_per_split[split_name].append(row_emb[i])
                labels_per_split[split_name].append(int(y[i]))
                paths_per_split[split_name].append(str(paths[i]))

            seen += len(paths)
            if seen % 200 < batch_size:
                rate = seen / max(1.0, time.monotonic() - t0)
                LOG.info("  [%s] %d/%d new=%d (%.1f clips/s)", split_name, seen, n, new_count, rate)

    all_emb = np.stack(
        embeddings_per_split["train"] + embeddings_per_split["val"] + embeddings_per_split["test"]
    )
    all_lab = np.array(
        labels_per_split["train"] + labels_per_split["val"] + labels_per_split["test"],
        dtype=np.int64,
    )
    all_paths = np.array(
        paths_per_split["train"] + paths_per_split["val"] + paths_per_split["test"]
    )
    n_train = len(embeddings_per_split["train"])
    n_val = len(embeddings_per_split["val"])
    n_test = len(embeddings_per_split["test"])
    train_idx = np.arange(n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test, dtype=np.int64)

    LOG.info(
        "embeddings: shape=%s | classes=%s | train=%d val=%d test=%d",
        all_emb.shape, classes, n_train, n_val, n_test,
    )
    return {
        "embeddings": all_emb,
        "labels": all_lab,
        "paths": all_paths,
        "classes": np.array(classes),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def save(features: dict, path: Path = FEATURES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **features)
    LOG.info("wrote %s (%.1f MB)", path, path.stat().st_size / 1024 / 1024)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Cache AST embeddings for all clips")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=FEATURES_PATH)
    parser.add_argument("--reuse-existing", type=Path, default=None,
                        help="Reuse embeddings from this previous .npz where paths overlap")
    args = parser.parse_args(argv)

    feats = extract_features(args.batch_size, reuse_from=args.reuse_existing)
    save(feats, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
