"""
Trainer for the CrashSense Audio_Detector (Requirement 3).

- Loads spectrograms from data/spectrograms/ via ImageFolder.
- 80/20 deterministic train/val split using torch.utils.data.random_split (seed=42).
- Augments train only:
    - RandomHorizontalFlip(0.5)  — temporal flip
    - ColorJitter(0.2, 0.2, 0.2, 0.05)
    - SpecAugment-style frequency + time masking
- Adam(lr=1e-4, wd=1e-4), CrossEntropyLoss, 20 epochs (configurable).
- Saves the best checkpoint to backend/audio_model/crash_detector.pth after
  every epoch where validation accuracy improves, so partial training is
  recoverable.
- If best val acc < 80%, falls back to timm EfficientNet-B0 once.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ._eval_helpers import compute_checkpoint_sha256
from .data_split import load_split
from .model import build_model
from .snr_mix_dataset import (
    DEFAULT_AMBIENT_MIX_PROBABILITY,
    DEFAULT_MIX_PROBABILITY,
    DEFAULT_SNR_CHOICES,
    AMBIENT_MIX_PROBABILITY_MAX,
    AMBIENT_MIX_PROBABILITY_MIN,
    SnrMixDataset,
)

LOG = logging.getLogger("train")

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECTROGRAM_ROOT = REPO_ROOT / "data" / "spectrograms"
RAW_AUDIO_ROOT = REPO_ROOT / "data" / "raw_audio"
AMBIENT_ROOT = REPO_ROOT / "data" / "ambient_long"
AMBIENT_MANIFEST = AMBIENT_ROOT / "manifest.json"

#: Phase 1 baseline checkpoint path. Preserved for backward compatibility:
#: when ``--snr-mix`` is NOT enabled the trainer continues to write here.
CHECKPOINT_PATH = Path(__file__).resolve().parent / "crash_detector.pth"

#: Phase 2 SNR-mixed checkpoints land in a parallel namespace alongside
#: the existing ``<arch>_v<n>_<tag>.pth`` checkpoints (R2.6, task 2.4).
CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"

SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4

# Mixup: blend two batch samples with weight ~ Beta(alpha, alpha). 0.0 disables.
MIXUP_ALPHA = 0.2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = MIXUP_ALPHA):
    """Apply mixup to a batch. Returns (mixed_x, y_a, y_b, lam)."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    lam = max(lam, 1.0 - lam)  # bias toward the dominant sample to keep gradients stable
    perm = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1.0 - lam) * x[perm]
    return mixed, y, y[perm], lam


def mixup_loss(criterion, logits, y_a, y_b, lam: float):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# Focal loss focuses gradient on poorly-classified samples. gamma=2 is the
# value from Lin et al., 2017. Useful here because hard negatives (siren,
# jackhammer) are otherwise easy enough to coast through with CE loss.
FOCAL_GAMMA = 2.0


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = FOCAL_GAMMA, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_p = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_p = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_w = (1 - target_p).pow(self.gamma)
        loss = -focal_w * target_log_p
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


class SpecAugment(nn.Module):
    """SpecAugment frequency + time masking applied to a CHW tensor.

    Park et al., 2019 (https://arxiv.org/abs/1904.08779). On a 224x224 mel
    image we treat axis 1 as frequency and axis 2 as time, mask out
    horizontal stripes (frequency masks) and vertical stripes (time masks)
    by setting the corresponding pixels to zero.
    """

    def __init__(
        self,
        freq_mask_param: int = 24,
        time_mask_param: int = 24,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        prob: float = 0.7,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.prob:
            return x
        c, h, w = x.shape
        for _ in range(self.n_freq_masks):
            f = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
            if f > 0 and h - f > 0:
                f0 = int(torch.randint(0, h - f, (1,)).item())
                x[:, f0:f0 + f, :] = 0.0
        for _ in range(self.n_time_masks):
            t = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
            if t > 0 and w - t > 0:
                t0 = int(torch.randint(0, w - t, (1,)).item())
                x[:, :, t0:t0 + t] = 0.0
        return x


def _make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        SpecAugment(freq_mask_param=24, time_mask_param=24,
                    n_freq_masks=2, n_time_masks=2, prob=0.7),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, val


class _SubsetWithTransform(torch.utils.data.Dataset):
    """Wrap a Subset so each split can apply a different transform."""

    def __init__(self, subset: torch.utils.data.Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform
        self.dataset = subset.dataset

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img, label = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image
        with Image.open(img) as pil:
            pil = pil.convert("RGB")
            return self.transform(pil), label


def _build_loaders(
    snr_mix: bool = False,
    snr_noise_stem: Path | None = None,
    ambient_mix: bool = False,
    ambient_manifest: Path | None = None,
    ambient_root: Path | None = None,
    ambient_mix_probability: float = DEFAULT_AMBIENT_MIX_PROBABILITY,
) -> tuple[DataLoader, DataLoader, list[str]]:
    if not SPECTROGRAM_ROOT.exists():
        raise FileNotFoundError(
            f"Spectrogram root missing: {SPECTROGRAM_ROOT}. "
            "Run dataset_prep.py and spectrogram_gen.py first."
        )
    train_idx, val_idx, _test_idx, samples, classes = load_split()

    base = ImageFolder(str(SPECTROGRAM_ROOT))
    train_subset = torch.utils.data.Subset(base, train_idx)
    val_subset = torch.utils.data.Subset(base, val_idx)

    train_tx, val_tx = _make_transforms()
    # ``--ambient-mix`` composes cleanly with ``--snr-mix``: when both are
    # set the SnrMixDataset draws independent Bernoulli triggers per
    # sample and ambient takes precedence when both fire (R6.3). When
    # ``--ambient-mix`` is set without ``--snr-mix`` we still wrap with
    # SnrMixDataset because the wrapper is the single owner of the WAV
    # decode + SNR-scaled mixing path; we just hold the synthetic
    # mix_probability at zero so only the ambient trigger can fire.
    use_wrapper = bool(snr_mix or ambient_mix)
    if use_wrapper:
        synth_mix_prob = (
            DEFAULT_MIX_PROBABILITY if snr_mix else 0.0
        )
        train_ds: torch.utils.data.Dataset = SnrMixDataset(
            train_subset,
            train_transform=train_tx,
            spec_root=SPECTROGRAM_ROOT,
            raw_root=RAW_AUDIO_ROOT,
            noise_path=snr_noise_stem,
            snr_choices=DEFAULT_SNR_CHOICES,
            mix_probability=synth_mix_prob,
            seed=SEED,
            ambient_mix=ambient_mix,
            ambient_manifest=ambient_manifest,
            ambient_root=ambient_root,
            ambient_mix_probability=ambient_mix_probability,
        )
        if snr_mix:
            LOG.info(
                "snr_mix enabled: p=%.2f, snr_choices=%s, noise_stem=%s",
                synth_mix_prob,
                DEFAULT_SNR_CHOICES,
                "synthetic" if snr_noise_stem is None else str(snr_noise_stem),
            )
        if ambient_mix:
            # ``train_ds.ambient_mix`` reflects the post-load decision: the
            # dataset disables the ambient path when the manifest yields
            # zero non-placeholder sessions and emits an INFO line of its
            # own ("no real ambient sessions available, skipping ambient
            # mix"). Mirror that signal in the trainer log so a user
            # tailing train.py output sees the fall-back too.
            if getattr(train_ds, "ambient_mix", False):
                LOG.info(
                    "ambient_mix enabled: p=%.2f, manifest=%s, sessions=%d",
                    ambient_mix_probability,
                    ambient_manifest,
                    len(getattr(train_ds, "_ambient_sessions", [])),
                )
            else:
                LOG.info(
                    "ambient_mix requested but no real ambient sessions "
                    "available (manifest=%s); falling back to synthetic-only",
                    ambient_manifest,
                )
    else:
        train_ds = _SubsetWithTransform(train_subset, train_tx)
    val_ds = _SubsetWithTransform(val_subset, val_tx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, classes


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += y.numel()
    return 0.0 if total == 0 else 100.0 * correct / total


def train_once(
    backbone: str,
    epochs: int = EPOCHS,
    snr_mix: bool = False,
    snr_noise_stem: Path | None = None,
    save_path: Path | None = None,
    ambient_mix: bool = False,
    ambient_manifest: Path | None = None,
    ambient_root: Path | None = None,
    ambient_mix_probability: float = DEFAULT_AMBIENT_MIX_PROBABILITY,
) -> tuple[float, dict]:
    """Train one full schedule. Returns (best_val_acc_percent, best_state_dict).

    Saves the best checkpoint after every improving epoch.

    ``save_path`` controls where the per-epoch best snapshot is written. When
    omitted, defaults to the legacy ``crash_detector.pth`` (Phase 1 baseline).
    For Phase 2 ``--snr-mix`` runs the caller resolves the SNR target path via
    :func:`resolve_snr_checkpoint_path` and passes it here so improving epochs
    don't clobber the baseline checkpoint mid-run (R2.6).

    ``ambient_mix`` enables R6.3: with per-batch probability
    ``ambient_mix_probability`` (default 0.3, range
    ``[AMBIENT_MIX_PROBABILITY_MIN, AMBIENT_MIX_PROBABILITY_MAX]``), the
    SnrMixDataset swaps the noise stem from synthetic highway noise to a
    randomly-selected non-placeholder session from
    ``ambient_manifest``. Composes with ``snr_mix`` cleanly — when both
    flags are set, ambient takes precedence on a per-sample basis.
    """
    train_loader, val_loader, classes = _build_loaders(
        snr_mix=snr_mix,
        snr_noise_stem=snr_noise_stem,
        ambient_mix=ambient_mix,
        ambient_manifest=ambient_manifest,
        ambient_root=ambient_root,
        ambient_mix_probability=ambient_mix_probability,
    )
    LOG.info("classes=%s | train_batches=%d val_batches=%d", classes, len(train_loader), len(val_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)

    best_acc = 0.0
    best_state: dict = {}
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            x_mixed, y_a, y_b, lam = mixup_batch(x, y, alpha=MIXUP_ALPHA)
            logits = model(x_mixed)
            loss = mixup_loss(criterion, logits, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
        val_acc = _evaluate(model, val_loader, device)
        LOG.info("epoch %02d/%02d val_acc=%.2f%%", epoch, epochs, val_acc)
        print(f"epoch {epoch:02d}/{epochs} val_acc={val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(best_state, backbone, best_acc, path=save_path)
    return best_acc, best_state


def save_checkpoint(
    state: dict,
    backbone: str,
    val_acc: float,
    path: Path | None = None,
) -> None:
    """Save a checkpoint dict to ``path`` (default: legacy baseline).

    ``path`` defaults to :data:`CHECKPOINT_PATH` so non-SNR runs continue to
    write the Phase 1 baseline at ``backend/audio_model/crash_detector.pth``.
    SNR runs override via :func:`resolve_snr_checkpoint_path`.
    """
    target = path or CHECKPOINT_PATH
    payload = {
        "state_dict": state,
        "backbone": backbone,
        "val_acc": val_acc,
        "num_classes": 2,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(target))
    LOG.info("saved %s (val_acc=%.2f%%, backbone=%s)", target, val_acc, backbone)


# ---------------------------------------------------------------------------
# Phase 2 SNR-mixed checkpoint helpers (R2.6, task 2.4)
# ---------------------------------------------------------------------------


def resolve_git_sha() -> str:
    """Return ``git rev-parse --short HEAD`` or ``"unknown"`` on any failure.

    Pure stdlib subprocess; the trainer must keep running outside a git
    checkout (e.g. CI tarball builds) so we swallow every failure mode and
    fall back to a deterministic placeholder.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    sha = result.stdout.strip()
    return sha or "unknown"


def resolve_snr_checkpoint_path(arch: str, git_sha: str) -> Path:
    """Build the Phase 2 SNR-mixed checkpoint target path.

    Convention: ``backend/audio_model/checkpoints/<arch>_snr_<git_sha>.pth``
    (design §3.2, R2.6). Returned path may not yet exist; callers that
    intend to write should first invoke :func:`assert_distinct_from_baseline`
    and the refuse-to-overwrite check in :func:`save_snr_checkpoint`.
    """
    if not arch:
        raise ValueError("arch must be a non-empty backbone name")
    if not git_sha:
        raise ValueError("git_sha must be a non-empty string (use 'unknown' as fallback)")
    return CHECKPOINTS_DIR / f"{arch}_snr_{git_sha}.pth"


def assert_distinct_from_baseline(snr_path: Path) -> None:
    """Refuse to overwrite the Phase 1 baseline checkpoint (R2.6).

    Resolves both paths to absolute form and asserts they're distinct. By
    construction (``checkpoints/<arch>_snr_<sha>.pth`` vs
    ``crash_detector.pth``) they always will be — this assertion is a
    defensive guardrail in case future refactors collapse the directories.
    """
    snr_resolved = Path(snr_path).resolve()
    baseline_resolved = CHECKPOINT_PATH.resolve()
    assert snr_resolved != baseline_resolved, (
        f"refusing to overwrite Phase 1 baseline checkpoint at {baseline_resolved}; "
        f"resolved SNR target {snr_resolved} collided with the baseline path"
    )


def save_snr_checkpoint(
    state: dict,
    backbone: str,
    val_acc: float,
    *,
    git_sha: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist an SNR-mixed checkpoint and emit its sha256 to the training log.

    Implements R2.6 + task 2.4:

    * Resolve target as ``checkpoints/<arch>_snr_<git_sha>.pth``.
    * Assert distinct from the Phase 1 baseline path.
    * When ``overwrite`` is ``False`` (default) and the SNR target already
      exists, refuse to overwrite — print an actionable error to stderr and
      raise ``FileExistsError`` so the CLI can exit non-zero.
    * Write the payload, compute its sha256 via
      :func:`compute_checkpoint_sha256`, and emit ``saved <path>
      (sha256=<first16>...)`` at INFO.

    The ``overwrite`` flag is set to ``True`` by the trainer's per-epoch
    "improving validation accuracy" path, which legitimately rewrites the
    same target multiple times within a single run; the up-front guard in
    :func:`main` covers the cross-run case.

    Returns the resolved target path.
    """
    sha = git_sha if git_sha is not None else resolve_git_sha()
    target = resolve_snr_checkpoint_path(backbone, sha)
    assert_distinct_from_baseline(target)

    if target.exists() and not overwrite:
        msg = (
            f"ERROR: refusing to overwrite existing SNR checkpoint at {target}. "
            f"Move or delete the existing file, or re-run with a different "
            f"--git-sha override."
        )
        print(msg, file=sys.stderr)
        raise FileExistsError(msg)

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": state,
        "backbone": backbone,
        "val_acc": val_acc,
        "num_classes": 2,
    }
    torch.save(payload, str(target))

    sha256 = compute_checkpoint_sha256(target)
    LOG.info("saved %s (sha256=%s...)", target, sha256[:16])
    return target


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train the CrashSense audio classifier")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--snr-mix",
        action="store_true",
        help=(
            "Augment crash training samples with highway road noise mixed at a "
            "uniformly random SNR drawn from {0, 10, 20} dB, applied with "
            "probability 0.5 per sample (R2.3)."
        ),
    )
    parser.add_argument(
        "--snr-noise-stem",
        type=Path,
        default=None,
        help=(
            "Optional path to a real highway-noise WAV file used as the noise "
            "stem for --snr-mix. When omitted the trainer synthesizes highway "
            "noise per sample. The stem is tiled or cropped to match each clean "
            "clip's length (R2.4)."
        ),
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help=(
            "Override the git short SHA used in the SNR checkpoint filename "
            "(R2.6). Defaults to `git rev-parse --short HEAD`, falling back "
            "to 'unknown' on failure. Only consulted when --snr-mix is set."
        ),
    )
    parser.add_argument(
        "--ambient-mix",
        action="store_true",
        help=(
            "Sample negative-class noise stems from the long-stretch ambient "
            "corpus (data/ambient_long/) on a per-sample basis with "
            "probability --ambient-mix-prob (R6.3). Placeholder sessions "
            "(is_placeholder=true in the manifest) are skipped. When no real "
            "ambient session is available, an INFO log is emitted and the "
            "trainer falls back to the synthetic-only path. Composes with "
            "--snr-mix cleanly: each sample independently rolls for synth "
            "and ambient triggers, and ambient takes precedence when both "
            "fire."
        ),
    )
    parser.add_argument(
        "--ambient-mix-prob",
        type=float,
        default=DEFAULT_AMBIENT_MIX_PROBABILITY,
        help=(
            "Per-sample ambient sampling probability used by --ambient-mix "
            "(R6.3, range [%(min)s, %(max)s], default %(default)s)."
        ) % {
            "min": AMBIENT_MIX_PROBABILITY_MIN,
            "max": AMBIENT_MIX_PROBABILITY_MAX,
            "default": DEFAULT_AMBIENT_MIX_PROBABILITY,
        },
    )
    parser.add_argument(
        "--ambient-manifest",
        type=Path,
        default=AMBIENT_MANIFEST,
        help=(
            "Path to the ambient corpus manifest used by --ambient-mix "
            "(default: data/ambient_long/manifest.json)."
        ),
    )
    parser.add_argument(
        "--ambient-root",
        type=Path,
        default=AMBIENT_ROOT,
        help=(
            "Root directory for ambient WAV files referenced by the "
            "manifest (default: data/ambient_long/)."
        ),
    )
    args = parser.parse_args(argv)

    # Validate the ambient sampling probability up front so a bad value
    # fails fast at the CLI rather than mid-epoch (R6.3).
    if args.ambient_mix and not (
        AMBIENT_MIX_PROBABILITY_MIN
        <= args.ambient_mix_prob
        <= AMBIENT_MIX_PROBABILITY_MAX
    ):
        print(
            f"ERROR: --ambient-mix-prob must be in "
            f"[{AMBIENT_MIX_PROBABILITY_MIN}, "
            f"{AMBIENT_MIX_PROBABILITY_MAX}], got {args.ambient_mix_prob}",
            file=sys.stderr,
        )
        return 2

    # Phase 2 SNR-mixed runs persist to a parallel namespace and refuse to
    # overwrite the baseline (R2.6, task 2.4). Resolve the target once up
    # front so per-epoch saves and the final save share the same path, and
    # so we fail fast if the target file already exists.
    if args.snr_mix:
        git_sha = args.git_sha if args.git_sha is not None else resolve_git_sha()
        snr_path = resolve_snr_checkpoint_path(args.backbone, git_sha)
        assert_distinct_from_baseline(snr_path)
        if snr_path.exists():
            msg = (
                f"ERROR: refusing to overwrite existing SNR checkpoint at "
                f"{snr_path}. Move or delete the existing file, or re-run "
                f"with a different --git-sha override."
            )
            print(msg, file=sys.stderr)
            return 2
        per_epoch_save_path: Path | None = snr_path
    else:
        git_sha = None
        per_epoch_save_path = None  # legacy CHECKPOINT_PATH

    best_acc, best_state = train_once(
        args.backbone,
        args.epochs,
        snr_mix=args.snr_mix,
        snr_noise_stem=args.snr_noise_stem,
        save_path=per_epoch_save_path,
        ambient_mix=args.ambient_mix,
        ambient_manifest=args.ambient_manifest,
        ambient_root=args.ambient_root,
        ambient_mix_probability=args.ambient_mix_prob,
    )
    if args.snr_mix:
        save_snr_checkpoint(
            best_state, args.backbone, best_acc, git_sha=git_sha, overwrite=True
        )
    else:
        save_checkpoint(best_state, args.backbone, best_acc)

    if best_acc < 80.0 and args.backbone == "resnet18":
        LOG.warning("ResNet-18 best=%0.2f%% < 80%%. Falling back to EfficientNet-B0.", best_acc)
        if args.snr_mix:
            fb_path = resolve_snr_checkpoint_path("efficientnet_b0", git_sha)
            assert_distinct_from_baseline(fb_path)
            if fb_path.exists():
                msg = (
                    f"ERROR: refusing to overwrite existing SNR checkpoint at "
                    f"{fb_path}. Move or delete the existing file, or re-run "
                    f"with a different --git-sha override."
                )
                print(msg, file=sys.stderr)
                return 2
            fb_save_path: Path | None = fb_path
        else:
            fb_save_path = None
        fb_acc, fb_state = train_once(
            "efficientnet_b0",
            args.epochs,
            snr_mix=args.snr_mix,
            snr_noise_stem=args.snr_noise_stem,
            save_path=fb_save_path,
            ambient_mix=args.ambient_mix,
            ambient_manifest=args.ambient_manifest,
            ambient_root=args.ambient_root,
            ambient_mix_probability=args.ambient_mix_prob,
        )
        if args.snr_mix:
            save_snr_checkpoint(
                fb_state, "efficientnet_b0", fb_acc, git_sha=git_sha, overwrite=True
            )
        else:
            save_checkpoint(fb_state, "efficientnet_b0", fb_acc)
        best_acc = fb_acc

    LOG.info("training complete; best val_acc=%.2f%%", best_acc)
    return 0 if best_acc >= 80.0 else 2


if __name__ == "__main__":
    sys.exit(main())
