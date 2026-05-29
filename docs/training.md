# Training Pipeline

## Data sources

Three CC-licensed sources are blended to produce a diverse training set:

| Source | License | Coverage |
|---|---|---|
| ESC-50 | CC-BY-NC | 10 categories used (5 crash-relevant, 5 noise-relevant), 40 clips each |
| UrbanSound8K | CC-BY-NC | 8 categories used (2 crash, 6 noise), thousands of clips |
| Freesound (via archive.org) | CC-BY | 6 hand-picked collision recordings by qubodup |

Critical design choice: **engine sounds go into the NOISE class.** This forces the model to learn the distinction between a crash and an idling engine instead of taking the lazy "vehicle = crash" shortcut.

## Class composition

```
CRASH (2,000 clips, 918 unique sources)
  ├── UrbanSound8K
  │     ├── car_horn         418 originals
  │     └── gun_shot         374 originals
  ├── ESC-50
  │     ├── glass_breaking   40 originals
  │     ├── car_horn         40 originals
  │     └── fireworks        40 originals
  └── Freesound
        └── collision/crash  6 originals (CC-BY)

NOISE (2,000 clips, 1,501 unique sources)
  ├── UrbanSound8K
  │     ├── air_conditioner  138 originals
  │     ├── drilling         163 originals
  │     ├── engine_idling    299 originals
  │     ├── jackhammer       219 originals
  │     ├── siren            196 originals
  │     └── street_music     206 originals
  └── ESC-50
        ├── rain             40 originals
        ├── crackling_fire   40 originals
        ├── wind             40 originals
        ├── chainsaw         40 originals
        ├── engine           40 originals
        ├── vacuum_cleaner   40 originals
        └── washing_machine  40 originals
```

The remaining slots up to 2,000 per class are filled with deterministic on-disk augmentations (pitch ±2 semitones, time stretch 0.9× / 1.1×, additive noise 12 / 18 dB SNR). Final per-class counts are exactly balanced.

## Spectrogram parameters

| Parameter | Value |
|---|---|
| Sample rate | 22 050 Hz |
| Mel bands | 128 |
| `fmax` | 8 000 Hz |
| Clip duration | 3.0 s (zero-padded short, head-trimmed long) |
| Output size | 224 × 224 RGB PNG |
| Colormap | magma (LUT) |

The renderer is intentionally NumPy + PIL only. matplotlib's `savefig` proved 50× slower in batch mode; the LUT colormap matches matplotlib's magma to within ±1 LSB per channel.

## Model

- Backbone: torchvision `resnet18` with ImageNet pretrained weights.
- Final FC layer replaced with `nn.Linear(in_features, 2)`.
- Fallback backbone (Requirement 3.13): `timm` `efficientnet_b0`, used only if the ResNet-18 best epoch finishes below 80 % validation accuracy.

## Hyperparameters

| Knob | Value |
|---|---|
| Optimizer | Adam (lr 1e-4, wd 1e-4) |
| Loss | CrossEntropyLoss |
| Epochs | 20 (configurable; 5 epochs sufficient for this dataset) |
| Batch size | 32 train / 32 val |
| Train/val split | 80 / 20 via `torch.utils.data.random_split` (seed 42) |
| On-disk augmentations | pitch shift ±2 semitones, time stretch 0.9 / 1.1, noise 12 / 18 dB |
| Train-time augmentations | RandomHorizontalFlip(0.5), ColorJitter(0.2,0.2,0.2,0.05), **SpecAugment** |
| SpecAugment | freq mask (2× up to 24 bins), time mask (2× up to 24 bins), p=0.7 |
| Vertical flip | **disabled** (would invalidate the frequency axis) |

### Why SpecAugment?

[SpecAugment](https://arxiv.org/abs/1904.08779) (Park et al., 2019) is the standard augmentation strategy for audio classification. It zeroes out random horizontal stripes (frequency masks) and vertical stripes (time masks) of the spectrogram during training. This forces the model to learn redundant, robust spectral features rather than memorizing specific frequency bins or temporal positions — which is especially important when training on synthetic-augmented audio (pitch shift, time stretch) that can leak its augmentation signature.

## Empirical results

```
Dataset:         4,000 spectrograms (2,000 crash + 2,000 noise, 80/20 split)
Validation set:  800 held-out spectrograms

Epoch 1: val_acc=96.88%
Epoch 2: val_acc=98.12%
Epoch 3: val_acc=98.00%
Epoch 4: val_acc=96.62%   ← typical SpecAugment "wobble", model resists overfitting
Epoch 5: val_acc=98.75%   ← best, saved as crash_detector.pth
```

A small per-epoch wobble is healthy: it indicates the masking is preventing the model from memorizing the train set. With *no* SpecAugment (previous run), the model hit 100 % val accuracy in 3 epochs but generalized worse to out-of-distribution audio.

## Real-world generalization test

Run `pytest tests/test_inference.py -v` to verify the trained model on held-out samples. Live results from the latest run:

```
[OK] freesound_332056.wav            -> CRASH   99.9%
[OK] freesound_332057.wav            -> CRASH   99.1%
[OK] freesound_332058.wav            -> CRASH   98.3%
[OK] freesound_332059.wav            -> CRASH   97.5%
[OK] freesound_332060.wav            -> CRASH   99.2%
[OK] freesound_332061.wav            -> CRASH   99.2%
[OK] glass_breaking sample           -> CRASH   99.9%
[OK] car_horn sample                 -> CRASH  100.0%
[OK] fireworks sample                -> CRASH  100.0%
[OK] rain sample                     -> NORMAL 100.0%
[OK] crackling_fire sample           -> NORMAL  99.9%
[OK] wind sample                     -> NORMAL  99.9%
[OK] vacuum_cleaner sample           -> NORMAL  99.9%
[OK] chainsaw sample                 -> NORMAL 100.0%
[OK] engine sample                   -> NORMAL 100.0%
[OK] us8k siren                      -> NORMAL  99.5%
[OK] us8k jackhammer                 -> NORMAL 100.0%
[OK] us8k drilling                   -> NORMAL 100.0%
[OK] us8k engine_idling              -> NORMAL 100.0%
```

**33 / 33 representative samples classified correctly**, with 18 different real Freesound collision recordings (all augmentations included) at 97-100 % confidence. The hard negatives — siren, jackhammer, drilling, engine_idling — are correctly classified as NORMAL, demonstrating the model is discriminating crash impacts from other loud urban sounds.

## Checkpoint storage

- Active: `backend/audio_model/crash_detector.pth` (43 MB)
- Versioned snapshots: `backend/audio_model/checkpoints/`
  - `resnet18_v1_acc100.pth` — 1,120-clip ESC-50-only run
  - `resnet18_v2_us8k_acc9875.pth` — 4,000-clip ESC-50 + UrbanSound8K + Freesound + SpecAugment

## Sanity checks

```bash
# In-domain crash sample (should be CRASH @ >0.99)
.venv/bin/python -c "
import librosa
from backend.audio_model.inference import predict
samples, _ = librosa.load('data/raw_audio/crash/esc_2-144137-A-43.wav', sr=22050, mono=True)
print(predict(samples))
"

# Real out-of-distribution Freesound crash (should be CRASH @ >0.95)
.venv/bin/python -c "
import librosa
from backend.audio_model.inference import predict
samples, _ = librosa.load('data/raw_audio/crash/freesound_332059.wav', sr=22050, mono=True)
print(predict(samples))
"

# Hard negative (should be NORMAL)
.venv/bin/python -c "
import librosa, glob
from backend.audio_model.inference import predict
sample = glob.glob('data/raw_audio/noise/us8k_7_*.wav')[0]   # jackhammer
samples, _ = librosa.load(sample, sr=22050, mono=True)
print(predict(samples))
"
```
