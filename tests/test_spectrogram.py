"""Tests for spectrogram generation (Requirement 2)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.audio_model.spectrogram_gen import (
    IMG_SIZE,
    SAMPLE_RATE,
    TARGET_LEN,
    audio_to_mel_spectrogram,
    fix_length,
    samples_to_mel_db,
    samples_to_mel_image,
    wav_to_mel_image,
)


class TestFixLength:
    def test_short_padded_with_zeros(self):
        x = np.ones(100, dtype=np.float32)
        out = fix_length(x)
        assert out.shape[0] == TARGET_LEN
        assert np.all(out[100:] == 0)

    def test_long_trimmed(self):
        x = np.ones(TARGET_LEN * 2, dtype=np.float32)
        out = fix_length(x)
        assert out.shape[0] == TARGET_LEN

    def test_exact_length_unchanged(self):
        x = np.ones(TARGET_LEN, dtype=np.float32)
        out = fix_length(x)
        assert np.array_equal(x, out)


class TestSamplesToMelDb:
    def test_output_finite(self):
        rng = np.random.default_rng(0)
        samples = rng.standard_normal(SAMPLE_RATE * 3).astype(np.float32) * 0.1
        mel_db = samples_to_mel_db(samples, SAMPLE_RATE)
        assert np.all(np.isfinite(mel_db))

    def test_resamples_when_sr_differs(self):
        # Pass at 16 kHz; should still produce a valid mel-db spectrogram
        samples = np.zeros(16000 * 3, dtype=np.float32)
        mel_db = samples_to_mel_db(samples, 16000)
        assert mel_db.ndim == 2

    def test_handles_stereo_input(self):
        # 2-channel input should be downmixed to mono
        samples = np.zeros((SAMPLE_RATE * 3, 2), dtype=np.float32)
        mel_db = samples_to_mel_db(samples, SAMPLE_RATE)
        assert np.all(np.isfinite(mel_db))


class TestImageOutput:
    def test_samples_to_mel_image_size(self):
        samples = np.zeros(TARGET_LEN, dtype=np.float32)
        img = samples_to_mel_image(samples, SAMPLE_RATE)
        assert img.size == (IMG_SIZE, IMG_SIZE)
        assert img.mode == "RGB"

    def test_audio_to_mel_spectrogram_writes_png(self, tmp_path):
        # Synthesize a tiny WAV
        samples = (np.sin(np.linspace(0, 100, SAMPLE_RATE * 3)) * 0.3).astype(np.float32)
        wav_path = tmp_path / "in.wav"
        sf.write(str(wav_path), samples, SAMPLE_RATE, subtype="PCM_16")
        out_path = tmp_path / "nested" / "out.png"
        result = audio_to_mel_spectrogram(wav_path, out_path)
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_wav_to_mel_image_loads_from_disk(self, tmp_path):
        samples = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
        wav_path = tmp_path / "silent.wav"
        sf.write(str(wav_path), samples, SAMPLE_RATE, subtype="PCM_16")
        img = wav_to_mel_image(wav_path)
        assert img.size == (IMG_SIZE, IMG_SIZE)
