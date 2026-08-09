"""Pure-numpy audio helpers.

Resamples Discord's 48 kHz stereo int16 PCM frames down to 16 kHz mono
float32 for mlx-whisper.  Only stdlib + numpy at module scope (load-bearing:
this module must import in a bare numpy-only environment).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["resample_48k_stereo_to_16k_mono", "is_speech_block"]

_SAMPLE_RATE_48K = 48000
_DECIMATION = 3
_CUTOFF_HZ = 8000.0
_FIR_TAPS = 63


def _design_lowpass(
    taps: int = _FIR_TAPS,
    cutoff: float = _CUTOFF_HZ,
    sample_rate: float = _SAMPLE_RATE_48K,
) -> np.ndarray:
    """Windowed-sinc (Hamming) low-pass FIR with unit DC gain."""
    if taps % 2 == 0:
        taps += 1
    half = taps // 2
    fc = cutoff / sample_rate
    kernel = np.zeros(taps, dtype=np.float64)
    for i in range(taps):
        x = i - half
        if x == 0:
            kernel[i] = 2.0 * fc
        else:
            kernel[i] = np.sin(2.0 * np.pi * fc * x) / (np.pi * x)
        kernel[i] *= 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (taps - 1))
    kernel /= kernel.sum()
    return kernel


_LOWPASS = _design_lowpass()


def resample_48k_stereo_to_16k_mono(pcm: bytes) -> np.ndarray:
    """int16 PCM (48 kHz, 2ch, 20 ms/frame = 3840 bytes) -> float32 mono 16 kHz.

    Averages the two channels, applies the anti-aliasing low-pass (cutoff
    8 kHz, ~63 taps, windowed-sinc) and decimates by 3.  Decimation without
    the filter would alias Thai fricatives and tones into the passband.
    """
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    if len(pcm) % 2:
        pcm = pcm[: len(pcm) - 1]  # drop an odd trailing byte if present
    raw = np.frombuffer(pcm, dtype=np.int16)
    if raw.size == 0:
        return np.zeros(0, dtype=np.float32)
    if raw.size % 2:
        raw = raw[:-1]
    stereo = raw.reshape(-1, 2).astype(np.float64)
    mono = stereo.mean(axis=1) / 32768.0
    filtered = np.convolve(mono, _LOWPASS, mode="same")
    out = filtered[::_DECIMATION].astype(np.float32)
    # Debug: log resampler output properties (only on first call to avoid spam)
    if not hasattr(resample_48k_stereo_to_16k_mono, "_logged"):
        resample_48k_stereo_to_16k_mono._logged = True
        log.info(
            "audio resample: pcm_bytes=%d int16_samples=%d stereo_pairs=%d mono_len=%d out_len=%d "
            "dtype=%s min=%.6f max=%.6f mean=%.6f",
            len(pcm), raw.size, raw.size // 2, mono.size, out.size,
            out.dtype, out.min(), out.max(), out.mean(),
        )
    return out


def is_speech_block(samples: np.ndarray, threshold: float) -> bool:
    """RMS of the block >= threshold => speech."""
    samples = np.asarray(samples)
    if samples.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return rms >= threshold
