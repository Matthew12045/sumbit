"""Pure-logic tests for meeting_bot.audio (no Discord/network)."""

import numpy as np
import pytest

from meeting_bot.audio import is_speech_block, resample_48k_stereo_to_16k_mono


def _stereo_sine_pcm(freq, seconds, amplitude=0.5, sample_rate=48000):
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    tone = amplitude * np.sin(2 * np.pi * freq * t)
    vals = (tone * 32768.0).astype(np.int16)
    stereo = np.empty(n * 2, dtype=np.int16)
    stereo[0::2] = vals
    stereo[1::2] = vals
    return stereo.tobytes()


def test_single_frame_length():
    # 20 ms frame = 3840 bytes -> 320 samples at 16 kHz.
    out = resample_48k_stereo_to_16k_mono(bytes(3840))
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert len(out) == 320


def test_output_length_one_second():
    # ⌊48000*960/3⌋-derived: 1 s at 48 kHz decimated by 3 -> 16000 samples.
    pcm = _stereo_sine_pcm(440, seconds=1.0)
    out = resample_48k_stereo_to_16k_mono(pcm)
    assert len(out) == 16000


def test_output_range():
    pcm = _stereo_sine_pcm(440, seconds=0.5, amplitude=0.5)
    out = resample_48k_stereo_to_16k_mono(pcm)
    assert np.all(np.abs(out) <= 1.0)


def test_1khz_tone_maps_to_correct_samples():
    pcm = _stereo_sine_pcm(1000, seconds=1.0, amplitude=0.5)
    out = resample_48k_stereo_to_16k_mono(pcm)
    assert len(out) == 16000
    # Phase-independent I/Q amplitude estimate on the steady-state region.
    mid = out[300:-300]
    t = np.arange(len(mid)) / 16000.0 + 300 / 16000.0
    sin_part = np.sum(mid * np.sin(2 * np.pi * 1000.0 * t))
    cos_part = np.sum(mid * np.cos(2 * np.pi * 1000.0 * t))
    amp = 2.0 * np.hypot(sin_part, cos_part) / len(mid)
    assert abs(amp - 0.5) < 0.05


def test_silence_has_no_energy():
    out = resample_48k_stereo_to_16k_mono(bytes(3840 * 10))
    assert np.max(np.abs(out)) < 1e-6


def test_is_speech_block_threshold():
    assert not is_speech_block(np.zeros(320, dtype=np.float32), 0.01)
    assert is_speech_block(np.full(320, 0.5, dtype=np.float32), 0.01)
    assert not is_speech_block(np.full(320, 0.005, dtype=np.float32), 0.01)
    assert not is_speech_block(np.array([], dtype=np.float32), 0.01)
