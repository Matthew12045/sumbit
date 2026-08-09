"""Env-gated WAV dump of whisper input chunks (stdlib only).

Set ``DUMP_CHUNKS_DIR`` to a directory and every 16 kHz mono float32 segment
the transcriber receives is written as a 16-bit PCM mono ``.wav`` *before*
transcription, so a failed live run leaves inspectable audio behind (was the
audio that reached whisper clean Thai speech, or corrupted/noisy?).

Inert when the env var is unset. No numpy at module scope (``array``/``wave``/
``re`` only), so it stays importable in a bare numpy-only environment and is
unit-testable without the heavy deps.
"""

from __future__ import annotations

import array
import os
import re
import wave
from pathlib import Path

__all__ = ["dump_segment_wav", "wav_dump_dir"]

_SAMPLE_RATE = 16000
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def wav_dump_dir() -> str | None:
    """The directory chunks are dumped to, or None when dumping is disabled."""
    d = os.environ.get("DUMP_CHUNKS_DIR")
    return d.strip() if d and d.strip() else None


def _sanitize(name: str) -> str:
    return _SAFE_RE.sub("_", str(name)) or "speaker"


def dump_segment_wav(
    speaker: str,
    start: float,
    duration: float,
    samples,
    sample_rate: int = _SAMPLE_RATE,
) -> str | None:
    """Write ``samples`` (float32 mono; numpy array or any iterable) to a .wav.

    Returns the absolute path written, or None when dumping is disabled.
    float32 -> int16 is clamped to [-1.0, 1.0] and scaled by 32767. Filenames
    sort by speaker then start time: ``{speaker}_{start:09.3f}_{dur:.1f}s.wav``.
    """
    out_dir = wav_dump_dir()
    if out_dir is None:
        return None
    filename = f"{_sanitize(speaker)}_{start:09.3f}_{duration:.1f}s.wav"
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    ints = array.array(
        "h",
        (int(round(max(-1.0, min(1.0, float(s))) * 32767)) for s in samples),
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(ints.tobytes())
    return str(path)
