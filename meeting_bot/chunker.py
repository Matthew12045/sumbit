"""Silence-based, per-speaker audio chunking (pure numpy/stdlib).

Consumes 16 kHz mono float32 samples and emits :class:`Segment` objects that
are handed to the transcriber.  Deliberately asyncio-free so it can run on the
py-cord router thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .audio import is_speech_block

log = logging.getLogger(__name__)

__all__ = ["Segment", "SilenceChunker"]


@dataclass
class Segment:
    speaker_key: str
    speaker_name: str
    start: float        # seconds since meeting start (monotonic clock)
    samples: np.ndarray  # 16 kHz mono float32
    duration: float


class SilenceChunker:
    """Turns a stream of 16 kHz mono float32 frames into speech segments.

    ``speaker_key``/``speaker_name`` are assigned by the sink after
    construction (they are per-user and unknown to the factory).
    """

    speaker_key: str = ""
    speaker_name: str = ""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        threshold: float = 0.01,
        silence_seconds: float = 0.8,
        min_chunk_seconds: float = 1.0,
        max_chunk_seconds: float = 30.0,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_size = int(self.sample_rate * self.frame_ms / 1000)
        self.threshold = float(threshold)
        self.silence_seconds = float(silence_seconds)
        self.min_chunk_seconds = float(min_chunk_seconds)
        self.max_chunk_seconds = float(max_chunk_seconds)

        self.speaker_key = ""
        self.speaker_name = ""
        self._open = np.zeros(0, dtype=np.float32)
        self._open_start: float | None = None
        self._last_speech: float | None = None

    # -- public API -----------------------------------------------------

    def feed(self, samples: np.ndarray, now: float) -> list[Segment]:
        """Feed samples whose batch *ends* at monotonic time ``now``.

        Returns the segments closed by this batch (never raises).
        """
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("SilenceChunker expects 1-D mono samples")
        segments: list[Segment] = []
        i = 0
        while i + self.frame_size <= samples.size:
            block = samples[i : i + self.frame_size]
            frame_time = now - (samples.size - i) / self.sample_rate
            self._apply_frame(block, frame_time, segments)
            i += self.frame_size
        # A trailing partial frame (< frame_size) is dropped; the sink feeds
        # exact 20 ms frames, so this never loses audio in practice.
        return segments

    def flush(self, now: float) -> list[Segment]:
        """Close the trailing partial chunk if >= min_chunk_seconds."""
        if self._open_start is None or self._open.size == 0:
            self._reset_open()
            return []
        segments: list[Segment] = []
        self._close(segments)
        return segments

    def reset(self) -> None:
        self._reset_open()

    def stats(self) -> dict:
        """Return diagnostic stats about this chunker's state."""
        return {
            "open_duration": self.open_duration,
            "open_start": self._open_start,
            "last_speech": self._last_speech,
            "open_size": self._open.size,
            "threshold": self.threshold,
        }

    @property
    def open_duration(self) -> float:
        """Seconds of buffered audio in the currently open chunk."""
        if self._open_start is None:
            return 0.0
        return self._open.size / self.sample_rate

    # -- internals ------------------------------------------------------

    def _apply_frame(
        self,
        block: np.ndarray,
        frame_time: float,
        segments: list[Segment],
    ) -> None:
        frame_duration = self.frame_ms / 1000.0
        rms = float(np.sqrt(np.mean(np.square(block.astype(np.float64)))))
        if is_speech_block(block, self.threshold):
            if self._open_start is None:
                self._open_start = frame_time
                self._open = block.copy()
                log.debug(
                    "chunker[%s]: speech start rms=%.4f threshold=%.4f",
                    self.speaker_name, rms, self.threshold,
                )
            elif frame_time + frame_duration - self._open_start >= self.max_chunk_seconds:
                # Force-close even mid-speech, then start a fresh chunk with
                # the current frame so no audio is lost.
                log.debug(
                    "chunker[%s]: force-close dur=%.2fs",
                    self.speaker_name,
                    frame_time + frame_duration - self._open_start,
                )
                self._close(segments)
                self._open_start = frame_time
                self._open = block.copy()
            else:
                self._open = np.concatenate([self._open, block])
            self._last_speech = frame_time
        else:
            if self._open_start is not None:
                if frame_time - (self._last_speech or self._open_start) >= self.silence_seconds:
                    # Trailing silence exceeded: close; don't include this frame.
                    log.debug(
                        "chunker[%s]: silence close open_dur=%.2fs",
                        self.speaker_name, self.open_duration,
                    )
                    self._close(segments)
                else:
                    self._open = np.concatenate([self._open, block])

    def _close(self, segments: list[Segment]) -> None:
        if self._open_start is None or self._open.size == 0:
            self._reset_open()
            return
        duration = self._open.size / self.sample_rate
        speech_end = (self._last_speech or self._open_start) + self.frame_ms / 1000.0
        speech_duration = speech_end - self._open_start
        if speech_duration >= self.min_chunk_seconds:
            segments.append(
                Segment(
                    speaker_key=self.speaker_key,
                    speaker_name=self.speaker_name,
                    start=self._open_start,
                    samples=self._open.copy(),
                    duration=duration,
                )
            )
        self._reset_open()

    def _reset_open(self) -> None:
        self._open = np.zeros(0, dtype=np.float32)
        self._open_start = None
        self._last_speech = None
