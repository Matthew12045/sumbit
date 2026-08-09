"""Per-user PCM capture sink for py-cord's SinkEventRouter."""

from __future__ import annotations

import logging
import threading
import time

import discord.sinks  # noqa: F401  (importing the submodule pulls in the base class)
from discord.sinks import Sink

from . import audio
from .chunker import SilenceChunker

log = logging.getLogger(__name__)


class MeetingSink(Sink):
    """Resamples each user's PCM frames and feeds their per-user chunker.

    ``write`` runs on the py-cord router thread and must never call asyncio —
    it only resamples, chunks, and pushes closed segments onto the
    transcriber's input queue.  ``flush_user`` is driven by ``bot.py`` from
    ``on_voice_state_update`` (it is NOT a py-cord router hook).
    """

    def __init__(
        self,
        transcriber,
        chunker_factory,
        names,
        *,
        threshold: float | None = None,
        **chunker_kw,
    ):
        super().__init__()
        self.transcriber = transcriber
        self.chunker_factory = chunker_factory
        self.names = dict(names or {})
        self.threshold = float(threshold) if threshold is not None else None
        self.chunker_kw = dict(chunker_kw)
        self._chunkers: dict[int, SilenceChunker] = {}
        self._lock = threading.Lock()
        self._last_frame_time = time.monotonic()
        self._frame_count = 0
        self._ever_received_frame = False
        self._resample_failures = 0

    @property
    def last_frame_time(self) -> float:
        """Monotonic time of the most recently written PCM frame (watchdog)."""
        return self._last_frame_time

    @property
    def frame_count(self) -> int:
        """Total PCM frames received (watchdog diagnostics)."""
        return self._frame_count

    @property
    def ever_received_frame(self) -> bool:
        """True if at least one PCM frame was successfully resampled."""
        return self._ever_received_frame

    @property
    def resample_failures(self) -> int:
        """Total resample failures since the sink was created."""
        return self._resample_failures

    def diagnostics(self) -> dict:
        """Return diagnostic state for runtime debugging (thread-safe)."""
        with self._lock:
            chunker_stats = {}
            for uid, c in self._chunkers.items():
                chunker_stats[uid] = c.stats()
            return {
                "frame_count": self._frame_count,
                "ever_received_frame": self._ever_received_frame,
                "resample_failures": self._resample_failures,
                "last_frame_age": time.monotonic() - self._last_frame_time,
                "chunker_count": len(self._chunkers),
                "chunker_stats": chunker_stats,
            }

    def _chunker_for(self, user_id: int, display_name: str | None) -> SilenceChunker:
        chunker = self._chunkers.get(user_id)
        if chunker is None:
            chunker = self.chunker_factory()
            chunker.speaker_key = str(user_id)
            chunker.speaker_name = (
                display_name or self.names.get(user_id) or str(user_id)
            )
            if self.threshold is not None:
                chunker.threshold = self.threshold
            for key, value in self.chunker_kw.items():
                setattr(chunker, key, value)
            self._chunkers[user_id] = chunker
        return chunker

    def write(self, data, user) -> None:
        """Router thread entrypoint. No asyncio, no blocking I/O.

        ``data`` is a ``VoiceData`` in py-cord 2.7+ (extract ``.pcm``) or a raw
        ``bytes``/``bytearray`` frame in earlier versions.
        """
        pcm = getattr(data, "pcm", data)
        if isinstance(pcm, (memoryview, bytearray)):
            pcm = bytes(pcm)
        if not isinstance(pcm, (bytes, bytearray)) or not pcm:
            return

        # Debug: log raw PCM properties on first frame
        if self._frame_count == 0:
            data_bytes = bytes(data) if hasattr(data, "__len__") else b"N/A"
            log.info(
                "sink: raw data type=%s pcm_type=%s pcm_len=%d data_len=%d",
                type(data).__name__, type(pcm).__name__, len(pcm), len(data_bytes) if isinstance(data_bytes, bytes) else "N/A",
            )

        try:
            samples = audio.resample_48k_stereo_to_16k_mono(bytes(pcm))
        except Exception:  # noqa: BLE001
            log.exception("resample failed for user %s", user.id)
            self._resample_failures += 1
            return
        if samples.size == 0:
            return

        # Debug: log first 100 frames' resampled audio properties
        if self._frame_count == 1:
            log.info(
                "sink: first frame resampled pcm_bytes=%d samples_size=%d samples_dtype=%s "
                "samples_min=%.6f samples_max=%.6f samples_mean=%.6f",
                len(pcm), samples.size, samples.dtype,
                samples.min(), samples.max(), samples.mean(),
            )

        now = time.monotonic()
        segments = []
        with self._lock:
            self._last_frame_time = now
            self._frame_count += 1
            self._ever_received_frame = True
            chunker = self._chunker_for(user.id, getattr(user, "display_name", None))
            segments.extend(chunker.feed(samples, now))
        for segment in segments:
            log.debug(
                "sink: segment closed for user=%s dur=%.2fs",
                user.name, segment.duration,
            )
            self.transcriber.submit(segment)

        # Diagnostic: log every 100th frame so we can see if frames arrive.
        if self._frame_count % 100 == 0:
            log.debug(
                "sink: user=%s frames=%d last_frame=%.1fs ago chunkers=%d",
                user.name, self._frame_count,
                time.monotonic() - self._last_frame_time,
                len(self._chunkers),
            )

    def flush_user(self, user_id: int) -> None:
        """Flush a user's open chunker into the transcript.

        Called by ``bot.py`` when a human leaves the channel but the meeting
        continues.  Guarded with a lock because sink state is also touched from
        the router thread.
        """
        with self._lock:
            chunker = self._chunkers.pop(user_id, None)
        if chunker is None:
            return
        now = time.monotonic()
        for segment in chunker.flush(now):
            self.transcriber.submit(segment)

    def flush_all(self) -> None:
        """Flush every chunker and drop them (used during final drain)."""
        with self._lock:
            chunkers = dict(self._chunkers)
            self._chunkers.clear()
        now = time.monotonic()
        for uid, chunker in chunkers.items():
            for segment in chunker.flush(now):
                log.debug(
                    "sink: flush_all segment for uid=%s dur=%.2fs",
                    uid, segment.duration,
                )
                self.transcriber.submit(segment)

    def debug_dump(self) -> dict:
        """Return diagnostic state for runtime debugging (delegates to diagnostics)."""
        return self.diagnostics()
