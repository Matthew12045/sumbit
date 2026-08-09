"""Background mlx-whisper transcription worker (single daemon thread).

``mlx_whisper`` is imported lazily (inside ``__init__``) so the module itself
stays importable without the MLX stack installed.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter

import numpy as np

from .chunker import Segment
from .transcript import TranscriptEvent

log = logging.getLogger(__name__)

# -- post-whisper garbage detection -----------------------------------

_MAX_CHAR_RUN = 30      # any single char repeated > this many times => garbage
_MAX_TOKEN_RATIO = 0.7  # most-frequent token / total tokens > this => garbage
_MIN_TOKENS = 10        # only apply token-ratio check when there are enough tokens
_MAX_REPEAT_PERIOD = 10  # characters; covers single chars up to short syllables/words


def _max_repeat_run_length(text: str, max_period: int = _MAX_REPEAT_PERIOD) -> int:
    """Longest consecutive run built from a short repeating substring.

    For each candidate period p (1..max_period), scans for the longest
    stretch where text[j] == text[j - p], i.e. text[i:i+p] repeating back
    to back. period=1 is the original single-character-run check; larger
    periods catch multi-character repeating units (e.g. a 3-char Thai
    syllable repeated hundreds of times with no whitespace).
    """
    n = len(text)
    best = 1
    for period in range(1, max_period + 1):
        i = 0
        while i < n - period:
            j = i + period
            run_len = period
            while j < n and text[j] == text[j - period]:
                run_len += 1
                j += 1
            if run_len > best:
                best = run_len
            i = j if j > i else i + 1
    return best


def is_garbage_transcription(text: str) -> bool:
    """Return True if *text* looks like a whisper hallucination on noise.

    Two heuristics (either one firing is enough):

    1. **Repeating-substring run length** — if any short substring (period 1
       up to ``_MAX_REPEAT_PERIOD`` characters, so single chars through short
       syllables/words) repeats back-to-back for more than ``_MAX_CHAR_RUN``
       total characters (e.g. ``"ZZZZZZZZZ..."`` or a Thai syllable repeated
       with no whitespace, ``"ตามตามตาม..."``).
    2. **Token repetition ratio** — split on whitespace; if there are at least
       ``_MIN_TOKENS`` tokens and the most frequent one accounts for more than
       ``_MAX_TOKEN_RATIO`` of the total (e.g. ``"Se Se Se Se Se..."``).

    This is a pure function (no mlx-whisper dependency) so it can be smoke-
    tested with only numpy present.
    """
    if not text or not text.strip():
        return False

    # 1. Max repeating-substring run length (subsumes the old single-char run).
    if _max_repeat_run_length(text) > _MAX_CHAR_RUN:
        return True

    # 2. Token repetition ratio
    tokens = text.strip().split()
    if len(tokens) >= _MIN_TOKENS:
        counts = Counter(tokens)
        top_count = counts.most_common(1)[0][1]
        if top_count / len(tokens) > _MAX_TOKEN_RATIO:
            return True

    return False


class Transcriber:
    """Serializes MLX inference on one daemon thread so the GPU isn't contended
    with the summarization gateway."""

    def __init__(self, model: str, language: str):
        self.model = model
        self.language = language
        try:
            import mlx_whisper
            from mlx_whisper.load_models import load_model
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "mlx_whisper is required for transcription "
                "(pip install 'mlx-whisper>=0.4.3')"
            ) from exc
        self._mlx_whisper = mlx_whisper
        self._load_model = load_model
        self._input: queue.Queue = queue.Queue()
        self._output: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Spawn the single daemon worker thread (idempotent)."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._run,
                name="mlx-whisper",
                daemon=True,
            )
            self._worker.start()

    def submit(self, segment: Segment) -> None:
        """Thread-safe: enqueue a closed segment for transcription."""
        self._input.put(segment)

    def events(self) -> queue.Queue:
        """Output queue of :class:`~meeting_bot.transcript.TranscriptEvent`."""
        return self._output

    def stop(self, flush: bool = True) -> None:
        """Stop the worker. ``flush=True`` drains the input queue to completion
        before the worker exits (a flushed trailing chunk is transcribed, not
        dropped)."""
        with self._lock:
            worker = self._worker
        if worker is None or not worker.is_alive():
            return
        self._input.put(None)  # sentinel: worker drains then exits
        if flush:
            worker.join()
        else:
            worker.join(timeout=0.5)

    def drain(self, timeout: float = 5.0) -> list[TranscriptEvent]:
        """Wait up to ``timeout`` for pending transcription events to land on
        the output queue and return them."""
        events: list[TranscriptEvent] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                events.append(self._output.get(timeout=remaining))
            except queue.Empty:
                break
        return events

    def _run(self) -> None:
        try:
            model = self._load_model(self.model)
        except Exception:  # noqa: BLE001
            log.exception("failed to load whisper model %r", self.model)
            return
        while True:
            item = self._input.get()
            if item is None:
                break
            try:
                event = self._transcribe(model, item)
                if event is not None:
                    self._output.put(event)
            except Exception:  # noqa: BLE001
                log.exception(
                    "transcription failed for segment starting at %.2f", item.start
                )

    def _transcribe(self, model, segment: Segment) -> TranscriptEvent | None:
        samples = np.asarray(segment.samples)
        # mlx-whisper does NOT validate or resample array input; a wrong rate
        # or dtype silently yields garbage, so a pipeline regression must fail
        # loudly instead of transcribing nonsense.
        # The sample rate is guaranteed 16 kHz by the upstream resampler
        # (audio.resample_48k_stereo_to_16k_mono always decimates by 3), so
        # we assert dtype + ndim rather than carrying rate metadata through
        # every pipeline stage.
        assert samples.dtype == np.float32, "transcriber requires float32 mono audio"
        assert samples.ndim == 1, "transcriber requires 1-D mono audio"
        log.info("transcribing segment: speaker=%s start=%.2f dur=%.2f samples=%d dtype=%s min=%.6f max=%.6f mean=%.6f",
                 segment.speaker_name or segment.speaker_key, segment.start, segment.duration, len(samples),
                 samples.dtype, samples.min(), samples.max(), samples.mean())
        result = self._mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self.model,
            language=self.language,
        )
        text = (result.get("text") or "").strip()

        # Extract no_speech_prob from the first segment for quality gating.
        segs = result.get("segments") or []
        no_speech_prob = segs[0].get("no_speech_prob", 0.0) if segs else 0.0
        log.info("transcription result: text=%r segments=%d no_speech_prob=%.4f",
                 text, len(segs), no_speech_prob)

        if not text:
            log.warning("transcription returned empty text for segment")
            return None

        # Gate: whisper is telling us this isn't speech.
        if no_speech_prob > 0.6:
            log.warning(
                "dropping segment: high no_speech_prob=%.4f (text=%r)",
                no_speech_prob, text[:120],
            )
            return None

        # Gate: detect whisper hallucination on noise/encrypted audio.
        if is_garbage_transcription(text):
            log.warning(
                "dropping segment: garbage transcription detected (text[:120]=%r)",
                text[:120],
            )
            return None

        return TranscriptEvent(
            speaker=segment.speaker_name or segment.speaker_key,
            start=segment.start,
            text=text,
        )
