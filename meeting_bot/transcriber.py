"""Background mlx-whisper transcription worker (single daemon thread).

``mlx_whisper`` is imported lazily (inside ``__init__``) so the module itself
stays importable without the MLX stack installed.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .chunker import Segment
from .transcript import TranscriptEvent
from .wav_dump import dump_segment_wav

log = logging.getLogger(__name__)

# -- post-whisper garbage detection -----------------------------------

_MAX_CHAR_RUN = 30      # any single char repeated > this many times => garbage
_MAX_TOKEN_RATIO = 0.7  # most-frequent token / total tokens > this => garbage
_MIN_TOKENS = 10        # only apply token-ratio check when there are enough tokens
_MAX_REPEAT_PERIOD = 10  # characters; covers single chars up to short syllables/words


# -- whisper decode settings + anti-loop retry ---------------------------
#
# mlx-whisper's built-in temperature fallback (decode_with_fallback) only
# escalates when compression_ratio > 2.4 OR avg_logprob < -1.0. Confident
# repetition loops fail BOTH checks (repetition compresses well -> LOW ratio;
# each token is confident -> HIGH avg_logprob), so the fallback never fires and
# the T=0 garbage survives. We therefore drive the escalation ourselves: a
# suspicious primary decode (garbage, or whisper flagged no-speech) is re-decoded
# ONCE with a small temperature bump + a Thai preamble to push the decoder off a
# deterministic repetition attractor, and only dropped if the retry is still bad.
#
# Because loops never trip them, compression_ratio_threshold / logprob_threshold
# are deliberately NOT passed to transcribe() — tightening them to catch loops
# would flag ordinary Thai speech (similar low compression / high logprob) and
# force temperature fallbacks on everything. Do not "add" them back in.
#
# Knobs are read from the environment (NOT Config) so the frozen Config
# dataclass / .env.example key set / spec acceptance criterion 6 stay untouched.
# WHISPER_FP16 must be constant for the whole process (ModelHolder caches the
# model by path only, not dtype) — set it before the first decode and restart
# to change it. WHISPER_BEAM_SIZE flows through transcribe(**decode_options)
# into DecodingOptions.beam_size — but the INSTALLED mlx-whisper raises
# ``NotImplementedError("Beam search decoder is not yet implemented")`` for
# beam_size > 1 (and best_of is incompatible with T=0 greedy), so the knob
# defaults to 0 = greedy kwarg omitted. Bump it only after upgrading
# mlx-whisper to a build that implements grouped decoding.


@dataclass(frozen=True)
class _DecodeSettings:
    """Per-decode knobs passed through to ``mlx_whisper.transcribe``."""

    temperature: float = 0.0
    condition_on_previous_text: bool = False
    initial_prompt: str | None = None
    no_speech_threshold: float = 0.6
    fp16: bool = True
    beam_size: int = 0  # ≤1 ⇒ omit kwarg (greedy); >1 ⇒ pass beam_size


# Retry-only preamble: anchors a deterministic repetition attractor. Never
# emitted in output (mlx decodes all_tokens[len(initial_prompt_tokens):]).
_THAI_PREAMBLE = "ต่อไปนี้คือการประชุม: "


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    """Unset env var -> *default*; a set var (even empty) -> its stripped value.

    Unlike ``_env_bool``/``_env_float``, unset and set-empty must be
    distinguishable here: ``WHISPER_INITIAL_PROMPT=""`` means "disable the
    preamble", which the caller turns into None via ``... or None``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def primary_decode_settings() -> _DecodeSettings:
    """T=0 greedy, clean, unbiased. Same effective behavior as today except
    ``condition_on_previous_text`` is now explicitly False."""
    return _DecodeSettings(
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=None,
        no_speech_threshold=0.6,
        fp16=_env_bool("WHISPER_FP16", True),
        beam_size=_env_int("WHISPER_BEAM_SIZE", 0),
    )


def retry_decode_settings() -> _DecodeSettings:
    """Anti-loop retry: small temperature bump + Thai preamble to push the
    decoder off a deterministic greedy attractor. ``fp16`` must match the
    primary decode (ModelHolder caches the model by path, not dtype). The
    beam size is inherited from the primary decode; mlx-whisper itself
    ignores it whenever temperature > 0."""
    return _DecodeSettings(
        temperature=_env_float("WHISPER_RETRY_TEMPERATURE", 0.2),
        condition_on_previous_text=False,
        initial_prompt=_env_str("WHISPER_INITIAL_PROMPT", _THAI_PREAMBLE) or None,
        no_speech_threshold=0.6,
        fp16=_env_bool("WHISPER_FP16", True),
        beam_size=_env_int("WHISPER_BEAM_SIZE", 0),
    )


def build_decode_kwargs(settings: _DecodeSettings, *, language: str) -> dict:
    """Translate :class:`_DecodeSettings` into mlx_whisper ``transcribe`` kwargs.

    ``beam_size`` is passed through only when > 1 — mlx-whisper's DecodingOptions
    defaults it to None (= greedy), and 0/negative values are treated as "omit"
    so the kwarg never reaches transcribe with a confusing sentinel.
    """
    kwargs: dict = {
        "language": language,
        "temperature": settings.temperature,
        "condition_on_previous_text": settings.condition_on_previous_text,
        "no_speech_threshold": settings.no_speech_threshold,
        "fp16": settings.fp16,
    }
    if settings.beam_size > 1:
        kwargs["beam_size"] = settings.beam_size
    if settings.initial_prompt:
        kwargs["initial_prompt"] = settings.initial_prompt
    return kwargs


def should_retry(
    text: str,
    no_speech_prob: float,
    *,
    no_speech_threshold: float = 0.6,
) -> bool:
    """True when a decode should be retried once: text present AND either the
    garbage filter fired or whisper flagged the window as no-speech.

    Empty (or whitespace-only) text stays a terminal drop (current behavior) —
    it costs nothing to skip retrying a genuinely empty window.
    """
    if not text.strip():
        return False
    return is_garbage_transcription(text) or no_speech_prob > no_speech_threshold


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

        # Diagnostic: dump exactly what whisper will receive (env-gated).
        dump_segment_wav(
            segment.speaker_name or segment.speaker_key,
            segment.start,
            segment.duration,
            samples,
        )

        log.info("transcribing segment: speaker=%s start=%.2f dur=%.2f samples=%d dtype=%s min=%.6f max=%.6f mean=%.6f",
                 segment.speaker_name or segment.speaker_key, segment.start, segment.duration, len(samples),
                 samples.dtype, samples.min(), samples.max(), samples.mean())

        primary = self._decode(samples, primary_decode_settings())
        if primary is None:
            log.warning("transcription returned empty text for segment")
            return None
        text, no_speech_prob, _ = primary

        if not should_retry(text, no_speech_prob):
            return self._event(segment, text)

        # One bounded anti-loop retry on the SAME audio. Whisper's own fallback
        # never fires for confident loops (repetition compresses well, tokens
        # are high-confidence), so we drive the escalation ourselves.
        log.info(
            "primary decode suspicious (garbage=%s no_speech_prob=%.4f) — retrying once",
            is_garbage_transcription(text),
            no_speech_prob,
        )
        retry = self._decode(samples, retry_decode_settings())
        if retry is None:
            log.warning(
                "anti-loop retry returned empty text — dropping segment (primary=%r)",
                text[:120],
            )
            return None
        rtext, r_nsp, _ = retry
        if not should_retry(rtext, r_nsp):
            log.info(
                "recovered after anti-loop retry: primary=%r retry=%r no_speech_prob=%.4f",
                text[:60],
                rtext[:120],
                r_nsp,
            )
            return self._event(segment, rtext)
        log.warning(
            "still garbage after anti-loop retry — dropping segment "
            "(primary=%r retry=%r no_speech_prob=%.4f)",
            text[:120],
            rtext[:120],
            r_nsp,
        )
        return None

    def _decode(
        self,
        samples: np.ndarray,
        settings: _DecodeSettings,
    ) -> tuple[str, float, list] | None:
        """One mlx-whisper decode returning ``(text, no_speech_prob, segs)``.

        Returns None when the decode produced empty text (a terminal drop).
        """
        result = self._mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self.model,
            **build_decode_kwargs(settings, language=self.language),
        )
        text = (result.get("text") or "").strip()

        # Extract no_speech_prob from the first segment for quality gating.
        segs = result.get("segments") or []
        no_speech_prob = segs[0].get("no_speech_prob", 0.0) if segs else 0.0
        log.info(
            "transcription result (temperature=%.2f): text=%r segments=%d no_speech_prob=%.4f",
            settings.temperature,
            text[:120],
            len(segs),
            no_speech_prob,
        )
        if not text:
            return None
        return text, no_speech_prob, segs

    @staticmethod
    def _event(segment: Segment, text: str) -> TranscriptEvent:
        return TranscriptEvent(
            speaker=segment.speaker_name or segment.speaker_key,
            start=segment.start,
            text=text,
        )
