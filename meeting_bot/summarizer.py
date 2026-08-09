"""Gateway summarization via the Anthropic-compatible API (lazy import).

``anthropic`` is imported lazily inside ``__init__`` to keep the pure-module
import rule intact (config/audio/chunker/transcript/summary_parse import only
stdlib + numpy at module scope).

Redesign notes (2026-08-10): summarization now streams instead of blocking on
a single ``messages.create()`` call. qwen is a thinking model that can loop
instead of terminating (see ``EmptySummaryError``); streaming lets us detect
that live -- via a stall timeout and an exact-repeat check on the generated
text -- and abort in tens of seconds instead of riding out the full
``summarize_timeout_seconds`` ceiling.

The old "retry once on any timeout" policy is gone. A stall/timeout AFTER
real output had started is NOT retried: retrying an identical
temperature=0.0 prompt against the same stuck reasoning trace is very likely
to reproduce it, so paying for a second full wait buys nothing. We still
retry once for a stall/timeout that produced literally zero bytes, which is
the "gateway hiccuped, no generation ever started" case -- a genuine
transient failure, not a reproducible pathology.

ASSUMPTION TO VERIFY BEFORE SHIPPING: this relies on the self-hosted gateway
(gateway.9arm.co) correctly implementing SSE streaming for
``messages.stream()`` -- i.e. emitting real ``content_block_delta`` events
(including ``thinking_delta`` for qwen's reasoning trace) rather than just
buffering the whole completion and returning it once at the end. If it does
the latter, stall/repetition detection never gets a chance to fire early
(the first "event" IS the finished response) and this whole mechanism
degrades to "wait for the full call, then check once" -- not wrong, but not
the improvement described above. Confirm with a raw probe against the
gateway before relying on this in production.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from .config import Config

log = logging.getLogger(__name__)

__all__ = [
    "Summarizer",
    "EmptySummaryError",
    "StalledGenerationError",
]

_SYSTEM_PROMPT = """\
คุณคือผู้ช่วยสรุปการประชุมภาษาไทยที่ละเอียดและมีบริบทครบถ้วน
จงตอบเป็นภาษาไทยเท่านั้น และให้ส่งออกเฉพาะ JSON ตามโครงสร้างต่อไปนี้
โดยไม่มีเครื่องหมาย markdown fence และไม่มีข้อความอื่นใดนอกเหนือจาก JSON:

{
  "overview": "...",
  "topics": [{"title": "...", "detail": "..."}],
  "decisions": [{"decision": "...", "rationale": "..."}],
  "action_items": [{"action": "...", "owner": "...", "due": "..."}],
  "open_questions": ["..."]
}

ความหมายของแต่ละช่อง:
- overview: ย่อหน้าสรุปภาพรวมการประชุม 3-6 ประโยค ครอบคลุมบริบท ลำดับเหตุการณ์
  และน้ำเสียงโดยรวมของการสนทนา
- topics: หัวข้อที่พูดคุยในการประชุม แต่ละหัวข้อมี "title" (ชื่อหัวข้อสั้น ๆ)
  และ "detail" (อธิบายเนื้อหาการสนทนาในหัวข้อนั้นอย่างละเอียด 2-4 ประโยค
  รวมถึงมุมมองต่าง ๆ ที่ถูกพูดถึง)
- decisions: การตัดสินใจที่เกิดขึ้น แต่ละรายการมี "decision" (สิ่งที่ตัดสินใจ)
  และ "rationale" (เหตุผลหรือบริบทที่นำไปสู่การตัดสินใจนั้น ถ้าไม่มีให้ใช้ "")
- action_items: รายการสิ่งที่ต้องทำ โดยระบุ "owner" หากทราบผู้รับผิดชอบ
  (ถ้าไม่ทราบให้ใช้ null) และ "due" หากมีการระบุกำหนดเวลา (ถ้าไม่มีให้ใช้ null)
- open_questions: ประเด็นหรือคำถามที่ถูกพูดถึงแต่ยังไม่ได้ข้อสรุปในที่ประชุม
  (ถ้าไม่มีให้ใช้ [])

ห้ามละเว้นบริบทที่สำคัญ จงสรุปให้ครบถ้วนและมีรายละเอียดเพียงพอที่จะเข้าใจ
การประชุมได้โดยไม่ต้องฟังเทปซ้ำ
"""

_USER_SUFFIX = (
    "\n\nโปรดตอบเฉพาะ JSON ตามรูปแบบที่กำหนดเท่านั้น "
    "ห้ามมีข้อความอื่นนอกจาก JSON และถ้าส่วนใดไม่มีเนื้อหาให้ใช้ [] หรือ \"\" "
    "หรือ null ตามชนิดของช่องนั้น"
)


class EmptySummaryError(RuntimeError):
    """Gateway returned a completed message but no usable text block.

    Almost always the qwen thinking-budget issue: the model spends its whole
    ``max_tokens`` on internal reasoning and never emits a final ``text``
    block, so the call "succeeds" while ``summarize`` yields ``""``.
    """


class StalledGenerationError(RuntimeError):
    """Streaming produced no new content for ``stall_timeout_seconds``, or
    detected an exact-repeat loop in the generated text.

    ``progressed`` records whether *any* content had streamed before the
    stall/loop fired -- the caller uses this to decide whether a retry is
    worth the wait (see module docstring).
    """

    def __init__(self, message: str, *, progressed: bool):
        super().__init__(message)
        self.progressed = progressed


def _non_text_char_count(blocks) -> int:
    """Total characters in non-text blocks, checking .text and .thinking."""
    total = 0
    for block in blocks:
        if getattr(block, "type", "") == "text":
            continue
        for attr in ("text", "thinking"):
            value = getattr(block, attr, None)
            if isinstance(value, str):
                total += len(value)
    return total


def _is_looping(buf: str, window: int, min_repeats: int) -> bool:
    """True if the last ``min_repeats`` windows of ``buf`` are identical.

    Cheap and false-positive-resistant: exact repetition of a multi-hundred
    character window doesn't happen in normal generation, even for
    repetitive schema output (bullet lists still vary word to word). This
    catches the same class of "confident repetition-loop hallucination"
    CLAUDE.md already documents for whisper, now for the gateway's thinking
    trace.
    """
    if window <= 0 or min_repeats < 2 or len(buf) < window * min_repeats:
        return False
    tail = buf[-window:]
    return all(
        buf[-window * i : -window * (i - 1)] == tail
        for i in range(2, min_repeats + 1)
    )


# Tags the pump thread puts on the queue alongside terminal payloads.
_DONE = "__done__"
_ERROR = "__error__"


class Summarizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            import anthropic
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "anthropic is required for summarization (pip install anthropic)"
            ) from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            base_url=cfg.anthropic_base_url,   # e.g. https://gateway.9arm.co (no /v1)
            auth_token=cfg.anthropic_auth_token,
            timeout=cfg.summarize_timeout_seconds,
            # Backstop only now -- stall detection below should always fire
            # first. Our own summarize() loop below owns the one no-progress
            # retry, so the SDK's internal 429/5xx retry is disabled to avoid
            # stacking two independent retry policies on top of each other.
            max_retries=0,
        )

    def summarize(self, transcript_text: str) -> str:
        """Blocking call (intended to run via ``asyncio.to_thread``).

        Retries once, but ONLY when the previous attempt produced zero
        streamed bytes before failing. See module docstring for why a
        stall/timeout after real output started is not retried.
        """
        last_exc: Exception | None = None
        for attempt in range(2):  # 0 = first try, 1 = one no-progress retry
            try:
                return self._summarize_once(transcript_text)
            except StalledGenerationError as exc:
                last_exc = exc
                if exc.progressed or attempt == 1:
                    raise
                log.warning("stalled with zero output, retrying once...")
                time.sleep(2.0)
                continue
            except self._anthropic.APITimeoutError as exc:
                last_exc = exc
                if attempt == 1:
                    raise
                log.warning(
                    "summarizer connection timed out after %.0fs with no "
                    "stream ever starting, retrying in 2s...",
                    self.cfg.summarize_timeout_seconds,
                )
                time.sleep(2.0)
                continue
            # EmptySummaryError and anything else: not retried.
        raise last_exc  # type: ignore[misc]  # pragma: no cover

    def _summarize_once(self, transcript_text: str) -> str:
        """Stream one summarization call, enforcing a stall/repetition guard.

        The SDK's blocking stream iteration runs on a background thread and
        is drained here via a queue with a timeout -- that's what lets us
        detect "no new event for N seconds" even though the iterator itself
        blocks on network I/O and can't be timed out directly from the
        consuming side.
        """
        q: "queue.Queue[object]" = queue.Queue()
        stop = threading.Event()

        def _pump() -> None:
            try:
                with self._client.messages.stream(
                    model=self.cfg.gateway_model,
                    max_tokens=self.cfg.summary_max_tokens,
                    temperature=0.0,
                    system=_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": transcript_text + _USER_SUFFIX}
                    ],
                ) as stream:
                    for event in stream:
                        if stop.is_set():
                            return
                        q.put(event)
                    q.put((_DONE, stream.get_final_message()))
            except Exception as exc:  # noqa: BLE001
                q.put((_ERROR, exc))

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()

        buf = ""  # accumulated thinking/text so far; loop-detected either way
        progressed = False
        final_message = None

        while final_message is None:
            try:
                item = q.get(timeout=self.cfg.stall_timeout_seconds)
            except queue.Empty:
                stop.set()
                log.warning(
                    "stall: %d chars streamed so far (progressed=%s), "
                    "last 200 chars: %r",
                    len(buf), progressed, buf[-200:],
                )
                raise StalledGenerationError(
                    f"no new content for {self.cfg.stall_timeout_seconds:.0f}s",
                    progressed=progressed,
                )

            if isinstance(item, tuple):
                tag, payload = item
                if tag == _ERROR:
                    raise payload
                final_message = payload  # tag == _DONE
                break

            event = item
            if getattr(event, "type", "") == "content_block_delta":
                delta = event.delta
                piece = getattr(delta, "text", None) or getattr(delta, "thinking", None)
                if piece:
                    progressed = True
                    buf += piece
                    if _is_looping(
                        buf,
                        self.cfg.repetition_window_chars,
                        self.cfg.repetition_min_repeats,
                    ):
                        stop.set()
                        raise StalledGenerationError(
                            "generation is repeating (loop detected)",
                            progressed=True,
                        )

        pump.join(timeout=5.0)

        text = "".join(
            block.text for block in final_message.content
            if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            block_types = [
                getattr(block, "type", "?") for block in final_message.content
            ]
            log.warning(
                "summarizer returned no text (stop_reason=%s, blocks=%s, "
                "non-text chars=%d) -- the model likely spent its whole "
                "token budget on reasoning; try raising SUMMARY_MAX_TOKENS "
                "in .env",
                getattr(final_message, "stop_reason", "unknown"),
                block_types,
                _non_text_char_count(final_message.content),
            )
            raise EmptySummaryError(
                "gateway returned no text (stop_reason=%s, blocks=%s)"
                % (getattr(final_message, "stop_reason", "unknown"), block_types)
            )
        return text
