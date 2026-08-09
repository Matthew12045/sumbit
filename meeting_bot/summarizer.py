"""Gateway summarization via the Anthropic-compatible API (lazy import).

``anthropic`` is imported lazily inside ``__init__`` to keep the pure-module
import rule intact (config/audio/chunker/transcript/summary_parse import only
stdlib + numpy at module scope).

Design notes (2026-08-10): probed against the real gateway
(``tools/probe_stream.py``), gateway.9arm.co does NOT stream incrementally in
a way a live watchdog can use. It emits zero ``thinking_delta`` events during
the entire (silent) thinking phase, then streams the final ``text`` block in
a sub-second burst just before ``message_stop``; the pre-``message_start``
delay alone can exceed 20s on a cold run. So the streaming stall/repetition
guards could never fire early -- any thinking phase longer than
``STALL_TIMEOUT_SECONDS`` aborted a perfectly healthy generation. We reverted
to a plain blocking ``client.messages.create()`` call and rely on the SDK
client timeout (``SUMMARIZE_TIMEOUT_SECONDS``) as the ceiling.

The exact-repeat loop check is kept, but now runs POST-HOC on the completed
output (thinking + text) instead of live during streaming: qwen is a thinking
model that can loop instead of terminating, and a completed message whose
output is an exact-repeat loop is still worthless and would confuse the JSON
parser. A detected loop raises ``StalledGenerationError(progressed=True)`` so
``bot.py``'s existing ⚠️ handler fires.

Retry policy: retry once on ``APITimeoutError`` only -- the "gateway
hiccuped, no response ever started" transient case. ``EmptySummaryError``
(model spent its whole token budget on reasoning) and
``StalledGenerationError`` (loop) are never retried: re-running
temperature=0.0 against the same reasoning trace would reproduce them.
"""

from __future__ import annotations

import logging
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
    """A completed summary was rejected because its output is an exact-repeat
    loop.

    ``progressed`` is always True here: the check is post-hoc on the completed
    message, so content always existed. It is kept as a keyword arg so the
    exception's shape matches what ``bot.py`` and tests already expect.
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
            # Backstop only now -- our own summarize() loop below owns the one
            # APITimeoutError retry, so the SDK's internal 429/5xx retry is
            # disabled to avoid stacking two independent retry policies on top
            # of each other.
            max_retries=0,
        )

    def summarize(self, transcript_text: str) -> str:
        """Blocking call (intended to run via ``asyncio.to_thread``).

        Retries once, but ONLY on ``APITimeoutError`` -- the "gateway
        hiccuped, no response ever started" transient case.
        ``EmptySummaryError`` (the model spent its whole token budget on
        reasoning) and ``StalledGenerationError`` (post-hoc repetition-loop
        detection) are never retried: re-running temperature=0.0 against the
        same reasoning trace would reproduce them.
        """
        last_exc: Exception | None = None
        for attempt in range(2):  # 0 = first try, 1 = one timeout retry
            try:
                return self._summarize_once(transcript_text)
            except EmptySummaryError:
                raise  # never retried -- see docstring
            except StalledGenerationError:
                raise  # post-hoc loop, always progressed=True -- never retried
            except self._anthropic.APITimeoutError as exc:
                last_exc = exc
                if attempt == 1:
                    raise
                log.warning(
                    "summarizer connection timed out after %.0fs with no "
                    "response ever starting, retrying in 2s...",
                    self.cfg.summarize_timeout_seconds,
                )
                time.sleep(2.0)
                continue
            # Anything else: not retried.
        raise last_exc  # type: ignore[misc]  # pragma: no cover

    def _summarize_once(self, transcript_text: str) -> str:
        """Run one blocking summarization call (no streaming).

        The gateway buffers the whole completion server-side and returns it
        in a single response, so streaming buys nothing (see module
        docstring): we rely on the SDK client timeout
        (``SUMMARIZE_TIMEOUT_SECONDS``) as the ceiling, and run the exact-
        repeat loop check post-hoc on the completed output instead of live
        during generation.
        """
        message = self._client.messages.create(
            model=self.cfg.gateway_model,
            max_tokens=self.cfg.summary_max_tokens,
            temperature=0.0,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": transcript_text + _USER_SUFFIX}
            ],
        )

        text = "".join(
            block.text for block in message.content
            if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            block_types = [
                getattr(block, "type", "?") for block in message.content
            ]
            log.warning(
                "summarizer returned no text (stop_reason=%s, blocks=%s, "
                "non-text chars=%d) -- the model likely spent its whole "
                "token budget on reasoning; try raising SUMMARY_MAX_TOKENS "
                "in .env",
                getattr(message, "stop_reason", "unknown"),
                block_types,
                _non_text_char_count(message.content),
            )
            raise EmptySummaryError(
                "gateway returned no text (stop_reason=%s, blocks=%s)"
                % (getattr(message, "stop_reason", "unknown"), block_types)
            )

        # Post-hoc loop detection. Mirror the old streaming buffer's scope:
        # qwen's thinking trace can loop without terminating, so check the
        # combined thinking + text output, not just the final text.
        buf = ""
        for block in message.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                buf += getattr(block, "text", "")
            elif btype == "thinking":
                buf += getattr(block, "thinking", "")
        if _is_looping(
            buf,
            self.cfg.repetition_window_chars,
            self.cfg.repetition_min_repeats,
        ):
            log.warning(
                "summarizer completed but output is an exact-repeat loop "
                "(window=%d x%d) -- posting a failure note",
                self.cfg.repetition_window_chars,
                self.cfg.repetition_min_repeats,
            )
            raise StalledGenerationError(
                "generation is repeating (loop detected on completed text)",
                progressed=True,
            )
        return text
