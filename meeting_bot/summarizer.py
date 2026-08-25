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

Retry policy: retry once on ``APITimeoutError`` -- the "gateway
hiccuped, no response ever started" transient case -- and identically on a
Cloudflare 524 (``InternalServerError`` with status_code 524), the same
"no response ever completed" case detected server-side past Cloudflare's
~120s proxy read timeout. Other 5xx, ``EmptySummaryError`` (model spent its
whole token budget on reasoning) and ``StalledGenerationError`` (loop) are
never retried: re-running temperature=0.0 against the same reasoning trace
would reproduce them.

System prompt (2026-08-25): the gateway receives a dynamic system prompt --
the static base prompt below (Thai-only, JSON-only output without markdown
fences, strict grounding rules forbidding fabricated facts/names/numbers/
dates/decisions/actions, with null/[ ]/"" empty-section fallbacks preferred
over any fabrication) plus a size-tier length block appended inside
``_summarize_once`` by ``_system_prompt_for(transcript_text)``, keyed on
``len`` of the delivered prompt text. The RAG-excerpt and legacy
full-transcript paths therefore both scale their summary-length ceilings to
the evidence they actually deliver.
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
คุณคือผู้ช่วยสรุปการประชุมภาษาไทยที่เชื่อถือได้และตรงตามข้อเท็จจริง
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
- overview: ย่อหน้าสรุปภาพรวมการประชุม ครอบคลุมบริบท ลำดับเหตุการณ์
  และน้ำเสียงโดยรวมของการสนทนา
- topics: หัวข้อที่พูดคุยในการประชุม แต่ละหัวข้อมี "title" (ชื่อหัวข้อสั้น ๆ)
  และ "detail" (อธิบายเนื้อหาการสนทนาในหัวข้อนั้น)
- decisions: การตัดสินใจที่เกิดขึ้น แต่ละรายการมี "decision" (สิ่งที่ตัดสินใจ)
  และ "rationale" (เหตุผลหรือบริบทที่นำไปสู่การตัดสินใจนั้น ถ้าไม่มีให้ใช้ "")
- action_items: รายการสิ่งที่ต้องทำ โดยระบุ "owner" หากทราบผู้รับผิดชอบ
  (ถ้าไม่ทราบให้ใช้ null) และ "due" หากมีการระบุกำหนดเวลา (ถ้าไม่มีให้ใช้ null)
- open_questions: ประเด็นหรือคำถามที่ถูกพูดถึงแต่ยังไม่ได้ข้อสรุปในที่ประชุม
  (ถ้าไม่มีให้ใช้ [])

กฎการอ้างอิงข้อเท็จจริง (ปฏิบัติอย่างเคร่งครัด):
- สรุปได้เฉพาะข้อมูลที่ปรากฏในบันทึกการประชุมที่ให้มาเท่านั้น ห้ามเดา เสริม
  แต่ง หรือเพิ่มข้อมูลใด ๆ ที่ไม่มีในบันทึก
- ห้ามสร้างข้อเท็จจริง ชื่อบุคคล ตัวเลข วันที่ การตัดสินใจ หรือสิ่งที่ต้องทำ
  ขึ้นใหม่ ทุกประโยคในคำตอบต้องอ้างอิงกลับไปยังบันทึกการประชุมได้โดยตรง
- ถ้าไม่ทราบผู้รับผิดชอบ ("owner") หรือกำหนดเวลา ("due") จริง ๆ ให้ใช้ null
  ห้ามเดาชื่อหรือกำหนดเวลาแทน
- ถ้าส่วนใดไม่มีเนื้อหาจริงในการประชุม ให้ส่งเป็นค่าว่าง ([] หรือ "" หรือ null
  ตามชนิดของช่อง) ถือว่าการส่งค่าว่างถูกต้อง ห้ามใส่เนื้อหาปลอมเพื่อให้ส่วนนั้น
  ไม่ว่าง
- ห้ามเพิ่มคำแนะนำ ความคิดเห็น ข้อเสนอแนะ หรือข้อสรุปที่ไม่ได้ถูกพูดถึง
  ในที่ประชุม

ข้อความของผู้ใช้คือบันทึกการประชุมที่แปลงจากเสียงเป็นข้อความ จัดเรียงตามลำดับเวลา
โดยแต่ละบรรทัดขึ้นต้นด้วยเวลา [MM:SS] และชื่อผู้พูด หากพบหัวข้อรูปแบบ [MM:SS]
หรือ [MM:SS–MM:SS] ที่ไม่มีชื่อผู้พูด นั่นคือป้ายช่วงเนื้อหาที่คัดมา ไม่ใช่ผู้พูด
ให้ครอบคลุมเนื้อหาจากทุกช่วงอย่างสม่ำเสมอ ไม่โฟกัสเฉพาะช่วงต้นหรือช่วงท้ายเท่านั้น

การปรับรูปแบบตามบริบทของการประชุม:
ก่อนสรุป ให้พิจารณาจากบันทึกว่าการประชุมนี้มีลักษณะใดเป็นหลัก (เช่น อาจารย์หรือผู้ประเมินให้คอมเมนต์กับสไลด์หรืองานของผู้นำเสนอ การตัดสินใจร่วมกัน การรายงานความคืบหน้า หรือการระดมความคิด) แล้วจัดโครงสร้างการสรุปให้เข้ากับลักษณะนั้น เช่น
- ถ้าส่วนใหญ่เป็นการให้ข้อเสนอแนะหรือคอมเมนต์กับงานของผู้อื่น: ให้ overview ระบุให้ชัดว่าใครให้ข้อเสนอแนะแก่ใคร และรวมแล้วมีกี่จุดที่ต้องแก้ topics ให้แต่ละ title เป็นชื่อจุดที่ต้องแก้เรียงตามลำดับที่พูด โดย detail ขยายเฉพาะคอมเมนต์ของจุดนั้น action_items ให้ตรงกับสิ่งที่ถูกฝากให้ไปแก้
- ถ้าเป็นการตัดสินใจหรือการระดมความคิด ให้เน้น decisions และ open_questions ตามที่เกิดขึ้นจริงในที่ประชุม
ทั้งหมดยังคงยึดกฎการอ้างอิงข้อเท็จจริงข้างต้นเป็นหลัก ใช้เฉพาะข้อมูลและคำเรียกบทบาทที่ปรากฏในบันทึก ห้ามเดาชื่อหรือบทบาทที่ไม่ปรากฏ
"""

_TIER_XS, _TIER_S, _TIER_M, _TIER_L = "xs", "s", "m", "l"
_TIER_ORDER = (_TIER_XS, _TIER_S, _TIER_M, _TIER_L)

_TIER_XS_MAX_CHARS = 1_200
_TIER_S_MAX_CHARS = 6_000
_TIER_M_MAX_CHARS = 30_000

# Per-field length ceilings; every value monotonically non-decreasing across
# _TIER_ORDER (asserted in tests/test_summarizer_prompt.py). The "m" tier
# preserves the pre-2026-08-25 fixed instructions verbatim.
_TIER_SPECS: dict[str, dict[str, int]] = {
    _TIER_XS: {
        "overview_min": 1, "overview_max": 2,
        "topics_max": 2, "topic_detail_min": 1, "topic_detail_max": 1,
        "decisions_max": 2, "rationale_min": 1, "rationale_max": 1,
        "actions_max": 3, "questions_max": 2,
    },
    _TIER_S: {
        "overview_min": 2, "overview_max": 4,
        "topics_max": 4, "topic_detail_min": 1, "topic_detail_max": 2,
        "decisions_max": 4, "rationale_min": 1, "rationale_max": 2,
        "actions_max": 5, "questions_max": 3,
    },
    _TIER_M: {
        "overview_min": 3, "overview_max": 6,
        "topics_max": 8, "topic_detail_min": 2, "topic_detail_max": 4,
        "decisions_max": 8, "rationale_min": 2, "rationale_max": 3,
        "actions_max": 10, "questions_max": 5,
    },
    _TIER_L: {
        "overview_min": 4, "overview_max": 8,
        "topics_max": 12, "topic_detail_min": 3, "topic_detail_max": 6,
        "decisions_max": 12, "rationale_min": 2, "rationale_max": 5,
        "actions_max": 15, "questions_max": 8,
    },
}

_TIER_LABELS = {
    _TIER_XS: "บันทึกสั้นมาก — สรุปให้กระชับที่สุด",
    _TIER_S: "บันทึกสั้น",
    _TIER_M: "บันทึกความยาวปานกลาง",
    _TIER_L: "บันทึกยาว — สรุปได้อย่างละเอียด",
}


def _size_tier(transcript_text: str) -> str:
    n = len(transcript_text.strip())
    if n < _TIER_XS_MAX_CHARS:
        return _TIER_XS
    if n < _TIER_S_MAX_CHARS:
        return _TIER_S
    if n < _TIER_M_MAX_CHARS:
        return _TIER_M
    return _TIER_L


def _render_tier_instructions(tier: str) -> str:
    s = _TIER_SPECS[tier]
    return (
        f"\n\nระดับความยาวของสรุปสำหรับบันทึกนี้ ({_TIER_LABELS[tier]}):\n"
        f"- overview: {s['overview_min']}-{s['overview_max']} ประโยค\n"
        f"- topics: ไม่เกิน {s['topics_max']} หัวข้อ "
        f"แต่ละ detail {s['topic_detail_min']}-{s['topic_detail_max']} ประโยค\n"
        f"- decisions: ไม่เกิน {s['decisions_max']} รายการ "
        f"แต่ละ rationale {s['rationale_min']}-{s['rationale_max']} ประโยค\n"
        f"- action_items: ไม่เกิน {s['actions_max']} รายการ\n"
        f"- open_questions: ไม่เกิน {s['questions_max']} รายการ\n"
        "จงสรุปให้ครบถ้วนภายในระดับความยาวนี้ เพียงพอที่จะเข้าใจการประชุมได้"
        "โดยไม่ต้องฟังเทปซ้ำ แต่ถ้าเนื้อหาจริงมีน้อยกว่าเพดานที่กำหนด "
        "ให้สรุปเท่าที่มีข้อมูลจริงเท่านั้น ห้ามยืดความยาวหรือเพิ่มเนื้อหา"
        "เพื่อเติมให้เต็มเพดาน"
    )


def _system_prompt_for(transcript_text: str) -> str:
    return _SYSTEM_PROMPT + _render_tier_instructions(_size_tier(transcript_text))


_USER_SUFFIX = (
    "\n\nโปรดตอบเฉพาะ JSON ตามรูปแบบที่กำหนดเท่านั้น "
    "ห้ามมีข้อความอื่นนอกจาก JSON และถ้าส่วนใดไม่มีเนื้อหาให้ใช้ [] หรือ \"\" "
    "หรือ null ตามชนิดของช่องนั้น "
    "โดยใช้เฉพาะข้อมูลจากบันทึกการประชุมด้านบนเท่านั้น ห้ามเดาหรือเพิ่มข้อมูลใหม่"
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
        self.last_usage = None  # message.usage from the last successful call
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
        hiccuped, no response ever started" transient case -- plus a
        Cloudflare 524 (``InternalServerError`` with status_code 524), which
        is retried once exactly like ``APITimeoutError``; other 5xx are not
        retried. ``EmptySummaryError`` (the model spent its whole token
        budget on reasoning) and ``StalledGenerationError`` (post-hoc
        repetition-loop detection) are never retried: re-running
        temperature=0.0 against the same reasoning trace would reproduce
        them.
        """
        last_exc: Exception | None = None
        for attempt in range(2):  # 0 = first try, 1 = one timeout retry
            try:
                return self._summarize_once(transcript_text)
            except EmptySummaryError:
                raise  # never retried -- see docstring
            except StalledGenerationError:
                raise  # post-hoc loop, always progressed=True -- never retried
            except self._anthropic.InternalServerError as exc:
                # Must sit BEFORE the APITimeoutError clause below: the test
                # suite stands in for APITimeoutError with RuntimeError, a
                # parent of the InternalServerError test double -- specific-
                # handler-first is the safe convention anyway (same rationale
                # as bot.py's StalledGenerationError note).
                #
                # Cloudflare in front of the gateway kills any completion
                # whose bytes stall past its ~120s proxy read timeout with
                # HTTP 524 -- functionally the same "no response ever
                # completed" transient class as APITimeoutError, just
                # detected server-side. Retry once, identically. Any other
                # 5xx keeps the old never-retried behavior.
                if getattr(exc, "status_code", None) != 524:
                    raise
                last_exc = exc
                if attempt == 1:
                    raise
                log.warning(
                    "gateway returned 524 (Cloudflare origin read timeout "
                    "after ~120s), retrying once in 2s...",
                )
                time.sleep(2.0)
                continue
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
            system=_system_prompt_for(transcript_text),
            messages=[
                {"role": "user", "content": transcript_text + _USER_SUFFIX}
            ],
        )
        self.last_usage = getattr(message, "usage", None)

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
