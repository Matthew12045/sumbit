"""Gateway summarization via the Anthropic-compatible API (lazy import).

``anthropic`` is imported lazily inside ``__init__`` to keep the pure-module
import rule intact (config/audio/chunker/transcript/summary_parse import only
stdlib + numpy at module scope).
"""

from __future__ import annotations

import logging
import time

from .config import Config

log = logging.getLogger(__name__)

__all__ = ["Summarizer", "EmptySummaryError"]

_SYSTEM_PROMPT = """\
คุณคือผู้ช่วยสรุปการประชุม จงตอบเป็นภาษาไทยเท่านั้น
ให้ส่งออกเฉพาะ JSON ตามโครงสร้างต่อไปนี้ โดยไม่มีเครื่องหมาย markdown fence และไม่มีข้อความอื่นใด:

{ "topics": ["..."], "decisions": ["..."], "action_items": [{"action": "...", "owner": "..."}] }

ความหมายของแต่ละช่อง:
- topics: หัวข้อที่พูดคุยในการประชุม
- decisions: การตัดสินใจที่เกิดขึ้น
- action_items: รายการสิ่งที่ต้องทำ โดยระบุ "owner" หากทราบผู้รับผิดชอบ
"""

_USER_SUFFIX = (
    "\n\nโปรดตอบเฉพาะ JSON ตามรูปแบบที่กำหนดเท่านั้น "
    "ห้ามมีข้อความอื่นนอกจาก JSON และถ้าส่วนใดไม่มีเนื้อหาให้ใช้ []"
)


class EmptySummaryError(RuntimeError):
    """Gateway returned HTTP 200 OK but no usable text block.

    Almost always the qwen thinking-budget issue: the model spends its whole
    ``max_tokens`` on internal reasoning and never emits a final ``text``
    block, so the API call "succeeds" while ``summarize`` yields ``""``.
    """


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
        )

    def summarize(self, transcript_text: str) -> str:
        """Blocking SDK call. Retries once on ``APITimeoutError`` after a 2 s delay.

        The caller drains the text through :func:`~meeting_bot.summary_parse.parse_summary`.
        """
        last_exc: Exception | None = None
        for attempt in range(2):  # 0 = first try, 1 = one retry
            try:
                response = self._client.messages.create(
                    model=self.cfg.gateway_model,
                    max_tokens=self.cfg.summary_max_tokens,
                    temperature=0.0,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": transcript_text + _USER_SUFFIX}],
                )
                text = "".join(
                    block.text for block in response.content
                    if getattr(block, "type", "") == "text"
                )
                if not text.strip():
                    block_types = [
                        getattr(block, "type", "?") for block in response.content
                    ]
                    log.warning(
                        "summarizer returned no text (stop_reason=%s, blocks=%s, "
                        "non-text chars=%d) — the model likely spent its whole "
                        "token budget on reasoning; try raising SUMMARY_MAX_TOKENS "
                        "in .env",
                        getattr(response, "stop_reason", "unknown"),
                        block_types,
                        _non_text_char_count(response.content),
                    )
                    raise EmptySummaryError(
                        "gateway returned no text (stop_reason=%s, blocks=%s)"
                        % (
                            getattr(response, "stop_reason", "unknown"),
                            block_types,
                        )
                    )
                return text
            except self._anthropic.APITimeoutError as exc:
                last_exc = exc
                if attempt == 0:
                    log.warning(
                        "summarizer timed out after %.0fs, retrying in 2s...",
                        self.cfg.summarize_timeout_seconds,
                    )
                    time.sleep(2.0)
                    continue
                raise
            # Non-timeout errors (auth, bad request, etc.) are not retried.
        raise last_exc  # type: ignore[misc]
