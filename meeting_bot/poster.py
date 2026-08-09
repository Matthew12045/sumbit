"""Discord embed building and posting.

The richer summary does not fit inside a single embed (1024 chars/field,
6000 total), so the output is split: the embed is a scannable index (overview,
truncated ``title — detail`` bullets, action items, open questions) and the
*complete* summary is attached as a Markdown file in the same message.
"""

from __future__ import annotations

import asyncio
import io
import logging
from datetime import datetime, timedelta

import discord

from .config import Config
from .summary_parse import Summary

log = logging.getLogger(__name__)

_EMBED_TITLE = "📝 สรุปการประชุม"
# Discord embed limits: 1024 per field value, 4096 per description, and the
# SUM of title + description + all field names/values + footer must stay <= 6000.
_EMBED_TOTAL_MAX = 6000
_FIELD_MAX = 1024
# Soft cap on the overview inside the embed.  The full overview always lives in
# the attached summary.md, so the embed only needs a scannable summary; the real
# ceiling is the shared 6000-char pool enforced in ``build_embed``.
_DESCRIPTION_MAX = 2000
_TRUNCATE_NOTE = "…ดูรายละเอียดเพิ่มเติมในไฟล์แนบ"

# Field names are fixed (non-value) embed text and count toward the 6000 total.
_FIELD_NAMES = (
    "หัวข้อที่พูดคุย",
    "การตัดสินใจ",
    "รายการที่ต้องทำ",
    "คำถามที่ยังไม่ได้ข้อสรุป",
)


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to **at most** ``limit`` chars, appending the note."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    budget = limit - len(_TRUNCATE_NOTE) - 1
    if budget <= 0:
        return _TRUNCATE_NOTE[:limit]
    return text[:budget] + "\n" + _TRUNCATE_NOTE


def _fit_to_pool(desired: list[int], pool: int) -> list[int]:
    """Scale per-piece desired char budgets to fit a shared ``pool``.

    Returns budgets whose sum is ``<= pool``.  When the desired total already
    fits, returns the desired values unchanged (nothing is truncated
    needlessly); otherwise every piece is scaled down proportionally, so a
    genuinely rich meeting can never blow Discord's 6000-char total embed budget.
    """
    total = sum(desired)
    if total <= pool:
        return list(desired)
    return [d * pool // total for d in desired]


def _duration_text(duration: timedelta) -> str:
    total_seconds = max(0, int(duration.total_seconds()))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes} นาที {seconds} วินาที" if minutes else f"{seconds} วินาที"


def build_embed(
    summary: Summary,
    *,
    started_at: datetime,
    duration: timedelta,
    member_count: int,
    meeting_title: str | None = None,
) -> discord.Embed:
    """Build the meeting summary embed (a scannable index into the file).

    ``meeting_title`` is optional; the caller resolves it to the guild/voice
    channel name before posting.
    """
    title = meeting_title or "การประชุม"
    footer_text = (
        f"{title} · ระยะเวลา {_duration_text(duration)} · "
        f"ผู้เข้าร่วม {member_count} คน · {started_at:%d %b %Y}"
    )

    overview_text = summary.overview or title

    topic_text = "\n".join(
        f"• **{item.title}** — {item.detail}" if item.detail else f"• **{item.title}**"
        for item in summary.topics
    ) or "—"
    decision_text = "\n".join(
        f"• **{item.decision}** — {item.rationale}"
        if item.rationale
        else f"• **{item.decision}**"
        for item in summary.decisions
    ) or "—"
    action_lines = []
    for item in summary.action_items:
        line = f"• {item.action}"
        if item.owner:
            line += f" — {item.owner}"
        if item.due:
            line += f" (due: {item.due})"
        action_lines.append(line)
    action_text = "\n".join(action_lines) or "—"
    question_text = "\n".join(f"• {q}" for q in summary.open_questions) or "—"

    # Discord caps the SUM of title + description + field names/values + footer
    # at 6000.  Fixed text (title, field names, footer) is counted first, then
    # the remaining pool is shared across the description and the four field
    # values — so a rich meeting never overflows even when every section is at
    # its per-item maximum (the full untruncated summary is in the attachment).
    fixed = (
        len(_EMBED_TITLE)
        + len(footer_text)
        + sum(len(name) for name in _FIELD_NAMES)
    )
    texts = [overview_text, topic_text, decision_text, action_text, question_text]
    desired = [
        min(_DESCRIPTION_MAX, len(overview_text)),
        min(_FIELD_MAX, len(topic_text)),
        min(_FIELD_MAX, len(decision_text)),
        min(_FIELD_MAX, len(action_text)),
        min(_FIELD_MAX, len(question_text)),
    ]
    limits = _fit_to_pool(desired, _EMBED_TOTAL_MAX - fixed)
    description = _truncate(overview_text, limits[0])
    field_values = [_truncate(t, limits[i]) for i, t in enumerate(texts[1:], 1)]

    embed = discord.Embed(
        title=_EMBED_TITLE,
        description=description,
        color=discord.Color.blurple(),
        timestamp=started_at,
    )
    for name, value in zip(_FIELD_NAMES, field_values):
        embed.add_field(name=name, value=value, inline=False)
    embed.set_footer(text=footer_text)
    return embed


def render_markdown(
    summary: Summary,
    *,
    started_at: datetime,
    duration: timedelta,
    member_count: int,
    meeting_title: str | None = None,
) -> str:
    """Render the *complete* summary as a Markdown document.

    Unlike the embed, nothing is truncated: full ``detail``/``rationale`` text
    for every topic and decision, the full action-item table, and all open
    questions. This is where the "more summarized context" actually lives.
    """
    title = meeting_title or "การประชุม"
    lines: list[str] = [f"# 📝 สรุปการประชุม: {title}", ""]
    if summary.overview:
        lines += [summary.overview, ""]

    lines += ["## หัวข้อที่พูดคุย", ""]
    if summary.topics:
        for item in summary.topics:
            lines.append(f"### {item.title}")
            if item.detail:
                lines.append(item.detail)
            lines.append("")
    else:
        lines.append("ไม่มี\n")

    lines += ["## การตัดสินใจ", ""]
    if summary.decisions:
        for item in summary.decisions:
            lines.append(f"### {item.decision}")
            if item.rationale:
                lines.append(item.rationale)
            lines.append("")
    else:
        lines.append("ไม่มี\n")

    lines += ["## รายการที่ต้องทำ", ""]
    if summary.action_items:
        lines += ["| # | สิ่งที่ต้องทำ | ผู้รับผิดชอบ | กำหนดส่ง |", "|---|---|---|---|"]
        for index, item in enumerate(summary.action_items, 1):
            lines.append(
                f"| {index} | {item.action} | {item.owner or '—'} | {item.due or '—'} |"
            )
    else:
        lines.append("ไม่มี\n")

    lines += ["## คำถามที่ยังไม่ได้ข้อสรุป", ""]
    if summary.open_questions:
        lines += [f"- {q}" for q in summary.open_questions]
    else:
        lines.append("ไม่มี\n")

    lines += [
        "---",
        f"ระยะเวลา {_duration_text(duration)} · ผู้เข้าร่วม {member_count} คน · "
        f"{started_at:%d %b %Y}",
    ]
    return "\n".join(lines) + "\n"


class Poster:
    """Sends the summary embed + full Markdown file to the target channel."""

    def __init__(self, config: Config):
        self.config = config

    async def post(self, channel, summary: Summary, *, meta: dict) -> discord.Message:
        """Post ``summary`` to ``channel``. Retries up to 3x on 429/5xx.

        The ``discord.File`` is rebuilt inside the retry loop: a failed send
        consumes the ``BytesIO`` buffer, so a retry would otherwise attach an
        empty file.
        """
        embed = build_embed(summary, **meta)
        markdown = render_markdown(summary, **meta)
        for attempt in range(1, 4):
            try:
                file = discord.File(
                    io.BytesIO(markdown.encode("utf-8")), filename="summary.md"
                )
                return await channel.send(embed=embed, file=file)
            except discord.HTTPException as exc:
                if exc.status not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                await asyncio.sleep(attempt)
        raise RuntimeError("unreachable")  # pragma: no cover
