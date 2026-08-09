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
# Discord embed limits: 1024 per field, 4096 per description, 6000 total.
_FIELD_MAX = 1024
# Conservative cap on the description so overview + 4 fields stay under the
# embed's 6000-char total (the full overview lives in the attached file).
_DESCRIPTION_MAX = 2000
_TRUNCATE_NOTE = "…ดูรายละเอียดเพิ่มเติมในไฟล์แนบ"


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars, appending the attachment note."""
    if len(text) <= limit:
        return text
    budget = limit - len(_TRUNCATE_NOTE) - 1
    if budget <= 0:
        return _TRUNCATE_NOTE
    return text[:budget] + "\n" + _TRUNCATE_NOTE


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
    embed = discord.Embed(
        title=_EMBED_TITLE,
        description=_truncate(summary.overview or title, _DESCRIPTION_MAX),
        color=discord.Color.blurple(),
        timestamp=started_at,
    )

    topic_lines = []
    for item in summary.topics:
        if item.detail:
            topic_lines.append(f"• **{item.title}** — {item.detail}")
        else:
            topic_lines.append(f"• **{item.title}**")
    embed.add_field(
        name="หัวข้อที่พูดคุย",
        value=_truncate("\n".join(topic_lines) if topic_lines else "—", _FIELD_MAX),
        inline=False,
    )

    decision_lines = []
    for item in summary.decisions:
        if item.rationale:
            decision_lines.append(f"• **{item.decision}** — {item.rationale}")
        else:
            decision_lines.append(f"• **{item.decision}**")
    embed.add_field(
        name="การตัดสินใจ",
        value=_truncate("\n".join(decision_lines) if decision_lines else "—", _FIELD_MAX),
        inline=False,
    )

    action_lines = []
    for item in summary.action_items:
        line = f"• {item.action}"
        if item.owner:
            line += f" — {item.owner}"
        if item.due:
            line += f" (due: {item.due})"
        action_lines.append(line)
    embed.add_field(
        name="รายการที่ต้องทำ",
        value=_truncate("\n".join(action_lines) if action_lines else "—", _FIELD_MAX),
        inline=False,
    )

    open_question_lines = [f"• {q}" for q in summary.open_questions]
    embed.add_field(
        name="คำถามที่ยังไม่ได้ข้อสรุป",
        value=_truncate("\n".join(open_question_lines) if open_question_lines else "—", _FIELD_MAX),
        inline=False,
    )

    embed.set_footer(
        text=(
            f"{title} · ระยะเวลา {_duration_text(duration)} · "
            f"ผู้เข้าร่วม {member_count} คน · {started_at:%d %b %Y}"
        )
    )
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
