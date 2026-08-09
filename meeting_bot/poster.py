"""Discord embed building and posting."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import discord

from .config import Config
from .summary_parse import Summary

log = logging.getLogger(__name__)

_EMBED_TITLE = "📝 สรุปการประชุม"


def build_embed(
    summary: Summary,
    *,
    started_at: datetime,
    duration: timedelta,
    member_count: int,
    meeting_title: str | None = None,
) -> discord.Embed:
    """Build the meeting summary embed.

    ``meeting_title`` is optional; the caller resolves it to the guild/voice
    channel name before posting.
    """
    title = meeting_title or "การประชุม"
    embed = discord.Embed(
        title=_EMBED_TITLE,
        description=title,
        color=discord.Color.blurple(),
        timestamp=started_at,
    )

    def _bullets(items: list[str]) -> str:
        return "\n".join(f"• {item}" for item in items) if items else "—"

    embed.add_field(name="หัวข้อที่พูดคุย", value=_bullets(summary.topics), inline=False)
    embed.add_field(name="การตัดสินใจ", value=_bullets(summary.decisions), inline=False)

    action_lines = []
    for item in summary.action_items:
        line = f"• {item.action}"
        if item.owner:
            line += f" — {item.owner}"
        action_lines.append(line)
    embed.add_field(
        name="รายการที่ต้องทำ",
        value="\n".join(action_lines) if action_lines else "—",
        inline=False,
    )

    total_seconds = max(0, int(duration.total_seconds()))
    minutes, seconds = divmod(total_seconds, 60)
    duration_text = (
        f"{minutes} นาที {seconds} วินาที" if minutes else f"{seconds} วินาที"
    )
    embed.set_footer(
        text=(
            f"{title} · ระยะเวลา {duration_text} · "
            f"ผู้เข้าร่วม {member_count} คน · {started_at:%d %b %Y}"
        )
    )
    return embed


class Poster:
    """Sends the summary embed to the configured target channel."""

    def __init__(self, config: Config):
        self.config = config

    async def post(self, channel, summary: Summary, *, meta: dict) -> discord.Message:
        """Post ``summary`` to ``channel``. Retries up to 3x on 429/5xx."""
        embed = build_embed(summary, **meta)
        for attempt in range(1, 4):
            try:
                return await channel.send(embed=embed)
            except discord.HTTPException as exc:
                if exc.status not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                await asyncio.sleep(attempt)
        raise RuntimeError("unreachable")  # pragma: no cover
