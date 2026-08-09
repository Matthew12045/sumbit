"""Embed budget tests for meeting_bot.poster.

``poster`` imports ``discord`` at module scope, so this module skips in a
minimal environment without it; ``_fit_to_pool`` (the pure budget math) is
still exercised directly.
"""

import pytest

discord = pytest.importorskip("discord")

from meeting_bot import poster


def test_fit_to_pool_unchanged_when_it_fits():
    desired = [100, 1024, 1024, 1024, 1024]
    assert poster._fit_to_pool(desired, 6000) == desired


def test_fit_to_pool_scales_down_to_pool():
    # 2000 + 4x1024 = 6096 desired against a 5888 pool -> everyone shrinks, no
    # piece is starved, and the scaled budgets sum to <= the pool.
    limits = poster._fit_to_pool([2000, 1024, 1024, 1024, 1024], 5888)
    assert sum(limits) <= 5888
    assert all(v > 0 for v in limits)
    assert limits[0] < 2000


def test_fit_to_pool_zero_pool():
    assert poster._fit_to_pool([10, 20], 0) == [0, 0]


def _rich_summary():
    from meeting_bot.summary_parse import ActionItem, DecisionItem, Summary, TopicItem

    return Summary(
        overview="อ" * 2000,
        topics=[TopicItem(title=f"หัวข้อ {i}", detail="ร" * 300) for i in range(30)],
        decisions=[
            DecisionItem(decision=f"ตัดสินใจ {i}", rationale="น" * 300)
            for i in range(30)
        ],
        action_items=[
            ActionItem(action=f"ทำงาน {i}", owner="แมท", due="พรุ่งนี้")
            for i in range(40)
        ],
        open_questions=[f"คำถามที่ยังไม่มีคำตอบข้อ {i}" for i in range(30)],
        raw="",
    )


def _embed_char_total(embed) -> int:
    total = len(embed.title or "")
    total += len(embed.description or "")
    for field in embed.fields:
        total += len(field.name) + len(field.value)
    total += len(embed.footer.text or "")
    return total


def test_build_embed_never_exceeds_discord_total():
    """A legitimately rich meeting must stay under Discord's 6000-char total.

    Regression for the fixed independent caps (_DESCRIPTION_MAX=2000 +
    4x_FIELD_MAX=1024 = 6096, already over before title/names/footer): the
    shared pool must shrink everything proportionally instead.
    """
    embed = poster.build_embed(
        _rich_summary(),
        started_at=poster.datetime(2026, 8, 10, 12, 0),
        duration=poster.timedelta(minutes=47, seconds=12),
        member_count=5,
        meeting_title="การประชุมทีมยาวมากเพื่อทดสอบขีดจำกัดรวมของ embed",
    )
    assert _embed_char_total(embed) <= 6000
    # The scannable structure is preserved: all four sections still present.
    assert len(embed.fields) == 4
    assert [f.name for f in embed.fields] == list(poster._FIELD_NAMES)


def test_build_embed_short_summary_untruncated():
    """When content is small, nothing is needlessly truncated."""
    from meeting_bot.summary_parse import ActionItem, Summary

    summary = Summary(
        overview="ประชุมสั้น ๆ",
        topics=[],
        decisions=[],
        action_items=[ActionItem(action="ส่งรายงาน", owner="แมท", due=None)],
        open_questions=[],
        raw="",
    )
    embed = poster.build_embed(
        summary,
        started_at=poster.datetime(2026, 8, 10, 12, 0),
        duration=poster.timedelta(minutes=3),
        member_count=2,
        meeting_title="สั้น",
    )
    assert embed.description == "ประชุมสั้น ๆ"
    assert "ส่งรายงาน" in embed.fields[2].value
    assert "ดูรายละเอียดเพิ่มเติม" not in embed.description
    assert _embed_char_total(embed) <= 6000
