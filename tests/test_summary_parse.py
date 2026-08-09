"""Pure-logic tests for meeting_bot.summary_parse (no Discord/network)."""

from meeting_bot.summary_parse import ActionItem, Summary, parse_summary


def test_valid_json():
    text = (
        '{"topics": ["งบประมาณ", "Q3 แผน"], "decisions": ["อนุมัติงบ"], '
        '"action_items": [{"action": "ส่งรายงาน", "owner": "สมชาย"}]}'
    )
    summary = parse_summary(text)
    assert summary.topics == ["งบประมาณ", "Q3 แผน"]
    assert summary.decisions == ["อนุมัติงบ"]
    assert len(summary.action_items) == 1
    assert summary.action_items[0].action == "ส่งรายงาน"
    assert summary.action_items[0].owner == "สมชาย"


def test_fenced_json_with_prose():
    text = (
        "นี่คือสรุปการประชุม:\n"
        '```json\n{"topics": ["หัวข้อแรก"], "decisions": [], '
        '"action_items": [{"action": "ทำ X", "owner": null}]}\n```\nจบครับ'
    )
    summary = parse_summary(text)
    assert summary.topics == ["หัวข้อแรก"]
    assert summary.decisions == []
    assert summary.action_items[0].action == "ทำ X"
    assert summary.action_items[0].owner is None


def test_english_keys_and_string_action_items():
    text = (
        '{"topics": ["T1", "T2"], "decisions": ["D1"], '
        '"action_items": ["Buy milk", "Call back"]}'
    )
    summary = parse_summary(text)
    assert summary.topics == ["T1", "T2"]
    assert summary.decisions == ["D1"]
    assert [ai.action for ai in summary.action_items] == ["Buy milk", "Call back"]
    assert all(ai.owner is None for ai in summary.action_items)


def test_thai_keys_json():
    text = (
        '{"หัวข้อ": ["ก"], "การตัดสินใจ": ["ข"], '
        '"รายการที่ต้องทำ": [{"action": "ค", "owner": "ง"}]}'
    )
    summary = parse_summary(text)
    assert summary.topics == ["ก"]
    assert summary.decisions == ["ข"]
    assert summary.action_items[0].action == "ค"
    assert summary.action_items[0].owner == "ง"


def test_markdown_fallback():
    text = (
        "# หัวข้อ\n"
        "- พูดคุยเรื่องงบประมาณ\n"
        "- แผนงาน Q3\n\n"
        "## การตัดสินใจ\n"
        "- อนุมัติงบประมาณ\n\n"
        "## สิ่งที่ต้องทำ\n"
        "- สมชาย: ส่งรายงาน\n"
    )
    summary = parse_summary(text)
    assert "พูดคุยเรื่องงบประมาณ" in summary.topics
    assert "แผนงาน Q3" in summary.topics
    assert "อนุมัติงบประมาณ" in summary.decisions
    assert len(summary.action_items) == 1
    assert summary.action_items[0].owner == "สมชาย"
    assert summary.action_items[0].action == "ส่งรายงาน"


def test_garbage_last_resort_never_raises():
    summary = parse_summary("这不是有效的 JSON หรือ Markdown !!!@@@")
    assert isinstance(summary, Summary)
    assert summary.topics == ["这不是有效的 JSON หรือ Markdown !!!@@@"]
    assert summary.decisions == []
    assert summary.action_items == []


def test_empty_never_raises():
    summary = parse_summary("")
    assert isinstance(summary, Summary)
    assert summary.topics == []
    assert summary.decisions == []
    assert summary.action_items == []


def test_action_item_parse_plain_string():
    item = ActionItem.parse("ทำรายงาน")
    assert item.action == "ทำรายงาน"
    assert item.owner is None
