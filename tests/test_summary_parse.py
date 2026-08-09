"""Pure-logic tests for meeting_bot.summary_parse (no Discord/network)."""

from meeting_bot.summary_parse import (
    ActionItem,
    DecisionItem,
    Summary,
    TopicItem,
    parse_summary,
)


def test_valid_json():
    text = (
        '{"overview": "ประชุมเรื่องงบประมาณ", '
        '"topics": [{"title": "งบประมาณ", "detail": "พูดคุยรายละเอียด"}], '
        '"decisions": [{"decision": "อนุมัติงบ", "rationale": "ตัวเลขตรงแผน"}], '
        '"action_items": [{"action": "ส่งรายงาน", "owner": "สมชาย", "due": "วันศุกร์"}], '
        '"open_questions": ["ใครทำเอกสาร?"]}'
    )
    summary = parse_summary(text)
    assert summary.overview == "ประชุมเรื่องงบประมาณ"
    assert summary.topics == [TopicItem(title="งบประมาณ", detail="พูดคุยรายละเอียด")]
    assert summary.decisions == [
        DecisionItem(decision="อนุมัติงบ", rationale="ตัวเลขตรงแผน")
    ]
    assert len(summary.action_items) == 1
    assert summary.action_items[0].action == "ส่งรายงาน"
    assert summary.action_items[0].owner == "สมชาย"
    assert summary.action_items[0].due == "วันศุกร์"
    assert summary.open_questions == ["ใครทำเอกสาร?"]


def test_fenced_json_with_prose():
    text = (
        "นี่คือสรุปการประชุม:\n"
        '```json\n{"overview": "ภาพรวม", '
        '"topics": [{"title": "หัวข้อแรก", "detail": "รายละเอียด"}], '
        '"decisions": [], '
        '"action_items": [{"action": "ทำ X", "owner": null, "due": null}], '
        '"open_questions": []}\n```\nจบครับ'
    )
    summary = parse_summary(text)
    assert summary.overview == "ภาพรวม"
    assert summary.topics == [TopicItem(title="หัวข้อแรก", detail="รายละเอียด")]
    assert summary.decisions == []
    assert summary.action_items[0].action == "ทำ X"
    assert summary.action_items[0].owner is None
    assert summary.action_items[0].due is None
    assert summary.open_questions == []


def test_english_keys_and_string_action_items():
    text = (
        '{"overview": "Quick sync", '
        '"topics": [{"title": "T1", "detail": "d1"}, {"title": "T2"}], '
        '"decisions": [{"decision": "D1", "rationale": "r1"}], '
        '"action_items": ["Buy milk", "Call back"], '
        '"open_questions": ["Q?"]}'
    )
    summary = parse_summary(text)
    assert summary.overview == "Quick sync"
    assert summary.topics[0] == TopicItem(title="T1", detail="d1")
    assert summary.topics[1] == TopicItem(title="T2", detail="")
    assert summary.decisions == [DecisionItem(decision="D1", rationale="r1")]
    assert [ai.action for ai in summary.action_items] == ["Buy milk", "Call back"]
    assert all(ai.owner is None for ai in summary.action_items)
    assert summary.open_questions == ["Q?"]


def test_backward_compat_plain_string_topics_and_decisions():
    # Old schema: topics/decisions as bare strings must still parse, wrapped
    # into TopicItem/DecisionItem with empty detail/rationale.
    text = (
        '{"topics": ["งบประมาณ", "Q3 แผน"], "decisions": ["อนุมัติงบ"], '
        '"action_items": [{"action": "ส่งรายงาน", "owner": "สมชาย"}]}'
    )
    summary = parse_summary(text)
    assert summary.overview == ""
    assert summary.topics == [TopicItem(title="งบประมาณ"), TopicItem(title="Q3 แผน")]
    assert summary.decisions == [DecisionItem(decision="อนุมัติงบ")]
    assert len(summary.action_items) == 1
    assert summary.action_items[0].action == "ส่งรายงาน"
    assert summary.action_items[0].owner == "สมชาย"
    assert summary.action_items[0].due is None
    assert summary.open_questions == []


def test_thai_keys_json():
    text = (
        '{"ภาพรวม": "สรุป", '
        '"หัวข้อ": [{"หัวข้อ": "ก", "รายละเอียด": "ข"}], '
        '"การตัดสินใจ": [{"การตัดสินใจ": "ค", "เหตุผล": "ง"}], '
        '"รายการที่ต้องทำ": [{"action": "จ", "owner": "ฉ", "กำหนดเวลา": "พรุ่งนี้"}], '
        '"คำถามที่ยังไม่ได้ข้อสรุป": ["ช"]}'
    )
    summary = parse_summary(text)
    assert summary.overview == "สรุป"
    assert summary.topics == [TopicItem(title="ก", detail="ข")]
    assert summary.decisions == [DecisionItem(decision="ค", rationale="ง")]
    assert summary.action_items[0].action == "จ"
    assert summary.action_items[0].owner == "ฉ"
    assert summary.action_items[0].due == "พรุ่งนี้"
    assert summary.open_questions == ["ช"]


def test_markdown_fallback():
    text = (
        "# ภาพรวม\n"
        "ประชุมกันเรื่องแผนงานประจำไตรมาส โดยมีทั้งเรื่องงบประมาณและทีมงาน\n\n"
        "## หัวข้อ\n"
        "- **งบประมาณ** — พูดคุยเรื่องงบประมาณปีหน้า\n"
        "- แผนงาน Q3\n\n"
        "## การตัดสินใจ\n"
        "- **อนุมัติงบประมาณ** — ตัวเลขตรงกับแผน\n\n"
        "## สิ่งที่ต้องทำ\n"
        "- สมชาย: ส่งรายงาน\n\n"
        "## คำถามที่ยังไม่ได้ข้อสรุป\n"
        "- ใครรับผิดชอบเอกสาร?\n"
    )
    summary = parse_summary(text)
    assert "ประชุมกันเรื่องแผนงาน" in summary.overview
    assert TopicItem(title="งบประมาณ", detail="พูดคุยเรื่องงบประมาณปีหน้า") in summary.topics
    assert "แผนงาน Q3" in [t.title for t in summary.topics]
    assert DecisionItem(decision="อนุมัติงบประมาณ", rationale="ตัวเลขตรงกับแผน") in summary.decisions
    assert len(summary.action_items) == 1
    assert summary.action_items[0].owner == "สมชาย"
    assert summary.action_items[0].action == "ส่งรายงาน"
    assert summary.open_questions == ["ใครรับผิดชอบเอกสาร?"]


def test_garbage_last_resort_never_raises():
    raw = "这不是有效的 JSON หรือ Markdown !!!@@@"
    summary = parse_summary(raw)
    assert isinstance(summary, Summary)
    assert summary.overview == raw
    assert summary.topics == []
    assert summary.decisions == []
    assert summary.action_items == []
    assert summary.open_questions == []


def test_empty_never_raises():
    summary = parse_summary("")
    assert isinstance(summary, Summary)
    assert summary.overview == ""
    assert summary.topics == []
    assert summary.decisions == []
    assert summary.action_items == []
    assert summary.open_questions == []


def test_action_item_parse_plain_string():
    item = ActionItem.parse("ทำรายงาน")
    assert item.action == "ทำรายงาน"
    assert item.owner is None
    assert item.due is None


def test_action_item_parse_with_due():
    item = ActionItem.parse({"action": "ส่งงาน", "owner": "แมท", "due": "วันจันทร์"})
    assert item.action == "ส่งงาน"
    assert item.owner == "แมท"
    assert item.due == "วันจันทร์"
