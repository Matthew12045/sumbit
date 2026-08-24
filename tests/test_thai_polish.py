"""Pure-logic tests for meeting_bot.thai_polish (no network).

Uses a fake OpenAI client to mock the polish API calls.  The ``openai``
package is installed (via ``openai`` in requirements.txt) so we can import
the module; we just never touch the network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from meeting_bot.thai_polish import (
    ThaiPolisher,
    _parse_polished,
    _PolishInput,
    _PolishOutput,
    _reconstruct,
    _fields,
    _blanks_text,
    _input_from_output,
    _to_input,
    _to_output,
)
from meeting_bot.summary_parse import (
    ActionItem,
    DecisionItem,
    Summary,
    TopicItem,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_summary(**overrides: Any) -> Summary:
    defaults = Summary(
        overview="ภาพรวมของการประชุมวันนี้",
        topics=[TopicItem(title="งบประมาณ", detail="พิจารณางบประมาณปหนา")],
        decisions=[DecisionItem(decision="อนุมัติงบประมาณ", rationale="เพราะจําเปนตองใชงาน")],
        action_items=[ActionItem(action="เตรียมนําเสนอ", owner="สมชาย", due="2569-01-15")],
        open_questions=["แผนสำรองเปนอย่างไร"],
        raw=(
            "**ภาพรวม**\n\nภาพรวมของการประชุมวันนี้\n\n"
            "**หัวข้อ**\n\n- **งบประมาณ** — พิจารณางบประมาณปหนา\n\n"
            "**การตัดสินใจ**\n\n- **อนุมัติงบประมาณ** — เพราะจําเปนตองใชงาน\n\n"
            "**รายการที่ตองดําเนินการ**\n\n"
            "| รายการ | ผูป้ ฏิบัติ | กำหนดการ |\n"
            "|--------|-----------|----------|\n"
            "| เตรียมนําเสนอ | สมชาย | 2569-01-15 |\n\n"
            "**คำถามที่ยังไมไดข้อสรุป**\n\n- แผนสำรองเปนอย่างไร\n"
        ),
    )
    for k, v in overrides.items():
        setattr(defaults, k, v)
    return defaults


def _make_polisher(fake_responses: list[str] | None = None) -> ThaiPolisher:
    """Build a ThaiPolisher with a fake OpenAI client."""
    p = ThaiPolisher(
        base_url="http://localhost:9999/v1",
        auth_token="test-key",
        model="test-model",
        max_passes=3,
        timeout_seconds=10.0,
    )
    p._skill_bundle = "dummy skill bundle"

    if fake_responses is None:
        # Default: report no edits needed immediately (converged in 1 pass)
        fake_responses = [
            json.dumps({
                "overview": "ภาพรวมของการประชุมวันนี้",
                "topics": [{"title": "งบประมาณ", "detail": "พิจารณางบประมาณปหนา"}],
                "decisions": [{"decision": "อนุมัติงบประมาณ", "rationale": "เพราะจําเปนตองใชงาน"}],
                "action_items": [{"action": "เตรียมนําเสนอ", "owner": "สมชาย", "due": "2569-01-15"}],
                "open_questions": ["แผนสำรองเปนอย่างไร"],
                "edits_needed": False,
                "edit_notes": [],
            }, ensure_ascii=False),
        ]

    call_count = 0

    def fake_create(*args, **kwargs):
        nonlocal call_count
        content = fake_responses[min(call_count, len(fake_responses) - 1)]
        call_count += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    p._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    return p


# ---------------------------------------------------------------------------
# _to_input / _to_output round-trip
# ---------------------------------------------------------------------------

class TestToInput:
    def test_maps_all_fields(self):
        s = _make_summary()
        inp = _to_input(s)
        assert inp.overview == "ภาพรวมของการประชุมวันนี้"
        assert len(inp.topics) == 1
        assert inp.topics[0]["title"] == "งบประมาณ"
        assert inp.topics[0]["detail"] == "พิจารณางบประมาณปหนา"
        assert len(inp.decisions) == 1
        assert inp.decisions[0]["rationale"] == "เพราะจําเปนตองใชงาน"
        assert len(inp.action_items) == 1
        assert inp.action_items[0]["owner"] == "สมชาย"
        assert len(inp.open_questions) == 1

    def test_empty_summary(self):
        s = Summary(
            overview="", topics=[], decisions=[],
            action_items=[], open_questions=[], raw="",
        )
        inp = _to_input(s)
        assert inp.overview == ""
        assert inp.topics == []
        assert inp.open_questions == []


class TestToOutput:
    def test_parses_edits_needed_true(self):
        data = {
            "overview": "x", "topics": [], "decisions": [],
            "action_items": [], "open_questions": [],
            "edits_needed": True, "edit_notes": ["fix grammar"],
        }
        out = _to_output(data)
        assert out.edits_needed is True
        assert out.edit_notes == ["fix grammar"]

    def test_parses_edits_needed_false(self):
        data = {
            "overview": "x", "topics": [], "decisions": [],
            "action_items": [], "open_questions": [],
            "edits_needed": False, "edit_notes": [],
        }
        out = _to_output(data)
        assert out.edits_needed is False


# ---------------------------------------------------------------------------
# _reconstruct
# ---------------------------------------------------------------------------

class TestReconstruct:
    def test_rebuilds_markdown_structure(self):
        s = _make_summary()
        polished = _PolishOutput(
            overview="ภาพรวมใหม่",
            topics=[{"title": "งบประมาณ", "detail": "รายละเอียดใหม่"}],
            decisions=[{"decision": "อนุมัติ", "rationale": "เหตุผลใหม่"}],
            action_items=[{"action": "เตรียมนําเสนอ", "owner": "สมชาย", "due": "2569-01-15"}],
            open_questions=["แผนสำรองเปนอย่างไร"],
            edits_needed=False, edit_notes=[],
        )
        result = _reconstruct(s, polished)
        # result is a Summary; check its raw markdown and structured fields
        assert "ภาพรวมใหม่" in result.raw
        assert "รายละเอียดใหม่" in result.raw
        assert "เหตุผลใหม่" in result.raw
        assert result.overview == "ภาพรวมใหม่"
        assert len(result.topics) == 1
        assert result.topics[0].detail == "รายละเอียดใหม่"
        # Action items and open questions kept from original
        assert len(result.action_items) == 1
        assert result.action_items[0].action == "เตรียมนําเสนอ"
        assert len(result.open_questions) == 1
        assert result.open_questions[0] == "แผนสำรองเปนอย่างไร"

    def test_empty_polished_fields_keeps_structure(self):
        s = _make_summary()
        polished = _PolishOutput(
            overview="", topics=[], decisions=[],
            action_items=[], open_questions=[],
            edits_needed=False, edit_notes=[],
        )
        result = _reconstruct(s, polished)
        # Should still have action items and open questions from original
        assert len(result.action_items) == 1
        assert result.action_items[0].action == "เตรียมนําเสนอ"
        assert len(result.open_questions) == 1
        assert result.open_questions[0] == "แผนสำรองเปนอย่างไร"


# ---------------------------------------------------------------------------
# Field comparison (replaces the old raw-string _summaries_equal check)
# ---------------------------------------------------------------------------

class TestFieldComparison:
    def test_fields_identical(self):
        inp = _to_input(_make_summary())
        assert _fields(inp) == _fields(_to_input(_make_summary()))

    def test_fields_detect_overview_change(self):
        s = _make_summary()
        fields_before = _fields(_to_input(s))
        changed = _PolishInput(
            overview="เปลี่ยนแล้ว",
            topics=_to_input(s).topics,
            decisions=_to_input(s).decisions,
            action_items=_to_input(s).action_items,
            open_questions=_to_input(s).open_questions,
        )
        assert _fields(changed) != fields_before

    def test_blanks_text_detects_emptied_field(self):
        prev = ("มีข้อความ", ("รายละเอียด",), ())
        nxt = ("", ("รายละเอียด",), ())
        assert _blanks_text(prev, nxt) is True

    def test_blanks_text_allows_real_edits(self):
        prev = ("ข้อความเดิม", ("รายละเอียดเดิม",), ())
        nxt = ("ข้อความใหม่", ("รายละเอียดใหม่",), ())
        assert _blanks_text(prev, nxt) is False

    def test_blanks_text_empty_to_empty_ok(self):
        assert _blanks_text(("", (), ()), ("", (), ())) is False


# ---------------------------------------------------------------------------
# _parse_polished
# ---------------------------------------------------------------------------

class TestParsePolished:
    def test_plain_json(self):
        raw = json.dumps({
            "overview": "x", "topics": [], "decisions": [],
            "action_items": [], "open_questions": [],
            "edits_needed": False, "edit_notes": [],
        }, ensure_ascii=False)
        out = _parse_polished(raw, 1)
        assert out.overview == "x"
        assert out.edits_needed is False

    def test_markdown_fence_stripping(self):
        raw = '```json\n{"overview": "y", "topics": [], "decisions": [], "action_items": [], "open_questions": [], "edits_needed": false, "edit_notes": []}\n```'
        out = _parse_polished(raw, 1)
        assert out.overview == "y"

    def test_no_fence(self):
        raw = '{"overview": "z", "topics": [], "decisions": [], "action_items": [], "open_questions": [], "edits_needed": false, "edit_notes": []}'
        out = _parse_polished(raw, 1)
        assert out.overview == "z"

    def test_garbage_raises(self):
        """Malformed JSON must raise (never masquerade as an empty no-edits
        result — that was the data-loss bug)."""
        import pytest

        with pytest.raises(ValueError):
            _parse_polished("not json at all", 1)

    def test_wrong_topic_count_raises(self):
        raw = json.dumps({
            "overview": "x",
            "topics": [{"title": "a", "detail": "b"}],
            "decisions": [],
            "action_items": [],
            "open_questions": [],
            "edits_needed": False,
            "edit_notes": [],
        }, ensure_ascii=False)
        expected = _PolishInput(overview="orig", topics=[], decisions=[], action_items=[], open_questions=[])
        import pytest

        with pytest.raises(ValueError):
            _parse_polished(raw, 1, expected=expected)

    def test_wrong_decision_count_raises(self):
        """Both directions: fewer AND more decisions than the input had."""
        import pytest

        expected = _PolishInput(
            overview="orig",
            topics=[],
            decisions=[{"decision": "c", "rationale": ""}],
            action_items=[],
            open_questions=[],
        )
        fewer = json.dumps({
            "overview": "x",
            "topics": [],
            "decisions": [],
            "action_items": [],
            "open_questions": [],
            "edits_needed": False,
            "edit_notes": [],
        }, ensure_ascii=False)
        more = json.dumps({
            "overview": "x",
            "topics": [],
            "decisions": [
                {"decision": "c", "rationale": ""},
                {"decision": "invented", "rationale": "junk"},
            ],
            "action_items": [],
            "open_questions": [],
            "edits_needed": False,
            "edit_notes": [],
        }, ensure_ascii=False)
        with pytest.raises(ValueError):
            _parse_polished(fewer, 1, expected=expected)
        with pytest.raises(ValueError):
            _parse_polished(more, 1, expected=expected)

    def test_matching_shape_parses(self):
        raw = json.dumps({
            "overview": "x",
            "topics": [{"title": "a", "detail": "b"}],
            "decisions": [{"decision": "c", "rationale": "d"}],
            "action_items": [],
            "open_questions": [],
            "edits_needed": True,
            "edit_notes": [],
        }, ensure_ascii=False)
        expected = _PolishInput(
            overview="orig",
            topics=[{"title": "a", "detail": "b"}],
            decisions=[{"decision": "c", "rationale": ""}],
            action_items=[],
            open_questions=[],
        )
        out = _parse_polished(raw, 1, expected=expected)
        assert out.edits_needed is True
        assert len(out.topics) == 1

    def test_wrong_types_raise(self):
        import pytest

        base = {
            "overview": "x", "topics": [], "decisions": [],
            "action_items": [], "open_questions": [],
            "edits_needed": False, "edit_notes": [],
        }
        # topics not a list
        with pytest.raises(ValueError):
            _to_output({**base, "topics": "nope"})
        # edits_needed not a bool
        with pytest.raises(ValueError):
            _to_output({**base, "edits_needed": "yes"})
        # overview not a string
        with pytest.raises(ValueError):
            _to_output({**base, "overview": 42})
        # non-object payload
        with pytest.raises(ValueError):
            _to_output(["list", "payload"])
        # topic items must be dicts
        with pytest.raises(ValueError):
            _to_output({**base, "topics": ["plain string"]})


# ---------------------------------------------------------------------------
# Full polish loop — convergence
# ---------------------------------------------------------------------------

class TestPolishConvergence:
    def test_converges_immediately(self):
        """If the first pass reports no edits, return the polished version."""
        s = _make_summary()
        p = _make_polisher()
        result = p.polish(s)
        # The rebuilt Summary must carry the original's polished fields.
        assert _fields(_to_input(result)) == _fields(_to_input(s))
        assert p.last_stats == {"passes": 1, "outcome": "converged"}

    def test_converges_after_edits(self):
        """First pass suggests edits, second pass converges."""
        s = _make_summary()
        responses = [
            # Pass 1: edits needed
            json.dumps({
                "overview": "ภาพรวมที่ปรบปรุงแล้ว",
                "topics": [{"title": "งบประมาณ", "detail": "พิจารณางบประมาณปหนา"}],
                "decisions": [{"decision": "อนุมัติงบประมาณ", "rationale": "เพราะจําเปนตองใชงาน"}],
                "action_items": [{"action": "เตรียมนําเสนอ", "owner": "สมชาย", "due": "2569-01-15"}],
                "open_questions": ["แผนสำรองเปนอย่างไร"],
                "edits_needed": True,
                "edit_notes": ["ปรับปรุงภาษา"],
            }, ensure_ascii=False),
            # Pass 2: no more edits
            json.dumps({
                "overview": "ภาพรวมที่ปรบปรุงแล้ว",
                "topics": [{"title": "งบประมาณ", "detail": "พิจารณางบประมาณปหนา"}],
                "decisions": [{"decision": "อนุมัติงบประมาณ", "rationale": "เพราะจําเปนตองใชงาน"}],
                "action_items": [{"action": "เตรียมนําเสนอ", "owner": "สมชาย", "due": "2569-01-15"}],
                "open_questions": ["แผนสำรองเปนอย่างไร"],
                "edits_needed": False,
                "edit_notes": [],
            }, ensure_ascii=False),
        ]
        p = _make_polisher(fake_responses=responses)
        result = p.polish(s)
        # Pass 1's edited text was accepted; pass 2 saw zero diffs.
        assert result.overview == "ภาพรวมที่ปรบปรุงแล้ว"
        assert _fields(_to_input(result)) == (
            "ภาพรวมที่ปรบปรุงแล้ว",
            ("พิจารณางบประมาณปหนา",),
            ("เพราะจําเปนตองใชงาน",),
        )
        assert p.last_stats == {"passes": 2, "outcome": "converged"}

    def test_safety_cap_returns_original(self):
        """If max_passes is hit without convergence, return the original."""
        s = _make_summary()
        # Every pass proposes DIFFERENT text and claims edits_needed=true,
        # so field comparison never sees a zero diff and the loop runs to
        # the cap (the fake client clamps to its last response, so supply
        # strictly more unique responses than max_passes).
        alternating = [
            json.dumps({
                "overview": f"v{i}", "topics": [{"title": "a", "detail": f"d{i}"}],
                "decisions": [{"decision": "c", "rationale": "d"}],
                "action_items": [{"action": "e", "owner": "f", "due": "g"}],
                "open_questions": ["h"],
                "edits_needed": True,
                "edit_notes": ["keep editing"],
            }, ensure_ascii=False)
            for i in range(8)
        ]
        p = _make_polisher(fake_responses=alternating)
        result = p.polish(s)
        # Should return original on safety cap
        assert result is s
        assert p.last_stats == {"passes": 3, "outcome": "cap"}

    def test_stable_text_converges_even_when_edits_claimed(self):
        """Zero field diffs = converged, regardless of edits_needed."""
        s = _make_summary()
        unchanged_but_claiming = json.dumps({
            "overview": "ภาพรวมของการประชุมวันนี้",
            "topics": [{"title": "งบประมาณ", "detail": "พิจารณางบประมาณปหนา"}],
            "decisions": [{"decision": "อนุมัติงบประมาณ", "rationale": "เพราะจําเปนตองใชงาน"}],
            "action_items": [{"action": "เตรียมนําเสนอ", "owner": "สมชาย", "due": "2569-01-15"}],
            "open_questions": ["แผนสำรองเปนอย่างไร"],
            "edits_needed": True,
            "edit_notes": [],
        }, ensure_ascii=False)
        p = _make_polisher(fake_responses=[unchanged_but_claiming])
        result = p.polish(s)
        assert result is not s  # rebuilt from identical fields
        assert _fields(_to_input(result)) == _fields(_to_input(s))


# ---------------------------------------------------------------------------
# Regression tests: the old data-loss bug
# ---------------------------------------------------------------------------

class TestParseFailureFallsBackToOriginal:
    def test_malformed_json_pass_returns_original(self):
        """A garbage pass must return the ORIGINAL summary — not an empty one.

        Regression: _parse_polished used to return an empty edits_needed=False
        output on JSONDecodeError, which polish() then rebuilt into a gutted
        empty summary.
        """
        s = _make_summary()
        p = _make_polisher(fake_responses=["this is not json"])
        result = p.polish(s)
        assert result is s
        assert result.overview == "ภาพรวมของการประชุมวันนี้"
        assert len(result.topics) == 1
        assert p.last_stats == {"passes": 1, "outcome": "hard_failure"}

    def test_shape_mismatch_returns_original(self):
        """A pass that drops topics must fall back to the original."""
        s = _make_summary()  # has exactly 1 topic
        dropped_topic = json.dumps({
            "overview": "x",
            "topics": [],
            "decisions": [],
            "action_items": [],
            "open_questions": [],
            "edits_needed": False,
            "edit_notes": [],
        }, ensure_ascii=False)
        p = _make_polisher(fake_responses=[dropped_topic])
        assert p.polish(s) is s
        assert p.last_stats == {"passes": 1, "outcome": "hard_failure"}

    def test_extra_decisions_returns_original(self):
        """A pass that INVENTS decisions (count > input) must also fall back —
        this mirrors the real OpenTyphoon failure seen in e2e probing where
        open questions were re-classified as extra decisions."""
        s = _make_summary()  # has exactly 1 decision
        inflated = json.dumps({
            "overview": "x",
            "topics": [{"title": "งบประมาณ", "detail": "พิจารณางบประมาณปหนา"}],
            "decisions": [
                {"decision": "อนุมัติงบประมาณ", "rationale": "เพราะจําเปนตองใชงาน"},
                {"decision": "ประเด็นแปลกปลอม", "rationale": "ไม่ได้อยู่ในอินพุต"},
            ],
            "action_items": [],
            "open_questions": [],
            "edits_needed": False,
            "edit_notes": [],
        }, ensure_ascii=False)
        p = _make_polisher(fake_responses=[inflated])
        assert p.polish(s) is s
        assert p.last_stats == {"passes": 1, "outcome": "hard_failure"}

    def test_blanked_out_text_returns_original(self):
        """edits_needed=false with emptied fields must NOT be accepted."""
        s = _make_summary()
        blanking = json.dumps({
            "overview": "",
            "topics": [{"title": "งบประมาณ", "detail": ""}],
            "decisions": [{"decision": "อนุมัติงบประมาณ", "rationale": ""}],
            "action_items": [{"action": "เตรียมนําเสนอ", "owner": "สมชาย", "due": "2569-01-15"}],
            "open_questions": ["แผนสำรองเปนอย่างไร"],
            "edits_needed": False,
            "edit_notes": [],
        }, ensure_ascii=False)
        p = _make_polisher(fake_responses=[blanking])
        result = p.polish(s)
        assert result is s
        assert p.last_stats == {"passes": 1, "outcome": "blanked"}

    def test_protected_fields_survive_mangled_output(self):
        """Titles/labels/action_items/open_questions come from the ORIGINAL,
        no matter what the model returns for them."""
        s = _make_summary()
        mangled = json.dumps({
            "overview": "ภาพรวมที่ขัดเกลาแล้ว",
            # model renamed/mangled the title and returned extra junk keys
            "topics": [{"title": "หัวข้อผิด", "detail": "พิจารณางบประมาณปหนา"}],
            "decisions": [{"decision": "คำตัดสินผิด", "rationale": "เพราะจําเปนตองใชงาน"}],
            "action_items": [],   # model dropped them — still passed through
            "open_questions": [], # model dropped them — still passed through
            "edits_needed": False,
            "edit_notes": [],
        }, ensure_ascii=False)
        p = _make_polisher(fake_responses=[mangled])
        result = p.polish(s)
        assert result.overview == "ภาพรวมที่ขัดเกลาแล้ว"
        assert result.topics[0].title == "งบประมาณ"  # protected title
        assert result.decisions[0].decision == "อนุมัติงบประมาณ"  # protected label
        # Passed through BYTE-IDENTICAL to the original items, not merely
        # same-length: the model returned [] for both.
        assert result.action_items == s.action_items
        assert result.open_questions == s.open_questions


# ---------------------------------------------------------------------------
# last_stats observability
# ---------------------------------------------------------------------------

class TestLastStats:
    """ThaiPolisher.last_stats must record {passes, outcome} for every path."""

    def test_skipped_on_empty_raw(self):
        s = Summary(
            overview="", topics=[], decisions=[],
            action_items=[], open_questions=[], raw="",
        )
        p = _make_polisher()
        assert p.polish(s) is s
        assert p.last_stats == {"passes": 0, "outcome": "skipped"}

    def test_final_edit_outcome(self):
        """Fields changed + edits_needed=false (no blanking) = final_edit."""
        s = _make_summary()
        improved = json.dumps({
            "overview": "ภาพรวมที่ขัดเกลาแล้ว",
            "topics": [{"title": "งบประมาณ", "detail": "พิจารณางบประมาณปหนา"}],
            "decisions": [{"decision": "อนุมัติงบประมาณ", "rationale": "เพราะจําเปนตองใชงาน"}],
            "action_items": [{"action": "เตรียมนําเสนอ", "owner": "สมชาย", "due": "2569-01-15"}],
            "open_questions": ["แผนสำรองเปนอย่างไร"],
            "edits_needed": False,
            "edit_notes": [],
        }, ensure_ascii=False)
        p = _make_polisher(fake_responses=[improved])
        result = p.polish(s)
        assert result.overview == "ภาพรวมที่ขัดเกลาแล้ว"
        assert p.last_stats == {"passes": 1, "outcome": "final_edit"}


# ---------------------------------------------------------------------------
# Log emission (kienthai.md: outcomes distinguishable from logs alone)
# ---------------------------------------------------------------------------

class TestLogEmission:
    def _messages(self, caplog, polisher, summary):
        import logging

        with caplog.at_level(logging.INFO, logger="meeting_bot.thai_polish"):
            polisher.polish(summary)
        return [record.message for record in caplog.records]

    def test_convergence_logged(self, caplog):
        msgs = self._messages(caplog, _make_polisher(), _make_summary())
        assert any("converged after 1 pass(es)" in m for m in msgs)

    def test_final_edit_logged(self, caplog):
        improved = json.dumps({
            "overview": "ภาพรวมที่ขัดเกลาแล้ว",
            "topics": [{"title": "งบประมาณ", "detail": "พิจารณางบประมาณปหนา"}],
            "decisions": [{"decision": "อนุมัติงบประมาณ", "rationale": "เพราะจําเปนตองใชงาน"}],
            "action_items": [{"action": "เตรียมนําเสนอ", "owner": "สมชาย", "due": "2569-01-15"}],
            "open_questions": ["แผนสำรองเปนอย่างไร"],
            "edits_needed": False,
            "edit_notes": [],
        }, ensure_ascii=False)
        msgs = self._messages(caplog, _make_polisher(fake_responses=[improved]), _make_summary())
        assert any("accepted final edit after 1 pass(es)" in m for m in msgs)

    def test_safety_cap_logged(self, caplog):
        alternating = [
            json.dumps({
                "overview": f"v{i}", "topics": [{"title": "a", "detail": f"d{i}"}],
                "decisions": [{"decision": "c", "rationale": "d"}],
                "action_items": [{"action": "e", "owner": "f", "due": "g"}],
                "open_questions": ["h"],
                "edits_needed": True,
                "edit_notes": ["keep editing"],
            }, ensure_ascii=False)
            for i in range(8)
        ]
        msgs = self._messages(caplog, _make_polisher(fake_responses=alternating), _make_summary())
        assert any("hit safety cap (3 passes)" in m for m in msgs)

    def test_hard_failure_logged(self, caplog):
        msgs = self._messages(caplog, _make_polisher(fake_responses=["garbage"]), _make_summary())
        assert any("hard failure on pass 1" in m for m in msgs)

    def test_blanked_logged(self, caplog):
        blanking = json.dumps({
            "overview": "",
            "topics": [{"title": "งบประมาณ", "detail": ""}],
            "decisions": [{"decision": "อนุมัติงบประมาณ", "rationale": ""}],
            "action_items": [],
            "open_questions": [],
            "edits_needed": False,
            "edit_notes": [],
        }, ensure_ascii=False)
        msgs = self._messages(caplog, _make_polisher(fake_responses=[blanking]), _make_summary())
        assert any("emptied polished text" in m for m in msgs)


# ---------------------------------------------------------------------------
# API failure fallback
# ---------------------------------------------------------------------------

class TestApiFailureFallback:
    def test_api_error_returns_original(self):
        """If the API raises, the original summary is returned."""
        s = _make_summary()
        p = _make_polisher()

        # Make the fake client raise
        def fake_raise(*args, **kwargs):
            raise RuntimeError("connection refused")

        p._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_raise))
        )

        result = p.polish(s)
        assert result is s
        assert p.last_stats == {"passes": 1, "outcome": "hard_failure"}

    def test_empty_response_returns_original(self):
        """If the model returns None content, return original."""
        s = _make_summary()
        p = _make_polisher()

        def fake_none(*args, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

        p._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_none))
        )

        result = p.polish(s)
        assert result is s


# ---------------------------------------------------------------------------
# Polish disabled
# ---------------------------------------------------------------------------

class TestPolishDisabled:
    def test_polish_module_works_when_disabled(self):
        """The ThaiPolisher class exists and can be instantiated."""
        p = ThaiPolisher(
            base_url="http://localhost:9999/v1",
            auth_token="test",
            model="test",
            max_passes=1,
            timeout_seconds=1.0,
        )
        assert p.max_passes == 1
        assert p.model == "test"
