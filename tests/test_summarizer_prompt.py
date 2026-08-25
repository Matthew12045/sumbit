"""Pure-logic tests for the summarizer's dynamic system prompt (no network).

Covers grounding-rule sentinels, the digit-free base prompt, size-tier
boundaries, spec monotonicity, rendered scaling, the M-tier legacy-wording
pin, the kwargs ``_summarize_once`` passes to the gateway, and the
``tools/gateway_test_matrix.py`` import contract.
"""

from types import SimpleNamespace

import meeting_bot.summarizer as sm


_SENTINELS = (
    "ห้ามเดา",
    "ห้ามสร้าง",
    "ให้ใช้ null",
    "ห้ามเพิ่มคำแนะนำ",
    "ค่าว่าง",
    "markdown fence",
)


def _tier_midpoints() -> dict[str, int]:
    return {
        sm._TIER_XS: sm._TIER_XS_MAX_CHARS // 2,
        sm._TIER_S: (sm._TIER_XS_MAX_CHARS + sm._TIER_S_MAX_CHARS) // 2,
        sm._TIER_M: (sm._TIER_S_MAX_CHARS + sm._TIER_M_MAX_CHARS) // 2,
        sm._TIER_L: sm._TIER_M_MAX_CHARS * 2,
    }


def test_grounding_rules_present_in_base_and_all_tiers():
    for sentinel in _SENTINELS:
        assert sentinel in sm._SYSTEM_PROMPT
    for n_chars in _tier_midpoints().values():
        prompt = sm._system_prompt_for("x" * n_chars)
        assert prompt.startswith(sm._SYSTEM_PROMPT)
        for sentinel in _SENTINELS:
            assert sentinel in prompt


def test_base_prompt_contains_no_digits():
    assert not any(ch.isdigit() for ch in sm._SYSTEM_PROMPT)


def test_size_tier_boundaries():
    assert sm._size_tier("") == sm._TIER_XS
    assert sm._size_tier("x") == sm._TIER_XS
    assert sm._size_tier("x" * sm._TIER_XS_MAX_CHARS) == sm._TIER_S
    assert sm._size_tier("x" * (sm._TIER_S_MAX_CHARS - 1)) == sm._TIER_S
    assert sm._size_tier("x" * sm._TIER_S_MAX_CHARS) == sm._TIER_M
    assert sm._size_tier("x" * (sm._TIER_M_MAX_CHARS - 1)) == sm._TIER_M
    assert sm._size_tier("x" * sm._TIER_M_MAX_CHARS) == sm._TIER_L
    assert sm._size_tier("x" * 200_000) == sm._TIER_L


def test_tier_specs_monotonic():
    keys = list(sm._TIER_SPECS[sm._TIER_ORDER[0]])
    assert set(sm._TIER_SPECS) == set(sm._TIER_ORDER)
    for key in keys:
        values = [sm._TIER_SPECS[t][key] for t in sm._TIER_ORDER]
        assert values == sorted(values), key
    for t in sm._TIER_ORDER:
        s = sm._TIER_SPECS[t]
        assert s["overview_min"] <= s["overview_max"]
        assert s["topic_detail_min"] <= s["topic_detail_max"]
        assert s["rationale_min"] <= s["rationale_max"]


def test_system_prompt_for_scales_monotonically():
    prompts = {
        tier: sm._system_prompt_for("x" * n)
        for tier, n in _tier_midpoints().items()
    }
    for prompt in prompts.values():
        assert prompt.startswith(sm._SYSTEM_PROMPT)
        assert len(prompt) > len(sm._SYSTEM_PROMPT)
    assert len(set(prompts.values())) == 4
    # Raw appended-block length grows from S upward; the XS label is
    # deliberately wordier than S's, so XS->S is covered by the ceiling
    # monotonicity in test_tier_specs_monotonic instead.
    lens = [len(sm._render_tier_instructions(t)) for t in sm._TIER_ORDER[1:]]
    assert lens == sorted(lens)


def test_rendered_tier_numbers_match_spec():
    rendered = sm._render_tier_instructions(sm._TIER_M)
    assert "3-6 ประโยค" in rendered
    assert "ไม่เกิน 8 หัวข้อ" in rendered


_FAKE_COMPLETION_TEXT = (
    '{"overview": "สรุปการประชุม", "topics": [], "decisions": [], '
    '"action_items": [], "open_questions": []}'
)


def _summarizer_capturing(captured: list[dict]) -> sm.Summarizer:
    """Summarizer with a fake client capturing messages.create(**kwargs)."""
    s = object.__new__(sm.Summarizer)  # bypass __init__: no real anthropic client
    s.cfg = SimpleNamespace(
        gateway_model="qwen3.8-27b-fp8",
        summary_max_tokens=8192,
        repetition_window_chars=300,
        repetition_min_repeats=3,
    )
    s._anthropic = SimpleNamespace(APITimeoutError=RuntimeError)

    class _Messages:
        def create(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=_FAKE_COMPLETION_TEXT)],
                stop_reason="stop",
            )

    s._client = SimpleNamespace(messages=_Messages())
    return s


def test_summarize_once_passes_dynamic_system_prompt():
    captured: list[dict] = []
    s = _summarizer_capturing(captured)
    short = "[00:10] สมชาย: เริ่มการประชุมครับ\n" * 6   # XS-sized input
    long = "[00:10] ปณิธาน: ตรวจสอบรายการสินค้า\n" * 1200  # >30k chars -> L
    assert len(short.strip()) < sm._TIER_XS_MAX_CHARS
    assert len(long) > sm._TIER_M_MAX_CHARS

    s.summarize(short)
    s.summarize(long)

    assert len(captured) == 2
    first, second = captured
    assert first["system"] == sm._system_prompt_for(short)
    assert second["system"] == sm._system_prompt_for(long)
    assert first["system"] != second["system"]
    assert first["messages"] == [
        {"role": "user", "content": short + sm._USER_SUFFIX}
    ]
    assert second["messages"][0]["content"] == long + sm._USER_SUFFIX
    assert first["temperature"] == 0.0
    assert second["temperature"] == 0.0
    assert first["model"] == "qwen3.8-27b-fp8"
    assert first["max_tokens"] == 8192


def test_gateway_test_matrix_compat():
    from meeting_bot.summarizer import _SYSTEM_PROMPT, _USER_SUFFIX

    assert isinstance(_SYSTEM_PROMPT, str)
    assert isinstance(_USER_SUFFIX, str)
    assert "JSON" in _USER_SUFFIX
    assert "null" in _USER_SUFFIX
