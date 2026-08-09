"""Pure-logic tests for meeting_bot.summarizer (no Discord/network).

``summarizer`` imports only stdlib + ``.config`` at module scope (anthropic is
lazy), so these run in the minimal test environment.
"""

import pytest
from types import SimpleNamespace

import meeting_bot.summarizer as sm
from meeting_bot.summarizer import StalledGenerationError


# -- _is_looping ---------------------------------------------------------


def test_is_looping_fires_at_threshold():
    # Exactly min_repeats identical trailing windows -> loop.
    assert sm._is_looping("abcabcabc", 3, 3) is True


def test_is_looping_below_threshold_false():
    # Only two windows: len < window * min_repeats.
    assert sm._is_looping("abcabc", 3, 3) is False
    # Three windows but not identical.
    assert sm._is_looping("aabbccaabbcc", 3, 3) is False


def test_is_looping_with_two_repeats():
    assert sm._is_looping("aaaaaa", 3, 2) is True
    assert sm._is_looping("aaa", 3, 2) is False  # len < window * min_repeats


def test_is_looping_guard_clauses():
    assert sm._is_looping("abcabcabc", 0, 3) is False
    assert sm._is_looping("abcabcabc", 3, 1) is False
    assert sm._is_looping("abcabcabc", 3, 4) is False  # len < window * min_repeats


def test_is_looping_no_false_positive_on_repetitive_json():
    # Repeated "owner": keys are legitimate schema output, not a loop.
    buf = (
        '{"action": "a1", "owner": "x"}, '
        '{"action": "a2", "owner": "y"}, '
        '{"action": "a3", "owner": "z"}'
    )
    assert sm._is_looping(buf, 12, 3) is False


# -- StalledGenerationError -----------------------------------------------


def test_stalled_generation_error_tracks_progressed():
    assert StalledGenerationError("stalled", progressed=False).progressed is False
    assert StalledGenerationError("loop", progressed=True).progressed is True


# -- Summarizer retry policy (no anthropic) --------------------------------


def _make_summarizer() -> sm.Summarizer:
    """Build a Summarizer without touching anthropic (bypass __init__)."""
    s = object.__new__(sm.Summarizer)
    s.cfg = SimpleNamespace(summarize_timeout_seconds=180.0)
    s._anthropic = SimpleNamespace(APITimeoutError=RuntimeError)
    s._client = None
    return s


def test_api_timeout_retries_once_then_raises(monkeypatch):
    monkeypatch.setattr(sm.time, "sleep", lambda _seconds: None)
    s = _make_summarizer()
    calls: list[str] = []

    def fake_once(text: str) -> str:
        calls.append(text)
        raise RuntimeError  # stands in for anthropic.APITimeoutError

    s._summarize_once = fake_once
    with pytest.raises(RuntimeError):
        s.summarize("t")
    # First attempt + one retry, then give up.
    assert len(calls) == 2


def test_emptysummary_error_never_retries(monkeypatch):
    monkeypatch.setattr(sm.time, "sleep", lambda _seconds: None)
    s = _make_summarizer()
    calls: list[str] = []

    def fake_once(text: str) -> str:
        calls.append(text)
        raise sm.EmptySummaryError("no text")

    s._summarize_once = fake_once
    with pytest.raises(sm.EmptySummaryError):
        s.summarize("t")
    # The model spent its whole token budget on reasoning; re-running
    # temperature=0 would reproduce it, so never retry.
    assert len(calls) == 1


def test_stalled_generation_loop_never_retries(monkeypatch):
    monkeypatch.setattr(sm.time, "sleep", lambda _seconds: None)
    s = _make_summarizer()
    calls: list[str] = []

    def fake_once(text: str) -> str:
        calls.append(text)
        raise StalledGenerationError("loop detected", progressed=True)

    s._summarize_once = fake_once
    with pytest.raises(StalledGenerationError):
        s.summarize("t")
    # Post-hoc loop, always progressed=True -> never retried.
    assert len(calls) == 1


# -- _summarize_once (blocking, post-hoc loop check) -----------------------


def _summarizer_with_client(blocks, stop_reason="stop") -> sm.Summarizer:
    """Summarizer with a fake client whose messages.create() returns blocks."""
    s = object.__new__(sm.Summarizer)
    s.cfg = SimpleNamespace(
        gateway_model="qwen3.6-35b-a3b",
        summary_max_tokens=8192,
        repetition_window_chars=3,
        repetition_min_repeats=3,
    )
    s._anthropic = SimpleNamespace(APITimeoutError=RuntimeError)

    class _Messages:
        def create(self, **kwargs):
            return SimpleNamespace(content=blocks, stop_reason=stop_reason)

    s._client = SimpleNamespace(messages=_Messages())
    return s


def test_summarize_once_returns_joined_text_blocks():
    s = _summarizer_with_client([
        SimpleNamespace(type="text", text="hello "),
        SimpleNamespace(type="text", text="world"),
    ])
    assert s._summarize_once("t") == "hello world"


def test_summarize_once_raises_empty_summary_when_no_text():
    s = _summarizer_with_client([
        SimpleNamespace(type="thinking", thinking="x" * 100),
    ])
    with pytest.raises(sm.EmptySummaryError):
        s._summarize_once("t")


def test_summarize_once_raises_loop_on_repeating_output():
    # "abcabcabc" with window=3, min_repeats=3 => _is_looping True.
    s = _summarizer_with_client([
        SimpleNamespace(type="text", text="abcabcabc"),
    ])
    with pytest.raises(StalledGenerationError) as exc_info:
        s._summarize_once("t")
    assert exc_info.value.progressed is True
