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
    s.cfg = None  # unused by the retry loop
    s._anthropic = SimpleNamespace(APITimeoutError=RuntimeError)
    s._client = None
    return s


def test_zero_progress_stall_retries_once_then_raises(monkeypatch):
    monkeypatch.setattr(sm.time, "sleep", lambda _seconds: None)
    s = _make_summarizer()
    calls: list[str] = []

    def fake_once(text: str) -> str:
        calls.append(text)
        raise StalledGenerationError("stalled", progressed=False)

    s._summarize_once = fake_once
    with pytest.raises(StalledGenerationError):
        s.summarize("t")
    # First attempt + one retry, then give up.
    assert len(calls) == 2


def test_progressed_stall_raises_without_retry(monkeypatch):
    monkeypatch.setattr(sm.time, "sleep", lambda _seconds: None)
    s = _make_summarizer()
    calls: list[str] = []

    def fake_once(text: str) -> str:
        calls.append(text)
        raise StalledGenerationError("loop detected", progressed=True)

    s._summarize_once = fake_once
    with pytest.raises(StalledGenerationError):
        s.summarize("t")
    # Output had started -> retrying the same trace would reproduce the loop.
    assert len(calls) == 1
