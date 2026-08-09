"""Pure-logic tests for meeting_bot.transcript (no Discord/network)."""

from meeting_bot.transcript import Transcript, TranscriptEvent


def test_prompt_text_ordering_and_format():
    transcript = Transcript(started_at=100.0)
    transcript.add(TranscriptEvent(speaker="Alice", start=101.5, text="สวัสดี"))
    transcript.add(TranscriptEvent(speaker="Bob", start=100.5, text="สวัสดีครับ"))
    transcript.add(TranscriptEvent(speaker="Alice", start=102.75, text="ลาก่อน"))

    text = transcript.to_prompt_text()
    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[0] == "[00:00] Bob: สวัสดีครับ"
    assert lines[1] == "[00:01] Alice: สวัสดี"
    assert lines[2] == "[00:02] Alice: ลาก่อน"


def test_prompt_text_is_chronological_even_when_added_out_of_order():
    transcript = Transcript(started_at=0.0)
    transcript.add(TranscriptEvent(speaker="A", start=30.0, text="later"))
    transcript.add(TranscriptEvent(speaker="B", start=5.0, text="earlier"))
    lines = transcript.to_prompt_text().splitlines()
    assert lines[0] == "[00:05] B: earlier"
    assert lines[1] == "[00:30] A: later"


def test_events_returns_copy():
    transcript = Transcript(started_at=0.0)
    transcript.add(TranscriptEvent(speaker="A", start=0.1, text="hello"))
    events = transcript.events()
    events.append(TranscriptEvent(speaker="B", start=0.2, text="bye"))
    assert len(transcript.events()) == 1


def test_is_empty():
    transcript = Transcript(started_at=0.0)
    assert transcript.is_empty()
    transcript.add(TranscriptEvent(speaker="A", start=0.5, text="x"))
    assert not transcript.is_empty()


# -- truncation ---------------------------------------------------------

def test_to_prompt_text_truncation_applies():
    transcript = Transcript(started_at=0.0)
    for i in range(20):
        transcript.add(TranscriptEvent(
            speaker=f"Speaker{i}", start=float(i),
            text="x" * 50,
        ))
    text = transcript.to_prompt_text(max_chars=200)
    assert len(text) <= 200
    assert text.endswith("...(truncated)")


def test_to_prompt_text_no_truncation_when_under_limit():
    transcript = Transcript(started_at=0.0)
    transcript.add(TranscriptEvent(speaker="A", start=0.0, text="short"))
    text = transcript.to_prompt_text(max_chars=6000)
    assert "short" in text
    assert "truncated" not in text


def test_to_prompt_text_none_disables_truncation():
    transcript = Transcript(started_at=0.0)
    for i in range(100):
        transcript.add(TranscriptEvent(
            speaker=f"S{i}", start=float(i),
            text="x" * 100,
        ))
    text = transcript.to_prompt_text(max_chars=None)
    assert "truncated" not in text
    assert len(text) > 10000


def test_to_prompt_text_default_max_chars():
    """Default max_chars=48000 applies when called with no arguments."""
    transcript = Transcript(started_at=0.0)
    transcript.add(TranscriptEvent(speaker="A", start=0.0, text="short"))
    text = transcript.to_prompt_text()
    assert "short" in text
