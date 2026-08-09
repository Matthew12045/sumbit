"""Pure-logic tests for meeting_bot.chunker (no Discord/network)."""

import numpy as np
import pytest

from meeting_bot.chunker import Segment, SilenceChunker

SAMPLE_RATE = 16000


def _tone(seconds, freq=440.0, amp=0.5):
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def test_silence_tone_silence_yields_one_segment():
    chunker = SilenceChunker(sample_rate=SAMPLE_RATE)
    chunker.speaker_key = "123"
    chunker.speaker_name = "Alice"

    # ``feed(samples, now)`` means the batch *ends* at ``now``, so each feed
    # advances ``now`` by exactly that batch's duration for contiguous audio.
    now = 100.0
    assert chunker.feed(_silence(0.5), now) == []       # spans [99.5, 100.0]
    assert chunker.feed(_tone(2.0), now + 2.0) == []    # spans [100.0, 102.0]
    segments = chunker.feed(_silence(1.0), now + 3.0)   # spans [102.0, 103.0]
    assert len(segments) == 1
    segment = segments[0]
    assert isinstance(segment, Segment)
    assert segment.speaker_key == "123"
    assert segment.speaker_name == "Alice"
    assert abs(segment.start - 100.0) < 0.05
    assert 2.0 <= segment.duration <= 3.2


def test_short_blip_dropped():
    chunker = SilenceChunker(sample_rate=SAMPLE_RATE)
    chunker.speaker_key = "x"
    chunker.speaker_name = "X"

    now = 10.0
    chunker.feed(_silence(0.5), now)                 # spans [9.5, 10.0]
    chunker.feed(_tone(0.4), now + 0.4)              # spans [10.0, 10.4]
    segments = chunker.feed(_silence(1.0), now + 1.4)  # spans [10.4, 11.4]
    assert segments == []


def test_continuous_speech_force_split():
    chunker = SilenceChunker(
        sample_rate=SAMPLE_RATE,
        max_chunk_seconds=1.0,
        min_chunk_seconds=0.2,
    )
    chunker.speaker_key = "y"
    chunker.speaker_name = "Y"

    now = 20.0
    segments = chunker.feed(_tone(2.5), now)
    assert len(segments) == 2
    assert all(s.duration >= 0.9 for s in segments)
    flushed = chunker.flush(now + 2.5)
    assert len(flushed) == 1
    total = sum(s.duration for s in segments + flushed)
    assert abs(total - 2.5) < 0.05


def test_flush_closes_trailing_partial():
    chunker = SilenceChunker(sample_rate=SAMPLE_RATE, min_chunk_seconds=0.5)
    chunker.speaker_key = "z"
    chunker.speaker_name = "Z"

    now = 5.0
    chunker.feed(_silence(0.3), now)             # spans [4.7, 5.0]
    chunker.feed(_tone(1.5), now + 1.8)          # spans [5.3, 6.8]
    assert chunker.open_duration == pytest.approx(1.5, abs=0.05)
    segments = chunker.flush(now + 3.3)
    assert len(segments) == 1
    assert abs(segments[0].start - 5.3) < 0.05
    assert abs(segments[0].duration - 1.5) < 0.05


def test_reset_clears_state():
    chunker = SilenceChunker(sample_rate=SAMPLE_RATE)
    chunker.speaker_key = "a"
    chunker.speaker_name = "A"
    now = 1.0
    chunker.feed(_tone(0.5), now)
    chunker.reset()
    assert chunker.open_duration == 0.0
    assert chunker.flush(2.0) == []
