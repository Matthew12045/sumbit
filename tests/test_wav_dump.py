"""Pure-logic tests for meeting_bot.wav_dump (stdlib only, no numpy/Discord).

Verifies the diagnostic is inert when disabled, writes a well-formed 16 kHz
mono 16-bit PCM file, clamps float32 -> int16, and sanitizes filenames.
"""

import os
import struct
import wave

import pytest

from meeting_bot.wav_dump import dump_segment_wav, wav_dump_dir


def _read_wav(path: str) -> tuple[int, int, int, bytes]:
    with wave.open(path, "rb") as w:
        return (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes()))


class TestInert:
    def test_wav_dump_dir_none_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("DUMP_CHUNKS_DIR", raising=False)
        assert wav_dump_dir() is None

    def test_wav_dump_dir_none_when_blank(self, monkeypatch) -> None:
        monkeypatch.setenv("DUMP_CHUNKS_DIR", "   ")
        assert wav_dump_dir() is None

    def test_dump_returns_none_when_disabled(self, monkeypatch) -> None:
        monkeypatch.delenv("DUMP_CHUNKS_DIR", raising=False)
        assert dump_segment_wav("alice", 1.25, 2.0, [0.0, 0.1]) is None


class TestWrite:
    def test_round_trip_pcm(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("DUMP_CHUNKS_DIR", str(tmp_path))
        samples = [0.0, 0.5, -0.5, 1.0, -1.0, 0.25]
        path = dump_segment_wav("alice", 1.25, 2.0, samples)
        assert path is not None

        ch, sw, rate, frames = _read_wav(path)
        assert ch == 1
        assert sw == 2
        assert rate == 16000
        assert len(frames) == len(samples) * 2

        ints = struct.unpack("<%dh" % len(samples), frames)
        assert ints == tuple(round(max(-1.0, min(1.0, s)) * 32767) for s in samples)

    def test_clamps_to_int16_range(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("DUMP_CHUNKS_DIR", str(tmp_path))
        path = dump_segment_wav("alice", 0.0, 1.0, [2.0, -2.0])
        _, _, _, frames = _read_wav(path)
        ints = struct.unpack("<2h", frames)
        # Clamp is [-1.0, 1.0] * 32767 (see dump_segment_wav), so +1.0 -> 32767,
        # -1.0 -> -32767 (the float->int rounding, not the int16 floor).
        assert ints == (32767, -32767)

    def test_filenames_sort_by_speaker_then_start(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("DUMP_CHUNKS_DIR", str(tmp_path))
        p1 = dump_segment_wav("alice", 9.5, 1.0, [0.0])
        p2 = dump_segment_wav("alice", 1.0, 1.0, [0.0])
        # lexicographic sort is by speaker then zero-padded start
        # ({start:09.3f} → "00001.000" / "00009.500"); 00001 < 00009
        assert os.path.basename(p2).startswith("alice_00001.000")
        assert os.path.basename(p1).startswith("alice_00009.500")
        assert os.path.basename(p2) < os.path.basename(p1)


class TestSanitization:
    def test_unsafe_chars_replaced(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("DUMP_CHUNKS_DIR", str(tmp_path))
        path = dump_segment_wav("Mat the /0", 0.0, 1.0, [0.0])
        name = os.path.basename(path)
        # spaces and "/" each become "_": "Mat the /0" -> "Mat_the__0"
        assert name.startswith("Mat_the__0_")

    def test_empty_name_falls_back(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("DUMP_CHUNKS_DIR", str(tmp_path))
        path = dump_segment_wav("", 0.0, 1.0, [0.0])
        assert os.path.basename(path).startswith("speaker_")
