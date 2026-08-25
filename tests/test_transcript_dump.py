"""Pure-logic tests for meeting_bot.transcript_dump (stdlib only).

No Discord/network: the module under test imports only stdlib +
``.transcript``, so these run in the minimal test environment.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from meeting_bot.transcript import Transcript, TranscriptEvent
from meeting_bot.transcript_dump import dump_transcript, resolve_dump_dir

STARTED = datetime(2026, 8, 25, 10, 12, 11)
STAMP = "20260825-101211"


def _make_transcript() -> Transcript:
    t = Transcript(started_at=100.0)
    t.add(TranscriptEvent(speaker="Alice", start=100.0, text="สวัสดีทุกคน"))
    t.add(TranscriptEvent(speaker="Bob", start=165.0, text="เริ่มวาระแรกกันเลย"))
    return t


def _dump(transcript: Transcript | None = None, **kwargs) -> str | None:
    return dump_transcript(
        transcript if transcript is not None else _make_transcript(),
        meeting_title=kwargs.pop("meeting_title", "Test Guild · room-1"),
        started_wall=kwargs.pop("started_wall", STARTED),
        duration_seconds=kwargs.pop("duration_seconds", 560),
        member_count=kwargs.pop("member_count", 2),
    )


@pytest.fixture
def dump_dir(monkeypatch, tmp_path):
    """Point TRANSCRIPT_DUMP_DIR at a per-test tmp dir."""
    monkeypatch.setenv("TRANSCRIPT_DUMP_DIR", str(tmp_path))
    return tmp_path


# -- resolve_dump_dir ------------------------------------------------------


def test_resolve_default_dir_when_unset(monkeypatch):
    monkeypatch.delenv("TRANSCRIPT_DUMP_DIR", raising=False)
    assert resolve_dump_dir() == (
        Path(tempfile.gettempdir()) / "meeting_bot_transcripts"
    )


@pytest.mark.parametrize("off_value", ["off", "0", "false", "OFF", "False"])
def test_resolve_off_values_disable(monkeypatch, off_value):
    monkeypatch.setenv("TRANSCRIPT_DUMP_DIR", off_value)
    assert resolve_dump_dir() is None


def test_resolve_empty_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_DUMP_DIR", "   ")
    assert resolve_dump_dir() == (
        Path(tempfile.gettempdir()) / "meeting_bot_transcripts"
    )


def test_resolve_custom_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSCRIPT_DUMP_DIR", str(tmp_path))
    assert resolve_dump_dir() == tmp_path


# -- dump_transcript -------------------------------------------------------


def test_dump_content_equals_to_prompt_text(dump_dir):
    transcript = _make_transcript()
    path = _dump()

    assert path is not None
    written = Path(path).read_text(encoding="utf-8")
    assert written == transcript.to_prompt_text(max_chars=None)


def test_dump_filename_derives_from_started_wall(dump_dir):
    path = _dump()

    assert path == str(dump_dir / f"{STAMP}_transcript.txt")
    assert Path(path).exists()


def test_dump_sidecar_has_five_typed_fields(dump_dir):
    title = "Guild · voice-room"
    path = _dump(meeting_title=title)
    sidecar = dump_dir / f"{STAMP}_meta.json"

    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert set(meta) == {
        "title",
        "started_at",
        "duration_seconds",
        "members",
        "transcript_file",
    }
    assert meta["title"] == title
    assert isinstance(meta["title"], str)
    assert meta["started_at"] == STARTED.isoformat()
    assert meta["duration_seconds"] == 560
    assert isinstance(meta["duration_seconds"], int)
    assert meta["members"] == 2
    assert isinstance(meta["members"], int)
    assert meta["transcript_file"] == Path(path).name


def test_dump_creates_missing_nested_override_dir(monkeypatch, tmp_path):
    nested = tmp_path / "deeper" / "dumps"
    monkeypatch.setenv("TRANSCRIPT_DUMP_DIR", str(nested))

    path = _dump()
    assert path == str(nested / f"{STAMP}_transcript.txt")
    assert nested.joinpath(f"{STAMP}_meta.json").exists()


@pytest.mark.parametrize("off_value", ["off", "0", "false"])
def test_dump_disabled_writes_nothing(monkeypatch, tmp_path, off_value):
    monkeypatch.setenv("TRANSCRIPT_DUMP_DIR", off_value)
    assert _dump() is None
    assert list(tmp_path.iterdir()) == []


def test_dump_empty_transcript_returns_none_and_writes_nothing(dump_dir):
    empty = Transcript(started_at=100.0)
    assert _dump(empty) is None
    assert list(dump_dir.iterdir()) == []


def test_dump_unwritable_dir_returns_none_no_raise(dump_dir, monkeypatch):
    def _boom(self, *args, **kwargs):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(Path, "mkdir", _boom)
    assert _dump() is None
