"""Env-gated dump of finalized meeting transcripts (stdlib only).

Every finalize writes the raw transcript — byte-identical to
``Transcript.to_prompt_text(max_chars=None)`` so it feeds straight into
``tools/manual_summary.py`` — plus a small JSON sidecar (exact recovery
flags) into the temp folder, BEFORE the gateway summarize call. Any
downstream failure (Cloudflare 524, timeout, repetition loop, parse,
posting) then leaves a recoverable file behind.

Directory: ``$TMPDIR/meeting_bot_transcripts``; override with the
``TRANSCRIPT_DUMP_DIR`` env var, where ``off`` / ``0`` / ``false``
disable dumping entirely. No retention/purge logic — the OS clears temp
on reboot. Never fatal: any write error logs a warning and moves on.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

from .transcript import Transcript

__all__ = ["resolve_dump_dir", "dump_transcript"]

log = logging.getLogger(__name__)

_DUMP_SUBDIR = "meeting_bot_transcripts"
_OFF_VALUES = {"off", "0", "false"}


def resolve_dump_dir() -> Path | None:
    """The directory transcripts are dumped to, or None when disabled.

    ``TRANSCRIPT_DUMP_DIR=off|0|false`` (case-insensitive) disables dumping;
    any other non-empty value is used as the directory verbatim; unset or
    empty falls back to ``$TMPDIR/meeting_bot_transcripts``.
    """
    raw = os.environ.get("TRANSCRIPT_DUMP_DIR")
    if raw is not None:
        value = raw.strip()
        if value.lower() in _OFF_VALUES:
            return None
        if value:
            return Path(value)
    return Path(tempfile.gettempdir()) / _DUMP_SUBDIR


def dump_transcript(
    transcript: Transcript,
    *,
    meeting_title: str,
    started_wall: datetime,
    duration_seconds: int,
    member_count: int,
) -> str | None:
    """Write the rendered transcript + meta sidecar; return the txt path.

    Returns None when the transcript is empty, dumping is disabled, or any
    I/O error occurs (logged as a warning; never raises past this function).
    Filenames carry only the wall-clock start stamp — no display-name
    sanitization, collision-free per meeting start.
    """
    if transcript.is_empty():
        return None
    out_dir = resolve_dump_dir()
    if out_dir is None:
        return None

    stamp = started_wall.strftime("%Y%m%d-%H%M%S")
    txt_path = out_dir / f"{stamp}_transcript.txt"
    meta = {
        "title": meeting_title,
        "started_at": started_wall.isoformat(),
        "duration_seconds": int(duration_seconds),
        "members": int(member_count),
        "transcript_file": txt_path.name,
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(
            transcript.to_prompt_text(max_chars=None), encoding="utf-8"
        )
        sidecar = out_dir / f"{stamp}_meta.json"
        sidecar.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        log.warning("transcript dump failed (%s: %s)", type(exc).__name__, exc)
        return None
    return str(txt_path)
