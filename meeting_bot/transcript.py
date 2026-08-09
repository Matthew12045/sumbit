"""Accumulated transcript (pure stdlib)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TranscriptEvent", "Transcript"]


@dataclass
class TranscriptEvent:
    speaker: str
    start: float
    text: str


class Transcript:
    """Ordered transcription events with a ``[MM:SS]`` prompt renderer."""

    def __init__(self, started_at: float):
        self.started_at = float(started_at)
        self._events: list[TranscriptEvent] = []

    def add(self, event: TranscriptEvent) -> None:
        self._events.append(event)

    def events(self) -> list[TranscriptEvent]:
        return list(self._events)

    def to_prompt_text(self, max_chars: int | None = 48000) -> str:
        """Render chronologically as ``[MM:SS] ผู้พูด: ...`` lines.

        If ``max_chars`` is not None and the rendered text exceeds it, the text
        is truncated and a ``...(truncated)`` suffix is appended.  Truncation
        preserves whole lines (does not split mid-line).
        """
        lines = []
        for event in sorted(self._events, key=lambda e: e.start):
            elapsed = max(0.0, event.start - self.started_at)
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            lines.append(f"[{minutes:02d}:{seconds:02d}] {event.speaker}: {event.text}")

        if max_chars is None:
            return "\n".join(lines)

        suffix = "...(truncated)"
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text

        # Truncate line by line to avoid splitting mid-line.
        budget = max_chars - len(suffix)
        kept: list[str] = []
        for line in lines:
            if len(line) + (1 if kept else 0) <= budget - sum(len(l) + 1 for l in kept):
                kept.append(line)
            else:
                break
        if not kept:
            # A single line is longer than the budget — slice it.
            return text[:budget] + suffix
        return "\n".join(kept) + "\n" + suffix

    def is_empty(self) -> bool:
        return not self._events
