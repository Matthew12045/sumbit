"""Meeting Summary Discord Bot package.

``import meeting_bot`` stays lightweight: the heavy dependencies (discord,
mlx-whisper, anthropic) are imported lazily via module ``__getattr__`` so the
pure modules (config, audio, chunker, transcript, summary_parse) can be
imported and tested with only numpy installed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["Config", "MeetingBot", "__version__"]


def __getattr__(name: str):
    if name == "Config":
        from .config import Config

        return Config
    if name == "MeetingBot":
        from .bot import MeetingBot

        return MeetingBot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"Config", "MeetingBot"})
