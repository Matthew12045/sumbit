"""Meeting-summary Discord bot package.

Uses lazy ``__getattr__`` / ``__dir__`` to defer importing heavy dependencies
(`discord`, ``mlx-whisper``, ``openai``) until they are first accessed.  The
pure modules (config, audio, chunker, transcript, summary_parse) are
importable with only ``numpy`` present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meeting_bot.config import Config
    from meeting_bot.bot import MeetingBot
    from meeting_bot.thai_polish import ThaiPolisher

__all__ = ["Config", "MeetingBot", "ThaiPolisher"]

_config: Config | None = None
_bot_cls: type[MeetingBot] | None = None
_polisher_cls: type[ThaiPolisher] | None = None


def __getattr__(name: str):
    global _config, _bot_cls, _polisher_cls
    if name == "Config":
        if _config is None:
            from meeting_bot.config import Config as _C
            _config = _C
        return _config
    if name == "MeetingBot":
        if _bot_cls is None:
            from meeting_bot.bot import MeetingBot as _B
            _bot_cls = _B
        return _bot_cls
    if name == "ThaiPolisher":
        if _polisher_cls is None:
            from meeting_bot.thai_polish import ThaiPolisher as _P
            _polisher_cls = _P
        return _polisher_cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
