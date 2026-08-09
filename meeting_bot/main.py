"""CLI entrypoint: run the bot or print the doctor report."""

from __future__ import annotations

import argparse
import logging
import sys

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting_bot",
        description=(
            "Meeting Summary Discord Bot — transcribes Thai meetings live and "
            "posts a structured Topics / Decisions / Action Items summary when "
            "the last human leaves the voice channel."
        ),
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="run environment checks and exit (0 if all ok, else 1)",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="path to the .env file (default: .env)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    log_level = logging.DEBUG if args.debug else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.debug:
        try:
            import os as _os
            _log_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "meeting_bot_debug.log")
            handlers.append(logging.FileHandler(_log_path, encoding="utf-8"))
        except OSError:
            pass  # can't write log file — stream handler is enough
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    from .config import Config, doctor, load_config

    cfg: Config | None = None
    try:
        cfg = load_config(args.env)
    except Exception as exc:  # noqa: BLE001
        if not args.doctor:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        # --doctor still works without a complete .env: run the checks against
        # an empty config so every check reports instead of aborting.
        cfg = Config(
            discord_token="",
            guild_id=0,
            voice_channel_id=0,
            target_channel_id=0,
            anthropic_base_url="",
            anthropic_auth_token="",
            gateway_model="",
            whisper_model="",
            whisper_language="",
        )

    if args.doctor:
        lines = doctor(cfg)
        for line in lines:
            print(line)
        return 0 if all(line.startswith("ok:") for line in lines) else 1

    from .bot import MeetingBot

    bot = MeetingBot(cfg)
    bot.run(cfg.discord_token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
