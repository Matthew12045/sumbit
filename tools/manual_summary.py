"""Manual summary regeneration CLI: rerun the finalize pipeline offline.

The live bot summarizes and posts when the last human leaves the voice
channel; if that finalize fails (e.g. a Cloudflare 524 killed the gateway
call), this tool regenerates the missed summary from a saved transcript file
through the same pipeline -- gateway summarize -> robust parse -> OpenTyphoon
Thai polish -> markdown render -- with no Discord involvement. It also works
as an ad-hoc summarizer for any exported ``[MM:SS] speaker: text`` transcript.

Usage (from repo root, .env must be populated):
    python3 tools/manual_summary.py TRANSCRIPT_FILE \
        [--title "Meeting Room 1"] [--started-at "2026-08-25 10:12:11"] \
        [--duration-seconds 560] [--members 2] [--no-polish] [--out PATH] \
        [--post]

With ``--post`` the finished embed + ``summary.md`` are delivered straight
to ``TARGET_CHANNEL_ID`` over the Discord REST API using a plain httpx
multipart request -- no ``discord.Client``, no gateway session -- so it is
safe to run while the live bot is connected.

The markdown is written to --out (default: alongside the transcript file,
``<stem>_summary.md``) and printed to stdout; a small timing/usage report
goes to stderr. No config secret is ever printed.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_bot.config import Config, load_config  # noqa: E402
from meeting_bot.summary_parse import Summary, parse_summary  # noqa: E402
from meeting_bot.summarizer import Summarizer  # noqa: E402
from meeting_bot.thai_polish import ThaiPolisher  # noqa: E402

log = logging.getLogger("manual_summary")

_DISCORD_API_BASE = "https://discord.com/api/v10"
_ATTACHMENT_NAME = "summary.md"
_MAX_ATTEMPTS = 3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate a meeting summary from a transcript file "
        "(offline mirror of the bot's finalize pipeline)."
    )
    parser.add_argument("transcript_file", help="Path to the transcript text file")
    parser.add_argument("--title", default="การประชุม", help="Meeting title (default: การประชุม)")
    parser.add_argument(
        "--started-at",
        default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        help='Meeting start time, "YYYY-MM-DD HH:MM:SS" (default: now)',
    )
    parser.add_argument(
        "--duration-seconds", type=int, default=0, help="Meeting duration in seconds"
    )
    parser.add_argument("--members", type=int, default=0, help="Human participant count")
    parser.add_argument("--no-polish", action="store_true", help="Skip the Thai polish pass")
    parser.add_argument("--out", default=None, help="Output markdown path")
    parser.add_argument(
        "--post",
        action="store_true",
        help="Deliver the finished embed + summary.md to TARGET_CHANNEL_ID via "
        "the Discord REST API (no gateway session)",
    )
    return parser.parse_args(argv)


def _polish(cfg: Config, summary: Summary) -> tuple[Summary, dict | None]:
    """Run the Thai polish pass, falling back to the unpolished summary.

    Returns ``(summary, stats_or_None)``. Never raises past this function.
    """
    if not (cfg.polish_enabled and cfg.polish_api_key):
        log.info("polish skipped (POLISH_ENABLED/POLISH_API_KEY not configured)")
        return summary, None
    try:
        polisher = ThaiPolisher(
            base_url=cfg.polish_base_url,
            auth_token=cfg.polish_api_key,
            model=cfg.polish_model,
            max_passes=cfg.polish_max_passes,
            timeout_seconds=cfg.polish_timeout_seconds,
        )
        polished = polisher.polish(summary)
    except Exception as exc:  # noqa: BLE001 - never crash past polish
        log.warning(
            "polish failed (%s: %s) - using unpolished summary",
            type(exc).__name__,
            exc,
        )
        return summary, None
    return polished, getattr(polisher, "last_stats", None)


def _post_to_discord(
    cfg: Config,
    summary: Summary,
    markdown: str,
    *,
    started_at: datetime,
    duration: timedelta,
    member_count: int,
    meeting_title: str | None = None,
) -> str:
    """Deliver the finished summary to Discord over the plain REST API.

    Builds the embed through :func:`meeting_bot.poster.build_embed` (imported
    lazily here because ``poster`` imports ``discord`` at module scope) and
    posts it together with the complete markdown attached as ``summary.md``
    in one ``POST /channels/{target}/messages`` multipart call. No
    ``discord.Client`` and no gateway connection is ever opened, so this
    cannot clash with a concurrently running bot.

    Retries up to 3 attempts on HTTP 429/5xx, sleeping Discord's
    ``retry_after`` from the error body when present and attempt-number
    seconds otherwise; every other status fails immediately without retry.
    HTTP 401/403/404 get a one-line diagnosis and a non-zero exit; HTTP 400
    includes Discord's error body.

    Returns the created message's jump URL. Never logs the bot token.
    """
    from meeting_bot.poster import build_embed  # lazy: poster imports discord

    embed = build_embed(
        summary,
        started_at=started_at,
        duration=duration,
        member_count=member_count,
        meeting_title=meeting_title,
    )
    payload_json = json.dumps(
        {
            "embeds": [embed.to_dict()],
            "attachments": [{"id": 0, "filename": _ATTACHMENT_NAME}],
        }
    )
    url = f"{_DISCORD_API_BASE}/channels/{cfg.target_channel_id}/messages"
    headers = {"Authorization": f"Bot {cfg.discord_token}"}  # never logged
    diagnoses = {
        401: "bad DISCORD_TOKEN (Discord rejected the bot token)",
        403: "missing permissions (bot cannot send/embed in TARGET_CHANNEL_ID)",
        404: "wrong TARGET_CHANNEL_ID (channel not found)",
    }

    with httpx.Client(timeout=30.0) as client:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            response = client.post(
                url,
                headers=headers,
                data={"payload_json": payload_json},
                files={
                    "files[0]": (
                        _ATTACHMENT_NAME,
                        io.BytesIO(markdown.encode("utf-8")),
                        "text/markdown",
                    )
                },
            )
            status = response.status_code
            if 200 <= status < 300:
                message_id = str(response.json().get("id", ""))
                return (
                    f"https://discord.com/channels/{cfg.guild_id}/"
                    f"{cfg.target_channel_id}/{message_id}"
                )
            if status in diagnoses:
                raise SystemExit(f"posting failed: HTTP {status} — {diagnoses[status]}")
            if status == 400:
                raise SystemExit(
                    f"posting failed: HTTP 400 — Discord error body: {response.text}"
                )
            if status == 429 or 500 <= status <= 599:
                if attempt == _MAX_ATTEMPTS:
                    raise SystemExit(
                        f"posting failed: HTTP {status} persisted after "
                        f"{_MAX_ATTEMPTS} attempts: {response.text[:500]}"
                    )
                delay = float(attempt)
                try:
                    retry_after = response.json().get("retry_after")
                    if isinstance(retry_after, (int, float)) and retry_after > 0:
                        delay = float(retry_after)
                except ValueError:
                    pass
                log.warning(
                    "HTTP %d posting summary (attempt %d/%d); retrying in %.1fs",
                    status,
                    attempt,
                    _MAX_ATTEMPTS,
                    delay,
                )
                time.sleep(delay)
                continue
            raise SystemExit(
                f"posting failed: unexpected HTTP {status}: {response.text[:500]}"
            )
    raise RuntimeError("unreachable")  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    cfg = load_config()

    transcript_path = Path(args.transcript_file)
    transcript_text = transcript_path.read_text(encoding="utf-8")
    log.info("loaded transcript %s (%d chars)", transcript_path, len(transcript_text))

    t0 = time.monotonic()
    summarizer = Summarizer(cfg)
    raw = summarizer.summarize(transcript_text)
    summarize_elapsed = time.monotonic() - t0
    log.info("summarize finished in %.1fs", summarize_elapsed)

    summary = parse_summary(raw)
    if args.no_polish:
        log.info("polish disabled by --no-polish")
        polished, polish_stats = summary, {"outcome": "skipped (--no-polish)"}
    else:
        polished, polish_stats = _polish(cfg, summary)

    started_at = datetime.strptime(args.started_at, "%Y-%m-%d %H:%M:%S")
    # Lazy import: poster pulls in discord at module scope, so only load it
    # once the pipeline has actually produced something to render.
    from meeting_bot.poster import render_markdown

    markdown = render_markdown(
        polished,
        started_at=started_at,
        duration=timedelta(seconds=args.duration_seconds),
        member_count=args.members,
        meeting_title=args.title,
    )

    out_path = (
        Path(args.out)
        if args.out
        else transcript_path.with_name(transcript_path.stem + "_summary.md")
    )
    out_path.write_text(markdown, encoding="utf-8")

    jump_url: str | None = None
    if args.post:
        jump_url = _post_to_discord(
            cfg,
            polished,
            markdown,
            started_at=started_at,
            duration=timedelta(seconds=args.duration_seconds),
            member_count=args.members,
            meeting_title=args.title,
        )

    sys.stdout.write(markdown)
    if jump_url:
        print(jump_url)

    usage = summarizer.last_usage
    usage_text = (
        f"input_tokens={getattr(usage, 'input_tokens', '?')} "
        f"output_tokens={getattr(usage, 'output_tokens', '?')}"
        if usage is not None
        else "unavailable"
    )
    print("--- report ---", file=sys.stderr)
    print(f"transcript: {transcript_path} ({len(transcript_text)} chars)", file=sys.stderr)
    print(f"summarize: {summarize_elapsed:.1f}s, {usage_text}", file=sys.stderr)
    print(f"polish: {polish_stats}", file=sys.stderr)
    print(f"wrote: {out_path}", file=sys.stderr)
    if jump_url:
        print(f"posted: {jump_url}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - report cleanly, exit non-zero
        logging.getLogger("manual_summary").error(
            "FAILED: %s: %s", type(exc).__name__, exc, exc_info=True
        )
        raise SystemExit(1) from exc
