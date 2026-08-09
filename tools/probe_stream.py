"""Standalone probe: does gateway.9arm.co stream smoothly, and where do gaps land?

Prints every SSE event with elapsed time and the gap since the previous
event, so a >5s gap is visible immediately instead of inferred after the
fact from a StalledGenerationError.

Run it twice to separate "gateway is inherently bursty" from "whisper is
stealing the GPU":
  1. Idle baseline  — with no meeting running (whisper worker idle).
  2. Contended       — start this at the same moment you'd normally trigger
                       summarization (e.g. right as a meeting ends, or with
                       a whisper decode forced to run concurrently).

If gaps only show up in run 2, it's GPU contention, not the gateway or the
model. If gaps show up in both, it's the gateway/model itself.

Usage (from repo root):
    python3 tools/probe_stream.py
    python3 tools/probe_stream.py --chars 48000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_bot.config import load_config  # noqa: E402

_SAMPLE_LINE = (
    "[00:00] ผู้พูด1: ทดสอบข้อความภาษาไทยสำหรับตรวจสอบพฤติกรรม streaming ของ gateway "
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chars", type=int, default=48000,
        help="approx prompt size to send (default matches MAX_PROMPT_CHARS)",
    )
    parser.add_argument(
        "--gap-warn", type=float, default=5.0,
        help="flag any inter-event gap larger than this many seconds",
    )
    args = parser.parse_args()

    cfg = load_config()
    import anthropic  # lazy, matches the pure-import rule elsewhere in the repo

    client = anthropic.Anthropic(
        base_url=cfg.anthropic_base_url,
        auth_token=cfg.anthropic_auth_token,
        timeout=cfg.summarize_timeout_seconds,
    )

    # Repeat a Thai sample line up to roughly the real transcript size so the
    # probe reflects a production-sized prompt, not a toy one-liner.
    body = (_SAMPLE_LINE * (args.chars // len(_SAMPLE_LINE) + 1))[: args.chars]
    prompt = body + "\n\nโปรดสรุปข้อความข้างต้นเป็น JSON สั้น ๆ"

    print(
        f"prompt length: {len(prompt)} chars | model={cfg.gateway_model} | "
        f"max_tokens={cfg.summary_max_tokens}"
    )
    print("-" * 78)

    t0 = time.monotonic()
    last = t0
    event_count = 0
    total_delta_chars = 0
    biggest_gap = 0.0

    with client.messages.stream(
        model=cfg.gateway_model,
        max_tokens=cfg.summary_max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            now = time.monotonic()
            gap = now - last
            elapsed = now - t0
            biggest_gap = max(biggest_gap, gap)
            event_count += 1
            etype = getattr(event, "type", "?")
            piece = ""
            if etype == "content_block_delta":
                delta = event.delta
                piece = getattr(delta, "text", None) or getattr(delta, "thinking", None) or ""
                total_delta_chars += len(piece)
            flag = f"  <-- GAP {gap:.1f}s" if gap > args.gap_warn else ""
            print(f"[{elapsed:7.2f}s] (+{gap:6.2f}s) {etype:24s} {len(piece):4d} chars{flag}")
            last = now
        final = stream.get_final_message()

    total = time.monotonic() - t0
    print("-" * 78)
    print(
        f"done: {event_count} events | {total_delta_chars} content chars | "
        f"{total:.2f}s total | biggest single gap: {biggest_gap:.2f}s | "
        f"stop_reason={getattr(final, 'stop_reason', '?')}"
    )
    # Production no longer streams (blocking messages.create(), no stall guard
    # -- see summarizer.py). This probe is pure characterization: did events
    # trickle in throughout, or did the gateway buffer the whole completion
    # and emit them in a burst near the end?
    print(
        "NOTE: streaming characterization only. If most events landed in a "
        "burst near the end, the gateway buffers the whole completion. If "
        "events trickled in with real gaps, the gateway does stream "
        "incrementally."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
