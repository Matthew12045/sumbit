"""Standalone end-to-end probe: real gateway summarize -> parse -> embed.

NOT a pytest test. Exercises the exact production pipeline ``bot.py`` runs
when a meeting finalizes:

    Transcript.to_prompt_text(max_chars=cfg.max_prompt_chars)
      -> Summarizer(cfg).summarize(prompt_text)   # REAL gateway call
      -> parse_summary(raw)
      -> build_embed(summary, ...)                # constructs discord.Embed only

It builds a synthetic but realistic 20-line Thai meeting transcript (multiple
speakers, short + long utterances, ~3k chars — representative of a real
prompt, not a toy one-liner), calls the REAL unmocked gateway, and prints the
raw model output and the parsed Summary so a human can eyeball whether the
Thai output is actually sensible, not just well-formed.

Exit code:
    0   clean run — summarize() returned text, parse_summary() did NOT fall
        back to the last-resort raw-text branch, embed built.
    1   summarize() raised (EmptySummaryError / StalledGenerationError /
        any other exception) OR parse_summary() fell back to last resort.

Usage (from repo root, .env must be populated):
    python3 tools/e2e_summarize_probe.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_bot.config import load_config  # noqa: E402
from meeting_bot.transcript import Transcript, TranscriptEvent  # noqa: E402
from meeting_bot.summary_parse import (  # noqa: E402
    parse_summary,
    _try_json,
    _try_markdown,
)
from meeting_bot.poster import build_embed, render_markdown  # noqa: E402

# (seconds_into_meeting, speaker, text) — a fictional product-team meeting.
# Realistic Thai dialogue: a couple of short back-and-forths, longer
# explanations, a bug, roadmap planning, budget, and a closing recap.
_LINES: list[tuple[float, str, str]] = [
    (0, "อนันต์", "สวัสดีทุกคนครับ วันนี้เรามีเรื่องต้องคุยกัน 3 เรื่องหลัก ๆ คือ roadmap "
                  "ไตรมาสหน้า เรื่องบั๊กในฟีเจอร์ส่งออก PDF และงบประมาณค่า server"),
    (45, "มะลิ", "เรากำลังเจอบั๊กตัวหนึ่ง เวลาผู้ใช้ส่งออก PDF ไฟล์ขนาดใหญ่ ไฟล์มันว่างเปล่า "
                  "ในบางครั้ง ทั้ง ๆ ที่ข้อมูลครบ"),
    (90, "สมชาย", "ผมว่าน่าจะเป็นเรื่องของ timezone ครับ ตอนที่เราจัดรูปแบบวันที่ใน template "
                   "ภาษาไทย ข้อมูลมันถูกแปลงผิดไปถ้าเป็นเวลาหลังเที่ยงคืน"),
    (135, "จิ๊บ", "ใช่ ฉันจำลองได้แล้ว ปัญหาเกิดตอนผู้ใช้อยู่ในเขตเวลา UTC+7 แล้ว export "
                  "ตอน 00:30 ของวันใหม่ วันที่ในเอกสารจะขยับไปอีกวันหนึ่ง"),
    (170, "อนันต์", "แล้วสาเหตุหลักคืออะไร เป็นที่ library จัดการวันที่ หรือเป็นที่โค้ดของเราเอง"),
    (215, "มะลิ", "เป็นที่โค้ดเราครับ เราใช้ moment แบบเก่าแล้วไม่ได้ตั้ง timezone ให้ชัดเจน "
                  "ผมเสนอให้เปลี่ยนไปใช้ date-fns-tz และตั้งค่า Asia/Bangkok เป็นค่าเริ่มต้น"),
    (260, "สมชาย", "ดีครับ แล้วผมจะอัปเดต template ให้ใช้การ format ผ่านตัวเดียวกัน "
                   "เพื่อไม่ให้มีจุดที่ format วันที่แยกกันหลายที่"),
    (305, "จิ๊บ", "ฉันจะเขียน test case ครอบคลุมทุก timezone ที่เราสนใจ และเพิ่มกรณีข้ามปี "
                  "กับข้ามเดือน เพื่อกันไม่ให้ปัญหากลับมาเป็นซ้ำ"),
    (345, "อนันต์", "สรุปว่าเราจะแก้บั๊กนี้ด้วยการเปลี่ยน library จัดการวันที่ และเพิ่ม "
                    "test coverage ใช่ไหม ทุกคนเห็นด้วยไหม"),
    (380, "มะลิ", "เห็นด้วยครับ กำหนดส่งภายในวันศุกร์นี้ได้เลย เพราะมันกระทบลูกค้าองค์กร "
                  "ที่ใช้ฟีเจอร์นี้ทำรายงานประจำเดือน"),
    (420, "อนันต์", "ต่อไป เรื่อง roadmap ไตรมาสหน้า เราวางแผนจะเพิ่มฟีเจอร์ dashboard "
                    "สรุปยอดขายแบบ real-time"),
    (460, "สมชาย", "ฝั่ง design ผมเสนอให้เริ่มจาก wireframe ก่อน แล้วให้ทีม sales ดู "
                   "เพื่อรับฟีดแบ็กภายใน 2 สัปดาห์"),
    (500, "มะลิ", "ทางเทคนิค feasibility สูงครับ เราใช้ WebSocket อยู่แล้ว เพิ่ม topic ใหม่ "
                  "ใน event stream ได้ไม่ยาก"),
    (545, "จิ๊บ", "แต่ฉันกังวลเรื่อง performance นะ ถ้า dashboard ยิง query หนัก ๆ ทุก 5 "
                  "วินาที มันจะโหลด database มากขึ้น เราควรมี rate limit หรือ cache"),
    (590, "อนันต์", "เป็นข้อกังวลที่ดี จิ๊บช่วยทำ spike เรื่อง caching strategy "
                    "ภายในสิ้นเดือนนี้ แล้วค่อยกลับมาคุยกันอีกที"),
    (630, "สมชาย", "แล้วงบประมาณล่ะครับ ค่า server ตอนนี้เพิ่มขึ้นเรื่อย ๆ เพราะผู้ใช้เติบโต "
                   "เราตั้งงบไว้เท่าไหร่"),
    (660, "อนันต์", "งบปีหน้าตั้งไว้ประมาณ 2 ล้านบาท คิดว่าเพียงพอ แต่ถ้า dashboard "
                    "real-time มา ก็ต้องขอเพิ่มอีก 20%"),
    (695, "มะลิ", "ส่วนงานฝั่ง infra ที่ทนต่อการถูกยกเลิกได้ ผมว่าย้ายไป spot instance "
                  "จะช่วยประหยัดค่าใช้จ่ายได้เยอะ"),
    (730, "จิ๊บ", "ฉันจะสำรวจข้อมูลค่าใช้จ่ายปัจจุบันให้ภายในสัปดาห์หน้า แล้วสรุปว่า "
                  "เราควรย้ายตรงไหนบ้าง"),
    (760, "อนันต์", "สรุปการประชุมวันนี้ หนึ่ง แก้บั๊ก PDF ภายในวันศุกร์ สอง ทำ wireframe "
                    "dashboard แล้วให้ทีม sales ฟีดแบ็ก สาม จิ๊บทำ spike เรื่อง cache "
                    "และสำรวจค่าใช้จ่าย ขอบคุณทุกคนครับ"),
]

_MEETING_TITLE = "ประชุมทีมผลิตภัณฑ์ (e2e probe)"
_SPEAKERS = sorted({speaker for _, speaker, _ in _LINES})


def build_transcript() -> tuple[Transcript, datetime]:
    """Build the synthetic transcript the same way the bot accumulates one."""
    started_at = datetime(2026, 8, 10, 9, 0, 0)
    transcript = Transcript(started_at.timestamp())
    for offset, speaker, text in _LINES:
        transcript.add(
            TranscriptEvent(speaker=speaker, start=started_at.timestamp() + offset, text=text)
        )
    return transcript, started_at


def _format_summary(summary) -> str:
    """Render the parsed Summary as readable text for eyeballing."""
    lines = [
        f"overview: {summary.overview}",
        "",
        f"topics ({len(summary.topics)}):",
    ]
    for item in summary.topics:
        lines.append(f"  • {item.title}")
        if item.detail:
            lines.append(f"      detail: {item.detail}")
    lines.append("")
    lines.append(f"decisions ({len(summary.decisions)}):")
    for item in summary.decisions:
        lines.append(f"  • {item.decision}")
        if item.rationale:
            lines.append(f"      rationale: {item.rationale}")
    lines.append("")
    lines.append(f"action_items ({len(summary.action_items)}):")
    for item in summary.action_items:
        lines.append(
            f"  • {item.action} | owner: {item.owner!r} | due: {item.due!r}"
        )
    lines.append("")
    lines.append(f"open_questions ({len(summary.open_questions)}):")
    for q in summary.open_questions:
        lines.append(f"  • {q}")
    return "\n".join(lines)


def _embed_summary(embed) -> str:
    """Render a discord.Embed's fields for printing (no Discord connection)."""
    lines = [f"title: {embed.title!r}", f"description: {embed.description!r}", "fields:"]
    for field in embed.fields:
        lines.append(f"  {field.name}: {field.value!r}")
    lines.append(f"footer: {embed.footer.text!r}")
    return "\n".join(lines)


def main() -> int:
    cfg = load_config()

    transcript, started_at = build_transcript()
    prompt_text = transcript.to_prompt_text(max_chars=cfg.max_prompt_chars)

    print("=" * 78)
    print(
        f"e2e summarize probe | model={cfg.gateway_model} | "
        f"max_tokens={cfg.summary_max_tokens} | prompt={len(prompt_text)} chars | "
        f"{len(_LINES)} transcript lines | speakers: {', '.join(_SPEAKERS)}"
    )
    print("=" * 78)

    from meeting_bot.summarizer import Summarizer, EmptySummaryError, StalledGenerationError

    summarizer = Summarizer(cfg)
    t0 = time.monotonic()
    try:
        raw = summarizer.summarize(prompt_text)
    except EmptySummaryError as exc:
        print(f"E2E FAIL: EmptySummaryError — {exc}")
        return 1
    except StalledGenerationError as exc:
        print(f"E2E FAIL: StalledGenerationError — {exc} (progressed={exc.progressed})")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"E2E FAIL: summarize() raised {type(exc).__name__}: {exc}")
        return 1
    elapsed = time.monotonic() - t0

    # Token accounting: real chars/token for Thai/qwen on this gateway model.
    usage = getattr(summarizer, "last_usage", None)
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        print(f"usage: input={input_tokens} output={output_tokens}")
        if output_tokens and len(raw) > 0:
            ratio = (len(prompt_text) + len(raw)) / max(1, (input_tokens or 0) + output_tokens)
            out_ratio = len(raw) / max(1, output_tokens)
            in_ratio = len(prompt_text) / max(1, input_tokens or 1)
            print(
                f"chars/token: overall≈{ratio:.2f}  prompt≈{in_ratio:.2f}  "
                f"output≈{out_ratio:.2f}  (retune MAX_PROMPT_CHARS if this "
                f"differs materially from the assumed ~1.5)"
            )

    print(f"summarize() OK in {elapsed:.2f}s — raw response ({len(raw)} chars):")
    print("-" * 78)
    print(raw)
    print("-" * 78)

    # Which parse branch did the parser actually take?
    summary = parse_summary(raw)
    raw_stripped = raw.strip()
    is_last_resort = (
        bool(raw_stripped)
        and summary.overview == raw_stripped
        and not summary.topics
        and not summary.decisions
        and not summary.action_items
        and not summary.open_questions
    )
    if is_last_resort:
        branch = "LAST-RESORT (raw text as overview)"
    elif _try_json(raw) is not None:
        branch = "JSON"
    elif _try_markdown(raw) is not None:
        branch = "markdown"
    else:
        branch = "structured (unknown channel)"

    print(f"\n--- parsed Summary (branch: {branch}) ---")
    print(_format_summary(summary))

    if is_last_resort:
        print("\nE2E FAIL: parse_summary() fell back to the LAST-RESORT raw-text branch — "
              "the model did not return parseable JSON or markdown.")
        return 1

    # Embed construction is pure object-building (no network / Discord gateway).
    duration = timedelta(seconds=_LINES[-1][0])
    meta = dict(
        started_at=started_at,
        duration=duration,
        member_count=len(_SPEAKERS),
        meeting_title=_MEETING_TITLE,
    )
    embed = build_embed(summary, **meta)
    print("\n--- discord.Embed (constructed, not sent) ---")
    print(_embed_summary(embed))

    markdown = render_markdown(summary, **meta)
    print("\n--- summary.md attachment (first 500 chars) ---")
    print(markdown[:500])

    print("\nE2E PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
