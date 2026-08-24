"""Standalone end-to-end probe: real OpenTyphoon polish call -> diff.

NOT a pytest test. Exercises the Thai polish pass against the REAL OpenTyphoon
API endpoint. Builds a synthetic Thai meeting summary, runs it through
``ThaiPolisher``, and prints a before/after diff so a human can eyeball
whether the polish actually improved the prose.

Exit code:
    0   polish returned a result (converged or safety-capped)
    1   polish raised an unexpected exception

Usage (from repo root, .env must be populated):
    python3 tools/e2e_polish_probe.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meeting_bot.config import load_config  # noqa: E402
from meeting_bot.transcript import Transcript, TranscriptEvent  # noqa: E402
from meeting_bot.summary_parse import parse_summary  # noqa: E402
from meeting_bot.thai_polish import ThaiPolisher  # noqa: E402

# (seconds_into_meeting, speaker, text) — a fictional product-team meeting.
_LINES = [
    (0, "อนันต์", "สวัสดีทุกคนครับ วันนี้เรามีกิจกรรมที่ตองคุยกัน 3 เรื่องหลัก คือกวางทางไตรมาสหนวา เรื่่องบั๊กในฟีเจอร์สงออก PDF และงบประมาณคา server"),
    (45, "มะลิ", "เรากำลังเจอบักตัวหนึ่ง เวลาผใชสงออก PDF ไฟลขนาดใหญไฟลมันว่างเปลในบางครัง ทั้ง ๆ ที่ขอมูลครบ"),
    (90, "สมชาย", "ผมวาอาจเปนเรื่ องของ timezone ครับ ตอนที่เราจัดรูปแบบวันที่ใน template ภาษาไทย ขอมูลมันถูกแปลงผิดไปถาเปนเวลาหลังเที่ยงคืน"),
    (135, "จิ๊บ", "ใช่ ฉันจำลองไดแล้ว ปญหาเกิดตอนผใชอยูในเขตเวลา UTC+7 แลว export ตอน 00:30 ของวันใหม วันในเอกสารจะขยับไปอีกวันหนึ่ง"),
    (170, "อนันต์", "แล้วสาเหตุหลักคืออะไร เปนที่ library จัดการวันหรือเปนที่โค้ดของเราเอง"),
    (215, "มะลิ", "เปนที่โค้ดเราครับ เราใช moment แบบดาวแลวไมไดตั้ง timezone ใหชัดเจน ผมเสนอใหเปลี่ยนไปใช date-fns-tz และตั้งคา Asia/Bangkokเปนคาเริ่ มต้น"),
    (260, "สมชาย", "ดีครับ แลวผมจะอัปเดต template ใหใชการ format ผานตัวเดียวกัน เพื่อไมให้มีจุดที่ format วันแยกกันหลายที่"),
    (305, "จิ๊บ", "ฉันจะเขียน test case ครอบคลุมทุก timezone ที่เราสนใจ และเพิ่มกรณีกาปกับกาเดือน เพื่อกันไมใหปัญหาเกิดซ้ำ"),
    (345, "อนันต์", "สรุปรวมวาเราจะแกบั๊กนี้ดวยการเปลี่ยน library จัดการวัน และเพิ่ ม test coverage ใชไหม ทุกคนเห็นดวยไหม"),
    (380, "มะลิ", "เห็นดวยครับ กำหนดสงภายในวันศุกรนี้ไดเลย เพราะมันกระทบลูกคาองค์กรที่ใชฟีเจอร์นี้ทำรายงานประจำเดือน"),
    (420, "อนันต์", "ตอไป เรื่ อง roadmap ไตรมาสหนา เราวางแผนจะเพิ่มฟีเจอร์ dashboard สรุปยอดขายแบบ real-time"),
    (460, "สมชาย", "ฝาย design ผมเสนอใหเริ่มจาก wireframe กอน แลวใหทีม sales ดู เพื่อรับฟีดแบ็กภายใน 2 สัปดาห"),
    (500, "มะลิ", "ทางเทคนิค feasibility สูงครับ เราใช WebSocket อยูแลว เพิ่ม topic ใหมใน event stream ไดไมยาก"),
    (545, "จิ๊บ", "แตฉันทกังวลเรื่ อง performance นะ ถา dashboard ยิง query หนัก ๆ ทุก 5 นาที มันจะโหลด database มากขึ้น เราควรมี rate limit หรือ cache"),
    (590, "อนันต์", "เปนขอกังวลที่ดี จิ๊บชวยทำ spike เรื่ อง caching strategy ภายในสิ้นเดือนนี้ แลวกลับมาคูนกันอีกที"),
    (630, "สมชาย", "แลวงบประมาณละครับ คา server ตอนนีเพิ่ มขึ้ นเรื่อย ๆ เพราะผใชเติบโต เราตั้งงบไวเทาไหร"),
    (660, "อนันต์", "งบปีหน้าตั้งไวประมาณ 2 ลานบาท คิดวาเพียงพอ แตถา dashboard real-time มา ก็ตองขอเพิ่ มอีก 20%"),
    (695, "มะลิ", "สวนงานฝาย infra ที่ทนต่อการถูกยกเลิกได ผมวายายไป spot instance จะชวยประหยัดคาใชจายไดเยอะ"),
    (730, "จิ๊บ", "ฉันจะสำรวจขอมูลคาใชจายปจจุบันใหภายในสัปดาหหนา แลวสรุปรวมวาเราควรยายตรงไหนบาง"),
    (760, "อนันต์", "สรุปการประชุมวันนี้ หนึ่ง แกบั๊ก PDF ภายในวันศุกร สอง ทำ wireframe dashboard ใหทีม sales ติชม สาม จิ๊บทำ spike เรื่ อง cache และสำรวจคาใชจาย ขอบคุนทุกคนครับ"),
]

_MEETING_TITLE = "ประชุมทีมผลิ ตภณั ฑ (e2e polish probe)"
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


def _make_raw_summary(transcript: Transcript) -> str:
    """Build a realistic but intentionally imperfect Thai summary.

    Structural headers use the canonical spellings ``summary_parse`` expects
    (the bot's real input always arrives pre-parsed); the *prose* carries the
    Register 6 violations the polish pass should fix.
    """
    lines = [
        "**ภาพรวม**\n\n",
        "การประชุมวันนี้เปนการคูนเกี่ยวกับ roadmap ไตรมาสหนา เรื่ องบั๊ก PDF\n",
        "และงบประมาณ server มีทกคนเขารวม 4 คน\n\n",
        "**หัวข้อ**\n\n",
        "- **บั๊ก PDF export** — มีปัญหากับ timezone ตอน export ไฟลใหญ่\n",
        "  ฝาย tech เสนอใหเปลี่ยน library จัดการวัน\n",
        "- **dashboard real-time** — ใช WebSocket อยูแลว ฝาย design จะทำ wireframe\n",
        "- **งบประมาณ** — ปหน้าตั้งไว 2 ลาน บาท อาจตองขอเพิ่ มถา dashboard มา\n\n",
        "**การตัดสินใจ**\n\n",
        "- **แกบั๊ก PDF ดวยการเปลี่ยน library** — เพราะกระทบลูกคาองคกร\n",
        "  ครบทุกคนเห็นดวย\n",
        "- **เริ่ม dashboard จาก wireframe** — ใหทีม sales ติชมกอน\n\n",
        "**สิ่งที่ต้องทำ**\n\n",
        "- เปลี่ยน library จัดการวันที่ — มะลิ (ภายใน 2569-01-15)\n",
        "- เขียน test case timezone — จิ๊บ (2569-01-20)\n",
        "- ทำ spike caching strategy — จิ๊บ (สิ้นเดือนนี้)\n\n",
        "**คำถามที่ยังไม่ได้ข้อสรุป**\n\n",
        "- spot instance จะประหยัดค่าใช้จ่ายไดกี่เปอร์เซ็นต์\n",
        "- rate limit ที่เหมาะสมสำหรับ dashboard คือเท่าไหร่\n",
    ]
    return "".join(lines)


def _diff_fields(original, polished) -> str:
    """Field-by-field before/after for exactly the polished fields."""
    lines: list[str] = []

    def show(label: str, before: str, after: str) -> None:
        if before == after:
            lines.append(f"= {label}: (unchanged) {after[:100]}")
        else:
            lines.append(f"- {label}: {before}")
            lines.append(f"+ {label}: {after}")

    show("overview", original.overview, polished.overview)
    for i, (o, p) in enumerate(zip(original.topics, polished.topics)):
        show(f"topics[{i}].detail", o.detail, p.detail)
        if o.title != p.title:
            lines.append(f"! topics[{i}].title CHANGED (protected field): {p.title!r}")
    for i, (o, p) in enumerate(zip(original.decisions, polished.decisions)):
        show(f"decisions[{i}].rationale", o.rationale, p.rationale)
        if o.decision != p.decision:
            lines.append(f"! decisions[{i}].decision CHANGED (protected field): {p.decision!r}")

    if len(original.action_items) != len(polished.action_items):
        lines.append(
            f"! action_items count changed: {len(original.action_items)} -> "
            f"{len(polished.action_items)} (protected field)"
        )
    if original.open_questions != polished.open_questions:
        lines.append("! open_questions changed (protected field)")
    return "\n".join(lines)


def main() -> int:
    cfg = load_config()

    transcript, started_at = build_transcript()
    raw_summary = _make_raw_summary(transcript)
    original = parse_summary(raw_summary)

    print("=" * 78)
    print(
        f"e2e polish probe | model={cfg.polish_model} | "
        f"max_passes={cfg.polish_max_passes} | timeout={cfg.polish_timeout_seconds}s | "
        f"speakers: {', '.join(_SPEAKERS)}"
    )
    print("=" * 78)

    if not cfg.polish_api_key:
        print("SKIP: POLISH_API_KEY not set")
        return 0

    t0 = time.monotonic()
    try:
        polisher = ThaiPolisher(
            base_url=cfg.polish_base_url,
            auth_token=cfg.polish_api_key,
            model=cfg.polish_model,
            max_passes=cfg.polish_max_passes,
            timeout_seconds=cfg.polish_timeout_seconds,
        )
        polished = polisher.polish(original)
    except Exception as exc:  # noqa: BLE001
        print(f"POLISH FAIL: {type(exc).__name__}: {exc}")
        return 1

    elapsed = time.monotonic() - t0
    stats = getattr(polisher, "last_stats", None) or {}
    print(f"\npolish() OK in {elapsed:.2f}s (skill_bundle={polisher.skill_bundle_size()} bytes)")
    print(f"passes: {stats.get('passes')} outcome: {stats.get('outcome')}")
    print("-" * 78)

    # Sanity: protected fields must survive byte-identical.
    protected_ok = (
        [t.title for t in original.topics] == [t.title for t in polished.topics]
        and [d.decision for d in original.decisions] == [d.decision for d in polished.decisions]
        and original.action_items == polished.action_items
        and original.open_questions == polished.open_questions
    )
    print(f"protected fields intact: {'YES' if protected_ok else 'NO — REGRESSION'}")

    print("\n--- FIELD DIFF (polished fields only) ---")
    print(_diff_fields(original, polished))

    unchanged = _diff_fields(original, polished).count("(unchanged)")
    total_fields = 1 + len(original.topics) + len(original.decisions)
    if stats.get("outcome") in ("cap", "hard_failure", "blanked") :
        print("\nNOTE: polish fell back to the ORIGINAL summary "
              f"(outcome={stats.get('outcome')})")
    elif unchanged == total_fields:
        print("\n(no changes — original and polished are field-identical)")

    print("\nE2E PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
