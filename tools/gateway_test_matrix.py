#!/usr/bin/env python3
"""Gateway behavior test matrix for gateway.9arm.co (Anthropic-compatible).

Systematically characterizes every failure mode around the Cloudflare 120 s
proxy window observed on 2026-08-24/25:

    Fast cases (--fast):
      tiny_probe        doctor-style max_tokens=1 call
      small_output      small prompt, max_tokens=256
      large_input       ~150k-char prompt, max_tokens=64 (isolates INPUT size)
      stream_small      successful streamed call — per-event arrival timeline
                        (reveals whether SSE bytes flow early or are buffered)
      err_old_model     qwen3.6-35b-a3b (expected off the allowlist)
      err_bogus_model   unknown model id
      err_bad_token     invalid bearer token

    Slow cases (--slow, each may cost up to ~135 s on failure):
      pure_decode       forces long literal output (minimal thinking) —
                        measures true decode tok/s vs the ~7.5 tok/s effective
      full_nonstream    the production-shaped summarize call (known 524)
      full_stream       same call streamed (connection killed mid-flight)

Usage:
    python3 tools/gateway_test_matrix.py --fast
    python3 tools/gateway_test_matrix.py --slow
    python3 tools/gateway_test_matrix.py            # both
    python3 tools/gateway_test_matrix.py --json-out /tmp/gw.json

Never prints or stores any auth token.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402


def _mk_client(cfg, timeout: float = 135.0, token: str | None = None):
    import anthropic

    return anthropic.Anthropic(
        base_url=cfg.anthropic_base_url,
        auth_token=token or cfg.anthropic_auth_token,
        timeout=timeout,
        max_retries=0,
    )


def _text_of(message) -> str:
    return "".join(
        b.text for b in message.content if getattr(b, "type", "") == "text"
    )


# ---------------------------------------------------------------------------
# Case implementations — each returns a result dict
# ---------------------------------------------------------------------------

def case_tiny_probe(cfg) -> dict:
    client = _mk_client(cfg)
    t0 = time.monotonic()
    try:
        m = client.messages.create(
            model=cfg.gateway_model, max_tokens=1, temperature=0.0,
            messages=[{"role": "user", "content": "hi"}],
        )
        u = getattr(m, "usage", None)
        return {"ok": True, "elapsed": round(time.monotonic() - t0, 1),
                "detail": f"out_tok={getattr(u, 'output_tokens', '?')}"}
    except Exception as exc:
        return {"ok": False, "elapsed": round(time.monotonic() - t0, 1),
                "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}


def case_small_output(cfg) -> dict:
    client = _mk_client(cfg)
    t0 = time.monotonic()
    try:
        m = client.messages.create(
            model=cfg.gateway_model, max_tokens=256, temperature=0.0,
            messages=[{"role": "user", "content":
                       "ตอบสั้น ๆ ครับ: 1+1 เท่ากับเท่าไร"}],
        )
        dt = time.monotonic() - t0
        u = m.usage
        tps = u.output_tokens / dt if dt else 0
        text = _text_of(m)
        return {"ok": True, "elapsed": round(dt, 1),
                "detail": f"out_tok={u.output_tokens} ({tps:.1f} tok/s) "
                          f"text_chars={len(text)}"}
    except Exception as exc:
        return {"ok": False, "elapsed": round(time.monotonic() - t0, 1),
                "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}


def case_large_input(cfg) -> dict:
    """~150k-char prompt (production MAX_PROMPT_CHARS scale), tiny output."""
    line = "[00:00] ผู้เข้าร่วม: ประโยคทดสอบความยาวพร้อมมูลสำหรับการวัดขนาดอินพุต "
    prompt = line * (150_000 // len(line) + 1)
    prompt = prompt[:150_000]
    client = _mk_client(cfg)
    t0 = time.monotonic()
    try:
        m = client.messages.create(
            model=cfg.gateway_model, max_tokens=64, temperature=0.0,
            messages=[{"role": "user", "content": prompt + "\n\nตอบเพียงคำว่า: รับทราบ"}],
        )
        dt = time.monotonic() - t0
        u = m.usage
        return {"ok": True, "elapsed": round(dt, 1),
                "detail": f"prompt_chars={len(prompt)} in_tok={getattr(u, 'input_tokens', '?')} "
                          f"out_tok={getattr(u, 'output_tokens', '?')}"}
    except Exception as exc:
        return {"ok": False, "elapsed": round(time.monotonic() - t0, 1),
                "detail": f"prompt_chars={len(prompt)} "
                          f"{type(exc).__name__}: {str(exc)[:120]}"}


def case_stream_small(cfg) -> dict:
    """Successful stream — records WHEN each SSE event type first arrives."""
    client = _mk_client(cfg)
    t0 = time.monotonic()
    marks: dict[str, float] = {}
    text_parts: list[str] = []
    try:
        with client.messages.stream(
            model=cfg.gateway_model, max_tokens=256, temperature=0.0,
            messages=[{"role": "user", "content":
                       "ตอบสั้น ๆ ครับ: 2+2 เท่ากับเท่าไร"}],
        ) as stream:
            for event in stream:
                etype = getattr(event, "type", "?")
                marks.setdefault(etype, round(time.monotonic() - t0, 1))
                d = getattr(event, "delta", None)
                if etype == "content_block_delta" and getattr(d, "type", "") == "text_delta":
                    text_parts.append(d.text)
            total = time.monotonic() - t0
        first_text = next((v for k, v in marks.items()
                           if k == "content_block_delta"), None)
        return {"ok": True, "elapsed": round(total, 1),
                "detail": f"message_start@{marks.get('message_start', '?')}s "
                          f"first_text@{first_text}s total={total:.1f}s "
                          f"chars={len(''.join(text_parts))} "
                          f"(early bytes={'YES' if first_text is not None and total - first_text > 1 else 'NO'})"}
    except Exception as exc:
        return {"ok": False, "elapsed": round(time.monotonic() - t0, 1),
                "detail": f"marks={marks} {type(exc).__name__}: {str(exc)[:120]}"}


def case_pure_decode(cfg) -> dict:
    """Minimal-thinking long literal output → true decode throughput."""
    client = _mk_client(cfg, timeout=135.0)
    t0 = time.monotonic()
    try:
        m = client.messages.create(
            model=cfg.gateway_model, max_tokens=6000, temperature=0.0,
            messages=[{"role": "user", "content":
                       "เขียนตัวเลข 1 ถึง 2500 คั่นด้วยจุลภาค บรรทัดเดียว "
                       "ห้ามอธิบายเพิ่มเติม ห้ามสรุป"}],
        )
        dt = time.monotonic() - t0
        u = m.usage
        tps = u.output_tokens / dt if dt else 0
        return {"ok": True, "elapsed": round(dt, 1),
                "detail": f"out_tok={u.output_tokens} PURE DECODE {tps:.1f} tok/s "
                          f"(→ 8192-token worst case ≈ {8192 / tps:.0f}s)"}
    except Exception as exc:
        return {"ok": False, "elapsed": round(time.monotonic() - t0, 1),
                "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}


def case_full_nonstream(cfg) -> dict:
    """Production-shaped summarize call through the Summarizer itself."""
    from meeting_bot.summarizer import Summarizer
    from tools.e2e_summarize_probe import build_transcript

    transcript, _ = build_transcript()
    prompt = transcript.to_prompt_text(max_chars=cfg.max_prompt_chars)
    summarizer = Summarizer(cfg)
    summarizer._client = _mk_client(cfg, timeout=135.0)  # outlive CF's 120s
    t0 = time.monotonic()
    try:
        raw = summarizer.summarize(prompt)
        return {"ok": True, "elapsed": round(time.monotonic() - t0, 1),
                "detail": f"raw_chars={len(raw)} "
                          f"usage={getattr(summarizer, 'last_usage', None)}"}
    except Exception as exc:
        return {"ok": False, "elapsed": round(time.monotonic() - t0, 1),
                "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}


def case_full_stream(cfg) -> dict:
    """Same production-shaped call, streamed — records event timeline."""
    from meeting_bot.summarizer import _SYSTEM_PROMPT, _USER_SUFFIX
    from tools.e2e_summarize_probe import build_transcript

    transcript, _ = build_transcript()
    prompt = transcript.to_prompt_text(max_chars=cfg.max_prompt_chars)
    client = _mk_client(cfg, timeout=600.0)
    t0 = time.monotonic()
    marks: dict[str, float] = {}
    n_text_deltas = 0
    try:
        with client.messages.stream(
            model=cfg.gateway_model, max_tokens=cfg.summary_max_tokens,
            temperature=0.0, system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt + _USER_SUFFIX}],
        ) as stream:
            for event in stream:
                etype = getattr(event, "type", "?")
                marks.setdefault(etype, round(time.monotonic() - t0, 1))
                if etype == "content_block_delta":
                    n_text_deltas += 1
            total = time.monotonic() - t0
        return {"ok": True, "elapsed": round(total, 1),
                "detail": f"completed; message_start@"
                          f"{marks.get('message_start', '?')}s "
                          f"first_delta@{marks.get('content_block_delta', '?')}s "
                          f"deltas={n_text_deltas}"}
    except Exception as exc:
        total = time.monotonic() - t0
        return {"ok": False, "elapsed": round(total, 1),
                "detail": f"killed at {total:.1f}s; marks={marks} "
                          f"{type(exc).__name__}: {str(exc)[:100]}"}


def case_err_old_model(cfg) -> dict:
    """qwen3.6-35b-a3b must stay rejected (off the allowlist) with HTTP 403."""
    client = _mk_client(cfg, timeout=30.0)
    t0 = time.monotonic()
    try:
        client.messages.create(model="qwen3.6-35b-a3b", max_tokens=16,
                               temperature=0.0,
                               messages=[{"role": "user", "content": "hi"}])
        return {"ok": False, "elapsed": round(time.monotonic() - t0, 1),
                "detail": "unexpectedly succeeded — old model is back?!"}
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return {"ok": status == 403,
                "elapsed": round(time.monotonic() - t0, 1),
                "detail": f"{type(exc).__name__} status={status} (expected 403)"}


def case_err_bogus_model(cfg) -> dict:
    return _expect_error(cfg, "nonexistent-model-xyz", None, "")


def case_err_bad_token(cfg) -> dict:
    return _expect_error(cfg, cfg.gateway_model, "sk-invalid-token-for-testing", "")


def _expect_error(cfg, model: str, token: str | None, needle: str) -> dict:
    client = _mk_client(cfg, timeout=30.0, token=token)
    t0 = time.monotonic()
    try:
        client.messages.create(model=model, max_tokens=16, temperature=0.0,
                               messages=[{"role": "user", "content": "hi"}])
        return {"ok": True, "elapsed": round(time.monotonic() - t0, 1),
                "detail": "unexpectedly succeeded"}
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        name = type(exc).__name__
        match = needle.lower() in (str(exc)[:200]).lower() if needle else True
        return {"ok": match, "elapsed": round(time.monotonic() - t0, 1),
                "detail": f"{name} status={status}"
                          + (f" (contains {needle!r})" if needle and match else "")
                          + ("" if match else f" MISSING {needle!r}")}


# ---------------------------------------------------------------------------

FAST_CASES = [
    ("tiny_probe", case_tiny_probe),
    ("small_output", case_small_output),
    ("large_input_150k", case_large_input),
    ("stream_small_timeline", case_stream_small),
    ("err_old_model", case_err_old_model),
    ("err_bogus_model", case_err_bogus_model),
    ("err_bad_token", case_err_bad_token),
]

SLOW_CASES = [
    ("pure_decode_rate", case_pure_decode),
    ("full_summarize_nonstream", case_full_nonstream),
    ("full_summarize_stream", case_full_stream),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true", help="run only fast cases")
    ap.add_argument("--slow", action="store_true", help="run only slow cases")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    from meeting_bot.config import load_config
    cfg = load_config()

    cases = []
    if args.fast or not (args.fast or args.slow):
        cases += FAST_CASES
    if args.slow or not (args.fast or args.slow):
        cases += SLOW_CASES

    print("=" * 78)
    print(f"gateway test matrix | base={cfg.anthropic_base_url} "
          f"model={cfg.gateway_model}")
    print(f"started {datetime.now().isoformat(timespec='seconds')} | "
          f"{len(cases)} cases")
    print("=" * 78)

    results = []
    for name, fn in cases:
        label = {"large_input_150k": "large_input(~150k)"}.get(name, name)
        print(f"\n>>> {label} ...", flush=True)
        res = fn(cfg)
        res["name"] = name
        results.append(res)
        flag = "PASS" if res["ok"] else "FAIL"
        print(f"    [{flag}] {res['elapsed']}s — {res['detail']}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'CASE':<28}{'RESULT':<8}{'TIME':>8}  DETAIL")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<28}{'PASS' if r['ok'] else 'FAIL':<8}"
              f"{r['elapsed']:>7}s  {r['detail'][:70]}")
    passed = sum(1 for r in results if r["ok"])
    print("-" * 78)
    print(f"{passed}/{len(results)} cases passed")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"results written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
