#!/usr/bin/env python3
"""Offline A/B harness for the whisper anti-loop fix (not a pytest test).

Routes a 48 kHz stereo int16 WAV through the **real** capture pipeline —
``audio.resample_48k_stereo_to_16k_mono`` -> ``SilenceChunker`` (0.8 s silence,
1.0 s min / 30.0 s max chunk) fed in 20 ms frames — then decodes each closed
segment three ways with mlx-whisper:

  (a) the old all-defaults call (what the bot did pre-fix)
  (b) the hardened primary  (T=0, condition_on_previous_text=False, no prompt)
  (c) the anti-loop retry    (temperature bump + Thai preamble)

so you can confirm the fix decodes clean Thai where the defaults hallucinate,
and A/B the knobs (``--whisper-fp16 0``, ``WHISPER_RETRY_TEMPERATURE``,
``WHISPER_INITIAL_PROMPT``) against real dumps from ``DUMP_CHUNKS_DIR``.

Usage:
  say -v Kanya -o /tmp/thai.aiff \\
    "สวัสดีครับ วันนี้เราจะพูดคุยเรื่องงบประมาณและแผนงานในไตรมาสหน้า"
  afconvert -f WAVE -d LEI16@48000 -c 2 /tmp/thai.aiff /tmp/thai_48k_stereo.wav
  python3 tools/offline_repro.py --wav /tmp/thai_48k_stereo.wav
  python3 tools/offline_repro.py --wav /tmp/dump.wav --whisper-fp16 0
"""

from __future__ import annotations

import argparse
import os
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

# Make the repo root importable when run as `python3 tools/offline_repro.py`
# (sys.path[0] would otherwise be tools/). Same bootstrap as tests/conftest.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Heavy deps are imported lazily so --help / arg-parse still works without the
# MLX stack. meeting_bot.audio + meeting_bot.chunker are pure (numpy only).
from meeting_bot.audio import resample_48k_stereo_to_16k_mono
from meeting_bot.chunker import SilenceChunker
from meeting_bot.transcriber import (
    build_decode_kwargs,
    is_garbage_transcription,
    primary_decode_settings,
    retry_decode_settings,
    should_retry,
)

FRAME_BYTES_48K_STEREO = 3840   # 20 ms @ 48 kHz × 2 ch × int16
FRAME_SECONDS = 20e-3


@dataclass
class _DecodeResult:
    label: str
    text: str
    no_speech_prob: float
    garbage: bool
    retry: bool


def _decode_all_defaults(model: str, language: str, samples):
    """Replicate the pre-fix call: no decode kwargs, mlx defaults."""
    import mlx_whisper

    result = mlx_whisper.transcribe(samples, path_or_hf_repo=model, language=language)
    text = (result.get("text") or "").strip()
    segs = result.get("segments") or []
    nsp = segs[0].get("no_speech_prob", 0.0) if segs else 0.0
    return text, nsp


def _decode_with_kwargs(model: str, language: str, samples, label: str, settings) -> _DecodeResult:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        samples,
        path_or_hf_repo=model,
        **build_decode_kwargs(settings, language=language),
    )
    text = (result.get("text") or "").strip()
    segs = result.get("segments") or []
    nsp = segs[0].get("no_speech_prob", 0.0) if segs else 0.0
    return _DecodeResult(
        label=label,
        text=text,
        no_speech_prob=nsp,
        garbage=is_garbage_transcription(text),
        retry=should_retry(text, nsp),
    )


def _iter_48k_stereo_frames(path: str):
    """Yield 3840-byte int16 frames from a 48 kHz stereo WAV."""
    with wave.open(path, "rb") as w:
        if w.getframerate() != 48000:
            raise SystemExit(
                f"--wav must be 48 kHz (got {w.getframerate()}); produce it with "
                "`afconvert -f WAVE -d LEI16@48000 -c 2`"
            )
        if w.getnchannels() != 2:
            raise SystemExit(f"--wav must be stereo (got {w.getnchannels()} ch)")
        if w.getsampwidth() != 2:
            raise SystemExit(f"--wav must be int16 PCM (got {w.getsampwidth()} bytes)")
        while True:
            # 960 stereo frames = 3840 bytes = 20 ms @ 48 kHz, exactly what
            # resample_48k_stereo_to_16k_mono expects per call.
            frame = w.readframes(960)
            if len(frame) == 0:
                break
            if len(frame) != FRAME_BYTES_48K_STEREO:
                # Truncated tail: pad with silence to keep the resampler happy.
                frame += b"\x00" * (FRAME_BYTES_48K_STEREO - len(frame))
            yield frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", required=True, help="48 kHz stereo int16 WAV file")
    ap.add_argument(
        "--model",
        default=os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-mlx"),
        help="mlx-whisper model id (default: $WHISPER_MODEL)",
    )
    ap.add_argument("--language", default="th", help="whisper language (default: th)")
    ap.add_argument(
        "--whisper-fp16",
        default=os.environ.get("WHISPER_FP16", "1"),
        help="fp16 compute for all three decodes (0/1; must be constant per process)",
    )
    ap.add_argument(
        "--frame-ms",
        default=20,
        type=int,
        help="20 ms resample frames (matches the live sink; no need to change)",
    )
    args = ap.parse_args()

    os.environ["WHISPER_FP16"] = args.whisper_fp16

    # 1. Resample + chunk through the real pipeline.
    chunker = SilenceChunker(
        silence_seconds=0.8,
        min_chunk_seconds=1.0,
        max_chunk_seconds=30.0,
    )
    chunker.speaker_key = "offline"
    chunker.speaker_name = "offline"
    segments = []
    now = 0.0
    total_frames = 0
    for frame in _iter_48k_stereo_frames(args.wav):
        mono = resample_48k_stereo_to_16k_mono(frame)
        now += FRAME_SECONDS
        total_frames += 1
        segments.extend(chunker.feed(mono, now))
    segments.extend(chunker.flush(now))

    if not segments:
        print(
            f"No speech segments (fed {total_frames} frames, {total_frames * FRAME_SECONDS:.1f}s). "
            "Is the WAV actually speech? Try SILENCE_THRESHOLD=0.01 (default) or a louder file."
        )
        return 1

    # 2. Decode each segment three ways.
    print(f"model={args.model} fp16={args.whisper_fp16}")
    print(f"segments={len(segments)} total_audio={total_frames * FRAME_SECONDS:.1f}s\n")
    all_results: list[list[_DecodeResult]] = []
    for i, seg in enumerate(segments, 1):
        samples = seg.samples
        print(f"--- segment {i}: speaker={seg.speaker_name} start={seg.start:.2f}s "
              f"dur={seg.duration:.2f}s samples={len(samples)} "
              f"min={samples.min():.4f} max={samples.max():.4f}")

        # (a) old all-defaults call
        text, nsp = _decode_all_defaults(args.model, args.language, samples)
        results = [
            _DecodeResult(
                label="(a) all-defaults",
                text=text,
                no_speech_prob=nsp,
                garbage=is_garbage_transcription(text),
                retry=should_retry(text, nsp),
            )
        ]
        # (b) hardened primary, (c) anti-loop retry
        results.append(
            _decode_with_kwargs(args.model, args.language, samples, "(b) hardened-primary",
                                primary_decode_settings())
        )
        results.append(
            _decode_with_kwargs(args.model, args.language, samples, "(c) anti-loop-retry",
                                retry_decode_settings())
        )

        for r in results:
            print(
                f"  {r.label}: no_speech_prob={r.no_speech_prob:.4f} "
                f"garbage={r.garbage} should_retry={r.retry}\n"
                f"    {r.text!r}"
            )
        print()
        all_results.append(results)

    # 3. Verdict: the fix succeeds when any hardened-path decode is clean.
    clean = [
        r for segment_results in all_results for r in segment_results
        if r.text.strip() and not r.garbage
    ]
    print("verdict:", "OK — clean Thai decodes on the hardened paths"
          if clean else "STILL GARBAGE — A/B the knobs (fp16/retry-temperature/prompt)")
    return 0 if clean else 2


if __name__ == "__main__":
    sys.exit(main())
