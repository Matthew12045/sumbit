#!/usr/bin/env python3
"""
End-to-end comparison of two MLX Whisper models on Thai ASR.

Compares two models side-by-side on the same audio samples:
  - Current model (default: mlx-community/whisper-large-v3-mlx)
  - Candidate model (default: scb10x/typhoon-whisper-large-v3 converted to MLX)

Uses Google FLEURS Thai test set as the audio + ground-truth source.
Does NOT modify .env, WHISPER_MODEL, or any live config.

Usage:
    python3 tools/e2e_whisper_compare.py              # defaults (20 samples)
    python3 tools/e2e_whisper_compare.py -n 50         # 50 samples
    python3 tools/e2e_whisper_compare.py --model2 /tmp/typhoon-whisper-large-v3-mlx
    python3 tools/e2e_whisper_compare.py --seed 42     # reproducible sample selection
    python3 tools/e2e_whisper_compare.py --all          # all 1021 FLEURS Thai samples (slow)

Output: per-sample transcripts, WER/CER, garbage/repetition flags, decode times,
and an aggregate report. No live config is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Imports — these are lazy; script fails fast if mlx-whisper isn't available
# ---------------------------------------------------------------------------
import mlx_whisper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLEURS_REPO = "google/fleurs"
LANG = "th_th"
AUDIO_DIR_IN_TAR = "test"  # FLEURS test audio lives in test/ within the tar
GROUND_TRUTH_TSV = f"data/{LANG}/test.tsv"

# Default models
DEFAULT_MODEL_CURRENT = "mlx-community/whisper-large-v3-mlx"
DEFAULT_MODEL_CANDIDATE = "/tmp/typhoon-whisper-large-v3-mlx"

# Decode settings — identical for both models, mirroring production
# (transcriber.build_decode_kwargs): T=0 greedy primary decode, no
# context carry-over, beam search per WHISPER_BEAM_SIZE. The
# compression_ratio/logprob fallback gates are deliberately NOT passed —
# they never fire for confident loops and only distort the benchmark
# (see transcriber.py header).
COMMON_DECODE_KWAGS_BEAM = int(os.environ.get("WHISPER_BEAM_SIZE", "0") or 0)
COMMON_DECODE_KWARGS = dict(
    temperature=0.0,
    condition_on_previous_text=False,
    no_speech_threshold=0.6,
    fp16=True,
    language="th",
    task="transcribe",
)
if COMMON_DECODE_KWAGS_BEAM > 1:
    COMMON_DECODE_KWARGS["beam_size"] = COMMON_DECODE_KWAGS_BEAM

# Garbage detection heuristics (same as transcriber.py)
GARBAGE_MIN_CHARS = 10
GARBAGE_SPECIAL_RATIO = 0.30
GARBAGE_REPETITION_NGRAM = 4
GARBAGE_REPETITION_THRESHOLD = 0.5

# Repetition-loop detection (same as summarizer.py)
REPETITION_WINDOW_CHARS = 300
REPETITION_MIN_REPEATS = 3

# ---------------------------------------------------------------------------
# FLEURS data management
# ---------------------------------------------------------------------------


def ensure_fleurs_data(cache_dir: Path) -> List[Tuple[str, str, str]]:
    """Download and extract FLEURS Thai test set if not already cached.

    Returns list of (entry_id, wav_path, transcription).

    Layout note: ``hf_hub_download(..., cache_dir=...)`` stores files in the
    hub snapshot layout (``datasets--google--fleurs/snapshots/<sha>/…``),
    while the extracted tar places wavs directly at ``cache_dir/test/``.
    """
    wav_dir = cache_dir  # _load_fleurs_entries appends AUDIO_DIR_IN_TAR ("test")
    if any(wav_dir.glob("test/*.wav")):
        candidates = sorted(cache_dir.rglob("test.tsv"))
        if candidates:
            return _load_fleurs_entries(candidates[0], wav_dir)

    # Download
    from huggingface_hub import hf_hub_download

    print("Downloading FLEURS Thai test.tsv ...")
    hf_hub_download(
        FLEURS_REPO, GROUND_TRUTH_TSV, repo_type="dataset", cache_dir=str(cache_dir)
    )
    print("Downloading FLEURS Thai test audio tarball ...")
    tar_path = hf_hub_download(
        FLEURS_REPO,
        f"data/{LANG}/audio/test.tar.gz",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )

    print("Extracting audio ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=str(cache_dir))

    candidates = sorted(cache_dir.rglob("test.tsv"))
    if not candidates:
        raise FileNotFoundError(f"FLEURS test.tsv not found under {cache_dir}")
    return _load_fleurs_entries(candidates[0], wav_dir)


def _load_fleurs_entries(tsv_path: Path, wav_dir: Path) -> List[Tuple[str, str, str]]:
    """Parse FLEURS TSV and return valid (id, path, transcription) tuples."""
    entries = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            entry_id, filename, transcription = parts[0], parts[1], parts[2]
            wav_path = str(wav_dir / AUDIO_DIR_IN_TAR / filename)
            if os.path.exists(wav_path):
                entries.append((entry_id, wav_path, transcription))
    return entries


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def select_samples(
    entries: List[Tuple[str, str, str]],
    n: int,
    seed: int = 0,
) -> List[Tuple[str, str, str]]:
    """Select a diverse subset of samples.

    Strategy: group by entry_id (same transcription, different recordings),
    pick one per group for diversity.
    """
    rng = np.random.RandomState(seed)
    if n >= len(entries):
        return entries

    # Group by entry_id
    groups: dict[str, List[Tuple[str, str, str]]] = {}
    for entry in entries:
        groups.setdefault(entry[0], []).append(entry)

    # Pick one random entry per group
    representatives = [
        group[int(rng.randint(len(group)))]
        for group in groups.values()
    ]
    # Shuffle and take top n
    rng.shuffle(representatives)
    return [tuple(r) for r in representatives[:n]]


# ---------------------------------------------------------------------------
# Garbage / repetition detection (mirrors transcriber.py / summarizer.py)
# ---------------------------------------------------------------------------


def _has_too_many_special_chars(text: str) -> bool:
    """Flag text with an abnormally high ratio of non-alphanumeric/non-Thai chars."""
    if len(text) < GARBAGE_MIN_CHARS:
        return False
    # Thai + alphanumeric + whitespace + common punctuation
    allowed = re.compile(
        r"^[฀-๿ัิภ-ษก-ส"
        r"a-zA-Z0-9\s.,;:!?\"\'()-_\/ะ์ํ๎]+$",
        re.UNICODE,
    )
    special = sum(1 for c in text if not allowed.match(c))
    return (special / len(text)) > GARBAGE_SPECIAL_RATIO


def _has_repetition_ngram(text: str, n: int = 4) -> bool:
    """Flag text where an n-gram repeats excessively."""
    if len(text) < n * 2:
        return False
    chars = list(text)
    ngrams = ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]
    if not ngrams:
        return False
    from collections import Counter

    counts = Counter(ngrams)
    max_count = max(counts.values())
    return (max_count / len(ngrams)) > GARBAGE_REPETITION_THRESHOLD


def is_garbage(text: str) -> bool:
    """Return True if the transcript looks like model garbage / hallucination."""
    return _has_too_many_special_chars(text) or _has_repetition_ngram(text)


def is_repetition_loop(text: str) -> bool:
    """Return True if the text appears stuck in a confident repeat loop.

    Mirrors the summarizer's _is_looping logic: slide a window and check if
    the second half repeats the first half.
    """
    if len(text) < REPETITION_WINDOW_CHARS:
        return False
    window = text[-REPETITION_WINDOW_CHARS:]
    half = REPETITION_WINDOW_CHARS // 2
    first_half = window[:half]
    second_half = window[half:]
    return second_half == first_half


# ---------------------------------------------------------------------------
# WER / CER computation
# ---------------------------------------------------------------------------


def _edit_distance(a: list, b: list) -> int:
    """Standard Levenshtein edit distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate — Thai is better evaluated at character level."""
    if not reference and not hypothesis:
        return 0.0
    if not reference or not hypothesis:
        return 1.0
    dist = _edit_distance(list(reference), list(hypothesis))
    return dist / max(len(reference), len(hypothesis))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate — Thai has no spaces, so we use 2-char bins as proxy."""
    if not reference and not hypothesis:
        return 0.0
    if not reference or not hypothesis:
        return 1.0
    # Thai doesn't have word boundaries; use 2-char "pseudo-words"
    ref_bins = [reference[i : i + 2] for i in range(0, len(reference), 2)]
    hyp_bins = [hypothesis[i : i + 2] for i in range(0, len(hypothesis), 2)]
    dist = _edit_distance(ref_bins, hyp_bins)
    return dist / max(len(ref_bins), len(hyp_bins))


# ---------------------------------------------------------------------------
# Transcription runner
# ---------------------------------------------------------------------------


@dataclass
class TranscriptResult:
    text: str
    decode_time_sec: float
    was_garbage: bool
    is_loop: bool
    cer: float
    wer: float
    error: Optional[str] = None


def _load_wav_as_float32(wav_path: str) -> np.ndarray:
    """Load a 16 kHz mono WAV as float32 in [-1, 1] — mirroring the exact
    input the production Transcriber hands mlx-whisper (numpy array, no
    ffmpeg dependency).

    Uses a small RIFF reader because FLEURS ships IEEE-float WAVs (format
    tag 3), which the stdlib ``wave`` module rejects.
    """
    import struct

    with open(wav_path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {wav_path}")

    pos = 12
    fmt = None
    raw = None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            audio_format, channels, rate = struct.unpack("<HHI", body[:8])
            fmt = audio_format
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size % 2)

    if fmt not in (1, 3) or raw is None:
        raise ValueError(f"unsupported WAV format {fmt}: {wav_path}")
    if fmt == 1:
        if len(raw) % 2:
            raw = raw[: len(raw) - len(raw) % 2]
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        samples = np.frombuffer(raw, dtype=np.float32)
    # FLEURS is mono; average channels just in case a future set isn't.
    if channels > 1:
        samples = samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        raise ValueError(f"expected 16 kHz WAV, got {rate} Hz: {wav_path}")
    return np.ascontiguousarray(samples, dtype=np.float32)


def transcribe(
    model_path: str, wav_path: str, kwargs: dict
) -> TranscriptResult:
    """Run one transcription and collect metrics.

    ``model_path`` is a local dir or HF repo id passed via
    ``path_or_hf_repo`` — mlx-whisper caches loaded models by path
    internally, so explicit preloading is unnecessary. Audio is decoded to
    a float32 array first so the benchmark matches production's array input.
    """
    t0 = time.monotonic()
    try:
        result = mlx_whisper.transcribe(
            _load_wav_as_float32(wav_path), path_or_hf_repo=model_path, **kwargs
        )
        decode_time = time.monotonic() - t0
        text = result.get("text", "").strip()
    except Exception as exc:
        decode_time = time.monotonic() - t0
        return TranscriptResult(
            text="",
            decode_time_sec=decode_time,
            was_garbage=False,
            is_loop=False,
            cer=1.0,
            wer=1.0,
            error=str(exc),
        )

    return TranscriptResult(
        text=text,
        decode_time_sec=decode_time,
        was_garbage=is_garbage(text),
        is_loop=is_repetition_loop(text),
        cer=0.0,  # filled in by caller
        wer=0.0,
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _print_sample(
    idx: int,
    entry_id: str,
    wav_name: str,
    ground_truth: str,
    current: TranscriptResult,
    candidate: TranscriptResult,
    model1_name: str,
    model2_name: str,
    max_field: int = 80,
):
    """Print a single sample comparison block."""
    name1 = model1_name.split('/')[-1]
    name2 = model2_name.split('/')[-1]
    print(f"\n{'=' * 80}")
    print(f"Sample {idx + 1:>3d}  [entry_id={entry_id}]  {wav_name}")
    print(f"{'=' * 80}")
    print(f"  Ground truth: {ground_truth[:max_field]}...")

    # Model 1
    print(f"\n  ── {name1} ──")
    if current.error:
        print(f"    ERROR: {current.error}")
    else:
        print(f"    Transcript: {current.text[:max_field]}...")
        flags = []
        if current.was_garbage:
            flags.append("⚠️ GARBAGE")
        if current.is_loop:
            flags.append("🔁 LOOP")
        if flags:
            print(f"    Flags: {', '.join(flags)}")
        print(f"    CER={current.cer:.3f}  WER={current.wer:.3f}  time={current.decode_time_sec:.2f}s")

    # Model 2
    print(f"\n  ── {name2} ──")
    if candidate.error:
        print(f"    ERROR: {candidate.error}")
    else:
        print(f"    Transcript: {candidate.text[:max_field]}...")
        flags = []
        if candidate.was_garbage:
            flags.append("⚠️ GARBAGE")
        if candidate.is_loop:
            flags.append("🔁 LOOP")
        if flags:
            print(f"    Flags: {', '.join(flags)}")
        print(f"    CER={candidate.cer:.3f}  WER={candidate.wer:.3f}  time={candidate.decode_time_sec:.2f}s")

    # Verdict
    if not current.error and not candidate.error:
        if current.cer < candidate.cer:
            delta = candidate.cer - current.cer
            print(f"\n  ✅ Current wins by CER delta {delta:.3f}")
        elif candidate.cer < current.cer:
            delta = current.cer - candidate.cer
            print(f"\n  ✅ Candidate wins by CER delta {delta:.3f}")
        else:
            print(f"\n  ➖ Tie on CER")


def _print_report(
    samples: List[dict],
    n: int,
    model_current_name: str,
    model_candidate_name: str,
):
    """Print the aggregate comparison report."""
    valid = [s for s in samples if s["current"].error is None]
    n_valid = len(valid)

    print(f"\n\n{'#' * 80}")
    print(f"#  COMPARISON REPORT  ({n_valid}/{n} samples with valid output)")
    print(f"{'#' * 80}")

    if n_valid == 0:
        print("No valid transcriptions from either model.")
        return

    # Gather metrics
    current_cers = [s["current"].cer for s in valid]
    candidate_cers = [s["candidate"].cer for s in valid]
    current_wers = [s["current"].wer for s in valid]
    candidate_wers = [s["candidate"].wer for s in valid]
    current_times = [s["current"].decode_time_sec for s in valid]
    candidate_times = [s["candidate"].decode_time_sec for s in valid]

    current_garbage_count = sum(1 for s in valid if s["current"].was_garbage)
    candidate_garbage_count = sum(1 for s in valid if s["candidate"].was_garbage)
    current_loop_count = sum(1 for s in valid if s["current"].is_loop)
    candidate_loop_count = sum(1 for s in valid if s["candidate"].is_loop)

    # CER stats
    def _stats(vals):
        if not vals:
            return {}
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    print(f"\n{'Model':<45} {'CER-mean':>10} {'CER-median':>12} {'CER-stdev':>12} "
          f"{'WER-mean':>10} {'Time-mean':>12} {'Time-median':>13}")
    print(f"{'-' * 110}")

    for name, cers, wers, times in [
        (model_current_name[-45:], current_cers, current_wers, current_times),
        (model_candidate_name[-45:], candidate_cers, candidate_wers, candidate_times),
    ]:
        cs = _stats(cers)
        ws = _stats(wers)
        ts = _stats(times)
        print(f"{name:<45} "
              f"{cs.get('mean', 0):>10.4f} "
              f"{cs.get('median', 0):>12.4f} "
              f"{cs.get('stdev', 0):>12.4f} "
              f"{ws.get('mean', 0):>10.4f} "
              f"{ts.get('mean', 0):>12.3f} "
              f"{ts.get('median', 0):>13.3f}")

    # Garbage / loop rates
    print(f"\n{'Metric':<45} {'Current':>12} {'Candidate':>12}")
    print(f"{'-' * 70}")

    curr_garbage_pct = current_garbage_count / max(n_valid, 1) * 100
    cand_garbage_pct = candidate_garbage_count / max(n_valid, 1) * 100
    curr_garbage_str = f"{current_garbage_count}/{n_valid} ({curr_garbage_pct:.1f}%)"
    cand_garbage_str = f"{candidate_garbage_count}/{n_valid} ({cand_garbage_pct:.1f}%)"
    print(f"{'Garbage rate':<45} {curr_garbage_str:>14} {cand_garbage_str:>14}")

    curr_loop_pct = current_loop_count / max(n_valid, 1) * 100
    cand_loop_pct = candidate_loop_count / max(n_valid, 1) * 100
    curr_loop_str = f"{current_loop_count}/{n_valid} ({curr_loop_pct:.1f}%)"
    cand_loop_str = f"{candidate_loop_count}/{n_valid} ({cand_loop_pct:.1f}%)"
    print(f"{'Loop rate':<45} {curr_loop_str:>14} {cand_loop_str:>14}")

    # CER winners
    current_wins = sum(1 for s in valid if s["current"].cer < s["candidate"].cer)
    candidate_wins = sum(1 for s in valid if s["candidate"].cer < s["current"].cer)
    ties = n_valid - current_wins - candidate_wins
    print(f"\n{'CER Winner':<45} Count")
    print(f"{'-' * 50}")
    print(f"{model_current_name[-45:] + ' wins':<45} {current_wins:>6d}  ({current_wins/max(n_valid,1)*100:.1f}%)")
    print(f"{model_candidate_name[-45:] + ' wins':<45} {candidate_wins:>6d}  ({candidate_wins/max(n_valid,1)*100:.1f}%)")
    print(f"{'Ties':<45} {ties:>6d}  ({ties/max(n_valid,1)*100:.1f}%)")

    # Speed
    if statistics.mean(current_times) > 0:
        speedup = statistics.mean(current_times) / statistics.mean(candidate_times)
        print(f"\nSpeed: candidate is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than current")

    # Overall verdict
    print(f"\n{'=' * 80}")
    if candidate_cers and current_cers:
        cand_mean = statistics.mean(candidate_cers)
        curr_mean = statistics.mean(current_cers)
        delta = curr_mean - cand_mean
        if abs(delta) < 0.01:
            print(f"VERDICT: CER is approximately equal (delta={delta:+.4f}). "
                  f"Consider other factors (model size, loop propensity, speed).")
        elif delta > 0:
            print(f"VERDICT: Candidate {model_candidate_name.split('/')[-1]} wins on CER by {delta:.4f}. "
                  f"RECOMMENDED for evaluation as replacement.")
        else:
            print(f"VERDICT: Current {model_current_name.split('/')[-1]} wins on CER by {-delta:.4f}. "
                  f"NOT recommended to swap.")
    print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare two MLX Whisper models on Thai ASR (FLEURS test set)."
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=20,
        help="Number of test samples to evaluate (default: 20, use --all for all)",
    )
    parser.add_argument(
        "--model1", type=str, default=DEFAULT_MODEL_CURRENT,
        help=f"Current model path/name (default: {DEFAULT_MODEL_CURRENT})",
    )
    parser.add_argument(
        "--model2", type=str, default=DEFAULT_MODEL_CANDIDATE,
        help=f"Candidate model path (default: {DEFAULT_MODEL_CANDIDATE})",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="/tmp/fleurs_cache",
        help="Directory to cache FLEURS data (default: /tmp/fleurs_cache)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run on all available Thai test samples",
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Write per-sample results as JSON to this path",
    )
    args = parser.parse_args()

    n_samples = 1021 if args.all else args.num_samples

    print("=" * 80)
    print("  MLX Whisper Thai ASR Comparison")
    print(f"  Model 1 (current):  {args.model1}")
    print(f"  Model 2 (candidate): {args.model2}")
    print(f"  Samples: {n_samples}")
    print("=" * 80)

    # 1. Get FLEURS data
    cache_dir = Path(args.cache_dir)
    print(f"\nFetching FLEURS Thai test data (cache: {cache_dir}) ...")
    entries = ensure_fleurs_data(cache_dir)
    print(f"Found {len(entries)} Thai test entries.")

    # 2. Select samples
    samples = select_samples(entries, n=n_samples, seed=args.seed)
    print(f"Selected {len(samples)} diverse samples (seed={args.seed}).")

    # 3. Run transcriptions (models load lazily via path_or_hf_repo;
    # mlx-whisper caches by path so each model loads exactly once)
    results = []
    for idx, (entry_id, wav_path, ground_truth) in enumerate(samples):
        wav_name = os.path.basename(wav_path)
        print(
            f"\rProcessing {idx + 1}/{len(samples)}: {wav_name[:40]}...",
            end="",
            flush=True,
        )

        r1 = transcribe(args.model1, wav_path, COMMON_DECODE_KWARGS)
        r2 = transcribe(args.model2, wav_path, COMMON_DECODE_KWARGS)

        # Fill in CER/WER
        r1.cer = compute_cer(ground_truth, r1.text)
        r1.wer = compute_wer(ground_truth, r1.text)
        r2.cer = compute_cer(ground_truth, r2.text)
        r2.wer = compute_wer(ground_truth, r2.text)

        results.append({
            "idx": idx + 1,
            "entry_id": entry_id,
            "wav_name": wav_name,
            "ground_truth": ground_truth,
            "current": r1,
            "candidate": r2,
        })

    print()  # newline after progress line

    # 5. Print per-sample comparisons (first 10 samples)
    print_samples = min(10, len(results))
    for i in range(print_samples):
        _print_sample(
            i,
            results[i]["entry_id"],
            results[i]["wav_name"],
            results[i]["ground_truth"],
            results[i]["current"],
            results[i]["candidate"],
            args.model1,
            args.model2,
        )

    if len(results) > 10:
        print(f"\n  ... ({len(results) - 10} more samples not shown individually)")

    # 6. Aggregate report
    _print_report(
        results,
        len(samples),
        args.model1,
        args.model2,
    )

    # 7. Optional JSON output
    if args.json_out:
        # Make results JSON-serializable
        serializable = []
        for r in results:
            serializable.append({
                "idx": r["idx"],
                "entry_id": r["entry_id"],
                "wav_name": r["wav_name"],
                "ground_truth": r["ground_truth"],
                "current": {
                    "text": r["current"].text,
                    "decode_time_sec": round(r["current"].decode_time_sec, 3),
                    "was_garbage": r["current"].was_garbage,
                    "is_loop": r["current"].is_loop,
                    "cer": round(r["current"].cer, 4),
                    "wer": round(r["current"].wer, 4),
                    "error": r["current"].error,
                },
                "candidate": {
                    "text": r["candidate"].text,
                    "decode_time_sec": round(r["candidate"].decode_time_sec, 3),
                    "was_garbage": r["candidate"].was_garbage,
                    "is_loop": r["candidate"].is_loop,
                    "cer": round(r["candidate"].cer, 4),
                    "wer": round(r["candidate"].wer, 4),
                    "error": r["candidate"].error,
                },
            })
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"\nJSON results written to {args.json_out}")


if __name__ == "__main__":
    main()
