"""Per-meeting RAG store: transcript chunking + cosine vector index.

Pure stdlib(+numpy) at module scope — same import rule as
config/audio/chunker/transcript/summary_parse, so the jailed smoke-import
and ``pytest`` never touch Discord/MLX/network.

Sized for a single meeting (hundreds of chunks): a brute-force dot-product
index over L2-normalized float32 rows is more than fast enough, so there is
no ANN dependency. Nothing here persists across meetings.

Chunking mirrors :meth:`meeting_bot.transcript.Transcript.to_prompt_text`
rendering (``[MM:SS] speaker: text`` lines, chronological) and its
truncation semantics (whole lines only, ``...(truncated)`` suffix).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .transcript import TranscriptEvent

__all__ = ["Chunk", "chunk_transcript", "VectorIndex", "truncate_block"]

_TRUNCATION_SUFFIX = "...(truncated)"


@dataclass(frozen=True)
class Chunk:
    """One line-aligned window of the rendered transcript."""

    chunk_id: int
    header: str      # e.g. "[02:10–05:40]" — time span the chunk covers
    text: str        # complete "[MM:SS] speaker: text" lines, newline-joined
    start_sec: float  # offset of the chunk's first line from meeting start

    def as_embedding_text(self) -> str:
        """Text handed to the embedder: header gives temporal context."""
        return f"{self.header}\n{self.text}"


def _format_mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _span_header(lines: list[tuple[str, float]]) -> str:
    first = _format_mmss(lines[0][1])
    if len(lines) == 1:
        return f"[{first}]"
    return f"[{first}\u2013{_format_mmss(lines[-1][1])}]"


def chunk_transcript(
    events: list["TranscriptEvent"],
    started_at: float,
    chunk_chars: int = 800,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Sliding-window chunking over chronological transcript lines.

    Lines are rendered exactly like ``Transcript.to_prompt_text`` does and
    packed greedily up to ``chunk_chars`` characters. A line is never split:
    a line longer than the window becomes its own (oversized) chunk. When a
    window closes, the trailing lines totalling up to ``overlap_chars``
    characters are repeated at the start of the next chunk so consecutive
    chunks share context.

    Returns ``[]`` for an empty transcript.
    """
    lines: list[tuple[str, float]] = []
    for event in sorted(events, key=lambda e: e.start):
        elapsed = max(0.0, float(event.start) - float(started_at))
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        lines.append((f"[{minutes:02d}:{seconds:02d}] {event.speaker}: {event.text}", elapsed))

    chunks: list[Chunk] = []
    window: list[tuple[str, float]] = []
    window_len = 0

    def _flush() -> None:
        if not window:
            return
        chunks.append(
            Chunk(
                chunk_id=len(chunks),
                header=_span_header(window),
                text="\n".join(text for text, _ in window),
                start_sec=window[0][1],
            )
        )

    for line, sec in lines:
        extra = len(line) + (1 if window else 0)
        if window and window_len + extra > chunk_chars:
            _flush()
            # Carry trailing lines into the next window as overlap context.
            # Even the most recent line only qualifies when it fits the
            # overlap budget, so overlap_chars=0 repeats nothing.
            overlap: list[tuple[str, float]] = []
            overlap_len = 0
            for text, ts in reversed(window):
                add = len(text) + (1 if overlap else 0)
                if overlap_len + add > overlap_chars:
                    break
                overlap.append((text, ts))
                overlap_len += add
            overlap.reverse()
            window = overlap
            window_len = overlap_len
            extra = len(line) + (1 if window else 0)
        window.append((line, sec))
        window_len += extra
    _flush()
    return chunks


def truncate_block(text: str, max_chars: int) -> str:
    """Line-preserving truncation, byte-compatible with the suffix/split
    behavior of ``Transcript.to_prompt_text`` (whole lines kept, a single
    oversized line is sliced, ``...(truncated)`` appended when cut)."""
    if max_chars is None or len(text) <= max_chars:
        return text
    budget = max_chars - len(_TRUNCATION_SUFFIX)
    if budget <= 0:
        return text[:max_chars]
    kept: list[str] = []
    used = 0
    for line in text.split("\n"):
        cost = len(line) + (1 if kept else 0)
        if used + cost <= budget:
            kept.append(line)
            used += cost
        else:
            break
    if not kept:
        return text[:budget] + _TRUNCATION_SUFFIX
    return "\n".join(kept) + "\n" + _TRUNCATION_SUFFIX


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0  # leave zero vectors as-is
    return mat / norms


class VectorIndex:
    """Brute-force cosine index over L2-normalized chunk vectors.

    Vectors are normalized on ``add`` and query vectors on ``query``, so
    every similarity is a plain dot product. Empty-index queries are safe
    and return ``[]``.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # (n, d) float32, unit rows

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, vectors, chunks: list[Chunk]) -> None:
        """Append a batch of chunk vectors (rows aligned with *chunks*)."""
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(f"vectors must be 2-D (n, d), got shape {mat.shape}")
        if len(chunks) != mat.shape[0]:
            raise ValueError(
                f"got {mat.shape[0]} vectors for {len(chunks)} chunks"
            )
        if mat.shape[0] == 0:
            return
        if self._matrix is not None and mat.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                f"dimension mismatch: index has {self._matrix.shape[1]}, "
                f"batch has {mat.shape[1]}"
            )
        mat = _l2_normalize(mat)
        if self._matrix is None:
            self._matrix = mat
            self._chunks = list(chunks)
        else:
            self._matrix = np.vstack([self._matrix, mat])
            self._chunks.extend(chunks)

    def query(self, vector, k: int) -> list[tuple[Chunk, float]]:
        """Top-*k* ``(chunk, score)`` by cosine similarity, best first."""
        if self._matrix is None or k <= 0:
            return []
        q = _l2_normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        scores = self._matrix @ q[0]
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._chunks[int(i)], float(scores[int(i)])) for i in order]

    def query_multi(self, vectors, k: int) -> list[tuple[Chunk, float]]:
        """Union of the top-*k* hits per query row.

        Duplicates (same chunk hit by several queries) keep their **max**
        score; the final list is re-sorted chronologically by the chunk's
        position in the meeting so the summarized excerpt reads in order.
        """
        if self._matrix is None or k <= 0:
            return []
        qs = np.asarray(vectors, dtype=np.float32)
        if qs.ndim != 2 or qs.shape[0] == 0:
            return []
        qs = _l2_normalize(qs)
        best: dict[int, float] = {}
        for qi in range(qs.shape[0]):
            scores = self._matrix @ qs[qi]
            for i in np.argsort(-scores, kind="stable")[:k]:
                idx = int(i)
                score = float(scores[idx])
                if idx not in best or score > best[idx]:
                    best[idx] = score
        ranked = sorted(
            best.items(),
            key=lambda item: (self._chunks[item[0]].start_sec, self._chunks[item[0]].chunk_id),
        )
        return [(self._chunks[idx], score) for idx, score in ranked]
