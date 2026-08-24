"""Pure-logic tests for meeting_bot.rag_store (no MLX / Discord / network)."""

import numpy as np
import pytest

from meeting_bot.rag_store import Chunk, VectorIndex, chunk_transcript, truncate_block
from meeting_bot.transcript import TranscriptEvent


def _events(*pairs):
    return [TranscriptEvent(speaker=s, start=t, text=txt) for s, t, txt in pairs]


def _vec(i: int, d: int = 4) -> np.ndarray:
    v = np.zeros((1, d), dtype=np.float32)
    v[0, i % d] = 1.0
    return v


# ---------------------------------------------------------------------------
# chunk_transcript
# ---------------------------------------------------------------------------

class TestChunkTranscript:
    def test_empty_events_give_no_chunks(self):
        assert chunk_transcript([], started_at=0.0) == []

    def test_single_short_line_is_one_chunk(self):
        chunks = chunk_transcript(
            _events(("สมชาย", 5.0, "สวัสดีครับ")), started_at=0.0
        )
        assert len(chunks) == 1
        assert chunks[0].chunk_id == 0
        assert chunks[0].start_sec == 5.0
        assert "[00:05]" in chunks[0].text
        assert "สมชาย: สวัสดีครับ" in chunks[0].text

    def test_timestamps_are_relative_to_started_at(self):
        chunks = chunk_transcript(
            _events(("a", 65.0, "hello")), started_at=60.0
        )
        assert "[01:05]" not in chunks[0].text  # absolute time must not appear
        assert "[00:05]" in chunks[0].text

    def test_lines_never_split(self):
        lines = [f"[00:{i:02d}] sp{i}: {'x' * 50}" for i in range(10)]
        events = [
            TranscriptEvent(speaker=f"sp{i}", start=float(i), text="x" * 50)
            for i in range(10)
        ]
        chunks = chunk_transcript(events, 0.0, chunk_chars=120, overlap_chars=0)
        for c in chunks:
            for line in c.text.split("\n"):
                assert line in lines  # every rendered line is complete

    def test_window_respects_chunk_chars(self):
        events = [
            TranscriptEvent(speaker="s", start=float(i), text="w" * 40)
            for i in range(20)
        ]
        chunks = chunk_transcript(events, 0.0, chunk_chars=100, overlap_chars=0)
        assert len(chunks) > 1
        for c in chunks:
            # a single line is 40+ chars; window may exceed only via one line
            assert len(c.text) <= 100 + 45

    def test_overlap_repeats_trailing_lines(self):
        events = [
            TranscriptEvent(speaker="s", start=float(i), text=f"line{i} " + "y" * 30)
            for i in range(6)
        ]
        chunks = chunk_transcript(events, 0.0, chunk_chars=90, overlap_chars=200)
        assert len(chunks) >= 2
        prev_lines = chunks[0].text.split("\n")
        first_new_line = next(
            l for l in chunks[1].text.split("\n") if l not in prev_lines
        )
        idx_prev = {l for l in prev_lines}
        overlap_part = chunks[1].text.split(first_new_line)[0]
        overlap_lines = [l for l in overlap_part.strip().split("\n") if l]
        assert overlap_lines, "overlap must repeat at least one trailing line"
        for l in overlap_lines:
            assert l in idx_prev

    def test_zero_overlap_repeats_nothing(self):
        events = [
            TranscriptEvent(speaker="s", start=float(i), text="z" * 40)
            for i in range(8)
        ]
        chunks = chunk_transcript(events, 0.0, chunk_chars=80, overlap_chars=0)
        seen = set()
        for c in chunks:
            for line in c.text.split("\n"):
                assert line not in seen  # no repeats anywhere
                seen.add(line)

    def test_chronological_sort_of_unordered_events(self):
        events = [
            TranscriptEvent(speaker="b", start=20.0, text="later"),
            TranscriptEvent(speaker="a", start=5.0, text="early"),
        ]
        chunks = chunk_transcript(events, 0.0)
        assert chunks[0].start_sec == 5.0
        assert "early" in chunks[0].text

    def test_header_covers_span(self):
        events = [
            TranscriptEvent(speaker="s", start=10.0, text="one"),
            TranscriptEvent(speaker="s", start=130.0, text="two"),
        ]
        chunks = chunk_transcript(events, 0.0)
        assert chunks[0].header.startswith("[00:10")
        assert chunks[0].header.endswith("02:10]")

    def test_oversized_single_line_becomes_own_chunk(self):
        big = TranscriptEvent(speaker="s", start=0.0, text="L" * 5000)
        after = TranscriptEvent(speaker="s", start=1.0, text="after")
        chunks = chunk_transcript([big, after], 0.0, chunk_chars=800, overlap_chars=150)
        assert any(len(c.text) > 4000 for c in chunks)  # unsplit
        assert any("after" in c.text for c in chunks)


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------

class TestVectorIndexBasics:
    def test_empty_index_queries_return_empty(self):
        idx = VectorIndex()
        assert len(idx) == 0
        assert idx.query(_vec(0), 3) == []
        assert idx.query_multi(np.zeros((2, 4)), 3) == []

    def test_add_then_len(self):
        idx = VectorIndex()
        idx.add(np.vstack([_vec(0), _vec(1)]), [
            Chunk(0, "[00:00]", "a", 0.0),
            Chunk(1, "[00:01]", "b", 1.0),
        ])
        assert len(idx) == 2

    def test_add_mismatched_lengths_raise(self):
        idx = VectorIndex()
        with pytest.raises(ValueError):
            idx.add(np.vstack([_vec(0), _vec(1)]), [Chunk(0, "", "a", 0.0)])

    def test_dim_mismatch_raises(self):
        idx = VectorIndex()
        idx.add(_vec(0, 4), [Chunk(0, "", "a", 0.0)])
        with pytest.raises(ValueError):
            idx.add(_vec(0, 8), [Chunk(1, "", "b", 1.0)])

    def test_query_topk_ordering_by_cosine(self):
        d = np.vstack([_vec(0), _vec(1), _vec(2)])
        chunks = [Chunk(i, "", f"c{i}", float(i)) for i in range(3)]
        idx = VectorIndex()
        idx.add(d, chunks)
        q = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
        hits = idx.query(q, 3)
        assert [c.chunk_id for c, _ in hits] == [0, 1, 2]
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)
        assert hits[0][1] > 0.99

    def test_query_k_limits_results(self):
        d = np.vstack([_vec(i) for i in range(5)])
        chunks = [Chunk(i, "", f"c{i}", float(i)) for i in range(5)]
        idx = VectorIndex()
        idx.add(d, chunks)
        assert len(idx.query(_vec(0), 2)) == 2

    def test_vectors_normalized_on_add(self):
        idx = VectorIndex()
        idx.add(np.array([[3.0, 4.0]], dtype=np.float32), [Chunk(0, "", "a", 0.0)])
        hits = idx.query(np.array([[6.0, 8.0]], dtype=np.float32), 1)
        # both normalized -> cosine similarity exactly 1.0
        assert abs(hits[0][1] - 1.0) < 1e-5

    def test_zero_vector_does_not_crash(self):
        idx = VectorIndex()
        idx.add(np.array([[0.0, 0.0]]), [Chunk(0, "", "a", 0.0)])
        assert idx.query(np.array([[1.0, 0.0]]), 1)[0][1] == pytest.approx(0.0)


class TestQueryMulti:
    def _index(self):
        # chunk i points along axis i; queries select disjoint sets
        d = np.vstack([_vec(i) for i in range(6)])
        chunks = [Chunk(i, "", f"c{i}", float(i)) for i in range(6)]
        idx = VectorIndex()
        idx.add(d, chunks)
        return idx

    def test_union_of_topk_per_query(self):
        idx = self._index()
        qs = np.vstack([_vec(0), _vec(1)])
        hits = idx.query_multi(qs, 1)
        ids = [c.chunk_id for c, _ in hits]
        assert set(ids) == {0, 1}

    def test_dedupe_keeps_max_score(self):
        idx = self._index()
        # both queries hit chunk 0; its best score wins, single entry kept
        qs = np.array([[1.0, 0.0, 0.0, 0.0], [0.7, 0.7, 0.0, 0.0]], dtype=np.float32)
        hits = idx.query_multi(qs, 2)
        zero_hits = [(c, s) for c, s in hits if c.chunk_id == 0]
        assert len(zero_hits) == 1

    def test_chronological_resort(self):
        idx = self._index()
        qs = np.vstack([_vec(5), _vec(0)])  # query order: late chunk first
        hits = idx.query_multi(qs, 1)
        ids = [c.chunk_id for c, _ in hits]
        assert ids == sorted(ids)  # re-sorted by start_sec

    def test_k_larger_than_index(self):
        idx = self._index()
        hits = idx.query_multi(np.vstack([_vec(0)]), 100)
        assert len(hits) == 6


# ---------------------------------------------------------------------------
# truncate_block (mirrors Transcript.to_prompt_text semantics)
# ---------------------------------------------------------------------------

class TestTruncateBlock:
    def test_short_text_untouched(self):
        assert truncate_block("hello\nworld", 100) == "hello\nworld"

    def test_truncates_on_line_boundary(self):
        text = "\n".join(f"line{i} " + "x" * 10 for i in range(20))
        out = truncate_block(text, 80)
        assert out.endswith("...(truncated)")
        body = out[: -len("...(truncated)")].rstrip("\n")
        for line in body.split("\n"):
            assert line in text.split("\n") or line == ""

    def test_single_long_line_sliced_with_suffix(self):
        out = truncate_block("A" * 500, 50)
        assert out.endswith("...(truncated)")
        assert len(out) == 50
        assert out.startswith("A" * (50 - len("...(truncated)")))

    def test_matches_transcript_semantics(self):
        from meeting_bot.transcript import Transcript

        t = Transcript(started_at=0.0)
        for i in range(30):
            t.add(TranscriptEvent(speaker=f"s{i}", start=float(i), text="word " * 8))
        legacy = t.to_prompt_text(max_chars=300)
        assert truncate_block(t.to_prompt_text(max_chars=None), 300) == legacy
