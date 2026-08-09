"""Pure-logic tests for meeting_bot.transcriber garbage detection
(no mlx-whisper / Discord / network).
"""

from meeting_bot.transcriber import is_garbage_transcription


class TestIsGarbageTranscription:
    # -- normal text (not garbage) ---------------------------------------

    def test_normal_thai_text_passes(self) -> None:
        assert not is_garbage_transcription(
            "สวัสดีครับ วันนี้เรามีประชุมเรื่องงบประมาณประจำปี"
        )

    def test_empty_string_passes(self) -> None:
        assert not is_garbage_transcription("")

    def test_whitespace_only_passes(self) -> None:
        assert not is_garbage_transcription("   ")

    def test_single_short_word_passes(self) -> None:
        assert not is_garbage_transcription("OK")

    def test_normal_english_passes(self) -> None:
        assert not is_garbage_transcription(
            "The quick brown fox jumps over the lazy dog several times today"
        )

    # -- heuristic 1: max character run > _MAX_CHAR_RUN (30) -------------

    def test_long_z_run_detected(self) -> None:
        # "SeZC" + 31 consecutive 'Z's  →  31-char run  →  garbage
        text = "SeZC" + "Z" * 31
        assert is_garbage_transcription(text)

    def test_exactly_30_char_run_passes(self) -> None:
        # 30 consecutive 'A's  →  at threshold, not over  →  ok
        text = "A" * 30 + " normal text"
        assert not is_garbage_transcription(text)

    def test_31_char_run_detected(self) -> None:
        text = "prefix" + "Z" * 31
        assert is_garbage_transcription(text)

    def test_run_in_middle_of_text_detected(self) -> None:
        text = "B" + "A" * 31 + "B"
        assert is_garbage_transcription(text)

    # -- heuristic 2: token repetition ratio > _MAX_TOKEN_RATIO (0.7) ----

    def test_repeated_se_token_detected(self) -> None:
        # "แล้ว Se Se Se Se Se..." — Se dominates >70% with >10 tokens
        text = "แล้ว " + "Se " * 20
        assert is_garbage_transcription(text)

    def test_balanced_tokens_passes(self) -> None:
        # 50% each  →  under 70%  →  ok
        text = " ".join(["hello"] * 10 + ["world"] * 10)
        assert not is_garbage_transcription(text)

    def test_high_repetition_but_few_tokens_passes(self) -> None:
        # Only 5 tokens total — below _MIN_TOKENS (10)
        text = " ".join(["yes"] * 4 + ["no"])
        assert not is_garbage_transcription(text)

    def test_eleven_tokens_72_percent_detected(self) -> None:
        # 8 "a" + 3 "b" = 11 tokens, "a" = 8/11 = 72.7% > 70%
        text = " ".join(["a"] * 8 + ["b"] * 3)
        assert is_garbage_transcription(text)

    def test_exactly_ten_tokens_70_percent_passes(self) -> None:
        # 7 "a" + 3 "b" = 10 tokens, "a" = 7/10 = 70%  →  NOT > 70%
        text = " ".join(["a"] * 7 + ["b"] * 3)
        assert not is_garbage_transcription(text)

    # -- edge cases ------------------------------------------------------

    def test_thai_with_some_repetition_passes(self) -> None:
        # Normal Thai conversation with some word repetition
        text = "ครับ ครับ ครับ เราเห็นด้วย เราเห็นด้วย งั้นเริ่มเลยนะครับ"
        assert not is_garbage_transcription(text)
