"""Pure-logic tests for meeting_bot.transcriber garbage detection + the
anti-loop decode settings (no mlx-whisper / Discord / network).
"""

from meeting_bot.transcriber import (
    build_decode_kwargs,
    is_garbage_transcription,
    primary_decode_settings,
    retry_decode_settings,
    should_retry,
)


def _clear_env(monkeypatch) -> None:
    """Make decode-settings tests deterministic regardless of shell env."""
    monkeypatch.delenv("WHISPER_FP16", raising=False)
    monkeypatch.delenv("WHISPER_RETRY_TEMPERATURE", raising=False)
    monkeypatch.delenv("WHISPER_INITIAL_PROMPT", raising=False)


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
        # 50% each, so the single-token ratio heuristic (2) would NOT fire.
        # But each word repeats consecutively many times in a row, which is
        # not normal speech — the generalized repeating-run check (heuristic
        # 1, period covers "hello"/"world" themselves) catches it.
        text = " ".join(["hello"] * 10 + ["world"] * 10)
        assert is_garbage_transcription(text)

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

    def test_thai_no_whitespace_repetition_detected(self) -> None:
        # Real whisper hallucination from a meeting log: the syllable "ตาม"
        # repeated ~111 times with no whitespace (Thai has no spaces between
        # words). Previously slipped past both heuristics — single-char run
        # doesn't match (ต ≠ า ≠ ม) and the whole string is one whitespace
        # token, below _MIN_TOKENS. Generalized run check catches it.
        text = "ตาม" * 111
        assert is_garbage_transcription(text)


class TestThaiParticlesPass:
    """Legitimate Thai politeness particles must never be flagged garbage.

    These are the exact repeated-particle patterns that could otherwise be
    mistaken for whisper repetition loops now that the retry path keys off
    :func:`should_retry` (a false positive costs a wasted re-decode).
    """

    def test_krub_spaced(self) -> None:
        assert not is_garbage_transcription("ครับ ครับ ครับ ครับ")

    def test_ka_spaced(self) -> None:
        assert not is_garbage_transcription("ค่ะ ค่ะ ค่ะ")

    def test_percent_krub(self) -> None:
        assert not is_garbage_transcription("100% ครับ")

    def test_krub_run_no_whitespace(self) -> None:
        # "ครับครับครับ" — the 12-char run (period 4 × 3) is well under
        # _MAX_CHAR_RUN (30) and the single whitespace token is below
        # _MIN_TOKENS, so neither heuristic fires.
        assert not is_garbage_transcription("ครับครับครับ")


class TestShouldRetry:
    """The retry trigger: garbage OR whisper's no-speech flag, never empty."""

    def test_garbage_retries(self) -> None:
        assert should_retry("ตาม" * 111, 0.0)

    def test_clean_high_no_speech_retries(self) -> None:
        assert should_retry("สวัสดีครับ วันนี้เรามีประชุม", 0.9)

    def test_clean_low_no_speech_no_retry(self) -> None:
        assert not should_retry("สวัสดีครับ วันนี้เรามีประชุม", 0.1)

    def test_empty_text_never_retries(self) -> None:
        # High no_speech alone must not resurrect a genuinely empty window.
        assert not should_retry("", 0.9)
        assert not should_retry("   ", 0.9)

    def test_boundary_no_speech_threshold_exclusive(self) -> None:
        # Strictly > threshold; 0.6 exactly is not a retry.
        assert not should_retry("สวัสดีครับ", 0.6)


class TestDecodeSettings:
    """build_decode_kwargs reflects primary vs retry behavior."""

    def test_primary_is_greedy_and_unbiased(self, monkeypatch) -> None:
        _clear_env(monkeypatch)
        kwargs = build_decode_kwargs(primary_decode_settings(), language="th")
        assert kwargs["language"] == "th"
        assert kwargs["temperature"] == 0.0
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["no_speech_threshold"] == 0.6
        assert kwargs["fp16"] is True
        assert "initial_prompt" not in kwargs

    def test_retry_bumps_temperature_and_adds_preamble(self, monkeypatch) -> None:
        _clear_env(monkeypatch)
        kwargs = build_decode_kwargs(retry_decode_settings(), language="th")
        assert kwargs["temperature"] > 0.0
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["fp16"] is True  # must match primary (model cached by path)
        assert kwargs["initial_prompt"].strip()  # Thai preamble present

    def test_fp16_env_toggle(self, monkeypatch) -> None:
        monkeypatch.setenv("WHISPER_FP16", "0")
        assert primary_decode_settings().fp16 is False

    def test_retry_temperature_env_toggle(self, monkeypatch) -> None:
        monkeypatch.setenv("WHISPER_RETRY_TEMPERATURE", "0.5")
        assert retry_decode_settings().temperature == 0.5

    def test_empty_initial_prompt_env_disables_preamble(self, monkeypatch) -> None:
        monkeypatch.setenv("WHISPER_INITIAL_PROMPT", "")
        assert retry_decode_settings().initial_prompt is None
