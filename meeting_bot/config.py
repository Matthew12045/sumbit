"""Configuration loading and ``--doctor`` environment checks.

This module must stay importable with only stdlib (+ numpy) present, so
python-dotenv is imported lazily inside :func:`load_config` rather than at
module scope.
"""

from __future__ import annotations

import ctypes.util
import importlib
import os
from dataclasses import dataclass

__all__ = ["Config", "load_config", "doctor"]

# Required env vars and the Config field they map to.  ``doctor`` uses this so
# the "all required keys present" check never has to guess.
_REQUIRED_ENV = (
    ("DISCORD_TOKEN", "discord_token"),
    ("GUILD_ID", "guild_id"),
    ("VOICE_CHANNEL_ID", "voice_channel_id"),
    ("TARGET_CHANNEL_ID", "target_channel_id"),
    ("ANTHROPIC_BASE_URL", "anthropic_base_url"),
    ("ANTHROPIC_AUTH_TOKEN", "anthropic_auth_token"),
    ("GATEWAY_MODEL", "gateway_model"),
    ("WHISPER_MODEL", "whisper_model"),
    ("WHISPER_LANGUAGE", "whisper_language"),
)

_OPTIONAL_FLOAT_ENV = (
    ("SILENCE_THRESHOLD", "silence_threshold", 0.01),
    ("SILENCE_SECONDS", "silence_seconds", 0.5),
    ("MIN_CHUNK_SECONDS", "min_chunk_seconds", 0.15),
    ("MAX_CHUNK_SECONDS", "max_chunk_seconds", 30.0),
    ("SUMMARIZE_TIMEOUT_SECONDS", "summarize_timeout_seconds", 300.0),
    ("MAX_PROMPT_CHARS", "max_prompt_chars", 6000.0),
    ("SUMMARY_MAX_TOKENS", "summary_max_tokens", 4096.0),
)


@dataclass(frozen=True)
class Config:
    discord_token: str
    guild_id: int
    voice_channel_id: int
    target_channel_id: int
    anthropic_base_url: str      # no trailing /v1 (the SDK appends /v1/messages)
    anthropic_auth_token: str
    gateway_model: str           # qwen3.6-35b-a3b
    whisper_model: str           # mlx-community/whisper-large-v3-turbo
    whisper_language: str        # "th"
    silence_threshold: float = 0.01     # RMS speech threshold (~ -40 dBFS)
    silence_seconds: float = 0.5        # trailing silence to close a chunk
    min_chunk_seconds: float = 0.15     # shorter closed chunks are dropped
    max_chunk_seconds: float = 30.0     # force-close cap
    summarize_timeout_seconds: float = 300.0  # SDK client timeout for gateway
    max_prompt_chars: int = 6000        # transcript truncation limit
    summary_max_tokens: int = 4096      # gateway output-token budget (qwen thinking)


def load_config(path: str | os.PathLike = ".env") -> Config:
    """Read ``.env`` (via python-dotenv) plus process env, build a Config.

    Raises a clear ``ValueError`` naming the first missing required key.
    """
    from dotenv import load_dotenv  # lazy: python-dotenv is not stdlib/numpy

    path = os.fspath(path)
    # `.env` must win over the process environment.  load_dotenv defaults to
    # override=False, so a shell-exported ANTHROPIC_BASE_URL (e.g. a Claude
    # Code alias pointing at api.deepseek.com) would silently shadow this
    # repo's gateway.9arm.co value and every --doctor probe would 401.
    load_dotenv(path, override=True)

    def _required(name: str) -> str:
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(
                f"Missing required environment variable: {name} "
                f"(expected in {path!r})"
            )
        return value

    def _optional_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return default
        return float(raw)

    return Config(
        discord_token=_required("DISCORD_TOKEN"),
        guild_id=int(_required("GUILD_ID")),
        voice_channel_id=int(_required("VOICE_CHANNEL_ID")),
        target_channel_id=int(_required("TARGET_CHANNEL_ID")),
        anthropic_base_url=_required("ANTHROPIC_BASE_URL"),
        anthropic_auth_token=_required("ANTHROPIC_AUTH_TOKEN"),
        gateway_model=_required("GATEWAY_MODEL"),
        whisper_model=_required("WHISPER_MODEL"),
        whisper_language=_required("WHISPER_LANGUAGE"),
        silence_threshold=_optional_float("SILENCE_THRESHOLD", 0.01),
        silence_seconds=_optional_float("SILENCE_SECONDS", 0.5),
        min_chunk_seconds=_optional_float("MIN_CHUNK_SECONDS", 0.15),
        max_chunk_seconds=_optional_float("MAX_CHUNK_SECONDS", 30.0),
        summarize_timeout_seconds=_optional_float("SUMMARIZE_TIMEOUT_SECONDS", 300.0),
        max_prompt_chars=int(_optional_float("MAX_PROMPT_CHARS", 6000.0)),
        summary_max_tokens=int(_optional_float("SUMMARY_MAX_TOKENS", 4096.0)),
    )


def doctor(cfg: Config) -> list[str]:
    """Environment checks. One ``ok: ...``/``fail: ...`` line per check.

    Never raises and never logs the auth token.  ``--doctor`` exits non-zero
    when any line starts with ``fail:``.
    """
    lines: list[str] = []

    # 1. All required config keys present and non-empty.
    missing = [
        field_name
        for env_name, field_name in _REQUIRED_ENV
        if not getattr(cfg, field_name)
    ]
    if missing:
        lines.append("fail: missing config keys: " + ", ".join(missing))
    else:
        lines.append("ok: all required config keys present")

    # 2. Optional heavy dependencies, each checked individually.
    for mod_name in ("numpy", "discord", "mlx_whisper", "anthropic"):
        try:
            importlib.import_module(mod_name)
            lines.append(f"ok: {mod_name} importable")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"fail: {mod_name} not importable ({type(exc).__name__})")

    # 3. System libopus (py-cord ships libopus only as Windows DLLs).
    try:
        opus = ctypes.util.find_library("opus")
        if opus:
            lines.append(f"ok: libopus found ({opus})")
        else:
            lines.append("fail: libopus not found (brew install opus)")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"fail: libopus lookup errored ({type(exc).__name__})")

    # 4. Whisper model name non-empty.
    if cfg.whisper_model:
        lines.append("ok: whisper model name set")
    else:
        lines.append("fail: whisper model name empty")

    # 5. Gateway probe (Anthropic-compatible API, ~10 s timeout).
    lines.append(_gateway_probe_line(cfg))

    return lines


def _gateway_probe_line(cfg: Config) -> str:
    """Probe the gateway; never raises. The token is never logged."""
    if not cfg.anthropic_base_url or not cfg.anthropic_auth_token:
        return "fail: gateway probe skipped (missing base URL or auth token)"
    try:
        import anthropic
    except Exception as exc:  # noqa: BLE001
        return f"fail: anthropic not importable ({type(exc).__name__})"
    try:
        client = anthropic.Anthropic(
            base_url=cfg.anthropic_base_url,
            auth_token=cfg.anthropic_auth_token,
            timeout=10.0,
        )
        response = client.messages.create(
            model=cfg.gateway_model,
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
    except Exception as exc:  # noqa: BLE001
        status_code = getattr(exc, "status_code", None)
        if status_code in (401, 403):
            return "fail: gateway auth rejected (bad token)"
        return f"fail: gateway probe failed ({type(exc).__name__})"
    # The SDK's Message response exposes the httpx status; default to 2xx if
    # it isn't reachable (older SDKs).
    try:
        status_code = response._http_response.status_code
    except Exception:  # noqa: BLE001
        status_code = None
    if status_code is None or 200 <= status_code < 300:
        return "ok: gateway probe succeeded"
    return f"fail: gateway probe HTTP {status_code}"
