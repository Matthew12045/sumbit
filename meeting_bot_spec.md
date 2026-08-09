# Meeting Summary Discord Bot — Build Spec

You are implementing a complete, runnable Discord bot package from this spec. Read carefully; every module, signature, and constraint below is mandatory. Where this spec and your own judgment conflict, the spec wins.

## Overview

A Discord bot that joins a voice channel, transcribes the meeting live in **Thai**, and when the **last human** leaves the voice channel, automatically posts a structured summary — **Topics / Decisions / Action Items** — to a configured text channel. No manual command triggers it.

Data flow:

```
Discord voice channel
  → py-cord per-user voice receive (DAVE-patched branch) → per-user PCM
  → resample 48 kHz stereo int16 → 16 kHz mono float32 (numpy, no ffmpeg)
  → silence-based chunking per speaker (worker thread safety)
  → mlx-whisper (local, Metal) transcribes each closed chunk in Thai
  → accumulate labeled transcript
  → on voice channel empty: local gateway (Anthropic-compatible) summarizes
  → post structured embed to target text channel
```

## Locked decisions (do not change)

1. **Voice receive:** py-cord pinned to the DAVE-patched unmerged branch — released 2.8.1's voice reception is buggy under Discord's DAVE encryption (py-cord issue #3139); the fix reworks per-user receive. `requirements.txt` must pin the **exact commit** `git+https://github.com/Pycord-Development/pycord.git@326b72acc8d1d952ac002fe07ca65581cf5952bc` (branch `fix/voice-rec-2`; a moving-branch pin is not reproducible). Do not substitute released 2.8.1 — receive will be broken.

   **macOS prerequisite:** `brew install opus` is required for voice (py-cord ships libopus only as Windows DLLs). The bot's `doctor()` checks `ctypes.util.find_library("opus")`. First whisper run downloads ~1.6 GB into `~/.cache/huggingface` (host default is fine; set `HF_HOME` only if the cache must live elsewhere).
2. **STT:** local `mlx-whisper`, model `mlx-community/whisper-large-v3-turbo`, `language="th"`. No cloud API. Runs on the same Apple Silicon MacBook as the AI gateway, so GPU contention is a real concern — serialized worker, fp16, model cached once.
3. **Summarization:** the user's own gateway via the **Anthropic-compatible** API — `anthropic` SDK, `base_url=https://gateway.9arm.co`, `auth_token` from env (`ANTHROPIC_AUTH_TOKEN`), model `qwen3.6-35b-a3b`. **Do not put `/v1` in the base URL** (the SDK appends `/v1/messages`). qwen's structured-output reliability is the known risk — the summary parse must never raise.
4. **Secrets/config:** `.env` (never committed) via python-dotenv. A `.env.example` mirrors every key with placeholders and comments, no real secrets.

## File tree — produce exactly this

```
meeting_bot/
  __init__.py          # __version__ = "0.1.0"; re-export Config, MeetingBot
  config.py            # Config dataclass, load_config(), doctor()
  audio.py             # resample + RMS helpers (numpy only)
  chunker.py           # SilenceChunker, Segment (numpy/stdlib only)
  sink.py              # MeetingSink(discord.sinks.Sink) per-user PCM capture
  transcriber.py       # Transcriber: background mlx-whisper worker
  transcript.py        # TranscriptEvent, Transcript accumulator (stdlib only)
  summary_parse.py     # Summary, ActionItem, parse_summary() (stdlib only)
  summarizer.py        # Summarizer: anthropic gateway call (lazy anthropic import)
  poster.py            # build_embed(), Poster
  bot.py               # MeetingBot(discord.Client): voice trigger + orchestration
  main.py              # argparse entrypoint: run | --doctor
  __main__.py          # raise SystemExit(main())
tests/
  test_audio.py
  test_chunker.py
  test_transcript.py
  test_summary_parse.py
requirements.txt
.env.example
```

**Import rule (load-bearing).** `config.py`, `audio.py`, `chunker.py`, `transcript.py`, `summary_parse.py` must import **only stdlib + numpy** at module scope. `summarizer.py` imports `anthropic` lazily (inside `__init__`/methods, not at module scope). `sink.py`, `bot.py`, `poster.py` may import `discord`. `config.py` imports `dotenv`/`load_dotenv` **inside `load_config()` only** — python-dotenv is neither stdlib nor numpy, so a module-scope import breaks the pure-import rule. This lets the pure modules be imported and tested without the heavy deps installed.

## Module contracts

### `config.py`
```python
@dataclass(frozen=True)
class Config:
    discord_token: str
    guild_id: int
    voice_channel_id: int
    target_channel_id: int
    anthropic_base_url: str      # no trailing /v1
    anthropic_auth_token: str
    gateway_model: str           # qwen3.6-35b-a3b
    whisper_model: str           # mlx-community/whisper-large-v3-turbo
    whisper_language: str        # "th"
    silence_threshold: float = 0.01     # RMS speech threshold (−40 dBFS)
    silence_seconds: float = 0.8        # trailing silence to close a chunk
    min_chunk_seconds: float = 1.0      # shorter closed chunks are dropped
    max_chunk_seconds: float = 30.0     # force-close cap

def load_config(path: str | os.PathLike = ".env") -> Config: ...
def doctor(cfg: Config) -> list[str]:   # list of "ok: ..."/"fail: ..." lines
```
`load_config` reads `.env` with `python-dotenv` (imported inside the function), validates required keys present and non-empty, raises a clear error naming the missing key. `doctor` returns one `ok: ...`/`fail: ...` line per check and **never raises**: all required keys present; `discord`/`mlx_whisper`/`anthropic`/`numpy` importable (each individually — a missing one is a `fail` line, not an exception); system `libopus` resolvable via `ctypes.util.find_library("opus")`; whisper model name non-empty; and a **gateway probe**: construct the anthropic client from cfg and call `messages.create(model=cfg.gateway_model, max_tokens=1)` with a ~10 s timeout — `ok` iff no exception and HTTP 2xx, `fail` on 401/403 (bad token) or network error, `fail` (not an exception) if `anthropic` isn't installed. Never log the auth token.

### `audio.py` (pure numpy)
```python
def resample_48k_stereo_to_16k_mono(pcm: bytes) -> np.ndarray:
    """int16 PCM (48 kHz, 2ch, 20 ms/frame = 3840 bytes) -> float32 mono 16 kHz."""
def is_speech_block(samples: np.ndarray, threshold: float) -> bool:
    """RMS of the block >= threshold => speech."""
```
Resample by decimation-by-3 with an anti-aliasing low-pass (windowed-sinc FIR, ~63 taps, cutoff 8 kHz) via `np.convolve`, then `x[::3]`. Divide int16 by 32768.0, average the two channels. Do not use a naive boxcar decimate (aliases Thai fricatives/tones). A `librosa.resample` fallback is acceptable but the numpy path is primary and must be deterministic and testable.

### `chunker.py` (pure numpy)
```python
@dataclass
class Segment:
    speaker_key: str
    speaker_name: str
    start: float        # seconds since meeting start (monotonic clock)
    samples: np.ndarray # 16 kHz mono float32
    duration: float

class SilenceChunker:
    def __init__(self, *, sample_rate=16000, frame_ms=20,
                 threshold=0.01, silence_seconds=0.8,
                 min_chunk_seconds=1.0, max_chunk_seconds=30.0): ...
    def feed(self, samples: np.ndarray, now: float) -> list[Segment]: ...
    def flush(self, now: float) -> list[Segment]: ...
    def reset(self) -> None: ...
    @property
    def open_duration(self) -> float: ...
```
- Compute RMS per 20 ms block (320 samples @16 kHz). Speech when `rms >= threshold`.
- After `silence_seconds` of trailing silence, close the open chunk; drop it if shorter than `min_chunk_seconds`.
- Force-close at `max_chunk_seconds` even mid-speech.
- `flush` closes the trailing partial chunk if `>= min_chunk_seconds`, else drops it.

### `sink.py` (py-cord)
```python
class MeetingSink(discord.sinks.Sink):
    def __init__(self, transcriber, chunker_factory, names, *, threshold=..., **chunker_kw): ...
    def write(self, data, user) -> None: ...
    def flush_user(self, user_id: int) -> None: ...   # flush that user's chunker (NOT a py-cord hook)
```
- `write()` runs on the py-cord **router thread** — must **never call asyncio**. Only: extract PCM via `getattr(data, "pcm", data)` (VoiceData in 2.7+ vs raw bytes), resample to 16 kHz mono f32, feed the per-user `SilenceChunker`, and call `transcriber.submit(segment)` for each closed `Segment`.
- Per-user state keyed by `user.id`; resolve `speaker_name` from `user.display_name` (fall back to `str(user.id)` if unresolved).
- `flush_user(user_id)` flushes that user's open chunker. It is **not** a py-cord hook — the SinkEventRouter only dispatches `rtcp_packet`/`member_speaking_start`/`member_speaking_stop` — so `bot.py` must call it from `on_voice_state_update` when a human leaves the channel but the meeting continues. Guard it with a lock (sink state is touched from the router thread).

### `transcriber.py`
```python
class Transcriber:
    def __init__(self, model: str, language: str): ...
    def start(self) -> None: ...       # spawn ONE daemon worker thread
    def submit(self, segment: Segment) -> None: ...   # thread-safe input queue
    def events(self) -> queue.Queue: ...   # output queue of TranscriptionEvent
    def stop(self, flush: bool = True) -> None: ...
    def drain(self, timeout: float = 5.0) -> list[TranscriptionEvent]: ...
```
One worker thread pulls segments, calls `mlx_whisper.transcribe(samples_array, path_or_hf_repo=model, language=language)["text"]`, and pushes `TranscriptionEvent(speaker, start, text)` onto the output queue. Passing a numpy array skips ffmpeg (input is 16 kHz mono f32). Load the model once and reuse (the MLX `ModelHolder` caches it). Serializes MLX so the GPU isn't contended with the gateway. If `mlx_whisper` import fails at construction, raise a clear error. **Right before transcribing, assert the array is exactly 16 kHz mono float32** (e.g. `assert samples.dtype == np.float32 and samples.ndim == 1` and the rate is 16000) — mlx-whisper does **not** validate or resample array input; wrong rate/dtype silently yields garbage, so a pipeline regression must fail loudly instead. `stop(flush=True)` drains the input queue to completion before the worker exits (so a flushed trailing chunk is transcribed, not dropped); `drain(timeout)` waits up to `timeout` for pending transcriptions to land on the output queue and returns them. `_finalize` calls `stop(flush=True)` (or `drain`) before building the transcript.

### `transcript.py` (pure stdlib)
```python
@dataclass
class TranscriptEvent:
    speaker: str
    start: float
    text: str

class Transcript:
    def __init__(self, started_at: float): ...
    def add(self, event: TranscriptEvent) -> None: ...
    def events(self) -> list[TranscriptEvent]: ...
    def to_prompt_text(self) -> str: ...   # "[MM:SS] ผู้พูด: ..." per line, chronological
    def is_empty(self) -> bool: ...
```

### `summary_parse.py` (pure stdlib — the reliability mitigation)
```python
@dataclass
class ActionItem:
    action: str
    owner: str | None = None
    @classmethod
    def parse(cls, obj) -> "ActionItem": ...   # dict or plain string

@dataclass
class Summary:
    topics: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    raw: str

def parse_summary(text: str) -> Summary: ...   # never raises
```
Parse order:
1. **JSON attempt:** strip ```json``` fences and prose before `{`/after `}`; `json.loads`; tolerate Thai or English keys (`topics`/`หัวข้อ`, `decisions`/`การตัดสินใจ`, `action_items`/`รายการที่ต้องทำ`); coerce each action item via `ActionItem.parse` (dict or plain string).
2. **Markdown fallback:** split on section headers matching `หัวข้อ|Topics`, `การตัดสินใจ|Decisions`, `สิ่งที่ต้องทำ|Action Items`; collect `-`/`*` bullets.
3. **Last resort:** `Summary(topics=[raw.strip()], ...)`. Never raise.

### `summarizer.py` (anthropic, lazy import)
```python
class Summarizer:
    def __init__(self, cfg: Config): ...
    def summarize(self, transcript_text: str) -> str: ...   # blocking SDK call
```
```python
client = anthropic.Anthropic(
    base_url=cfg.anthropic_base_url,     # e.g. https://gateway.9arm.co  (no /v1)
    auth_token=cfg.anthropic_auth_token, # -> Authorization: Bearer ...
    timeout=120.0,
)
resp = client.messages.create(
    model=cfg.gateway_model,             # qwen3.6-35b-a3b
    max_tokens=2000,
    temperature=0.0,
    system=<Thai JSON-contract system prompt>,
    messages=[{"role": "user", "content": transcript_text}],
)
text = "".join(b.text for b in resp.content if b.type == "text")
```
System prompt (Thai): instruct the model to reply only in Thai and output only the JSON schema below, no markdown fence, no other text. The user message additionally instructs empty sections must be `[]`.

```json
{ "topics": ["..."], "decisions": ["..."], "action_items": [{"action": "...", "owner": "..."}] }
```
The bot drains the summary text through `summary_parse.parse_summary`. Call `summarize` via `asyncio.to_thread` from `bot.py` so the event loop isn't blocked.

### `poster.py`
```python
def build_embed(summary, *, started_at: datetime, duration: timedelta,
                member_count: int, meeting_title: str | None = None) -> discord.Embed: ...
class Poster:
    def __init__(self, config): ...
    async def post(self, channel, summary, *, meta) -> discord.Message: ...
```
Embed titled `📝 สรุปการประชุม`, one field per section (`หัวข้อที่พูดคุย`, `การตัดสินใจ`, `รายการที่ต้องทำ`, `•` bullets), footer with guild/voice-channel name, date, duration, member count. `meeting_title` is optional — when `None`, default to the guild/voice-channel name. Send to `target_channel_id` via `bot.get_channel(...)`. Retry up to 3× on 429/5xx.

### `bot.py`
```python
class MeetingBot(discord.Client):
    def __init__(self, cfg: Config, *, intents=...): ...
    async def on_ready(self): ...
    async def on_voice_state_update(self, member, before, after): ...
    async def _start_meeting(self, channel): ...
    async def _finalize(self, channel): ...
```
- Intents: `guilds`, `voice_states`, and `members` (members is a **privileged** intent — note in `.env.example`/README that it must be enabled in the Developer Portal).
- Join self-muted/deaf (correct py-cord sequence): `vc = await channel.connect()` (VocalGuildChannel.connect accepts only `timeout`/`reconnect`/`cls`); then `await guild.change_voice_state(channel=channel, self_mute=True, self_deaf=True)` (returns `None` — it does **not** create the VoiceClient); then `vc.start_recording(sink)`.
- **Trigger:** in `on_voice_state_update`, count **humans** — `[m for m in channel.members if not m.bot]` — never gate on `member == self.user and before.channel.members == 1` (members includes the bot).
  - Humans present in the target voice channel and not connected → `_start_meeting`.
  - Target channel now has no humans and we're connected → `_finalize`.
  - A human left the target channel but humans remain and the meeting is active → `sink.flush_user(member.id)` (flush that speaker's trailing audio into the transcript).
- **Finalize:** stop recording, then `transcriber.stop(flush=True)` (drains the input queue so the flushed trailing chunk is transcribed) and `drain(timeout)` for pending events, then build the transcript. If `transcript.is_empty()`, post a "no speech detected" note instead of calling the summarizer. Otherwise summarize (via `asyncio.to_thread`), parse, post the embed to the target channel, then disconnect. Do **not** rely on `stop_recording`'s `once_done` callback firing (the pinned branch has a known truthiness bug with empty args) — drive finalize from `on_voice_state_update` directly.
- **Race guard:** `_meeting_active` flag + lock; re-check humans before disconnecting; a member joining mid-finalize must not double-post.
- **Watchdog:** if no PCM frames arrive N seconds (e.g. 10 s) after `start_recording`, log a prominent warning referencing py-cord issue #3139 and the required py-cord pin.

### `main.py` / `__main__.py`
```python
def main(argv: list[str] | None = None) -> int:
    # python -m meeting_bot            -> run the bot
    # python -m meeting_bot --doctor   -> print doctor report, exit 0 if all ok else 1
    # python -m meeting_bot --help
```

### `tests/` (pytest, pure logic only — no Discord/network)
- `test_audio.py`: resample output length `⌊n*960/3⌋`-derived; dtype float32; range ⊆ [−1,1]; a 1 kHz tone at 48 kHz maps to correct 16 kHz samples; silence → ~0 energy; `is_speech_block` threshold behavior.
- `test_chunker.py`: [silence, 2 s tone, silence] → exactly one Segment with correct speaker/timestamps; sub-`min_chunk_seconds` blip dropped; continuous > `max_chunk_seconds` forced into multiple segments; `flush()` closes trailing partial.
- `test_transcript.py`: `to_prompt_text()` ordering + `[MM:SS]` format; `is_empty()`.
- `test_summary_parse.py`: valid JSON → Summary; fenced JSON → parses; markdown sections → fallback; English-key JSON → parses; garbage → non-raising last resort.

## `requirements.txt` (pin exactly)

```
py-cord[voice] @ git+https://github.com/Pycord-Development/pycord.git@326b72acc8d1d952ac002fe07ca65581cf5952bc
mlx-whisper>=0.4.3
mlx>=0.32
anthropic
python-dotenv
numpy>=2.3.2          # cp314-wheel floor; numba (via mlx-whisper) caps the top at <2.5
pytest>=9,<10
pytest-asyncio>=1.3.0
```

## `.env.example` (mirror every Config key, placeholders + comments, no secrets)

```
DISCORD_TOKEN=       # bot token (Developer Portal) — never commit the real one
GUILD_ID=            # test guild id
VOICE_CHANNEL_ID=    # voice channel to watch
TARGET_CHANNEL_ID=   # text channel that receives the summary
ANTHROPIC_BASE_URL=https://gateway.9arm.co
ANTHROPIC_AUTH_TOKEN=sk-...   # gateway bearer token
GATEWAY_MODEL=qwen3.6-35b-a3b
WHISPER_MODEL=mlx-community/whisper-large-v3-turbo
WHISPER_LANGUAGE=th
SILENCE_THRESHOLD=0.01
SILENCE_SECONDS=0.8
MIN_CHUNK_SECONDS=1.0
MAX_CHUNK_SECONDS=30.0
```

## Acceptance criteria

1. `python3 -m compileall -q meeting_bot tests` exits 0.
2. The five pure modules (`config`, `audio`, `chunker`, `transcript`, `summary_parse`) import with only numpy present (no discord/anthropic/mlx_whisper).
3. Every class/function above exists with the documented signature.
4. `main.py` supports `--doctor` and `--help` without connecting.
5. No tokens or secrets appear in any file.
6. `.env.example` contains exactly the keys `config.py` reads.
