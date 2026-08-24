# Meeting Summary Discord Bot — Build Spec

You are implementing a complete, runnable Discord bot package from this spec. Read carefully; every module, signature, and constraint below is mandatory. Where this spec and your own judgment conflict, the spec wins.

## Overview

A Discord bot that joins a voice channel, transcribes the meeting live in **Thai**, and when the **last human** leaves the voice channel, automatically posts a structured summary — **overview / Topics / Decisions / Action Items / open questions** — to a configured text channel. No manual command triggers it.

Data flow:

```
Discord voice channel
  → py-cord per-user voice receive (DAVE-patched branch) → per-user PCM
  → resample 48 kHz stereo int16 → 16 kHz mono float32 (numpy, no ffmpeg)
  → silence-based chunking per speaker (worker thread safety)
  → mlx-whisper (local, Metal) transcribes each closed chunk in Thai
  → accumulate labeled transcript
  → on voice channel empty: BLOCKING local gateway (Anthropic-compatible) summarizes
  → post compact embed + full Markdown file attachment to target text channel
```

## Locked decisions (do not change)

1. **Voice receive:** py-cord pinned to the DAVE-patched unmerged branch — released 2.8.1's voice reception is buggy under Discord's DAVE encryption (py-cord issue #3139); the fix reworks per-user receive. `requirements.txt` must pin the **exact commit** `git+https://github.com/Pycord-Development/pycord.git@326b72acc8d1d952ac002fe07ca65581cf5952bc` (branch `fix/voice-rec-2`; a moving-branch pin is not reproducible). Do not substitute released 2.8.1 — receive will be broken.

   **macOS prerequisite:** `brew install opus` is required for voice (py-cord ships libopus only as Windows DLLs). The bot's `doctor()` checks `ctypes.util.find_library("opus")`. First whisper run downloads ~3 GB into `~/.cache/huggingface` (host default is fine; set `HF_HOME` only if the cache must live elsewhere).
2. **STT:** local `mlx-whisper`, model `mlx-community/whisper-large-v3-mlx` (non-turbo — large-v3-turbo is prone to confident repetition-loop hallucinations on Thai), `language="th"`. No cloud API. Runs on the same Apple Silicon MacBook as the AI gateway, so GPU contention is a real concern — serialized worker, fp16, model cached once. The transcriber hardens the decode (`condition_on_previous_text=False`, greedy T=0) and re-decodes once with a small temperature bump + Thai preamble when a decode is flagged garbage; see `tools/offline_repro.py` for an A/B harness.
3. **Summarization:** the user's own gateway via the **Anthropic-compatible** API — `anthropic` SDK, `base_url=https://gateway.9arm.co`, `auth_token` from env (`ANTHROPIC_AUTH_TOKEN`), model `qwen3.8-27b-fp8` (quantized variant, 128k-token context; must match the gateway's allowlist exactly — the doctor probe catches mismatches). **Do not put `/v1` in the base URL** (the SDK appends `/v1/messages`). The call is a **blocking** `client.messages.create()` — probing the gateway (`tools/probe_stream.py`) showed it buffers the whole completion server-side (SSE events arrive in a burst near the end), so streaming adds no early-abort value; the SDK client timeout `SUMMARIZE_TIMEOUT_SECONDS=300` is the ceiling. Prompt budget `MAX_PROMPT_CHARS=135000` (declared 128k-token window on the fp8 variant; measured 2026-08-25 at ≈1.23 chars/token for Thai prompt text — 135k chars ≈110k tokens, ~18k-token headroom for system + full output; re-measure from `tools/e2e_summarize_probe.py` if the model changes), output budget `SUMMARY_MAX_TOKENS=8192`. qwen's structured-output reliability is the known risk — the summary parse must never raise. The completed output is checked post-hoc for an exact-repeat loop (`REPETITION_WINDOW_CHARS` × `REPETITION_MIN_REPEATS`); a loop raises `StalledGenerationError(progressed=True)` (never retried), which `bot.py` surfaces as a visible ⚠️ note in the target channel. An empty completion raises `EmptySummaryError` (never retried).
4. **Secrets/config:** `.env` (never committed) via python-dotenv. A `.env.example` mirrors every key with placeholders and comments, no real secrets.

## File tree — produce exactly this

```
meeting_bot/
  __init__.py          # __version__ = "0.1.0"; lazy re-export Config, MeetingBot, ThaiPolisher
  config.py            # Config dataclass, load_config(), doctor()
  audio.py             # resample + RMS helpers (numpy only)
  chunker.py           # SilenceChunker, Segment (numpy/stdlib only)
  sink.py              # MeetingSink(discord.sinks.Sink) per-user PCM capture
  transcriber.py       # Transcriber: background mlx-whisper worker
  transcript.py        # TranscriptEvent, Transcript accumulator (stdlib only)
  rag_store.py         # Chunk, chunk_transcript(), VectorIndex, truncate_block (numpy/stdlib only)
  embedder.py          # Embedder: local MLX sentence embeddings (lazy mlx_embedding_models import)
  summary_parse.py     # Summary, TopicItem, DecisionItem, ActionItem, parse_summary() (stdlib only)
  summarizer.py        # Summarizer: BLOCKING anthropic gateway call (lazy anthropic import)
  thai_polish.py       # ThaiPolisher: kien-thai audit+fix loop via OpenTyphoon (lazy openai usage)
  poster.py            # build_embed(), render_markdown(), Poster
  bot.py               # MeetingBot(discord.Bot): voice trigger, /leave + /join slash commands, orchestration
  main.py              # argparse entrypoint: run | --doctor
  __main__.py          # raise SystemExit(main())
  wav_dump.py          # env-gated DUMP_CHUNKS_DIR .wav writer (stdlib only)
tests/
  test_audio.py
  test_chunker.py
  test_transcript.py
  test_summary_parse.py
  test_transcriber.py
  test_wav_dump.py
  test_summarizer.py
  test_poster.py
  test_thai_polish.py
  test_rag_store.py
  test_embedder.py
  test_bot_rejoin.py
requirements.txt
.env.example
```

**Import rule (load-bearing).** `config.py`, `audio.py`, `chunker.py`, `transcript.py`, `summary_parse.py`, `summarizer.py`, `wav_dump.py` **and the new `rag_store.py`** must import **only stdlib + numpy** at module scope (`wav_dump.py` is stdlib-only; `summarizer.py` imports only `logging`, `time`, and `.config` at module scope). `summarizer.py` imports `anthropic` **lazily** (inside `__init__`/methods, not at module scope); `embedder.py` likewise lazy-imports `mlx_embedding_models` inside `Embedder.__init__`; `thai_polish.py` imports `openai` at module scope but is itself imported lazily by the package `__getattr__`. `sink.py`, `bot.py`, `poster.py` may import `discord`. `config.py` imports `dotenv`/`load_dotenv` **inside `load_config()` only** — python-dotenv is neither stdlib nor numpy, so a module-scope import breaks the pure-import rule. This lets the pure modules be imported and tested without the heavy deps installed.

## Module contracts

### `config.py`
```python
@dataclass(frozen=True)
class Config:
    discord_token: str
    guild_id: int
    voice_channel_id: int
    target_channel_id: int
    anthropic_base_url: str      # no trailing /v1 (the SDK appends /v1/messages)
    anthropic_auth_token: str
    gateway_model: str           # qwen3.8-27b-fp8
    whisper_model: str           # mlx-community/whisper-large-v3-mlx (see STT decision below)
    whisper_language: str        # "th"
    silence_threshold: float = 0.01     # RMS speech threshold (~ −40 dBFS)
    silence_seconds: float = 0.8        # trailing silence to close a chunk
    min_chunk_seconds: float = 1.0      # shorter closed chunks are dropped
    max_chunk_seconds: float = 30.0     # force-close cap
    summarize_timeout_seconds: float = 300.0  # SDK client timeout for gateway
    max_prompt_chars: int = 135000      # transcript truncation limit (128k-context fp8 gateway)
    summary_max_tokens: int = 8192     # gateway output-token budget (qwen thinking + richer schema)
    repetition_window_chars: int = 300   # exact-repeat loop detection window
    repetition_min_repeats: int = 3      # identical consecutive windows before declaring a loop
    rag_enabled: bool = True             # per-meeting RAG retrieval over the transcript
    embedding_model: str = "BAAI/bge-m3" # local MLX embeddings for RAG
    rag_top_k: int = 8                   # top-k chunks retrieved per query
    rag_chunk_chars: int = 800           # sliding-window chunk size (chars)
    rag_overlap_chars: int = 150         # trailing lines repeated per chunk
    polish_enabled: bool = False          # kien-thai Thai-writing audit+fix loop
    polish_api_key: str = ""              # OpenAI-compatible key for OpenTyphoon
    polish_base_url: str = "https://api.opentyphoon.ai/v1"
    polish_model: str = "typhoon-v2.5-30b-a3b-instruct"
    polish_max_passes: int = 20
    polish_timeout_seconds: float = 120.0

def load_config(path: str | os.PathLike = ".env") -> Config: ...
def doctor(cfg: Config) -> list[str]:   # list of "ok: ..."/"fail: ..." lines
```
`load_config` reads `.env` with `python-dotenv` (imported inside the function), validates required keys present and non-empty, raises a clear error naming the missing key. `doctor` returns one `ok: ...`/`fail: ...` line per check and **never raises**: all required keys present; `discord`/`mlx_whisper`/`anthropic`/`numpy` importable (each individually — a missing one is a `fail` line, not an exception); system `libopus` resolvable via `ctypes.util.find_library("opus")`; whisper model name non-empty; a **gateway probe**: construct the anthropic client from cfg and call `messages.create(model=cfg.gateway_model, max_tokens=1)` with a 30 s timeout and one retry on timeout only (the self-hosted qwen thinking model can exceed 10 s on cold load, so a slow-but-up gateway must not fail the check) — `ok` iff no exception and HTTP 2xx, `fail` on 401/403 (bad token), network error, or two consecutive timeouts, `fail` (not an exception) if `anthropic` isn't installed. When `polish_enabled`: a key-presence check plus a cheap **OpenTyphoon reachability probe** (`max_tokens=1`, 15 s timeout, ok/fail line — never raises, key never logged). When `rag_enabled`: an `mlx_embedding_models` importability line naming `EMBEDDING_MODEL`. Never log any auth token.

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
One worker thread pulls segments, optionally dumps the audio to `.wav` (`DUMP_CHUNKS_DIR`, env-gated — see `meeting_bot/wav_dump.py`), and calls `mlx_whisper.transcribe(samples_array, path_or_hf_repo=model, **kwargs)["text"]`, pushing `TranscriptionEvent(speaker, start, text)` onto the output queue. The decode kwargs harden against repetition-loop hallucinations: **greedy `temperature=0`, `condition_on_previous_text=False`, `no_speech_threshold=0.6`, `fp16` (env toggle `WHISPER_FP16`)**. mlx-whisper's own temperature fallback cannot catch confident loops (they compress well and are high-confidence, so `compression_ratio > 2.4` / `avg_logprob < -1.0` never fire), so a suspicious primary decode — garbage per `is_garbage_transcription`, or `no_speech_prob` above threshold — is re-decoded **once** with a small temperature bump (`WHISPER_RETRY_TEMPERATURE`, default 0.2) + Thai preamble (`WHISPER_INITIAL_PROMPT`, default `"ต่อไปนี้คือการประชุม: "`, empty disables) and only dropped if the retry is still bad. See `tools/offline_repro.py` for the A/B harness. Passing a numpy array skips ffmpeg (input is 16 kHz mono f32). Load the model once and reuse (the MLX `ModelHolder` caches it; `fp16` must therefore be constant per process). Serializes MLX so the GPU isn't contended with the gateway. If `mlx_whisper` import fails at construction, raise a clear error. **Right before transcribing, assert the array is exactly 16 kHz mono float32** (e.g. `assert samples.dtype == np.float32 and samples.ndim == 1` and the rate is 16000) — mlx-whisper does **not** validate or resample array input; wrong rate/dtype silently yields garbage, so a pipeline regression must fail loudly instead. `stop(flush=True)` drains the input queue to completion before the worker exits (so a flushed trailing chunk is transcribed, not dropped); `drain(timeout)` waits up to `timeout` for pending transcriptions to land on the output queue and returns them. `_finalize` calls `stop(flush=True)` (or `drain`) before building the transcript.

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
    def to_prompt_text(self, max_chars: int | None = 48000) -> str: ...
    # "[MM:SS] ผู้พูด: ..." per line, chronological.  Truncates to max_chars
    # (whole lines only, with a "...(truncated)" suffix); None disables.
    def is_empty(self) -> bool: ...
```

### `rag_store.py` (pure stdlib + numpy — per-meeting RAG index)
```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    header: str        # "[MM:SS–MM:SS]" time span the chunk covers
    text: str          # complete "[MM:SS] speaker: text" lines, newline-joined
    start_sec: float   # offset of the chunk's first line from meeting start
    def as_embedding_text(self) -> str: ...  # header + "\n" + text

def chunk_transcript(events, started_at, chunk_chars=800,
                     overlap_chars=150) -> list[Chunk]: ...
def truncate_block(text: str, max_chars: int) -> str: ...
class VectorIndex:
    def __len__(self) -> int: ...
    def add(self, vectors, chunks: list[Chunk]) -> None: ...
    def query(self, vector, k: int) -> list[tuple[Chunk, float]]: ...
    def query_multi(self, vectors, k: int) -> list[tuple[Chunk, float]]: ...
```
- `chunk_transcript` renders lines exactly like `Transcript.to_prompt_text` (`[MM:SS] speaker: text`, chronological), packs them greedily up to `chunk_chars`, **never splits a line** (an oversized line becomes its own chunk), and repeats trailing lines totalling ≤ `overlap_chars` at the head of the next chunk (`0` ⇒ no overlap). Returns `[]` for an empty transcript.
- `VectorIndex` stores L2-normalized float32 rows; cosine similarity is a plain dot product. `add` raises on length/dimension mismatch; zero rows stay zero. `query` returns top-k `(chunk, score)` best-first; empty-index queries return `[]`. `query_multi` unions the top-k hits per query row, dedupes by chunk keeping the **max** score, and re-sorts **chronologically** so the summarized excerpt reads in meeting order.
- `truncate_block` mirrors `Transcript.to_prompt_text` truncation semantics (whole lines, single oversized line sliced, `...(truncated)` suffix).
- Brute-force by design — hundreds of chunks per meeting needs no ANN. Nothing persists across meetings.

### `embedder.py` (lazy `mlx_embedding_models` import)
```python
class Embedder:
    def __init__(self, model: str = "BAAI/bge-m3"): ...  # lazy import here
    @property
    def dim(self) -> int: ...
    def embed_texts(self, texts) -> np.ndarray: ...  # (n, d) float32, L2-normalized
    def embed_query(self, text: str) -> np.ndarray: ...  # (d,)
```
- Wraps `mlx_embedding_models.embedding.EmbeddingModel.from_pretrained(...)` / `.encode(...)`. ImportError names the pip package. bge-m3 is symmetric (no query/passage prefixes). Outputs L2-normalized float32 to match `VectorIndex`.

### `summary_parse.py` (pure stdlib — the reliability mitigation)```python
@dataclass
class TopicItem:
    title: str
    detail: str = ""

@dataclass
class DecisionItem:
    decision: str
    rationale: str = ""

@dataclass
class ActionItem:
    action: str
    owner: str | None = None
    due: str | None = None
    @classmethod
    def parse(cls, obj) -> "ActionItem": ...   # dict or plain string

@dataclass
class Summary:
    overview: str
    topics: list[TopicItem]
    decisions: list[DecisionItem]
    action_items: list[ActionItem]
    open_questions: list[str]
    raw: str

def parse_summary(text: str) -> Summary: ...   # never raises
```
`__all__ = ["ActionItem", "TopicItem", "DecisionItem", "Summary", "parse_summary"]`.

Parse order:
1. **JSON attempt:** strip ```json``` fences and prose before `{`/after `}`; `json.loads`; tolerate Thai or English keys: `overview`/`ภาพรวม`/`สรุปภาพรวม`, `topics`/`หัวข้อ`, `decisions`/`การตัดสินใจ`, `action_items`/`รายการที่ต้องทำ`, `open_questions`/`คำถามที่ยังไม่ได้ข้อสรุป`/`ประเด็นค้าง`. Item-level keys too: topic `title`/`หัวข้อ` + `detail`/`รายละเอียด`; decision `decision`/`การตัดสินใจ` + `rationale`/`เหตุผล`; action `action` + `owner` + `due`/`กำหนดเวลา`/`ครบกำหนด`. Coerce items via `TopicItem`/`DecisionItem`/`ActionItem` — **plain-string list items wrap into `TopicItem(title=s, detail="")` / `DecisionItem(decision=s, rationale="")` (backward-compat with the old schema)**; `ActionItem.parse` accepts a dict or a plain string.
2. **Markdown fallback:** split on section headers matching `หัวข้อ|Topics`, `การตัดสินใจ|Decisions`, `สิ่งที่ต้องทำ|Action Items`, plus `ภาพรวม|Overview` and `คำถามที่ยังไม่ได้ข้อสรุป|Open Questions`; collect `-`/`*` bullets. Overview is collected as a **paragraph** (non-bullet lines until the next header); open questions as bullets. Topic/decision bullets split on `label: value`, `—`, or `–` into title/detail and decision/rationale (a bare bullet with no separator is the title with empty detail; strip `**` from a bold title). Return the result if **any** of overview/topics/decisions/action_items/open_questions was found.
3. **Last resort:** `Summary(overview=text, topics=[], decisions=[], action_items=[], open_questions=[], raw=text)`. Empty input → `overview=""`. Never raise.

### `summarizer.py` (anthropic, lazy import — BLOCKING)
```python
class EmptySummaryError(RuntimeError): ...
class StalledGenerationError(RuntimeError):
    def __init__(self, message: str, *, progressed: bool) -> None: ...
    # progressed = whether ANY output was produced (post-hoc loop check: always True)

class Summarizer:
    def __init__(self, cfg: Config): ...    # max_retries=0; anthropic imported here (lazy)
    def summarize(self, transcript_text: str) -> str: ...   # blocking; never returns non-text
    def _summarize_once(self, transcript_text: str) -> str: ...  # one blocking attempt
```
`__all__ = ["Summarizer", "EmptySummaryError", "StalledGenerationError"]`. Module scope imports are **only** `logging`, `time`, and `from .config import Config` — `anthropic` is imported lazily inside `__init__`.

**Blocking mechanics (`_summarize_once`).** Use `client.messages.create(...)` (non-streaming). The gateway buffers the whole completion server-side and returns it in one response, so there is no live stream to inspect — the SDK client timeout `SUMMARIZE_TIMEOUT_SECONDS` is the ceiling. Join the `text` blocks of the returned message; if no text at all, raise `EmptySummaryError` (qwen sometimes spends the entire token budget in its thinking trace and never emits a text block — the rich schema + `SUMMARY_MAX_TOKENS` budget is the mitigation, but the bot must still never post an empty summary).

**Post-hoc loop guard.** After a successful completion, run `_is_looping(buf, window, min_repeats)` on the combined thinking + text output (the thinking trace can loop without terminating). If True, raise `StalledGenerationError(progressed=True)`. `_is_looping` returns True when the last `min_repeats` consecutive trailing `window`-char slices are byte-identical; guard clauses return False for `window <= 0`, `min_repeats < 2`, or `len(buf) < window * min_repeats`. The exact-repeat check must not false-positive on legitimate repetitive JSON schema output (repeated `"owner":` keys, etc.).

**Retry policy.** `summarize()` retries **once, and only on `APITimeoutError`** — a gateway hiccup with no response ever starting. `EmptySummaryError` and `StalledGenerationError` are **never** retried (`StalledGenerationError` is always `progressed=True`): `temperature=0` means re-running reproduces the same trace. `max_retries=0` disables the SDK's own retry loop (the manual policy above is authoritative).

**Contract & schema.** `timeout=cfg.summarize_timeout_seconds` on the client; `max_tokens=cfg.summary_max_tokens`, `temperature=0.0`. The Thai system prompt instructs the model to reply only in Thai and output **only** the JSON below (no markdown fence, no other text); the user message appends a suffix demanding empty sections be `[]`/`""`/`null` per type:

```json
{
  "overview": "...",
  "topics": [{"title": "...", "detail": "..."}],
  "decisions": [{"decision": "...", "rationale": "..."}],
  "action_items": [{"action": "...", "owner": "...", "due": "..."}],
  "open_questions": ["..."]
}
```
The bot drains the returned text through `summary_parse.parse_summary`. Call `summarize` via `asyncio.to_thread` from `bot.py` so the event loop isn't blocked.

### `poster.py`
```python
def build_embed(summary, *, started_at: datetime, duration: timedelta,
                member_count: int, meeting_title: str | None = None) -> discord.Embed: ...
def render_markdown(summary, *, started_at: datetime, duration: timedelta,
                    member_count: int, meeting_title: str | None = None) -> str: ...
class Poster:
    def __init__(self, config): ...
    async def post(self, channel, summary, *, meta) -> discord.Message: ...
```
**Two-output design: a compact embed as a scannable index + the full detail as an attached `summary.md` file** (the embed is capped by Discord's 6000-char/embed limit; the file is where "more context" actually lives, untruncated).

- `build_embed` — titled `📝 สรุปการประชุม`; `description = summary.overview or meeting_title or "การประชุม"`. One field per section: `หัวข้อที่พูดคุย` (`• **{title}** — {detail}`; bare `• **{title}**` when detail empty), `การตัดสินใจ` (same pattern with `decision`/`rationale`), `รายการที่ต้องทำ` (`• {action} — {owner}` + ` (due: …)` when present), `คำถามที่ยังไม่ได้ข้อสรุป` (`—` when empty). Footer with guild/voice-channel name, date, duration, member count. `meeting_title` is optional — when `None`, default to the guild/voice-channel name. **Shared 6000-char budget, not fixed independent caps:** Discord rejects embeds whose title + description + all field names/values + footer total more than 6000, so `build_embed` counts the fixed text (title, the four field names, footer) first and carves the remaining pool across the description (soft cap `_DESCRIPTION_MAX=2000`) and the four field values (each ≤ `_FIELD_MAX=1024`) via a pure `_fit_to_pool(desired, pool)` that scales every piece **proportionally** when the desired total overflows — a rich meeting shrinks all sections instead of 400ing. Truncation always appends `_TRUNCATE_NOTE = "…ดูรายละเอียดเพิ่มเติมในไฟล์แนบ"` within the limit and never exceeds it (even for tiny budgets).
- `render_markdown` — the full untruncated document: title + overview, `### {title}` per topic + full `detail`, `### {decision}` per decision + full `rationale`, an action-item table (`# | สิ่งที่ต้องทำ | ผู้รับผิดชอบ | กำหนดส่ง`), all open questions.
- `Poster.post` — render the markdown once; build `discord.File(io.BytesIO(md.encode("utf-8")), filename="summary.md")` **inside the retry loop** (a failed send consumes the BytesIO, so re-create it per attempt); `channel.send(embed=embed, file=file)`. Send to `target_channel_id` via `bot.get_channel(...)`. Retry up to 3× on 429/5xx. ASCII filename.

### `bot.py`
```python
def rejoin_allowed(manual_leave: bool, human_count: int) -> bool: ...

class MeetingBot(discord.Bot):   # py-cord Bot subclasses Client — run/intents/events unchanged
    def __init__(self, cfg: Config, *, intents=...): ...
    async def on_ready(self): ...
    async def on_voice_state_update(self, member, before, after): ...
    async def _start_meeting(self, channel): ...
    async def _finalize(self, channel, *, force_disconnect: bool = False): ...
```
- **Base class is py-cord's `discord.Bot`** (verified against the pinned DAVE commit: `Bot.__init__(description=None, *args, **options)` forwards to `Client`, so `intents=` and `run(token)` behave identically). Two guild-scoped slash commands are registered idempotently in `__init__` (guard flag) via the instance decorator with `guild_ids=[cfg.guild_id]` (instant sync; no ~1 hr global propagation); anyone in the guild may invoke them:
  - `/leave` — if no meeting active: disconnect any lingering voice client + ephemeral "ไม่มีการประชุมอยู่ตอนนี้". Otherwise `ctx.respond("⏳ กำลังสรุปการประชุม…")` **immediately** (interaction tokens expire in 15 min; finalize takes minutes) and spawn `_finalize(channel, force_disconnect=True)` as a background task. Never raises past the handler.
  - `/join` — clears `_manual_leave`; ephemeral "already connected" note if a meeting/connection exists; ephemeral ack + background `_start_meeting(channel)` when humans are present; polite ephemeral note when the channel is empty.
- **Force-disconnect finalize:** `force_disconnect=False` keeps every behavior below unchanged. When `True` (`/leave`), set `_manual_leave = True`, skip the humans-present keep-connection race-guard branch entirely (a human joining mid-forced-finalize must not resurrect the meeting), always tear down + disconnect, and still serialize on `_meeting_lock` (a concurrent auto-finalize makes the forced call fall through to the connection-teardown path).
- **Rejoin suppression:** while `_manual_leave` is set, block `_start_meeting` from the voice-state handler; clear the flag only when the target channel is observed with zero humans, then normal auto-rejoin resumes. The gate decision is the module-level pure function `rejoin_allowed(manual_leave, human_count)`.
- Intents: `guilds`, `voice_states`, and `members` (members is a **privileged** intent — note in `.env.example`/README that it must be enabled in the Developer Portal). No Message Content intent.
- Join self-muted/deaf (correct py-cord sequence): `vc = await channel.connect()` (VocalGuildChannel.connect accepts only `timeout`/`reconnect`/`cls`); then `await guild.change_voice_state(channel=channel, self_mute=True, self_deaf=True)` (returns `None` — it does **not** create the VoiceClient); then `vc.start_recording(sink)`.
- **Trigger:** in `on_voice_state_update`, count **humans** — `[m for m in channel.members if not m.bot]` — never gate on `member == self.user and before.channel.members == 1` (members includes the bot).
  - Humans present in the target voice channel and not connected → `_start_meeting`.
  - Target channel now has no humans and we're connected → `_finalize`.
  - A human left the target channel but humans remain and the meeting is active → `sink.flush_user(member.id)` (flush that speaker's trailing audio into the transcript).
- **Finalize:** stop recording, then `transcriber.stop(flush=True)` (drains the input queue so the flushed trailing chunk is transcribed) and `drain(timeout)` for pending events, then build the transcript. If `transcript.is_empty()`, post a "no speech detected" note instead of calling the summarizer. Otherwise build the prompt via `_build_prompt(transcript, cfg)` (RAG retrieval when `rag_enabled`, legacy truncated render otherwise — see below), summarize via `asyncio.to_thread`, parse, run the optional kien-thai polish pass, post the embed + file to the target channel, then disconnect. Handle summarizer failures visibly rather than silently: `except EmptySummaryError` posts a Thai "no summary generated" note; `except StalledGenerationError` (a `RuntimeError` subclass, so catch it **before** the generic `except Exception`) posts a ⚠️ note explaining the output was detected as a repetition loop and pointing at `REPETITION_WINDOW_CHARS` / `REPETITION_MIN_REPEATS`. Do **not** rely on `stop_recording`'s `once_done` callback firing (the pinned branch has a known truthiness bug with empty args) — drive finalize from `on_voice_state_update` directly.
- **RAG prompt construction (`_build_prompt`).** Runs inside the same worker thread as the gateway call (the embedder may load a ~2 GB model). When `cfg.rag_enabled`: chunk the transcript (`chunk_chars=cfg.rag_chunk_chars`, `overlap_chars=cfg.rag_overlap_chars`), embed chunks with `Embedder(cfg.embedding_model)`, index them, run one fixed five-query Thai set (overall gist / topics discussed / decisions made / action items & owners / unresolved questions), union top-k per query (`cfg.rag_top_k`), dedupe + chronological order via `query_multi`, join chunk texts, hard-cap at `cfg.max_prompt_chars` with `truncate_block`. Log `rag: chunks=%d retrieved=%d prompt_chars=%d`. Any failure (ImportError, embed error, empty index) logs a warning and falls back to `Transcript.to_prompt_text(max_chars=...)`; posting is never blocked. `RAG_ENABLED=false` ⇒ byte-for-byte legacy prompt.
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
- `test_transcript.py`: `to_prompt_text()` ordering + `[MM:SS]` format; default `max_chars=48000`; explicit truncation (≤ limit, `...(truncated)` suffix, whole lines) and `max_chars=None` disables; `is_empty()`.
- `test_summary_parse.py`: new-schema JSON (overview + `TopicItem`/`DecisionItem`/`ActionItem` with due) → Summary; fenced JSON + prose → parses; Thai keys (`ภาพรวม`, `หัวข้อ`, `การตัดสินใจ`, `รายการที่ต้องทำ` with `กำหนดเวลา`, `คำถามที่ยังไม่ได้ข้อสรุป`) and English keys → parse; **backward-compat**: old bare-string topics/decisions wrap into `TopicItem`/`DecisionItem` with empty detail/rationale; markdown fallback with `ภาพรวม`/`คำถามที่ยังไม่ได้ข้อสรุป` headers and `label: value`/`—` splits; garbage → non-raising last resort (`overview == raw`); empty → all empty; `ActionItem.parse` plain string and dict-with-due.
- `test_transcriber.py`: `is_garbage_transcription` heuristics (long char runs, token-ratio, Thai no-whitespace repetition); Thai politeness particles (`ครับ ครับ ครับ`, `ค่ะ ค่ะ`) never flagged; `should_retry` (garbage → retry, high `no_speech_prob` → retry, empty/whitespace → never, threshold is strictly `>`); `build_decode_kwargs` primary (T=0, `condition_on_previous_text=False`, no prompt) vs retry (temperature > 0, Thai preamble); env toggles `WHISPER_FP16`/`WHISPER_RETRY_TEMPERATURE`/`WHISPER_INITIAL_PROMPT`.
- `test_wav_dump.py`: inert when `DUMP_CHUNKS_DIR` unset; writes well-formed 16 kHz mono 16-bit PCM; float32→int16 clamp; filename sanitization + zero-padded sort.
- `test_summarizer.py`: `_is_looping` fires at exactly `min_repeats` identical trailing windows and never false-positives on repetitive JSON (`"owner":` keys); guard clauses (`window<=0`, `min_repeats<2`, `len < window*min_repeats`) → False; `StalledGenerationError` stores `.progressed`; retry policy via a fake `_summarize_once` — `APITimeoutError` retries once then raises, `EmptySummaryError` / `StalledGenerationError`(progressed=True) raise without retry; a fake `messages.create` drives the post-hoc loop and empty-summary branches. `summarizer` imports only stdlib + `.config` at module scope, so this test runs with no `anthropic` installed (build the Summarizer via `object.__new__` to bypass `__init__`).
- `test_poster.py` (guarded by `pytest.importorskip("discord")` — skips in a minimal env; `poster` imports `discord` at module scope): the pure `_fit_to_pool` allocator (unchanged when desired fits, proportional scaling when it overflows, zero pool → all zeros); a **rich-summary regression** that builds a full embed with a 2000-char overview and every field near-maxed and asserts the title + description + field names/values + footer total stays `<= 6000` (Discord's hard limit — the old fixed caps reached 6286 and would 400); a short-summary case asserting nothing is needlessly truncated.

## `requirements.txt` (pin exactly)

```
py-cord[voice] @ git+https://github.com/Pycord-Development/pycord.git@326b72acc8d1d952ac002fe07ca65581cf5952bc
mlx-whisper>=0.4.3
mlx>=0.32
anthropic
mlx-embedding-models    # local RAG embeddings (BAAI/bge-m3); import name mlx_embedding_models
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
GATEWAY_MODEL=qwen3.8-27b-fp8
WHISPER_MODEL=mlx-community/whisper-large-v3-mlx
WHISPER_LANGUAGE=th
SILENCE_THRESHOLD=0.01
SILENCE_SECONDS=0.8
MIN_CHUNK_SECONDS=1.0
MAX_CHUNK_SECONDS=30.0
SUMMARIZE_TIMEOUT_SECONDS=300
MAX_PROMPT_CHARS=135000
SUMMARY_MAX_TOKENS=8192
REPETITION_WINDOW_CHARS=300
REPETITION_MIN_REPEATS=3
RAG_ENABLED=true
EMBEDDING_MODEL=BAAI/bge-m3
RAG_TOP_K=8
RAG_CHUNK_CHARS=800
RAG_OVERLAP_CHARS=150
POLISH_ENABLED=false
POLISH_API_KEY=
POLISH_BASE_URL=https://api.opentyphoon.ai/v1
POLISH_MODEL=typhoon-v2.5-30b-a3b-instruct
POLISH_MAX_PASSES=20
POLISH_TIMEOUT_SECONDS=120
```

`.env.example` must contain **exactly** the keys `config.py` reads (`_REQUIRED_ENV` + `_OPTIONAL_FLOAT_ENV` names plus the RAG/polish keys read in `load_config`). Runtime-only toggles (`DUMP_CHUNKS_DIR`, `WHISPER_FP16`, `WHISPER_RETRY_TEMPERATURE`, `WHISPER_INITIAL_PROMPT`, `WHISPER_BEAM_SIZE`) are read via `os.environ` by `transcriber.py`/`wav_dump.py` and deliberately do **not** appear as Config keys (they are documented in `.env.example`'s notes section only).

## Acceptance criteria

1. `python3 -m compileall -q meeting_bot tests` exits 0.
2. The pure modules (`config`, `audio`, `chunker`, `transcript`, `summary_parse`, `summarizer`, `wav_dump`, `rag_store`) import with only stdlib (+ numpy for `audio`/`chunker`/`rag_store`) present — no discord/anthropic/mlx_whisper/mlx_embedding_models/openai; `wav_dump` and `summarizer` need nothing beyond stdlib; `embedder` imports nothing heavy at module scope.
3. Every class/function above exists with the documented signature — including `Summarizer.summarize`, `_summarize_once`, `EmptySummaryError`, `StalledGenerationError(..., *, progressed)`, `render_markdown`, the RAG/polish `Config` fields, `Transcript.to_prompt_text(max_chars=48000)`, `chunk_transcript`, `VectorIndex.query_multi`, `Embedder.embed_texts`, `MeetingBot._finalize(channel, *, force_disconnect)`, and `rejoin_allowed`.
4. `main.py` supports `--doctor` and `--help` without connecting.
5. No tokens or secrets appear in any file.
6. `.env.example` contains exactly the keys `config.py` reads (`_REQUIRED_ENV` + `_OPTIONAL_FLOAT_ENV` names + RAG/polish keys), including `REPETITION_WINDOW_CHARS`/`REPETITION_MIN_REPEATS` and the `RAG_*`/`EMBEDDING_MODEL` block. Runtime toggles (`DUMP_CHUNKS_DIR`, `WHISPER_FP16`, `WHISPER_RETRY_TEMPERATURE`, `WHISPER_INITIAL_PROMPT`, `WHISPER_BEAM_SIZE`) are not Config keys and must not appear as keys.
