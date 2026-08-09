# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **Shepherd workspace** (shepherd-agents/shepherd, v0.3.0) — not a library. The two Python files at the root are quickstart agent tasks that run sandboxed agents inside this directory. The framework itself is installed in the user site-packages (`shepherd`, `shepherd_dialect`, `vcs_core`) — it is not vendored here, and there is no build, lint, or test setup to run.

## Meeting Summary Discord Bot

The active project: a real, long-running Discord bot that joins a voice channel, transcribes the meeting live in Thai, and — when the last human leaves the voice channel — auto-posts a structured summary (Topics / Decisions / Action Items) to a configured text channel. It is built **with** Shepherd but does **not** use Shepherd at runtime (Shepherd is one-shot/sync; the bot is plain py-cord).

**Build flow (how the bot is produced).** `build_bot_task.py` is the Shepherd build driver: it registers `build.meeting_bot` and runs a jailed `claude` agent against the contract in `meeting_bot_spec.md` (the revised design). Review the retained result with `shepherd run changeset --latest --read <path>` and settle it with `shepherd run select <ref>`; edit the spec and re-run to iterate. `build_bot_task.py` must keep **both** monkeypatches — the `ClaudeHeadlessProvider.command_argv` `CLAUDE_CODE_TMPDIR` injection (jail, same as `agent_task.py`) **and** the `_WORKSPACE_RUNTIME_PROVIDER_TRANSPORTS.claude` budget override (see the budget gotcha below). The first build died at exactly 240s before the budget patch existed.

**Decisions (user-revised from the original design doc):**
- **Voice:** py-cord **pinned to the DAVE-patched unmerged branch** `git+https://github.com/Pycord-Development/pycord.git@fix/voice-rec-2` (PR #3159). Released 2.8.1 voice receive is broken under Discord's DAVE end-to-end encryption (issue #3139) — don't "fix" this pin back to a release, capture will silently stop. A watchdog warns if no PCM frames arrive after recording starts.
- **STT:** local `mlx-whisper` (`mlx-community/whisper-large-v3-turbo`, `language="th"`) — free, Metal-accelerated on Apple Silicon; shares the GPU with the gateway stack.
- **Summarization:** the user's own gateway via the Anthropic-compatible API — `anthropic` SDK with `base_url=https://gateway.9arm.co`, `auth_token` (`ANTHROPIC_AUTH_TOKEN`), model `qwen3.6-35b-a3b`. **Do not append `/v1` to the base URL** (the SDK adds it). qwen's structured-output reliability is mitigated by a never-raising JSON→markdown parse.

**Bot layout** (`meeting_bot/`): `config` (`.env` → frozen dataclass + `doctor()`), `audio` (48k-stereo-int16 → 16k-mono-f32 numpy resample), `chunker` (silence segmentation), `sink` (py-cord per-user capture), `transcriber` (single worker thread → mlx-whisper), `transcript` (speaker/time events), `summary_parse` (robust parse), `summarizer` (gateway call), `poster` (embed), `bot` (`MeetingBot`, auto-trigger on voice-channel-empty), `main`/`__main__`. **Import rule:** `config`/`audio`/`chunker`/`transcript`/`summary_parse` import only stdlib + numpy at module scope, so the jailed agent can smoke-import them and `pytest` runs without Discord/MLX/anthropic.

**Lazy import pattern (`meeting_bot/__init__.py`).** The package uses `__getattr__`/`__dir__` to defer importing `Config` (from `config`) and `MeetingBot` (from `bot`) until they are first accessed. This keeps `import meeting_bot` lightweight — the heavy deps (`discord`, `mlx-whisper`, `anthropic`) are never loaded until the bot actually runs. The pure modules (`config`, `audio`, `chunker`, `transcript`, `summary_parse`) are importable with only numpy present, so `pytest` runs in a minimal environment.

**Build contract (`meeting_bot_spec.md`).** This is the authoritative architecture doc and module contract that `build_bot_task.py` feeds to the jailed claude agent. Every class, function signature, import rule, and acceptance criterion is defined here. After `shepherd run select`, verify the 6 acceptance criteria from the spec: (1) `compileall` exits 0, (2) pure modules import with only numpy, (3) `--help`/`--doctor` work without connecting, (4) all documented signatures exist and `.env.example` matches `config.py`, (5) no secrets in any file, (6) `.env.example` keys match `config.py` exactly.

**Commands:**
- `python3 build_bot_task.py` — run the Shepherd build (jailed claude agent → retained proposal).
- `python3 quickstart_demo.py` — deterministic lane (no LLM, no jail) to verify Shepherd mechanics; prints a JSON summary of a retained note output.
- `python3 -m meeting_bot --doctor` — config/import/gateway-probe readiness check, exits non-zero on failures.
- `python3 -m meeting_bot` — run the bot (needs a populated `.env`).
- `pytest -q` — pure-logic suites (no Discord/network).

**Running the bot (verified on this machine, 2026-08):** Python 3.14.6 (homebrew), deps installed into user site-packages via `pip install -r requirements.txt` (py-cord resolves to `2.8.1.dev91+g326b72acc` — the pin works), `brew install opus` done (libopus at `/opt/homebrew/lib/libopus.dylib`). `numpy` is pinned `>=2.3.2` but pip settles it to **2.4.6** because `numba` (pulled in via mlx-whisper's chain) caps it at `<2.5` — the requirements comment documents this; don't force 2.5 back in. Before running live: copy `.env.example` → `.env`, fill `DISCORD_TOKEN`/`GUILD_ID`/`VOICE_CHANNEL_ID`/`TARGET_CHANNEL_ID`/`ANTHROPIC_AUTH_TOKEN`, and enable the **Server Members** privileged intent in the Developer Portal. First whisper run downloads ~1.6 GB into `~/.cache/huggingface`.

## Common commands

Both demos are safe to run: outputs are **retained**, nothing is written to your working tree unless you explicitly settle a run.

- `python3 agent_task.py` — the real-agent lane. Registers `quickstart.write_program`, runs the `claude` provider under `placement="jail"`, retains a generated `donut.py`.
- `python3 quickstart_demo.py` — the deterministic lane. Runs the `static` provider (no LLM, no jail) and prints a JSON summary of a retained note output.
- `shepherd doctor claude` — readiness check (claude on PATH + authenticated) before the claude lane. `agent_task.py` also pre-flights this itself and exits with a `not ready — …` message if claude is missing/unauth'd; that early exit is expected behavior, not a bug.

Inspect and settle retained runs (`--latest` works wherever a `<ref>` like `run-…` is accepted):

- `shepherd run list` / `shepherd run show --latest` / `shepherd run trace --latest --events`
- `shepherd run changeset --latest --read <path>` — preview a retained file; pipe to `python3 -` to run the generated program.
- `shepherd run select <ref>` — apply a retained run's output onto the working tree; `shepherd run discard <ref>` — drop it.

Workspace setup: `shepherd init` (already done — creates `.vcscore/`), `shepherd demo` re-emits these demo scripts, `shepherd package` manages extension packages.

## Architecture

**Task model.** A task is a Python function whose *signature is the permission surface*: type-annotated parameters (e.g. `repo: sp.GitRepo`, `output_path: str`) declare what the sandboxed agent may access or write, and the docstring is the contract the agent implements — the body is intentionally empty. Register on a workspace via `workspace.tasks.register(task, task_id="…")` or the `@sp.task` decorator.

**Run lifecycle.** `with sp.open(".") as workspace:` binds the workspace; `workspace.run(task, repo=workspace.git_repo(), …)` executes the task in an isolated fork and returns a run whose `.output()` exposes the retained result (`.changeset()`, `.read_text()`, `.release()`). Runs never write your working files — output lives in the retained store until `shepherd run select`.

**Providers.** `runtime={"provider": "static"}` is a deterministic no-LLM provider that works with `placement="advisory"`. `runtime={"provider": "claude"}` shells out to the headless `claude` CLI inside the OS jail and *requires* `placement="jail"` (advisory is rejected for it). `jail` wraps the run in a macOS Seatbelt sandbox; `advisory` skips OS-level enforcement.

**The jail's writable root (critical).** Under `jail`, the run's only writable path is its ephemeral clone dir — a `vcs-core-overlay/…/clones/<run-scope>` tree under the system temp dir, **not this repo**. `agent_task.py` exists around this constraint: the `claude` CLI unconditionally `mkdir`s `/tmp/claude-{uid}` at startup, which the jail refuses (EPERM). The file's header comment and its `ClaudeHeadlessProvider.command_argv` monkeypatch inject `CLAUDE_CODE_TMPDIR=<working_path>/.claude-scratch` into Shepherd's env redirect so that mkdir lands inside the writable root. Don't simplify that monkeypatch away, and don't "fix" the EPERM by pointing `CLAUDE_CODE_TMPDIR` at this repo's path — the repo is outside the jail.

**.vcscore/.** The workspace's internal object store (analogous to `.git`), created by `shepherd init`. It is untracked runtime state, not source. A `.gitignore` now covers `.vcscore/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, and `.env` (bot secrets). The repo still has no commits — before the first commit, `git status` should now show only source files.

## Gotchas

- **Session-reuse masking.** A `claude` spawned from inside a running Claude Code session inherits `CLAUDE_CODE_SESSION_ID` and reuses the parent's existing workspace, skipping the startup mkdir — so runs launched from inside this session can pass when a fresh terminal would fail. For a definitive check of anything jail/EPERM-related, run `python3 agent_task.py` from a plain Terminal.
- claude-provider runs need network + auth inside the jail; Shepherd sets `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, but real API traffic is allowed.

- **Build budget cap.** `ClaudeHeadlessProvider` defaults to `budget_seconds=240` (a SIGALRM hard stop) and the runtime envelope *reserves* that field, so you can't pass a bigger budget through `workspace.run(runtime=...)`. `build_bot_task.py` replaces the module-level `_WORKSPACE_RUNTIME_PROVIDER_TRANSPORTS.claude` transport (the same private seam the framework uses for test transports) with one that passes `budget_seconds` from `SHEPHERD_BUILD_BUDGET_SECONDS` (default 1800). Don't simplify this away — a whole-package build dies at 240s.

- **`shepherd run select` can record a settlement without writing files.** On 2026-08-08 the select of build run `run-280e9dd0b877` exited 1 after the authority settlement — the store ground advanced and the output was left consumed (`got 'selected'` on retry) but **nothing was written to the working tree**. Suspected trigger: stale `running` runs in the ledger (a killed build + an old quickstart) trip the materialization admission check. There's no CLI to close a stale running run (`discard` needs a retained `workspace` output; `repair` only reclaims orphaned ops). Recovery — materialize the settled candidate world manually:
  ```
  # output world oid is refs/vcscore/scopes/<scope> in worlds.git (or from `shepherd run outputs <ref> --json`); its meta/world.json snapshot has workspace.head = the applied tree oid
  W=.vcscore/world-vectors/substrates/workspace.git
  git -C $W archive <workspace.head> | tar -x -C /tmp/settle      # -> /tmp/settle/workspace/...
  cp -R /tmp/settle/workspace/{meeting_bot,tests} . && cp /tmp/settle/workspace/{requirements.txt,.env.example,conftest.py} .
  ```
  The candidate tree matched the `changeset --read` preview exactly. If `select` fails this way again, report it upstream (framework bug) and use the extraction above rather than re-running the build.
