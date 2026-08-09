"""Your first agent task: a function with no body, implemented by a Claude agent.

Run from an initialized workspace (`shepherd init`; `shepherd doctor claude`
checks readiness). The agent's work is kept as a retained output for you to
review — nothing touches your files unless you `shepherd run select` it.
"""

import shutil
import sys
from pathlib import Path

from shepherd_dialect import claude_auth_status
from shepherd_dialect.providers import ClaudeHeadlessProvider

import shepherd as sp

# The claude CLI unconditionally mkdirs its per-user workspace base
# (<CLAUDE_CODE_TMPDIR or /tmp>/claude-{uid}) at startup, and the Shepherd jail
# only permits writes beneath the run's working path (an ephemeral clone dir, not
# this repo) — so the default /tmp/claude-501 mkdir is refused (EPERM) before the
# agent can start. Shepherd's env redirect (providers.py: command_argv) already
# points HOME/CLAUDE_CONFIG_DIR/TMPDIR into <working_path>/.claude-scratch
# (pre-created before launch, scrubbed after); it just misses CLAUDE_CODE_TMPDIR.
# The working path is only known inside the run, so the missing var is injected
# where it is: wrapping the provider's command_argv. No new writable path is
# granted — this only points claude's housekeeping at the root the jail already
# allows.
_ORIG_COMMAND_ARGV = ClaudeHeadlessProvider.command_argv


def _command_argv_with_claude_tmpdir(self, working_path, cli, prompt=None):
    argv = _ORIG_COMMAND_ARGV(self, working_path, cli, prompt)
    try:
        env_index = argv.index("/usr/bin/env")
    except ValueError:
        return argv
    argv.insert(env_index + 1, f"CLAUDE_CODE_TMPDIR={Path(working_path) / self._SCRATCH}")
    return argv


ClaudeHeadlessProvider.command_argv = _command_argv_with_claude_tmpdir

# The ask. Change it and re-run — the contract below stays the same.
PROMPT = """# Meeting Summary Discord Bot — Design Doc

## Overview
A Discord bot that joins a voice channel, transcribes the meeting live in Thai,
and automatically posts a structured summary to a text channel once the
meeting ends.

## Decisions

### D1 — Live voice capture, not pasted transcripts
The bot joins the voice channel and transcribes as the meeting happens.

### D2 — Auto-trigger on voice channel empty
Summarization + posting fires automatically when the last person leaves the
voice channel. No manual command needed to kick it off.

### D3 — Claude API for summarization
Chosen over the local Thai LLM setup (qwen3.6-35b-a3b via the existing
gateway) to avoid resource contention on the MacBook, and because Claude
follows structured-output instructions (topics/decisions/action items) more
reliably for a final artifact people will actually read.
**Trade-off:** transcript text leaves the device.

### D4 — Cloud speech-to-text, not local Whisper
`faster-whisper` has no Metal/MPS backend on Apple Silicon (CPU-only via
CTranslate2), which is risky for real-time transcription while the same
machine runs the AI gateway stack. A cloud Thai STT API (e.g. Google Cloud
Speech-to-Text, `th-TH`) offloads that load. Per-user audio streams from
Discord give speaker separation regardless of which STT is used.
**Trade-off:** recurring per-minute cost, one more external dependency.

### D5 — Python + py-cord
Chosen for voice-receive support (per-user PCM audio streams) and
straightforward local audio/STT plumbing.

### D6 — Runs on the existing MacBook
Alongside the current AI gateway/agent setup — no separate server for now.

## Defaults assumed (flag if you want these different)
- **Summary format:** structured post — Topics Discussed / Decisions /
  Action Items (with owner if mentioned)
- **Target channel:** one fixed channel ID per server, set in config
- **Audio chunking:** silence-based segmentation per speaker stream, sent to
  STT as each chunk closes
- **Secrets:** Discord bot token, Claude API key, STT API key all via `.env`
  (never committed)

## Architecture

```
Discord voice channel
  → py-cord voice receive sink (per-user PCM streams)
  → silence-based chunking per speaker
  → STT (Thai) per chunk → text with speaker label
  → accumulate into a running transcript
  → on voice channel empty:
      → full transcript → Claude API (summarize, Thai in/out)
      → post structured summary to target text channel
```

## Open items for later
- Exact cloud STT provider (Google vs Azure vs a Thai-specific ASR API)
- Per-server config for target channel (vs hardcoded)
- Handling STT failures / partial transcripts gracefully"""


# The signature is the permission surface: the grant on `repo` is what lets the
# agent write the bound repository (see "Permissions" in the README).
def write_program(repo: sp.GitRepo, prompt: str, output_path: str = "program.py") -> None:
    """Write a small, self-contained Python program that does what `prompt` asks.

    Save it to output_path. It must run with plain `python3`, read no input,
    and finish on its own within about ten seconds.
    """


if shutil.which("claude") is None:
    sys.exit("not ready — `claude` is not on PATH; run `shepherd doctor claude`")
_auth = claude_auth_status()
if not _auth.ok:
    # An expired/absent login is caught here rather than failing mid-run.
    sys.exit(f"not ready — {_auth.detail}")

with sp.open(".") as workspace:
    workspace.tasks.register(write_program, task_id="quickstart.write_program")
    run = workspace.run(
        "quickstart.write_program",
        repo=workspace.git_repo(),
        prompt=PROMPT,
        output_path="donut.py",
        placement="jail",
        runtime={"provider": "claude"},
    )
    output = run.output()
    changed = ", ".join(output.changeset().changed_paths)
    print(f"retained: {run.run_ref} wrote {changed} (nothing applied to your files)")
    print()
    print("run the agent's program straight from the retained output:")
    print("  shepherd run changeset --latest --read donut.py | python3 -")
    print()
    print("keep it, or not:")
    print(f"  shepherd run select {run.run_ref}")
    print(f"  shepherd run discard {run.run_ref}")
