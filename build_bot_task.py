"""Shepherd build driver for the Meeting Summary Discord Bot.

Run from an initialized workspace (`shepherd init`; `shepherd doctor claude`
checks readiness). Registers the `build.meeting_bot` task and runs it with the
`claude` provider under `placement="jail"`, passing the contract in
`meeting_bot_spec.md` as `spec`. The agent's generated `meeting_bot/` tree is
kept as a retained output for you to review — nothing touches your files unless
you `shepherd run select` it.
"""

import os
import shutil
import sys
from pathlib import Path

from shepherd_dialect import claude_auth_status
from shepherd_dialect.providers import ClaudeHeadlessProvider
from shepherd_dialect.workspace_control import runtime_provider as _runtime_provider

import shepherd as sp

# Same jail constraint as agent_task.py: the claude CLI unconditionally mkdirs
# its per-user workspace base (<CLAUDE_CODE_TMPDIR or /tmp>/claude-{uid}) at
# startup, and the Shepherd jail only permits writes beneath the run's working
# path (an ephemeral clone dir, not this repo) — so the default /tmp/claude-501
# mkdir is refused (EPERM). Shepherd's env redirect (providers.py: command_argv)
# points HOME/CLAUDE_CONFIG_DIR/TMPDIR into <working_path>/.claude-scratch but
# misses CLAUDE_CODE_TMPDIR, so it is injected here. No new writable path is
# granted — this only points claude's housekeeping at the root the jail already
# allows. Do not simplify this away or point CLAUDE_CODE_TMPDIR at this repo.
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

# Second framework default that doesn't fit this task: the built-in Claude lane
# caps every run at `ClaudeHeadlessProvider.budget_seconds = 240` (a SIGALRM hard
# stop). That's fine for the quickstart demos but far too small for generating a
# whole bot package — the first build died at exactly 240s (`budget exceeded`).
# The runtime envelope reserves `budget_seconds`, so it can't be passed through
# `workspace.run(runtime=...)`. The provider is built by the module-level
# `_WORKSPACE_RUNTIME_PROVIDER_TRANSPORTS.claude` seam (the same private seam the
# framework uses to inject test transports), so we replace it with one that
# passes a generous budget. Override with SHEPHERD_BUILD_BUDGET_SECONDS.
_ORIG_CLAUDE_TRANSPORT = _runtime_provider._default_claude_transport


def _claude_transport_with_build_budget(invocation):
    budget = int(os.environ.get("SHEPHERD_BUILD_BUDGET_SECONDS", "1800"))
    return _runtime_provider.ClaudeHeadlessProvider(
        provider_id=invocation.provider_id,
        prompt=invocation.prompt,
        model=invocation.model_name,
        budget_seconds=budget,
    )


_runtime_provider._WORKSPACE_RUNTIME_PROVIDER_TRANSPORTS = (
    _runtime_provider._WorkspaceRuntimeProviderTransports(claude=_claude_transport_with_build_budget)
)

SPEC_PATH = Path(__file__).resolve().parent / "meeting_bot_spec.md"


# The signature is the permission surface; the docstring is the contract the
# sandboxed agent implements. The full spec is passed in as `spec`.
def build_meeting_bot(repo: sp.GitRepo, spec: str, output_path: str = "meeting_bot") -> None:
    """Create a complete, runnable Meeting Summary Discord Bot package in this repo.

    Read `spec` (a markdown architecture doc) and implement it as Python source.
    Produce EXACTLY this file tree at the repo root (in addition to your own
    output directory):

      meeting_bot/
        __init__.py  config.py  audio.py  chunker.py  sink.py
        transcriber.py  transcript.py  summary_parse.py  summarizer.py
        poster.py  bot.py  main.py  __main__.py
      tests/test_audio.py  tests/test_chunker.py  tests/test_transcript.py
      tests/test_summary_parse.py
      requirements.txt  .env.example

    Follow the spec's locked decisions exactly: py-cord pinned to the
    DAVE-patched branch (git+...@fix/voice-rec-2), local mlx-whisper STT
    (whisper-large-v3-turbo, language=th), and summarization via the user's
    Anthropic-compatible gateway (base_url without /v1, auth_token, model
    qwen3.6-35b-a3b). Honor the import rule: config/audio/chunker/transcript/
    summary_parse must import only stdlib + numpy at module scope; summarizer
    imports anthropic lazily. sink.write() must never call asyncio (router
    thread). The summary parser must never raise.

    ACCEPTANCE (verify inside this sandbox where possible; the driver verifies
    the rest after settlement):
    1. `python3 -m compileall -q meeting_bot tests` exits 0.
    2. These import with only numpy installed:
       `import meeting_bot.config, meeting_bot.audio, meeting_bot.chunker,
        meeting_bot.transcript, meeting_bot.summary_parse`
    3. `python3 -m meeting_bot --help` and `--doctor` work without connecting.
    4. Every class/function named in the spec exists with the documented
       signature. .env.example contains exactly the keys config.py reads.
    5. No tokens or secrets in any file.
    """


if __name__ == "__main__":
    if shutil.which("claude") is None:
        sys.exit("not ready — `claude` is not on PATH; run `shepherd doctor claude`")
    _auth = claude_auth_status()
    if not _auth.ok:
        # An expired/absent login is caught here rather than failing mid-run.
        sys.exit(f"not ready — {_auth.detail}")
    if not SPEC_PATH.exists():
        sys.exit(f"missing spec file: {SPEC_PATH}")

    with sp.open(".") as workspace:
        workspace.tasks.register(build_meeting_bot, task_id="build.meeting_bot")
        run = workspace.run(
            "build.meeting_bot",
            repo=workspace.git_repo(),
            spec=SPEC_PATH.read_text(),
            output_path="meeting_bot",
            placement="jail",
            runtime={"provider": "claude"},
        )
        output = run.output()
        changed = ", ".join(output.changeset().changed_paths)
        print(f"retained: {run.run_ref} wrote {changed} (nothing applied to your files)")
        print()
        print("preview the generated package straight from the retained output:")
        print("  shepherd run changeset --latest --read meeting_bot/bot.py")
        print()
        print("keep it, or not:")
        print(f"  shepherd run select {run.run_ref}")
        print(f"  shepherd run discard {run.run_ref}")
        print()
        print("after selecting: pip install -r requirements.txt, pytest -q,")
        print("python3 -m meeting_bot --doctor (needs .env), then run the bot.")
