"""Discord client: voice trigger + orchestration."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta

import discord

from .chunker import SilenceChunker
from .config import Config
from .poster import Poster
from .sink import MeetingSink
from .summarizer import EmptySummaryError, StalledGenerationError, Summarizer
from .summary_parse import parse_summary
from .thai_polish import ThaiPolisher
from .transcriber import Transcriber
from .transcript import Transcript
from .transcript_dump import dump_transcript
from . import _pycord_diag  # noqa: F401  # monkeypatches py-cord for DAVE diagnostics

log = logging.getLogger(__name__)

_WATCHDOG_SECONDS = 10.0
_WATCHDOG_POLL = 5.0
_WATCHDOG_WARN_INTERVAL = 60.0
_VOICE_CONNECT_MAX_RETRIES = 3
_VOICE_CONNECT_BACKOFFS = (2.0, 5.0, 10.0)
_VOICE_MONITOR_POLL = 5.0
_VOICE_MONITOR_MAX_FAILURES = 3
# Discard incoming PCM this long after a fresh voice (re)connect: DAVE takes
# ~5s to derive per-sender keys, and frames in that window fail
# NoValidCryptorFound. Small buffer over the observed ~5s.
_AUDIO_GRACE_SECONDS = 6.5

# Fixed multi-query set for RAG retrieval (one query per summary facet).
# Union of top-k per query, deduped and re-sorted chronologically, gives
# broad meeting coverage without needing an LLM to formulate queries.
_RAG_QUERIES = (
    "สรุปภาพรวมของการประชุมและบทสนทนาหลัก",
    "หัวข้อที่ถูกพูดถึงในการประชุม",
    "การตัดสินใจที่ที่ประชุมตกลงกันและเหตุผล",
    "สิ่งที่ต้องทำ ผู้รับผิดชอบ และกำหนดเวลา",
    "คำถามหรือประเด็นที่ยังไม่ได้ข้อสรุป",
)


def _rag_prompt(transcript: Transcript, cfg: Config) -> str:
    """Build the summarized-excerpt prompt via per-meeting RAG retrieval.

    Heavy work (loading the embedding model, embedding chunks/queries) —
    must run off the event loop (inside ``asyncio.to_thread``).
    """
    from .embedder import Embedder
    from .rag_store import VectorIndex, chunk_transcript, truncate_block

    chunks = chunk_transcript(
        transcript.events(),
        transcript.started_at,
        chunk_chars=cfg.rag_chunk_chars,
        overlap_chars=cfg.rag_overlap_chars,
    )
    if not chunks:
        raise ValueError("no transcript chunks to index")

    embedder = Embedder(cfg.embedding_model)
    index = VectorIndex()
    index.add(embedder.embed_texts([c.as_embedding_text() for c in chunks]), chunks)
    hits = index.query_multi(embedder.embed_texts(list(_RAG_QUERIES)), cfg.rag_top_k)
    if not hits:
        raise ValueError("retrieval returned no chunks")

    block = "\n\n".join(f"{chunk.header}\n{chunk.text}" for chunk, _ in hits)
    prompt = truncate_block(block, cfg.max_prompt_chars)
    log.info("rag: chunks=%d retrieved=%d prompt_chars=%d", len(chunks), len(hits), len(prompt))
    return prompt


def _build_prompt(transcript: Transcript, cfg: Config) -> str:
    """Gateway prompt: RAG retrieval when enabled, else the legacy full-
    transcript truncated render. Never raises — any RAG failure falls back
    to the legacy path so posting is never blocked."""
    if cfg.rag_enabled and not transcript.is_empty():
        try:
            return _rag_prompt(transcript, cfg)
        except Exception:  # noqa: BLE001
            log.warning(
                "rag retrieval failed — falling back to full-transcript truncation",
                exc_info=True,
            )
    return transcript.to_prompt_text(max_chars=cfg.max_prompt_chars)


def rejoin_allowed(manual_leave: bool, human_count: int) -> bool:
    """Auto-rejoin gate after a manual ``/leave``.

    While ``manual_leave`` is set, auto-start stays suppressed until the
    voice channel has been observed with zero humans (the caller clears the
    flag on that observation). Pure function so the transition table is
    unit-testable.
    """
    return not manual_leave or human_count == 0


class MeetingBot(discord.Bot):
    """Joins the target voice channel, transcribes live, and auto-posts a
    structured summary when the last human leaves — or on demand via the
    ``/leave`` slash command; ``/join`` brings it back manually."""

    def __init__(self, cfg: Config, *, intents=None):
        if intents is None:
            intents = discord.Intents.default()
            intents.guilds = True
            intents.voice_states = True
            intents.members = True  # privileged — enable in Developer Portal
        # discord.Bot subclasses Client and forwards kwargs (incl. intents)
        # to it, so run(token)/event handlers behave identically.
        super().__init__(intents=intents)
        self.cfg = cfg
        self._meeting_active = False
        self._meeting_lock = asyncio.Lock()
        self._voice_client: discord.VoiceClient | None = None
        self._sink: MeetingSink | None = None
        self._transcriber: Transcriber | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._voice_monitor_task: asyncio.Task | None = None
        self._last_watchdog_warn: float | None = None
        self._meeting_started_mono: float | None = None
        self._meeting_started_wall: datetime | None = None
        self._seen_humans: set[int] = set()
        self._manual_leave = False  # set by /leave; lifted when channel observed empty
        self._commands_registered = False
        self._register_slash_commands()

    def _register_slash_commands(self) -> None:
        """Register guild-scoped /leave and /join exactly once per instance.

        Guild-scoped commands sync instantly (no ~1 hr global propagation).
        The guard flag keeps gateway reconnects from re-adding duplicates;
        __init__ runs once anyway, this just makes idempotency explicit.
        """
        if self._commands_registered:
            return
        self._commands_registered = True

        @self.slash_command(
            guild_ids=[self.cfg.guild_id],
            name="leave",
            description="สรุปการประชุมตอนนี้ โพสต์สรุป แล้วออกจากห้องเสียง",
        )
        async def _leave(ctx) -> None:
            await self._handle_leave(ctx)

        @self.slash_command(
            guild_ids=[self.cfg.guild_id],
            name="join",
            description="ให้บอทเข้าห้องเสียงเพื่อบันทึกการประชุม",
        )
        async def _join(ctx) -> None:
            await self._handle_join(ctx)

    def _target_voice_channel(self):
        guild = self.get_guild(self.cfg.guild_id)
        if guild is None:
            return None
        return guild.get_channel(self.cfg.voice_channel_id)

    async def _handle_leave(self, ctx) -> None:
        """/leave: summarize now, post, disconnect — anyone may invoke."""
        try:
            if not self._meeting_active:
                # No meeting running: disconnect if somehow still connected,
                # then tell the user there is nothing to summarize.
                vc = self._voice_client
                if vc is not None and vc.is_connected():
                    try:
                        await vc.disconnect()
                    except Exception:  # noqa: BLE001
                        log.exception("disconnect failed (non-fatal)")
                    self._voice_client = None
                await ctx.respond("ไม่มีการประชุมอยู่ตอนนี้", ephemeral=True)
                return

            # Interaction tokens expire in 15 min and finalize takes minutes:
            # ack immediately, then run finalize as a background task.
            await ctx.respond(
                "⏳ กำลังสรุปการประชุม… บอทจะออกจากห้องเสียงเมื่อสรุปเสร็จ",
                ephemeral=True,
            )
            channel = self._target_voice_channel()
            if channel is None:
                log.error("voice channel %s not found for /leave", self.cfg.voice_channel_id)
                return
            asyncio.create_task(self._finalize(channel, force_disconnect=True))
        except Exception:  # noqa: BLE001 — never raise past the handler
            log.exception("/leave handler failed")
            try:
                await ctx.respond("⚠️ เกิดข้อผิดพลาด ลองใช้คำสั่งอีกครั้งภายหลัง", ephemeral=True)
            except Exception:  # noqa: BLE001
                pass

    async def _handle_join(self, ctx) -> None:
        """/join: manual way back in after /leave — anyone may invoke."""
        try:
            channel = self._target_voice_channel()
            if channel is None:
                await ctx.respond("ไม่พบห้องเสียงที่ตั้งค่าไว้", ephemeral=True)
                return
            vc = self._voice_client
            if self._meeting_active or (vc is not None and vc.is_connected()):
                await ctx.respond("บอทอยู่ในห้องเสียงอยู่แล้ว", ephemeral=True)
                return
            self._manual_leave = False
            humans = [m for m in channel.members if not m.bot]
            if humans:
                await ctx.respond(
                    f"🔌 กำลังเข้าร่วม {channel.name} เพื่อบันทึกการประชุม",
                    ephemeral=True,
                )
                asyncio.create_task(self._start_meeting(channel))
            else:
                await ctx.respond(
                    "ไม่มีผู้ใช้อยู่ในห้องเสียง — บอทจะเริ่มบันทึกอัตโนมัติเมื่อมีคนเข้าห้อง",
                    ephemeral=True,
                )
        except Exception:  # noqa: BLE001
            log.exception("/join handler failed")
            try:
                await ctx.respond("⚠️ เกิดข้อผิดพลาด ลองใช้คำสั่งอีกครั้งภายหลัง", ephemeral=True)
            except Exception:  # noqa: BLE001
                pass

    # -- event handlers --------------------------------------------------

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id)
        guild = self.get_guild(self.cfg.guild_id)
        if guild is None:
            log.error("Guild %s not found", self.cfg.guild_id)
            return
        channel = guild.get_channel(self.cfg.voice_channel_id)
        if channel is None:
            log.error("Voice channel %s not found", self.cfg.voice_channel_id)
            return
        humans = [m for m in channel.members if not m.bot]
        if humans:
            await self._start_meeting(channel)

    async def on_voice_state_update(self, member, before, after) -> None:
        if member.bot:
            return  # only humans drive start/finalize

        target = self.cfg.voice_channel_id
        was_in_target = before.channel is not None and before.channel.id == target
        is_in_target = after.channel is not None and after.channel.id == target
        if not (was_in_target or is_in_target):
            return

        guild = self.get_guild(self.cfg.guild_id)
        if guild is None:
            return
        channel = guild.get_channel(target)
        if channel is None:
            return

        if is_in_target:
            self._seen_humans.add(member.id)

        humans = [m for m in channel.members if not m.bot]
        human_count = len(humans)

        # Rejoin suppression: after a manual /leave, block auto-start until
        # the voice channel has been observed empty once — then lift the
        # flag and resume normal auto-rejoin behavior.
        if not rejoin_allowed(self._manual_leave, human_count):
            return
        if self._manual_leave and human_count == 0:
            self._manual_leave = False
            log.info("voice channel observed empty — /leave suppression lifted")

        if humans and not self._meeting_active:
            await self._start_meeting(channel)
        elif not humans and self._meeting_active:
            await self._finalize(channel)
        elif was_in_target and not is_in_target and self._meeting_active:
            # A human left but the meeting continues: flush that speaker's
            # trailing audio into the transcript.
            sink = self._sink
            if sink is not None:
                await asyncio.to_thread(sink.flush_user, member.id)

    # -- orchestration ---------------------------------------------------

    async def _start_meeting(self, channel) -> None:
        async with self._meeting_lock:
            if self._meeting_active:
                return
            humans = [m for m in channel.members if not m.bot]
            if not humans:
                return

            self._meeting_active = True
            self._meeting_started_mono = time.monotonic()
            self._meeting_started_wall = datetime.now()
            self._seen_humans = {m.id for m in humans}
            self._last_watchdog_warn = None

            transcriber = Transcriber(
                model=self.cfg.whisper_model,
                language=self.cfg.whisper_language,
            )
            transcriber.start()
            self._transcriber = transcriber

            names = {m.id: m.display_name for m in humans}

            def chunker_factory() -> SilenceChunker:
                return SilenceChunker(
                    sample_rate=16000,
                    frame_ms=20,
                    threshold=self.cfg.silence_threshold,
                    silence_seconds=self.cfg.silence_seconds,
                    min_chunk_seconds=self.cfg.min_chunk_seconds,
                    max_chunk_seconds=self.cfg.max_chunk_seconds,
                )

            self._sink = MeetingSink(transcriber, chunker_factory, names)

            # Reuse a still-connected VoiceClient (a human rejoined mid-
            # finalize); otherwise connect fresh with retry + backoff.
            vc = self._voice_client
            if vc is None or not vc.is_connected():
                vc = await self._connect_voice_with_retry(channel)
                if vc is None:
                    # All retries exhausted — _connect_voice_with_retry
                    # already logged and posted an error note.
                    self._meeting_active = False
                    self._transcriber = None
                    self._sink = None
                    return
                self._voice_client = vc

            # Log DAVE encryption status after successful connection.
            self._log_dave_diag(vc)

            # Correct self-mute/deaf sequence: change_voice_state returns None
            # and does NOT create the VoiceClient.
            await channel.guild.change_voice_state(
                channel=channel, self_mute=True, self_deaf=True
            )
            try:
                vc.start_recording(self._sink)
            except Exception:  # noqa: BLE001
                log.exception("failed to start recording")
                self._meeting_active = False
                return

            log.info("Meeting started in %s (%d humans)", channel, len(humans))
            self._watchdog_task = asyncio.create_task(self._watchdog(channel))
            self._voice_monitor_task = asyncio.create_task(
                self._voice_monitor(channel)
            )

    async def _finalize(self, channel, *, force_disconnect: bool = False) -> None:
        async with self._meeting_lock:
            if force_disconnect:
                # Manual /leave: suppress auto-rejoin until the channel is
                # observed empty once (see rejoin_allowed).
                self._manual_leave = True
                if not self._meeting_active:
                    # A finalize already ran (or never started) — just make
                    # sure the voice connection is torn down.
                    vc = self._voice_client
                    if vc is not None and vc.is_connected():
                        try:
                            await vc.disconnect()
                        except Exception:  # noqa: BLE001
                            log.exception("disconnect failed (non-fatal)")
                        self._voice_client = None
                        self._sink = None
                        self._transcriber = None
                    return
            elif not self._meeting_active:
                return
            self._meeting_active = False

            log.info("Meeting finalizing in %s", channel)

            if self._watchdog_task is not None:
                self._watchdog_task.cancel()
                self._watchdog_task = None
            if self._voice_monitor_task is not None:
                self._voice_monitor_task.cancel()
                self._voice_monitor_task = None

            vc = self._voice_client
            transcriber = self._transcriber
            started_mono = self._meeting_started_mono or time.monotonic()
            started_wall = self._meeting_started_wall or datetime.now()

            if vc is not None and vc.is_connected():
                try:
                    vc.stop_recording()
                except Exception:  # noqa: BLE001
                    log.exception("stop_recording failed (non-fatal)")

            # Flush all chunkers so trailing buffered speech is emitted
            # before the transcriber drains.
            sink = self._sink
            if sink is not None:
                await asyncio.to_thread(sink.flush_all)

            # Drive finalize directly rather than relying on stop_recording's
            # once_done callback (the pinned branch has a truthiness bug).
            events = []
            if transcriber is not None:

                def _drain():
                    transcriber.stop(flush=True)
                    return transcriber.drain(timeout=5.0)

                try:
                    events = await asyncio.to_thread(_drain)
                except Exception:  # noqa: BLE001
                    log.exception("failed to drain transcriptions")

            transcript = Transcript(started_at=started_mono)
            for event in events:
                transcript.add(event)

            target = self.get_channel(self.cfg.target_channel_id)
            duration = timedelta(seconds=max(0.0, time.monotonic() - started_mono))
            guild = channel.guild
            meeting_title = f"{guild.name} · {channel.name}" if guild else str(channel)
            meta = {
                "started_at": started_wall,
                "duration": duration,
                "member_count": len(self._seen_humans),
                "meeting_title": meeting_title,
            }

            # Debug: log what events we got
            evts = transcript.events()
            log.info("finalize: collected %d transcript events", len(evts))
            for evt in evts:
                log.info("  event: speaker=%s text=%r", evt.speaker, evt.text)

            if target is None:
                log.error("target channel %s not found", self.cfg.target_channel_id)
            elif transcript.is_empty():
                diag = sink.diagnostics() if sink else {}
                frames = diag.get("frame_count", 0)
                ever = diag.get("ever_received_frame", False)
                if not ever:
                    note = (
                        "🗒️ No audio frames received from Discord — the voice "
                        "connection may have a DAVE encryption issue. "
                        "(py-cord #3139 / DAVE protocol mismatch)"
                    )
                else:
                    note = (
                        f"🗒️ No speech detected in the meeting room — "
                        f"the meeting ended without a recording. "
                        f"({frames} audio frames received but no speech recognized)"
                    )
                log.info("no speech detected — posting note: %s", note)
                try:
                    await target.send(note)
                except Exception:  # noqa: BLE001
                    log.exception("failed to post no-speech note")
            else:
                # Persist the raw transcript BEFORE summarizing so every
                # downstream failure mode (524, timeout, loop, parse,
                # poster) leaves a recoverable file behind (KB-sized
                # write — synchronous is fine, no extra thread hop).
                dump_path = dump_transcript(
                    transcript,
                    meeting_title=meeting_title,
                    started_wall=started_wall,
                    duration_seconds=int(duration.total_seconds()),
                    member_count=len(self._seen_humans),
                )
                if dump_path:
                    log.info("transcript saved to %s", dump_path)

                def _recovery_hint(path: str | None) -> str:
                    """Thai one-liner appended to failure notes when the
                    raw transcript was saved, quoting a ready-made
                    manual_summary.py command with exact recovery flags."""
                    if not path:
                        return ""
                    return (
                        f"\n💾 บันทึกถอดความดิบไว้ที่ {path} — เรียกคืนสรุปด้วย "
                        f"`python3 tools/manual_summary.py \"{path}\" "
                        f'--title "{meeting_title}" '
                        f'--started-at "{started_wall:%Y-%m-%d %H:%M:%S}" '
                        f"--duration-seconds {int(duration.total_seconds())} "
                        f"--members {len(self._seen_humans)}`"
                    )

                prompt_chars = {"value": 0}
                try:
                    def _summarize():
                        # Prompt construction (RAG may load a ~2 GB embedding
                        # model) and the anthropic client bootstrap
                        # (~0.5-2s first call) both run off the loop so the
                        # gateway heartbeat never stalls.
                        prompt_text = _build_prompt(transcript, self.cfg)
                        prompt_chars["value"] = len(prompt_text)
                        log.info("summarizer prompt length: %d chars", prompt_chars["value"])
                        summarizer = Summarizer(self.cfg)
                        return summarizer.summarize(prompt_text)

                    summary_text = await asyncio.to_thread(_summarize)
                    log.info("summarizer raw response (first 1000 chars): %s", summary_text[:1000])
                    summary = parse_summary(summary_text)
                    log.info("parsed summary: topics=%d decisions=%d action_items=%d",
                             len(summary.topics), len(summary.decisions), len(summary.action_items))

                    # Polish Thai prose (kien-thai skill, Register 6)
                    if self.cfg.polish_enabled:
                        try:
                            def _polish():
                                polisher = ThaiPolisher(
                                    base_url=self.cfg.polish_base_url,
                                    auth_token=self.cfg.polish_api_key,
                                    model=self.cfg.polish_model,
                                    max_passes=self.cfg.polish_max_passes,
                                    timeout_seconds=self.cfg.polish_timeout_seconds,
                                )
                                result = polisher.polish(summary)
                                return result, getattr(polisher, "last_stats", None)

                            polished, polish_stats = await asyncio.to_thread(_polish)
                            summary = polished
                            log.info(
                                "thai_polish: %s (passes=%s, overview_len=%d)",
                                (polish_stats or {}).get("outcome", "done"),
                                (polish_stats or {}).get("passes", "?"),
                                len(summary.overview),
                            )
                        except Exception:  # noqa: BLE001 — polish failure is non-fatal
                            log.exception("thai_polish failed — posting unpolished summary")

                    poster = Poster(self.cfg)
                    await poster.post(target, summary, meta=meta)
                except EmptySummaryError:
                    log.exception(
                        "gateway returned no text for the summary — posting a "
                        "failure note instead of an empty summary"
                    )
                    note = (
                        "⚠️ การสร้างสรุปการประชุมล้มเหลว: โมเดลไม่ได้ตอบกลับเนื้อหาใด ๆ "
                        "(ใช้โทเค็นทั้งหมดไปกับการคิดคำนวณ) "
                        "ลองเพิ่ม SUMMARY_MAX_TOKENS ในไฟล์ .env"
                    )
                    note += _recovery_hint(dump_path)
                    try:
                        await target.send(note)
                    except Exception:  # noqa: BLE001
                        log.exception("failed to post empty-summary failure note")
                except StalledGenerationError:
                    # Must sit BEFORE the generic `except Exception` below:
                    # StalledGenerationError is a RuntimeError subclass and
                    # would otherwise be swallowed into the log-only branch.
                    log.exception(
                        "summarizer output detected as a repetition loop — "
                        "posting a failure note instead of an empty summary"
                    )
                    note = (
                        "⚠️ การสร้างสรุปการประชุมล้มเหลว: โมเดลสร้างข้อความซ้ำซ้อน "
                        "(ตรวจพบ repetition loop) ลองเพิ่มค่า REPETITION_WINDOW_CHARS / "
                        "REPETITION_MIN_REPEATS ในไฟล์ .env หรือตรวจสอบ Gateway"
                    )
                    note += _recovery_hint(dump_path)
                    try:
                        await target.send(note)
                    except Exception:  # noqa: BLE001
                        log.exception("failed to post stall/loop failure note")
                except Exception as exc:  # noqa: BLE001
                    exc_name = type(exc).__name__
                    if "Timeout" in exc_name or "timeout" in exc_name.lower():
                        log.exception(
                            "summarizer timed out (prompt was %d chars)", prompt_chars["value"],
                        )
                    else:
                        log.exception("failed to summarize/post meeting summary")
                    note = (
                        f"⚠️ การสร้างสรุปการประชุมล้มเหลว ({exc_name}): "
                        "gateway ไม่ตอบกลับสำเร็จภายในเวลาที่กำหนด "
                        "(เช่น Cloudflare 524 origin timeout) — ดูรายละเอียดใน log ของ bot"
                    )
                    note += _recovery_hint(dump_path)
                    try:
                        await target.send(note)
                    except Exception:  # noqa: BLE001
                        log.exception("failed to post summarize-failure note")

            # Race guard: re-check humans before disconnecting. A member
            # joining mid-finalize must not double-post — keep the connection
            # so a queued _start_meeting can reuse it. A forced /leave skips
            # this branch entirely: the bot always tears down and disconnects,
            # and /join is the explicit way back in.
            if not force_disconnect:
                humans_now = [m for m in channel.members if not m.bot]
                if humans_now:
                    log.info(
                        "humans present again — keeping voice connection for continued meeting"
                    )
                    self._sink = None
                    self._transcriber = None
                    return

            if vc is not None and vc.is_connected():
                try:
                    await vc.disconnect()
                except Exception:  # noqa: BLE001
                    log.exception("disconnect failed (non-fatal)")

            self._voice_client = None
            self._sink = None
            self._transcriber = None
            self._seen_humans = set()

    async def _watchdog(self, channel) -> None:
        """Warn if no PCM frames arrive after start_recording (py-cord #3139)."""
        while self._meeting_active:
            await asyncio.sleep(_WATCHDOG_POLL)
            if not self._meeting_active:
                return
            sink = self._sink
            if sink is None:
                continue
            idle = time.monotonic() - sink.last_frame_time
            if idle > _WATCHDOG_SECONDS:
                now = time.monotonic()
                if (
                    self._last_watchdog_warn is None
                    or now - self._last_watchdog_warn >= _WATCHDOG_WARN_INTERVAL
                ):
                    log.warning(
                        "No PCM frames received for %.0fs after start_recording. "
                        "Discord voice receive is likely broken — py-cord issue "
                        "#3139. requirements.txt must pin py-cord to the "
                        "DAVE-patched branch: "
                        "git+https://github.com/Pycord-Development/pycord.git@"
                        "326b72acc8d1d952ac002fe07ca65581cf5952bc",
                        idle,
                    )
                    self._last_watchdog_warn = now

    async def _connect_voice_with_retry(self, channel):
        """Connect to voice with retry and backoff on failure.

        py-cord's internal retry reuses the same voice server token on every
        attempt.  When Discord closes with code 4006 ("session no longer
        valid"), a fresh ``channel.connect()`` is needed to get a new token
        from the main gateway.

        Returns the VoiceClient on success, or None if all retries exhausted.
        """
        for attempt in range(_VOICE_CONNECT_MAX_RETRIES):
            if self.is_closed():
                # Bot is shutting down (e.g. Ctrl+C) — the main gateway
                # websocket is already closing, so any further connect
                # attempt will only fail noisily with
                # ClientConnectionResetError. Bail out quietly instead.
                log.debug("voice connect aborted — client is closing")
                return None
            try:
                vc = await channel.connect()
                log.info(
                    "voice connect succeeded (attempt %d/%d)",
                    attempt + 1,
                    _VOICE_CONNECT_MAX_RETRIES,
                )
                self._arm_audio_grace()
                return vc
            except Exception as exc:
                log.warning(
                    "voice connect attempt %d/%d failed: %s: %s",
                    attempt + 1,
                    _VOICE_CONNECT_MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                )
                # Disconnect any lingering client so the next attempt triggers
                # a fresh VOICE_SERVER_UPDATE / token from the main gateway.
                stale = self._voice_client
                if stale is not None:
                    try:
                        await stale.disconnect()
                    except Exception:
                        pass
                    self._voice_client = None

                if attempt < _VOICE_CONNECT_MAX_RETRIES - 1:
                    delay = _VOICE_CONNECT_BACKOFFS[attempt]
                    log.info("retrying voice connect in %.1fs...", delay)
                    await asyncio.sleep(delay)

        log.error(
            "voice connect failed after %d attempts — giving up",
            _VOICE_CONNECT_MAX_RETRIES,
        )
        # Post a note to the target channel so the user knows the bot tried.
        try:
            target = self.get_channel(self.cfg.target_channel_id)
            if target is not None:
                await target.send(
                    "⚠️ Bot attempted to join the voice channel but the "
                    "connection was rejected by Discord (error 4006). "
                    "This is usually temporary — the bot will retry when "
                    "someone joins or leaves the voice channel."
                )
        except Exception:
            log.exception("failed to post voice-connect-error note")
        return None

    def _arm_audio_grace(self) -> None:
        """Discard incoming PCM for _AUDIO_GRACE_SECONDS after a fresh connect.

        DAVE takes a few seconds to derive per-sender keys after a (re)connect,
        and frames in that window fail NoValidCryptorFound. The sink drops them
        so garbage never reaches the chunker/transcriber.
        """
        sink = self._sink
        if sink is None:
            return
        sink.set_grace_until(time.monotonic() + _AUDIO_GRACE_SECONDS)
        log.info(
            "audio grace period started — discarding PCM for %.1fs",
            _AUDIO_GRACE_SECONDS,
        )

    @staticmethod
    def _log_dave_diag(vc) -> None:
        """Log DAVE encryption status after a successful voice connection."""
        try:
            ws = getattr(vc, "ws", None)
            if ws is None:
                log.info("DAVE: voice WS not available for inspection")
                return
            dave_ver = getattr(ws, "dave_protocol_version", None)
            log.info("DAVE protocol version negotiated: %s", dave_ver)
            session = getattr(ws, "dave_session", None)
            if session is not None:
                status = getattr(session, "status", "unknown")
                log.info("DAVE session active: status=%s", status)
            else:
                log.info("DAVE session: not established (passthrough)")
        except Exception:
            log.debug("Could not read DAVE state (non-critical)", exc_info=True)

    async def _voice_monitor(self, channel) -> None:
        """Reconnect the voice client if it drops mid-meeting.

        Polls ``vc.is_connected()`` every _VOICE_MONITOR_POLL seconds.  On
        disconnect, attempts a fresh ``channel.connect()`` and re-starts
        recording into the existing sink.
        """
        failures = 0
        while self._meeting_active:
            await asyncio.sleep(_VOICE_MONITOR_POLL)
            if not self._meeting_active:
                return
            if self.is_closed():
                # Bot is shutting down — don't attempt a reconnect against a
                # gateway websocket that's already closing (only produces
                # confusing ClientConnectionResetError/TimeoutError noise).
                return
            vc = self._voice_client
            if vc is not None and vc.is_connected():
                failures = 0  # reset on healthy poll
                continue
            if failures >= _VOICE_MONITOR_MAX_FAILURES:
                if failures == _VOICE_MONITOR_MAX_FAILURES:
                    log.error(
                        "voice reconnect failed after %d attempts — "
                        "giving up (watchdog will warn on missing frames)",
                        _VOICE_MONITOR_MAX_FAILURES,
                    )
                    failures += 1  # silence further logs
                continue
            log.warning(
                "voice client disconnected mid-meeting — "
                "reconnect attempt %d/%d",
                failures + 1,
                _VOICE_MONITOR_MAX_FAILURES,
            )
            try:
                # Tear down any stale client before reconnecting.
                if vc is not None:
                    try:
                        await vc.disconnect()
                    except Exception:
                        pass
                    self._voice_client = None
                vc = await channel.connect()
                self._voice_client = vc
                self._arm_audio_grace()
                await channel.guild.change_voice_state(
                    channel=channel, self_mute=True, self_deaf=True,
                )
                sink = self._sink
                if sink is not None:
                    vc.start_recording(sink)
                log.info("voice reconnected successfully")
                failures = 0
            except Exception:
                log.exception("voice reconnect attempt failed")
                failures += 1
                # Clean up on failure so next poll starts fresh.
                if self._voice_client is not None:
                    try:
                        await self._voice_client.disconnect()
                    except Exception:
                        pass
                    self._voice_client = None
