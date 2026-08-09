"""Diagnostic monkeypatches and DAVE transition fix for py-cord.

Imported once at startup by ``bot.py`` — the patches activate at module
import time.

Patches 1-7: diagnostics + fixes.
The DAVE passthrough fix is applied in THREE places (patches 3, 4, 5):
  - Patch 4 (reinit_dave_session): enables passthrough on NEW session,
    BEFORE the MLS handshake completes.  This is the primary fix — it
    prevents the ratchet from drifting when dave.ready first flips True.
  - Patch 5 (execute_dave_transition): re-enables passthrough after
    every protocol transition (covers the 1→1 refresh case).
  - Patch 3 (decrypt_rtp): reactive fallback — enables passthrough on
    first DAVE decrypt failure (catches edge cases like epoch changes).

Patch 7 (UDPKeepAlive.run): fixes the macOS EISCONN busy-loop — the UDP
socket is already connect()ed by VoiceWebSocket.ready(), so send() is used
instead of sendto() with an explicit destination, with a bounded backoff on
the failure path.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAVE passthrough fix: seconds to keep passthrough enabled after a transition
# ---------------------------------------------------------------------------

_PASSTHROUGH_GRACE_SECONDS = 15

# ---------------------------------------------------------------------------
# Patch 1: VoiceWebSocket.load_secret_key
# ---------------------------------------------------------------------------

_orig_load_secret_key = None


async def _patched_load_secret_key(self, data: dict[str, Any]) -> None:
    global _orig_load_secret_key

    secret_key = data.get("secret_key")
    mode = data.get("mode")
    dave_ver = data.get("dave_protocol_version")

    if isinstance(secret_key, (list, tuple)):
        sk_bytes = bytes(secret_key)
        log.info(
            "DIAG load_secret_key: type=list len=%d first4=%s last4=%s",
            len(secret_key),
            sk_bytes[:4].hex(),
            sk_bytes[-4:].hex(),
        )
    elif isinstance(secret_key, str):
        log.info(
            "DIAG load_secret_key: type=str len=%d value_preview=%s...",
            len(secret_key),
            secret_key[:40],
        )
    elif isinstance(secret_key, bytes):
        log.info(
            "DIAG load_secret_key: type=bytes len=%d first4=%s last4=%s",
            len(secret_key),
            secret_key[:4].hex(),
            secret_key[-4:].hex(),
        )
    else:
        log.info(
            "DIAG load_secret_key: type=%s value=%r",
            type(secret_key).__name__,
            secret_key,
        )

    _load_sk_count = getattr(_patched_load_secret_key, "_count", 0) + 1
    _patched_load_secret_key._count = _load_sk_count

    log.info(
        "DIAG load_secret_key #%d: mode=%r dave_protocol_version=%r",
        _load_sk_count,
        mode,
        dave_ver,
    )

    # Call original.
    return await _orig_load_secret_key(self, data)

_patched_load_secret_key._count = 0


# ---------------------------------------------------------------------------
# Patch 2: PacketDecryptor._decrypt_rtp_aead_xchacha20_poly1305_rtpsize
# ---------------------------------------------------------------------------

_orig_aead_decrypt = None
_aead_failure_count = 0
_aead_success_logged = False
_MAX_FAILURE_LOGS = 5


def _patched_aead_decrypt(self, packet: Any) -> bytes:
    global _orig_aead_decrypt, _aead_failure_count, _aead_success_logged

    try:
        result = _orig_aead_decrypt(self, packet)
    except Exception as exc:
        _aead_failure_count += 1
        if _aead_failure_count <= _MAX_FAILURE_LOGS:
            # Gather packet state AFTER adjust_rtpsize has been called
            # (we can't call it here because the original already did,
            # so we read the modified state).
            pkt_data = getattr(packet, "data", b"")
            pkt_header = getattr(packet, "header", b"")
            pkt_nonce = getattr(packet, "nonce", b"")
            pkt_extended = getattr(packet, "extended", None)
            pkt_decrypted = getattr(packet, "decrypted_data", None)

            log.error(
                "DIAG AEAD decrypt FAIL #%d: mode=%r "
                "data_len=%d header=%s nonce=%s extended=%r "
                "decrypted_data_set=%r error=%s",
                _aead_failure_count,
                getattr(self, "mode", "?"),
                len(pkt_data),
                bytes(pkt_header).hex() if pkt_header else "N/A",
                bytes(pkt_nonce).hex() if pkt_nonce else "N/A",
                pkt_extended,
                pkt_decrypted is not None,
                exc,
            )
            # Log the box key snippet (first 4 bytes only for safety).
            try:
                box = getattr(self, "box", None)
                if box is not None:
                    # nacl.secret.Aead stores key internally; try to peek.
                    key_bytes = bytes(box._key) if hasattr(box, "_key") else b"?"
                    log.error(
                        "DIAG AEAD decrypt FAIL #%d: box_key_first4=%s",
                        _aead_failure_count,
                        key_bytes[:4].hex() if len(key_bytes) >= 4 else key_bytes.hex(),
                    )
            except Exception:
                pass

        if _aead_failure_count == _MAX_FAILURE_LOGS:
            log.warning(
                "DIAG AEAD: reached %d failure logs — suppressing further "
                "per-packet AEAD error logs (errors still occur)",
                _MAX_FAILURE_LOGS,
            )

        raise

    # Log the first successful decryption.
    if not _aead_success_logged:
        _aead_success_logged = True
        pkt_data = getattr(packet, "data", b"")
        log.info(
            "DIAG AEAD decrypt SUCCESS: data_len=%d result_len=%d "
            "failure_count_before=%d",
            len(pkt_data) if pkt_data else 0,
            len(result),
            _aead_failure_count,
        )

    return result


# ---------------------------------------------------------------------------
# Patch 3: PacketDecryptor.decrypt_rtp — DAVE layer diagnostics
# ---------------------------------------------------------------------------

_orig_decrypt_rtp = None
_dave_fail_count = 0
_dave_success_count = 0
_MAX_DAVE_LOGS = 5

# --- periodic failure-rate visibility ---------------------------------
#
# NOTE on what set_passthrough_mode actually fixes: it only prevents
# UnencryptedWhenPassthroughDisabled (raised when a plaintext packet
# arrives during a transition window). It does NOT fix
# NoValidCryptorFound (raised when a packet IS DAVE-encrypted but the
# local MLS session has no cryptor yet for that specific sender) —
# that failure mode can only resolve once davey finishes deriving the
# sender's key, which is outside application control. Capping the
# per-packet error log at _MAX_DAVE_LOGS hid just how much audio this
# was costing (a real run lost ~all audio despite these patches being
# active), so we log a periodic rate summary below in addition to the
# first few detailed failures.
_window_fail = 0
_window_success = 0
_last_summary_mono = 0.0
_SUMMARY_INTERVAL_SECONDS = 5.0


def _maybe_log_summary() -> None:
    global _window_fail, _window_success, _last_summary_mono

    now = time.monotonic()
    if _last_summary_mono == 0.0:
        _last_summary_mono = now
        return
    if now - _last_summary_mono < _SUMMARY_INTERVAL_SECONDS:
        return
    if _window_fail or _window_success:
        total = _window_fail + _window_success
        log.warning(
            "DIAG DAVE decrypt rate (last %.0fs): %d/%d packets failed "
            "(%.0f%%) — ongoing failures here are NOT fixed by the "
            "passthrough patches (see NoValidCryptorFound note)",
            _SUMMARY_INTERVAL_SECONDS,
            _window_fail,
            total,
            100.0 * _window_fail / total if total else 0.0,
        )
    _window_fail = 0
    _window_success = 0
    _last_summary_mono = now


def _patched_decrypt_rtp(self, packet: Any) -> bytes:
    global _orig_decrypt_rtp, _dave_fail_count, _dave_success_count
    global _window_fail, _window_success

    # Call the original first — it does NaCl decrypt + DAVE decrypt.
    result = _orig_decrypt_rtp(self, packet)

    # After call, inspect what happened.
    state = self.client._connection
    dave = state.dave_session

    if dave is not None:
        uid = state.ssrc_user_map.get(packet.ssrc) if hasattr(packet, "ssrc") else None
        decrypted = getattr(packet, "decrypted_data", None)

        if uid is not None and decrypted == b"":
            _dave_fail_count += 1
            _window_fail += 1
            if _dave_fail_count <= _MAX_DAVE_LOGS:
                dave_status = getattr(dave, "status", "unknown")
                dave_ready = getattr(dave, "ready", False)
                user_ids = getattr(dave, "get_user_ids", lambda: [])()
                log.error(
                    "DIAG DAVE decrypt FAIL #%d: ssrc=%s uid=%s "
                    "dave_ready=%r dave_status=%s known_users=%s",
                    _dave_fail_count,
                    getattr(packet, "ssrc", "?"),
                    uid,
                    dave_ready,
                    dave_status,
                    list(user_ids) if user_ids else "none",
                )
                if _dave_fail_count == _MAX_DAVE_LOGS:
                    log.warning(
                        "DIAG DAVE: reached %d failure logs — suppressing "
                        "per-packet detail, switching to periodic rate "
                        "summaries every %.0fs",
                        _MAX_DAVE_LOGS,
                        _SUMMARY_INTERVAL_SECONDS,
                    )

            # FIX: Reactively enable passthrough when DAVE decrypt starts
            # failing. This only helps the UnencryptedWhenPassthroughDisabled
            # case (catches transition windows the proactive patch in
            # execute_dave_transition might miss, e.g. epoch transitions
            # triggered by mls_prepare_epoch) — it is a no-op against
            # NoValidCryptorFound, which is the more common failure seen
            # in practice. Still harmless to call, so left in place.
            if _dave_fail_count == 1:
                try:
                    dave.set_passthrough_mode(True, _PASSTHROUGH_GRACE_SECONDS)
                    log.info(
                        "FIX: reactively enabled DAVE passthrough for %.0fs "
                        "after %d consecutive DAVE decrypt failures",
                        _PASSTHROUGH_GRACE_SECONDS,
                        _dave_fail_count,
                    )
                except Exception:
                    log.debug(
                        "FIX: reactive set_passthrough_mode failed (non-critical)",
                        exc_info=True,
                    )
        elif uid is not None and decrypted and decrypted != b"":
            _dave_success_count += 1
            _window_success += 1
            if _dave_success_count == 1:
                log.info(
                    "DIAG DAVE decrypt SUCCESS (first): ssrc=%s uid=%s "
                    "result_len=%d",
                    getattr(packet, "ssrc", "?"),
                    uid,
                    len(decrypted),
                )

        _maybe_log_summary()

    return result


# ---------------------------------------------------------------------------
# Patch 4: VoiceConnectionState.reinit_dave_session
# ---------------------------------------------------------------------------

_orig_reinit_dave = None


async def _patched_reinit_dave(self) -> None:
    global _orig_reinit_dave, _dave_fail_count, _dave_success_count
    global _window_fail, _window_success, _last_summary_mono

    proto = getattr(self, "dave_protocol_version", "?")
    old_session = getattr(self, "dave_session", None)
    old_ready = getattr(old_session, "ready", None) if old_session else None
    old_status = getattr(old_session, "status", None) if old_session else None

    log.info(
        "DIAG reinit_dave_session: proto=%s had_session=%r "
        "old_ready=%r old_status=%s channel_id=%s",
        proto,
        old_session is not None,
        old_ready,
        old_status,
        getattr(self, "channel_id", "?"),
    )

    await _orig_reinit_dave(self)

    # FIX: Enable passthrough on the NEW session immediately after creation.
    # Without this, the DAVE session's `ready` flag flips to True mid-stream
    # (MLS handshake completes) BEFORE Discord sends the DAVE transition op 22.
    # dave.decrypt() starts being called on transport-only-encrypted packets,
    # raises ValueError (UnencryptedWhenPassthroughDisabled), the MLS ratchet
    # drifts, and EVERY subsequent packet fails — even after the transition.
    new_session = getattr(self, "dave_session", None)
    new_proto = getattr(self, "dave_protocol_version", 0)
    if new_session is not None and new_proto and int(new_proto) > 0:
        try:
            new_session.set_passthrough_mode(True, _PASSTHROUGH_GRACE_SECONDS)
            log.info(
                "FIX: enabled DAVE passthrough for %.0fs on new session "
                "(proto=%s)",
                _PASSTHROUGH_GRACE_SECONDS,
                new_proto,
            )
        except Exception:
            log.debug(
                "FIX: set_passthrough_mode failed in reinit (non-critical)",
                exc_info=True,
            )

    # Reset failure/success counters so fresh diagnostics start each session.
    _dave_fail_count = 0
    _dave_success_count = 0
    _window_fail = 0
    _window_success = 0
    _last_summary_mono = 0.0


# ---------------------------------------------------------------------------
# Patch 5: VoiceConnectionState.execute_dave_transition
# ---------------------------------------------------------------------------

_orig_execute_dave = None


async def _patched_execute_dave(self, transition: int) -> None:
    global _orig_execute_dave

    old_proto = getattr(self, "dave_protocol_version", "?")
    pending = getattr(self, "dave_pending_transition", None)
    session = getattr(self, "dave_session", None)
    ready = getattr(session, "ready", None) if session else None

    log.info(
        "DIAG execute_dave_transition: transition=%s old_proto=%s "
        "pending=%s dave_ready_before=%r",
        transition,
        old_proto,
        pending,
        ready,
    )

    await _orig_execute_dave(self, transition)

    new_proto = getattr(self, "dave_protocol_version", "?")
    session_after = getattr(self, "dave_session", None)
    ready_after = getattr(session_after, "ready", None) if session_after else None
    status_after = getattr(session_after, "status", None) if session_after else None

    # FIX: Enable passthrough mode after every DAVE transition where
    # protocol_version > 0.  Discord sends a brief window of unencrypted
    # packets during transitions; without passthrough, dave.decrypt() raises
    # ValueError ("UnencryptedWhenPassthroughDisabled"), those packets are
    # dropped, and the MLS ratchet state drifts, breaking ALL subsequent
    # decryption.  A 15-second grace window covers the transition + epoch setup.
    if session_after is not None and new_proto and int(new_proto) > 0:
        try:
            session_after.set_passthrough_mode(True, _PASSTHROUGH_GRACE_SECONDS)
            log.info(
                "FIX: enabled DAVE passthrough for %.0fs after transition %s",
                _PASSTHROUGH_GRACE_SECONDS,
                transition,
            )
        except Exception:
            log.debug("FIX: set_passthrough_mode failed (non-critical)", exc_info=True)

    log.info(
        "DIAG execute_dave_transition AFTER: new_proto=%s "
        "dave_ready=%r dave_status=%s",
        new_proto,
        ready_after,
        status_after,
    )


# ---------------------------------------------------------------------------
# Patch 6: AudioReader.__init__ — log key at PacketDecryptor creation
# ---------------------------------------------------------------------------

_orig_audioreader_init = None
_reader_create_count = 0


def _patched_audioreader_init(self, sink, client, *, after=None, args=None, start=False):
    global _orig_audioreader_init, _reader_create_count
    _reader_create_count += 1

    sk = getattr(client, "secret_key", None)
    mode = getattr(client, "mode", None)
    if isinstance(sk, list):
        sk_bytes = bytes(sk)
        log.info(
            "DIAG AudioReader #%d created: mode=%r key_first4=%s key_last4=%s",
            _reader_create_count,
            mode,
            sk_bytes[:4].hex(),
            sk_bytes[-4:].hex(),
        )
    else:
        log.info(
            "DIAG AudioReader #%d created: mode=%r key_type=%s key=%r",
            _reader_create_count,
            mode,
            type(sk).__name__,
            sk,
        )

    return _orig_audioreader_init(self, sink, client, after=after, args=args, start=start)


# ---------------------------------------------------------------------------
# Patch 7: UDPKeepAlive.run — fix EISCONN busy-loop on macOS
# ---------------------------------------------------------------------------

_orig_udp_keep_alive_run = None
_KEEPALIVE_RETRY_WAIT_SECONDS = 1.0


def _patched_udp_keep_alive_run(self) -> None:
    global _orig_udp_keep_alive_run

    self.client.wait_until_connected()

    while not self._end_thread.is_set():
        vc = self.client

        try:
            packet = self.counter.to_bytes(8, "big")
        except OverflowError:
            self.counter = 0
            continue

        try:
            # The UDP socket was already connect()ed by VoiceWebSocket.ready()
            # (loop.sock_connect), so send() is correct — sendto() with an
            # explicit destination raises EISCONN (OSError 56) on macOS.
            vc._connection.socket.send(packet)
        except Exception as exc:
            log.debug(
                "Error while sending udp keep alive to socket %s at %s:%s",
                vc._connection.socket,
                vc._connection.endpoint_ip,
                vc._connection.voice_port,
                exc_info=exc,
            )
            vc.wait_until_connected()
            if vc.is_connected():
                # Bounded backoff so a persistent failure can't busy-loop /
                # log-flood the way the original sendto() path did. Event.wait
                # also stays responsive to stop().
                self._end_thread.wait(_KEEPALIVE_RETRY_WAIT_SECONDS)
                continue
            break
        else:
            self.counter += 1
            time.sleep(self.delay)


# ---------------------------------------------------------------------------
# Apply patches at import time
# ---------------------------------------------------------------------------


def _apply():
    global _orig_load_secret_key, _orig_aead_decrypt
    global _orig_decrypt_rtp, _orig_reinit_dave, _orig_execute_dave
    global _orig_audioreader_init, _orig_udp_keep_alive_run

    import discord.voice.gateway
    import discord.voice.receive.reader
    import discord.voice.state

    # Patch 1: load_secret_key
    _orig_load_secret_key = discord.voice.gateway.VoiceWebSocket.load_secret_key
    discord.voice.gateway.VoiceWebSocket.load_secret_key = _patched_load_secret_key

    # Patch 2: AEAD decrypt
    _orig_aead_decrypt = (
        discord.voice.receive.reader.PacketDecryptor
        ._decrypt_rtp_aead_xchacha20_poly1305_rtpsize
    )
    discord.voice.receive.reader.PacketDecryptor._decrypt_rtp_aead_xchacha20_poly1305_rtpsize = (
        _patched_aead_decrypt
    )

    # Patch 3: DAVE decrypt_rtp diagnostics
    _orig_decrypt_rtp = discord.voice.receive.reader.PacketDecryptor.decrypt_rtp
    discord.voice.receive.reader.PacketDecryptor.decrypt_rtp = _patched_decrypt_rtp

    # Patch 4: reinit_dave_session
    _orig_reinit_dave = discord.voice.state.VoiceConnectionState.reinit_dave_session
    discord.voice.state.VoiceConnectionState.reinit_dave_session = _patched_reinit_dave

    # Patch 5: execute_dave_transition
    _orig_execute_dave = discord.voice.state.VoiceConnectionState.execute_dave_transition
    discord.voice.state.VoiceConnectionState.execute_dave_transition = _patched_execute_dave

    # Patch 6: AudioReader init
    _orig_audioreader_init = discord.voice.receive.reader.AudioReader.__init__
    discord.voice.receive.reader.AudioReader.__init__ = _patched_audioreader_init

    # Patch 7: UDP keep-alive send() fix (EISCONN on macOS)
    _orig_udp_keep_alive_run = discord.voice.receive.reader.UDPKeepAlive.run
    discord.voice.receive.reader.UDPKeepAlive.run = _patched_udp_keep_alive_run

    log.info("DIAG: py-cord monkeypatches applied (7 patches)")


_apply()
