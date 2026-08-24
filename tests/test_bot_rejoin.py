"""Tests for the /leave rejoin-suppression gate (pure logic, no Discord).

Covers ``rejoin_allowed`` transitions: fresh meeting -> forced leave blocks
rejoin despite humans present -> channel observed empty lifts suppression ->
normal auto-rejoin resumes.
"""

import pytest

from meeting_bot.bot import rejoin_allowed


class TestRejoinAllowed:
    def test_normal_auto_rejoin_allowed(self):
        assert rejoin_allowed(manual_leave=False, human_count=3) is True

    def test_manual_leave_blocks_rejoin_despite_humans(self):
        assert rejoin_allowed(manual_leave=True, human_count=2) is False

    def test_empty_channel_observation_permits_gate_clear(self):
        # The caller clears manual_leave when it observes zero humans; the
        # gate itself must not block that observation.
        assert rejoin_allowed(manual_leave=True, human_count=0) is True

    def test_transition_fresh_meeting_to_forced_leave(self):
        # Before /leave: humans present, auto behavior normal.
        assert rejoin_allowed(False, 4) is True
        # After /leave while humans remain: suppressed.
        assert rejoin_allowed(True, 4) is False

    def test_transition_empties_then_cleared_then_normal(self):
        forced = True
        # Someone joins while suppression active -> still blocked.
        assert rejoin_allowed(forced, 1) is False
        # Channel empties -> caller clears the flag.
        assert rejoin_allowed(forced, 0) is True
        forced = False
        # Normal auto-rejoin resumes for later joins.
        assert rejoin_allowed(forced, 1) is True
        assert rejoin_allowed(forced, 5) is True

    def test_zero_humans_without_suppression_trivially_allowed(self):
        assert rejoin_allowed(False, 0) is True


class TestGateContract:
    """The exact boolean contract bot.on_voice_state_update relies on."""

    @pytest.mark.parametrize("human_count", [0, 1, 2, 10])
    def test_no_suppression_always_true(self, human_count):
        assert rejoin_allowed(False, human_count) is True

    def test_suppression_only_lifted_by_empty_channel(self):
        for n in range(1, 6):
            assert rejoin_allowed(True, n) is False
        assert rejoin_allowed(True, 0) is True
