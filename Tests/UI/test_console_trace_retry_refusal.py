"""TASK-34100.3 review G3V2-03: a refused Retry capture stays a trace pause.

With trace capture on, the third send of a saved Kimi chat stops before the
provider with "Trace capture blocked" (TASK-34100.3.1 owns that refusal). The
card's first action, Retry capture, met the same refusal and then showed the
false "Response delivery status is unknown ... Retry anyway may send a
duplicate request" card, with "Send blocked" in the composer, although the
card had just said the provider was not contacted.

The pause records that no call boundary was ever built (the planner refused
before reservation), so nothing of this send can have reached the provider.
A retry of that shape is refused again; it must leave the send paused at the
same card, with Send without capture still working.

Rebased onto dev (2026-10-19): dev's tasks-32951/32953 (d591a4c101) classify
a surface refusal before any boundary as ``not_established``, so the gateway
consumes that first-call proof, the retry re-enters the send, meets the same
refusal and re-pauses with dev's "Trace capture could not start" copy. That
made this branch's own re-pause helper unreachable, so it was removed; this
pilot now pins dev's behaviour for the saved-Kimi third turn.

Pass 3 (2026-11-12): turn 3 of a saved Kimi chat now sends normally (the
planner admits an append in each domain). The pause is reproduced here with
a request whose recorded history really changed: from turn 3 on, the replay
drops every saved reasoning checkpoint, so an active continuation the trace
recorded disappears. That is still refused (ADR-337 amendment 2026-11-12,
TASK-34100.3.1 AC#2), and the Retry capture / Cancel send pins below run on
that real refusal.

Both tests are regression pins of dev's d591a4c101 (tasks-32951/32953) and
pass on base 26b2e677dd; they are not RED guards of this branch.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_blocked_send_recovery import _frame_rows, _until
from Tests.UI.test_console_moonshot_reasoning_reply import (
    TRACE_BLOCKED,
    _mounted_console,
    _reply,
    _send,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Chat import (
    console_chat_controller,
    console_history_budget,
    console_provider_gateway,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

pytestmark = pytest.mark.bootstrap_profile

DELIVERY_UNKNOWN = ("delivery status is unknown", "Send blocked", "Retry anyway")
#: The trace card's status line after a refused retry (dev's re-pause copy).
REFUSED_RETRY = "Trace capture could not start"


def _drop_replay_from_turn_three(monkeypatch) -> None:
    """From turn 3 on, replay no saved reasoning: a really changed history.

    Turn 2 recorded one active continuation; dropping it removes an entry the
    trace recorded, which no admission rule accepts. Every module that
    builds the replay reads ``provider_continuation_owner_groups`` by name.
    """

    for module in (
        console_chat_controller,
        console_provider_gateway,
        console_history_budget,
        agent_service,
    ):
        original = module.provider_continuation_owner_groups

        def owner_groups(items, *args, _original=original, **kwargs):
            groups = _original(items, *args, **kwargs)
            return () if len(groups) >= 2 else groups

        monkeypatch.setattr(module, "provider_continuation_owner_groups", owner_groups)


def _screen(h) -> str:
    return "\n".join(_frame_rows(h.host))


def _checkpoints(h) -> list[tuple]:
    return [
        tuple(row)
        for row in h.database.get_connection().execute(
            "SELECT state, checkpoint_revision, assistant_message_version "
            "FROM console_dispatch_checkpoints ORDER BY created_at"
        )
    ]


def _first_replies(count: int) -> list[str]:
    """The replies of the first ``count`` provider calls (each call's own)."""
    return [_reply(call) for call in range(1, count + 1)]


def _replies(h, session_id: str) -> list[str]:
    return [
        message.content
        for message in h.store.messages_for_session(session_id)
        if message.role is ConsoleMessageRole.ASSISTANT and message.content
    ]


@pytest.mark.asyncio
async def test_refused_retry_capture_keeps_the_trace_pause(
    tmp_path, monkeypatch
) -> None:
    _drop_replay_from_turn_three(monkeypatch)
    async with _mounted_console(
        tmp_path, monkeypatch, provider="moonshot", model="kimi-k3"
    ) as h:
        saved = h.store.ensure_session()
        for turn, text in enumerate(("Say hello", "And again"), 1):
            await _send(h, text)
            await _until(
                h.pilot,
                lambda turn=turn: _replies(h, saved.id) == _first_replies(turn),
                timeout=30.0,
                what=f"turn {turn} did not reply",
            )
        await _send(h, "Once more")
        await _until(
            h.pilot,
            lambda: TRACE_BLOCKED in _screen(h),
            timeout=30.0,
            what="a dropped saved replay did not stop at the trace pause",
        )
        assert len(h.calls) == 2
        paused_checkpoints = _checkpoints(h)

        await _wait_for_selector(h.console, h.pilot, "#console-trace-retry")
        await h.pilot.click("#console-trace-retry")
        await _until(
            h.pilot,
            lambda: (
                REFUSED_RETRY in _screen(h)
                or any(copy in _screen(h) for copy in DELIVERY_UNKNOWN)
            ),
            timeout=30.0,
            what="Retry capture did not settle",
        )

        screen = _screen(h)
        assert [copy for copy in DELIVERY_UNKNOWN if copy in screen] == [], screen
        assert TRACE_BLOCKED in screen
        # The card's Impact line is dev's (it reads the dispatch checkpoint the
        # agent path writes before reservation, so it says the contact is
        # unknown); this pin covers what the retry does, not that copy.
        recovery = h.store.dispatch_recovery_for_session(saved.id)
        assert recovery is None or not recovery.recovery_needed
        assert h.controller.trace_call_recovery_preparation() is not None
        assert len(h.calls) == 2, "a refused retry must not reach the provider"
        # The refused retry wrote no dispatch state of its own.
        assert _checkpoints(h) == paused_checkpoints

        # The pause is still the ordinary one: Send without capture answers.
        await h.pilot.click("#console-trace-send-without")
        await _until(
            h.pilot,
            lambda: _replies(h, saved.id) == _first_replies(3),
            timeout=30.0,
            what="Send without capture did not reply after the refused retry",
        )
        await h.console._sync_native_console_chat_ui()
        await h.pilot.pause(0.3)
        assert len(h.calls) == 3
        assert h.store.dispatch_recovery_for_session(saved.id) is None
        assert [copy for copy in DELIVERY_UNKNOWN if copy in _screen(h)] == []
        assert (
            h.database.get_connection()
            .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
            .fetchone()[0]
            == 0
        )


@pytest.mark.asyncio
async def test_cancel_send_settles_the_trace_pause(tmp_path, monkeypatch) -> None:
    """TASK-34100.3.1 AC#3: Cancel send settles the saved-Kimi third-turn pause.

    It used to refuse with "Trace reservation status is unavailable", because
    the pause recorded an "unknown" reservation. Dev's tasks-32951/32953
    (d591a4c101) record a surface refusal before any boundary as
    "not_established", which the cancel path accepts; this pins that.
    """
    _drop_replay_from_turn_three(monkeypatch)
    async with _mounted_console(
        tmp_path, monkeypatch, provider="moonshot", model="kimi-k3"
    ) as h:
        saved = h.store.ensure_session()
        for turn, text in enumerate(("Say hello", "And again"), 1):
            await _send(h, text)
            await _until(
                h.pilot,
                lambda turn=turn: _replies(h, saved.id) == _first_replies(turn),
                timeout=30.0,
                what=f"turn {turn} did not reply",
            )
        await _send(h, "Once more")
        await _until(
            h.pilot,
            lambda: TRACE_BLOCKED in _screen(h),
            timeout=30.0,
            what="a dropped saved replay did not stop at the trace pause",
        )

        await _wait_for_selector(h.console, h.pilot, "#console-trace-cancel")
        await h.pilot.click("#console-trace-cancel")
        await _until(
            h.pilot,
            lambda: h.controller.trace_call_recovery_preparation() is None,
            timeout=30.0,
            what="Cancel send did not settle the trace pause",
        )

        screen = _screen(h)
        assert "Trace reservation status is unavailable" not in screen
        assert TRACE_BLOCKED not in screen
        assert [copy for copy in DELIVERY_UNKNOWN if copy in screen] == [], screen
        assert h.controller.run_state_for(saved.id).visible_copy == (
            "Trace-captured send canceled."
        )
        assert h.store.dispatch_recovery_for_session(saved.id) is None
        assert _checkpoints(h) == []
        assert len(h.calls) == 2, "a canceled send must not reach the provider"
