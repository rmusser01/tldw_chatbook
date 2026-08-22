"""task-15862: a wake turn's UI must stay fresh without user interaction.

PR 3a-2 Task 7's live pass proved the wake DELIVERY layer correct and
durable while the UI around a wake turn froze: the 0.2s transcript poll
is armed only by the user-driven send worker
(``_submit_console_native_draft``), and a wake turn enters through
``ConsoleFleetWakeCoordinator._deliver`` -> ``controller.submit_draft``
-- so nothing ever repainted the wake turn's streaming reply, its
terminal tab glyph, or the composer state. Observed live (Task 7,
findings 1a-1c): a stuck ``●`` on the woken session's tab for minutes; a
VIEWED session frozen mid-delivery with an empty reply row and the
misleading "finish provider setup" composer copy, healing instantly on a
session switch.

Under test here, all three ACs, against the REAL mounted ``ChatScreen``:

1. a wake turn completing in a non-viewed session flips that session's
   tab glyph off RUNNING at the terminal edge, with no interaction;
2. a wake turn's reply reaches the VIEWED session's rendered transcript
   with no session switch;
3. the composer's blocked-state copy during a wake turn names the wake,
   never provider setup (the observed lie: the queue presentation's
   "wait to be accepted" tooltip -- a chainless wake is never
   queue-accepted -- fell into ``build_console_disabled_reason``'s
   provider-setup fallback).

The 15664 pinned regression (no recurring idle repaint) is asserted
alongside: after the wake settles, the transcript poll must have stopped
itself again.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from Tests.Chat.test_console_fleet_wake import (
    _RecordingWakeGateway,
    _drain,
    _survivor,
    _terminal_subagent_run,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_panel import _AGENT_SECTION_SIZE
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


async def _settle(pilot, predicate, seconds: float = 8.0) -> bool:
    """Run the app loop until ``predicate()`` is true (or time out)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return bool(predicate())


def _drain_from_child_thread(wake, drain) -> None:
    """Deliver the drain from a plain thread, as production does.

    The bridge fan-out fires on the CHILD's daemon thread, so
    ``retry_soon``'s ``call_soon_threadsafe`` copies THAT thread's
    context -- which carries no Textual ``active_app``. The task-15862
    live diagnosis caught a transcript-poll timer created straight from
    that bare callback context dying on its very first tick (Textual's
    ``Timer._tick`` reads the ``active_app`` ContextVar, and an asyncio
    task inherits the context it was created in): "arm-poll" logged,
    zero beats, frozen transcript. This suite's first version injected
    the drain from the test coroutine -- whose context HAS active_app
    under ``run_test`` -- and passed against that broken arming. Every
    delivery-driving injection here must therefore come from a plain
    thread, so the screen's arming is exercised in the context
    production gives it.
    """
    thread = threading.Thread(target=lambda: wake.on_fleet_drained(drain))
    thread.start()
    thread.join(5)


async def _mounted_wake_rig(pilot, host, *, reply: str = "wake reply"):
    """Mounted Console + real bridge/runs DB + a swappable wake gateway."""
    from Tests.UI.test_destination_shells import _wait_for_selector

    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-session-surface")
    controller = console._ensure_console_chat_controller()
    bridge = console._ensure_console_agent_bridge()
    assert bridge is not None, (
        "harness must build the real bridge (chachanotes_db path wired)"
    )
    gateway = _RecordingWakeGateway(reply=reply)
    controller.provider_gateway = gateway
    store = console._ensure_console_chat_store()
    return console, controller, bridge, gateway, store


def _app_with_plain_provider_path(tmp_path):
    """Test app whose Console controller takes the plain-provider path, so
    the wake turn streams through the recording gateway double."""
    app = _build_test_app()
    console_cfg = app.app_config.setdefault("console", {})
    console_cfg["agent_runtime"] = False
    _attach_real_dbs(app, tmp_path)
    return app


def _tab_label(console, session_id: str) -> str:
    return str(console.query_one(f"#console-session-tab-{session_id}").label)


@pytest.mark.asyncio
async def test_wake_turn_in_a_nonviewed_session_flips_the_tab_glyph_off_running(
    tmp_path,
):
    """AC#2, the live finding 1a shape: the wake turn's ``●`` painted
    mid-turn (live: by the survivor tick's settle paint; here: by one
    explicit sync while the stream is gated) must flip off RUNNING at the
    terminal edge WITH NO USER INTERACTION -- live it sat stuck for
    minutes until the session was viewed."""
    app = _app_with_plain_provider_path(tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console, controller, bridge, gateway, store = await _mounted_wake_rig(
            pilot, host
        )
        viewed = store.ensure_session()
        target = store.create_session(
            title="Background research",
            settings=console._session._default_console_session_settings(),
        )
        store.switch_session(viewed.id)
        _parent, run_id = _terminal_subagent_run(bridge.runs_db, target.id)
        gate = asyncio.Event()
        gateway.stream_gate = gate

        _drain_from_child_thread(
            controller.fleet_wake,
            _drain(target.id, _survivor(run_id, session_id=target.id)),
        )
        assert await _settle(pilot, lambda: gateway.payloads), (
            "the wake turn never started streaming"
        )
        # The live pass's mid-turn paint (survivor-tick settle beat): the
        # woken session's tab shows RUNNING while the wake streams.
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        assert "●" in _tab_label(console, target.id), (
            "precondition: the streaming wake turn paints RUNNING on its tab"
        )

        gate.set()
        stamped = await _settle(
            pilot,
            lambda: bool(
                (bridge.runs_db.get_run(run_id) or {}).get("wake_delivered_at")
            ),
        )
        assert stamped, "the wake turn never completed/stamped its ledger row"

        # NO interaction from here on: the terminal edge itself must repaint.
        await pilot.pause(1.2)
        label = _tab_label(console, target.id)
        assert "●" not in label, (
            "task-15862: the wake turn ended but its tab glyph froze at "
            f"RUNNING with no user interaction: {label!r}"
        )
        assert "✓" in label, (
            "the settled wake turn's unvisited outcome must reach the tab "
            f"glyph without a session switch: {label!r}"
        )
        # 15664 AC#2 stays pinned: the poll must have stopped itself again.
        await _settle(pilot, lambda: console._console_transcript_sync_timer is None)
        assert console._console_transcript_sync_timer is None, (
            "the transcript poll must self-stop after the wake settles -- "
            "no recurring idle repaint (15664 AC#2)"
        )
        assert console._fleet._console_fleet_survivor_timer is None


@pytest.mark.asyncio
async def test_wake_reply_reaches_the_viewed_transcript_without_a_switch(
    tmp_path,
):
    """AC#1, the live finding 1b shape: a wake delivering into the VIEWED
    session froze mid-delivery -- empty assistant row for 4+ minutes while
    the full reply sat in the DB, healing only on a session switch. The
    rendered transcript must gain the wake's notice row and reply with no
    interaction at all."""
    app = _app_with_plain_provider_path(tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console, controller, bridge, gateway, store = await _mounted_wake_rig(
            pilot, host, reply="wake reply body"
        )
        session = store.ensure_session()
        _parent, run_id = _terminal_subagent_run(bridge.runs_db, session.id)

        _drain_from_child_thread(
            controller.fleet_wake,
            _drain(session.id, _survivor(run_id, session_id=session.id)),
        )
        stamped = await _settle(
            pilot,
            lambda: bool(
                (bridge.runs_db.get_run(run_id) or {}).get("wake_delivered_at")
            ),
        )
        assert stamped, "the wake turn never completed/stamped its ledger row"
        # The store has the reply the moment delivery returns; the defect
        # is the WIDGET never learning about it. No syncs, no switches.
        assert any(
            "wake reply body" in str(m.content)
            for m in store.messages_for_session(session.id)
        ), "precondition: the delivered reply is in the store"

        await pilot.pause(1.2)
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        synced = [str(m.content) for m in transcript._messages]
        assert any("wake reply body" in content for content in synced), (
            "task-15862: the wake turn's reply never reached the viewed "
            f"session's rendered transcript (widget rows: {len(synced)})"
        )
        await _settle(pilot, lambda: console._console_transcript_sync_timer is None)
        assert console._console_transcript_sync_timer is None, (
            "the transcript poll must self-stop after the wake settles -- "
            "no recurring idle repaint (15664 AC#2)"
        )


@pytest.mark.asyncio
async def test_composer_blocked_copy_names_the_wake_not_provider_setup(
    tmp_path,
):
    """AC#3, the live finding 1b copy lie: mid-wake the composer read
    "Send blocked — finish provider setup to continue" although provider
    setup was fine -- the queue presentation's not-yet-accepted tooltip (a
    chainless wake turn is never queue-accepted) fell into
    ``build_console_disabled_reason``'s provider-setup fallback. The
    blocked copy must name the actual reason: a sub-agent result being
    delivered."""
    app = _app_with_plain_provider_path(tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console, controller, bridge, gateway, store = await _mounted_wake_rig(
            pilot, host
        )
        # The live state under test: provider setup was FINE (the wake was
        # streaming through it) -- the lie was the queue tooltip falling
        # into the provider-setup fallback. Instance-level stub, the
        # workbench-contract suite's existing idiom.
        console._console_provider_blocker_copy = lambda: ""
        session = store.ensure_session()
        _parent, run_id = _terminal_subagent_run(bridge.runs_db, session.id)
        gate = asyncio.Event()
        gateway.stream_gate = gate

        _drain_from_child_thread(
            controller.fleet_wake,
            _drain(session.id, _survivor(run_id, session_id=session.id)),
        )
        assert await _settle(pilot, lambda: gateway.payloads), (
            "the wake turn never started streaming"
        )
        # One paint mid-wake (live: the last paint before the freeze).
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        reason = str(composer._send_disabled_reason or "")
        assert "provider setup" not in reason.lower(), (
            "task-15862 AC#3: mid-wake composer copy blamed provider setup "
            f"(the observed lie): {reason!r}"
        )
        assert "sub-agent" in reason.lower(), (
            "the blocked copy must name the actual blocker -- a sub-agent "
            f"result being delivered: {reason!r}"
        )
        from textual.widgets import Button

        tooltip = str(console.query_one("#console-send-message", Button).tooltip or "")
        assert "sub-agent" in tooltip.lower(), (
            "the send button's hover copy must name the wake too, not the "
            f"queue's not-yet-accepted line: {tooltip!r}"
        )

        gate.set()
        await _settle(
            pilot,
            lambda: bool(
                (bridge.runs_db.get_run(run_id) or {}).get("wake_delivered_at")
            ),
        )


@pytest.mark.asyncio
async def test_poll_survives_the_wake_scheduling_gap_then_stops_after(
    tmp_path,
):
    """The stop-guard race, pinned in isolation: the coordinator sets
    ``_delivering`` synchronously BEFORE its delivery task first runs, and
    a poll beat landing in that gap sees an idle viewed session and zero
    in-flight runs. Without the wake-delivery stop guard the poll would
    self-stop right there and the wake turn would stream unwatched -- the
    exact freeze the delivery hook exists to prevent. When the delivery
    ends, the poll must stop itself again (15664 AC#2)."""
    app = _app_with_plain_provider_path(tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console, controller, bridge, gateway, store = await _mounted_wake_rig(
            pilot, host
        )
        session = store.ensure_session()
        wake = controller.fleet_wake
        # The scheduling gap, held open: delivery claimed, task not yet
        # busy (no run state change, no in-flight run).
        wake._delivering = session.id
        try:
            console._start_console_transcript_sync_timer()
            await pilot.pause(0.7)
            assert console._console_transcript_sync_timer is not None, (
                "task-15862: a poll beat inside the wake scheduling gap "
                "(delivery claimed, turn not yet busy) must not self-stop"
            )
        finally:
            wake._delivering = None
        await _settle(pilot, lambda: console._console_transcript_sync_timer is None)
        assert console._console_transcript_sync_timer is None, (
            "with the delivery over and nothing busy the poll must stop "
            "itself (15664 AC#2: no recurring idle repaint)"
        )
