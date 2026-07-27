"""Two sessions run concurrently; interrupt is session-scoped (spec §2)."""

from __future__ import annotations

import asyncio

import pytest

from textual.widgets import Static

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunMarker,
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


def _transcript_text(console) -> str:
    """Return the plain text of every Static descendant of the native
    transcript widget -- scoped to the transcript itself (not the whole
    screen), so mode-bar/rail text changes elsewhere can't make an
    equality assertion on this flaky.
    """
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    return " ".join(
        getattr(widget.renderable, "plain", str(widget.renderable))
        for widget in transcript.query(Static)
        if widget.display and hasattr(widget, "renderable")
    )


@pytest.mark.asyncio
async def test_two_sessions_run_concurrently_and_interrupt_is_scoped() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session_a = store.active_session_id
        session_b = controller.new_session().id

        release_a = asyncio.Event()
        release_b = asyncio.Event()

        async def fake_run(session_id, release):
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STREAMING, "run"),
                session_id=session_id,
            )
            await release.wait()
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"),
                session_id=session_id,
            )

        console.run_worker(
            fake_run(session_a, release_a),
            exclusive=True,
            group=f"console-run-{session_a}",
        )
        console.run_worker(
            fake_run(session_b, release_b),
            exclusive=True,
            group=f"console-run-{session_b}",
        )
        await pilot.pause(0.2)
        assert controller.in_flight_run_count() == 2  # truly concurrent

        # Cancelling A's group leaves B running.
        console.workers.cancel_group(console, f"console-run-{session_a}")
        release_b.set()
        await pilot.pause(0.3)
        assert controller.run_state_for(session_b).status is ConsoleRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_second_session_send_does_not_cancel_first_sessions_worker() -> None:
    """Regression guard for the shared-group bug this task fixes: before the
    per-session group name, two `run_worker(..., exclusive=True,
    group="console-run")` dispatches from DIFFERENT sessions shared Textual's
    exclusive group and silently cancelled each other. Dispatching two long-
    running fake workers under the real per-session group names must leave
    both alive simultaneously.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session_a = store.active_session_id
        session_b = controller.new_session().id

        started_a = asyncio.Event()
        started_b = asyncio.Event()
        never = asyncio.Event()  # never set -- both workers just hang until cancelled

        async def fake_run(session_id, started):
            started.set()
            await never.wait()

        worker_a = console.run_worker(
            fake_run(session_a, started_a),
            exclusive=True,
            group=f"console-run-{session_a}",
        )
        await started_a.wait()
        worker_b = console.run_worker(
            fake_run(session_b, started_b),
            exclusive=True,
            group=f"console-run-{session_b}",
        )
        await started_b.wait()
        await pilot.pause(0.1)

        # If the groups collided, starting worker_b would have cancelled
        # worker_a via Textual's exclusive-group semantics.
        assert worker_a.is_running
        assert worker_b.is_running

        console.workers.cancel_group(console, f"console-run-{session_a}")
        console.workers.cancel_group(console, f"console-run-{session_b}")
        await pilot.pause(0.1)


@pytest.mark.asyncio
async def test_stop_visible_action_only_cancels_viewed_session_background_completes() -> None:
    """Requirement 5c (Task 3b): two concurrent fake runs, mirroring the
    tests above, but this time each REGISTERS itself in the controller's
    per-session stream/cancel maps like a real run would. Pressing the
    visible Stop action (the Stop button's own handler) for the VIEWED
    session must cancel only that session's run; the background session's
    run is untouched and reaches COMPLETED on its own.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session_a = store.active_session_id
        session_b = controller.new_session().id
        store.switch_session(session_a)  # A is VIEWED (new_session() activates B)

        def _seed(session_id: str) -> str:
            store.append_message(
                session_id, role=ConsoleMessageRole.USER, content="hi"
            )
            assistant = store.append_message(
                session_id, role=ConsoleMessageRole.ASSISTANT, content=""
            )
            store.append_stream_chunk(assistant.id, "partial")
            return assistant.id

        assistant_a = _seed(session_a)
        assistant_b = _seed(session_b)

        started_a = asyncio.Event()
        started_b = asyncio.Event()
        never_release_a = asyncio.Event()  # A is only ever stopped, not released
        release_b = asyncio.Event()

        async def fake_run(session_id, assistant_id, started, release):
            task = asyncio.current_task()
            controller._active_stream_tasks[session_id] = task
            controller._active_assistant_message_ids[session_id] = assistant_id
            controller._set_run_state(
                ConsoleRunState(ConsoleRunStatus.STREAMING, "run"),
                session_id=session_id,
            )
            started.set()
            try:
                await release.wait()
                controller._set_run_state(
                    ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"),
                    session_id=session_id,
                )
            except asyncio.CancelledError:
                raise
            finally:
                if controller._active_stream_tasks.get(session_id) is task:
                    controller._active_stream_tasks.pop(session_id, None)
                    controller._active_assistant_message_ids.pop(session_id, None)

        console.run_worker(
            fake_run(session_a, assistant_a, started_a, never_release_a),
            exclusive=True,
            group=f"console-run-{session_a}",
        )
        console.run_worker(
            fake_run(session_b, assistant_b, started_b, release_b),
            exclusive=True,
            group=f"console-run-{session_b}",
        )
        await started_a.wait()
        await started_b.wait()
        await pilot.pause(0.1)
        assert controller.in_flight_run_count() == 2  # truly concurrent

        # A is the VIEWED session -- press the visible Stop action.
        assert store.active_session_id == session_a
        await console._stop_console_generation_from_visible_action()
        await pilot.pause(0.2)

        assert controller.run_state_for(session_a).status is ConsoleRunStatus.STOPPED
        # B is completely untouched by A's Stop.
        assert controller.run_state_for(session_b).status is ConsoleRunStatus.STREAMING

        release_b.set()
        await pilot.pause(0.2)
        assert controller.run_state_for(session_b).status is ConsoleRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_background_run_never_mutates_viewed_transcript() -> None:
    """Task 4 (background-write audit): the real seam the audit found is
    ``ChatScreen._append_native_console_system_message``, the function every
    slash-command handler funnels its system-row output through (the
    "candidates ... callbacks that append/patch transcript widgets" grep in
    the task brief -- ``_append_.*message``). Its previous behavior always
    resolved "the store's currently ACTIVE session" via
    ``store.ensure_session()``, even for callers (like ``/generate-image``'s
    failure path) that had already anchored themselves to a specific
    OWNING session before an ``asyncio.to_thread`` await let the user
    switch tabs. Driving the gated ``session_id=`` keyword directly (the
    same seam ``_console_command_generate_image`` now uses) proves a
    background session's append can never land on the viewed transcript --
    and, per the store-first design, that switching to the background
    session later reveals it with no separate replay mechanism needed.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        store = controller.store
        viewed = store.active_session_id
        background = controller.new_session().id
        store.switch_session(viewed)  # keep viewing the first session

        before = _transcript_text(console)
        # Drive the audited append path for the BACKGROUND session directly,
        # via the real gated seam (not the illustrative
        # `_apply_console_stream_delta` name from the brief -- there is no
        # such method; this IS the method the audit found and fixed).
        await console._append_native_console_system_message(
            "SHOULD-NOT-APPEAR", session_id=background
        )
        await pilot.pause(0.2)
        after = _transcript_text(console)
        assert "SHOULD-NOT-APPEAR" not in after
        assert before == after

        # Store-first: the row IS there for the background session -- no
        # deferred-replay mechanism needed, switching tabs just rebuilds the
        # view from the store.
        store.switch_session(background)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.2)
        assert "SHOULD-NOT-APPEAR" in _transcript_text(console)


@pytest.mark.asyncio
async def test_background_run_sensitivity_reverting_the_gate_fails() -> None:
    """Sensitivity check for the test above (TDD requirement): with the
    ``session_id`` gate temporarily bypassed -- reproducing the pre-fix
    behavior where the append always targeted whatever session is active
    RIGHT NOW instead of the caller-supplied owning session -- the SAME
    background-session append DOES leak onto the viewed transcript. This
    proves the prior test is actually exercising the gate rather than
    passing vacuously.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        store = controller.store
        viewed = store.active_session_id
        background = controller.new_session().id
        store.switch_session(viewed)

        # Reproduce the pre-fix seam: ignore the caller-supplied
        # `session_id` and always append to the currently-active session
        # (exactly what `_append_native_console_system_message` did before
        # Task 4's fix).
        async def leaky_append(message: str, *, session_id: str | None = None) -> None:
            active = store.ensure_session()
            store.append_message(
                active.id, role=ConsoleMessageRole.SYSTEM, content=message
            )
            await console._sync_native_console_chat_ui()

        console._append_native_console_system_message = leaky_append
        try:
            await console._append_native_console_system_message(
                "SHOULD-LEAK", session_id=background
            )
            await pilot.pause(0.2)
            assert "SHOULD-LEAK" in _transcript_text(console)
        finally:
            del console._append_native_console_system_message


@pytest.mark.asyncio
async def test_tab_and_sidebar_show_run_markers_and_fleet_line() -> None:
    """Task 8 (parallel-agents spec §6): a background session's live run
    marks BOTH the session tab and its sidebar conversation-browser row
    with the fleet glyph, and the Agent rail grows a fleet summary line --
    all sourced from Task 7's `run_marker_for`/`fleet_summary_counts`.

    Brief's illustrative `controller.store.set_active_session(...)` does not
    exist (`ConsoleChatStore` has no such method, matching the finding
    already noted by `Tests/Chat/test_console_run_markers.py`) -- the real
    API is `switch_session`, mirrored here on the `viewed`/`background`
    idiom `test_background_run_never_mutates_viewed_transcript` above uses.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        store = controller.store
        viewed = store.active_session_id
        background = controller.new_session().id
        store.switch_session(viewed)  # keep viewing the first session

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "bg"),
            session_id=background,
        )
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.3)

        text = _visible_text(console)
        assert "●" in text  # running glyph on tab/row
        assert "1 other agents running, 0 waiting for approval." in text


@pytest.mark.asyncio
async def test_transcript_sync_timer_keeps_ticking_for_background_run_while_viewed_idle() -> None:
    """Fix round 1 / Critical 1 regression (PA-T8 review): `_poll_transcript`
    used to self-stop off `controller.run_state` alone -- a read-only facade
    for the VIEWED session ONLY (parallel-agents spec §2). That froze the
    0.2s poll (and therefore tab glyphs / the Agent-rail fleet line, both
    driven only by that poll's `_sync_native_console_chat_ui()` call) the
    instant the viewed tab went idle, even with a DIFFERENT session still
    streaming. The prior test in this file calls `_sync_native_console_
    chat_ui()` manually and structurally cannot catch this -- everything
    below relies on the timer ticking on its own; there is no manual sync
    call anywhere in this test after the timer starts.

    Reproduces the real ordering: the background session is active at the
    moment its (fake) send starts the timer -- exactly what
    `_submit_console_native_draft` does for whichever session is active at
    dispatch -- and only THEN does the user switch away to the idle
    `viewed` session, matching the reviewer's live repro.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        store = controller.store
        viewed = store.active_session_id
        background = controller.new_session().id  # background is active here

        # Same start call `_submit_console_native_draft` makes as its first
        # action, for whichever session is active at dispatch (background).
        console._start_console_transcript_sync_timer()
        assert console._console_transcript_sync_timer is not None
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "bg"),
            session_id=background,
        )

        store.switch_session(viewed)  # user switches away; viewed is idle

        # Advance several 0.2s poll ticks. No manual `_sync_native_console_
        # chat_ui()` call anywhere below -- only the timer can produce this.
        await pilot.pause(1.0)

        assert console._console_transcript_sync_timer is not None
        text = _visible_text(console)
        assert "●" in text
        assert "1 other agents running, 0 waiting for approval." in text

        # The background run finishes -- the fixed stop condition (viewed
        # idle AND nothing anywhere in flight) now correctly fires.
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"),
            session_id=background,
        )
        await pilot.pause(1.0)
        assert console._console_transcript_sync_timer is None


@pytest.mark.asyncio
async def test_background_approval_parks_with_badge_and_single_toast() -> None:
    """Task 9 (parked background approvals, parallel-agents spec): a run in
    a NON-viewed session that needs approval must not steal the mounted
    approval card out from under whatever the user is currently looking
    at -- it parks (fleet badge via `set_run_pending_approval` + one
    toast), and only mounts once the user actually visits it.

    Real interface names (the brief's own names were illustrative):
    - The approval card is `ChatApprovalCard(id="chat-approval-card")`
      (`Widgets/Chat_Widgets/chat_task_cards.py`), NOT
      `#console-approval-card` -- and it is a SINGLETON always present in
      the DOM (`ConsoleSessionSurface.compose` yields it once, not
      per-session), toggled via its own `.display` flag rather than
      mount/unmount. "Not mounted" is therefore verified as `display is
      False`, matching `ChatApprovalCard.set_batch`'s own visibility
      convention.
    - `ChatScreen._park_console_approval(session_id)` is the seam this
      task adds -- the UI-thread half of the park path (flag + toast),
      wired as `ConsoleChatController.park_pending_approval` and invoked
      via `call_from_thread` from `request_mcp_approvals`'s park branch.
      The controller's own `_parked_approval_payloads` map (populated by
      `request_mcp_approvals` before parking) is what `switch_session`
      later reads to mount the SAME payload through the existing
      `set_pending_approval` path -- seeded directly here to drive the
      seam without a live worker thread/round (mirrors how the sibling
      `test_tab_and_sidebar_show_run_markers_and_fleet_line` drives
      `_set_run_state` directly instead of a real streamed run).
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        store = controller.store
        viewed = store.active_session_id
        background = controller.new_session().id
        store.switch_session(viewed)  # keep viewing the first session

        notifications: list[str] = []
        app.notify = lambda message, **kwargs: notifications.append(str(message))

        # Seed the retained round payload `request_mcp_approvals` would
        # have stored before parking, then drive the UI-thread park seam
        # directly.
        controller._parked_approval_payloads[background] = {
            "calls": [
                {
                    "llm_name": "mcp__srv__tool",
                    "server_key": "local:srv",
                    "tool_name": "tool",
                    "server_label": "Srv",
                    "arguments": {},
                    "reason": "ask",
                    "options": ["approve_once", "deny"],
                }
            ],
            "timeout_seconds": 30.0,
        }
        console._park_console_approval(background)
        await pilot.pause(0.3)

        approval_card = console.query_one("#chat-approval-card")
        assert not approval_card.display  # parked: never mounted over the viewed tab
        approval_toasts = [n for n in notifications if "needs approval" in n]
        assert len(approval_toasts) == 1
        assert (
            controller.run_marker_for(background) is ConsoleRunMarker.NEEDS_APPROVAL
        )

        # Visiting mounts the card through the existing mount path.
        controller.switch_session(background)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.3)
        assert console.query_one("#chat-approval-card").display

        # A second visit-away-and-back (no new decision) re-mounts the SAME
        # card without a second toast -- card state derives from the run's
        # pending-approval state, not mounted-widget lifetime.
        controller.switch_session(viewed)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.1)
        assert not console.query_one("#chat-approval-card").display
        controller.switch_session(background)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.1)
        assert console.query_one("#chat-approval-card").display
        assert len(
            [n for n in notifications if "needs approval" in n]
        ) == 1


@pytest.mark.asyncio
async def test_background_completion_fires_single_toast() -> None:
    """Task 10 (background completion toasts, parallel-agents spec): a
    NON-viewed session's run finishing (COMPLETED) or failing (FAILED)
    fires exactly one toast -- the viewed session's own terminal
    transition is visible live in its transcript and gets none (spec's
    "the user is watching" rule). Real interface names, mirroring the
    Task 9 approval toast test above (the brief's illustrative
    ``store.set_active_session`` does not exist -- ``switch_session`` is
    the real activation API, on both ``store`` and ``controller``).
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        viewed = controller.store.active_session_id
        background = controller.new_session().id
        controller.store.switch_session(viewed)  # keep viewing the first session

        notifications: list[str] = []
        app.notify = lambda message, **kwargs: notifications.append(str(message))

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "bg"), session_id=background
        )
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=background
        )
        finished = [n for n in notifications if "finished" in n]
        assert len(finished) == 1

        # Re-setting the same terminal state must not double-toast.
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"), session_id=background
        )
        assert len([n for n in notifications if "finished" in n]) == 1


def _assert_widget_and_ancestors_displayed(widget) -> None:
    """Walk `widget`'s own ancestor chain and assert every node up to the
    screen is displayed and occupies real screen space.

    Fix round 2 (live-smoke finding): Textual's ``Widget.display`` is a
    PER-WIDGET property (``self.styles.display != "none"``), never
    ancestor-aggregated -- a widget can report ``display=True`` while an
    ancestor's own ``display=False`` hides it from the live surface
    entirely. `_visible_text` (used by the sibling fleet-line test above)
    collects text from every ``Static`` regardless of whether an ancestor
    section is mounted/displayed; that gap is exactly what let Task 8's
    original tests pass while the real TUI rendered nothing (the Agent
    rail section defaults collapsed, and nothing reopened it, so its
    body -- where the fleet Static lives -- stayed `display: none`
    permanently). This walks the real chain, the same bar a user's
    rendered terminal must clear.
    """
    node = widget
    while node is not None:
        assert node.display, (
            f"{getattr(node, 'id', None) or node!r} is not displayed "
            "(display=False) -- an ancestor collapse/display:none hides "
            "it from the live surface even though it is mounted"
        )
        node = node.parent
    assert widget.region.width > 0 and widget.region.height > 0, (
        f"{widget!r} has an empty rendered region {widget.region!r}"
    )


@pytest.mark.asyncio
async def test_fleet_summary_line_is_reachable_on_the_live_rendered_surface() -> None:
    """Fix round 2 (parallel-agents spec §6 live-smoke finding): the Agent
    rail section's persisted preference defaults COLLAPSED (`agent_open=
    False`) and nothing previously reopened it, so `#console-agent-fleet-
    summary` -- though always mounted -- lived inside a body whose
    `display` stayed `none` regardless of fleet state. A live reviewer
    scrolling the whole rail with a background session parked (fleet
    counts (0, 1)) found no Agent header/fleet line anywhere on screen.

    `test_tab_and_sidebar_show_run_markers_and_fleet_line` above already
    covers the TEXT/copy contract via `_visible_text`, which structurally
    cannot catch an ancestor-hidden widget (see `_assert_widget_and_
    ancestors_displayed`'s docstring) -- this test is the one that would
    have caught the live-smoke finding: same idle-viewer/busy-background
    setup, but it walks the actual mounted ancestor chain instead.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        store = controller.store
        viewed = store.active_session_id
        background = controller.new_session().id
        store.switch_session(viewed)  # viewed idle; background stays non-active

        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "bg"),
            session_id=background,
        )
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.3)

        fleet_summary = console.query_one("#console-agent-fleet-summary", Static)
        _assert_widget_and_ancestors_displayed(fleet_summary)
        assert (
            getattr(fleet_summary.renderable, "plain", str(fleet_summary.renderable))
            == "1 other agents running, 0 waiting for approval."
        )

        # Quiet fleet -- the section (and the line specifically) may
        # release back to the user's actual (collapsed) preference; the
        # line's own content must go empty either way (never a stale
        # count once nothing is left to report).
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "done"),
            session_id=background,
        )
        controller.mark_session_visited(background)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.3)
        fleet_summary = console.query_one("#console-agent-fleet-summary", Static)
        assert not fleet_summary.display or str(
            getattr(fleet_summary.renderable, "plain", fleet_summary.renderable)
        ) == ""


def _single_pending_call() -> MCPPendingCall:
    return MCPPendingCall(
        llm_name="mcp__srv__tool",
        server_key="local:srv",
        tool_name="tool",
        server_label="Srv",
        arguments={},
        reason="ask",
    )


@pytest.mark.asyncio
async def test_mounted_round_survives_switch_away_and_switch_back() -> None:
    """Final review CRITICAL 1: a round that MOUNTS immediately (its
    session was the active/viewed one when `request_mcp_approvals`
    started -- i.e. NEVER parked) must still be recoverable after the
    user switches away and back. Pre-fix, `_parked_approval_payloads` was
    populated ONLY inside the parked branch, so `switch_session`'s
    mount-from-retained-payload lookup found nothing for a round that had
    never been parked -- the session showed a NEEDS_APPROVAL badge with no
    card, unrecoverable short of the round's own 120s timeout. Asserted
    through the REAL widget (`#chat-approval-card.display`) via the real
    `switch_session` re-derive path, not a direct `resolve_pending_
    approval` call standing in for it.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        # `_ensure_console_chat_controller` wires `controller.app =
        # self.app_instance` -- the wrapped `TldwCli` instance, which
        # `ConsoleHarness` never itself `run_test()`s (only `host`, this
        # test's actual running App, is). Re-point at `host` so
        # `call_from_thread` genuinely marshals onto the running event
        # loop -- mounting/mutating REAL Textual widgets (as
        # `ChatApprovalCard.set_batch` does) from a foreign OS thread
        # without that marshal raises `RuntimeError: no running event
        # loop` (verified empirically while writing this test).
        controller.app = host
        store = controller.store
        session_a = store.active_session_id
        session_b = controller.new_session().id
        store.switch_session(session_a)  # A is viewed; the round mounts on it
        controller.mcp_approval_timeout_seconds = lambda: 30.0

        # `asyncio.to_thread` (awaited via a Task, never a blocking
        # `Thread.join()`) -- a raw `threading.Thread` + `.join()` here
        # would deadlock: `.join()` blocks THIS coroutine, which is running
        # ON the same event loop the worker thread's `call_from_thread`
        # needs free to marshal its widget mutations back onto (verified
        # empirically while writing this test -- the round never resolved,
        # `result_holder` stayed empty until the 2s join timeout).
        decisions_task = asyncio.create_task(
            asyncio.to_thread(
                controller.request_mcp_approvals,
                [_single_pending_call()],
                session_id=session_a,
            )
        )
        await pilot.pause(0.3)

        approval_card = console.query_one("#chat-approval-card")
        assert approval_card.display  # mounted immediately -- A was active

        controller.switch_session(session_b)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.2)
        assert not console.query_one("#chat-approval-card").display

        controller.switch_session(session_a)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.2)
        # The fix: the card re-mounts. Pre-fix this was `False` (no
        # retained payload to re-derive from).
        assert console.query_one("#chat-approval-card").display

        round_id = controller._parked_approval_payloads[session_a]["round_id"]
        controller.resolve_pending_approval(
            {"mcp__srv__tool": "approve_once"}, round_id=round_id
        )
        decisions = await asyncio.wait_for(decisions_task, timeout=2.0)
        assert decisions == {"mcp__srv__tool": "approve_once"}


@pytest.mark.asyncio
async def test_new_session_clears_a_mounted_card_from_the_session_being_left() -> None:
    """Final review IMPORTANT 2: `new_session` activates the created
    session but, pre-fix, never re-derived the approval card the way
    `switch_session`/`close_session` do -- a round mounted on the session
    being left behind stayed rendered over the brand-new tab. Asserted
    through the REAL widget, driven by the real `new_session` call.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        # See the sibling test above for why this must be `host` (the
        # actually-running App), not `app_instance`.
        controller.app = host
        controller.mcp_approval_timeout_seconds = lambda: 30.0
        session_a = controller.store.active_session_id

        # See the sibling test above for why this is `asyncio.to_thread` +
        # an awaited Task, never a blocking `Thread.join()`.
        decisions_task = asyncio.create_task(
            asyncio.to_thread(
                controller.request_mcp_approvals,
                [_single_pending_call()],
                session_id=session_a,
            )
        )
        await pilot.pause(0.3)
        assert console.query_one("#chat-approval-card").display

        new_session = controller.new_session()
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.2)
        # The fix: brand-new tab shows no stale card from session A.
        assert not console.query_one("#chat-approval-card").display

        # Composes with fix 1: switching back to A re-mounts it.
        controller.switch_session(session_a)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.2)
        assert console.query_one("#chat-approval-card").display

        round_id = controller._parked_approval_payloads[session_a]["round_id"]
        controller.resolve_pending_approval(
            {"mcp__srv__tool": "deny"}, round_id=round_id
        )
        decisions = await asyncio.wait_for(decisions_task, timeout=2.0)
        assert decisions == {"mcp__srv__tool": "deny"}
        assert new_session.id != session_a


@pytest.mark.asyncio
async def test_background_skill_install_confirm_parks_badges_toasts_and_mounts_on_visit() -> None:
    """TASK-910: `request_skill_install_confirm` now gets the SAME park/
    badge/toast/re-mount treatment as `request_mcp_approvals` -- see
    `test_background_approval_parks_with_badge_and_single_toast` above,
    which this mirrors, but through the REAL controller bridge (a genuine
    worker thread via `asyncio.to_thread`) rather than a seeded payload,
    exercising the full seam end-to-end: park -> badge -> one toast ->
    mount on visit -> re-mount on revisit -> resolve.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        # See `test_mounted_round_survives_switch_away_and_switch_back` for
        # why this must be `host` (the actually-running App), not
        # `app_instance`: `call_from_thread` needs a real running loop to
        # marshal the worker thread's widget mutations onto.
        controller.app = host
        controller.skill_install_confirm_timeout_seconds = lambda: 30.0
        store = controller.store
        viewed = store.active_session_id
        background = controller.new_session().id
        store.switch_session(viewed)  # keep viewing the first session

        notifications: list[str] = []
        app.notify = lambda message, **kwargs: notifications.append(str(message))

        decision_task = asyncio.create_task(
            asyncio.to_thread(
                controller.request_skill_install_confirm,
                "https://github.com/o/r",
                session_id=background,
            )
        )
        await pilot.pause(0.3)

        install_card = console.query_one("#chat-skill-install-card")
        assert not install_card.display  # parked: never mounted over the viewed tab
        approval_toasts = [n for n in notifications if "needs approval" in n]
        assert len(approval_toasts) == 1
        assert (
            controller.run_marker_for(background) is ConsoleRunMarker.NEEDS_APPROVAL
        )

        # Visiting mounts the card through the existing mount path.
        controller.switch_session(background)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.2)
        assert console.query_one("#chat-skill-install-card").display

        # Switch-away-and-back re-mounts the SAME round without a second toast.
        controller.switch_session(viewed)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.1)
        assert not console.query_one("#chat-skill-install-card").display
        controller.switch_session(background)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.1)
        assert console.query_one("#chat-skill-install-card").display
        assert len([n for n in notifications if "needs approval" in n]) == 1

        request_id = controller._parked_skill_install_payloads[background]["request_id"]
        controller.resolve_pending_skill_install(True, request_id=request_id)
        allowed = await asyncio.wait_for(decision_task, timeout=2.0)
        assert allowed is True
        assert background not in controller._pending_approvals


@pytest.mark.asyncio
async def test_background_skill_script_confirm_parks_badges_toasts_and_mounts_on_visit() -> None:
    """TASK-910: `request_skill_script_confirm` gets the identical
    treatment -- see the sibling skill-install test above."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        controller = console._ensure_console_chat_controller()
        controller.app = host
        controller.skill_script_confirm_timeout_seconds = lambda: 30.0
        store = controller.store
        viewed = store.active_session_id
        background = controller.new_session().id
        store.switch_session(viewed)  # keep viewing the first session

        notifications: list[str] = []
        app.notify = lambda message, **kwargs: notifications.append(str(message))

        decision_task = asyncio.create_task(
            asyncio.to_thread(
                controller.request_skill_script_confirm,
                {"skill_name": "demo", "script_path": "scripts/hello.py"},
                session_id=background,
            )
        )
        await pilot.pause(0.3)

        script_card = console.query_one("#chat-skill-script-card")
        assert not script_card.display  # parked: never mounted over the viewed tab
        approval_toasts = [n for n in notifications if "needs approval" in n]
        assert len(approval_toasts) == 1
        assert (
            controller.run_marker_for(background) is ConsoleRunMarker.NEEDS_APPROVAL
        )

        controller.switch_session(background)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.2)
        assert console.query_one("#chat-skill-script-card").display

        controller.switch_session(viewed)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.1)
        assert not console.query_one("#chat-skill-script-card").display
        controller.switch_session(background)
        await console._sync_native_console_chat_ui()
        await pilot.pause(0.1)
        assert console.query_one("#chat-skill-script-card").display
        assert len([n for n in notifications if "needs approval" in n]) == 1

        request_id = controller._parked_skill_script_payloads[background]["request_id"]
        controller.resolve_pending_skill_script(True, False, request_id=request_id)
        decision = await asyncio.wait_for(decision_task, timeout=2.0)
        assert decision == {"allow": True, "remember": False}
        assert background not in controller._pending_approvals
