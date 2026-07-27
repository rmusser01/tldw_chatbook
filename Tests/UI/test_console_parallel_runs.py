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
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
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
