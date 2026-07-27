"""Two sessions run concurrently; interrupt is session-scoped (spec §2)."""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
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
