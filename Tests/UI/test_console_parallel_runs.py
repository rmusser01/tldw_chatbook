"""Two sessions run concurrently; interrupt is session-scoped (spec §2)."""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus


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
