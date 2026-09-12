"""Production-path regression for a human decision armed while detached.

The historical probe measured whether a headless round denied itself.  The
navigation-lifetime contract is now exact: the round must remain pending
until a human decision or an explicit cancellation boundary resolves it.
"""

from __future__ import annotations

import threading

import pytest

from Tests.Chat.test_console_fleet_wake import _drain, _settle, _survivor
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_mcp_approval import _pending
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_store_continuity import (
    _StallingWakeGateway,
    _drain_from_child_thread,
    _navigate,
    _seed_console,
    _terminal_survivor_run,
)


def _risk_row():
    """The shape `build_tool_review_hook` emits for a risk-tagged tool."""
    return _pending(
        server_key="agent:builtin",
        tool_name="write_file",
        llm_name="builtin__write_file",
        reason="risk_floored",
    )


def _build_console_app(tmp_path):
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _StallingWakeGateway()
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    return app, gateway


@pytest.mark.asyncio
async def test_headless_approval_waits_until_exact_human_resolution(tmp_path):
    """A risk-tagged round remains pending with no Console mounted.

    Console is left through the production navigation path (the real
    `NavigateToScreen` + the real "Leave Console?" dialog), so the runtime
    is in exactly the state a headless wake turn runs in. A wake turn is
    then held in flight at the provider readiness probe, and the approval
    round is armed from a plain child thread -- the thread
    `build_tool_review_hook` calls `request_mcp_approvals` on.
    """
    app, gateway = _build_console_app(tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        wake = controller.fleet_wake
        runs_db = controller._agent_bridge.runs_db
        run_id = _terminal_survivor_run(runs_db, conversation_id)

        # -- hold a wake turn in flight, then leave Console for real --------
        gateway.stall = True
        _drain_from_child_thread(
            wake, _drain(conversation_id, _survivor(run_id, session_id=session_id))
        )
        assert await _settle(lambda: gateway.entered_stall.is_set(), seconds=10.0), (
            "wake turn never reached its readiness barrier"
        )

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack
        assert controller is app.console_runtime.chat_controller
        assert not controller._shutdown_requested.is_set()
        assert controller._disposed is False
        assert controller.set_pending_approval is None
        assert controller.park_pending_approval is None

        # -- arm the risk-tagged round from a plain worker thread -----------
        result: dict[str, object] = {}

        def _arm() -> None:
            result["decisions"] = controller.request_mcp_approvals(
                [_risk_row()], session_id=session_id
            )

        thread = threading.Thread(target=_arm, daemon=True)
        thread.start()
        assert await _settle(
            lambda: (
                controller._head_round_payload(
                    controller._parked_approval_payloads, session_id
                )
                is not None
            ),
            seconds=5.0,
        ), "headless approval payload was never retained"
        head = controller._head_round_payload(
            controller._parked_approval_payloads, session_id
        )
        assert head is not None
        round_id = str(head["round_id"])

        # Cross more than one legacy poll interval.  Navigation must neither
        # deny the round nor consume its answerable-time budget.
        await pilot.pause(1.2)
        assert thread.is_alive()
        assert "decisions" not in result
        assert controller.has_pending_approval_round(session_id)

        controller.resolve_pending_approval(
            {"builtin__write_file": "deny"}, round_id=round_id
        )
        await pilot.pause()
        thread.join(5)
        assert not thread.is_alive()
        assert result["decisions"] == {"builtin__write_file": "deny"}
        assert not controller.has_pending_approval_round(session_id)

        gateway.release.set()
        await pilot.pause()
