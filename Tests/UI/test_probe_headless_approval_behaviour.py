"""PROBE (task-15860 plan Task 5, step 1): what a headless approval ACTUALLY does.

NOT a regression test. One question, answered by execution before anything
is built:

    With Console unmounted through the REAL navigation API, does a
    risk-tagged tool's approval round burn the ~120s the plan's P4
    measured, or does it deny at the first poll because the visit's
    teardown Event was already set at arm time?

The plan (`Docs/superpowers/plans/2026-08-14-headless-wake.md` Task 5)
carries P4's number, 120.43s. The fires landing flagged -- and explicitly
did NOT execute -- the inference that the number is now wrong. Two things
changed under it since P4 ran:

* ADR-067 (`8403c12e6`) dropped `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS`
  from 120.0 to **0.0** (no deadline at all) and added a no-`app` guard
  that denies on the spot;
* the lifetime landing made `_shutdown_requested` per-VISIT and
  `request_mcp_approvals` binds it at ARM time
  (`_bind_visit_cancel_signal`), so a round armed while detached captures
  an Event that is ALREADY SET.

The probe reports the number rather than asserting a threshold: the
measurement is the finding.
"""
from __future__ import annotations

import threading
import time

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
from tldw_chatbook.Chat.console_chat_controller import (
    _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS,
)


def _report(title: str, lines: list[str]) -> None:
    print(f"\n===== {title} =====")
    for line in lines:
        print(f"  {line}")


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
async def test_probe_headless_approval_round_wall_time(tmp_path):
    """Measure the real thing: arm a risk-tagged round with NO Console mounted.

    Console is left through the production navigation path (the real
    `NavigateToScreen` + the real "Leave Console?" dialog), so the runtime
    is in exactly the state a headless wake turn runs in. A wake turn is
    then held in flight at the provider readiness probe, and the approval
    round is armed from a plain child thread -- the thread
    `build_tool_review_hook` calls `request_mcp_approvals` on.
    """
    findings: list[str] = []
    app, gateway = _build_console_app(tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        wake = controller.fleet_wake
        runs_db = controller._agent_bridge.runs_db
        run_id = _terminal_survivor_run(runs_db, conversation_id)

        findings.append(
            "config: _DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS = "
            f"{_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS}"
        )
        findings.append(
            "config: effective [mcp] approval_timeout_seconds = "
            f"{controller._resolve_mcp_approval_timeout_seconds()}"
        )

        # -- hold a wake turn in flight, then leave Console for real --------
        gateway.stall = True
        _drain_from_child_thread(
            wake, _drain(conversation_id, _survivor(run_id, session_id=session_id))
        )
        entered = await _settle(lambda: gateway.entered_stall.is_set(), seconds=10.0)
        findings.append(f"wake turn parked at the readiness probe: {entered}")

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        findings.append(f"Console unmounted: {chat not in app.screen_stack}")
        findings.append(
            f"runtime outlived the screen: "
            f"{controller is app.console_runtime.chat_controller}"
        )
        findings.append(
            f"visit Event set (_shutdown_requested): "
            f"{controller._shutdown_requested.is_set()}"
        )
        findings.append(f"controller._disposed: {controller._disposed}")
        findings.append(f"controller.app wired: {controller.app is not None}")
        findings.append(
            f"set_pending_approval wired: {controller.set_pending_approval is not None}"
        )
        findings.append(
            f"park_pending_approval wired: "
            f"{controller.park_pending_approval is not None}"
        )

        # -- arm the risk-tagged round from a plain worker thread -----------
        result: dict[str, object] = {}

        def _arm() -> None:
            started = time.monotonic()
            result["decisions"] = controller.request_mcp_approvals(
                [_risk_row()], session_id=session_id
            )
            result["elapsed"] = time.monotonic() - started

        thread = threading.Thread(target=_arm)
        thread.start()
        # Give the arming thread room; poll the loop so anything it
        # marshals through `call_from_thread` can actually run.
        deadline = time.monotonic() + 30.0
        while thread.is_alive() and time.monotonic() < deadline:
            await pilot.pause(0.05)
        thread.join(5)

        findings.append(f"MEASURED decisions = {result.get('decisions')}")
        elapsed = result.get("elapsed")
        findings.append(
            "MEASURED wall time to verdict = "
            + (f"{elapsed:.2f}s" if isinstance(elapsed, float) else "NEVER RETURNED")
        )
        findings.append(
            "registry: round still registered after the verdict = "
            f"{controller.has_pending_approval_round(session_id)}"
        )
        head = controller._head_round_payload(
            controller._parked_approval_payloads, session_id
        )
        findings.append(f"registry: parked payload retained = {head is not None}")

        gateway.release.set()
        await pilot.pause()

    _report("PROBE -- headless approval, measured through the production path", findings)
