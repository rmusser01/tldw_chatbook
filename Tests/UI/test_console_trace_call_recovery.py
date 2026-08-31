"""Mounted recovery actions for pre-dispatch trace persistence failures."""

from __future__ import annotations

import asyncio

from textual.app import ComposeResult
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Console_Modules.trace_call_recovery import (
    TraceCallRecoveryCallout,
    TraceCallRecoveryState,
    dispatch_trace_call_recovery_action,
)


class _Controller:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def retry_library_preparation(self, preparation_id: str) -> object:
        self.actions.append(("retry", preparation_id))
        self.started.set()
        await self.release.wait()
        return object()

    async def send_without_capture(self, preparation_id: str) -> object:
        self.actions.append(("send_without_capture", preparation_id))
        self.started.set()
        await self.release.wait()
        return object()

    def cancel_library_preparation(self, preparation_id: str) -> object:
        self.actions.append(("cancel", preparation_id))
        return object()


class _RecoveryApp(ConsolidatedCSSApp):
    def __init__(self, controller: _Controller) -> None:
        super().__init__()
        self.controller = controller

    def compose(self) -> ComposeResult:
        yield TraceCallRecoveryCallout(
            state=TraceCallRecoveryState("preparation-1"),
            on_action=lambda action, preparation_id: (
                dispatch_trace_call_recovery_action(
                    self.controller, action, preparation_id
                )
            ),
        )


async def test_trace_recovery_callout_is_wrapped_labeled_and_keyboard_focusable() -> (
    None
):
    controller = _Controller()
    app = _RecoveryApp(controller)
    async with app.run_test(size=(34, 18)) as pilot:
        await pilot.pause()
        rendered = "\n".join(
            str(widget.render())
            for widget in app.screen.query("TraceCallRecoveryCallout *")
            if hasattr(widget, "render")
        )
        assert "Owner: This Console send" in rendered
        assert "Problem: Trace capture could not be saved." in rendered
        assert "Impact: The provider was not contacted." in rendered
        assert "Retry capture" in rendered
        assert "Send without capture" in rendered
        assert "Cancel send" in rendered

        retry = app.screen.query_one("#console-trace-retry", Button)
        retry.focus()
        await pilot.pause()
        assert app.focused is retry
        await pilot.press("enter")
        await asyncio.wait_for(controller.started.wait(), timeout=1)
        assert all(button.disabled for button in app.screen.query(Button))
        controller.release.set()
        await app.workers.wait_for_complete()


async def test_trace_recovery_action_is_consumed_before_double_press() -> None:
    controller = _Controller()
    app = _RecoveryApp(controller)
    async with app.run_test(size=(52, 16)) as pilot:
        retry = app.screen.query_one("#console-trace-retry", Button)
        retry.focus()
        await pilot.press("enter", "enter")
        await asyncio.wait_for(controller.started.wait(), timeout=1)
        await pilot.pause()

        assert controller.actions == [("retry", "preparation-1")]
        status = app.screen.query_one("#console-trace-status", Static)
        assert "Working" in str(status.render())
        controller.release.set()
        await app.workers.wait_for_complete()


async def test_trace_recovery_cancel_calls_terminal_controller_entrypoint() -> None:
    controller = _Controller()
    app = _RecoveryApp(controller)
    async with app.run_test(size=(42, 18)) as pilot:
        cancel = app.screen.query_one("#console-trace-cancel", Button)
        cancel.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        assert controller.actions == [("cancel", "preparation-1")]


async def test_trace_recovery_refusal_reenables_all_actions() -> None:
    class RefusingApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield TraceCallRecoveryCallout(
                state=TraceCallRecoveryState("preparation-1"),
                on_action=lambda *_args: False,
            )

    app = RefusingApp()
    async with app.run_test(size=(42, 18)) as pilot:
        retry = app.screen.query_one("#console-trace-retry", Button)
        retry.focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()

        callout = app.screen.query_one(TraceCallRecoveryCallout)
        assert callout.display
        assert all(not button.disabled for button in callout.query(Button))
        assert "did not complete" in str(
            callout.query_one("#console-trace-status", Static).render()
        )
