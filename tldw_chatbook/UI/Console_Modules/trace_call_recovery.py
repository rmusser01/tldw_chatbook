"""Inline recovery for a Capture On send blocked before provider entry."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
)


@dataclass(frozen=True, slots=True)
class TraceCallRecoveryState:
    """Content-free identity for one blocked send recovery callout."""

    preparation_id: str


def trace_call_recovery_state(
    preparation: ConsoleTurnPreparation | None,
) -> TraceCallRecoveryState | None:
    """Project only an actionable TRACE_CALL pause into the UI."""

    if (
        preparation is None
        or preparation.state is not ConsoleTurnPreparationState.PAUSED
        or preparation.pause_kind is not ConsolePreparationPauseKind.TRACE_CALL
    ):
        return None
    return TraceCallRecoveryState(preparation.preparation_id)


TraceCallAction = Callable[[str, str], object | Awaitable[object]]


async def dispatch_trace_call_recovery_action(
    controller: Any,
    action: str,
    preparation_id: str,
    *,
    on_started: Callable[[], object] | None = None,
    on_finished: Callable[[], object | Awaitable[object]] | None = None,
) -> object:
    """Route a visible action to the existing controller entrypoint."""

    handler = {
        "retry": controller.retry_library_preparation,
        "send_without_capture": controller.send_without_capture,
        "cancel": controller.cancel_library_preparation,
    }.get(action)
    if handler is None:
        return False
    if on_started is not None:
        on_started()
    try:
        result = handler(preparation_id)
        return await result if inspect.isawaitable(result) else result
    finally:
        if on_finished is not None:
            finished = on_finished()
            if inspect.isawaitable(finished):
                await finished


class TraceCallRecoveryCallout(Vertical):
    """Terminal-native warning with three explicit, focusable actions."""

    BUNDLED_CSS = """
    TraceCallRecoveryCallout {
        width: 100%;
        height: auto;
        border: round $warning;
        background: $boost;
        color: $text;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    TraceCallRecoveryCallout #console-trace-title {
        width: 100%;
        height: auto;
        text-style: bold;
        color: $warning-lighten-2;
    }
    TraceCallRecoveryCallout .console-trace-detail,
    TraceCallRecoveryCallout #console-trace-status {
        width: 100%;
        height: auto;
    }
    TraceCallRecoveryCallout #console-trace-actions {
        width: 100%;
        height: auto;
        margin-top: 1;
    }
    TraceCallRecoveryCallout Button {
        width: 100%;
        min-width: 20;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        *,
        state: TraceCallRecoveryState | None,
        on_action: TraceCallAction,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.recovery_state = state
        self._on_action = on_action
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Static("Trace capture blocked", id="console-trace-title")
        yield Static("Owner: This Console send", classes="console-trace-detail")
        yield Static(
            "Problem: Trace capture could not be saved.",
            classes="console-trace-detail",
        )
        yield Static(
            "Impact: The provider was not contacted. Choose how to continue.",
            classes="console-trace-detail",
        )
        yield Static("Choose one action.", id="console-trace-status")
        with Vertical(id="console-trace-actions"):
            yield Button("Retry capture", id="console-trace-retry", variant="warning")
            yield Button(
                "Send without capture",
                id="console-trace-send-without",
                variant="warning",
            )
            yield Button("Cancel send", id="console-trace-cancel")

    def on_mount(self) -> None:
        self.sync_recovery(self.recovery_state)

    def sync_recovery(self, state: TraceCallRecoveryState | None) -> None:
        """Update the always-mounted placeholder without recomposition."""

        self.recovery_state = state
        self.display = state is not None
        for button in self.query(Button):
            button.display = state is not None
            button.disabled = self._busy or state is None
        self.query_one("#console-trace-status", Static).update(
            "Working… actions are temporarily disabled."
            if self._busy
            else "Choose one action."
        )

    @on(Button.Pressed)
    def _handle_action(self, event: Button.Pressed) -> None:
        action = {
            "console-trace-retry": "retry",
            "console-trace-send-without": "send_without_capture",
            "console-trace-cancel": "cancel",
        }.get(event.button.id or "")
        state = self.recovery_state
        if action is None or state is None or self._busy:
            return
        event.stop()
        self._busy = True
        self.sync_recovery(state)
        self.run_worker(
            self._dispatch(action, state),
            exclusive=True,
            group=f"trace-call-recovery-{state.preparation_id}",
        )

    async def _dispatch(self, action: str, state: TraceCallRecoveryState) -> None:
        try:
            result = self._on_action(action, state.preparation_id)
            if inspect.isawaitable(result):
                result = await result
        except asyncio.CancelledError:
            self._busy = False
            self.sync_recovery(self.recovery_state)
            raise
        except Exception:
            self._busy = False
            self.sync_recovery(self.recovery_state)
            self.query_one("#console-trace-status", Static).update(
                "Recovery did not complete. Try again or cancel the send."
            )
            return
        self._busy = False
        if result:
            self.display = False
            return
        self.sync_recovery(self.recovery_state)
        self.query_one("#console-trace-status", Static).update(
            "Recovery did not complete. Try again or cancel the send."
        )
