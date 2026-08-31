"""Inline, privacy-safe recovery for interrupted provider tool runs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Static

from ...Chat.console_chat_models import ConsoleChatMessage
from ...Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparation,
    ConsoleTurnPreparationState,
)
from .frame import frame_console_region
from .transcript import ConsoleTranscriptRegion


@dataclass(frozen=True, slots=True)
class TraceCallRecoveryState:
    """Content-free identity for one blocked send recovery callout."""

    preparation_id: str
    temporary_capture: bool = False


def trace_call_recovery_state(
    preparation: ConsoleTurnPreparation | None,
) -> TraceCallRecoveryState | None:
    """Project only an actionable trace pause into the transcript UI.

    Args:
        preparation: Current session preparation, if one exists.

    Returns:
        Content-free recovery identity for a supported pause, otherwise None.
    """

    if (
        preparation is None
        or preparation.state is not ConsoleTurnPreparationState.PAUSED
        or preparation.pause_kind
        not in {
            ConsolePreparationPauseKind.TRACE_CALL,
            ConsolePreparationPauseKind.TEMPORARY_CAPTURE,
        }
    ):
        return None
    return TraceCallRecoveryState(
        preparation.preparation_id,
        temporary_capture=(
            preparation.pause_kind
            is ConsolePreparationPauseKind.TEMPORARY_CAPTURE
        ),
    )


TraceCallAction = Callable[[str, str], object | Awaitable[object]]


async def dispatch_trace_call_recovery_action(
    controller: Any,
    action: str,
    preparation_id: str,
    *,
    on_started: Callable[[], object] | None = None,
    on_finished: Callable[[], object | Awaitable[object]] | None = None,
) -> object:
    """Route a visible trace-recovery action to the controller.

    Args:
        controller: Controller exposing the selected recovery entrypoint.
        action: Stable visible action identifier.
        preparation_id: Exact paused preparation identity.
        on_started: Optional callback invoked before controller entry.
        on_finished: Optional callback invoked after controller completion.

    Returns:
        The awaited controller result, or False for an unknown action or missing
        handler.
    """

    handler_name = {
        "retry": "retry_library_preparation",
        "save_and_send": "save_and_send",
        "send_without_capture": "send_without_capture",
        "cancel": "cancel_library_preparation",
    }.get(action)
    handler = getattr(controller, handler_name, None) if handler_name else None
    if not callable(handler):
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
    """Terminal-native warning with explicit, focusable recovery actions."""

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
            yield Button(
                "Save & Send",
                id="console-trace-save-send",
                variant="warning",
            )
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
        temporary = bool(state is not None and state.temporary_capture)
        self.query_one("#console-trace-title", Static).update(
            "Save chat to capture this send" if temporary else "Trace capture blocked"
        )
        detail_rows = tuple(self.query(".console-trace-detail"))
        if len(detail_rows) >= 2:
            detail_rows[1].update(
                (
                    "Problem: Temporary chats cannot store durable captures."
                    if temporary
                    else "Problem: Trace capture could not be saved."
                )
            )
        for button in self.query(Button):
            button.display = state is not None
            button.disabled = self._busy or state is None
        self.query_one("#console-trace-save-send", Button).display = temporary
        self.query_one("#console-trace-retry", Button).display = (
            state is not None and not temporary
        )
        self.query_one("#console-trace-status", Static).update(
            "Working… actions are temporarily disabled."
            if self._busy
            else "Choose one action."
        )

    @on(Button.Pressed)
    def _handle_action(self, event: Button.Pressed) -> None:
        action = {
            "console-trace-save-send": "save_and_send",
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


@dataclass(frozen=True)
class ProviderContinuationRecoveryState:
    """Only the bounded, non-private facts the recovery UI may consume."""

    message_id: str
    message_version: int
    mode: str
    impact: str
    replay_available: bool = False
    actions_enabled: bool = True


def provider_continuation_recovery_state(
    message: ConsoleChatMessage | None,
    *,
    replay_available: bool = False,
    owner_live: bool = False,
) -> ProviderContinuationRecoveryState | None:
    """Project an assistant checkpoint into fixed user-facing recovery copy."""
    if message is None or owner_live:
        return None
    if message.generation_projection_quarantined:
        return ProviderContinuationRecoveryState(
            message.id,
            message.generation_projection_quarantine_version or 0,
            "reload",
            message.generation_projection_quarantine_reason
            or "Canonical generation is unavailable; reload required.",
        )
    if message.provider_continuation is None and message.provider_continuation_warning:
        return ProviderContinuationRecoveryState(
            message.id,
            message.provider_continuation_message_version or 0,
            "notice",
            message.provider_continuation_warning,
        )
    if message.provider_continuation is None:
        return None
    checkpoint = message.provider_continuation
    if checkpoint.state != "active":
        return None
    version = message.provider_continuation_message_version
    if type(version) is not int or version <= 0:
        return None
    states = {call.state for round_ in checkpoint.rounds for call in round_.calls}
    if "executing" in states:
        return ProviderContinuationRecoveryState(
            message.id,
            version,
            "ambiguous",
            "A tool may already have run. Resume is blocked to avoid repeating a side effect; discard this interrupted run to continue.",
            False,
            message.provider_continuation_actions_enabled,
        )
    if message.provider_continuation_remote:
        impact = (
            "This run was active elsewhere. The other device may still be running it; "
            "take over only after checking there, or discard it."
        )
        if not replay_available:
            impact = (
                "Take over is unavailable until continuation replay support is enabled "
                "for this provider integration. The other device may still be running; "
                "you can still discard this interrupted run."
            )
        return ProviderContinuationRecoveryState(
            message.id,
            version,
            "remote",
            message.provider_continuation_warning or impact,
            replay_available,
            message.provider_continuation_actions_enabled,
        )
    impact = (
        "The provider paused while tools may not have finished. Resume after reviewing "
        "approvals, or discard this interrupted run."
    )
    if not replay_available:
        impact = (
            "Resume is unavailable until continuation replay support is enabled for this "
            "provider integration. You can still discard this interrupted run."
        )
    return ProviderContinuationRecoveryState(
        message.id,
        version,
        "local",
        message.provider_continuation_warning or impact,
        replay_available,
        message.provider_continuation_actions_enabled,
    )


ContinuationAction = Callable[[str, str, int], bool | Awaitable[bool]]


class ProviderContinuationRecoveryCallout(Vertical):
    """Bounded inline warning with keyboard-reachable recovery actions."""

    DEFAULT_CSS = """
    ProviderContinuationRecoveryCallout {
        width: 100%;
        height: auto;
        border: round $warning;
        background: $boost;
        color: $text;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    ProviderContinuationRecoveryCallout #console-continuation-title {
        width: 100%;
        height: auto;
        text-style: bold;
        color: $warning-lighten-2;
    }
    ProviderContinuationRecoveryCallout #console-continuation-impact,
    ProviderContinuationRecoveryCallout #console-continuation-status {
        width: 100%;
        height: auto;
    }
    ProviderContinuationRecoveryCallout #console-continuation-actions {
        width: 100%;
        height: 3;
        margin-top: 1;
    }
    ProviderContinuationRecoveryCallout Button {
        min-width: 10;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        *,
        state: ProviderContinuationRecoveryState | None,
        on_action: ContinuationAction,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.recovery_state = state
        self._on_action = on_action
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Static("Interrupted tool run", id="console-continuation-title")
        yield Static("", id="console-continuation-impact")
        yield Static("Choose an action to continue.", id="console-continuation-status")
        with Horizontal(id="console-continuation-actions"):
            yield Button(
                "Resume",
                id="console-continuation-resume",
                variant="warning",
            )
            yield Button(
                "Take over",
                id="console-continuation-take-over",
                variant="warning",
            )
            yield Button("Discard", id="console-continuation-discard")
            yield Button(
                "Reload",
                id="console-continuation-reload",
                variant="warning",
            )

    def on_mount(self) -> None:
        """Apply initial state after the fixed controls are mounted."""
        self.sync_recovery(self.recovery_state)

    def sync_recovery(
        self,
        state: ProviderContinuationRecoveryState | None,
    ) -> None:
        """Update fixed callout controls without removing or recomposing them."""
        self.recovery_state = state
        if state is None:
            self.display = False
            return
        self.display = True
        self.query_one("#console-continuation-title", Static).update(
            "Generation unavailable"
            if state.mode == "reload"
            else (
                "Continuation warning"
                if state.mode == "notice"
                else "Interrupted tool run"
            )
        )
        self.query_one("#console-continuation-impact", Static).update(state.impact)
        self.query_one("#console-continuation-status", Static).update(
            "Working… actions are temporarily disabled."
            if self._busy
            else (
                "Reload the canonical generation before continuing."
                if state.mode == "reload"
                else (
                    "The visible message is unchanged. No action is required."
                    if state.mode == "notice"
                    else "Choose an available action to continue."
                )
            )
        )
        resume = self.query_one("#console-continuation-resume", Button)
        take_over = self.query_one("#console-continuation-take-over", Button)
        discard = self.query_one("#console-continuation-discard", Button)
        reload_button = self.query_one("#console-continuation-reload", Button)
        resume.display = state.mode == "local"
        take_over.display = state.mode == "remote"
        discard.display = state.mode not in {"notice", "reload"}
        reload_button.display = state.mode == "reload"
        resume.disabled = (
            self._busy or not state.actions_enabled or not state.replay_available
        )
        take_over.disabled = (
            self._busy or not state.actions_enabled or not state.replay_available
        )
        discard.disabled = self._busy or not state.actions_enabled
        reload_button.disabled = self._busy or not state.actions_enabled

    @on(Button.Pressed)
    def _handle_action(self, event: Button.Pressed) -> None:
        action_by_id = {
            "console-continuation-resume": "resume",
            "console-continuation-take-over": "take_over",
            "console-continuation-discard": "discard",
            "console-continuation-reload": "reload",
        }
        action = action_by_id.get(event.button.id or "")
        state = self.recovery_state
        if action is None or self._busy or state is None or not state.actions_enabled:
            return
        event.stop()
        self._busy = True
        for button in self.query(Button):
            button.disabled = True
        self.query_one("#console-continuation-status", Static).update(
            "Working… actions are temporarily disabled."
        )
        self.run_worker(
            self._dispatch(action),
            exclusive=True,
            group=f"provider-continuation-{state.message_id}",
        )

    async def _dispatch(self, action: str) -> None:
        succeeded = False
        starting_state = self.recovery_state
        if starting_state is None:
            self._busy = False
            self.sync_recovery(None)
            return
        try:
            result = self._on_action(
                action,
                starting_state.message_id,
                starting_state.message_version,
            )
            succeeded = (
                bool(await result) if inspect.isawaitable(result) else bool(result)
            )
        except asyncio.CancelledError:
            self._busy = False
            self.sync_recovery(self.recovery_state)
            raise
        except Exception:
            succeeded = False
        if succeeded:
            self._busy = False
            if self.recovery_state != starting_state:
                self.sync_recovery(self.recovery_state)
            else:
                self.display = False
            self.screen.focus_next("#console-transcript-region *")
            return
        self._busy = False
        if self.recovery_state == starting_state:
            self.sync_recovery(starting_state)
            self.query_one("#console-continuation-status", Static).update(
                "Recovery did not complete. Reload the conversation and try again."
            )
        else:
            self.sync_recovery(self.recovery_state)
        first = next(
            (
                button
                for button in self.query(Button)
                if button.display and not button.disabled
            ),
            None,
        )
        if first is not None:
            first.focus()


class ProviderContinuationTranscriptRegion(ConsoleTranscriptRegion):
    """Transcript region with one active-path continuation recovery callout."""

    def __init__(
        self,
        *,
        session_surface_builder: Callable[[], Widget],
        recovery_message_builder: Callable[[], ConsoleChatMessage | None],
        on_recovery_action: ContinuationAction,
        recovery_replay_available_builder: Callable[[], bool] = lambda: False,
        recovery_owner_live_builder: Callable[[ConsoleChatMessage], bool] = (
            lambda _message: False
        ),
        trace_recovery_state_builder: Callable[
            [], TraceCallRecoveryState | None
        ] = lambda: None,
        on_trace_recovery_action: TraceCallAction = lambda *_args: False,
        **kwargs: object,
    ) -> None:
        super().__init__(session_surface_builder=session_surface_builder, **kwargs)
        self._recovery_message_builder = recovery_message_builder
        self._recovery_replay_available_builder = recovery_replay_available_builder
        self._recovery_owner_live_builder = recovery_owner_live_builder
        self._on_recovery_action = on_recovery_action
        self._trace_recovery_state_builder = trace_recovery_state_builder
        self._on_trace_recovery_action = on_trace_recovery_action

    def compose(self) -> ComposeResult:
        transcript_region = frame_console_region(
            Vertical(id="console-transcript-region", classes="console-region"),
            edges=(),
        )
        with transcript_region:
            yield TraceCallRecoveryCallout(
                state=self._trace_recovery_state_builder(),
                on_action=self._recover_trace_call,
            )
            yield ProviderContinuationRecoveryCallout(
                state=self._recovery_state(),
                on_action=self._recover,
            )
            yield self._session_surface_builder()

    def _recovery_state(self) -> ProviderContinuationRecoveryState | None:
        message = self._recovery_message_builder()
        return provider_continuation_recovery_state(
            message,
            replay_available=self._recovery_replay_available_builder(),
            owner_live=(
                self._recovery_owner_live_builder(message)
                if message is not None
                else False
            ),
        )

    def sync_recovery(self) -> None:
        """Synchronize the always-mounted recovery placeholder in place."""
        self.query_one(TraceCallRecoveryCallout).sync_recovery(
            self._trace_recovery_state_builder()
        )
        self.query_one(ProviderContinuationRecoveryCallout).sync_recovery(
            self._recovery_state()
        )

    async def _recover_trace_call(self, action: str, preparation_id: str) -> bool:
        result = self._on_trace_recovery_action(action, preparation_id)
        if inspect.isawaitable(result):
            await result
        self.sync_recovery()
        return self._trace_recovery_state_builder() is None

    async def _recover(self, action: str, message_id: str, version: int) -> bool:
        result = self._on_recovery_action(action, message_id, version)
        succeeded = bool(await result) if inspect.isawaitable(result) else bool(result)
        self.sync_recovery()
        return succeeded
