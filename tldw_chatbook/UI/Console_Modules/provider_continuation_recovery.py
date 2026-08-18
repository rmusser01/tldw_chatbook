"""Inline, privacy-safe recovery for interrupted provider tool runs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Static

from ...Chat.console_chat_models import ConsoleChatMessage
from .frame import frame_console_region
from .transcript import ConsoleTranscriptRegion


@dataclass(frozen=True)
class ProviderContinuationRecoveryState:
    """Only the bounded, non-private facts the recovery UI may consume."""

    message_id: str
    message_version: int
    mode: str
    impact: str
    replay_available: bool = False


def provider_continuation_recovery_state(
    message: ConsoleChatMessage | None,
    *,
    replay_available: bool = False,
    owner_live: bool = False,
) -> ProviderContinuationRecoveryState | None:
    """Project an assistant checkpoint into fixed user-facing recovery copy."""
    if message is None or owner_live:
        return None
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
            "Continuation warning" if state.mode == "notice" else "Interrupted tool run"
        )
        self.query_one("#console-continuation-impact", Static).update(state.impact)
        self.query_one("#console-continuation-status", Static).update(
            "Working… actions are temporarily disabled."
            if self._busy
            else (
                "The visible message is unchanged. No action is required."
                if state.mode == "notice"
                else "Choose an available action to continue."
            )
        )
        resume = self.query_one("#console-continuation-resume", Button)
        take_over = self.query_one("#console-continuation-take-over", Button)
        discard = self.query_one("#console-continuation-discard", Button)
        resume.display = state.mode == "local"
        take_over.display = state.mode == "remote"
        discard.display = state.mode != "notice"
        resume.disabled = self._busy or not state.replay_available
        take_over.disabled = self._busy or not state.replay_available
        discard.disabled = self._busy

    @on(Button.Pressed)
    def _handle_action(self, event: Button.Pressed) -> None:
        action_by_id = {
            "console-continuation-resume": "resume",
            "console-continuation-take-over": "take_over",
            "console-continuation-discard": "discard",
        }
        action = action_by_id.get(event.button.id or "")
        state = self.recovery_state
        if action is None or self._busy or state is None:
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
        **kwargs: object,
    ) -> None:
        super().__init__(session_surface_builder=session_surface_builder, **kwargs)
        self._recovery_message_builder = recovery_message_builder
        self._recovery_replay_available_builder = recovery_replay_available_builder
        self._recovery_owner_live_builder = recovery_owner_live_builder
        self._on_recovery_action = on_recovery_action

    def compose(self) -> ComposeResult:
        transcript_region = frame_console_region(
            Vertical(id="console-transcript-region", classes="console-region"),
            top=False,
            # TASK-17651: the workspace grid's own bottom border is the
            # bottom stack's single separator; the region ends flush.
            bottom=False,
        )
        with transcript_region:
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
        self.query_one(ProviderContinuationRecoveryCallout).sync_recovery(
            self._recovery_state()
        )

    async def _recover(self, action: str, message_id: str, version: int) -> bool:
        result = self._on_recovery_action(action, message_id, version)
        succeeded = bool(await result) if inspect.isawaitable(result) else bool(result)
        self.sync_recovery()
        return succeeded
