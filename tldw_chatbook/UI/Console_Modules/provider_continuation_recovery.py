"""Inline, privacy-safe recovery for interrupted provider tool runs."""

from __future__ import annotations

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


def provider_continuation_recovery_state(
    message: ConsoleChatMessage | None,
) -> ProviderContinuationRecoveryState | None:
    """Project an assistant checkpoint into fixed user-facing recovery copy."""
    if message is None or message.provider_continuation is None:
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
        return ProviderContinuationRecoveryState(
            message.id,
            version,
            "remote",
            "This run was active elsewhere. The other device may still be running it; take over only after checking there, or discard it.",
        )
    return ProviderContinuationRecoveryState(
        message.id,
        version,
        "local",
        "The provider paused while tools may not have finished. Resume after reviewing approvals, or discard this interrupted run.",
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
        state: ProviderContinuationRecoveryState,
        on_action: ContinuationAction,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self._on_action = on_action
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Static("Interrupted tool run", id="console-continuation-title")
        yield Static(self.state.impact, id="console-continuation-impact")
        yield Static("Choose an action to continue.", id="console-continuation-status")
        with Horizontal(id="console-continuation-actions"):
            if self.state.mode == "local":
                yield Button(
                    "Resume",
                    id="console-continuation-resume",
                    variant="warning",
                )
            elif self.state.mode == "remote":
                yield Button(
                    "Take over",
                    id="console-continuation-take-over",
                    variant="warning",
                )
            yield Button("Discard", id="console-continuation-discard")

    @on(Button.Pressed)
    def _handle_action(self, event: Button.Pressed) -> None:
        action_by_id = {
            "console-continuation-resume": "resume",
            "console-continuation-take-over": "take_over",
            "console-continuation-discard": "discard",
        }
        action = action_by_id.get(event.button.id or "")
        if action is None or self._busy:
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
            group=f"provider-continuation-{self.state.message_id}",
        )

    async def _dispatch(self, action: str) -> None:
        succeeded = False
        try:
            result = self._on_action(
                action,
                self.state.message_id,
                self.state.message_version,
            )
            succeeded = (
                bool(await result) if inspect.isawaitable(result) else bool(result)
            )
        except Exception:
            succeeded = False
        if succeeded:
            self.display = False
            return
        self._busy = False
        self.query_one("#console-continuation-status", Static).update(
            "Recovery did not complete. Reload the conversation and try again."
        )
        for button in self.query(Button):
            button.disabled = False
        first = self.query(Button).first()
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
        **kwargs: object,
    ) -> None:
        super().__init__(session_surface_builder=session_surface_builder, **kwargs)
        self._recovery_message_builder = recovery_message_builder
        self._on_recovery_action = on_recovery_action

    def compose(self) -> ComposeResult:
        state = provider_continuation_recovery_state(self._recovery_message_builder())
        if state is not None:
            yield ProviderContinuationRecoveryCallout(
                state=state,
                on_action=self._on_recovery_action,
            )
        transcript_region = frame_console_region(
            Vertical(id="console-transcript-region", classes="console-region"),
            top=False,
        )
        with transcript_region:
            yield self._session_surface_builder()
