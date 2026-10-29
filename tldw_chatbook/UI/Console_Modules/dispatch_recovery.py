"""Literal Textual projection for store-owned Console dispatch recovery."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import ConsoleDispatchRecoveryState


@dataclass(frozen=True, slots=True)
class ConsoleDispatchRecoveryActionPresentation:
    """One model-owned action projected without changing its vocabulary."""

    action_id: str
    label: str
    enabled: bool
    disabled_reason: str


@dataclass(frozen=True, slots=True)
class ConsoleDispatchRecoveryPresentation:
    """Markup-neutral immutable input for the recovery widget."""

    visible: bool
    visible_copy: str
    warning: str
    actions: tuple[ConsoleDispatchRecoveryActionPresentation, ...]
    markup: bool = False


DISPATCH_RECOVERY_BUSY_COPY = "Working… actions are disabled until it finishes."


def derive_dispatch_recovery_presentation(
    recovery: ConsoleDispatchRecoveryState | None,
) -> ConsoleDispatchRecoveryPresentation:
    """Project only literal UI-neutral state; never infer an action in Textual."""

    if recovery is None:
        return ConsoleDispatchRecoveryPresentation(False, "", "", ())
    # TASK-34100.5 AC#3: a turn refused before dispatch reads "Not sent".
    from tldw_chatbook.Chat.console_predispatch_block import presented_owner

    recovery = presented_owner(recovery) or recovery
    return ConsoleDispatchRecoveryPresentation(
        visible=True,
        visible_copy=recovery.visible_copy,
        # TASK-33621.6 AC#4: a claimed action says it is running.
        warning=DISPATCH_RECOVERY_BUSY_COPY if recovery.in_flight else recovery.warning,
        actions=tuple(
            ConsoleDispatchRecoveryActionPresentation(
                action_id=action.action_id.value,
                label=action.label,
                enabled=action.enabled,
                disabled_reason=action.disabled_reason,
            )
            for action in recovery.actions
        ),
    )


class ConsoleDispatchRecoveryRegion(Widget):
    """Small recovery surface whose repeated in-flight intents are inert."""

    BUNDLED_CSS = """
    ConsoleDispatchRecoveryRegion {
        display: none;
        height: auto;
        max-height: 8;
        padding: 0 1;
        overflow-y: auto;
    }

    ConsoleDispatchRecoveryRegion.-visible {
        display: block;
    }

    ConsoleDispatchRecoveryRegion Static {
        height: auto;
        width: 1fr;
    }

    ConsoleDispatchRecoveryRegion #console-dispatch-recovery-warning {
        color: $warning;
    }

    ConsoleDispatchRecoveryRegion #console-dispatch-recovery-actions {
        height: 3;
        width: 1fr;
    }

    ConsoleDispatchRecoveryRegion Button {
        min-width: 16;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        recovery: ConsoleDispatchRecoveryState | None = None,
        *,
        session_id: str = "",
        on_action: Callable[[str, str, str, Callable[[], None]], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._session_id = session_id
        self._recovery_snapshot = recovery
        self._presentation = derive_dispatch_recovery_presentation(recovery)
        self._on_action = on_action
        self._assistant_message_id = (
            recovery.assistant_message_id if recovery is not None else ""
        )
        self._model_in_flight = recovery.in_flight if recovery is not None else False
        self._pending_intent: object | None = None
        self.set_class(self._presentation.visible, "-visible")

    def compose(self) -> ComposeResult:
        presentation = self._presentation
        yield Static(
            presentation.visible_copy,
            id="console-dispatch-recovery-copy",
            markup=False,
        )
        yield Static(
            presentation.warning,
            id="console-dispatch-recovery-warning",
            markup=False,
        )
        buttons: list[Button] = []
        for action in presentation.actions:
            button = Button(
                action.label,
                id=f"console-dispatch-recovery-{action.action_id}",
                disabled=not action.enabled,
            )
            button.tooltip = action.disabled_reason or action.label
            buttons.append(button)
        yield Horizontal(*buttons, id="console-dispatch-recovery-actions")

    def sync_recovery(
        self,
        session_id: str,
        recovery: ConsoleDispatchRecoveryState | None,
    ) -> bool:
        """Replace one immutable projection and recompose only when changed."""

        updated = derive_dispatch_recovery_presentation(recovery)
        assistant_message_id = (
            recovery.assistant_message_id if recovery is not None else ""
        )
        dropped_pending = False
        if session_id != self._session_id or recovery is not self._recovery_snapshot:
            dropped_pending = self._pending_intent is not None
            self._pending_intent = None
        self._recovery_snapshot = recovery
        self._model_in_flight = recovery.in_flight if recovery is not None else False
        if (
            session_id == self._session_id
            and assistant_message_id == self._assistant_message_id
            and updated == self._presentation
        ):
            if dropped_pending:  # no recompose follows: release the busy state
                self._show_pending(None)
            return False
        self._session_id = session_id
        self._assistant_message_id = assistant_message_id
        self._presentation = updated
        self.set_class(updated.visible, "-visible")
        if self.is_mounted:
            self.refresh(recompose=True, layout=True)
        return True

    def _show_pending(self, action_id: str | None) -> None:
        """Acknowledge a pressed action at once, until its outcome arrives.

        TASK-33621.6 AC#4: Retry anyway left its row byte-identical for
        seconds until the reply came. The buttons now disable and the warning
        line says what is running; the model's own projection (in-flight
        reason, or the settled owner) replaces this on its next sync.
        """
        with contextlib.suppress(NoMatches):
            for action in self._presentation.actions:
                self.query_one(
                    f"#console-dispatch-recovery-{action.action_id}", Button
                ).disabled = action_id is not None or not action.enabled
            self.query_one("#console-dispatch-recovery-warning", Static).update(
                self._presentation.warning
                if action_id is None
                else "Discarding…"
                if action_id == "discard"
                else "Retrying… actions are disabled until it finishes."
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Forward one enabled intent; repeated presses are ignored."""

        prefix = "console-dispatch-recovery-"
        button_id = event.button.id or ""
        if (
            self._model_in_flight
            or self._pending_intent is not None
            or not button_id.startswith(prefix)
        ):
            return
        action_id = button_id.removeprefix(prefix)
        action = next(
            (
                item
                for item in self._presentation.actions
                if item.action_id == action_id and item.enabled
            ),
            None,
        )
        if action is None:
            return
        event.stop()
        if self._on_action is not None:
            token = self._pending_intent = object()
            self._show_pending(action_id)

            def complete() -> None:
                # A stale worker must not release a newer owner's/click's claim.
                if self._pending_intent is token:
                    self._pending_intent = None
                    self._show_pending(None)

            try:
                self._on_action(
                    self._session_id,
                    self._assistant_message_id,
                    action_id,
                    complete,
                )
            except Exception:
                complete()
                raise


__all__ = [
    "DISPATCH_RECOVERY_BUSY_COPY",
    "ConsoleDispatchRecoveryActionPresentation",
    "ConsoleDispatchRecoveryPresentation",
    "ConsoleDispatchRecoveryRegion",
    "derive_dispatch_recovery_presentation",
]
