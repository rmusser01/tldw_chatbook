"""Literal Textual projection for store-owned Console dispatch recovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from textual.app import ComposeResult
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


def derive_dispatch_recovery_presentation(
    recovery: ConsoleDispatchRecoveryState | None,
) -> ConsoleDispatchRecoveryPresentation:
    """Project only literal UI-neutral state; never infer an action in Textual."""

    if recovery is None:
        return ConsoleDispatchRecoveryPresentation(False, "", "", ())
    return ConsoleDispatchRecoveryPresentation(
        visible=True,
        visible_copy=recovery.visible_copy,
        warning=recovery.warning,
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

    def __init__(
        self,
        recovery: ConsoleDispatchRecoveryState | None = None,
        *,
        on_action: Callable[[str], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._presentation = derive_dispatch_recovery_presentation(recovery)
        self._on_action = on_action
        self._intent_in_flight = recovery.in_flight if recovery is not None else False

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
        for action in presentation.actions:
            button = Button(
                action.label,
                id=f"console-dispatch-recovery-{action.action_id}",
                disabled=not action.enabled,
            )
            button.tooltip = action.disabled_reason or action.label
            yield button

    def sync_recovery(self, recovery: ConsoleDispatchRecoveryState | None) -> bool:
        """Replace one immutable projection and recompose only when changed."""

        updated = derive_dispatch_recovery_presentation(recovery)
        if updated == self._presentation:
            return False
        self._presentation = updated
        self._intent_in_flight = recovery.in_flight if recovery is not None else False
        if self.is_mounted:
            self.refresh(recompose=True, layout=True)
        return True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Forward one enabled intent; repeated presses are ignored."""

        prefix = "console-dispatch-recovery-"
        button_id = event.button.id or ""
        if self._intent_in_flight or not button_id.startswith(prefix):
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
        self._intent_in_flight = True
        if self._on_action is not None:
            self._on_action(action_id)


__all__ = [
    "ConsoleDispatchRecoveryActionPresentation",
    "ConsoleDispatchRecoveryPresentation",
    "ConsoleDispatchRecoveryRegion",
    "derive_dispatch_recovery_presentation",
]
