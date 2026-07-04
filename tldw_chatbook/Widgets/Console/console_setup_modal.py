"""Blocking first-run setup modal for the native Console workbench.

While provider setup is incomplete (``ConsoleSetupCardState.mode == "card"``),
this widget overlays the Console workbench (rail + transcript + composer) with a
dim backdrop and a centered "Get started" card. The card shows the three live
setup steps and a single primary action that routes to provider recovery. It is
Console-scoped: it lives inside the ChatScreen (never an app-level modal), so the
top navigation tab bar stays reachable. The modal dismisses automatically the
moment readiness + model are satisfied (the guidance sync drives it).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_CARD_TITLE,
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested


CONSOLE_SETUP_MODAL_STEP_COUNT = 3
CONSOLE_SETUP_MODAL_ACTION_ID = "console-setup-modal-action"
_DEFAULT_ACTION_LABEL = "Choose model"
_DEFAULT_ACTION_TOOLTIP = "Choose the provider and model for this Console session."


def _coerce_card_state(value: object) -> ConsoleSetupCardState:
    """Guard against a transiently non-``ConsoleSetupCardState`` value."""
    if isinstance(value, ConsoleSetupCardState):
        return value
    return ConsoleSetupCardState(mode="quiet")


class ConsoleSetupModal(Vertical):
    """Console-scoped blocking overlay carrying the first-run setup card."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("id", "console-setup-modal")
        classes = kwargs.pop("classes", "")
        kwargs["classes"] = f"console-setup-modal-backdrop {classes}".strip()
        super().__init__(**kwargs)
        self._card_state = ConsoleSetupCardState(mode="quiet")
        self._action_label = _DEFAULT_ACTION_LABEL
        self._action_tooltip = _DEFAULT_ACTION_TOOLTIP
        # Hidden until a card-mode state is synced in.
        self.display = False

    @property
    def is_blocking(self) -> bool:
        """Return whether the modal is currently overlaying the workbench."""
        return self._card_state.mode == "card"

    def compose(self) -> ComposeResult:
        # Children mirror the container's blocking state so hidden-modal copy
        # never leaks into visible-text scrapes before the first guidance sync.
        blocking = self.is_blocking
        card = Vertical(
            id="console-setup-modal-card",
            classes="console-setup-modal-card",
        )
        card.display = blocking
        with card:
            title = Static(
                CONSOLE_SETUP_CARD_TITLE,
                id="console-setup-modal-title",
                classes="console-setup-modal-title",
            )
            title.display = blocking
            yield title
            for index in range(1, CONSOLE_SETUP_MODAL_STEP_COUNT + 1):
                step = self._step_at(index)
                step_row = Static(
                    self._step_text(index, step),
                    id=f"console-setup-step-{index}",
                    classes=self._step_classes(step),
                    markup=False,
                )
                step_row.display = blocking
                yield step_row
            action = Button(
                self._action_label,
                id=CONSOLE_SETUP_MODAL_ACTION_ID,
                classes="console-setup-modal-action",
                compact=True,
            )
            action.tooltip = self._action_tooltip
            action.display = blocking
            yield action

    def sync_card_state(
        self,
        card_state: ConsoleSetupCardState,
        *,
        action_label: str = "",
        action_tooltip: str = "",
    ) -> None:
        """Refresh steps + primary action and toggle overlay visibility in place."""
        self._card_state = _coerce_card_state(card_state)
        self._action_label = action_label.strip() or _DEFAULT_ACTION_LABEL
        self._action_tooltip = action_tooltip.strip() or _DEFAULT_ACTION_TOOLTIP
        blocking = self.is_blocking
        self.display = blocking
        if not self.is_mounted:
            return
        for index in range(1, CONSOLE_SETUP_MODAL_STEP_COUNT + 1):
            step = self._step_at(index)
            try:
                widget = self.query_one(f"#console-setup-step-{index}", Static)
            except Exception:
                continue
            widget.update(self._step_text(index, step))
            widget.set_classes(self._step_classes(step))
            # Own display must track blocking so hidden-modal content does not
            # leak into visible-text scrapes while the overlay is dismissed.
            widget.display = blocking
        for selector in ("#console-setup-modal-title", "#console-setup-modal-card"):
            try:
                self.query_one(selector).display = blocking
            except Exception:
                continue
        try:
            action = self.query_one(f"#{CONSOLE_SETUP_MODAL_ACTION_ID}", Button)
        except Exception:
            return
        action.label = self._action_label
        action.tooltip = self._action_tooltip
        action.display = blocking

    def focus_primary_action(self) -> None:
        """Move focus to the modal's primary action button while blocking."""
        if not (self.is_mounted and self.is_blocking):
            return
        try:
            self.query_one(f"#{CONSOLE_SETUP_MODAL_ACTION_ID}", Button).focus()
        except Exception:
            return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the primary action through the owning Workbench screen."""
        if event.button.id != CONSOLE_SETUP_MODAL_ACTION_ID:
            return
        event.stop()
        self.post_message(WorkbenchActionRequested("provider-recovery"))

    def _step_at(self, index: int) -> ConsoleSetupStep:
        steps = self._card_state.steps
        if 1 <= index <= len(steps):
            return steps[index - 1]
        return ConsoleSetupStep(state="pending", label="")

    @staticmethod
    def _step_text(index: int, step: ConsoleSetupStep) -> str:
        if not step.label:
            return ""
        text = f"{index}. {step.glyph} {step.label}"
        if step.detail:
            text = f"{text}  {step.detail}"
        return text

    @staticmethod
    def _step_classes(step: ConsoleSetupStep) -> str:
        return f"console-setup-step console-setup-step-{step.state}"
