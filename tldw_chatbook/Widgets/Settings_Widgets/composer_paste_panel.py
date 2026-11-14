"""Local presentation and edit events for Console composer paste settings."""

from __future__ import annotations

from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Checkbox, Input, Static

from ..state_checkbox import StateCheckbox


class ComposerPasteSettingsPanel(Vertical):
    """Own paste controls; the Settings screen owns validation and persistence."""

    class Edited(Message):
        """A local control changed after its displayed value was initialized."""

        def __init__(
            self,
            key: Literal["collapse_large_pastes", "paste_collapse_threshold"],
            value: bool | str,
        ) -> None:
            super().__init__()
            self.key = key
            self.value = value

    def __init__(
        self, *, enabled: bool, threshold: str, default_threshold: int
    ) -> None:
        super().__init__(id="settings-console-paste-panel", classes="h-auto")
        self._enabled = enabled
        self._threshold = threshold
        self._default_threshold = default_threshold

    def compose(self) -> ComposeResult:
        yield Static("Composer paste handling", classes="destination-section")
        yield Static(
            "Collapse large pasted chunks only when they exceed the threshold.",
            id="settings-console-collapse-large-pastes-label",
        )
        yield StateCheckbox(
            "Collapse large pastes",
            value=self._enabled,
            id="settings-console-collapse-large-pastes-toggle",
            tooltip="Toggle compact display for large pasted Console chunks.",
        )
        with Horizontal(classes="settings-input-row"):
            yield Static("Threshold (chars)", classes="settings-input-label")
            yield Input(
                value=self._threshold,
                id="settings-console-paste-collapse-threshold",
                classes="settings-compact-input",
                placeholder=str(self._default_threshold),
                restrict=r"^[0-9]*$",
            )
        yield Static(
            "Normal typing stays literal. The canonical message payload is preserved.",
            id="settings-console-collapse-large-pastes-help",
        )

    def sync_values(self, *, enabled: bool, threshold: str) -> None:
        """Refresh displayed values without emitting edits or replacing inputs.

        Args:
            enabled: Current draft-aware paste-collapse preference.
            threshold: Current draft-aware threshold, including invalid input.
        """
        self._enabled = enabled
        self._threshold = threshold
        toggle = self.query_one(Checkbox)
        with toggle.prevent(Checkbox.Changed):
            toggle.value = enabled
        control = self.query_one(Input)
        if control.value != threshold:
            with control.prevent(Input.Changed):
                control.value = threshold

    @on(Checkbox.Changed)
    def _toggle_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        if event.value != event.checkbox.value or event.value == self._enabled:
            return
        self._enabled = event.value
        self.post_message(self.Edited("collapse_large_pastes", event.value))

    @on(Input.Changed)
    def _threshold_changed(self, event: Input.Changed) -> None:
        event.stop()
        if event.value != event.input.value or event.value == self._threshold:
            return
        self._threshold = event.value
        self.post_message(self.Edited("paste_collapse_threshold", event.value))
