"""Keep-or-clear decision for saving the current Console draft."""

from __future__ import annotations

from typing import ClassVar, Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

PromptDraftSaveChoice = Literal["keep", "clear"]


class ConsolePromptDraftSaveDialog(
    SafeModalDismissMixin, ModalScreen[PromptDraftSaveChoice | None]
):
    """Explain the non-sending save and ask what happens to the composer."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "request_safe_cancel", "Cancel", show=False)
    ]
    SAFE_MODAL_CONTENT = "#console-prompt-draft-save-dialog"

    def compose(self) -> ComposeResult:
        with Vertical(id="console-prompt-draft-save-dialog"):
            yield Static("Save draft to shelf", classes="dialog-title", markup=False)
            yield Static(
                "Save this unsent message locally. Nothing will be sent.",
                id="console-prompt-draft-save-copy",
                markup=False,
            )
            with Horizontal(id="console-prompt-draft-save-actions"):
                yield Button(
                    "Cancel",
                    id="console-prompt-draft-save-cancel",
                )
                yield Button(
                    "Save and keep",
                    id="console-prompt-draft-save-keep",
                )
                yield Button(
                    "Save and clear",
                    id="console-prompt-draft-save-clear",
                    variant="primary",
                )

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        choices: dict[str, PromptDraftSaveChoice | None] = {
            "console-prompt-draft-save-cancel": None,
            "console-prompt-draft-save-keep": "keep",
            "console-prompt-draft-save-clear": "clear",
        }
        button_id = event.button.id or ""
        if button_id not in choices:
            return
        event.stop()
        self.dismiss_safe_once(choices[button_id])

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.dismiss_safe_once(None)


__all__ = ["ConsolePromptDraftSaveDialog", "PromptDraftSaveChoice"]
