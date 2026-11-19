"""Every tool slot an unproven cleanup holds, with Release (TASK-41307).

Opened from the status line's "Held tool slots" segment. It lists agent runs,
Raw CLI commands, Undo, change review, @git references and sub-agents alike,
using the transcript's own Release action. Imported only when opened.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Console.console_cleanup_release import (
    cleanup_release_widget,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

HELD_SLOTS_COPY = (
    "These tools could not prove that what they started has stopped, so their "
    "workspace, storage and reservation stay held until you restart the app. "
    "Release frees only the tool slot for other work."
)


class ConsoleHeldToolSlotsModal(SafeModalDismissMixin, ModalScreen[None]):
    """List each held slot by what held it; Release returns only the slot."""

    BUNDLED_CSS = """
    ConsoleHeldToolSlotsModal {
        align: center middle;
    }

    #console-held-tool-slots {
        width: 76;
        height: auto;
        max-height: 32;
        border: tall $tldw-boundary;
        background: $panel;
        color: $text;
        padding: 1 2;
    }

    #console-held-tool-slots-rows {
        height: auto;
        max-height: 20;
        margin: 1 0;
    }

    .console-held-tool-slot {
        height: auto;
    }
    """

    SAFE_MODAL_CONTENT = "#console-held-tool-slots"
    BINDINGS = [("escape", "request_safe_cancel", "Close")]

    def __init__(
        self,
        slots: Mapping[str, tuple[str, bool]],
        release: Callable[[str], bool],
    ) -> None:
        """Hold the slots to list.

        Args:
            slots: Record key -> (what held the slot, True once released).
            release: Frees one key's slots; it runs off the UI thread.
        """
        super().__init__()
        self.slots = dict(slots)
        self.release = release

    def compose(self) -> ComposeResult:
        with Vertical(id="console-held-tool-slots"):
            yield Static("Held tool slots", classes="console-modal-header")
            yield Static(HELD_SLOTS_COPY, markup=False)
            with VerticalScroll(id="console-held-tool-slots-rows"):
                for key, (labels, released) in self.slots.items():
                    yield Static(
                        f"The cleanup of {labels} could not be proven.",
                        classes="console-held-tool-slot",
                        markup=False,
                    )
                    yield cleanup_release_widget(key, released, release=self.release)
            yield Button("Close", id="console-held-tool-slots-close")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "console-held-tool-slots-close":
            event.stop()
            await self.request_safe_cancel(source="button")
