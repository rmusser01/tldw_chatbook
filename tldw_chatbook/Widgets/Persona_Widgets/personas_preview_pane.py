"""Ephemeral preview-conversation pane for the Personas workbench.

The pane renders a collapsible, in-memory transcript so a character or
persona can be test-driven without persisting anything. The owning screen
seeds the greeting, runs the provider call, and appends replies; nothing
here touches a database.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from .personas_pane_messages import (
    PreviewOpenInConsoleRequested,
    PreviewReplyRequested,
    PreviewResetRequested,
)


class PersonasPreviewPane(Vertical):
    """Collapsible test conversation: transcript, input, and actions."""

    DEFAULT_CSS = """
    PersonasPreviewPane {
        height: auto;
        max-height: 60%;
    }

    PersonasPreviewPane #personas-preview-toggle {
        width: 100%;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    PersonasPreviewPane #personas-preview-body {
        height: auto;
    }

    PersonasPreviewPane #personas-preview-status {
        height: 1;
        min-height: 1;
    }

    PersonasPreviewPane #personas-preview-transcript {
        height: auto;
        max-height: 10;
        min-height: 2;
    }

    PersonasPreviewPane .ds-toolbar {
        height: 1;
        min-height: 1;
    }

    PersonasPreviewPane .ds-toolbar Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        margin-right: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._greeting: str = ""
        # Rendered transcript line texts ("you: ..." / "character: ...");
        # the source of truth for transcript_text().
        self._lines: list[str] = []

    def compose(self) -> ComposeResult:
        yield Button(
            "Preview conversation",
            id="personas-preview-toggle",
            tooltip="Test the selected character or persona in an ephemeral conversation; nothing is saved.",
            classes="console-action-subdued",
        )
        with Vertical(id="personas-preview-body"):
            yield Static("", id="personas-preview-status")
            yield VerticalScroll(id="personas-preview-transcript")
            yield Input(placeholder="Test message...", id="personas-preview-input")
            with Horizontal(classes="ds-toolbar"):
                yield Button(
                    "Test Reply",
                    id="personas-preview-test-reply",
                    classes="console-action-secondary",
                )
                yield Button(
                    "Reset",
                    id="personas-preview-reset",
                    classes="console-action-subdued",
                )
                yield Button(
                    "Open in Console",
                    id="personas-preview-open-console",
                    classes="console-action-subdued",
                )

    def on_mount(self) -> None:
        self.query_one("#personas-preview-body").display = False

    # ===== Public API =====

    def expand(self) -> None:
        """Show the collapsible body."""
        self.query_one("#personas-preview-body").display = True

    async def seed_greeting(self, text: str) -> None:
        """Store the greeting and restart the transcript from it.

        Also clears the status line: the greeting changes when the selection
        does, and any prior run status belongs to the old selection.
        """
        self._greeting = str(text or "")
        self.set_status("")
        await self._render_seed_lines()

    def append_user(self, text: str) -> None:
        """Append a "you: ..." transcript line."""
        self._append_line(f"you: {text}", "personas-preview-line-you")

    def append_reply(self, text: str) -> None:
        """Append a "character: ..." transcript line."""
        self._append_line(f"character: {text}", "personas-preview-line-character")

    async def reset(self) -> None:
        """Drop the conversation back to the seeded greeting; clear status."""
        self.set_status("")
        await self._render_seed_lines()

    def set_status(self, text: str) -> None:
        """Update the readable status line."""
        self.query_one("#personas-preview-status", Static).update(str(text or ""))

    def transcript_text(self) -> str:
        """The visible transcript as plain text, one line per message."""
        return "\n".join(self._lines)

    # ===== Internals =====

    async def _render_seed_lines(self) -> None:
        """Replace the transcript with the greeting line (or nothing)."""
        self._lines = []
        container = self.query_one("#personas-preview-transcript", VerticalScroll)
        await container.remove_children()
        widgets: list[Static] = []
        if self._greeting:
            line = f"character: {self._greeting}"
            self._lines.append(line)
            widgets.append(
                # markup=False: greeting text must render literally, never as
                # Rich markup (unmatched tags raise MarkupError at render).
                Static(
                    line,
                    classes="personas-preview-line personas-preview-line-character",
                    markup=False,
                )
            )
        if widgets:
            await container.mount_all(widgets)

    def _append_line(self, line: str, role_class: str) -> None:
        self._lines.append(line)
        # markup=False: user/character text must render literally, never as
        # Rich markup (unmatched tags raise MarkupError at render).
        self.query_one("#personas-preview-transcript", VerticalScroll).mount(
            Static(line, classes=f"personas-preview-line {role_class}", markup=False)
        )

    # ===== Events =====

    @on(Button.Pressed, "#personas-preview-toggle")
    def _handle_toggle(self, event: Button.Pressed) -> None:
        event.stop()
        body = self.query_one("#personas-preview-body")
        body.display = not body.display

    @on(Button.Pressed, "#personas-preview-test-reply")
    def _handle_test_reply(self, event: Button.Pressed) -> None:
        event.stop()
        field = self.query_one("#personas-preview-input", Input)
        text = field.value.strip()
        if not text:
            return
        field.value = ""
        self.append_user(text)
        self.post_message(PreviewReplyRequested(text))

    @on(Button.Pressed, "#personas-preview-reset")
    async def _handle_reset(self, event: Button.Pressed) -> None:
        event.stop()
        await self.reset()
        self.post_message(PreviewResetRequested())

    @on(Button.Pressed, "#personas-preview-open-console")
    def _handle_open_console(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PreviewOpenInConsoleRequested())


__all__ = ["PersonasPreviewPane"]
