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

from ...Utils.input_validation import validate_text_input
from .personas_pane_messages import (
    PreviewOpenInConsoleRequested,
    PreviewReplyRequested,
    PreviewResetRequested,
)

#: Boundary cap for one preview test message; generous for a test exchange
#: while keeping pathological pastes out of the provider request.
PREVIEW_MESSAGE_MAX_CHARS = 4000


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
        # In-progress streamed reply line (begin_reply/append_reply_chunk).
        # finalize_reply commits it; discard_partial_reply removes it.
        self._partial_widget: Static | None = None
        self._partial_index: int | None = None
        self._partial_text: str = ""

    def compose(self) -> ComposeResult:
        yield Button(
            "Preview conversation",
            id="personas-preview-toggle",
            tooltip="Test the selected character or persona in an ephemeral conversation; nothing is saved.",
            classes="console-action-subdued",
        )
        with Vertical(id="personas-preview-body"):
            yield VerticalScroll(id="personas-preview-transcript")
            # The status line is a status region adjacent to the input, kept
            # BELOW the transcript so provider/error messages never render
            # above the chronological greeting -> you -> character history.
            yield Static("", id="personas-preview-status")
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

    def refresh_greeting_seed(self, text: str) -> None:
        """Update the greeting Reset restores without changing the transcript.

        Args:
            text: Greeting text that future Reset actions should restore. Empty
                values clear the stored greeting seed.
        """
        self._greeting = str(text or "")

    def append_user(self, text: str) -> None:
        """Append a "you: ..." transcript line."""
        self._append_line(f"you: {text}", "personas-preview-line-you")

    def append_reply(self, text: str) -> None:
        """Append a complete "character: ..." transcript line in one go."""
        self._append_line(f"character: {text}", "personas-preview-line-character")

    def begin_reply(self) -> None:
        """Start a streamed "character: ..." line, grown by append_reply_chunk.

        Only one streamed line can be in progress; beginning a new one while
        another is open simply re-targets the partial tracking (the owner is
        expected to finalize or discard first).
        """
        self._partial_text = ""
        line = "character:"
        widget = Static(
            line,
            classes="personas-preview-line personas-preview-line-character",
            # markup=False: streamed text must render literally, never as Rich
            # markup (unmatched tags raise MarkupError at render).
            markup=False,
        )
        self._partial_widget = widget
        self._partial_index = len(self._lines)
        self._lines.append(line)
        self.query_one("#personas-preview-transcript", VerticalScroll).mount(widget)

    def append_reply_chunk(self, text: str) -> None:
        """Grow the in-progress streamed reply line in place."""
        if self._partial_widget is None:
            self.begin_reply()
        self._partial_text += str(text)
        line = f"character: {self._partial_text}"
        if self._partial_index is not None and self._partial_index < len(self._lines):
            self._lines[self._partial_index] = line
        self._partial_widget.update(line)

    def finalize_reply(self) -> None:
        """Commit the streamed line: it is now a permanent transcript entry."""
        self._partial_widget = None
        self._partial_index = None
        self._partial_text = ""

    async def discard_partial_reply(self) -> None:
        """Remove an in-progress streamed line (stale/error); no-op otherwise."""
        widget, index = self._partial_widget, self._partial_index
        self.finalize_reply()
        if widget is None:
            return
        if index is not None and index < len(self._lines):
            del self._lines[index]
        try:
            await widget.remove()
        except Exception:
            # Tolerate a widget already removed by a transcript reseed.
            pass

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
        # remove_children below also removes any in-progress streamed line, so
        # the partial tracking must be dropped with it (a later discard must
        # not delete a line from the freshly seeded transcript).
        self.finalize_reply()
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

    def _submit_preview_message(self) -> None:
        """Shared Test Reply path: validate, clear, append, request a reply.

        Rejections keep the draft in the input (so it stays editable) and
        surface a readable status instead of posting the message.
        """
        field = self.query_one("#personas-preview-input", Input)
        text = field.value.strip()
        if not text:
            return
        if len(text) > PREVIEW_MESSAGE_MAX_CHARS:
            self.set_status(
                f"Message too long (max {PREVIEW_MESSAGE_MAX_CHARS} characters)."
            )
            return
        if not validate_text_input(
            text, max_length=PREVIEW_MESSAGE_MAX_CHARS, allow_html=False
        ):
            self.set_status("Message contains content that cannot be sent.")
            return
        field.value = ""
        self.set_status("")
        self.append_user(text)
        self.post_message(PreviewReplyRequested(text))

    @on(Button.Pressed, "#personas-preview-test-reply")
    def _handle_test_reply(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit_preview_message()

    @on(Input.Submitted, "#personas-preview-input")
    def _handle_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the input submits exactly like the Test Reply button."""
        event.stop()
        self._submit_preview_message()

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
