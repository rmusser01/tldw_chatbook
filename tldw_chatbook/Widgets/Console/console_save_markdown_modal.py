"""One-field path prompt for Save .md (TASK-25836).

Mirrors ``ConsoleWorkspaceRenameModal``'s contract: pushes with a callback
receiving the entered string or None on cancel/dismiss. The default value is
a slugified ``~/Downloads/<title>.md``; the screen validates the entered
path before writing.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


def markdown_filename_slug(title: str) -> str:
    """Return a filesystem-safe slug for one conversation title.

    Args:
        title: The conversation title, any length/content.

    Returns:
        A lowercase slug of alphanumerics and dashes (max 60 chars),
        never empty -- ``chat`` when the title has nothing usable.
    """
    slug = "".join(
        char if char.isalnum() else "-" for char in (title or "").lower()
    )
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")[:60]
    return slug or "chat"


class ConsoleSaveMarkdownModal(
    SafeModalDismissMixin, ModalScreen["str | None"]
):
    """Prompt for the .md destination path."""

    DEFAULT_CSS = """
    ConsoleSaveMarkdownModal {
        align: center middle;
    }
    #console-save-markdown-box {
        width: 64;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    #console-save-markdown-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #console-save-markdown-input {
        width: 1fr;
    }
    #console-save-markdown-actions {
        height: auto;
        align-horizontal: right;
        margin-top: 1;
    }
    """

    def __init__(self, *, default_path: str) -> None:
        """Create the prompt.

        Args:
            default_path: Pre-filled destination (already slugified by the
                caller via ``markdown_filename_slug``).
        """
        super().__init__()
        self._default_path = default_path

    def compose(self) -> ComposeResult:
        with Vertical(id="console-save-markdown-box"):
            yield Static("Save markdown to…", id="console-save-markdown-title")
            yield Input(
                self._default_path,
                id="console-save-markdown-input",
                placeholder="/path/to/chat.md",
            )
            with Horizontal(id="console-save-markdown-actions"):
                yield Button(
                    "Cancel", id="console-save-markdown-cancel", compact=True
                )
                yield Button(
                    "Save", id="console-save-markdown-save", compact=True
                )

    def on_mount(self) -> None:
        self.query_one("#console-save-markdown-input", Input).focus()

    @on(Button.Pressed, "#console-save-markdown-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#console-save-markdown-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Input.Submitted, "#console-save-markdown-input")
    def _submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def _submit(self) -> None:
        value = self.query_one("#console-save-markdown-input", Input).value.strip()
        if value:
            self.dismiss(value)
