"""Console-native composer action row."""

from __future__ import annotations

import textwrap
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Static


class ConsoleComposerBar(Horizontal):
    """Expose Console-owned composer actions while reusing active chat sessions."""

    DEFAULT_STATUS = "No active Console session."
    DRAFT_PLACEHOLDER = "Type message or command"
    MIN_DRAFT_ROWS = 1
    MAX_DRAFT_ROWS = 4
    COMPOSER_CHROME_ROWS = 4
    FALLBACK_DRAFT_WIDTH = 80

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.can_focus = True
        self.styles.height = 5
        self.styles.min_height = 5
        self.styles.max_height = self.MAX_DRAFT_ROWS + self.COMPOSER_CHROME_ROWS

    @staticmethod
    def _bounded_button(label: str, *, width: int, **kwargs: Any) -> Button:
        kwargs.setdefault("compact", True)
        button = Button(label, **kwargs)
        button.styles.width = width
        button.styles.min_width = width
        button.styles.height = 1
        button.styles.min_height = 1
        return button

    def draft_text(self) -> str:
        """Return the visible native Console draft."""
        try:
            return self.query_one("#console-command-input", Input).value
        except NoMatches:
            return ""

    @classmethod
    def _wrap_draft_lines(cls, text: str, width: int) -> list[str]:
        """Return wrapped draft lines for the visible bounded composer."""
        width = max(8, width)
        source_lines = text.splitlines() or [text]
        wrapped_lines: list[str] = []
        for line in source_lines:
            if not line:
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                    drop_whitespace=False,
                    replace_whitespace=False,
                )
                or [""]
            )
        return wrapped_lines or [""]

    @classmethod
    def _visible_draft_lines(cls, text: str, width: int) -> list[str]:
        """Return the bounded visible draft lines, biased toward the caret end."""
        lines = cls._wrap_draft_lines(text, width)
        if len(lines) <= cls.MAX_DRAFT_ROWS:
            return lines
        visible_lines = lines[-cls.MAX_DRAFT_ROWS :]
        first_line = visible_lines[0].lstrip()
        visible_lines[0] = f"... {first_line}" if first_line else "..."
        return visible_lines

    @classmethod
    def _draft_renderable(cls, text: str, *, width: int = FALLBACK_DRAFT_WIDTH) -> Text:
        if text:
            lines = cls._visible_draft_lines(text, width)
            return Text("\n".join(lines))
        return Text(cls.DRAFT_PLACEHOLDER, style="dim")

    @classmethod
    def _visible_draft_row_count(cls, text: str, width: int) -> int:
        if not text:
            return cls.MIN_DRAFT_ROWS
        return max(
            cls.MIN_DRAFT_ROWS,
            min(cls.MAX_DRAFT_ROWS, len(cls._wrap_draft_lines(text, width))),
        )

    def _draft_render_width(self) -> int:
        try:
            visible_draft = self.query_one("#console-command-visible-text", Static)
        except NoMatches:
            return self.FALLBACK_DRAFT_WIDTH
        width = visible_draft.region.width or self.FALLBACK_DRAFT_WIDTH
        # The visible draft keeps one-column horizontal padding on both sides.
        return max(8, width - 2)

    def _apply_draft_height(self, row_count: int) -> None:
        row_count = max(self.MIN_DRAFT_ROWS, min(self.MAX_DRAFT_ROWS, row_count))
        composer_height = row_count + self.COMPOSER_CHROME_ROWS
        try:
            visible_draft = self.query_one("#console-command-visible-text", Static)
            visible_draft.styles.height = row_count
            visible_draft.styles.min_height = row_count
            visible_draft.styles.max_height = self.MAX_DRAFT_ROWS
        except NoMatches:
            pass
        self.styles.height = composer_height
        self.styles.min_height = self.MIN_DRAFT_ROWS + self.COMPOSER_CHROME_ROWS
        self.styles.max_height = self.MAX_DRAFT_ROWS + self.COMPOSER_CHROME_ROWS
        self.refresh(layout=True)

    def _refresh_visible_draft(self) -> None:
        try:
            draft = self.query_one("#console-command-input", Input).value
            width = self._draft_render_width()
            row_count = self._visible_draft_row_count(draft, width)
            self.query_one("#console-command-visible-text", Static).update(
                self._draft_renderable(draft, width=width)
            )
            self._apply_draft_height(row_count)
        except NoMatches:
            return

    def on_mount(self) -> None:
        self._refresh_visible_draft()

    def on_resize(self, event: Any) -> None:
        self._refresh_visible_draft()

    def load_draft(self, text: str) -> None:
        """Replace the visible native Console draft."""
        try:
            self.query_one("#console-command-input", Input).value = text
        except NoMatches:
            return
        self._refresh_visible_draft()

    def clear_draft(self) -> None:
        """Clear the visible native Console draft."""
        self.load_draft("")

    def insert_text(self, text: str) -> None:
        """Append user-entered text to the visible Console draft."""
        if not text:
            return
        self.load_draft(f"{self.draft_text()}{text}")

    def delete_left(self) -> None:
        """Delete the last draft character for simple terminal-style editing."""
        self.load_draft(self.draft_text()[:-1])

    def sync_session_data(self, session_data: Any | None) -> None:
        """Refresh composer status copy from the active chat session contract."""
        if session_data is None:
            status = self.DEFAULT_STATUS
        else:
            title = getattr(session_data, "title", None) or "Untitled session"
            backend = getattr(session_data, "runtime_backend", None) or "local"
            assistant = getattr(session_data, "assistant_id", None) or getattr(
                session_data,
                "character_name",
                None,
            ) or "General"
            workspace = getattr(session_data, "workspace_id", None) or "global"
            status = (
                f"Active session: {title} | Backend: {backend} | "
                f"Assistant: {assistant} | Scope: {workspace}"
            )

        try:
            self.query_one("#console-composer-status", Static).update(escape(status))
        except NoMatches:
            return

    def compose(self) -> ComposeResult:
        title = Static("Composer:", id="console-composer-title", classes="destination-section")
        title.styles.width = 10
        title.styles.min_width = 10
        yield title
        visible_draft = Static(
            self._draft_renderable(""),
            id="console-command-visible-text",
            classes="console-command-visible-text",
        )
        visible_draft.can_focus = True
        yield visible_draft
        command_input = Input(
            value="",
            id="console-command-input",
            classes="console-command-input",
            placeholder="Type message or command",
            compact=True,
        )
        command_input.can_focus = False
        command_input.disabled = True
        command_input.styles.display = "none"
        command_input.styles.width = 0
        command_input.styles.min_width = 0
        command_input.styles.height = 1
        command_input.styles.min_height = 1
        yield command_input
        status = Static(
            self.DEFAULT_STATUS,
            id="console-composer-status",
            classes="console-composer-status console-hidden-control",
        )
        status.styles.display = "none"
        status.styles.width = 0
        status.styles.min_width = 0
        status.styles.height = 0
        status.styles.min_height = 0
        yield status
        yield self._bounded_button(
            "Send",
            width=8,
            id="console-send-message",
            classes="destination-action-button console-send-button",
            variant="primary",
            tooltip="Send the active Console session draft.",
        )
        yield self._bounded_button(
            "Stop",
            width=8,
            id="console-stop-generation",
            classes="destination-action-button console-stop-button",
            tooltip="Stop generation in the active Console session.",
        )
        yield self._bounded_button(
            "Attach",
            width=10,
            id="console-attach-context",
            classes="destination-action-button console-attach-button",
            tooltip="Attach files or context through the active Console session.",
        )
        yield self._bounded_button(
            "Save Chatbook",
            width=22,
            id="console-save-chatbook",
            classes="destination-action-button console-save-chatbook-button",
            tooltip="Compatibility adapter: save Chatbook export is still owned by Artifacts/Chatbooks.",
        )
