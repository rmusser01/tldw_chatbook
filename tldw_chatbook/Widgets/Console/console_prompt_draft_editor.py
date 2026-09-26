"""Plain-text editor and Library promotion controls for one Draft Shelf item."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Select, Static, TextArea


class ConsolePromptDraftEditor(VerticalScroll):
    """Render one local Draft Shelf entry without owning persistence."""

    def __init__(
        self,
        *,
        content: str,
        collection_options: Sequence[tuple[str, int]] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._content = str(content)
        self._collection_options = tuple(collection_options)
        self._delete_armed = False

    def compose(self) -> ComposeResult:
        yield Static(
            "Edit the saved draft, insert it at the current composer caret, or "
            "promote it to a reusable local Library Prompt.",
            id="console-prompt-draft-editor-copy",
            markup=False,
        )
        yield TextArea(self._content, id="console-prompt-draft-content")
        yield Static("", id="console-prompt-draft-status", markup=False)
        with Horizontal(id="console-prompt-draft-actions"):
            yield Button(
                "Update draft",
                id="console-prompt-draft-update",
                variant="primary",
            )
            yield Button("Insert at caret", id="console-prompt-draft-insert")
            yield Button(
                "Delete draft",
                id="console-prompt-draft-delete",
                variant="error",
            )
        yield Static(
            "Save a copy to Library > Prompts",
            id="console-prompt-draft-promote-title",
            markup=False,
        )
        with Horizontal(id="console-prompt-draft-promote-fields"):
            yield Input(
                placeholder="Prompt name",
                id="console-prompt-draft-library-name",
            )
            yield Select(
                self._collection_options,
                value=Select.NULL,
                allow_blank=True,
                prompt="No collection",
                id="console-prompt-draft-library-collection",
            )
            yield Button(
                "Save to Library",
                id="console-prompt-draft-promote",
            )

    @property
    def content(self) -> str:
        """Return the current exact editor text."""

        return self.query_one("#console-prompt-draft-content", TextArea).text

    @property
    def library_name(self) -> str:
        """Return the trimmed promotion name."""

        return self.query_one("#console-prompt-draft-library-name", Input).value.strip()

    @property
    def selected_collection_id(self) -> int | None:
        """Return the selected local collection, if any."""

        value = self.query_one("#console-prompt-draft-library-collection", Select).value
        return value if type(value) is int else None

    @property
    def delete_armed(self) -> bool:
        """Return whether the next delete press confirms deletion."""

        return self._delete_armed

    def show_status(self, message: str, *, error: bool = False) -> None:
        """Show concise action feedback in the editor."""

        status = self.query_one("#console-prompt-draft-status", Static)
        status.update(message)
        status.set_class(error, "error")

    def arm_delete(self) -> None:
        """Require the immediately following delete press to confirm."""

        self._delete_armed = True
        button = self.query_one("#console-prompt-draft-delete", Button)
        button.label = "Press again to delete"
        self.show_status("Deletion is permanent. Press Delete again to confirm.")

    def reset_delete_confirmation(self) -> None:
        """Cancel a pending delete confirmation after any intervening action."""

        if not self._delete_armed:
            return
        self._delete_armed = False
        self.query_one("#console-prompt-draft-delete", Button).label = "Delete draft"


__all__ = ["ConsolePromptDraftEditor"]
