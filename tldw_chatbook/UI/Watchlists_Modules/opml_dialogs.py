"""Modal dialogs for watchlist OPML import/export."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea


class OpmlImportDialog(ModalScreen[str | None]):
    """Modal dialog that prompts the user for OPML XML to import."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        with Vertical(id="opml-import-dialog", classes="opml-dialog"):
            yield Static("Import OPML", classes="dialog-title")
            yield TextArea("", id="opml-import-text")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Import", id="opml-import-confirm", variant="success")
                yield Button("Cancel", id="opml-import-cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        if button_id == "opml-import-confirm":
            text_area = self.query_one("#opml-import-text", TextArea)
            self.dismiss(text_area.text)
        elif button_id == "opml-import-cancel":
            self.dismiss(None)
        event.stop()


class OpmlExportDialog(ModalScreen[None]):
    """Modal dialog that displays OPML XML exported from watchlist sources."""

    BINDINGS = []

    def __init__(self, xml_text: str) -> None:
        super().__init__()
        self.xml_text = xml_text

    def compose(self) -> ComposeResult:
        with Vertical(id="opml-export-dialog", classes="opml-dialog"):
            yield Static("Export OPML", classes="dialog-title")
            yield TextArea(self.xml_text, id="opml-export-text", read_only=True)
            with Horizontal(classes="dialog-buttons"):
                yield Button("Close", id="opml-export-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if str(event.button.id) == "opml-export-close":
            self.dismiss(None)
        event.stop()
