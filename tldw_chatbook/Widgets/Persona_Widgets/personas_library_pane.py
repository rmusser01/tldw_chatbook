"""Mode-scoped library list pane for the Personas workbench."""

from __future__ import annotations

import re
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from .personas_messages import (
    PersonaActionRequested,
    PersonaEntitySelected,
    PersonaSearchChanged,
)

_ID_SAFE = re.compile(r"[^a-zA-Z0-9_-]")


def _row_dom_id(kind: str, item_id: str) -> str:
    return f"personas-library-row-{kind}-{_ID_SAFE.sub('-', str(item_id))}"


@dataclass(frozen=True)
class LibraryRow:
    """One selectable row in the workbench library list."""

    item_id: str
    kind: str  # "character" | "persona_profile"
    name: str
    is_unsaved: bool = False


class PersonasLibraryPane(Vertical):
    """Search, create/import toolbar, and selectable item rows."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._row_lookup: dict[str, LibraryRow] = {}

    def compose(self) -> ComposeResult:
        yield Static("Library", classes="destination-section personas-column-title")
        yield Input(placeholder="Search...", id="personas-library-search")
        with Horizontal(id="personas-library-toolbar", classes="ds-toolbar"):
            yield Button("New", id="personas-library-new", tooltip="Create a new item in this mode.")
            yield Button(
                "Import",
                id="personas-library-import",
                tooltip="Import a character card (PNG or JSON).",
            )
        yield VerticalScroll(id="personas-library-rows")
        yield Static("", id="personas-library-count", classes="destination-purpose")

    def set_mode(self, mode: str) -> None:
        """Show Import only where it applies (Characters mode)."""
        self.query_one("#personas-library-import", Button).display = mode == "characters"

    def update_rows(
        self,
        rows: tuple[LibraryRow, ...],
        *,
        total: int,
        noun: str,
        filtered: bool = False,
    ) -> None:
        """Replace the visible rows and count line."""
        container = self.query_one("#personas-library-rows", VerticalScroll)
        container.remove_children()
        self._row_lookup = {}
        if not rows:
            container.mount(
                Static(
                    f"No {noun} yet - use New or Import to add one.",
                    id="personas-library-empty",
                )
            )
        for row in rows:
            dom_id = _row_dom_id(row.kind, row.item_id)
            self._row_lookup[dom_id] = row
            classes = "personas-library-row"
            if row.is_unsaved:
                classes += " is-unsaved"
            container.mount(Button(row.name, id=dom_id, classes=classes))
        count = f"{len(rows)} of {total} {noun}" if filtered else f"{total} {noun}"
        self.query_one("#personas-library-count", Static).update(count)

    def mark_active_row(self, kind: str, item_id: str) -> None:
        """Apply .is-active to the selected row only."""
        active_id = _row_dom_id(kind, item_id)
        for button in self.query(".personas-library-row"):
            button.set_class(button.id == active_id, "is-active")

    @on(Input.Changed, "#personas-library-search")
    def _search_changed(self, event: Input.Changed) -> None:
        event.stop()
        self.post_message(PersonaSearchChanged(query=event.value))

    @on(Button.Pressed, ".personas-library-row")
    def _row_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        row = self._row_lookup.get(str(event.button.id or ""))
        if row is not None:
            self.post_message(
                PersonaEntitySelected(
                    entity_kind=row.kind,
                    entity_id=row.item_id,
                    entity_name=row.name,
                )
            )

    @on(Button.Pressed, "#personas-library-new")
    def _new_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaActionRequested(action="create"))

    @on(Button.Pressed, "#personas-library-import")
    def _import_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaActionRequested(action="import"))


__all__ = [
    "LibraryRow",
    "PersonasLibraryPane",
]
