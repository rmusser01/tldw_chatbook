"""Mode-scoped library list pane for the Personas workbench."""

from __future__ import annotations

import re
from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, ListItem, ListView, Static

from .personas_messages import (
    PersonaActionRequested,
    PersonaEntityKind,
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
    kind: PersonaEntityKind
    name: str
    is_unsaved: bool = False


class PersonasLibraryPane(Vertical):
    """Search, create/import toolbar, and a keyboard-first item list.

    Rows live in a ``ListView`` (the Notes-workbench idiom): arrow keys move
    the highlight, Enter (or click) selects. Selection is explicit - mere
    highlighting never posts ``PersonaEntitySelected``, so unsaved-edit
    guards stay quiet while the user browses.
    """

    # Structure only: colors come from the app stylesheet
    # (.console-action-subdued rows, ListView ListItem.--highlight, and
    # ListItem.personas-library-row.is-active in the bundle).
    DEFAULT_CSS = """
    PersonasLibraryPane #personas-library-rows ListItem {
        width: 100%;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
    }

    PersonasLibraryPane #personas-library-rows ListItem Static {
        width: 100%;
        height: 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._row_lookup: dict[str, LibraryRow] = {}
        self._import_visible: bool = True

    def compose(self) -> ComposeResult:
        yield Static("Library", classes="destination-section personas-column-title")
        yield Input(placeholder="Search...", id="personas-library-search")
        with Horizontal(id="personas-library-toolbar", classes="ds-toolbar"):
            yield Button(
                "New",
                id="personas-library-new",
                tooltip="Create a new item in this mode.",
                classes="console-action-secondary",
            )
            yield Button(
                "Import",
                id="personas-library-import",
                tooltip="Import a character card (PNG or JSON).",
                classes="console-action-secondary",
            )
        yield ListView(id="personas-library-rows")
        yield Static("", id="personas-library-count", classes="destination-purpose")

    def set_mode(self, mode: str) -> None:
        """Show Import only where it applies (Characters mode)."""
        self._import_visible = mode == "characters"
        self.query_one("#personas-library-import", Button).display = self._import_visible

    async def update_rows(
        self,
        rows: tuple[LibraryRow, ...],
        *,
        total: int,
        noun: str,
        filtered: bool = False,
    ) -> None:
        """Replace the visible rows and count line."""
        list_view = self.query_one("#personas-library-rows", ListView)
        await list_view.clear()
        self._row_lookup = {}
        items: list[ListItem] = []
        if not rows:
            hint = "use New or Import" if self._import_visible else "use New"
            items.append(
                ListItem(
                    Static(
                        f"No {noun} yet - {hint} to add one.",
                        id="personas-library-empty",
                        markup=False,
                    ),
                    disabled=True,
                )
            )
        seen: set[str] = set()
        for row in rows:
            dom_id = _row_dom_id(row.kind, row.item_id)
            if dom_id in seen:
                suffix = 2
                while f"{dom_id}-{suffix}" in seen:
                    suffix += 1
                dom_id = f"{dom_id}-{suffix}"
            seen.add(dom_id)
            self._row_lookup[dom_id] = row
            classes = "personas-library-row console-action-subdued"
            if row.is_unsaved:
                classes += " is-unsaved"
            items.append(
                ListItem(Static(row.name, markup=False), id=dom_id, classes=classes)
            )
        await list_view.extend(items)
        count = f"{len(rows)} of {total} {noun}" if filtered else f"{total} {noun}"
        self.query_one("#personas-library-count", Static).update(count)

    def mark_active_row(self, kind: str, item_id: str) -> None:
        """Move the list highlight and the .is-active marker to one row."""
        active_id = _row_dom_id(kind, item_id)
        list_view = self.query_one("#personas-library-rows", ListView)
        for index, item in enumerate(list_view.children):
            is_active = item.id == active_id
            item.set_class(is_active, "is-active")
            if is_active:
                list_view.index = index

    @on(Input.Changed, "#personas-library-search")
    def _search_changed(self, event: Input.Changed) -> None:
        event.stop()
        self.post_message(PersonaSearchChanged(query=event.value))

    @on(Input.Submitted, "#personas-library-search")
    def _search_submitted(self, event: Input.Submitted) -> None:
        """Enter in the search box jumps into the results list."""
        event.stop()
        list_view = self.query_one("#personas-library-rows", ListView)
        list_view.focus()
        if self._row_lookup:
            list_view.index = 0

    @on(ListView.Selected, "#personas-library-rows")
    def _row_selected(self, event: ListView.Selected) -> None:
        event.stop()
        row = self._row_lookup.get(str(event.item.id or ""))
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
