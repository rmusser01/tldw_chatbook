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


def _singular_noun(noun: str) -> str:
    """Return a compact singular label for count copy."""

    if noun.endswith("ies"):
        return f"{noun[:-3]}y"
    if noun.endswith("s"):
        return noun[:-1]
    return noun


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

    PersonasLibraryPane #personas-library-rows ListItem.personas-library-recovery-row {
        height: auto;
        min-height: 6;
    }

    PersonasLibraryPane #personas-library-rows ListItem.personas-library-recovery-row Static {
        height: auto;
        min-height: 6;
        text-wrap: wrap;
        text-overflow: clip;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._row_lookup: dict[str, LibraryRow] = {}
        self._import_visible: bool = True

    def compose(self) -> ComposeResult:
        """Compose the Library pane header, search controls, and rows.

        Returns:
            Textual compose result for the Library pane.
        """
        with Horizontal(classes="console-rail-header"):
            title = Static(
                "Library",
                classes="destination-section personas-column-title console-rail-title",
            )
            title.styles.width = "1fr"
            yield title
            collapse_button = Button(
                "<",
                id="personas-library-rail-collapse",
                classes="console-rail-collapse-button",
                compact=True,
            )
            collapse_button.tooltip = "Collapse Library rail"
            yield collapse_button
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
        filtered_total_unbounded: bool = False,
        recovery_copy: str | None = None,
        recovery_id: str = "personas-library-recovery",
    ) -> None:
        """Replace the visible rows and count line.

        Args:
            rows: Selectable library rows to render when no recovery state is
                active.
            total: Total number of rows known for the current mode.
            noun: User-facing noun used in empty and count copy.
            filtered: Whether ``rows`` is a filtered subset of ``total``.
            filtered_total_unbounded: Whether filtered rows came from a
                full-library search whose total match denominator is unknown.
            recovery_copy: Optional multi-line recovery copy. When present, the
                pane renders a disabled recovery row instead of list or empty
                rows.
            recovery_id: Stable DOM id for the recovery copy widget.

        Returns:
            None.
        """
        list_view = self.query_one("#personas-library-rows", ListView)
        await list_view.clear()
        self._row_lookup = {}
        items: list[ListItem] = []
        visible_rows = () if recovery_copy else rows
        if recovery_copy:
            items.append(
                ListItem(
                    Static(recovery_copy, id=recovery_id, markup=False),
                    classes="personas-library-recovery-row",
                    disabled=True,
                )
            )
        elif not visible_rows:
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
        for row in visible_rows:
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
        if recovery_copy:
            count = f"{noun.capitalize()} unavailable"
        elif filtered and filtered_total_unbounded:
            match_word = "match" if len(rows) == 1 else "matches"
            count = (
                f"Showing {len(rows)} {_singular_noun(noun)} "
                f"{match_word} from full library"
            )
        else:
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

    def set_row_unsaved(self, kind: str | None, item_id: str | None, unsaved: bool) -> None:
        """Toggle the ``.is-unsaved`` badge without rebuilding the rows.

        Only one row (the active editing session's) may carry the badge, so
        setting it also clears any stale badge elsewhere; passing
        ``unsaved=False`` (or no kind/id) clears the badge everywhere.
        """
        target = _row_dom_id(kind, item_id) if (kind and item_id) else None
        list_view = self.query_one("#personas-library-rows", ListView)
        for item in list_view.children:
            item.set_class(unsaved and item.id == target, "is-unsaved")

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
