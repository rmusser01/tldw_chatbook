"""Mode-scoped library list pane for the Personas workbench."""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Resize
from textual.widgets import Button, Input, ListItem, ListView, Static

from .personas_messages import (
    PersonaActionRequested,
    PersonaEntityKind,
    PersonaEntitySelected,
    PersonaMarksChanged,
    PersonaPageChanged,
    PersonaSearchChanged,
    PersonaSortCycleRequested,
    PersonaTagFilterRequested,
)

_ID_SAFE = re.compile(r"[^a-zA-Z0-9_-]")

logger = logger.bind(module="PersonasLibraryPane")

#: Columns one toolbar button occupies beyond its label: `padding: 0 1`
#: plus the `margin-right: 1` gap from the pane CSS below.
_TOOLBAR_BUTTON_CHROME_COLS = 3


def _row_dom_id(kind: str, item_id: str) -> str:
    return f"personas-library-row-{kind}-{_ID_SAFE.sub('-', str(item_id))}"


def _singular_noun(noun: str) -> str:
    """Return a compact singular label for count copy."""

    if noun.endswith("ies"):
        return f"{noun[:-3]}y"
    if noun.endswith("s"):
        return noun[:-1]
    return noun


def _noun_for_count(count: int, noun: str) -> str:
    """Return ``noun`` in the grammatical number matching ``count``.

    task-445: the count line used the plural ``noun`` unconditionally, so a
    total of exactly one item read "1 characters".
    """

    return _singular_noun(noun) if count == 1 else noun


@dataclass(frozen=True)
class LibraryRow:
    """One selectable row in the workbench library list."""

    item_id: str
    kind: PersonaEntityKind
    name: str
    is_unsaved: bool = False
    meta: str | None = None


class PersonasLibraryPane(Vertical):
    """Search, create/import toolbar, and a keyboard-first item list.

    Rows live in a ``ListView`` (the Notes-workbench idiom): arrow keys move
    the highlight, Enter (or click) selects. Selection is explicit - mere
    highlighting never posts ``PersonaEntitySelected``, so unsaved-edit
    guards stay quiet while the user browses.
    """

    BINDINGS = [
        ("space", "toggle_highlighted", "Toggle on/off"),
        # F-040: m marks rows for bulk delete/export; s cycles the sort
        # (both no-op outside their applicable modes/rows).
        ("m", "toggle_mark", "Mark row"),
        ("s", "cycle_sort", "Cycle sort"),
    ]

    # Structure only: colors come from the app stylesheet
    # (.console-action-subdued rows, ListView ListItem.--highlight, and
    # ListItem.personas-library-row.is-active in the bundle).
    BUNDLED_CSS = """
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

    /* The generic Button min-width:16 default would push "next" past the
       pane's narrow width (2fr of the workbench split); pin the prev/next
       buttons to a compact width and let the page-info Static fill the rest
       so the bar never overflows its container. */
    PersonasLibraryPane #personas-library-pagebar Button {
        width: 5;
        min-width: 5;
    }

    PersonasLibraryPane #personas-library-page-info {
        width: 1fr;
        text-align: center;
    }

    /* F-030: toolbar buttons size to their labels. The Textual Button
       default min-width:16 let "New" alone fill a narrow pane and clipped
       Import/Duplicate/Tag off the right edge at supported widths. */
    PersonasLibraryPane #personas-library-toolbar Button,
    PersonasLibraryPane #personas-library-filterbar Button {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        margin-right: 1;
    }

    /* F-030 narrow panes: stack each bar vertically so every action wraps
       onto its own full-width row (a Textual Horizontal never wraps, so one
       over-wide row would clip instead). */
    PersonasLibraryPane.personas-library-stacked-controls #personas-library-toolbar,
    PersonasLibraryPane.personas-library-stacked-controls #personas-library-filterbar {
        layout: vertical;
        height: auto;
    }

    PersonasLibraryPane.personas-library-stacked-controls #personas-library-toolbar Button,
    PersonasLibraryPane.personas-library-stacked-controls #personas-library-filterbar Button {
        width: 100%;
        margin-right: 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._row_lookup: dict[str, LibraryRow] = {}
        self._import_visible: bool = True
        # F-040: marked (multi-selected) rows for bulk delete/export, as row
        # dom ids; pruned to the rendered rows on every update_rows.
        self._marked_ids: set[str] = set()
        self._sort_visible: bool = True
        # The count line's filter-state text from the last update_rows; the
        # "N marked" summary overrides it while marks exist.
        self._base_count_text: str = ""

    def on_mount(self) -> None:
        """Initialize control visibility for default characters mode.

        ``PersonasScreen.on_mount`` calls ``set_mode`` immediately after this
        pane mounts, so this only sets the very first paint; Duplicate
        applies to characters (task-443), so it starts visible here too.
        """
        self.query_one("#personas-library-duplicate", Button).display = True
        self.query_one("#personas-library-pagebar").display = False
        self._sync_control_layout()

    def on_resize(self, event: Resize) -> None:
        """Re-wrap the toolbar bars when the pane width changes (F-030)."""
        self._sync_control_layout()

    def _required_toolbar_row_width(self) -> int:
        """Widest single-row width the currently visible bar buttons need.

        Derived from labels, not rendered sizes, so toggling the stacked
        class never changes the measurement (no layout oscillation).
        """
        required = 0
        for bar_id in ("#personas-library-toolbar", "#personas-library-filterbar"):
            row = 0
            for button in self.query(f"{bar_id} Button").results():
                if not button.display:
                    continue
                row += len(str(button.label)) + _TOOLBAR_BUTTON_CHROME_COLS
            required = max(required, row)
        return required

    def _sync_control_layout(self) -> None:
        """Stack the toolbar bars when one row would clip actions (F-030).

        A pane narrower than the single-row width clipped the rightmost
        buttons off-screen (at 100x30 that hid Import -- the roleplay
        onboarding path). Stacking switches each bar to a vertical layout so
        every action wraps onto its own row instead.
        """
        width = self.content_size.width
        if width <= 0:
            # Not laid out yet (on_mount); on_resize re-syncs once sized.
            return
        try:
            required = self._required_toolbar_row_width()
        except Exception:
            # Widths are label-derived, so a failure here is a teardown or
            # pre-compose race - but never something to swallow silently:
            # leaving the bars clipped with no trace was the bug under
            # review. Debug level, matching the teardown-race idiom used
            # across the personas widgets (the layout simply keeps its
            # previous state and re-syncs on the next resize).
            logger.opt(exception=True).debug(
                "PersonasLibraryPane toolbar width measurement failed; "
                "keeping the previous control layout."
            )
            return
        self.set_class(width < required, "personas-library-stacked-controls")

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
                "New Actor Pack",
                id="personas-library-new-actor-pack",
                tooltip=(
                    "Create a local actor with a required portrait and "
                    "portable identity."
                ),
                classes="console-action-secondary",
            )
            yield Button(
                "Import",
                id="personas-library-import",
                tooltip="Import a character card (PNG or JSON).",
                classes="console-action-secondary",
            )
            yield Button(
                "Duplicate",
                id="personas-library-duplicate",
                tooltip="Duplicate the selected item.",
                classes="console-action-secondary",
            )
        with Horizontal(id="personas-library-filterbar", classes="ds-toolbar"):
            yield Button(
                "Sort: Name",
                id="personas-library-sort",
                tooltip="Cycle the list sort order. (s)",
                classes="console-action-secondary",
            )
            yield Button(
                "Tag: All",
                id="personas-library-tag",
                tooltip="Filter characters by tag.",
                classes="console-action-secondary",
            )
        yield ListView(id="personas-library-rows")
        with Horizontal(id="personas-library-pagebar", classes="ds-toolbar"):
            yield Button(
                "<",
                id="personas-library-prev",
                compact=True,
                classes="console-action-secondary",
            )
            yield Static(
                "", id="personas-library-page-info", classes="destination-purpose"
            )
            yield Button(
                ">",
                id="personas-library-next",
                compact=True,
                classes="console-action-secondary",
            )
        yield Static("", id="personas-library-count", classes="destination-purpose")

    def set_mode(self, mode: str) -> None:
        """Gate the library toolbar's buttons for the active workbench mode.

        Import and Duplicate render for characters/dictionaries/lore (with a
        mode-appropriate Import tooltip); personas mode hides both (task-443).

        Args:
            mode: The active workbench mode id (``characters``/``personas``/
                ``dictionaries``/``lore``).
        """
        self._import_visible = mode in ("characters", "dictionaries", "lore")
        import_button = self.query_one("#personas-library-import", Button)
        import_button.display = self._import_visible
        if mode == "dictionaries":
            import_button.tooltip = "Import a dictionary (JSON or Markdown)."
        elif mode == "lore":
            import_button.tooltip = "Import a world book (JSON)."
        else:
            import_button.tooltip = "Import a character card (PNG or JSON)."
        self.query_one("#personas-library-duplicate", Button).display = mode in (
            "characters",
            "dictionaries",
            "lore",
        )
        self.query_one("#personas-library-new-actor-pack", Button).display = mode in (
            "characters",
            "personas",
        )
        sort_visible = mode in ("characters", "personas")
        self._sort_visible = sort_visible
        self.query_one("#personas-library-sort", Button).display = sort_visible
        self.query_one("#personas-library-tag", Button).display = mode == "characters"
        if not sort_visible:
            # dict/lore never paginate - keep the page bar hidden.
            self.query_one("#personas-library-pagebar").display = False
        # Which buttons render changed, so the single-row fit changed too.
        self._sync_control_layout()
        # Marks are mode-scoped: a mode switch drops them (F-040).
        self.clear_marks()

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
        page_offset: int | None = None,
        page_size: int | None = None,
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
            page_offset: Zero-based offset of ``rows`` within ``total``. When
                given together with ``page_size``, the pane renders a page bar
                instead of the plain count line. ``None`` (the default)
                preserves the pre-paging behavior for callers (dictionaries,
                lore) that never page.
            page_size: Page window size paired with ``page_offset``.

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
            # F-040: marked rows carry a glyph prefix on the name line.
            name_text = (
                f"● {row.name}" if dom_id in self._marked_ids else row.name
            )
            if row.meta:
                item = ListItem(
                    Vertical(
                        Static(name_text, markup=False),
                        Static(
                            row.meta,
                            markup=False,
                            classes="personas-library-row-meta destination-purpose",
                        ),
                    ),
                    id=dom_id,
                    classes=classes,
                )
                # Inline override, not CSS: app-level .console-action-subdued pins height:1 and
                # Textual ranks app CSS above widget DEFAULT_CSS regardless of specificity/!important;
                # inline styles beat both.
                item.styles.height = 2
                items.append(item)
            else:
                items.append(
                    ListItem(Static(name_text, markup=False), id=dom_id, classes=classes)
                )
        await list_view.extend(items)
        # F-040: a mark never outlives its row - a refresh that drops a
        # marked row drops the mark with it.
        pruned = self._marked_ids - seen
        if pruned:
            self._marked_ids -= pruned
        pagebar = self.query_one("#personas-library-pagebar")
        paginated = page_offset is not None and page_size is not None
        if recovery_copy:
            pagebar.display = False
            self._base_count_text = f"{noun.capitalize()} unavailable"
        elif paginated and total > page_size:
            start = page_offset + 1 if total else 0
            end = page_offset + len(rows)
            self.query_one("#personas-library-page-info", Static).update(
                f"{start}-{end} of {total} {noun}"
            )
            self.query_one("#personas-library-prev", Button).disabled = page_offset <= 0
            self.query_one("#personas-library-next", Button).disabled = (
                page_offset + page_size >= total
            )
            pagebar.display = True
            self._base_count_text = ""
        else:
            pagebar.display = False
            if filtered and filtered_total_unbounded:
                match_word = "match" if len(rows) == 1 else "matches"
                self._base_count_text = (
                    f"Showing {len(rows)} {_singular_noun(noun)} "
                    f"{match_word} from full library"
                )
            elif filtered:
                self._base_count_text = (
                    f"{len(rows)} of {total} {_noun_for_count(total, noun)}"
                )
            else:
                # F-033: the plain total renders once, in the screen's merged
                # purpose line ("Characters — who the AI plays · N") - the
                # pane's own count line only speaks for filtered states.
                self._base_count_text = ""
        self._sync_marked_count_line()
        if pruned:
            self._post_marks_changed()

    def mark_active_row(self, kind: str, item_id: str) -> None:
        """Move the list highlight and the .is-active marker to one row."""
        active_id = _row_dom_id(kind, item_id)
        list_view = self.query_one("#personas-library-rows", ListView)
        for index, item in enumerate(list_view.children):
            is_active = item.id == active_id
            item.set_class(is_active, "is-active")
            if is_active:
                list_view.index = index

    def highlight_row(self, kind: str, item_id: str) -> None:
        """Move only the ListView cursor to one row (no active-marker change)."""
        target = _row_dom_id(kind, item_id)
        list_view = self.query_one("#personas-library-rows", ListView)
        for index, item in enumerate(list_view.children):
            if item.id == target:
                list_view.index = index
                return

    def set_row_unsaved(
        self, kind: str | None, item_id: str | None, unsaved: bool
    ) -> None:
        """Toggle the ``.is-unsaved`` badge without rebuilding the rows.

        Only one row (the active editing session's) may carry the badge, so
        setting it also clears any stale badge elsewhere; passing
        ``unsaved=False`` (or no kind/id) clears the badge everywhere.
        """
        target = _row_dom_id(kind, item_id) if (kind and item_id) else None
        list_view = self.query_one("#personas-library-rows", ListView)
        for item in list_view.children:
            item.set_class(unsaved and item.id == target, "is-unsaved")

    def set_sort_label(self, text: str) -> None:
        """Update the sort button's label (the screen owns the sort cycle/copy)."""
        self.query_one("#personas-library-sort", Button).label = text
        # A longer/shorter label changes the single-row fit (F-030).
        self._sync_control_layout()

    # ===== F-040: marks (multi-select) =====

    def _sync_marked_count_line(self) -> None:
        """The count line reports active marks ahead of filter state."""
        text = (
            f"{len(self._marked_ids)} marked"
            if self._marked_ids
            else self._base_count_text
        )
        self.query_one("#personas-library-count", Static).update(text)

    def _post_marks_changed(self) -> None:
        marks = tuple(
            (row.kind, row.item_id, row.name)
            for dom_id, row in self._row_lookup.items()
            if dom_id in self._marked_ids
        )
        self.post_message(PersonaMarksChanged(marks))

    @staticmethod
    def _render_row_marker(item: ListItem, row: LibraryRow, marked: bool) -> None:
        """Prefix/unprefix the row's name line with the marked glyph."""
        name_static = item.query_one(Static)
        name_static.update(f"● {row.name}" if marked else row.name)

    def clear_marks(self) -> None:
        """Drop every mark (mode switch, or after a bulk action consumes them)."""
        if not self._marked_ids:
            return
        self._marked_ids = set()
        list_view = self.query_one("#personas-library-rows", ListView)
        for item in list_view.children:
            row = self._row_lookup.get(str(item.id or ""))
            if row is not None:
                self._render_row_marker(item, row, False)
        self._sync_marked_count_line()
        self._post_marks_changed()

    def action_toggle_mark(self) -> None:
        """m: mark/unmark the highlighted row for bulk delete/export (F-040)."""
        list_view = self.query_one("#personas-library-rows", ListView)
        index = list_view.index
        if index is None or not 0 <= index < len(list_view.children):
            return
        item = list_view.children[index]
        dom_id = str(item.id or "")
        row = self._row_lookup.get(dom_id)
        if row is None:
            # Placeholder/recovery rows are not markable.
            return
        marked = dom_id not in self._marked_ids
        if marked:
            self._marked_ids.add(dom_id)
        else:
            self._marked_ids.discard(dom_id)
        self._render_row_marker(item, row, marked)
        self._sync_marked_count_line()
        self._post_marks_changed()

    def action_cycle_sort(self) -> None:
        """s: cycle the list sort order where sorting applies (F-040)."""
        if self._sort_visible:
            self.post_message(PersonaSortCycleRequested())

    def set_tag_label(self, text: str) -> None:
        """Update the tag button's label (the screen owns the active tag)."""
        self.query_one("#personas-library-tag", Button).label = text
        self._sync_control_layout()

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

    @on(Button.Pressed, "#personas-library-new-actor-pack")
    def _new_actor_pack_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaActionRequested(action="create_actor_pack"))

    @on(Button.Pressed, "#personas-library-duplicate")
    def _duplicate_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaActionRequested(action="duplicate"))

    @on(Button.Pressed, "#personas-library-sort")
    def _sort_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaSortCycleRequested())

    @on(Button.Pressed, "#personas-library-tag")
    def _tag_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaTagFilterRequested())

    @on(Button.Pressed, "#personas-library-prev")
    def _prev_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaPageChanged(-1))

    @on(Button.Pressed, "#personas-library-next")
    def _next_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaPageChanged(1))

    def action_toggle_highlighted(self) -> None:
        """Space on a highlighted dictionary row requests an enable-toggle."""
        list_view = self.query_one("#personas-library-rows", ListView)
        index = list_view.index
        if index is None or not 0 <= index < len(list_view.children):
            return
        row = self._row_lookup.get(str(list_view.children[index].id or ""))
        if row is None or row.kind != "dictionary":
            return
        self.post_message(
            PersonaActionRequested(
                action="toggle_enabled", entity_kind=row.kind, entity_id=row.item_id
            )
        )


__all__ = [
    "LibraryRow",
    "PersonasLibraryPane",
]
