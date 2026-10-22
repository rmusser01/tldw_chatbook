"""Mode-scoped library list pane for the Personas workbench.

Frame slice B2a (spec 1.5, D14 = A): the items list is one
``RoleplayItemList``, an ``OptionList`` that keeps the id
``#personas-library-rows``. Each option is one row, id ``"<kind>:<id>"``,
whose prompt is literal ``Content`` built from
``roleplay_frame_state.build_item_prompt`` spans (markup off, spec R33). Rows
are option lines, not widgets, so a keystroke that changes the list calls
``set_options`` once and mounts nothing (PERF-22). Arrows only move the
cursor; Enter or a click loads (I6); a click on the gutter's mark column
toggles the mark instead (spec 4.10). ``space``, ``m`` and ``s`` are the
list's own keys, so they never fire from a focused toolbar button (RP-033).
Only the public ``OptionList`` API is used (``set_options``, ``add_options``,
``get_option``, ``get_option_index``, ``replace_option_prompt``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from loguru import logger
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.events import Resize
from textual.message import Message
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from ...UI.Persona_Modules import roleplay_frame_state as frame_state
from .personas_messages import (
    ActorPackImportRequested,
    PersonaActionRequested,
    PersonaEntityKind,
    PersonaEntitySelected,
    PersonaMarksChanged,
    PersonaPageChanged,
    PersonaSearchChanged,
    PersonaSortCycleRequested,
    PersonaTagFilterRequested,
)

logger = logger.bind(module="PersonasLibraryPane")

#: Columns one toolbar button occupies beyond its label: `padding: 0 1`
#: plus the `margin-right: 1` gap from the pane CSS below.
_TOOLBAR_BUTTON_CHROME_COLS = 3


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


def option_id_for(kind: str, item_id: str) -> str:
    """A row's option id: ``"<kind>:<id>"`` (spec 1.5.6; ids are not DOM ids)."""
    return f"{kind}:{item_id}"


@dataclass(frozen=True)
class LibraryRow:
    """One selectable row in the workbench library list.

    ``meta`` is line 2 (the scent, spec 1.5.2), ``state`` the book's
    ``on``/``off`` word, ``short`` the 1-line extra (a book's entry count).
    The list carries no unsaved marker (spec 1.5.3, R24).
    """

    item_id: str
    kind: PersonaEntityKind
    name: str
    meta: str | None = None
    state: str | None = None
    short: str | None = None


def _prompt_content(prompt: frame_state.ItemPrompt) -> Content:
    """Turn FS spans into one literal ``Content`` (markup never parsed)."""
    lines = [
        Content.assemble(*((text, style) if style else text for text, style in line))
        for line in prompt.lines
    ]
    return Content("\n").join(lines)


class RoleplayItemList(OptionList):
    """The line-based Roleplay items list (spec 1.5, D14 = A).

    Owns the rows (``LibraryRow`` per option id), their prompts, the loaded
    row, the marks per kind and the window requests; the pane and the screen
    decide what rows to show.
    """

    BINDINGS = [
        Binding("space", "toggle_highlighted", "Toggle on/off", show=False),
        Binding("m", "toggle_mark", "Mark row", show=False),
        Binding("s", "cycle_sort", "Cycle sort", show=False),
    ]

    class WindowWanted(Message):
        """More rows are wanted (spec 1.5.6 load-through).

        ``reason`` is ``"near_end"`` (the cursor or scroll came within a screen
        of the loaded end), ``"end"`` (End past the loaded rows) or ``"id"``
        (a row to load, select or restore is not loaded yet; ``target_id``).
        """

        def __init__(
            self,
            item_list: RoleplayItemList,
            *,
            reason: str,
            target_id: str | None = None,
        ) -> None:
            super().__init__()
            self.item_list = item_list
            self.reason = reason
            self.target_id = target_id

    def __init__(self, **kwargs) -> None:
        super().__init__(markup=False, **kwargs)
        self.add_class("roleplay-items-list")
        self._rows: dict[str, LibraryRow] = {}
        self._prompts: dict[str, frame_state.ItemPrompt] = {}
        self._marked: dict[str, dict[str, LibraryRow]] = {}
        self._kind = "character"
        self._kind_total = 0
        self._active_id: str | None = None
        self._pending_active: str | None = None
        self._density = 1
        self._width = frame_state.UNMEASURED_WIDTH
        self._more_available = False
        self._window_requested = False
        self._width_check_scheduled = False
        self._tooltip_key: tuple[str, ...] | None = None

    # ---- read side ----

    @property
    def kind(self) -> str:
        """The entity kind the list shows (``character``, ``persona``, ``lore``, ``dictionary``)."""
        return self._kind

    @property
    def density(self) -> int:
        """1 or 2 lines per row (spec 1.5.2)."""
        return self._density

    @property
    def active_id(self) -> str | None:
        """The option id loaded in the work pane, if any."""
        return self._active_id

    @property
    def more_available(self) -> bool:
        """Whether rows past the loaded end exist."""
        return self._more_available

    def row(self, option_id: str) -> LibraryRow | None:
        """The loaded row for ``option_id``."""
        return self._rows.get(option_id)

    def loaded_ids(self) -> tuple[str, ...]:
        """Every loaded option id, in list order."""
        return tuple(self._rows)

    def prompt_for(self, option_id: str) -> frame_state.ItemPrompt | None:
        """The prompt spans currently shown for ``option_id``."""
        return self._prompts.get(option_id)

    def highlighted_id(self) -> str | None:
        """The cursor row's option id."""
        option = self.highlighted_option
        return str(option.id) if option is not None else None

    def marked_rows(self, kind: str | None = None) -> tuple[LibraryRow, ...]:
        """The marked rows of ``kind`` (default: the list's kind), in marking order."""
        return tuple(self._marked.get(kind or self._kind, {}).values())

    def marked_ids(self, kind: str | None = None) -> frozenset[str]:
        """The marked option ids of ``kind`` (default: the list's kind)."""
        return frozenset(self._marked.get(kind or self._kind, {}))

    # ---- rows ----

    def set_kind(self, kind: str) -> None:
        """Switch the kind the list shows; marks of other kinds are kept."""
        self._kind = kind

    def _measured_width(self) -> int | None:
        width = self.scrollable_content_region.width
        return width if width > 0 else None

    def _content_for(self, option_id: str) -> Content:
        row = self._rows[option_id]
        prompt = frame_state.build_item_prompt(
            name=row.name,
            state=row.state,
            meta=row.meta,
            short=row.short,
            loaded=option_id == self._active_id,
            marked=option_id in self._marked.get(row.kind, {}),
            width=self._width,
            density=self._density,
        )
        self._prompts[option_id] = prompt
        return _prompt_content(prompt)

    def _top_index(self) -> int:
        return int(self.scroll_y) // max(1, self._density)

    def show_rows(
        self,
        rows: Sequence[LibraryRow],
        *,
        kind_total: int,
        anchor: frame_state.ListAnchor | None = None,
    ) -> tuple[str, ...]:
        """Replace every option with ``rows`` in one ``set_options``.

        The cursor comes back by id (or, with an ``anchor`` whose cursor row is
        gone, at the same index: the delete neighbour), and so does the top
        row when an ``anchor`` is given (spec 1.5.6 "State", "Writes").

        Returns:
            The duplicate option ids that were dropped (logged, never raised).
        """
        cursor_id = anchor.cursor_id if anchor is not None else self.highlighted_id()
        ids = [option_id_for(row.kind, row.item_id) for row in rows]
        keep, dropped = frame_state.new_option_ids((), ids)
        width = self._measured_width()
        self._kind_total = kind_total
        self._width = width or frame_state.UNMEASURED_WIDTH
        self._density = frame_state.item_density(
            inner_width=width or 0, kind_total=kind_total
        )
        self._rows = {ids[index]: rows[index] for index in keep}
        self._prompts = {}
        self.set_options(
            [
                Option(self._content_for(option_id), id=option_id)
                for option_id in self._rows
            ]
        )
        self._restore(
            cursor_id=cursor_id,
            cursor_index=anchor.cursor_index if anchor is not None else None,
            top_index=anchor.top_index if anchor is not None else 0,
        )
        self._apply_pending_active()
        return dropped

    def append_rows(self, rows: Sequence[LibraryRow]) -> tuple[str, ...]:
        """Append a window with ``add_options``, skipping ids already present.

        Textual 8.2.8 raises ``DuplicateID`` part-way through a batch, so only
        new ids reach ``add_options`` (spec 1.5.6 "No duplicates").

        Returns:
            The duplicate option ids that were dropped.
        """
        ids = [option_id_for(row.kind, row.item_id) for row in rows]
        keep, dropped = frame_state.new_option_ids(self._rows, ids)
        new_ids = [ids[index] for index in keep]
        for index in keep:
            self._rows[ids[index]] = rows[index]
        if new_ids:
            self.add_options(
                [
                    Option(self._content_for(option_id), id=option_id)
                    for option_id in new_ids
                ]
            )
        self._window_requested = False
        self._apply_pending_active()
        return dropped

    def _restore(
        self, *, cursor_id: str | None, cursor_index: int | None, top_index: int
    ) -> None:
        count = self.option_count
        if not count:
            return
        if top_index:
            self.scroll_to(
                y=min(top_index, count - 1) * self._density,
                animate=False,
                immediate=True,
            )
        if cursor_id is not None and cursor_id in self._rows:
            self.highlighted = self.get_option_index(cursor_id)
        elif cursor_index is not None:
            self.highlighted = min(max(cursor_index, 0), count - 1)

    def _rebuild(
        self, needs: Callable[[str, frame_state.ItemPrompt], bool], *, top: int
    ) -> int:
        """One ``set_options`` with new prompts for the rows ``needs`` picks.

        Every other row keeps its option (and Textual's cached visual), so a
        resize or a batch of marks rebuilds only what changed (RC-6).
        """
        options: list[Option] = []
        changed = 0
        for option_id in self._rows:
            prompt = self._prompts.get(option_id)
            if prompt is None or needs(option_id, prompt):
                options.append(Option(self._content_for(option_id), id=option_id))
                changed += 1
            else:
                options.append(self.get_option(option_id))
        if changed:
            cursor = self.highlighted_id()
            self.set_options(options)
            self._restore(cursor_id=cursor, cursor_index=None, top_index=top)
        return changed

    def _refresh_rows(self, option_ids: Sequence[str]) -> None:
        """Repaint a few rows: one ``replace_option_prompt`` each, a batch as one rebuild."""
        targets = [option_id for option_id in option_ids if option_id in self._rows]
        if len(targets) == 1:
            self.replace_option_prompt(targets[0], self._content_for(targets[0]))
        elif targets:
            wanted = set(targets)
            self._rebuild(
                lambda option_id, _prompt: option_id in wanted, top=self._top_index()
            )

    # ---- the loaded row and the cursor ----

    def set_active(self, option_id: str | None) -> bool:
        """Show ``option_id`` as loaded (``▸`` + bold) and move the cursor to it.

        Returns:
            False when the row is not loaded yet: it is kept as pending, and a
            window is requested if rows past the loaded end exist (load-through).
        """
        previous, self._active_id = self._active_id, option_id
        if previous != option_id:
            self._refresh_rows([x for x in (previous, option_id) if x is not None])
        if option_id is None or option_id in self._rows:
            self._pending_active = None
            if option_id is not None:
                self.highlighted = self.get_option_index(option_id)
            return True
        self._pending_active = option_id
        if self._more_available:
            self._request_window("id", option_id)
        return False

    def _apply_pending_active(self) -> None:
        pending = self._pending_active
        if pending is not None and pending in self._rows:
            self._pending_active = None
            self.set_active(pending)

    def highlight(self, option_id: str) -> bool:
        """Move only the cursor to ``option_id`` (False when it is not loaded)."""
        if option_id not in self._rows:
            return False
        self.highlighted = self.get_option_index(option_id)
        return True

    def move_to_end(self) -> None:
        """The cursor to the last loaded row (after End loaded every window)."""
        super().action_last()

    # ---- windows ----

    def set_more_available(self, more: bool) -> None:
        """Whether rows past the loaded end exist (set by the screen after each window)."""
        self._more_available = more
        self._window_requested = False

    def _request_window(self, reason: str, target_id: str | None = None) -> None:
        self._window_requested = True
        self.post_message(self.WindowWanted(self, reason=reason, target_id=target_id))

    def _check_near_end(self) -> None:
        if not self._more_available or self._window_requested or not self.option_count:
            return
        lines = max(1, self._density)
        page = max(1, self.scrollable_content_region.height // lines)
        bottom = int(self.scroll_y) // lines + page
        cursor = self.highlighted if self.highlighted is not None else 0
        if self.option_count - max(bottom, cursor) <= page:
            self._request_window("near_end")

    def watch_highlighted(self, highlighted: int | None) -> None:
        super().watch_highlighted(highlighted)
        self._check_near_end()

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self._check_near_end()

    def action_last(self) -> None:
        """End: load every remaining window first when rows are not all loaded."""
        if self._more_available:
            self._request_window("end")
            return
        super().action_last()

    # ---- width (spec 1.5.6 "Many rows loaded") ----

    def on_resize(self, event: Resize) -> None:
        """Rebuild only the prompts a new width changes, once per settled width.

        A height-only resize changes nothing. The check runs after the next
        refresh, so a burst of resizes in one frame costs one rebuild, and it
        reads no data and starts no timer or worker (spec 2.13).
        """
        if not self._width_check_scheduled:
            self._width_check_scheduled = True
            self.call_after_refresh(self._apply_width)

    def _apply_width(self) -> None:
        self._width_check_scheduled = False
        width = self._measured_width()
        if width is None:
            return
        density = frame_state.item_density(
            inner_width=width, kind_total=self._kind_total
        )
        if width == self._width and density == self._density:
            return
        top = self._top_index()
        self._width, self._density = width, density
        self._rebuild(
            lambda _option_id, prompt: prompt.needs_rebuild(
                width=width, density=density
            ),
            top=top,
        )

    # ---- marks ----

    def toggle_mark(self, option_id: str) -> None:
        """Mark or unmark one loaded row (``m`` or a click on the mark column)."""
        row = self._rows.get(option_id)
        if row is None:
            return
        marks = self._marked.setdefault(row.kind, {})
        if option_id in marks:
            del marks[option_id]
        else:
            marks[option_id] = row
        self._refresh_rows([option_id])
        pane = self._pane()
        if pane is not None:
            pane.marks_changed()

    def clear_marks(self, kind: str | None = None) -> bool:
        """Drop the marks of ``kind`` (default: the list's kind) in one batch."""
        marks = self._marked.pop(kind or self._kind, {})
        if not marks:
            return False
        self._refresh_rows(list(marks))
        return True

    def drop_marks(self, option_ids: Sequence[str]) -> None:
        """Forget marks whose rows are gone (deleted, or pruned by a refresh)."""
        for marks in self._marked.values():
            for option_id in option_ids:
                marks.pop(option_id, None)

    # ---- mouse ----

    def on_click(self, event: events.Click) -> None:
        """A click on the gutter's mark column toggles the mark and never loads (spec 4.10)."""
        index = event.style.meta.get("option")
        offset = event.get_content_offset(self)
        if index is None or offset is None:
            return
        if offset.x not in frame_state.gutter_mark_columns():
            return
        event.prevent_default()
        event.stop()
        option_id = str(self.get_option_at_index(index).id)
        self.highlighted = index
        self.toggle_mark(option_id)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Hover tooltip: "Click to mark" on the mark column, the full name on a cut one."""
        index = event.style.meta.get("option")
        offset = event.get_content_offset(self)
        key: tuple[str, ...] | None = None
        tip: Content | None = None
        if index is not None and offset is not None and 0 <= index < self.option_count:
            option_id = str(self.get_option_at_index(index).id)
            if offset.x in frame_state.gutter_mark_columns():
                key, tip = ("mark",), Content("Click to mark")
            else:
                prompt = self._prompts.get(option_id)
                row = self._rows.get(option_id)
                if prompt is not None and prompt.name_cut and row is not None:
                    key, tip = ("name", option_id), Content(row.name)
        if key != self._tooltip_key:
            self._tooltip_key = key
            self.tooltip = tip

    # ---- keys ----

    def _pane(self) -> PersonasLibraryPane | None:
        for node in self.ancestors:
            if isinstance(node, PersonasLibraryPane):
                return node
        return None

    def action_toggle_mark(self) -> None:
        """m: mark or unmark the cursor row."""
        option_id = self.highlighted_id()
        if option_id is not None:
            self.toggle_mark(option_id)

    def action_toggle_highlighted(self) -> None:
        """space: ask the screen to turn the cursor's book on or off."""
        pane = self._pane()
        if pane is not None:
            pane.request_toggle(self.highlighted_id())

    def action_cycle_sort(self) -> None:
        """s: cycle the sort where sorting applies."""
        pane = self._pane()
        if pane is not None:
            pane.action_cycle_sort()


class PersonasLibraryPane(Vertical):
    """Search, create/import toolbar, and the line-based items list.

    Selection is explicit: highlighting never posts ``PersonaEntitySelected``,
    so unsaved-edit guards stay quiet while the user browses.
    """

    # Structure only: colors come from the app stylesheet.
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

    /* The smallest supported terminal has no spare vertical chrome once
       both action bars stack.  Collapse the search field's stock three-row
       border to one row so the final Tag action remains inside the pane. */
    PersonasLibraryPane.personas-library-stacked-controls #personas-library-search {
        height: 1;
        min-height: 1;
        border: none;
        padding: 0 1;
    }

    PersonasLibraryPane.personas-library-stacked-controls #personas-library-toolbar Button,
    PersonasLibraryPane.personas-library-stacked-controls #personas-library-filterbar Button {
        width: 100%;
        margin-right: 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._import_visible: bool = True
        self._sort_visible: bool = True
        self._status_key: tuple[str, str] | None = None
        # The count line's filter-state text from the last update_rows; the
        # "N marked" summary overrides it while marks exist.
        self._base_count_text: str = ""

    @property
    def item_list(self) -> RoleplayItemList:
        """The items list."""
        return self.query_one("#personas-library-rows", RoleplayItemList)

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
            title.add_class("w-fill")
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
                tooltip="Import a character card or create a character from a Buddy archive.",
                classes="console-action-secondary",
            )
            yield Button(
                "Import Actor Pack",
                id="personas-library-import-actor-pack",
                tooltip="Review and activate a portable .tldw-actor-pack archive.",
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
        # The empty / recovery copy: one Static beside the list, mounted only
        # when the state changes (rows / empty / recovery), never per row.
        # Hidden while the list has rows, so it takes no row from it.
        status = Vertical(id="personas-library-status")
        status.display = False
        yield status
        yield RoleplayItemList(id="personas-library-rows")
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
            import_button.tooltip = (
                "Import a character card or create a character from a Buddy archive."
            )
        self.query_one("#personas-library-duplicate", Button).display = mode in (
            "characters",
            "dictionaries",
            "lore",
        )
        self.query_one("#personas-library-new-actor-pack", Button).display = mode in (
            "characters",
            "personas",
        )
        self.query_one("#personas-library-import-actor-pack", Button).display = (
            mode
            in (
                "characters",
                "personas",
            )
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
        self.item_list.set_kind(frame_state.KIND_FOR_MODE.get(mode, "character"))

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
        anchor: frame_state.ListAnchor | None = None,
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
                pane shows it instead of list or empty rows.
            recovery_id: Stable DOM id for the recovery copy widget.
            page_offset: Zero-based offset of ``rows`` within ``total``. When
                given together with ``page_size``, the pane renders a page bar
                instead of the plain count line. ``None`` (the default)
                preserves the pre-paging behavior for callers (dictionaries,
                lore) that never page.
            page_size: Page window size paired with ``page_offset``.
            anchor: Where the list stood before a reload from the anchor.

        Returns:
            None.
        """
        item_list = self.item_list
        visible_rows = () if recovery_copy else rows
        for dropped in item_list.show_rows(
            visible_rows, kind_total=total, anchor=anchor
        ):
            logger.debug("PersonasLibraryPane dropped a duplicate row id {}", dropped)
        if recovery_copy:
            status: tuple[str, str] | None = (recovery_id, recovery_copy)
        elif not visible_rows:
            hint = "use New or Import" if self._import_visible else "use New"
            status = ("personas-library-empty", f"No {noun} yet - {hint} to add one.")
        else:
            status = None
        await self._sync_status(status)
        # F-040: a mark never outlives its row - a refresh that drops a
        # marked row drops the mark with it.
        loaded = set(item_list.loaded_ids())
        pruned = [oid for oid in item_list.marked_ids() if oid not in loaded]
        if pruned:
            item_list.drop_marks(pruned)
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

    async def _sync_status(self, status: tuple[str, str] | None) -> None:
        """Show the empty or recovery copy; mount only when the state changes."""
        if status == self._status_key:
            return
        container = self.query_one("#personas-library-status", Vertical)
        await container.remove_children()
        if status is not None:
            await container.mount(
                Static(
                    status[1],
                    id=status[0],
                    markup=False,
                    classes="personas-library-status-copy",
                )
            )
        container.display = status is not None
        self._status_key = status

    def mark_active_row(self, kind: str, item_id: str) -> bool:
        """Show one row as loaded and move the cursor to it (loads through when needed)."""
        return self.item_list.set_active(option_id_for(kind, item_id))

    def highlight_row(self, kind: str, item_id: str) -> bool:
        """Move only the cursor to one row (no loaded-marker change)."""
        return self.item_list.highlight(option_id_for(kind, item_id))

    def set_sort_label(self, text: str) -> None:
        """Update the sort button's label (the screen owns the sort cycle/copy)."""
        self.query_one("#personas-library-sort", Button).label = text
        # A longer/shorter label changes the single-row fit (F-030).
        self._sync_control_layout()

    # ===== F-040: marks (multi-select) =====

    def _sync_marked_count_line(self) -> None:
        """The count line reports active marks ahead of filter state."""
        marked = len(self.item_list.marked_ids())
        text = f"{marked} marked" if marked else self._base_count_text
        self.query_one("#personas-library-count", Static).update(text)

    def _post_marks_changed(self) -> None:
        marks = tuple(
            (row.kind, row.item_id, row.name) for row in self.item_list.marked_rows()
        )
        self.post_message(PersonaMarksChanged(marks))

    def marks_changed(self) -> None:
        """The list's marks changed (``m``, a mark-column click, Clear)."""
        self._sync_marked_count_line()
        self._post_marks_changed()

    def clear_marks(self) -> None:
        """Drop every mark of the list's kind (or after a bulk action consumes them)."""
        if self.item_list.clear_marks():
            self.marks_changed()

    def action_cycle_sort(self) -> None:
        """s: cycle the list sort order where sorting applies (F-040)."""
        if self._sort_visible:
            self.post_message(PersonaSortCycleRequested())

    def request_toggle(self, option_id: str | None) -> None:
        """space on a book row asks the screen to turn it on or off."""
        row = self.item_list.row(option_id) if option_id else None
        if row is None or row.kind != "dictionary":
            return
        self.post_message(
            PersonaActionRequested(
                action="toggle_enabled", entity_kind=row.kind, entity_id=row.item_id
            )
        )

    def set_tag_label(self, text: str) -> None:
        """Update the tag button's label (the screen owns the active tag)."""
        # TASK-34400: the active tag is untrusted; a str label parses as markup.
        self.query_one("#personas-library-tag", Button).label = Content(text)
        self._sync_control_layout()

    @on(Input.Changed, "#personas-library-search")
    def _search_changed(self, event: Input.Changed) -> None:
        event.stop()
        self.post_message(PersonaSearchChanged(query=event.value))

    @on(Input.Submitted, "#personas-library-search")
    def _search_submitted(self, event: Input.Submitted) -> None:
        """Enter in the search box jumps into the results list."""
        event.stop()
        item_list = self.item_list
        item_list.focus()
        if item_list.option_count:
            item_list.highlighted = 0

    @on(OptionList.OptionSelected, "#personas-library-rows")
    def _row_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        row = self.item_list.row(str(event.option.id))
        if row is not None:
            self.post_message(
                PersonaEntitySelected(
                    entity_kind=row.kind,
                    entity_id=row.item_id,
                    entity_name=row.name,
                )
            )

    @on(OptionList.OptionHighlighted, "#personas-library-rows")
    def _row_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        # Highlight never loads (I6); it also fires on every programmatic
        # cursor restore, so it stays a no-op.
        event.stop()

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

    @on(Button.Pressed, "#personas-library-import-actor-pack")
    def _import_actor_pack_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(ActorPackImportRequested())

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


__all__ = [
    "LibraryRow",
    "PersonasLibraryPane",
    "RoleplayItemList",
    "option_id_for",
]
