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
from rich.cells import cell_len
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
    PersonaActionRequested,
    PersonaEntityKind,
    PersonaEntitySelected,
    PersonaMarksChanged,
    PersonaSearchChanged,
    PersonaSearchSubmitted,
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
    row and the marks per kind; the pane and the screen decide what rows to
    show. The window state (rows past the loaded end, the asked-for window,
    the owed cursor move) lives in a ``ListWindowState`` the screen shares.
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
        ``intent`` is the cursor move End or a load-through owes the user.
        """

        def __init__(
            self,
            item_list: RoleplayItemList,
            *,
            reason: str,
            target_id: str | None = None,
            intent: frame_state.WindowIntent | None = None,
        ) -> None:
            super().__init__()
            self.item_list = item_list
            self.reason = reason
            self.target_id = target_id
            self.intent = intent

    def __init__(
        self, *, window_state: frame_state.ListWindowState | None = None, **kwargs
    ) -> None:
        super().__init__(markup=False, **kwargs)
        self.add_class("roleplay-items-list")
        self._rows: dict[str, LibraryRow] = {}
        self._prompts: dict[str, frame_state.ItemPrompt] = {}
        self._marked: dict[str, dict[str, LibraryRow]] = {}
        self._kind = "character"
        self._kind_total = 0
        self._active_id: str | None = None
        self._density = 1
        self._width = frame_state.UNMEASURED_WIDTH
        self._window = (
            window_state if window_state is not None else frame_state.ListWindowState()
        )
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
        return self._window.more_available

    @property
    def window_state(self) -> frame_state.ListWindowState:
        """The window state this list shares with its screen (TASK-33910.34)."""
        return self._window

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
        """Switch the kind the list shows; marks of other kinds are kept.

        A new kind starts with no window state: rows past the loaded end, an
        asked-for window and an owed cursor move belong to the old kind's
        list (B2a's final review, I1: End did nothing in Lore books after a
        partial Characters window).
        """
        if kind != self._kind:
            self._window.kind_switched()
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
        self._position_changed()
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
        self._window.answered()
        self._apply_pending_active()
        self._position_changed()
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

    def set_active(
        self,
        option_id: str | None,
        *,
        move_cursor: bool = True,
        load_through: bool = True,
    ) -> bool:
        """Show ``option_id`` as loaded (``▸`` + bold) and, by default, move the cursor to it.

        Args:
            option_id: The loaded row's option id, or ``None`` for none.
            move_cursor: False after a reload from the anchor, whose cursor
                was already restored by id.
            load_through: False for a plain reload (a keystroke, a sort, a
                tag): it reads one window, so a loaded item past that window
                is marked when a later window brings it, never fetched.

        Returns:
            False when the row is not loaded yet: with ``load_through`` the
            list owes the user the move to it, and a window is requested if
            rows past the loaded end exist (also for the open row marked
            again, as on dev). A selection of another loaded row replaces any
            owed move; the screen marking the loaded open row again (a
            background refresh) leaves it.
        """
        previous, self._active_id = self._active_id, option_id
        if previous != option_id:
            self._refresh_rows([x for x in (previous, option_id) if x is not None])
        self._position_changed()
        if option_id is None or option_id in self._rows:
            if load_through and previous != option_id:
                self._window.user_moved()
            if option_id is not None and move_cursor:
                self.highlighted = self.get_option_index(option_id)
            return True
        if load_through:
            self._owe_row(option_id)
        return False

    def _owe_row(self, option_id: str) -> None:
        read = self._window.more_available
        intent = self._window.want("id", option_id, read=read)
        if read:
            self._request_window("id", option_id, intent)

    def ask_again(self, intent: frame_state.WindowIntent) -> None:
        """Ask again for an owed move a reload deferred or cancelled (once that list lands)."""
        if intent.reason == "end":
            self.action_last()
        elif intent.target_id in self._rows:
            self.set_active(intent.target_id)
        elif intent.target_id is not None:
            self._owe_row(intent.target_id)

    def _apply_pending_active(self) -> None:
        arrived = self._window.arrived(self._rows)
        if arrived is not None:
            self.set_active(arrived)

    def highlight(self, option_id: str) -> bool:
        """Move only the cursor to ``option_id`` (False when it is not loaded)."""
        if option_id not in self._rows:
            return False
        self.highlighted = self.get_option_index(option_id)
        return True

    def move_to_end(self) -> None:
        """The cursor to the last loaded row (after End loaded every window)."""
        super().action_last()

    def _position_changed(self) -> None:
        pane = self._pane()
        if pane is not None:
            pane.list_changed()

    # ---- windows ----

    def set_more_available(self, more: bool) -> None:
        """Whether rows past the loaded end exist (set by the screen after each window)."""
        self._window.set_more(more)

    def _request_window(
        self,
        reason: str,
        target_id: str | None = None,
        intent: frame_state.WindowIntent | None = None,
    ) -> None:
        self.post_message(
            self.WindowWanted(self, reason=reason, target_id=target_id, intent=intent)
        )

    def _check_near_end(self) -> None:
        window = self._window
        if not window.more_available or window.requested or not self.option_count:
            return
        lines = max(1, self._density)
        page = max(1, self.scrollable_content_region.height // lines)
        bottom = int(self.scroll_y) // lines + page
        cursor = self.highlighted if self.highlighted is not None else 0
        if self.option_count - max(bottom, cursor) <= page:
            window.ask()
            self._request_window("near_end")

    def watch_highlighted(self, highlighted: int | None) -> None:
        super().watch_highlighted(highlighted)
        self._check_near_end()

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self._check_near_end()

    def action_last(self) -> None:
        """End: load every remaining window first when rows are not all loaded."""
        if self._window.more_available:
            self._request_window("end", None, self._window.want("end", read=True))
            return
        self._window.user_moved()
        super().action_last()

    # A cursor key, or Enter on a row, is a newer move: an End or a
    # load-through still owed is not wanted any more (TASK-33910.34).

    def action_select(self) -> None:
        if self._window.defer_first_activation():
            # Open no stale row; matching publication fulfils this Enter once.
            return
        self._window.user_moved()
        super().action_select()

    def action_first(self) -> None:
        self._window.user_moved()
        super().action_first()

    def action_cursor_up(self) -> None:
        self._window.user_moved()
        super().action_cursor_up()

    def action_cursor_down(self) -> None:
        self._window.user_moved()
        super().action_cursor_down()

    def action_page_up(self) -> None:
        self._window.user_moved()
        super().action_page_up()

    def action_page_down(self) -> None:
        self._window.user_moved()
        super().action_page_down()

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

    def clear_all_marks(self) -> bool:
        """Drop the marks of every kind (the visit ended, spec R16); one batch."""
        shown = self._marked.pop(self._kind, {})
        had_marks = bool(shown) or any(self._marked.values())
        self._marked.clear()
        self._refresh_rows(list(shown))
        return had_marks

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
        # The meta can be a frame stale after set_options shrank the list.
        if index is None or offset is None or not 0 <= index < self.option_count:
            return
        self._window.user_moved()  # the click moves the cursor
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

    def __init__(
        self, *, window_state: frame_state.ListWindowState | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._window_state = window_state
        self._sort_visible: bool = True
        self._status_key: tuple[str, str] | None = None
        # The list's COUNT total (the M of "#N of M") and whether a search or
        # tag narrows the list (spec 1.5.1).
        self._list_total: int | None = None
        self._filtered: bool = False

    @property
    def item_list(self) -> RoleplayItemList:
        """The items list."""
        return self.query_one("#personas-library-rows", RoleplayItemList)

    def on_mount(self) -> None:
        """Initialize control visibility for default characters mode.

        ``PersonasScreen.on_mount`` calls ``set_mode`` immediately after this
        pane mounts, so this only sets the very first paint.
        """
        self._sync_control_layout()

    def on_resize(self, event: Resize) -> None:
        """Re-fit the toolbar bars (F-030) and the marks bar to the pane's width."""
        self._sync_control_layout()
        self._sync_marks_bar()

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
        """Compose the Library pane header, search controls, rows and marks bar.

        The counts live in the pane's border titles (spec 1.5.1): the kind and
        its count on the top border, the loaded row's ``#N of M`` on the
        bottom. The marks bar takes the pane's bottom row only while marks
        exist (spec 1.5.5), so the rows above it never move.

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
                tooltip="Create something new: pick a kind and a starting point.",
                classes="console-action-secondary",
            )
            # One Import picker for every kind (frame slice B2b, spec 1.4.6);
            # dev had Import (by kind) and Import Actor Pack.
            yield Button(
                "Import",
                id="personas-library-import",
                tooltip=(
                    "Import a character card, lore book, chat dictionary, "
                    "Actor Pack or Buddy from a file."
                ),
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
        yield RoleplayItemList(
            id="personas-library-rows", window_state=self._window_state
        )
        marks = Horizontal(id="personas-library-marks")
        marks.display = False
        with marks:
            yield Static("", id="personas-library-marks-count", markup=False)
            for action, label in (
                ("export", "Export…"),
                ("delete", "Delete…"),
                ("clear", "Clear esc"),
            ):
                yield Static(
                    "",
                    id=f"personas-library-marks-sep-{action}",
                    classes="personas-library-marks-sep",
                    markup=False,
                )
                yield Button(
                    Content(label),
                    id=f"personas-library-marks-{action}",
                    classes="personas-library-marks-action",
                    compact=True,
                )

    def set_mode(self, mode: str) -> None:
        """Gate the library toolbar's buttons for the active workbench mode.

        Import shows in every kind: one picker takes every Roleplay format
        (frame slice B2b; dev hid it in Personas, task-443). Duplicate is the
        Inspector's (frame slice B2b, spec 5.3).
        Marks are kept per kind for the visit (spec R16): a kind switch shows
        that kind's marks and never drops another kind's.

        Args:
            mode: The active workbench mode id (``characters``/``personas``/
                ``dictionaries``/``lore``).
        """
        sort_visible = mode in ("characters", "personas")
        self._sort_visible = sort_visible
        self.query_one("#personas-library-sort", Button).display = sort_visible
        self.query_one("#personas-library-tag", Button).display = mode == "characters"
        # Which buttons render changed, so the single-row fit changed too.
        self._sync_control_layout()
        self.item_list.set_kind(frame_state.KIND_FOR_MODE.get(mode, "character"))
        self.marks_changed()

    async def update_rows(
        self,
        rows: tuple[LibraryRow, ...],
        *,
        total: int,
        noun: str,
        filtered: bool = False,
        matched: int | None = None,
        title: str | None = None,
        recovery_copy: str | None = None,
        recovery_id: str = "personas-library-recovery",
        anchor: frame_state.ListAnchor | None = None,
    ) -> None:
        """Replace the visible rows (one ``set_options``) and the border titles.

        Marks are kept by id (spec 1.5.5): a row a search, sort or tag hides
        stays marked, and the marks bar counts it as hidden.

        Args:
            rows: Selectable library rows to render when no recovery state is
                active (for characters: the loaded windows, spec 1.5.6).
            total: The kind's total: the title's count when unfiltered, the
                "of M" when filtered, and the density rule's item count.
            noun: User-facing noun used in the empty copy.
            filtered: Whether a search or tag narrows the list.
            matched: The narrowed list's COUNT (default: ``len(rows)`` when
                filtered, else ``total``): the M of the bottom border's
                ``#N of M``.
            title: The top-border title; built from the kind and the counts
                when omitted (spec 1.5.1).
            recovery_copy: Optional multi-line recovery copy. When present, the
                pane shows it instead of list or empty rows.
            recovery_id: Stable DOM id for the recovery copy widget.
            anchor: Where the list stood before a reload from the anchor.

        Returns:
            None.
        """
        item_list = self.item_list
        visible_rows = () if recovery_copy else rows
        self._filtered = filtered
        self._list_total = (
            None
            if recovery_copy
            else (
                matched if matched is not None else (len(rows) if filtered else total)
            )
        )
        for dropped in item_list.show_rows(
            visible_rows, kind_total=total, anchor=anchor
        ):
            logger.debug("PersonasLibraryPane dropped a duplicate row id {}", dropped)
        if recovery_copy:
            status: tuple[str, str] | None = (recovery_id, recovery_copy)
        elif not visible_rows:
            status = (
                "personas-library-empty",
                f"No {noun} yet - use New or Import to add one.",
            )
        else:
            status = None
        await self._sync_status(status)
        if title is None:
            title = frame_state.items_border_title(
                item_list.kind,
                total=total,
                matched=self._list_total if filtered else None,
                failed=bool(recovery_copy),
            )
        self.set_title(title)

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

    def mark_active_row(
        self,
        kind: str,
        item_id: str,
        *,
        move_cursor: bool = True,
        load_through: bool = True,
    ) -> bool:
        """Show one row as loaded and move the cursor to it (``RoleplayItemList.set_active``)."""
        return self.item_list.set_active(
            option_id_for(kind, item_id),
            move_cursor=move_cursor,
            load_through=load_through,
        )

    def list_anchor(self) -> frame_state.ListAnchor:
        """Where the list stands now: the cursor (id and index), the top row and the loaded count."""
        item_list = self.item_list
        return frame_state.ListAnchor(
            cursor_id=item_list.highlighted_id(),
            cursor_index=item_list.highlighted,
            top_index=int(item_list.scroll_y) // max(1, item_list.density),
            loaded=item_list.option_count,
        )

    def set_title(self, title: str, *, total: int | None = None) -> None:
        """The top border's kind-and-count title (literal ``Content``: a tag is untrusted).

        Args:
            title: The text (``items_border_title``).
            total: A new COUNT total for ``#N of M`` (an append keeps it).
        """
        if total is not None:
            self._list_total = total
        self.border_title = Content(title)
        self.refresh_position()

    def refresh_position(self) -> None:
        """The bottom border's ``#N of M``: blank with nothing loaded or the row not listed (RC-10)."""
        item_list = self.item_list
        active = item_list.active_id
        index = (
            item_list.get_option_index(active)
            if active is not None and item_list.row(active) is not None
            else None
        )
        self.border_subtitle = Content(
            frame_state.loaded_position(index, self._list_total)
        )

    def list_changed(self) -> None:
        """Rows or the loaded row changed: re-read ``#N of M`` and the hidden-mark count."""
        self.refresh_position()
        self._sync_marks_bar()

    def _sync_marks_bar(self) -> None:
        """Show the one-row marks bar while this kind has marks (spec 1.5.5).

        Its text degrades with the pane's width (``marks_bar_view``); it reads
        no data and mounts nothing. When it hides while it holds focus, focus
        moves to the list, never to ``None`` (I1).
        """
        try:
            bar = self.query_one("#personas-library-marks", Horizontal)
        except Exception:
            return  # still composing
        item_list = self.item_list
        marked = item_list.marked_ids()
        can_export = item_list.kind in ("character", "persona")
        view = frame_state.marks_bar_view(
            len(marked),
            frame_state.hidden_mark_count(
                marked, item_list.loaded_ids(), filtered=self._filtered
            ),
            # The bar's own content width: the pane's, less the bar's padding.
            # Read from its styles, so it holds before the bar is first shown.
            self.content_size.width - bar.styles.gutter.width,
            can_export=can_export,
        )
        was_shown = bar.display
        if view is None:
            focused = self.app.focused if self.is_attached else None
            had_focus = focused is not None and bar in focused.ancestors
            bar.display = False
            if had_focus:
                item_list.focus()
            return
        count = self.query_one("#personas-library-marks-count", Static)
        count.update(view.count_text)
        count.tooltip = Content(view.count_tooltip)
        # The narrowest form drops the count, never Delete: Delete then
        # carries the count's words in its tooltip (literal Content, R33).
        count.display = bool(view.count_text)
        self.query_one("#personas-library-marks-delete", Button).tooltip = (
            None if view.count_text else Content(view.count_tooltip)
        )
        labels = dict(zip(view.actions, view.labels))
        for action in ("export", "delete", "clear"):
            shown = action in labels
            button = self.query_one(f"#personas-library-marks-{action}", Button)
            separator = self.query_one(f"#personas-library-marks-sep-{action}", Static)
            button.display = separator.display = shown
            if shown:
                label = labels[action]
                button.label = Content(label)
                # Exactly the label's cells, unpadded: the separators carry
                # the spaces. Textual 8.2.8's Button adds two cells to an auto
                # width and pads its label (line-pad), and its CSS parser
                # refuses `line-pad: 0`, so both are set here.
                # ds-runtime: the measured cell width of the label shown now.
                button.styles.width = cell_len(label)
                button.styles.line_pad = 0
                separator.update(view.separator)
        bar.display = True
        if not was_shown:
            # The bar takes the pane's bottom row: keep the cursor row in view.
            item_list.call_after_refresh(item_list.scroll_to_highlight)

    def land_on_first(self) -> None:
        """The cursor to the first row (a searched list's rows have arrived)."""
        item_list = self.item_list
        if item_list.option_count:
            item_list.highlighted = 0

    def highlight_row(self, kind: str, item_id: str) -> bool:
        """Move only the cursor to one row (no loaded-marker change)."""
        return self.item_list.highlight(option_id_for(kind, item_id))

    def set_sort_label(self, text: str) -> None:
        """Update the sort button's label (the screen owns the sort cycle/copy)."""
        button = self.query_one("#personas-library-sort", Button)
        button.label = text
        # Button.label does not re-measure an auto-width button, so a longer
        # label wrapped onto a second, unpainted line while the bar was one
        # row (exists on dev in Personas at 220; fixed by frame slice B2b).
        button.refresh(layout=True)
        # A longer/shorter label changes the single-row fit (F-030).
        self._sync_control_layout()

    # ===== F-040: marks (multi-select) =====

    def _post_marks_changed(self) -> None:
        marks = tuple(
            (row.kind, row.item_id, row.name) for row in self.item_list.marked_rows()
        )
        self.post_message(PersonaMarksChanged(marks))

    def marks_changed(self) -> None:
        """The list's marks changed (``m``, a mark-column click, Clear, a kind switch)."""
        self._sync_marks_bar()
        self._post_marks_changed()

    def clear_marks(self) -> None:
        """Drop every mark of the list's kind (Clear, Escape)."""
        if self.item_list.clear_marks():
            self.marks_changed()

    def end_visit(self) -> None:
        """The visit ended (spec 4.11 leave hook): every kind's marks go."""
        if self.item_list.clear_all_marks():
            self.marks_changed()

    def forget_marks(self, option_ids: Sequence[str]) -> None:
        """Deleted rows leave the marks (spec 4.11, Delete rows); others stay marked."""
        self.item_list.drop_marks(option_ids)
        self.marks_changed()

    def action_cycle_sort(self) -> None:
        """s: cycle the list sort order where sorting applies (F-040)."""
        if self._sort_visible:
            self.post_message(PersonaSortCycleRequested())

    def request_toggle(self, option_id: str | None) -> None:
        """space on a book row (lore book or chat dictionary) asks the screen to turn it on or off."""
        row = self.item_list.row(option_id) if option_id else None
        if row is None or row.kind not in ("dictionary", "lore"):
            return
        self.post_message(
            PersonaActionRequested(
                action="toggle_enabled", entity_kind=row.kind, entity_id=row.item_id
            )
        )

    def set_tag_label(self, text: str) -> None:
        """Update the tag button's label (the screen owns the active tag)."""
        # TASK-34400: the active tag is untrusted; a str label parses as markup.
        button = self.query_one("#personas-library-tag", Button)
        button.label = Content(text)
        button.refresh(layout=True)  # as set_sort_label: re-measure the label
        self._sync_control_layout()

    @on(Input.Changed, "#personas-library-search")
    def _search_changed(self, event: Input.Changed) -> None:
        event.stop()
        self.post_message(PersonaSearchChanged(query=event.value))

    @on(Input.Submitted, "#personas-library-search")
    def _search_submitted(self, event: Input.Submitted) -> None:
        """Enter in the search box jumps into the results list.

        The screen starts a still-pending search at once and owes the cursor
        on its first match, which the list takes when the rows arrive (spec
        1.6's 4-key find); Enter returns without waiting (TASK-33910.35).
        """
        event.stop()
        self.item_list.focus()
        self.post_message(PersonaSearchSubmitted())

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

    @on(Button.Pressed, "#personas-library-marks-export")
    def _marks_export_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaActionRequested(action="export_marked"))

    @on(Button.Pressed, "#personas-library-marks-delete")
    def _marks_delete_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaActionRequested(action="delete_marked"))

    @on(Button.Pressed, "#personas-library-marks-clear")
    def _marks_clear_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.clear_marks()

    @on(Button.Pressed, "#personas-library-new")
    def _new_pressed(self, event: Button.Pressed) -> None:
        """Open the New chooser (frame slice B2b, spec 1.4.6)."""
        event.stop()
        self.post_message(PersonaActionRequested(action="new"))

    @on(Button.Pressed, "#personas-library-import")
    def _import_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaActionRequested(action="import"))

    @on(Button.Pressed, "#personas-library-sort")
    def _sort_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaSortCycleRequested())

    @on(Button.Pressed, "#personas-library-tag")
    def _tag_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(PersonaTagFilterRequested())


__all__ = [
    "LibraryRow",
    "PersonasLibraryPane",
    "RoleplayItemList",
    "option_id_for",
]
