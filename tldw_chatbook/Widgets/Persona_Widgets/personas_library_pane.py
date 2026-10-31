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

from collections.abc import Callable, Collection, Mapping, Sequence
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
from textual.timer import Timer
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from ...UI.Persona_Modules import roleplay_frame_state as frame_state
from .personas_messages import (
    AlsoInRequested,
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

#: Every toolbar control's classes: the shared flat action look plus the
#: Roleplay toolbar's own sizing (css/features/_roleplay.tcss).
_CONTROL_CLASSES = "console-action-secondary roleplay-items-control"
#: The Tag button's words; the tag itself shows in the border title (B2a).
_TAG_OFF_TOOLTIP = "Filter characters by tag."
_TAG_ON_TOOLTIP = "A tag narrows the list (named in its title). Pick All to clear it."


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
        """End: load every remaining window first when rows are not all loaded.

        A newer move either way, before the End it owes (frame slice B2b).
        """
        self._window.user_moved()
        if self._window.more_available:
            self._request_window("end", None, self._window.want("end", read=True))
            return
        super().action_last()

    # A cursor key, or Enter on a row, is a newer move: an End or a
    # load-through still owed is not wanted any more (TASK-33910.34).

    def action_select(self) -> None:
        if self._window.defer_first_activation():
            # Open no stale row; matching publication fulfils this Enter once.
            return
        if self._to_also_in():
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
        if self._to_also_in():
            return
        self._window.user_moved()
        super().action_cursor_down()

    def _to_also_in(self) -> bool:
        """An empty list hands Enter and Down to the ``Also in:`` targets (spec 1.5.4)."""
        pane = self._pane()
        return not self.option_count and pane is not None and pane.focus_also_in()

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

    # Harness fallback only (spec 2.12 item 3): the unstyled test tier loads
    # no app sheet, and there the stock Static (the whole row), Input (3 rows,
    # 100% wide) and Button (16 cells) would push Import and the controls out
    # of the pane. The real rules are app-tier (css/features/_roleplay.tcss
    # and the rail header's), and win wherever both load. Id and class
    # subjects only (spec 2.12 item 1).
    BUNDLED_CSS = """
    PersonasLibraryPane #personas-library-title {
        width: 1fr;
    }

    PersonasLibraryPane .roleplay-items-search {
        width: 1fr;
        height: 1;
        border: none;
    }

    PersonasLibraryPane .roleplay-items-controls {
        width: auto;
        height: auto;
    }

    PersonasLibraryPane .roleplay-items-control {
        min-width: 0;
        margin-left: 1;
    }

    PersonasLibraryPane .roleplay-items-toolbar-wrapped,
    PersonasLibraryPane .roleplay-items-controls-stacked {
        layout: vertical;
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
        # What the toolbar fits (spec 1.5.1): the mode, the sort it names and
        # the fit last applied (None until the pane has a width).
        self._mode = "characters"
        self._sort_key = "name_asc"
        self._toolbar_fit: frame_state.ToolbarFit | None = None
        # The delayed "Loading <kind>…" (spec 1.5.4): its timer, and a serial
        # that a stop bumps, so a callback already queued says nothing.
        self._loading_timer: Timer | None = None
        self._loading_serial = 0
        # What the no-match message shows (query, tag, total), while it shows:
        # the other kinds' counts are laid over it when they arrive.
        self._no_match_args: tuple[str, str | None, int | None] | None = None
        # Whether counts were laid over that message (a leave can cut them short),
        # and the counts themselves: a sort or tag change with the same query
        # repaints the message with them instead of counting again.
        self._also_in_landed = False
        self._also_in_counts: (
            tuple[Mapping[str, int], Collection[str], Collection[str]] | None
        ) = None
        # Retry held the focus when a reload's blank hid it (``_show_retry``).
        self._retry_owed_focus = False

    @property
    def item_list(self) -> RoleplayItemList:
        """The items list."""
        return self.query_one("#personas-library-rows", RoleplayItemList)

    def on_mount(self) -> None:
        """Fit the toolbar for the default characters mode.

        ``PersonasScreen.on_mount`` calls ``set_mode`` immediately after this
        pane mounts, so this only sets the very first paint.
        """
        self._apply_toolbar_fit()

    def on_resize(self, event: Resize) -> None:
        """Re-fit the toolbar and the marks bar to the pane's width."""
        self._apply_toolbar_fit()
        self._sync_marks_bar()

    def _apply_toolbar_fit(self) -> None:
        """Lay the toolbar out for the pane's width (spec 1.5.1): display only.

        Two classes, the labels and one width per control; it mounts nothing,
        and a fit equal to the one applied does nothing at all, so a resize
        that keeps the fit costs one pure function call.
        """
        width = self.content_size.width
        if width <= 0:
            return  # not laid out yet (on_mount); on_resize fits it once sized
        fit = frame_state.fit_items_toolbar(
            self._mode, width=width, sort_key=self._sort_key
        )
        if fit == self._toolbar_fit:
            return
        self._toolbar_fit = fit
        self.query_one("#personas-library-toolbar", Horizontal).set_class(
            fit.rows > 1, "roleplay-items-toolbar-wrapped"
        )
        self.query_one("#personas-library-controls", Horizontal).set_class(
            fit.stacked, "roleplay-items-controls-stacked"
        )
        self.query_one("#personas-library-search", Input).placeholder = fit.labels[
            "search"
        ]
        self.query_one("#personas-library-title", Static).display = fit.title
        for part in ("import", "sort", "tag", "new"):
            label = fit.labels.get(part)
            if label is not None:
                button = self.query_one(f"#personas-library-{part}", Button)
                button.label = Content(label)
                # Exactly the label's cells, unpadded, as the marks bar's
                # actions: the controls' margin is the gap (Textual 8.2.8's
                # Button pads an auto width by two cells, and its CSS parser
                # refuses `line-pad: 0`).
                # ds-runtime: the cell width of the label this fit chose.
                button.styles.width = cell_len(label)
                button.styles.line_pad = 0

    def compose(self) -> ComposeResult:
        """Compose the Library pane header, search controls, rows and marks bar.

        The counts live in the pane's border titles (spec 1.5.1): the kind and
        its count on the top border, the loaded row's ``#N of M`` on the
        bottom. The marks bar takes the pane's bottom row only while marks
        exist (spec 1.5.5), so the rows above it never move.

        Returns:
            Textual compose result for the Library pane.
        """
        # The title row carries the interim Import (spec 5.3, owner decision
        # D2) until frame slice B6's rail takes it: the toolbar then fits one
        # row at 160 columns with whole labels. "Library" yields first.
        with Horizontal(classes="console-rail-header"):
            title = Static(
                frame_state.RAIL_TITLE,
                id="personas-library-title",
                classes="destination-section personas-column-title console-rail-title",
            )
            title.add_class("w-fill")
            yield title
            # One Import picker for every kind (frame slice B2b, spec 1.4.6);
            # dev had Import (by kind) and Import Actor Pack in the toolbar.
            yield Button(
                frame_state.import_label(),
                id="personas-library-import",
                tooltip=(
                    "Import a character card, lore book, chat dictionary, "
                    "Actor Pack or Buddy from a file."
                ),
                classes=_CONTROL_CLASSES,
                compact=True,
            )
            collapse_button = Button(
                "<",
                id="personas-library-rail-collapse",
                classes="console-rail-collapse-button",
                compact=True,
            )
            collapse_button.tooltip = "Collapse Library rail"
            yield collapse_button
        # The one-row toolbar (spec 1.5.1): the dense one-row search box
        # (.settings-compact-input keeps its height focused, RP-009) beside
        # the controls, which take the next row where they do not fit.
        with Horizontal(
            id="personas-library-toolbar", classes="roleplay-items-toolbar"
        ):
            yield Input(
                placeholder=frame_state.search_placeholder("characters", short=True),
                id="personas-library-search",
                classes="settings-compact-input roleplay-items-search",
            )
            with Horizontal(
                id="personas-library-controls", classes="roleplay-items-controls"
            ):
                yield Button(
                    frame_state.sort_label("name_asc"),
                    id="personas-library-sort",
                    tooltip=Content(frame_state.sort_tooltip("name_asc")),
                    classes=_CONTROL_CLASSES,
                    compact=True,
                )
                yield Button(
                    "Tag",
                    id="personas-library-tag",
                    tooltip=_TAG_OFF_TOOLTIP,
                    classes=_CONTROL_CLASSES,
                    compact=True,
                )
                yield Button(
                    "+ New",
                    id="personas-library-new",
                    tooltip="Create something new: pick a kind and a starting point.",
                    classes=_CONTROL_CLASSES,
                    compact=True,
                )
        # The empty / recovery copy: one Static beside the list, mounted only
        # when the state changes (rows / empty / recovery), never per row.
        # Hidden while the list has rows, so it takes no row from it.
        status = Vertical(id="personas-library-status")
        status.display = False
        yield status
        # A failed read's way back (spec 1.5.4 "Failure"): shown with the
        # recovery copy, in every kind (frame slice B2b; dev had none).
        retry = Button(
            Content("Retry"),
            id="personas-library-retry",
            tooltip="Read this list again.",
            classes="console-action-secondary",
            compact=True,
        )
        retry.display = False
        yield retry
        # The no-match message (spec 1.5.4): composed once and updated in
        # place, so a keystroke that keeps the list empty mounts nothing; the
        # query it echoes is untrusted, so both lines paint with markup off.
        no_match = Vertical(id="personas-library-nomatch")
        no_match.display = False
        with no_match:
            for part in ("line", "hint"):
                yield Static(
                    "",
                    id=f"personas-library-nomatch-{part}",
                    classes="personas-library-status-copy",
                    markup=False,
                )
            # "Also in:" (spec 1.5.4): one target per other kind, in rail
            # order, shown only where that kind matches; Enter switches to it
            # with the query applied.
            yield Static(
                frame_state.ALSO_IN_LABEL,
                id="personas-library-also-in-label",
                classes="personas-library-status-copy",
                markup=False,
            )
            for kind in frame_state.ALSO_IN_ORDER:
                yield Button(
                    "",
                    id=f"personas-library-also-in-{kind}",
                    classes="console-action-secondary personas-library-also-in-target",
                    compact=True,
                )
            yield Static(
                frame_state.NOT_FOUND_ELSEWHERE,
                id="personas-library-also-in-none",
                classes="personas-library-status-copy",
                markup=False,
            )
        # "Loading <kind>…", only once a kind's read takes 250 ms (spec 1.5.4).
        loading = Static(
            "",
            id="personas-library-loading",
            classes="personas-library-status-copy",
            markup=False,
        )
        loading.display = False
        yield loading
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
        controls = frame_state.toolbar_controls(mode)
        self._mode = mode
        self._sort_visible = "sort" in controls
        for part in ("sort", "tag"):
            button = self.query_one(f"#personas-library-{part}", Button)
            button.display = part in controls
        # Which controls show changed, so the toolbar's fit changed too.
        self._apply_toolbar_fit()
        self.item_list.set_kind(frame_state.KIND_FOR_MODE.get(mode, "character"))
        self.marks_changed()

    async def update_rows(
        self,
        rows: tuple[LibraryRow, ...],
        *,
        total: int,
        filtered: bool = False,
        matched: int | None = None,
        title: str | None = None,
        recovery_copy: str | None = None,
        recovery_id: str = "personas-library-recovery",
        anchor: frame_state.ListAnchor | None = None,
        query: str = "",
        tag: str | None = None,
    ) -> bool:
        """Replace the visible rows (one ``set_options``) and the border titles.

        Marks are kept by id (spec 1.5.5): a row a search, sort or tag hides
        stays marked, and the marks bar counts it as hidden.

        Args:
            rows: Selectable library rows to render when no recovery state is
                active (for characters: the loaded windows, spec 1.5.6).
            total: The kind's total: the title's count when unfiltered, the
                "of M" when filtered (and the no-match line's "N in total"),
                and the density rule's item count.
            filtered: Whether a search or tag narrows the list's count.
            matched: The narrowed list's COUNT (default: ``len(rows)`` when
                filtered, else ``total``): the M of the bottom border's
                ``#N of M``.
            title: The top-border title; built from the kind and the counts
                when omitted (spec 1.5.1).
            recovery_copy: Optional multi-line recovery copy. When present, the
                pane shows it instead of list or empty rows.
            recovery_id: Stable DOM id for the recovery copy widget.
            anchor: Where the list stood before a reload from the anchor.
            query: The search these rows answer; with ``tag``, an empty list
                says what found nothing instead of "No <kind> yet" (spec 1.5.4).
            tag: The tag these rows answer.

        Returns:
            Whether the no-match message shows and owes the other kinds'
            counts (the screen then counts them, spec 1.5.4); False when it
            kept the counts landed for the same query.
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
        no_match: frame_state.NoMatchView | None = None
        status: tuple[str, str] | None = None
        if recovery_copy:
            status = (recovery_id, recovery_copy)
        elif visible_rows:
            pass
        elif query or tag:
            # TASK-33788: dev said "No <kind> yet" for a search that matched
            # nothing; "N in total" only where the count says it (filtered).
            # The other kinds' counts depend on the query alone: a sort or
            # tag change that keeps it repaints them and owes no count (they
            # flickered and were counted again until B2b's final fix wave).
            previous = self._no_match_args
            kept = (
                self._also_in_counts
                if previous is not None
                and previous[0] == query
                and self._also_in_landed
                else None
            )
            if kept is not None:
                counts, failed, capped = kept
                no_match = frame_state.no_match_view(
                    item_list.kind,
                    query=query,
                    tag=tag,
                    total=total if filtered else None,
                    counts=counts,
                    failed=failed,
                    capped=capped,
                )
            else:
                no_match = frame_state.no_match_view(
                    item_list.kind,
                    query=query,
                    tag=tag,
                    total=total if filtered else None,
                )
                self._also_in_landed = False
                self._also_in_counts = None
            self._no_match_args = (query, tag, total if filtered else None)
        else:
            status = (
                "personas-library-empty",
                frame_state.empty_kind_text(item_list.kind),
            )
        self._stop_loading()
        await self._sync_status(status)
        self._show_no_match(no_match)
        self._show_retry(bool(recovery_copy))
        if title is None:
            title = frame_state.items_border_title(
                item_list.kind,
                total=total,
                matched=self._list_total if filtered else None,
                failed=bool(recovery_copy),
            )
        self.set_title(title)
        return no_match is not None and not self._also_in_landed

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

    def _show_no_match(self, view: frame_state.NoMatchView | None) -> None:
        """Show or hide the no-match message; updated in place, never mounted.

        A target that held focus and hides hands it to the list (I1).
        """
        block = self.query_one("#personas-library-nomatch", Vertical)
        focused = self.app.focused if self.is_attached else None
        held = focused is not None and block in focused.ancestors
        if view is None:
            self._no_match_args = None
        else:
            for part, text in zip(("line", "hint"), view.lines):
                self.query_one(f"#personas-library-nomatch-{part}", Static).update(text)
            targets = {target.kind: target for target in view.targets}
            self.query_one("#personas-library-also-in-label", Static).display = bool(
                targets
            )
            for kind in frame_state.ALSO_IN_ORDER:
                button = self.query_one(f"#personas-library-also-in-{kind}", Button)
                target = targets.get(kind)
                if target is not None:
                    button.label = Content(target.label)
                button.display = target is not None
            self.query_one("#personas-library-also-in-none", Static).display = (
                view.none_text is not None
            )
        block.display = view is not None
        if held and not (view is not None and focused.display):
            self.item_list.focus()

    def show_also_in(
        self,
        counts: Mapping[str, int],
        failed: Collection[str],
        capped: Collection[str] = (),
    ) -> None:
        """Lay the other kinds' counts over the no-match message (spec 1.5.4).

        The screen fences them to the list they were counted for; a message
        that is no longer shown takes nothing. ``capped`` names the kinds
        whose count is a lower bound (``100+``).
        """
        if self._no_match_args is None:
            return
        query, tag, total = self._no_match_args
        self._show_no_match(
            frame_state.no_match_view(
                self.item_list.kind,
                query=query,
                tag=tag,
                total=total,
                counts=counts,
                failed=failed,
                capped=capped,
            )
        )
        self._also_in_landed = True
        self._also_in_counts = (dict(counts), frozenset(failed), frozenset(capped))

    def hide_also_in(self) -> None:
        """Hide the targets: the search moved on from the query they were counted for.

        The no-match lines stay until the newer query's rows replace them.
        """
        if self._no_match_args is None:
            return
        query, tag, total = self._no_match_args
        self._show_no_match(
            frame_state.no_match_view(
                self.item_list.kind, query=query, tag=tag, total=total
            )
        )
        self._also_in_landed = False
        self._also_in_counts = None

    def also_in_owed(self, query: str) -> bool:
        """Whether the no-match message for ``query`` shows with no counts laid over it."""
        args = self._no_match_args
        return args is not None and args[0] == query and not self._also_in_landed

    def focus_also_in(self, *, query: str | None = None) -> bool:
        """Focus the first ``Also in:`` target, if one shows (for ``query``, when given)."""
        args = self._no_match_args
        if args is None or (query is not None and args[0] != query):
            return False
        for kind in frame_state.ALSO_IN_ORDER:
            button = self.query_one(f"#personas-library-also-in-{kind}", Button)
            if button.display:
                button.focus()
                return True
        return False

    def no_match_showing(self) -> bool:
        """Whether the no-match message shows: a search or tag listed nothing."""
        return bool(self.query_one("#personas-library-nomatch", Vertical).display)

    def _show_retry(self, shown: bool) -> None:
        """Show Retry with the recovery copy; a hidden Retry hands its focus to the list (I1).

        A server reload blanks the list first (``show_loading``), which hid
        Retry under the focus; a read that then failed again brought Retry
        back unfocused, so a second Enter did nothing (new with B2b, closed
        in its final fix wave). The blank remembers that Retry held the
        focus: a failed read gives it back, and rows leave it on the list.
        """
        retry = self.query_one("#personas-library-retry", Button)
        owed = self._retry_owed_focus
        self._retry_owed_focus = False
        if not shown and retry.has_focus:
            self._retry_owed_focus = True
            self.item_list.focus()
        retry.display = shown
        if shown and owed:
            retry.focus()

    async def show_loading(self) -> None:
        """A kind's first read started: blank the old kind's rows now, and say
        ``Loading <kind>…`` only if the read takes 250 ms (spec 1.5.4).

        On dev the previous kind's rows and title stayed up until the new
        kind's read landed, and Personas said "No personas yet" meanwhile.
        The next ``update_rows`` ends the wait.
        """
        self._stop_loading()
        self.item_list.show_rows((), kind_total=0)
        self._list_total = None
        self._filtered = False
        await self._sync_status(None)
        self._show_no_match(None)
        self._show_retry(False)
        self.set_title(frame_state.items_border_title(self.item_list.kind, total=None))
        serial = self._loading_serial
        self._loading_timer = self.set_timer(
            frame_state.LOADING_DELAY_SECONDS, lambda: self._loading_due(serial)
        )

    def _loading_due(self, serial: int) -> None:
        """The read is still running after the delay: say so (a stopped timer's late call says nothing)."""
        if serial != self._loading_serial:
            return
        self._loading_timer = None
        loading = self.query_one("#personas-library-loading", Static)
        loading.update(frame_state.loading_text(self.item_list.kind))
        loading.display = True

    def _stop_loading(self) -> None:
        """The wait ended (rows, an empty kind or a failure arrived)."""
        self._loading_serial += 1
        if self._loading_timer is not None:
            self._loading_timer.stop()
            self._loading_timer = None
        self.query_one("#personas-library-loading", Static).display = False

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

    def set_sort_label(self, sort_key: str) -> None:
        """Name the list's sort on its button (the screen owns the sort cycle).

        The button paints the sort's full or short label, whichever the
        toolbar's fit chose (spec 1.5.1); its tooltip always says it whole.

        Args:
            sort_key: The sort (``frame_state.SORT_LABELS``; ``SERVER_SORT_KEY``
                while browsing the server's own order).
        """
        self._sort_key = sort_key
        self.query_one("#personas-library-sort", Button).tooltip = Content(
            frame_state.sort_tooltip(sort_key)
        )
        # A longer or shorter label can change the toolbar's fit.
        self._apply_toolbar_fit()

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

    def set_tag_active(self, active: bool) -> None:
        """Say on the Tag button whether a tag narrows the list.

        The button keeps the one label ``Tag``, so its fit never changes; the
        tag itself (untrusted text) shows only in the literal border title
        (frame slice B2a). Dev's button read ``Tag: <tag>``.
        """
        self.query_one("#personas-library-tag", Button).tooltip = (
            _TAG_ON_TOOLTIP if active else _TAG_OFF_TOOLTIP
        )

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
        # The searched list is empty and another kind matches: the first
        # "Also in:" target takes focus, so Enter again switches (spec 1.5.4).
        if not self.focus_also_in(query=event.value.strip()):
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

    @on(Button.Pressed, ".personas-library-also-in-target")
    def _also_in_pressed(self, event: Button.Pressed) -> None:
        """Switch to the target's kind with the query applied (spec 1.5.4)."""
        event.stop()
        kind = str(event.button.id or "").removeprefix("personas-library-also-in-")
        self.post_message(AlsoInRequested(kind))

    @on(Button.Pressed, "#personas-library-retry")
    def _retry_pressed(self, event: Button.Pressed) -> None:
        """Read the failed list again (spec 1.5.4 "Failure")."""
        event.stop()
        self.post_message(PersonaActionRequested(action="retry"))

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
