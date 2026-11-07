"""Roleplay item-list widget and its literal row projection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .personas_library_pane import PersonasLibraryPane
from loguru import logger
from textual import events
from textual.binding import Binding
from textual.content import Content
from textual.events import Resize
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from ...UI.Persona_Modules import roleplay_frame_state as frame_state
from .personas_messages import (
    PersonaEntityKind,
)

logger = logger.bind(module="PersonasLibraryPane")


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
        from .personas_library_pane import PersonasLibraryPane

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
