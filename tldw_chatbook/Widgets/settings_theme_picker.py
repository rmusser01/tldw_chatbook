# tldw_chatbook/Widgets/settings_theme_picker.py
"""Settings ▸ Theme picker and the picker/editor pane (TASK-32948)."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Literal

from loguru import logger
from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.message import Message
from textual.widgets import Button, ContentSwitcher, Input, OptionList, Static
from textual.widgets.option_list import Option

from ..Backup_Recovery.bootstrap import RecoveryRequired
from ..config import get_user_themes_dir
from ..css.Themes.theme_catalog import (
    CACHE_REFRESH_FAILED,
    ORIGIN_LABELS,
    ThemeChange,
    ThemeEntry,
    build_catalog,
    current_launch_default,
    display_name,
    launch_default_restorable,
    persist_launch_default_async,
    revert_theme,
    use_theme,
    use_theme_toast,
    user_theme_names,
)
from ..css.Themes.themes import printable
from ..Utils.input_validation import escape_markup
from .settings_theme_editor import (
    THEME_STACK_BELOW,
    THEMES_UNAVAILABLE_LABEL,
    SettingsThemeEditor,
    theme_stack_width,
)
from .theme_preview import ThemePreview

_GROUP_TITLES = {"yours": "YOUR THEMES", "shipped": "SHIPPED", "textual": "BUILT-IN"}  # TASK-33073
THEMES_LOADING_LABEL = "Loading your themes…"
# TASK-32957: one exclusive group per picker for the off-thread folder scan.
_SCAN_GROUP = "settings-theme-scan"


def _bare_name(entry: ThemeEntry) -> str:
    """The entry's label without the origin word a shared name carries.

    build_catalog tells "Solarized Light" apart as "Solarized Light ·
    built-in"; the card title already names the origin, and the filter
    matches names only (P3 review M3/M4).
    """
    return entry.display_name.removesuffix(f" · {ORIGIN_LABELS[entry.origin]}")


_MIN_NAME_CELLS = 10  # a narrow row keeps at least this much of the name


def _row(entry: ThemeEntry, width: int | None = None) -> Text:
    """One list row: name, colour strip, then the active/launch/overrides markers.

    Args:
        entry: The theme to show.
        width: The cells the row may take. When the whole row doesn't fit,
            the tail gives way first, least important part first, until the
            name keeps ``_MIN_NAME_CELLS`` (TASK-33074): "overrides built-in"
            becomes "overrides", then goes, then the strip shrinks to three
            swatches, then "launch" goes. "active" always stays. Only then
            does the name take an ellipsis. None means no limit.
    """
    flags = [m for m, on_ in (("active", entry.is_active), ("launch", entry.is_launch_default)) if on_]
    overrides = [f"overrides {ORIGIN_LABELS[entry.overrides]}"] if entry.overrides else []
    strip = list(entry.strip)
    tails = [(strip, flags + overrides)]
    if width is not None:
        short = ["overrides"] if overrides else []
        tails += [(strip, flags + short), (strip, flags), (strip[:3], flags), (strip[:3], ["active"] if entry.is_active else [])]

    def build(colours: list[str], markers: list[str]) -> Text:
        tail = Text("  ")
        for colour in colours:
            # entry.strip is always an uppercase #RRGGBB (theme_catalog._colour_hex
            # resolves alpha hex and ANSI colour names before they reach here).
            tail.append("▮", Style(color=colour))
        if markers:
            tail.append("  " + " · ".join(markers))
        return tail

    name = Text(entry.display_name)
    if width is None:
        return name + build(*tails[0])
    need = min(name.cell_len, _MIN_NAME_CELLS)
    built = [build(*t) for t in tails]
    tail = next((t for t in built if t.cell_len + need <= width), built[-1])
    name.truncate(max(width - tail.cell_len, 1), overflow="ellipsis")
    return name + tail


class ThemeFilterInput(Input):
    BINDINGS: ClassVar[list[Binding]] = [Binding("down", "to_list", "List", show=False)]

    def action_to_list(self) -> None:
        self.parent_picker().focus_list()

    def parent_picker(self) -> ThemePicker:
        return self.query_ancestor(ThemePicker)


class ThemeOptionList(OptionList):
    BINDINGS: ClassVar[list[Binding]] = [
        # TASK-33072: vim keys too. The Settings screen's own j/k rail
        # handler stands aside while focus is in the detail pane.
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("t", "try_theme", "Try"),
        Binding("c", "clone_theme", "Clone"),
        Binding("n", "new_theme", "New"),
        Binding("e", "edit_theme", "Edit"),
        Binding("r", "rename_theme", "Rename"),
        Binding("delete", "delete_theme", "Delete"),
        Binding("i", "import_theme", "Import"),
    ]

    def row_width(self) -> int | None:
        """Cells a row may take (TASK-33074).

        The vertical scrollbar is always reserved: the ~95-row catalog
        overflows at every size, and it appearing sends no Resize.

        Returns:
            The width in cells left for a row's text after the scrollbar and
            the option padding; None before layout gives the list a width.
        """
        padding = self.get_component_styles("option-list--option").padding
        width = self.content_region.width - self.styles.scrollbar_size_vertical - padding.width
        return width if width > 0 else None

    def on_resize(self, event: events.Resize) -> None:
        """Refit the rows' names to the new width."""
        self.query_ancestor(ThemePicker).refit_rows()

    def _first_enabled_index(self) -> int | None:
        for index in range(self.option_count):
            if not self.get_option_at_index(index).disabled:
                return index
        return None

    def action_cursor_up(self) -> None:
        # OptionList wraps at the ends (find_next_enabled); the picker wants
        # the top row to hand focus back to the filter instead (spec §5).
        if self.highlighted is None or self.highlighted == self._first_enabled_index():
            self.query_ancestor(ThemePicker).query_one("#settings-theme-filter").focus()
            return
        super().action_cursor_up()

    async def _on_click(self, event: events.Click) -> None:
        # Spec D2: a click only highlights (repaints the preview). OptionList's
        # own handler also selects (= Use), so stop it running after this one;
        # Enter and the Use button stay the only ways to Use.
        event.prevent_default()
        clicked = event.style.meta.get("option")
        if clicked is not None and not self.get_option_at_index(clicked).disabled:
            self.highlighted = clicked

    def action_try_theme(self) -> None:
        self.query_ancestor(ThemePicker).try_highlighted()

    def action_clone_theme(self) -> None:
        self.query_ancestor(ThemePicker).request_edit("clone")

    def action_new_theme(self) -> None:
        self.query_ancestor(ThemePicker).request_edit("new")

    def action_edit_theme(self) -> None:
        self.query_ancestor(ThemePicker).request_edit("edit")

    def action_rename_theme(self) -> None:
        self.query_ancestor(ThemePicker).request_rename()

    def action_delete_theme(self) -> None:
        self.query_ancestor(ThemePicker).request_delete()

    def action_import_theme(self) -> None:
        self.query_ancestor(ThemePicker).request_import()


class ThemePicker(Vertical):
    class EditRequested(Message):
        """Open the editor on a theme (Clone, New or Edit).

        Args:
            theme_id: The source theme.
            mode: How the editor opens it.
        """

        def __init__(self, theme_id: str, mode: Literal["clone", "new", "edit"]) -> None:
            self.theme_id = theme_id
            self.mode = mode
            super().__init__()

    class RenameRequested(Message):
        """Rename… pressed on a saved theme; the screen prompts for the name.

        Args:
            theme_id: The saved theme to rename.
        """

        def __init__(self, theme_id: str) -> None:
            self.theme_id = theme_id
            super().__init__()

    class DeleteRequested(Message):
        """Delete pressed; the screen confirms, the editor deletes.

        Args:
            theme_id: The saved theme, or an ``unreadable:<stem>`` id.
        """

        def __init__(self, theme_id: str) -> None:
            self.theme_id = theme_id
            super().__init__()

    class ExportRequested(Message):
        """Export pressed on a saved theme; the editor writes the copy.

        Args:
            theme_id: The saved theme to export.
        """

        def __init__(self, theme_id: str) -> None:
            self.theme_id = theme_id
            super().__init__()

    class ImportRequested(Message):
        """Import… pressed; the screen owns the path prompt."""

    def __init__(
        self,
        list_themes: Callable[[], tuple[set[str], Mapping[str, str]]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.entries: list[ThemeEntry] = []
        self.highlighted_id: str | None = None
        self.files_available: bool = True
        # R27 (5): while paused, your themes keep their origin from the last
        # good listing instead of being relabelled "shipped".
        self._last_user_names: set[str] = set()
        self._last_unreadable: Mapping[str, str] = {}
        # R40(b): ONE listing per refresh -- (readable names, unreadable
        # stem -> short error); unreadable files are listed so they can be deleted.
        self._list_themes = list_themes or (lambda: (user_theme_names(get_user_themes_dir()), {}))
        # TASK-32948 Task 3: the last Export's full path, for Copy path.
        self._export_path: Path | None = None
        # TASK-32957: the folder scan runs in a thread worker. Only the newest
        # scan may land (a stale one never overwrites a newer listing), a
        # caller's highlight waits for it, and until the first one lands
        # YOUR THEMES shows a loading row. The pending highlight is
        # (requested id, the highlight when requested): a user move in
        # between wins over it (Qodo 4116061873).
        # TASK-33069: the highlight from before the filter was typed, so
        # clearing it returns there instead of to row 1.
        self._prefilter_highlight: str | None = None
        # Qodo 4118068766: the row the list fell back to because the filter
        # hid the highlight. Still on it when the filter clears = the user
        # never moved, so the pre-filter highlight wins; moved = theirs sticks.
        self._filter_fallback: str | None = None
        self._filtering = False
        self._scan_generation = 0
        self._pending_highlight: tuple[str, str | None] | None = None
        self._scanned = False

    # The pending Revert lives on the app, not this widget: the pane is
    # recomposed on every category switch, and spec §5 keeps Revert for the
    # rest of the session (ruling R9).
    @property
    def _revert(self) -> ThemeChange | None:
        return getattr(self.app, "theme_revert_change", None)

    @_revert.setter
    def _revert(self, change: ThemeChange | None) -> None:
        self.app.theme_revert_change = change

    def _sync_revert_chip(self) -> None:
        revert = self.query_one("#settings-theme-revert", Button)
        change = self._revert
        restores_launch = change is not None and change.persisted and launch_default_restorable(self.app, change)
        # TASK-33061: no chip when reverting would change nothing -- the
        # target is already active and the launch default would stay as is.
        # The pending change is kept: a later Try/Use merges into it.
        revert.display = change is not None and (
            change.previous_active != str(self.app.theme)
            or (restores_launch and change.previous_launch_default != current_launch_default())
        )
        self.query_one("#settings-theme-revert-row").display = revert.display
        if change is None:
            return
        # Button labels parse markup; theme names are untrusted file text (R28),
        # and the launch default is hand-editable config text (printable()).
        label = f"Revert to {escape_markup(printable(display_name(change.previous_active)))}"
        # Only worth naming the launch default too when it persisted AND
        # disagrees with the active theme it's reverting to -- otherwise
        # they're the same theme and the parenthetical is noise. A missing
        # launch default is never written back, so say it stays put.
        if change.persisted and not restores_launch:
            label = f"{label} (launch unchanged)"
        elif restores_launch and change.previous_active != change.previous_launch_default:
            label = f"{label} (launch: {escape_markup(printable(display_name(change.previous_launch_default)))})"
        revert.label = label

    def compose(self) -> ComposeResult:
        with Horizontal(id="settings-theme-picker-columns"):
            with Vertical(id="settings-theme-list-column"):
                yield ThemeFilterInput(placeholder="Filter themes", id="settings-theme-filter")
                yield Static(
                    "", id="settings-theme-launch-missing", classes="settings-help-copy", markup=False
                )
                yield ThemeOptionList(id="settings-theme-list")
                yield Static("", id="settings-theme-empty", classes="settings-help-copy", markup=False)
                # Spec §9: a no-match filter offers a way back.
                with Horizontal(id="settings-theme-clear-filter-row", classes="settings-action-row"):
                    yield Button("Clear filter", id="settings-theme-clear-filter", classes="theme-editor-action")
            with Vertical(id="settings-theme-card-column"):
                yield Static("", id="settings-theme-card-title", classes="destination-section", markup=False)
                yield Static("", id="settings-theme-card-error", classes="settings-help-copy", markup=False)
                yield ThemePreview("settings-theme-picker-preview", id="settings-theme-picker-preview")
                # TASK-33064: three action groups -- switching (Use, Try,
                # then Revert on its own row: its label names up to two
                # themes), creation, and your-theme file actions.
                with Horizontal(classes="settings-action-row theme-action-group"):
                    yield Button("Use this theme", id="settings-theme-use", variant="primary", classes="theme-editor-action")
                    yield Button("Try", id="settings-theme-try", classes="theme-editor-action")
                with Horizontal(id="settings-theme-revert-row", classes="settings-action-row"):
                    yield Button("Revert", id="settings-theme-revert", classes="theme-editor-action")
                with Horizontal(classes="settings-action-row theme-action-group"):
                    yield Button("Clone", id="settings-theme-picker-clone", classes="theme-editor-action")
                    yield Button("New", id="settings-theme-picker-new", classes="theme-editor-action")
                    yield Button("Import…", id="settings-theme-picker-import", classes="theme-editor-action")
                with Horizontal(id="settings-theme-yours-actions", classes="settings-action-row theme-action-group"):
                    yield Button("Edit", id="settings-theme-picker-edit", classes="theme-editor-action")
                    yield Button("Rename", id="settings-theme-picker-rename", classes="theme-editor-action")
                    yield Button("Export", id="settings-theme-picker-export", classes="theme-editor-action")
                    yield Button("Delete", id="settings-theme-picker-delete", variant="error", classes="theme-editor-action")
                with Horizontal(id="settings-theme-export-result", classes="settings-action-row"):
                    yield Static(
                        "", id="settings-theme-export-path", classes="settings-help-copy", markup=False
                    )
                    yield Button("Copy path", id="settings-theme-copy-path", classes="theme-editor-action")

    def on_mount(self) -> None:
        self._sync_revert_chip()
        self.query_one("#settings-theme-empty").display = False
        self.app.theme_changed_signal.subscribe(self, lambda _theme: self.refresh_catalog(rescan=False))
        self.refresh_catalog(highlight=str(self.app.theme))

    # -- catalog -------------------------------------------------------
    def refresh_catalog(self, highlight: str | None = None, *, rescan: bool = True) -> None:
        """Rebuild the list from the registered themes and the saved files.

        Args:
            highlight: The theme id to highlight; default keeps the current.
            rescan: False reuses the last good listing of the themes folder.
                A theme switch (Use, Try, Revert, the palette) only moves
                the active/launch markers, and the backup-scoped scan costs
                ~6 ms per file (Qodo 4107495860). While the files are paused
                it rescans anyway, to notice the resume.

        A rescan runs in a thread worker (TASK-32957); the last good listing
        stays up meanwhile and ``highlight`` is applied when it lands.
        """
        if not rescan and self.files_available:
            if highlight is not None:
                self._pending_highlight = None  # a newer, explicit highlight
            self._rebuild(self._last_user_names, self._last_unreadable, highlight)
            return
        self._scan_generation += 1
        if not self._scanned:
            # First open: show the registered themes now, a loading row for yours.
            self._rebuild(self._last_user_names, self._last_unreadable, highlight)
        if highlight is not None:
            self._pending_highlight = (highlight, self.highlighted_id)
        generation = self._scan_generation
        self.run_worker(
            lambda: self._scan(generation),
            thread=True,
            exclusive=True,
            group=_SCAN_GROUP,
            exit_on_error=False,
        )

    def _scan(self, generation: int) -> None:
        """Worker thread: list the themes folder, then apply on the UI thread."""
        try:
            outcome: tuple[str, set[str], Mapping[str, str]] = ("ok", *self._list_themes())
        except RecoveryRequired:
            outcome = ("paused", set(), {})
        except OSError as exc:
            # R27 (6): an unreadable themes dir reads as "no user themes".
            logger.warning(f"Could not list saved themes (scan {generation}): {exc.strerror or type(exc).__name__}")
            outcome = ("oserror", set(), {})
        except Exception as exc:  # noqa: BLE001 - never leave "Loading" up for good
            # Qodo 4116061876/4116061867: the frames, not str(exc) -- the
            # message may carry the themes path (R16).
            frames = "".join(traceback.format_tb(exc.__traceback__))
            logger.error(f"Saved-theme listing failed (scan {generation}): {type(exc).__name__}\n{frames}")
            outcome = ("failed", set(), {})
        self.app.call_from_thread(self._apply_scan, generation, outcome)

    def _apply_scan(self, generation: int, outcome: tuple[str, set[str], Mapping[str, str]]) -> None:
        """UI thread: land one scan unless a newer one has started since."""
        if generation != self._scan_generation or not self.is_attached:
            return
        status, user_names, unreadable = outcome
        self._scanned = True
        if status in ("paused", "failed"):
            # A failed scan is not "no user themes" (R27 (6) is OSError only):
            # keep the last good listing and the current availability.
            user_names, unreadable = self._last_user_names, self._last_unreadable
            if status == "paused":
                self.files_available = False
        else:
            self.files_available = True
            if status == "ok":
                self._last_user_names, self._last_unreadable = user_names, unreadable
        pending, self._pending_highlight = self._pending_highlight, None
        highlight = pending[0] if pending is not None and pending[1] == self.highlighted_id else None
        self._rebuild(user_names, unreadable, highlight)

    def _rebuild(
        self, user_names: set[str], unreadable: Mapping[str, str], highlight: str | None
    ) -> None:
        self.entries = build_catalog(
            self.app.available_themes,
            user_names,
            str(self.app.theme),
            current_launch_default(),
            unreadable=unreadable,
        )
        self._sync_revert_chip()  # a rename/delete may have retargeted it
        self._sync_launch_missing()
        self._render_list(highlight or self.highlighted_id)

    def _sync_launch_missing(self) -> None:
        """Spec §9: the launch default may point at a theme id that is no
        longer registered (its file was deleted or renamed elsewhere)."""
        launch = current_launch_default()
        notice = self.query_one("#settings-theme-launch-missing", Static)
        missing = launch not in self.app.available_themes
        notice.display = missing
        # Qodo 4109320405: a hand-edited config value may carry ESC.
        notice.update(f"Launch default missing: {printable(launch)} — Use any theme to fix it" if missing else "")

    def _render_list(self, highlight: str | None) -> None:
        query = self.query_one("#settings-theme-filter", Input).value.strip().casefold()
        # Names only, never origin words (P3 review M4): the group headers
        # already sort by origin, and matching "built-in"/"shipped" would make
        # every short query ("i", "p") list whole groups.
        shown = [e for e in self.entries if not query or query in _bare_name(e).casefold() or query in e.id.casefold()]
        lst = self.query_one("#settings-theme-list", ThemeOptionList)
        width = lst.row_width()
        options: list[Option] = []
        for origin, title in _GROUP_TITLES.items():
            group = [e for e in shown if e.origin == origin]
            placeholder = None
            if origin == "yours":
                if not self.files_available:
                    placeholder = THEMES_UNAVAILABLE_LABEL
                elif not self._scanned:
                    placeholder = THEMES_LOADING_LABEL
                elif not group and not query:
                    placeholder = "(none yet)"  # task-32945, spec §9
            if not group and placeholder is None:
                continue
            header = f"{title} ({len(group)})" if query else title
            options.append(Option(header, disabled=True))
            options.extend(Option(_row(e, width), id=e.id) for e in group)
            if placeholder is not None:
                options.append(Option(placeholder, disabled=True))
        lst.clear_options()
        lst.add_options(options)
        empty = self.query_one("#settings-theme-empty", Static)
        empty.display = not shown
        empty.update(f"No themes match '{query}'" if not shown else "")
        # The button too: its own display doesn't follow a hidden row's
        # (see _show's export-result note).
        self.query_one("#settings-theme-clear-filter-row").display = not shown
        self.query_one("#settings-theme-clear-filter", Button).display = not shown
        ids = [e.id for e in shown]
        # TASK-33069: a highlight the filter hid comes back when it shows
        # again; failing that the active theme, and only then row 1.
        target = next(
            (c for c in (highlight, self._prefilter_highlight, str(self.app.theme)) if c in ids),
            ids[0] if ids else None,
        )
        if target != highlight:
            self._filter_fallback = target
        if target is not None:
            lst.highlighted = lst.get_option_index(target)
        pending = self._pending_highlight
        if pending is not None and pending[1] == self.highlighted_id:
            # A re-render's move is not the user's: the request still stands.
            self._pending_highlight = (pending[0], target)
        self._show(target)

    def refit_rows(self) -> None:
        """Re-truncate the rows' names for the list's current width, in place."""
        lst = self.query_one("#settings-theme-list", ThemeOptionList)
        width = lst.row_width()
        by_id = {e.id: e for e in self.entries}
        for index in range(lst.option_count):
            entry = by_id.get(lst.get_option_at_index(index).id)
            if entry is not None:
                lst.replace_option_prompt_at_index(index, _row(entry, width))

    def _show(self, theme_id: str | None) -> None:
        self.highlighted_id = theme_id
        # Task 3: an Export result is only good for the theme it was made
        # from; a fresh highlight clears it (spec §7). A widget's own
        # `.display` doesn't inherit a hidden ancestor's, so the button also
        # gets its own flag -- otherwise it stays keyboard-reachable (and
        # counted by the reachability walk) while its row is invisible.
        self.query_one("#settings-theme-export-result").display = False
        self.query_one("#settings-theme-copy-path", Button).display = False
        entry = self._highlighted_entry()
        error = entry.error if entry is not None else None
        # Tooltips parse markup; the error quotes untrusted file content.
        unreadable_tip = f"This theme file can't be read: {escape_markup(error)}" if error else None
        title = self.query_one("#settings-theme-card-title", Static)
        for button_id in ("#settings-theme-use", "#settings-theme-try", "#settings-theme-picker-clone"):
            button = self.query_one(button_id, Button)
            button.disabled = entry is None or error is not None
            button.tooltip = unreadable_tip
        # R29: New is blocked on an unreadable row too (it would start from a
        # stale palette); with nothing highlighted it stays available.
        new = self.query_one("#settings-theme-picker-new", Button)
        new.disabled = error is not None
        new.tooltip = unreadable_tip
        is_yours = entry is not None and entry.origin == "yours"
        # The group row too, or its top margin leaves a gap under Import.
        self.query_one("#settings-theme-yours-actions").display = is_yours
        tooltip = None if self.files_available else THEMES_UNAVAILABLE_LABEL
        import_button = self.query_one("#settings-theme-picker-import", Button)
        import_button.disabled = not self.files_available
        import_button.tooltip = tooltip
        for button_id in (
            "#settings-theme-picker-edit",
            "#settings-theme-picker-rename",
            "#settings-theme-picker-delete",
            "#settings-theme-picker-export",
        ):
            button = self.query_one(button_id, Button)
            button.display = is_yours
            button.disabled = not self.files_available
            button.tooltip = tooltip
            # Delete stays available: removing a broken file is the fix.
            if error is not None and button_id != "#settings-theme-picker-delete":
                button.disabled = True
                button.tooltip = unreadable_tip
        card_error = self.query_one("#settings-theme-card-error", Static)
        card_error.display = error is not None
        card_error.update(error or "")
        # TASK-33067/33069: no preview for nothing, and an unreadable file's
        # error stands in its place (its palette is all grey, 1:1).
        preview = self.query_one(ThemePreview)
        preview.display = entry is not None and error is None
        if entry is None:
            title.update("")
            return
        if error is not None:
            title.update(entry.display_name)
        else:
            tone = "dark" if entry.dark else "light"
            title.update(f"{_bare_name(entry)}  ·  {tone} · {ORIGIN_LABELS[entry.origin]}")
        if preview.display:
            preview.paint(dict(entry.colours))

    def _highlighted_entry(self) -> ThemeEntry | None:
        return next((e for e in self.entries if e.id == self.highlighted_id), None)

    def _highlighted_readable(self) -> bool:
        """False for an unreadable file: only Delete acts on it (R29).

        R40(d): a blocked action says why, once, instead of doing nothing.
        """
        entry = self._highlighted_entry()
        if entry is None or entry.error is None:
            return True
        # notify parses markup; the error quotes untrusted file content.
        self.app.notify(f"This theme file can't be read: {escape_markup(entry.error)}", severity="warning")
        return False

    def focus_list(self) -> None:
        lst = self.query_one("#settings-theme-list", OptionList)
        if self.highlighted_id is not None:
            lst.focus()

    # -- events --------------------------------------------------------
    @on(Input.Changed, "#settings-theme-filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        event.stop()
        if not event.value.strip():
            self._end_filter()
            return
        if not self._filtering:
            self._filtering = True
            self._prefilter_highlight = self.highlighted_id
            self._filter_fallback = None
        self._render_list(self.highlighted_id)

    def _end_filter(self) -> None:
        """Re-list every theme once the filter is empty: back on the
        pre-filter highlight, unless the user moved while filtering."""
        moved = self.highlighted_id != self._filter_fallback
        self._render_list(self.highlighted_id if moved else None)
        self._prefilter_highlight = self._filter_fallback = None
        self._filtering = False

    @on(Button.Pressed, "#settings-theme-clear-filter")
    def _clear_filter_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        with self.prevent(Input.Changed):
            self.query_one("#settings-theme-filter", Input).value = ""
        self._end_filter()
        self.focus_list()

    @on(Input.Submitted, "#settings-theme-filter")
    def _filter_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.focus_list()

    @on(OptionList.OptionHighlighted, "#settings-theme-list")
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        self._show(event.option.id)

    @on(OptionList.OptionSelected, "#settings-theme-list")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.use_highlighted()

    @on(Button.Pressed, "#settings-theme-use")
    def _use_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.use_highlighted()

    @on(Button.Pressed, "#settings-theme-try")
    def _try_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.try_highlighted()

    @on(Button.Pressed, "#settings-theme-picker-clone")
    def _clone_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_edit("clone")

    @on(Button.Pressed, "#settings-theme-picker-new")
    def _new_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_edit("new")

    @on(Button.Pressed, "#settings-theme-picker-import")
    def _import_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_import()

    @on(Button.Pressed, "#settings-theme-picker-edit")
    def _edit_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_edit("edit")

    @on(Button.Pressed, "#settings-theme-picker-rename")
    def _rename_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_rename()

    @on(Button.Pressed, "#settings-theme-picker-delete")
    def _delete_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_delete()

    @on(Button.Pressed, "#settings-theme-picker-export")
    def _export_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_export()

    @on(Button.Pressed, "#settings-theme-copy-path")
    def _copy_path_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._export_path is None:
            return
        self.app.copy_to_clipboard(str(self._export_path))
        self.app.notify("Path copied", severity="information")

    @on(Button.Pressed, "#settings-theme-revert")
    def _revert_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._revert is None:
            return
        change, self._revert = self._revert, None
        try:
            restored, caches_reloaded = revert_theme(self.app, change)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Theme revert failed: {exc}")
            self.app.notify(f"Could not revert the theme: {escape_markup(exc)}", severity="error")
        else:
            if not restored:
                self.app.notify("Reverted the theme; the launch default was not restored", severity="warning")
            elif not caches_reloaded:
                self.app.notify(f"Reverted the theme; {CACHE_REFRESH_FAILED}", severity="warning")
        self._sync_revert_chip()
        self.refresh_catalog(rescan=False)

    # -- actions -------------------------------------------------------
    def use_highlighted(self) -> None:
        """Use: switch to the highlighted theme and save it as the launch default."""
        self._switch(persist=True)

    def try_highlighted(self) -> None:
        """Try: switch to the highlighted theme for this session only."""
        self._switch(persist=False)

    def request_edit(self, mode: Literal["clone", "new", "edit"]) -> None:
        """Ask the pane to open the editor on the highlighted theme.

        Args:
            mode: ``clone``/``edit`` need a highlighted, readable theme
                (``edit`` also one of yours, with files available). ``new``
                with nothing highlighted (a filter matched nothing) starts
                from the running theme (Qodo 4104047326).
        """
        theme_id = self.highlighted_id
        if theme_id is None:
            if mode != "new":
                return
            theme_id = str(self.app.theme)
        elif not self._highlighted_readable():
            return
        if mode == "edit" and not self._can_manage_files():
            return
        self.post_message(self.EditRequested(theme_id, mode))

    def request_rename(self) -> None:
        """Post ``RenameRequested`` for a readable saved theme while files are available."""
        if self._can_manage_files() and self._highlighted_readable():
            self.post_message(self.RenameRequested(self.highlighted_id))

    def request_delete(self) -> None:
        """Post ``DeleteRequested`` for a saved theme, readable or not, while files are available."""
        if self._can_manage_files():
            self.post_message(self.DeleteRequested(self.highlighted_id))

    def request_export(self) -> None:
        """Post ``ExportRequested`` for a readable saved theme while files are available."""
        if self._can_manage_files() and self._highlighted_readable():
            self.post_message(self.ExportRequested(self.highlighted_id))

    def request_import(self) -> None:
        """Post ``ImportRequested`` unless backup/recovery holds the files."""
        if self.files_available:
            self.post_message(self.ImportRequested())

    def show_export_result(self, path: Path) -> None:
        """Show where a saved theme was just exported, with Copy path (spec §7).

        Args:
            path: The written file's full path (the one notice allowed to
                show a path, R16).
        """
        self._export_path = path
        self.query_one("#settings-theme-export-path", Static).update(f"Exported to {path}")
        self.query_one("#settings-theme-export-result").display = True
        self.query_one("#settings-theme-copy-path", Button).display = True

    def _can_manage_files(self) -> bool:
        # Rename/Delete/Export/Edit all touch the theme file on disk: gated
        # on the highlighted entry being one of yours AND on files being
        # reachable (backup/recovery may be holding them, spec pause row).
        entry = self._highlighted_entry()
        return entry is not None and entry.origin == "yours" and self.files_available

    def _switch(self, *, persist: bool) -> None:
        theme_id = self.highlighted_id
        if theme_id is None or not self._highlighted_readable():
            return
        try:
            # TASK-33121: the switch shows now; Use's launch-default write
            # (~140 ms) runs off the UI thread in _persist_use.
            change = use_theme(self.app, theme_id, persist=False)
        except Exception as exc:  # noqa: BLE001 - stale entry / unregistered theme
            self.app.notify(f"Could not apply {escape_markup(display_name(theme_id))}: {escape_markup(exc)}", severity="error")
            self.refresh_catalog()
            return
        if persist:
            # Revert counts on the write landing: its own write queues behind
            # this one (theme_catalog's one writer thread), so it still wins.
            change = replace(change, persisted=True)
        if change.previous_active.startswith("custom_"):
            # Review I-1/P6: an editor Try's custom_* registration is never a
            # row; Revert targets the listed theme behind it, else the launch
            # default -- never a theme the picker cannot show.
            base = change.previous_active[len("custom_") :]
            listed = base if base in self.app.available_themes else change.previous_launch_default
            change = replace(change, previous_active=listed)
        self._revert = change if self._revert is None else self._revert.merge(change)
        self._sync_revert_chip()
        if not persist:
            self.app.notify(f"Trying {escape_markup(display_name(theme_id))} for this session", severity="information")
        else:
            # App-owned: leaving Settings mid-write must not drop the report.
            self.app.run_worker(self._persist_use(theme_id, change), group="settings-theme-use", exit_on_error=False)
        self.refresh_catalog(highlight=theme_id, rescan=False)

    async def _persist_use(self, theme_id: str, change: ThemeChange) -> None:
        """Write Use's launch default off the UI thread, then say how it went.

        Args:
            theme_id: The theme Use switched to.
            change: The switch, for the toast's "was:" name.
        """
        try:
            persisted, caches_reloaded = await persist_launch_default_async(self.app, theme_id)
        except Exception as exc:  # noqa: BLE001 - reported as "not saved" below
            logger.warning(f"Saving the launch default failed: {type(exc).__name__}")
            persisted, caches_reloaded = False, False
        outcome = replace(change, persisted=persisted, caches_reloaded=caches_reloaded)
        message, severity = use_theme_toast(theme_id, outcome)
        self.app.notify(message, severity=severity)
        if self.is_attached:
            self.refresh_catalog(rescan=False)  # the launch marker moves now


class ThemePane(ContentSwitcher):
    """Picker first; the editor swaps in for Clone/New/Edit (spec §4, D3).

    The picker lists and changes saved theme files only through the editor's
    backup-scoped file API. Rename bubbles to the screen, which owns the
    name prompt.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(initial="settings-theme-picker", **kwargs)
        self.add_class("-picker")  # `initial` bypasses watch_current

    # TASK-33064: own-width layout switches (the workbench's compact class
    # only fires at <=100 terminal cols; 101-180 squeezed the columns).
    # Below THEME_STACK_BELOW (shared with the editor, review M-2) the list
    # and card stack; below _NARROW_CARD_BELOW card cols the chips stack too
    # (four chips need ~48).
    _NARROW_CARD_BELOW = 48

    def on_resize(self, event: events.Resize) -> None:
        """Stack the card under the list, and narrow its chips, by body width.

        Args:
            event: The resize event (the width is re-measured on the body).
        """
        width = theme_stack_width(self)
        stacked = width < THEME_STACK_BELOW
        self.set_class(stacked, "-stacked")
        card_width = width if stacked else width // 2
        self.set_class(card_width < self._NARROW_CARD_BELOW, "-narrow-card")

    def watch_current(self, old: str | None, new: str | None) -> None:
        """Size the pane for the view being switched to.

        Args:
            old: The id of the view being left.
            new: The id of the view being shown.
        """
        super().watch_current(old, new)
        # TASK-33064: the picker fills the detail pane (CSS `-picker`); the
        # taller editor keeps the auto height so the pane body scrolls it.
        self.set_class(new == "settings-theme-picker", "-picker")

    def compose(self) -> ComposeResult:
        # The editor is composed first so the picker's first refresh_catalog
        # (its on_mount) can reach it through _editor_listing.
        with Vertical(id="settings-theme-editor-view"):
            with Horizontal(classes="settings-action-row"):
                yield Button("Back to themes", id="settings-theme-back", classes="theme-editor-action")
            # Held directly: the picker's listing runs on a worker thread,
            # which must not query the DOM (TASK-32957).
            self._theme_editor = SettingsThemeEditor(id="settings-theme-editor")
            yield self._theme_editor
        yield ThemePicker(
            list_themes=self._editor_listing,
            id="settings-theme-picker",
        )

    def _editor(self) -> SettingsThemeEditor:
        return self._theme_editor

    def _editor_listing(self) -> tuple[set[str], dict[str, str]]:
        files, unreadable = self._editor().user_theme_listing()
        return set(files), unreadable

    def show_picker(self) -> None:
        self._editor().discard_try()  # review I-1: every Back undoes an unsaved Try
        self.current = "settings-theme-picker"
        picker = self.query_one(ThemePicker)
        picker.refresh_catalog()
        picker.focus_list()

    def open_editor(self, theme_id: str, mode: Literal["clone", "new", "edit"]) -> None:
        editor = self._editor()
        picker = self.query_one(ThemePicker)
        entry = next((e for e in picker.entries if e.id == theme_id), None)
        if entry is not None and entry.origin == "yours":
            if not editor.load_user_theme(theme_id):
                # Qodo 4104047302: never open on the previous palette; the
                # load said why, and the listing may be stale.
                picker.refresh_catalog()
                return
        else:
            editor.load_theme(theme_id)
        if mode == "clone":
            editor.on_clone_theme()
        elif mode == "new":
            editor.on_new_theme()
        editor.set_editing_context(theme_id, mode)
        editor.set_files_available(picker.files_available)
        self.current = "settings-theme-editor-view"
        editor.query_one("#settings-theme-name").focus()

    @on(ThemePicker.EditRequested)
    def _edit_requested(self, event: ThemePicker.EditRequested) -> None:
        event.stop()
        self.open_editor(event.theme_id, event.mode)

    @on(ThemePicker.DeleteRequested)
    def _delete_requested(self, event: ThemePicker.DeleteRequested) -> None:
        event.stop()
        editor = self._editor()
        editor.run_file_action(editor.request_delete(event.theme_id))

    @on(ThemePicker.ExportRequested)
    def _export_requested(self, event: ThemePicker.ExportRequested) -> None:
        event.stop()
        editor = self._editor()
        editor.run_file_action(editor.export_theme(event.theme_id))

    @on(SettingsThemeEditor.Saved)
    def _saved(self, event: SettingsThemeEditor.Saved) -> None:
        """Save / Save as return to the picker with the saved theme highlighted.

        The category-leave Save tears this pane down right after saving, so
        a detached pane just ignores the message.
        """
        event.stop()
        if not self.is_attached:
            return
        try:
            self.show_picker()
            self.query_one(ThemePicker).refresh_catalog(highlight=event.theme_name)
        except QueryError:
            return

    @on(SettingsThemeEditor.ThemesChanged)
    def _themes_changed(self, event: SettingsThemeEditor.ThemesChanged) -> None:
        event.stop()
        self.query_one(ThemePicker).refresh_catalog(highlight=event.highlight)

    @on(SettingsThemeEditor.Exported)
    def _exported(self, event: SettingsThemeEditor.Exported) -> None:
        event.stop()
        if not self.is_attached:
            return
        try:
            self.query_one(ThemePicker).show_export_result(event.path)
        except QueryError:
            return
