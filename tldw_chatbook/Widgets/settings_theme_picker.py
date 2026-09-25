# tldw_chatbook/Widgets/settings_theme_picker.py
"""Settings ▸ Theme picker and the picker/editor pane (TASK-32948)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    ThemeChange,
    ThemeEntry,
    build_catalog,
    current_launch_default,
    display_name,
    revert_theme,
    use_theme,
    use_theme_toast,
    user_theme_names,
)
from .settings_theme_editor import THEMES_UNAVAILABLE_LABEL, SettingsThemeEditor
from .theme_preview import ThemePreview

_GROUP_TITLES = {"yours": "YOUR THEMES", "shipped": "SHIPPED", "textual": "TEXTUAL"}


def _row(entry: ThemeEntry) -> Text:
    text = Text(f"{entry.display_name}  ")
    for colour in entry.strip:
        # entry.strip is always an uppercase #RRGGBB (theme_catalog._colour_hex
        # resolves alpha hex and ANSI colour names before they reach here).
        text.append("▮", Style(color=colour))
    markers = [m for m, on_ in (("active", entry.is_active), ("launch", entry.is_launch_default)) if on_]
    if entry.overrides:
        markers.append(f"overrides {entry.overrides}")
    if markers:
        text.append("  " + " · ".join(markers))
    return text


class ThemeFilterInput(Input):
    BINDINGS: ClassVar[list[Binding]] = [Binding("down", "to_list", "List", show=False)]

    def action_to_list(self) -> None:
        self.parent_picker().focus_list()

    def parent_picker(self) -> ThemePicker:
        return self.query_ancestor(ThemePicker)


class ThemeOptionList(OptionList):
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("t", "try_theme", "Try"),
        Binding("c", "clone_theme", "Clone"),
        Binding("n", "new_theme", "New"),
        Binding("e", "edit_theme", "Edit"),
        Binding("r", "rename_theme", "Rename"),
        Binding("delete", "delete_theme", "Delete"),
    ]

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


class ThemePicker(Vertical):
    class EditRequested(Message):
        def __init__(self, theme_id: str, mode: Literal["clone", "new", "edit"]) -> None:
            self.theme_id = theme_id
            self.mode = mode
            super().__init__()

    class RenameRequested(Message):
        def __init__(self, theme_id: str) -> None:
            self.theme_id = theme_id
            super().__init__()

    class DeleteRequested(Message):
        def __init__(self, theme_id: str) -> None:
            self.theme_id = theme_id
            super().__init__()

    class ExportRequested(Message):
        def __init__(self, theme_id: str) -> None:
            self.theme_id = theme_id
            super().__init__()

    def __init__(
        self,
        list_user_names: Callable[[], set[str]] | None = None,
        list_unreadable: Callable[[], Mapping[str, str]] | None = None,
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
        self._list_user_names = list_user_names or (lambda: user_theme_names(get_user_themes_dir()))
        # Unreadable saved files (stem -> short error), listed so they can be deleted.
        self._list_unreadable = list_unreadable or dict

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
        revert.display = change is not None
        if change is None:
            return
        label = f"Revert to {display_name(change.previous_active)}"
        # Only worth naming the launch default too when it persisted AND
        # disagrees with the active theme it's reverting to -- otherwise
        # they're the same theme and the parenthetical is noise.
        if change.persisted and change.previous_active != change.previous_launch_default:
            label = f"{label} (launch: {display_name(change.previous_launch_default)})"
        revert.label = label

    def compose(self) -> ComposeResult:
        with Horizontal(id="settings-theme-picker-columns"):
            with Vertical(id="settings-theme-list-column"):
                yield ThemeFilterInput(placeholder="Filter themes", id="settings-theme-filter")
                yield ThemeOptionList(id="settings-theme-list")
                yield Static("", id="settings-theme-empty", classes="settings-help-copy", markup=False)
            with Vertical(id="settings-theme-card-column"):
                yield Static("", id="settings-theme-card-title", classes="destination-section", markup=False)
                yield Static("", id="settings-theme-card-error", classes="settings-help-copy", markup=False)
                yield ThemePreview("settings-theme-picker-preview", id="settings-theme-picker-preview")
                with Horizontal(classes="settings-action-row"):
                    yield Button("Use this theme", id="settings-theme-use", variant="primary", classes="theme-editor-action")
                    yield Button("Try", id="settings-theme-try", classes="theme-editor-action")
                    yield Button("Clone", id="settings-theme-picker-clone", classes="theme-editor-action")
                    yield Button("New", id="settings-theme-picker-new", classes="theme-editor-action")
                    yield Button("Revert", id="settings-theme-revert", classes="theme-editor-action")
                with Horizontal(classes="settings-action-row"):
                    yield Button("Edit", id="settings-theme-picker-edit", classes="theme-editor-action")
                    yield Button("Rename", id="settings-theme-picker-rename", classes="theme-editor-action")
                    yield Button("Delete", id="settings-theme-picker-delete", variant="error", classes="theme-editor-action")
                    yield Button("Export", id="settings-theme-picker-export", classes="theme-editor-action")

    def on_mount(self) -> None:
        self._sync_revert_chip()
        self.query_one("#settings-theme-empty").display = False
        self.app.theme_changed_signal.subscribe(self, lambda _theme: self.refresh_catalog())
        self.refresh_catalog(highlight=str(self.app.theme))

    # -- catalog -------------------------------------------------------
    def refresh_catalog(self, highlight: str | None = None) -> None:
        try:
            user_names = self._list_user_names()
            unreadable = self._list_unreadable()
            self.files_available = True
            self._last_user_names, self._last_unreadable = user_names, unreadable
        except RecoveryRequired:
            user_names, unreadable = self._last_user_names, self._last_unreadable
            self.files_available = False
        except OSError as exc:
            # R27 (6): an unreadable themes dir reads as "no user themes".
            logger.warning(f"Could not list saved themes: {exc.strerror or type(exc).__name__}")
            user_names, unreadable = set(), {}
            self.files_available = True
        self.entries = build_catalog(
            self.app.available_themes,
            user_names,
            str(self.app.theme),
            current_launch_default(),
            unreadable=unreadable,
        )
        self._sync_revert_chip()  # a rename/delete may have retargeted it
        self._render_list(highlight or self.highlighted_id)

    def _render_list(self, highlight: str | None) -> None:
        query = self.query_one("#settings-theme-filter", Input).value.strip().casefold()
        shown = [e for e in self.entries if not query or query in e.display_name.casefold() or query in e.id.casefold()]
        options: list[Option] = []
        for origin, title in _GROUP_TITLES.items():
            group = [e for e in shown if e.origin == origin]
            placeholder = None
            if origin == "yours":
                if not self.files_available:
                    placeholder = THEMES_UNAVAILABLE_LABEL
                elif not group and not query:
                    placeholder = "(none yet)"  # task-32945, spec §9
            if not group and placeholder is None:
                continue
            header = f"{title} ({len(group)})" if query else title
            options.append(Option(header, disabled=True))
            options.extend(Option(_row(e), id=e.id) for e in group)
            if placeholder is not None:
                options.append(Option(placeholder, disabled=True))
        lst = self.query_one("#settings-theme-list", OptionList)
        lst.clear_options()
        lst.add_options(options)
        empty = self.query_one("#settings-theme-empty", Static)
        empty.display = not shown
        empty.update(f"No themes match '{query}'" if not shown else "")
        ids = [e.id for e in shown]
        target = highlight if highlight in ids else (ids[0] if ids else None)
        if target is not None:
            lst.highlighted = lst.get_option_index(target)
        self._show(target)

    def _show(self, theme_id: str | None) -> None:
        self.highlighted_id = theme_id
        entry = self._highlighted_entry()
        error = entry.error if entry is not None else None
        unreadable_tip = f"This theme file can't be read: {error}" if error else None
        title = self.query_one("#settings-theme-card-title", Static)
        for button_id in ("#settings-theme-use", "#settings-theme-try", "#settings-theme-picker-clone"):
            button = self.query_one(button_id, Button)
            button.disabled = entry is None or error is not None
            button.tooltip = unreadable_tip
        is_yours = entry is not None and entry.origin == "yours"
        tooltip = None if self.files_available else THEMES_UNAVAILABLE_LABEL
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
        if entry is None:
            title.update("")
            return
        if error is not None:
            title.update(entry.display_name)
        else:
            tone = "dark" if entry.dark else "light"
            title.update(f"{entry.display_name}  ·  {tone} · {entry.origin}")
        self.query_one(ThemePreview).paint(dict(entry.colours))

    def _highlighted_entry(self) -> ThemeEntry | None:
        return next((e for e in self.entries if e.id == self.highlighted_id), None)

    def _highlighted_readable(self) -> bool:
        """False for an unreadable file: only Delete (and New) act on it."""
        entry = self._highlighted_entry()
        return entry is None or entry.error is None

    def focus_list(self) -> None:
        lst = self.query_one("#settings-theme-list", OptionList)
        if self.highlighted_id is not None:
            lst.focus()

    # -- events --------------------------------------------------------
    @on(Input.Changed, "#settings-theme-filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._render_list(self.highlighted_id)

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

    @on(Button.Pressed, "#settings-theme-revert")
    def _revert_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._revert is None:
            return
        change, self._revert = self._revert, None
        try:
            restored = revert_theme(self.app, change)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Theme revert failed: {exc}")
            self.app.notify(f"Could not revert the theme: {exc}", severity="error")
        else:
            if not restored:
                self.app.notify("Reverted the theme; the launch default was not restored", severity="warning")
        self._sync_revert_chip()
        self.refresh_catalog()

    # -- actions -------------------------------------------------------
    def use_highlighted(self) -> None:
        self._switch(persist=True)

    def try_highlighted(self) -> None:
        self._switch(persist=False)

    def request_edit(self, mode: Literal["clone", "new", "edit"]) -> None:
        if self.highlighted_id is None:
            return
        if mode != "new" and not self._highlighted_readable():
            return
        if mode == "edit" and not self._can_manage_files():
            return
        self.post_message(self.EditRequested(self.highlighted_id, mode))

    def request_rename(self) -> None:
        if self._can_manage_files() and self._highlighted_readable():
            self.post_message(self.RenameRequested(self.highlighted_id))

    def request_delete(self) -> None:
        if self._can_manage_files():
            self.post_message(self.DeleteRequested(self.highlighted_id))

    def request_export(self) -> None:
        if self._can_manage_files() and self._highlighted_readable():
            self.post_message(self.ExportRequested(self.highlighted_id))

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
            change = use_theme(self.app, theme_id, persist=persist)
        except Exception as exc:  # noqa: BLE001 - stale entry / unregistered theme
            self.app.notify(f"Could not apply {display_name(theme_id)}: {exc}", severity="error")
            self.refresh_catalog()
            return
        self._revert = change if self._revert is None else self._revert.merge(change)
        self._sync_revert_chip()
        if not persist:
            self.app.notify(f"Trying {display_name(theme_id)} for this session", severity="information")
        else:
            message, severity = use_theme_toast(theme_id, change)
            self.app.notify(message, severity=severity)
        self.refresh_catalog(highlight=theme_id)


class ThemePane(ContentSwitcher):
    """Picker first; the editor swaps in for Clone/New/Edit (spec §4, D3).

    The picker lists and changes saved theme files only through the editor's
    backup-scoped file API. Rename bubbles to the screen, which owns the
    name prompt.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(initial="settings-theme-picker", **kwargs)

    def compose(self) -> ComposeResult:
        # The editor is composed first so the picker's first refresh_catalog
        # (its on_mount) can reach it through _editor_names.
        with Vertical(id="settings-theme-editor-view"):
            with Horizontal(classes="settings-action-row"):
                yield Button("Back to themes", id="settings-theme-back", classes="theme-editor-action")
            yield SettingsThemeEditor(id="settings-theme-editor")
        yield ThemePicker(
            list_user_names=self._editor_names,
            list_unreadable=lambda: self._editor().user_theme_listing()[1],
            id="settings-theme-picker",
        )

    def _editor(self) -> SettingsThemeEditor:
        return self.query_one(SettingsThemeEditor)

    def _editor_names(self) -> set[str]:
        return self._editor().list_user_theme_names()

    def show_picker(self) -> None:
        self.current = "settings-theme-picker"
        picker = self.query_one(ThemePicker)
        picker.refresh_catalog()
        picker.focus_list()

    def open_editor(self, theme_id: str, mode: Literal["clone", "new", "edit"]) -> None:
        editor = self._editor()
        picker = self.query_one(ThemePicker)
        entry = next((e for e in picker.entries if e.id == theme_id), None)
        if entry is not None and entry.origin == "yours":
            editor.load_user_theme(theme_id)
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
        self._editor().request_delete(event.theme_id)

    @on(ThemePicker.ExportRequested)
    def _export_requested(self, event: ThemePicker.ExportRequested) -> None:
        event.stop()
        self._editor().export_theme(event.theme_id)

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
        self.query_one(ThemePicker).refresh_catalog()
