# tldw_chatbook/Widgets/settings_theme_picker.py
"""Settings ▸ Theme picker and the picker/editor pane (TASK-32948)."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from loguru import logger
from rich.style import Style
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from ..config import get_user_themes_dir
from ..css.Themes.theme_catalog import (
    ThemeChange,
    ThemeEntry,
    build_catalog,
    current_launch_default,
    display_name,
    revert_theme,
    use_theme,
    user_theme_names,
)
from .theme_preview import ThemePreview

_GROUP_TITLES = {"yours": "YOUR THEMES", "shipped": "SHIPPED", "textual": "TEXTUAL"}


def _row(entry: ThemeEntry) -> Text:
    text = Text(f"{entry.display_name}  ")
    for colour in entry.strip:
        try:
            style = Style(color=colour)
        except Exception:  # noqa: BLE001 - an 8-digit alpha hex or an ANSI colour
            # name (e.g. deep_dive_cyberspace's "#FF33AACC", ansi-dark/-light's
            # "ansi_default") is a valid textual.color.Color but not a valid
            # rich.color.Color; an unstyled swatch must not break the whole row.
            style = None
        text.append("▮", style)
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

    def action_try_theme(self) -> None:
        self.query_ancestor(ThemePicker).try_highlighted()

    def action_clone_theme(self) -> None:
        self.query_ancestor(ThemePicker).request_edit("clone")

    def action_new_theme(self) -> None:
        self.query_ancestor(ThemePicker).request_edit("new")


class ThemePicker(Vertical):
    class EditRequested(Message):
        def __init__(self, theme_id: str, mode: Literal["clone", "new"]) -> None:
            self.theme_id = theme_id
            self.mode = mode
            super().__init__()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.entries: list[ThemeEntry] = []
        self.highlighted_id: str | None = None
        self._revert: ThemeChange | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="settings-theme-picker-columns"):
            with Vertical(id="settings-theme-list-column"):
                yield ThemeFilterInput(placeholder="Filter themes", id="settings-theme-filter")
                yield ThemeOptionList(id="settings-theme-list")
                yield Static("", id="settings-theme-empty", classes="settings-help-copy", markup=False)
            with Vertical(id="settings-theme-card-column"):
                yield Static("", id="settings-theme-card-title", classes="destination-section", markup=False)
                yield ThemePreview("settings-theme-picker-preview", id="settings-theme-picker-preview")
                with Horizontal(classes="settings-action-row"):
                    yield Button("Use this theme", id="settings-theme-use", variant="primary", classes="theme-editor-action")
                    yield Button("Try", id="settings-theme-try", classes="theme-editor-action")
                    yield Button("Clone", id="settings-theme-clone", classes="theme-editor-action")
                    yield Button("New", id="settings-theme-new", classes="theme-editor-action")
                    yield Button("Revert", id="settings-theme-revert", classes="theme-editor-action")

    def on_mount(self) -> None:
        self.query_one("#settings-theme-revert").display = False
        self.query_one("#settings-theme-empty").display = False
        self.app.theme_changed_signal.subscribe(self, lambda _theme: self.refresh_catalog())
        self.refresh_catalog(highlight=str(self.app.theme))

    # -- catalog -------------------------------------------------------
    def refresh_catalog(self, highlight: str | None = None) -> None:
        self.entries = build_catalog(
            self.app.available_themes,
            user_theme_names(get_user_themes_dir()),
            str(self.app.theme),
            current_launch_default(),
        )
        self._render_list(highlight or self.highlighted_id)

    def _render_list(self, highlight: str | None) -> None:
        query = self.query_one("#settings-theme-filter", Input).value.strip().casefold()
        shown = [e for e in self.entries if not query or query in e.display_name.casefold() or query in e.id.casefold()]
        options: list[Option] = []
        for origin, title in _GROUP_TITLES.items():
            group = [e for e in shown if e.origin == origin]
            if not group:
                continue
            header = f"{title} ({len(group)})" if query else title
            options.append(Option(header, disabled=True))
            options.extend(Option(_row(e), id=e.id) for e in group)
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
        entry = next((e for e in self.entries if e.id == theme_id), None)
        title = self.query_one("#settings-theme-card-title", Static)
        for button_id in ("#settings-theme-use", "#settings-theme-try", "#settings-theme-clone"):
            self.query_one(button_id, Button).disabled = entry is None
        if entry is None:
            title.update("")
            return
        tone = "dark" if entry.dark else "light"
        title.update(f"{entry.display_name}  ·  {tone} · {entry.origin}")
        self.query_one(ThemePreview).paint(dict(entry.colours))

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

    @on(Button.Pressed, "#settings-theme-clone")
    def _clone_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_edit("clone")

    @on(Button.Pressed, "#settings-theme-new")
    def _new_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.request_edit("new")

    @on(Button.Pressed, "#settings-theme-revert")
    def _revert_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._revert is None:
            return
        change, self._revert = self._revert, None
        try:
            revert_theme(self.app, change)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Theme revert failed: {exc}")
            self.app.notify(f"Could not revert the theme: {exc}", severity="error")
        self.query_one("#settings-theme-revert").display = False
        self.refresh_catalog()

    # -- actions -------------------------------------------------------
    def use_highlighted(self) -> None:
        self._switch(persist=True)

    def try_highlighted(self) -> None:
        self._switch(persist=False)

    def request_edit(self, mode: Literal["clone", "new"]) -> None:
        if self.highlighted_id is not None:
            self.post_message(self.EditRequested(self.highlighted_id, mode))

    def _switch(self, *, persist: bool) -> None:
        theme_id = self.highlighted_id
        if theme_id is None:
            return
        try:
            change = use_theme(self.app, theme_id, persist=persist)
        except Exception as exc:  # noqa: BLE001 - stale entry / unregistered theme
            self.app.notify(f"Could not apply {display_name(theme_id)}: {exc}", severity="error")
            self.refresh_catalog()
            return
        self._revert = change if self._revert is None else self._revert.merge(change)
        was = display_name(self._revert.previous_active)
        revert = self.query_one("#settings-theme-revert", Button)
        revert.label = f"Revert to {was}"
        revert.display = True
        if not persist:
            self.app.notify(f"Trying {display_name(theme_id)} for this session", severity="information")
        elif change.persisted:
            self.app.notify(f"{display_name(theme_id)} is now your theme (was: {display_name(change.previous_active)})", severity="information")
        else:
            self.app.notify(f"{display_name(theme_id)} applied; the launch default was not saved", severity="warning")
        self.refresh_catalog(highlight=theme_id)
