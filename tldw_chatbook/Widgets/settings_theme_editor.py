"""Settings-native theme editor widget."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any, ClassVar, Literal

import toml
from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.color import Color
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.events import Click, Key
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.theme import BUILTIN_THEMES, Theme
from textual.widgets import Button, Checkbox, Input, Select, Static

from ..Backup_Recovery import raw_participants as raw
from ..Backup_Recovery.bootstrap import RecoveryRequired
from ..Backup_Recovery.settings_file_participants import NOT_REGULAR
from ..css.Themes.theme_catalog import (
    CACHE_REFRESH_FAILED,
    UNREADABLE_ID_PREFIX,
    display_name,
    is_catalog_theme,
)
from ..css.Themes.themes import (
    ALL_THEMES,
    PREVIEW_CHARS,
    RESERVED_NAME_RULE,
    create_theme_from_dict,
    is_hex_colour,
    is_reserved_theme_name,
    pinned_text_hues,
    printable,
    sanitize_theme_variables,
    theme_file_dark,
    theme_from_file_data,
)
from ..Utils.input_validation import escape_markup
from ..Utils.path_validation import validate_browsing_path, validate_filename
from .confirmation_dialog import ConfirmationDialog
from .theme_preview import ThemePreview

#: TASK-32942: shown in the picker while backup/recovery holds the theme files.
THEMES_UNAVAILABLE_LABEL = "Theme files unavailable while backup/recovery is in progress"

#: TASK-32948 PR 3: the largest theme file Import reads.
IMPORT_MAX_BYTES = 64 * 1024


ThemeLeaveChoice = Literal["save", "discard", "cancel"]


class ThemeLeaveModal(ModalScreen[ThemeLeaveChoice]):
    """Ask before leaving Theme with unsaved edits (TASK-32941).

    Mirrors the Speech & TTS leave guard (``_GlobalSpeechTTSLeaveModal``).
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Stay", show=False)]

    def compose(self) -> ComposeResult:
        """Build the prompt: a title, one line of copy, Stay/Discard/Save.

        Returns:
            The modal's widgets.
        """
        with Vertical(id="settings-theme-leave-modal", classes="settings-rag-profile-modal"):
            yield Static("Unsaved theme changes", classes="destination-section")
            yield Static(
                "Save this theme before leaving, or discard the changes?",
                classes="settings-detail-row",
                markup=False,
            )
            with Horizontal(classes="settings-action-row"):
                yield Button("Stay", id="settings-theme-leave-stay")
                yield Button("Discard", id="settings-theme-leave-discard")
                yield Button("Save", id="settings-theme-leave-save", variant="primary")

    def action_cancel(self) -> None:
        """Escape: dismiss with ``"cancel"`` (Stay), keeping the edits."""
        self.dismiss("cancel")

    @on(Button.Pressed, "#settings-theme-leave-stay")
    def _stay(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("cancel")

    @on(Button.Pressed, "#settings-theme-leave-discard")
    def _discard(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("discard")

    @on(Button.Pressed, "#settings-theme-leave-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("save")

_INVALID_COLOUR_TEXT = "Invalid — use #RRGGBB"


class SettingsThemeEditor(Vertical):
    """Theme editor styled for the Settings screen."""

    class ThemeModifiedStatus(Message):
        """Message sent when the theme editor's modified state changes."""

        def __init__(self, is_modified: bool) -> None:
            self.is_modified = is_modified
            super().__init__()

    class LaunchDefaultChanged(Message):
        """Publish a saved launch preference to the surrounding Settings draft."""

        def __init__(self, theme_name: str) -> None:
            self.theme_name = theme_name
            super().__init__()

    class ThemesChanged(Message):
        """The saved theme files changed (delete, rename, save, import);
        rebuild lists, selecting ``highlight`` when given."""

        def __init__(self, highlight: str | None = None) -> None:
            self.highlight = highlight
            super().__init__()

    class Exported(Message):
        """A saved theme file was exported to ``path`` (spec §7)."""

        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    class Saved(Message):
        """A Save or Save as wrote ``theme_name``; the pane returns to the picker."""

        def __init__(self, theme_name: str) -> None:
            self.theme_name = theme_name
            super().__init__()

    class SaveAsRequested(Message):
        """Save as was pressed; the screen owns the name prompt."""

        def __init__(self, current_name: str) -> None:
            self.current_name = current_name
            super().__init__()

    current_theme_name = reactive("textual-dark")
    current_theme_data: reactive[dict[str, str]] = reactive(dict, layout=False)
    is_dark_theme = reactive(True)
    # init=False: the init-time watch call would post ThemeModifiedStatus(False)
    # on every mount, and SettingsScreen recomposes on that reactive -- paired
    # with the load-time True post below, each recompose mounted an editor
    # whose posts forced the next recompose (an event-loop-starving storm
    # that froze the whole app while Theme was open).
    is_modified = reactive(False, init=False)

    BASE_COLORS = [
        "primary",
        "secondary",
        "accent",
        "background",
        "surface",
        "panel",
        "foreground",
        "success",
        "warning",
        "error",
    ]

    COLOR_PRESETS = {
        "Blues": ["#0099FF", "#006FB3", "#004D80", "#003366", "#002244"],
        "Greens": ["#00CC66", "#009944", "#006633", "#004422", "#002211"],
        "Reds": ["#FF3333", "#CC0000", "#990000", "#660000", "#330000"],
        "Purples": ["#9966FF", "#7744DD", "#5522BB", "#330099", "#220066"],
        "Grays": ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333"],
        "Material": ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"],
        "Pastels": ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#FFDFBA"],
        "Dark": ["#1A1A1A", "#2D2D2D", "#404040", "#525252", "#656565"],
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from ..config import get_user_themes_dir

        self.custom_themes_path = get_user_themes_dir()
        try:
            with raw._scope(self, "theme_directory", writing=True) as operation:
                raw._mkdirs(operation)
        except RecoveryRequired:
            # TASK-32942: a backup/recovery pause must not crash Settings;
            # the picker reports the files as unavailable instead.
            pass
        self.color_inputs: dict[str, Input] = {}
        self.color_swatches: dict[str, Static] = {}
        # TASK-31258: the user theme file the palette was loaded from (or last
        # saved to); re-saving it is an update, saving over another is an
        # overwrite that asks first.
        self._loaded_user_theme: str | None = None
        # TASK-32940: the loaded theme's extra colour variables (shipped AA
        # fixes such as text-muted), carried through Apply/Save/Export.
        self._theme_variables: dict[str, str] = {}
        # Review #2: the base colours + dark flag those variables belong to;
        # once the palette diverges they are not carried (Textual derives).
        self._variables_palette: tuple[dict[str, str], bool] | None = None
        # The catalog theme currently loaded, so an unmodified Apply can
        # select it by name instead of registering a lossy copy.
        self._loaded_catalog_theme: str | None = None
        # What the header says the editor was opened from (ThemePane sets it).
        self._editing_context: tuple[str, str] | None = None

    def compose(self) -> ComposeResult:
        """Compose the theme editor widget.

        Yields:
            ComposeResult: The theme editor UI sections.
        """
        # Title is rendered by SettingsScreen._render_detail_pane()
        with Vertical(id="settings-theme-card", classes="settings-focus-card"):
            # TASK-31256: Actions before the palette and presets so Apply/Save
            # are reachable without Tabbing through 10 inputs and 40 swatches.
            yield Static("", id="settings-theme-editor-header", markup=False)
            yield from self._compose_theme_section()
            yield from self._compose_actions_section()
            yield from self._compose_palette_section()
            yield from self._compose_preview_section()

    def _compose_theme_section(self) -> ComposeResult:
        # TASK-32948: the theme list and New/Clone/Delete/Export moved to
        # the picker; the editor keeps only the theme's own identity.
        yield Static("Theme", classes="destination-section")
        with Horizontal(classes="settings-input-row"):
            yield Static("Name", classes="settings-input-label")
            yield Input(
                placeholder="Theme name",
                id="settings-theme-name",
                classes="settings-compact-input",
                disabled=True,
            )
        with Horizontal(classes="settings-input-row"):
            yield Static("Dark theme", classes="settings-input-label")
            # TASK-31254: the label carries the state in text ("On"/"Off") so
            # colour is never the only carrier.
            yield Checkbox("On", value=True, id="settings-theme-dark-mode")

    def _compose_palette_section(self) -> ComposeResult:
        yield Static("Color Palette", classes="destination-section")
        for color_name in self.BASE_COLORS:
            with Horizontal(classes="settings-input-row"):
                yield Static(color_name.title(), classes="settings-input-label")
                yield Input(
                    placeholder="#RRGGBB",
                    id=f"settings-theme-color-{color_name}",
                    classes="settings-compact-input",
                    max_length=7,
                )
                yield Static("", id=f"settings-theme-swatch-{color_name}", classes="color-swatch")
        yield Static("Color Presets", classes="destination-section")
        # TASK-31254/31256: the preset target is an explicit, visible choice.
        # It used to follow keyboard focus, so Tabbing through the colour
        # inputs to reach a swatch silently moved the target to Error.
        with Horizontal(classes="settings-input-row settings-select-row"):
            yield Static("Presets fill", classes="settings-input-label")
            yield Select(
                [(name.title(), name) for name in self.BASE_COLORS],
                value="primary",
                allow_blank=False,
                id="settings-theme-preset-target",
                classes="settings-compact-select",
            )
        # task-1369: swatches are focusable and apply on Enter/Space, not
        # just on mouse click.
        yield Static(
            "Pick the colour above, then click a swatch or focus it and press Enter or Space.",
            classes="settings-help-copy",
        )
        for palette_name, colors in self.COLOR_PRESETS.items():
            with Horizontal(classes="settings-input-row preset-row"):
                yield Static(palette_name, classes="settings-input-label")
                for idx, color in enumerate(colors):
                    swatch = Static(
                        "",
                        id=f"settings-theme-preset-{palette_name}-{idx}",
                        classes="color-preset-swatch",
                    )
                    swatch.add_class(f"theme-preset-{palette_name.lower()}-{idx}")
                    swatch.can_focus = True
                    swatch.tooltip = f"Apply {color} to the selected color"
                    yield swatch

    def _compose_actions_section(self) -> ComposeResult:
        yield Static("Actions", classes="destination-section")
        # task-1369: Apply re-themes the whole app instantly; say so with the
        # Settings screen's instant-apply phrasing
        # (INSTANT_APPLY_BEHAVIOR_COPY in UI/Screens/settings_screen.py -- the
        # widget must not import the screen).
        yield Static(
            "Try previews this palette now - no Save needed; Save stores the theme file.",
            id="settings-theme-apply-hint",
            classes="settings-help-copy",
        )
        # `theme-editor-action` keys the compact-mode width rule to these
        # buttons only (the CSS fast-path ratchet forbids a bare `Button`
        # subject, TASK-31279). The id stays `apply` (TASK-32948: "Try").
        with Horizontal(classes="settings-action-row"):
            yield Button("Try", id="settings-theme-apply", variant="primary", classes="theme-editor-action")
            yield Button("Save", id="settings-theme-save", variant="success", classes="theme-editor-action")
            yield Button("Save as…", id="settings-theme-save-as", classes="theme-editor-action")
            yield Button("Reset", id="settings-theme-reset", variant="warning", classes="theme-editor-action")
            yield Button(
                "Generate from Primary",
                id="settings-theme-generate",
                classes="theme-editor-action",
            )

    def _compose_preview_section(self) -> ComposeResult:
        yield Static("Live Preview", classes="destination-section")
        # TASK-31259: painted from the palette being edited (see
        # _refresh_preview), so it follows every keystroke, not just Apply.
        yield ThemePreview("settings-theme-preview", id="settings-theme-preview")

    def on_mount(self) -> None:
        """Initialize after composed descendants are mounted."""
        self.call_after_refresh(self._initialize_editor)

    def _initialize_editor(self) -> None:
        """Bind composed controls and load the active theme."""
        try:
            color_inputs = {
                color_name: self.query_one(f"#settings-theme-color-{color_name}", Input)
                for color_name in self.BASE_COLORS
            }
            color_swatches = {
                color_name: self.query_one(
                    f"#settings-theme-swatch-{color_name}", Static
                )
                for color_name in self.BASE_COLORS
            }
            self.color_inputs = color_inputs
            self.color_swatches = color_swatches
            app_theme = str(self.app.theme)
            self.load_theme(app_theme)
            if app_theme.startswith("custom_"):
                # Apply registers the working palette under a custom_ prefix
                # so it never clobbers a shipped registration; the editor shows
                # the user-facing name and keeps it editable (TASK-31252).
                display_name = app_theme[len("custom_") :]
                self.current_theme_name = display_name
                name_input = self.query_one("#settings-theme-name", Input)
                name_input.value = display_name
                name_input.disabled = False
        except QueryError:
            # Settings can recompose while this callback is queued. A stale,
            # detached editor must not fail the replacement screen.
            self.color_inputs.clear()
            self.color_swatches.clear()
            return

    def set_editing_context(self, source: str, mode: Literal["clone", "new", "edit"]) -> None:
        """Say in the header what the editor was opened from (TASK-32948)."""
        self._editing_context = (source, mode)
        self._render_header()

    def watch_current_theme_name(self) -> None:
        self._render_header()

    def _render_header(self) -> None:
        if self._editing_context is None:
            return
        source, mode = self._editing_context
        suffix = {
            "clone": f"copy of {display_name(source)}",
            "new": "new",
            "edit": "saved theme",
        }[mode]
        try:
            header = self.query_one("#settings-theme-editor-header", Static)
        except QueryError:
            return
        header.update(f"Editing {self.current_theme_name} · {suffix}")

    def watch_is_modified(self, is_modified: bool) -> None:
        """Notify parent screen when modified state changes."""
        self.post_message(SettingsThemeEditor.ThemeModifiedStatus(is_modified))

    def _scan_theme_files(self) -> tuple[dict[str, Path], dict[str, tuple[Path, str]]]:
        """One pass over the themes directory: readable and unreadable files.

        Returns ``(name -> path, stem -> (path, short error))``. A file is
        unreadable when it is not TOML or ``theme_from_file_data`` rejects it
        -- the same test the startup loader applies (``load_user_themes``) --
        when it is not a regular single-link file (never opened), or when a
        later file claims the same name: startup registers the last one, so
        the earlier is listed as a duplicate that can be seen and deleted.

        Raises:
            RecoveryRequired: backup/recovery holds the theme files, including
                a pause that starts mid-scan (R15: never a partial map).
        """
        files: dict[str, Path] = {}
        unreadable: dict[str, tuple[Path, str]] = {}
        with raw._scope(self, "theme_directory") as operation:
            state = raw._check(operation)
            for theme_file in sorted(state.rejected_files):
                # Qodo 4109320408: refused by the backup layer, never opened.
                unreadable[theme_file.stem] = (theme_file, state.rejected_files[theme_file])
            for theme_file in sorted(state.observed_files):
                theme_data: Any = None
                try:
                    with raw._file(operation, theme_file, "r") as f:
                        theme_data = toml.load(f)
                    # Qodo 4107495864: key by the name startup registers.
                    name = theme_from_file_data(theme_data, theme_file.stem, theme_file.name).name
                except RecoveryRequired:
                    raise
                except Exception as e:
                    error = self._theme_file_error(e, theme_data)
                    # File name only: the themes directory is a user path.
                    # WARNING: the picker re-lists on every theme change.
                    logger.warning(f"Unreadable user theme {printable(theme_file.name)}: {error}")
                    unreadable[theme_file.stem] = (theme_file, error)
                    continue
                if name in files:
                    # Qodo 4107495872: file order is startup's order; the
                    # last file registers the name, the earlier is shadowed.
                    earlier = files[name]
                    unreadable[earlier.stem] = (earlier, f"duplicate of '{printable(name)}'")
                files[name] = theme_file
        return files, unreadable

    def user_theme_listing(self) -> tuple[dict[str, Path], dict[str, str]]:
        """``(readable name -> path, unreadable stem -> short error)``.

        Raises:
            RecoveryRequired: see ``_scan_theme_files``.
        """
        files, unreadable = self._scan_theme_files()
        return files, {stem: error for stem, (_path, error) in unreadable.items()}

    def _user_theme_files(self) -> dict[str, Path]:
        """Each readable saved theme's ``[theme].name`` (else stem) -> its file.

        R12: the one identity every theme-file operation resolves through, so
        ``a.toml`` holding ``name = "b"`` is listed, deleted, renamed and
        exported as ``b``.

        Raises:
            RecoveryRequired: see ``_scan_theme_files``.
        """
        return self._scan_theme_files()[0]

    @classmethod
    def _theme_file_error(cls, exc: Exception, data: Any) -> str:
        """A short, path-free, printable (R39) reason a theme file can't be read."""
        if data is None and isinstance(exc, (toml.TomlDecodeError, UnicodeDecodeError)):
            return "not valid TOML"
        message = str(exc)
        if isinstance(exc, TypeError) and "'primary'" in message and "missing" in message:
            colors = data.get("colors") if isinstance(data, dict) else None
            if isinstance(colors, dict) and "primary" in colors:
                return "invalid colour 'primary'"
            return "missing [colors].primary"
        return cls._failure_reason(exc)

    def list_user_theme_names(self) -> set[str]:
        """The ``[theme].name`` (else the stem) of each readable saved theme.

        Raises:
            RecoveryRequired: backup/recovery holds the theme files; the
                caller decides how to show that.
        """
        return set(self._user_theme_files())

    @staticmethod
    def _failure_reason(exc: Exception) -> str:
        """R16: an error's reason without the file path OSError carries.

        R39: printable only -- a parse error can quote file bytes.
        """
        if isinstance(exc, OSError):
            return exc.strerror or type(exc).__name__
        return printable(exc)

    def load_theme(self, theme_name: str) -> None:
        """Load a theme for editing."""
        self.current_theme_name = theme_name
        self._loaded_user_theme = None
        self._loaded_catalog_theme = None

        name_input = self.query_one("#settings-theme-name", Input)
        name_input.value = theme_name
        name_input.disabled = theme_name in ["textual-dark", "textual-light"]

        # TASK-31255: loading is read-only for the running app -- Apply is the
        # only action that changes app.theme. Colours come from the registered
        # Theme object (built-in, shipped, or saved), not a hardcoded table.
        theme = self.app.available_themes.get(theme_name) or next(
            (t for t in ALL_THEMES if getattr(t, "name", None) == theme_name),
            None,
        )
        if theme is None:
            logger.warning(
                f"Theme editor: unknown theme '{theme_name}', keeping current palette"
            )
        else:
            self.current_theme_data = self._extract_theme_colors(theme)
            self.is_dark_theme = bool(getattr(theme, "dark", True))
            # The text-* tints the AA guard generated are re-derived per
            # palette by create_theme_from_dict, so only the theme's own
            # (hand-set) extras carry (review #3).
            pinned = pinned_text_hues(theme)
            self._theme_variables = {
                key: value
                for key, value in (theme.variables or {}).items()
                if key not in pinned
            }
            self._snapshot_variables_palette()
            self._loaded_catalog_theme = theme_name

        self._update_color_inputs()
        self._update_dark_mode_checkbox()
        self.is_modified = False

    def load_user_theme(self, theme_name: str) -> bool:
        """Load a saved theme file into the editor.

        Args:
            theme_name: The saved theme's ``[theme].name``.

        Returns:
            True once loaded; False (with a notice) when the name is invalid,
            the files are paused, or no readable file holds it any more --
            the editor then keeps its previous palette (Qodo 4104047302).
        """
        try:
            validate_filename(theme_name)
        except ValueError as exc:
            self.app.notify(f"Invalid theme name: {escape_markup(exc)}", severity="error")
            return False

        # R12: resolve by [theme].name, so a.toml holding name "b" loads as b.
        try:
            theme_path = self._user_theme_files().get(theme_name)
        except RecoveryRequired:
            self.app.notify(THEMES_UNAVAILABLE_LABEL, severity="warning")
            return False
        if theme_path is None:
            self.app.notify(
                f"No saved custom theme named '{escape_markup(printable(theme_name))}'",
                severity="warning",
            )
            return False
        if theme_path is not None:
            try:
                with raw._scope(self, "theme_file", selected_read=theme_path) as operation:
                    with raw._file(operation, theme_path, "r") as f:
                        theme_data = toml.load(f)
                # Follow-up (R41): validate what was read, not the earlier
                # scan's copy -- the file can change in between.
                theme_from_file_data(theme_data, theme_path.stem, theme_path.name)

                self.current_theme_name = theme_name
                self.current_theme_data = dict(theme_data.get("colors", {}))
                self._theme_variables = sanitize_theme_variables(
                    theme_data.get("variables", {}) or {}, theme_path.name
                )
                self._loaded_catalog_theme = None
                self.is_dark_theme = theme_file_dark(theme_data.get("theme", {}).get("dark", True))
                self._snapshot_variables_palette()

                name_input = self.query_one("#settings-theme-name", Input)
                name_input.value = theme_name
                name_input.disabled = False

                self._update_color_inputs()
                self._update_dark_mode_checkbox()
                self.is_modified = False
                self._loaded_user_theme = theme_name

            except Exception as e:
                logger.error(f"Failed to load user theme {printable(theme_name)}: {self._failure_reason(e)}")
                self.app.notify(f"Failed to load theme: {escape_markup(self._failure_reason(e))}", severity="error")
                return False
        return True

    def _extract_theme_colors(self, theme: Theme) -> dict[str, str]:
        """Resolve a Theme's ten base colours as uppercase hex.

        Built-in Themes leave background/surface/panel/foreground unset and
        derive them; ``to_color_system().generate()`` resolves every base
        colour the same way the app does at runtime (TASK-31255).
        """
        try:
            generated = theme.to_color_system().generate()
        except Exception as exc:  # noqa: BLE001 - a malformed theme must not break the editor
            logger.warning(f"Theme editor: could not resolve colours for {theme.name!r}: {exc}")
            generated = {}
        colors: dict[str, str] = {}
        for our_name in self.BASE_COLORS:
            # A colour the Theme sets explicitly comes back byte-exact (the
            # resolved system round-trips through float maths: a saved
            # "#FFD700" came back "#FED700" live); only unset ones resolve.
            raw = getattr(theme, our_name, None)
            if isinstance(raw, Color):
                value: str | None = raw.hex
            elif raw:
                value = str(raw)
            else:
                value = generated.get(our_name)
            try:
                colors[our_name] = Color.parse(str(value)).hex.upper() if value else "#808080"
            except Exception:  # noqa: BLE001
                colors[our_name] = "#808080"
        return colors

    def _update_color_inputs(self) -> None:
        """Update color input fields with current theme data."""
        for color_name, color_value in self.current_theme_data.items():
            if color_name in self.color_inputs:
                self.color_inputs[color_name].value = color_value
                self._update_color_swatch(color_name, color_value)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        """Paint the preview rows from the palette being edited (TASK-31259)."""
        try:
            self.query_one("#settings-theme-preview", ThemePreview).paint(self.current_theme_data)
        except QueryError:
            return

    def _update_dark_mode_checkbox(self) -> None:
        """Update the dark mode checkbox."""
        checkbox = self.query_one("#settings-theme-dark-mode", Checkbox)
        checkbox.value = self.is_dark_theme
        checkbox.label = "On" if self.is_dark_theme else "Off"

    def _update_color_swatch(self, color_name: str, color_value: str) -> None:
        """Update a color swatch preview."""
        if color_name in self.color_swatches:
            try:
                parsed_color = Color.parse(color_value)
                swatch = self.color_swatches[color_name]
                swatch.remove_class("theme-preview-invalid")
                swatch.set_class(parsed_color.brightness > 0.5, "theme-preview-dark-ink")
                swatch.set_class(parsed_color.brightness <= 0.5, "theme-preview-light-ink")
                # ds-runtime: preview the user-entered theme color before applying it.
                swatch.set_styles(background=color_value)
                swatch.update(color_value.upper())
            except Exception:
                swatch = self.color_swatches[color_name]
                swatch.set_styles(background=None)
                swatch.remove_class("theme-preview-dark-ink", "theme-preview-light-ink")
                swatch.add_class("theme-preview-invalid")
                swatch.update(_INVALID_COLOUR_TEXT)

    def _validate_color_input(self, color_value: str) -> bool:
        """Validate a color input value (``#RGB``/``#RRGGBB``/``#RRGGBBAA``, R41)."""
        return is_hex_colour(color_value)

    def _preset_target(self) -> str:
        """The colour a preset swatch fills: the visible Select's value."""
        try:
            value = self.query_one("#settings-theme-preset-target", Select).value
        except QueryError:
            value = "primary"
        return value if value in self.color_inputs else "primary"

    @on(Input.Changed)
    def on_color_input_changed(self, event: Input.Changed) -> None:
        """Handle color input changes.

        Args:
            event: The Input.Changed event from a color field; events whose
                value matches the stored theme data are programmatic reloads
                and do not mark the editor modified.
        """
        if event.input.id and event.input.id.startswith("settings-theme-color-"):
            color_name = event.input.id[len("settings-theme-color-") :]
            color_value = event.value.strip()

            if color_value:
                if self._validate_color_input(color_value):
                    self._update_color_swatch(color_name, color_value)
                    # task-1338: programmatic loads (_update_color_inputs during
                    # load_theme) deliver Input.Changed asynchronously, AFTER
                    # load_theme reset is_modified -- treat a value identical
                    # to the loaded theme data as not-a-modification, otherwise
                    # every mount flags modified and the screen's
                    # recompose-on-modified loops forever.
                    if self.current_theme_data.get(color_name) != color_value:
                        self.current_theme_data[color_name] = color_value
                        self.is_modified = True
                        self._refresh_preview()
                    event.input.remove_class("settings-invalid-input")
                else:
                    # TASK-31254: Settings' invalid-input convention (styled in
                    # the bundle) instead of an unstyled class, and the swatch
                    # says so instead of silently turning black.
                    event.input.add_class("settings-invalid-input")
                    if color_name in self.color_swatches:
                        self.color_swatches[color_name].update(_INVALID_COLOUR_TEXT)

    @on(Input.Changed, "#settings-theme-name")
    def on_theme_name_changed(self, event: Input.Changed) -> None:
        """Keep current_theme_name in step with the Name box (TASK-31251).

        Try, Save as and Reset all read ``current_theme_name``;
        without this they acted on the name from the last load. Programmatic
        loads set the box to the name already held, so the equality guard
        makes those echoes no-ops.

        Qodo 4100998919: a real edit of the name marks the editor modified,
        so Back and leaving Theme ask before dropping it. An echo of an
        earlier programmatic set (load then clone queues two) is stale --
        the box already holds a newer value -- and is ignored.
        """
        if event.value != event.input.value:
            return
        name = event.value.strip()
        if name != self.current_theme_name:
            # An emptied box clears the name too (PR #2375 review #6), so no
            # action can fall back to the previously loaded theme.
            self.current_theme_name = name
            self.is_modified = True

    def _require_theme_name(self) -> str | None:
        """Return the current theme name if it is non-empty and file-safe.

        Notifies and returns ``None`` otherwise, so Try and Reset share
        Save's guard instead of trusting a raw value.
        """
        name = self.current_theme_name.strip()
        if not name:
            self.app.notify("Please enter a theme name", severity="warning")
            return None
        try:
            validate_filename(name)
        except ValueError as exc:
            self.app.notify(f"Invalid theme name: {escape_markup(exc)}", severity="warning")
            return None
        return name

    @on(Checkbox.Changed, "#settings-theme-dark-mode")
    def on_dark_mode_changed(self, event: Checkbox.Changed) -> None:
        """Handle dark mode checkbox changes."""
        # task-1338: ignore programmatic syncs from _update_dark_mode_checkbox;
        # only a real change counts as a modification.
        event.checkbox.label = "On" if event.value else "Off"
        if event.value != self.is_dark_theme:
            self.is_dark_theme = event.value
            self.is_modified = True

    @on(Button.Pressed, "#settings-theme-apply")
    def on_apply_theme(self) -> None:
        """Apply the current theme to the app."""
        if self._require_theme_name() is None:
            return
        try:
            if (
                not self.is_modified
                and self._loaded_catalog_theme == self.current_theme_name
            ):
                # TASK-32940: an untouched catalog theme applies as itself.
                theme = self.app.available_themes.get(self.current_theme_name) or next(
                    t for t in ALL_THEMES if t.name == self.current_theme_name
                )
            else:
                theme = create_theme_from_dict(
                    name=f"custom_{self.current_theme_name}",
                    theme_dict=self._theme_dict(),
                )
            self.app.register_theme(theme)
            self.app.theme = theme.name
            self.app.notify(
                f"Theme '{escape_markup(self.current_theme_name)}' applied", severity="information"
            )
        except Exception as e:
            logger.error(f"Failed to apply theme: {e}")
            self.app.notify(f"Failed to apply theme: {escape_markup(e)}", severity="error")

    @on(Button.Pressed, "#settings-theme-save")
    def on_save_theme(self) -> None:
        """Save the current theme under the Name box's name."""
        self._save_under(self.query_one("#settings-theme-name", Input).value.strip())

    @on(Button.Pressed, "#settings-theme-save-as")
    def on_save_as_pressed(self, event: Button.Pressed) -> None:
        """Ask the screen for a new name (it owns the prompt)."""
        event.stop()
        self.post_message(self.SaveAsRequested(self.current_theme_name))

    def save_as(self, name: str) -> None:
        """Write the working palette as a new theme ``name``.

        The file the editor was loaded from is left untouched unless it is
        ``name`` itself. R21: an existing ``name.toml`` always asks first --
        even the loaded theme's own file, since Save as means "a new file".
        """
        self._save_under(name.strip(), always_confirm=True)

    def set_files_available(self, available: bool) -> None:
        """R20 / spec §9: Save and Save as write theme files, so a
        backup/recovery pause disables them with the reason as a tooltip."""
        for button_id in ("#settings-theme-save", "#settings-theme-save-as"):
            button = self.query_one(button_id, Button)
            button.disabled = not available
            button.tooltip = None if available else THEMES_UNAVAILABLE_LABEL

    def _save_under(self, theme_name: str, *, always_confirm: bool = False) -> None:
        """Validate ``theme_name``, confirm an overwrite, then write it."""
        if not theme_name:
            self.app.notify("Please enter a theme name", severity="warning")
            return

        try:
            validate_filename(theme_name)
        except ValueError as exc:
            self.app.notify(f"Invalid theme name: {escape_markup(exc)}", severity="warning")
            return

        if theme_name in ["textual-dark", "textual-light"]:
            self.app.notify("Cannot overwrite built-in themes", severity="warning")
            return
        if is_reserved_theme_name(theme_name):
            self.app.notify(f"Invalid theme name: {RESERVED_NAME_RULE}", severity="warning")
            return

        theme_data = self._theme_file_data(theme_name)
        # R12: a saved theme named ``theme_name`` may live in another file
        # (a.toml holding name "b"); write back to it, never a second file
        # claiming the same name.
        theme_path = self._resolve_write_target(theme_name)
        if theme_path is None:
            return

        # TASK-31258: writing over another saved theme is one keypress from
        # destroying it; re-saving the theme loaded from that very file is an
        # update and needs no dialog (Save as always asks, R21).
        if theme_path.exists() and (always_confirm or self._loaded_user_theme != theme_name):

            async def _confirmed_overwrite() -> None:
                # Qodo 4109320416: the file may have moved while asking.
                target = self._resolve_write_target(theme_name)
                if target is not None:
                    self._write_theme_file(theme_name, target, theme_data)

            self.app.push_screen(
                ConfirmationDialog(
                    title="Overwrite theme",
                    message=(
                        f"A saved theme named '{theme_name}' already exists. "
                        "Replace it with the current palette?"
                    ),
                    confirm_label="Overwrite",
                    cancel_label="Keep existing",
                    confirm_callback=_confirmed_overwrite,
                )
            )
            return

        self._write_theme_file(theme_name, theme_path, theme_data)

    def _resolve_write_target(self, theme_name: str) -> Path | None:
        """The file a write of ``theme_name`` goes to, re-read now (R12).

        The file already holding the name, else ``<name>.toml``; None (with
        the pause notice) while backup/recovery holds the files.
        """
        try:
            theme_path = self._user_theme_files().get(theme_name)
        except RecoveryRequired:
            self.app.notify(THEMES_UNAVAILABLE_LABEL, severity="warning")
            return None
        return theme_path or self.custom_themes_path / f"{theme_name}.toml"

    def _write_theme_file(
        self, theme_name: str, theme_path: Path, theme_data: dict[str, Any]
    ) -> None:
        """Write the theme TOML and register it; post ``Saved`` on success."""
        try:
            self._write_toml(theme_path, theme_data)

            # TASK-31250: register at once so Appearance and the palette can
            # offer the theme without a restart.
            self.app.register_theme(create_theme_from_dict(theme_name, self._theme_dict()))

            self.app.notify(f"Theme '{escape_markup(theme_name)}' saved", severity="success")
            self.is_modified = False
            self._loaded_user_theme = theme_name
            if self.current_theme_name != theme_name:
                # Save as: the editor now edits the new file. Name first, so
                # the Name box's Changed echo is a no-op.
                self.current_theme_name = theme_name
                self.query_one("#settings-theme-name", Input).value = theme_name
            self._reapply_if_active(theme_name)
            self.post_message(self.ThemesChanged())
            self.post_message(self.Saved(theme_name))
        except Exception as e:
            logger.error(f"Failed to save theme: {e}")
            self.app.notify(f"Failed to save theme: {escape_markup(self._failure_reason(e))}", severity="error")

    def _write_toml(self, theme_path: Path, data: dict[str, Any]) -> None:
        """Temp-then-replace ``data`` into ``theme_path`` in the backup scope.

        The one theme-file write path (Save, Rename, Import).

        Raises:
            RecoveryRequired: backup/recovery holds the theme files.
            OSError: the write failed; the old file is untouched.
        """
        with raw._scope(self, "theme_file", writing=True, selected_read=theme_path) as operation:
            temporary = theme_path.with_suffix(theme_path.suffix + ".tmp")
            try:
                with raw._file(operation, temporary, "w") as f:
                    toml.dump(data, f)
                raw._replace(operation, temporary, theme_path)
            finally:
                raw._remove_temporary(operation, temporary)

    def _reapply_if_active(self, name: str) -> None:
        """Spec §6: saving the theme the app is showing re-applies it.

        Done here, not in the pane, so the category-leave Save (which tears
        the pane down before its messages run) re-applies too.
        """
        from ..css.Themes.theme_catalog import use_theme

        active = str(self.app.theme)
        if active not in (name, f"custom_{name}"):
            return
        use_theme(self.app, name, persist=False)
        if active == name:
            # Same name: Textual's theme watcher does not re-fire, so the
            # re-registered palette needs an explicit CSS refresh.
            self.app.refresh_css(animate=False)

    def _snapshot_variables_palette(self) -> None:
        self._variables_palette = (dict(self.current_theme_data), bool(self.is_dark_theme))

    def _carried_variables(self) -> dict[str, str]:
        """The loaded extras, only while the palette is still the loaded one.

        Review #2: a variable such as text-muted was tuned for the loaded
        palette (modern_dark_dracula made light kept #afb8d1 = 1.80:1).
        Compared at emit time so every edit path (inputs, dark flag,
        Generate, presets, New) is covered.
        """
        if self._variables_palette != (
            dict(self.current_theme_data),
            bool(self.is_dark_theme),
        ):
            return {}
        return dict(self._theme_variables)

    def _theme_dict(self) -> dict[str, Any]:
        """The working palette as ``create_theme_from_dict`` input."""
        theme_dict: dict[str, Any] = {**self.current_theme_data, "dark": self.is_dark_theme}
        if variables := self._carried_variables():
            theme_dict["variables"] = variables
        return theme_dict

    def _theme_file_data(self, theme_name: str) -> dict[str, Any]:
        """The working palette as saved/exported TOML (``[variables]`` optional)."""
        data: dict[str, Any] = {
            "theme": {"name": theme_name, "dark": self.is_dark_theme},
            "colors": self.current_theme_data,
        }
        if variables := self._carried_variables():
            data["variables"] = variables
        return data

    def _save_launch_default(self, name: str, success_message: str) -> None:
        from ..config import apply_settings_mutation_to_cli_config

        result = apply_settings_mutation_to_cli_config(
            {"general": {"default_theme": name}}
        )
        if not result.file_replaced:
            self.app.notify(
                "Could not save the launch default; check the config file",
                severity="error",
            )
            return
        self.post_message(self.LaunchDefaultChanged(name))
        if result.caches_reloaded:
            self.app.notify(success_message, severity="success")
        else:
            self.app.notify(
                "Launch default saved, but configuration refresh failed. Reopen Settings to refresh.",
                severity="warning",
            )

    @on(Button.Pressed, "#settings-theme-reset")
    def on_reset_theme(self) -> None:
        """Reset theme to original values (confirms before discarding edits)."""
        # task-1371: Reset throws away unapplied edits; the Settings screen
        # confirms its equivalent discard (revert) per ADR-031 rule 3, so the
        # editor follows the same confirmation rule.
        if self._require_theme_name() is None:
            return
        if not self.is_modified:
            # TASK-31280: nothing to discard -- say so instead of claiming a
            # reset happened (still no confirmation dialog, task-1371).
            self.app.notify("No changes to reset", severity="information")
            return

        async def _confirmed_reset() -> None:
            self._reset_theme()

        self.app.push_screen(
            ConfirmationDialog(
                title="Reset theme",
                message=(
                    f"Discard all unsaved changes to '{self.current_theme_name}'?"
                ),
                confirm_label="Discard changes",
                cancel_label="Keep editing",
                confirm_callback=_confirmed_reset,
            )
        )

    def _reset_theme(self) -> None:
        """Reload original theme values (post-confirmation when modified)."""
        # R28: resolve by [theme].name, so a.toml holding name "b" resets
        # from a.toml, like every other file operation (R12).
        try:
            saved = self.current_theme_name in self._user_theme_files()
        except RecoveryRequired:
            self.app.notify(THEMES_UNAVAILABLE_LABEL, severity="warning")
            return
        if saved:
            if not self.load_user_theme(self.current_theme_name):
                return
        elif is_catalog_theme(self.current_theme_name):
            self.load_theme(self.current_theme_name)
        else:
            # TASK-31251: a renamed, never-saved theme has nothing to go back
            # to; say so instead of claiming a reset happened.
            self.app.notify(
                f"No saved version of '{escape_markup(self.current_theme_name)}' to reset to",
                severity="warning",
            )
            return
        self.app.notify("Theme reset to original values", severity="information")

    def on_new_theme(self) -> None:
        """Create a new theme (confirms before discarding unsaved edits)."""
        # task-1371: starting a new theme replaces the working palette, so it
        # follows the same discard confirmation rule as Reset/revert.
        if not self.is_modified:
            self._new_theme()
            return

        async def _confirmed_new() -> None:
            self._new_theme()

        self.app.push_screen(
            ConfirmationDialog(
                title="New theme",
                message=(
                    f"Discard all unsaved changes to '{self.current_theme_name}' "
                    "and start a new theme?"
                ),
                confirm_label="Discard changes",
                cancel_label="Keep editing",
                confirm_callback=_confirmed_new,
            )
        )

    def _new_theme(self) -> None:
        """Start a new theme from the current palette (post-confirmation when modified).

        TASK-31257: New starts from the current palette, the way Clone
        works; the hardcoded blue set is only the fallback for an editor that
        has nothing loaded yet. The dark flag is kept as well.
        """
        defaults = {
            "primary": "#0099FF",
            "secondary": "#006FB3",
            "accent": "#FFD700",
            "background": "#1E1E1E",
            "surface": "#2C2C2C",
            "panel": "#252525",
            "foreground": "#FFFFFF",
            "success": "#008000",
            "warning": "#FFD700",
            "error": "#FF0000",
        }
        self.current_theme_name = "new_theme"
        self._loaded_user_theme = None
        self.current_theme_data = dict(self.current_theme_data) or defaults

        name_input = self.query_one("#settings-theme-name", Input)
        name_input.value = "new_theme"
        name_input.disabled = False
        name_input.focus()

        self._update_color_inputs()
        self._update_dark_mode_checkbox()
        # TASK-32948: a new theme starts clean; only a real edit marks it, so
        # Back without edits does not prompt.
        self.is_modified = False

        self.app.notify("Creating new theme", severity="information")

    def on_clone_theme(self) -> None:
        """Clone the current theme."""
        new_name = f"{self.current_theme_name}_copy"

        name_input = self.query_one("#settings-theme-name", Input)
        name_input.value = new_name
        name_input.disabled = False
        name_input.focus()

        self.current_theme_name = new_name
        self._loaded_user_theme = None
        self.is_modified = False  # TASK-32948: clean until the first real edit

        self.app.notify(f"Cloned theme as '{escape_markup(new_name)}'", severity="information")

    def request_delete(self, name: str) -> None:
        """Confirm, then delete the saved theme ``name`` (see ``_delete_user_theme``)."""
        built_in_names = set(BUILTIN_THEMES)
        shipped_names = {t.name for t in ALL_THEMES if hasattr(t, "name")}
        try:
            files, unreadable = self._scan_theme_files()
        except RecoveryRequired:
            self.app.notify(THEMES_UNAVAILABLE_LABEL, severity="warning")
            return
        # PR 3 / R40(a): an unreadable file is listed (and deletable) under
        # its own id, so a broken nord.toml can't hide behind shipped nord.
        theme_path = files.get(name)
        if theme_path is None and name.startswith(UNREADABLE_ID_PREFIX):
            theme_path, reason = unreadable.get(name.removeprefix(UNREADABLE_ID_PREFIX), (None, ""))
            if theme_path is not None and reason == NOT_REGULAR:
                # The backup layer never writes through a link; say so rather
                # than fail the unlink with its internal reason code.
                self.app.notify(
                    f"'{escape_markup(printable(theme_path.name))}' is a link, not a regular file; "
                    "remove it outside the app",
                    severity="warning",
                )
                return

        # File existence decides: anything saved in the user themes directory
        # is a user theme and deletable, even when its name shadows a shipped
        # catalog theme. The built-in/shipped guard only applies when no user
        # file exists for the name.
        if theme_path is None:
            if name in built_in_names:
                self.app.notify(
                    f"'{escape_markup(name)}' is a built-in theme and cannot be deleted",
                    severity="warning",
                )
            elif name in shipped_names:
                self.app.notify(
                    f"'{escape_markup(name)}' is a shipped theme and cannot be deleted",
                    severity="warning",
                )
            else:
                self.app.notify(
                    # R41: a stale ``unreadable:<stem>`` echoes file-name text.
                    f"No saved custom theme named '{escape_markup(printable(name))}'",
                    severity="warning",
                )
            return

        # task-1367: unlinking a user theme file is irreversible -- confirm
        # first; ``name`` is bound here, so a theme switch while the dialog is
        # up cannot delete the wrong file. An unreadable file is named by its
        # file name: it never registered a theme, so nothing falls back.
        readable = name in files
        theme_name = name if readable else printable(theme_path.name)

        async def _confirmed_delete() -> None:
            self._delete_user_theme(theme_path, theme_name, registered=readable)

        self.app.push_screen(
            ConfirmationDialog(
                title="Delete theme",
                message=(
                    f"Delete the saved theme '{theme_name}'?\n"
                    "This removes the theme file and cannot be undone."
                ),
                confirm_label="Delete theme",
                cancel_label="Keep theme",
                confirm_callback=_confirmed_delete,
            )
        )

    def _delete_user_theme(
        self, theme_path: Path, theme_name: str, *, registered: bool = True
    ) -> None:
        """Unlink a user theme file and reset the editor (post-confirmation).

        ``registered=False`` (an unreadable file, R41): nothing was registered
        under ``theme_name`` -- a readable theme may be literally named
        ``b.toml`` -- so no registration, pending revert or default moves.
        """
        try:
            with raw._scope(
                self, "theme_file", writing=True, selected_read=theme_path
            ) as operation:
                raw._unlink(operation, theme_path)

            if registered:
                self._release_registration(theme_name)
                from ..css.Themes.theme_catalog import retarget_pending_revert

                retarget_pending_revert(self.app, theme_name, None)
                self._fall_back_after_delete(theme_name)
            else:
                self.app.notify(f"Deleted theme '{escape_markup(theme_name)}'", severity="success")
                self.post_message(self.ThemesChanged())
            # The editor must not keep offering the deleted file as loaded;
            # a delete of some other theme (from the picker) keeps its edits.
            # An unreadable file was never loaded under ``theme_name``.
            if registered and self.current_theme_name == theme_name:
                self.load_theme("textual-dark")
        except Exception as e:
            logger.error(f"Failed to delete theme '{theme_name}': {e}")
            self.app.notify(f"Failed to delete theme: {escape_markup(self._failure_reason(e))}", severity="error")

    def _release_registration(self, name: str) -> None:
        """Drop ``name``'s user registration once its file is gone.

        PR #2375 review #9: Appearance and the palette stop offering it; a
        user file that shadowed a shipped theme hands the shipped
        registration back. Review #4: a file may shadow a Textual built-in
        too (nord).
        """
        shipped = next(
            (t for t in ALL_THEMES if getattr(t, "name", None) == name), None
        ) or BUILTIN_THEMES.get(name)
        if shipped is not None:
            self.app.register_theme(shipped)
        else:
            self.app.unregister_theme(name)

    def _fall_back_after_delete(self, name: str) -> None:
        """Move the running theme / launch default off a deleted theme.

        User decision 2026-09-25: deleting the launch default changes only
        the *setting* unless that theme is the one on screen -- deleting an
        inactive launch default must not switch the running theme.
        """
        from ..css.Themes.theme_catalog import (
            current_launch_default,
            persist_launch_default,
            use_theme,
        )

        launch = current_launch_default()
        was_active = str(self.app.theme) in (name, f"custom_{name}")
        if launch == name and was_active:
            change = use_theme(self.app, "textual-dark", persist=True)
            if change.persisted:
                self.post_message(self.LaunchDefaultChanged("textual-dark"))
                self._notify_saved(
                    f"Deleted '{escape_markup(name)}'; launch default and theme reset to Textual Dark",
                    change.caches_reloaded,
                )
            else:
                self.app.notify(
                    f"Deleted '{escape_markup(name)}', but could not save the launch default; check the config file",
                    severity="error",
                )
        elif launch == name:
            persisted, caches_reloaded = persist_launch_default(self.app, "textual-dark")
            if persisted:
                self.post_message(self.LaunchDefaultChanged("textual-dark"))
                self._notify_saved(
                    f"Deleted '{escape_markup(name)}'; launch default reset to Textual Dark",
                    caches_reloaded,
                )
            else:
                self.app.notify(
                    f"Deleted '{escape_markup(name)}', but could not save the launch default; check the config file",
                    severity="error",
                )
        elif was_active:
            use_theme(
                self.app,
                launch if launch in self.app.available_themes else "textual-dark",
                persist=False,
            )
            self.app.notify(
                f"Deleted '{escape_markup(name)}'; switched to your launch default", severity="success"
            )
        else:
            self.app.notify(f"Deleted theme '{escape_markup(name)}'", severity="success")
        self.post_message(self.ThemesChanged())

    def _notify_saved(self, message: str, caches_reloaded: bool) -> None:
        """A success toast, or a warning when the config reload failed
        (Qodo 4107495906 / 4107495911; the ``use_theme_toast`` wording)."""
        if caches_reloaded:
            self.app.notify(message, severity="success")
        else:
            self.app.notify(f"{message}; {CACHE_REFRESH_FAILED}", severity="warning")

    def rename_user_theme(self, old: str, new: str) -> bool:
        """Rename the saved theme ``old`` to ``new``: file, registration,
        running theme and launch default.

        Args:
            old: The saved theme's current name (or an ``unreadable:`` id,
                which is refused with its reason).
            new: The new name, already stripped by the prompt.

        Returns:
            True once renamed (also when the old file could not be removed;
            the user is told both files exist). False, with a notice, when
            nothing changed: invalid, reserved or taken name, unreadable or
            missing source, a pause, or a launch default that could not be
            moved (the new file is then removed again).
        """
        from ..css.Themes.theme_catalog import (
            current_launch_default,
            persist_launch_default,
            retarget_pending_revert,
            use_theme,
        )

        try:
            validate_filename(new)
        except ValueError as exc:
            self.app.notify(f"Invalid theme name: {escape_markup(exc)}", severity="error")
            return False
        if is_reserved_theme_name(new):
            self.app.notify(f"Invalid theme name: {RESERVED_NAME_RULE}", severity="error")
            return False
        if new == old:
            return True
        try:
            files, unreadable = self._scan_theme_files()
        except RecoveryRequired:
            self.app.notify(THEMES_UNAVAILABLE_LABEL, severity="warning")
            return False
        old_path = files.get(old)
        stem = old.removeprefix(UNREADABLE_ID_PREFIX)
        if old_path is None and stem in unreadable:
            self.app.notify(
                # notify parses markup; name and error are untrusted file text.
                f"Can't rename '{escape_markup(printable(stem))}': this theme file can't be read: "
                f"{escape_markup(unreadable[stem][1])}",
                severity="error",
            )
            return False
        if old_path is None:
            self.app.notify(f"No saved custom theme named '{escape_markup(printable(old))}'", severity="warning")
            return False
        shipped_names = {getattr(t, "name", None) for t in ALL_THEMES}
        new_path = self.custom_themes_path / f"{new}.toml"
        if (
            new in BUILTIN_THEMES
            or new in shipped_names
            or new in files
            or new_path.exists()
            or new in self.app.available_themes
        ):
            self.app.notify(f"Name taken: '{escape_markup(new)}'", severity="warning")
            return False

        try:
            with (
                raw._scope(self, "theme_file", selected_read=old_path) as operation,
                raw._file(operation, old_path, "r") as f,
            ):
                data = toml.load(f)
            data.setdefault("theme", {})["name"] = new
            # Review item 1: build the Theme before anything is written, so a
            # file Textual rejects (no primary, unknown colour key) fails the
            # rename with nothing on disk changed.
            theme = theme_from_file_data(data, new, new_path.name)
            self._write_toml(new_path, data)
        except RecoveryRequired:
            self.app.notify(THEMES_UNAVAILABLE_LABEL, severity="warning")
            return False
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            # TomlDecodeError is a ValueError; Theme(**args) raises TypeError.
            logger.error(f"Failed to rename theme '{old}': {self._failure_reason(exc)}")
            self.app.notify(
                f"Failed to rename theme '{escape_markup(old)}': "
                f"{escape_markup(self._failure_reason(exc))}",
                severity="error",
            )
            return False

        # Qodo 4107495893: move the launch default while the old file still
        # exists; if that write fails, take the new file back out so the
        # next launch still finds the theme its config names.
        caches_reloaded = True
        if current_launch_default() == old:
            # Spec §7 step 4: config only; the running theme is handled below.
            persisted, caches_reloaded = persist_launch_default(self.app, new)
            if not persisted:
                try:
                    with raw._scope(
                        self, "theme_file", writing=True, selected_read=new_path
                    ) as operation:
                        raw._unlink(operation, new_path)
                except (RecoveryRequired, OSError) as exc:
                    logger.error(
                        f"Could not remove '{new}' after a failed rename: {self._failure_reason(exc)}"
                    )
                self.app.notify(
                    f"Could not rename '{escape_markup(old)}': the launch default could not be saved; "
                    "check the config file",
                    severity="error",
                )
                return False
            self.post_message(self.LaunchDefaultChanged(new))

        # The new file exists; only now may the old one go.
        self.app.register_theme(theme)
        try:
            with raw._scope(
                self, "theme_file", writing=True, selected_read=old_path
            ) as operation:
                raw._unlink(operation, old_path)
        except (RecoveryRequired, OSError) as exc:
            logger.error(
                f"Renamed theme '{old}' but could not remove its old file: "
                f"{self._failure_reason(exc)}"
            )
            self.app.notify(
                f"Saved '{escape_markup(new)}', but could not remove '{escape_markup(old)}'; both files now exist",
                severity="warning",
            )
            self.post_message(self.ThemesChanged(highlight=new))
            return True

        if old in (self._loaded_user_theme, self.current_theme_name):
            # R13: the editor follows the file, so Save writes new.toml
            # instead of recreating old.toml. current_theme_name first, so
            # the Name box's Changed echo is a no-op.
            self.current_theme_name = new
            self._loaded_user_theme = new
            self.query_one("#settings-theme-name", Input).value = new
        if str(self.app.theme) in (old, f"custom_{old}"):
            use_theme(self.app, new, persist=False)
        self._release_registration(old)
        retarget_pending_revert(self.app, old, new)
        self.post_message(self.ThemesChanged(highlight=new))
        self._notify_saved(f"Renamed '{escape_markup(old)}' to '{escape_markup(new)}'", caches_reloaded)
        return True

    def export_theme(self, name: str) -> None:
        """Export the saved theme file ``name`` (not the editor's palette)."""
        try:
            validate_filename(name)  # it names the file in ~/Downloads
            theme_path = self._user_theme_files().get(name)
            if theme_path is None:
                self.app.notify(f"No saved custom theme named '{escape_markup(printable(name))}'", severity="warning")
                return
            with (
                raw._scope(self, "theme_file", selected_read=theme_path) as operation,
                raw._file(operation, theme_path, "r") as f,
            ):
                theme_data = toml.load(f)
        except RecoveryRequired:
            self.app.notify(THEMES_UNAVAILABLE_LABEL, severity="warning")
            return
        except (ValueError, OSError) as exc:  # TomlDecodeError is a ValueError
            reason = self._failure_reason(exc)
            logger.error(f"Failed to read theme '{printable(name)}' for export: {reason}")
            self.app.notify(
                f"Failed to export theme '{escape_markup(printable(name))}': {escape_markup(reason)}",
                severity="error",
            )
            return
        self._export(name, theme_data)

    def _export(self, name: str, theme_data: dict[str, Any]) -> None:
        """Write ``theme_data`` to ~/Downloads, confirming an overwrite."""
        export_path = Path.home() / "Downloads" / f"{name}_theme.toml"

        if export_path.exists():
            # TASK-31258: never silently replace an earlier export.
            async def _confirmed_export() -> None:
                self._write_export(export_path, theme_data)

            self.app.push_screen(
                ConfirmationDialog(
                    title="Overwrite export",
                    message=f"{export_path} already exists. Replace it?",
                    confirm_label="Overwrite",
                    cancel_label="Keep existing",
                    confirm_callback=_confirmed_export,
                )
            )
            return

        self._write_export(export_path, theme_data)

    def _write_export(self, export_path: Path, theme_data: dict[str, Any]) -> None:
        """Write the export TOML and report the path."""
        try:
            with raw._scope(self, "theme_export", writing=True, selected_read=export_path) as operation:
                raw._mkdirs(operation)
                temporary = export_path.with_suffix(export_path.suffix + ".tmp")
                try:
                    with raw._file(operation, temporary, "w") as f:
                        toml.dump(theme_data, f)
                    raw._replace(operation, temporary, export_path)
                finally:
                    raw._remove_temporary(operation, temporary)

            self.app.notify(f"Theme exported to: {escape_markup(export_path)}", severity="success")
            self.post_message(self.Exported(export_path))
        except Exception as e:
            logger.error(f"Failed to export theme: {e}")
            self.app.notify(f"Failed to export theme: {escape_markup(self._failure_reason(e))}", severity="error")

    def import_theme(self, source: str) -> str | None:
        """Import the theme file at ``source`` (a typed, pasted or dropped path).

        Returns:
            The imported theme's name once written; None when refused, paused,
            or waiting on the Replace confirmation (which writes, registers
            and posts ``ThemesChanged(highlight=name)`` itself).

        Args:
            source: The path the user gave; validated with
                ``path_validation.validate_browsing_path`` before any read.
        """
        try:
            files = self._user_theme_files()
        except RecoveryRequired:
            self.app.notify(THEMES_UNAVAILABLE_LABEL, severity="warning")
            return None
        parsed = self._parse_import(source)
        if isinstance(parsed, str):
            # R16: the reason only, never the source path; file text is
            # untrusted and notify parses markup.
            # R39: and may carry control characters.
            self.app.notify(escape_markup(printable(parsed)), severity="error")
            return None
        name, data, theme = parsed
        # R12: write back to the file that already claims the name.
        target = files.get(name, self.custom_themes_path / f"{name}.toml")
        if name in files or target.exists():

            async def _confirmed_replace() -> None:
                # Qodo 4109320416: the file may have moved while asking.
                current = self._resolve_write_target(name)
                if current is not None:
                    self._write_import(name, current, data, theme)

            self.app.push_screen(
                ConfirmationDialog(
                    title="Replace theme",
                    message=f"Replace the saved theme '{escape_markup(name)}'?",
                    confirm_label="Replace",
                    cancel_label="Keep existing",
                    confirm_callback=_confirmed_replace,
                )
            )
            return None
        return name if self._write_import(name, target, data, theme) else None

    def _parse_import(self, source: str) -> tuple[str, dict[str, Any], Theme] | str:
        """``(name, normalised file data, theme)``, or a path-free refusal reason."""
        text = source.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1]  # a terminal drop can paste a quoted path
        elif os.sep == "/":
            # R35/R40(e): macOS Terminal/iTerm drop `/a/My\ \&\ Theme.toml`,
            # backslash-escaping any shell-special character: `\X` -> `X`.
            text = re.sub(r"\\(.)", r"\1", text)
        try:
            path = validate_browsing_path(os.path.expanduser(text))
        except ValueError:
            return "Import needs the full path to a .toml file"
        if path.suffix.lower() != ".toml":
            return "Import needs a .toml file"
        try:
            # R34: check and read ONE descriptor, opened non-blocking, so a
            # FIFO (or one swapped in after a path check) can't stall the UI.
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    return "Import needs a regular file"
                if info.st_size > IMPORT_MAX_BYTES:
                    return "Theme file is larger than 64 KB"
                # Bounded: the file may grow after fstat.
                with os.fdopen(fd, "rb", closefd=False) as f:
                    content = f.read(IMPORT_MAX_BYTES + 1)
            finally:
                os.close(fd)
        except OSError as exc:
            return f"Could not read the file: {self._failure_reason(exc)}"
        if len(content) > IMPORT_MAX_BYTES:
            return "Theme file is larger than 64 KB"
        try:
            raw_data = toml.loads(content.decode("utf-8"))
        except Exception:  # noqa: BLE001 - any parser failure is "not TOML"
            return "File is not valid TOML"
        meta = raw_data.get("theme", {})
        if not isinstance(meta, dict):
            return "[theme] must be a table"
        colors = raw_data.get("colors")
        if not isinstance(colors, dict) or "primary" not in colors:
            return "Missing [colors].primary"
        for key, value in colors.items():
            # Spec §7 / R40(c): what the editor's colour fields accept.
            if not (isinstance(value, str) and self._validate_color_input(value)):
                return (
                    f"{str(key)[:PREVIEW_CHARS]}: '{str(value)[:PREVIEW_CHARS]}' "
                    "is not #RGB, #RRGGBB or #RRGGBBAA"
                )
        name = str(meta.get("name") or path.stem).strip() or path.stem
        try:
            validate_filename(name)
            if "[" in name:
                # Review Focus 1: a name is shown in toasts and titles.
                raise ValueError("'[' is not allowed")
            if not name.isprintable():
                raise ValueError("control characters are not allowed")  # R39
            if is_reserved_theme_name(name):
                raise ValueError(RESERVED_NAME_RULE)
        except ValueError as exc:
            return f"Invalid theme name: {exc}"
        if name in ("textual-dark", "textual-light"):
            return "Cannot overwrite built-in themes"
        data: dict[str, Any] = {
            "theme": {"name": name, "dark": theme_file_dark(meta.get("dark", True))},
            "colors": dict(colors),
        }
        if variables := sanitize_theme_variables(raw_data.get("variables"), f"{name}.toml"):
            data["variables"] = variables
        try:
            theme = theme_from_file_data(data, name, f"{name}.toml")
        except (TypeError, ValueError, AttributeError) as exc:
            return self._theme_file_error(exc, data)
        return name, data, theme

    def _write_import(self, name: str, target: Path, data: dict[str, Any], theme: Theme) -> bool:
        try:
            self._write_toml(target, data)
        except RecoveryRequired:
            self.app.notify(THEMES_UNAVAILABLE_LABEL, severity="warning")
            return False
        except OSError as exc:
            logger.error(f"Failed to import theme '{name}': {self._failure_reason(exc)}")
            self.app.notify(
                f"Failed to import theme: {escape_markup(self._failure_reason(exc))}",
                severity="error",
            )
            return False
        self.app.register_theme(theme)
        self._reapply_if_active(name)
        self.post_message(self.ThemesChanged(highlight=name))
        self.app.notify(f"Imported '{escape_markup(name)}'", severity="success")
        return True

    def _apply_preset_swatch(self, swatch: Static) -> None:
        """Apply a preset swatch's color to the last focused color input."""
        background = swatch.styles.background
        # str(Color) is "Color(r, g, b)", not a usable hex value -- normalize.
        color = background.hex if isinstance(background, Color) else str(background)

        target = self._preset_target()

        self.color_inputs[target].value = color
        self._update_color_swatch(target, color)
        self.current_theme_data[target] = color
        self.is_modified = True
        self._refresh_preview()

    @on(Click, ".color-preset-swatch")
    def on_preset_color_clicked(self, event: Click) -> None:
        """Handle clicks on color preset swatches."""
        self._apply_preset_swatch(event.control)

    def on_key(self, event: Key) -> None:
        """Apply a focused color preset swatch on Enter/Space (task-1369).

        Args:
            event: The key event; only ``enter`` and ``space`` are handled,
                and only when a preset swatch Static has focus. All other
                keys pass through untouched.
        """
        if event.key not in ("enter", "space"):
            return
        focused = self.app.focused
        if (
            isinstance(focused, Static)
            and focused.has_class("color-preset-swatch")
            and str(focused.id or "").startswith("settings-theme-preset-")
        ):
            event.stop()
            event.prevent_default()
            self._apply_preset_swatch(focused)

    @on(Button.Pressed, "#settings-theme-generate")
    def on_generate_theme(self) -> None:
        """Generate a complete theme based on the primary color."""
        primary_color = self.current_theme_data.get("primary", "#0099FF")

        try:
            primary = Color.parse(primary_color)
            generated_theme = self._generate_theme_from_primary(primary)

            for color_name, color_value in generated_theme.items():
                if color_name in self.color_inputs:
                    self.color_inputs[color_name].value = color_value
                    self._update_color_swatch(color_name, color_value)

            self.current_theme_data.update(generated_theme)
            self.is_modified = True
            self._refresh_preview()

            self.app.notify("Theme generated from primary color.", severity="success")
        except Exception as e:
            logger.error(f"Failed to generate theme: {e}")
            self.app.notify(f"Failed to generate theme: {e}", severity="error")

    def _generate_theme_from_primary(self, primary: Color) -> dict[str, str]:
        """Generate a complete theme based on a primary color."""
        # Textual's Color.hsl reports hue in the 0-1 range; _adjust_color works
        # in degrees (TASK-31253: every primary used to yield red/cyan).
        hsl = primary.hsl
        hue, saturation, lightness = hsl.h * 360, hsl.s, hsl.l

        return {
            "primary": primary.hex,
            "secondary": self._adjust_color(
                hue, saturation * 0.8, lightness * 0.8
            ),
            "accent": self._adjust_color(
                (hue + 180) % 360, saturation, lightness
            ),
            "background": self._adjust_color(
                hue, saturation * 0.1, 0.08 if self.is_dark_theme else 0.95
            ),
            "surface": self._adjust_color(
                hue, saturation * 0.1, 0.12 if self.is_dark_theme else 0.92
            ),
            "panel": self._adjust_color(
                hue, saturation * 0.1, 0.10 if self.is_dark_theme else 0.94
            ),
            "foreground": "#FFFFFF" if self.is_dark_theme else "#000000",
            "success": self._adjust_color(120, 0.7, 0.4),
            "warning": self._adjust_color(45, 0.9, 0.5),
            "error": self._adjust_color(0, 0.8, 0.5),
        }

    def _adjust_color(
        self, hue: float, saturation: float, lightness: float
    ) -> str:
        """Create a color from HSL values."""
        try:
            hue = hue % 360
            saturation = max(0, min(1, saturation))
            lightness = max(0, min(1, lightness))

            c = (1 - abs(2 * lightness - 1)) * saturation
            x = c * (1 - abs((hue / 60) % 2 - 1))
            m = lightness - c / 2

            if hue < 60:
                r, g, b = c, x, 0
            elif hue < 120:
                r, g, b = x, c, 0
            elif hue < 180:
                r, g, b = 0, c, x
            elif hue < 240:
                r, g, b = 0, x, c
            elif hue < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            r = max(0, min(255, int((r + m) * 255)))
            g = max(0, min(255, int((g + m) * 255)))
            b = max(0, min(255, int((b + m) * 255)))

            return f"#{r:02X}{g:02X}{b:02X}"
        except Exception:
            return "#808080"
