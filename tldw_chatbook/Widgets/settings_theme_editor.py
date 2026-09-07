"""Settings-native theme editor widget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml
from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.color import Color
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.events import Click, Key
from textual.message import Message
from textual.reactive import reactive
from textual.theme import Theme
from textual.widgets import Button, Checkbox, Input, Select, Static, Tree

from ..css.Themes.themes import ALL_THEMES, create_theme_from_dict
from ..Utils.path_validation import validate_filename
from .confirmation_dialog import ConfirmationDialog


class SettingsThemeEditor(Vertical):
    """Theme editor styled for the Settings screen."""

    class ThemeModifiedStatus(Message):
        """Message sent when the theme editor's modified state changes."""

        def __init__(self, is_modified: bool) -> None:
            self.is_modified = is_modified
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

    # TASK-31259: the Live Preview is a Console-shaped stub. Each row is
    # (id suffix, text); _PREVIEW_STYLE maps the suffix to the BASE_COLORS
    # keys used for its background and text, painted by _refresh_preview.
    _PREVIEW_ROWS = (
        ("rail", " Console ▸ Conversation · ready"),
        ("user", " You: summarise the attached paper"),
        ("assistant", " Assistant: Here is the summary…"),
        ("success", " ✓ tool web_search finished"),
        ("warning", " ! approval needed before the next call"),
        ("error", " ✗ provider returned 401"),
        ("accent", " [ Send ]   Ctrl+P palette"),
    )
    _PREVIEW_STYLE = {
        "rail": ("panel", "foreground"),
        "user": ("primary", "foreground"),
        "assistant": ("surface", "foreground"),
        "success": ("background", "success"),
        "warning": ("background", "warning"),
        "error": ("background", "error"),
        "accent": ("background", "accent"),
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from ..config import get_user_themes_dir

        self.custom_themes_path = get_user_themes_dir()
        self.custom_themes_path.mkdir(parents=True, exist_ok=True)
        self.color_inputs: dict[str, Input] = {}
        self.color_swatches: dict[str, Static] = {}
        # TASK-31258: the user theme file the palette was loaded from (or last
        # saved to); re-saving it is an update, saving over another is an
        # overwrite that asks first.
        self._loaded_user_theme: str | None = None

    def compose(self) -> ComposeResult:
        """Compose the theme editor widget.

        Yields:
            ComposeResult: The theme editor UI sections.
        """
        # Title is rendered by SettingsScreen._render_detail_pane()
        with Vertical(id="settings-theme-card", classes="settings-focus-card"):
            # TASK-31256: Actions before the palette and presets so Apply/Save
            # are reachable without Tabbing through 10 inputs and 40 swatches.
            yield from self._compose_library_section()
            yield from self._compose_actions_section()
            yield from self._compose_palette_section()
            yield from self._compose_preview_section()

    def _compose_library_section(self) -> ComposeResult:
        yield Static("Theme Library", classes="destination-section")
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
        # `theme-editor-action` keys the compact-mode width rule to these
        # buttons only (the CSS fast-path ratchet forbids a bare `Button`
        # subject, TASK-31279).
        with Horizontal(classes="settings-action-row"):
            yield Button("New", id="settings-theme-new", classes="theme-editor-action")
            yield Button("Clone", id="settings-theme-clone", classes="theme-editor-action")
            yield Button("Delete", id="settings-theme-delete", variant="error", classes="theme-editor-action")
            yield Button("Export", id="settings-theme-export", classes="theme-editor-action")
        yield Tree("Themes", id="settings-theme-tree")
        # task-1585: the collapsed tree left a large blank region with no
        # explanation of what fills it.
        yield Static(
            "Your themes come first; expand Shipped themes to browse the catalog. "
            "New starts a theme from the current palette.",
            id="settings-theme-tree-hint",
            classes="settings-detail-row",
        )

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
                    swatch.styles.background = color
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
            "Apply applies immediately - no Save needed; "
            "Save stores the theme; Set as launch default makes it load at startup.",
            id="settings-theme-apply-hint",
            classes="settings-help-copy",
        )
        with Horizontal(classes="settings-action-row"):
            yield Button("Apply", id="settings-theme-apply", variant="primary", classes="theme-editor-action")
            yield Button("Save", id="settings-theme-save", variant="success", classes="theme-editor-action")
            yield Button("Reset", id="settings-theme-reset", variant="warning", classes="theme-editor-action")
            yield Button(
                "Generate from Primary",
                id="settings-theme-generate",
                classes="theme-editor-action",
            )
            yield Button(
                "Set as launch default",
                id="settings-theme-set-default",
                classes="theme-editor-action",
            )

    def _compose_preview_section(self) -> ComposeResult:
        yield Static("Live Preview", classes="destination-section")
        # TASK-31259: painted from the palette being edited (see
        # _refresh_preview), so it follows every keystroke, not just Apply.
        with Vertical(id="settings-theme-preview", classes="settings-theme-preview"):
            for suffix, text in self._PREVIEW_ROWS:
                yield Static(
                    text,
                    id=f"settings-theme-preview-{suffix}",
                    classes="settings-theme-preview-row",
                )

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

    def watch_is_modified(self, is_modified: bool) -> None:
        """Notify parent screen when modified state changes."""
        self.post_message(SettingsThemeEditor.ThemeModifiedStatus(is_modified))

    def _load_user_themes(self, parent_node) -> None:
        """Load user-created themes from the themes directory."""
        for theme_file in sorted(self.custom_themes_path.glob("*.toml")):
            try:
                with open(theme_file, "r", encoding="utf-8") as f:
                    theme_data = toml.load(f)
                theme_name = theme_data.get("theme", {}).get("name", theme_file.stem)
                # TASK-31256: the leaf's data says which loader applies; the
                # label is the bare name (no "user:" prefix).
                parent_node.add_leaf(theme_name, data="user")
            except Exception as e:
                logger.error(f"Failed to load user theme {theme_file}: {e}")

    @on(Tree.NodeSelected)
    def on_theme_selected(self, event: Tree.NodeSelected) -> None:
        """Handle theme selection from the tree."""
        theme_name = str(event.node.label)
        if event.node.data == "user":
            self.load_user_theme(theme_name)
        elif event.node.data == "catalog":
            self.load_theme(theme_name)

    def load_theme(self, theme_name: str) -> None:
        """Load a theme for editing."""
        self.current_theme_name = theme_name
        self._loaded_user_theme = None

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

        self._update_color_inputs()
        self._update_dark_mode_checkbox()
        self.is_modified = False

    def load_user_theme(self, theme_name: str) -> None:
        """Load a user-created theme from file."""
        try:
            validate_filename(theme_name)
        except ValueError as exc:
            self.app.notify(f"Invalid theme name: {exc}", severity="error")
            return

        theme_path = self.custom_themes_path / f"{theme_name}.toml"
        if theme_path.exists():
            try:
                with open(theme_path, "r", encoding="utf-8") as f:
                    theme_data = toml.load(f)

                self.current_theme_name = theme_name
                self.current_theme_data = theme_data.get("colors", {})
                self.is_dark_theme = theme_data.get("theme", {}).get("dark", True)

                name_input = self.query_one("#settings-theme-name", Input)
                name_input.value = theme_name
                name_input.disabled = False

                self._update_color_inputs()
                self._update_dark_mode_checkbox()
                self.is_modified = False
                self._loaded_user_theme = theme_name

            except Exception as e:
                logger.error(f"Failed to load user theme {theme_name}: {e}")
                self.app.notify(f"Failed to load theme: {e}", severity="error")

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
        for suffix, (bg_key, fg_key) in self._PREVIEW_STYLE.items():
            try:
                row = self.query_one(f"#settings-theme-preview-{suffix}", Static)
            except QueryError:
                return
            background = self.current_theme_data.get(bg_key)
            foreground = self.current_theme_data.get(fg_key)
            try:
                if background:
                    row.styles.background = background
                if foreground:
                    row.styles.color = foreground
            except Exception:  # noqa: BLE001 - a half-typed hex must not break painting
                continue

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
                self.color_swatches[color_name].styles.background = color_value
                self.color_swatches[color_name].update(color_value.upper())
                self.color_swatches[color_name].styles.color = (
                    "black" if parsed_color.brightness > 0.5 else "white"
                )
            except Exception:
                self.color_swatches[color_name].styles.background = "#808080"
                self.color_swatches[color_name].update("Invalid")
                self.color_swatches[color_name].styles.color = "white"

    def _validate_color_input(self, color_value: str) -> bool:
        """Validate a color input value."""
        try:
            if not color_value.startswith("#"):
                return False
            hex_part = color_value[1:]
            if len(hex_part) not in {3, 6}:
                return False
            Color.parse(color_value)
            return True
        except Exception:
            return False

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
                        self.color_swatches[color_name].update("invalid")

    @on(Input.Changed, "#settings-theme-name")
    def on_theme_name_changed(self, event: Input.Changed) -> None:
        """Keep current_theme_name in step with the Name box (TASK-31251).

        Apply, Export, Reset and Delete all read ``current_theme_name``;
        without this they acted on the name from the last load. Programmatic
        loads set the box to the name already held, so the equality guard
        makes those echoes no-ops.
        """
        name = event.value.strip()
        if name != self.current_theme_name:
            # An emptied box clears the name too (PR #2375 review #6), so no
            # action can fall back to the previously loaded theme.
            self.current_theme_name = name

    def _require_theme_name(self) -> str | None:
        """Return the current theme name if it is non-empty and file-safe.

        Notifies and returns ``None`` otherwise, so Apply/Export/Delete/Reset/
        Set-as-default share Save's guard instead of trusting a raw value.
        """
        name = self.current_theme_name.strip()
        if not name:
            self.app.notify("Please enter a theme name", severity="warning")
            return None
        try:
            validate_filename(name)
        except ValueError as exc:
            self.app.notify(f"Invalid theme name: {exc}", severity="warning")
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
            theme_dict = {**self.current_theme_data, "dark": self.is_dark_theme}
            theme = create_theme_from_dict(
                name=f"custom_{self.current_theme_name}",
                theme_dict=theme_dict,
            )
            self.app.register_theme(theme)
            self.app.theme = theme.name
            self.app.notify(
                f"Theme '{self.current_theme_name}' applied", severity="information"
            )
        except Exception as e:
            logger.error(f"Failed to apply theme: {e}")
            self.app.notify(f"Failed to apply theme: {e}", severity="error")

    @on(Button.Pressed, "#settings-theme-save")
    def on_save_theme(self) -> None:
        """Save the current theme."""
        theme_name = self.query_one("#settings-theme-name", Input).value.strip()

        if not theme_name:
            self.app.notify("Please enter a theme name", severity="warning")
            return

        try:
            validate_filename(theme_name)
        except ValueError as exc:
            self.app.notify(f"Invalid theme name: {exc}", severity="warning")
            return

        if theme_name in ["textual-dark", "textual-light"]:
            self.app.notify("Cannot overwrite built-in themes", severity="warning")
            return

        theme_data = {
            "theme": {"name": theme_name, "dark": self.is_dark_theme},
            "colors": self.current_theme_data,
        }
        theme_path = self.custom_themes_path / f"{theme_name}.toml"

        # TASK-31258: writing over another saved theme is one keypress from
        # destroying it; re-saving the theme loaded from that very file is an
        # update and needs no dialog.
        if theme_path.exists() and self._loaded_user_theme != theme_name:

            async def _confirmed_overwrite() -> None:
                self._write_theme_file(theme_name, theme_path, theme_data)

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

    def _write_theme_file(
        self, theme_name: str, theme_path: Path, theme_data: dict[str, Any]
    ) -> None:
        """Write the theme TOML, register it, and update the tree."""
        try:
            with open(theme_path, "w", encoding="utf-8") as f:
                toml.dump(theme_data, f)

            # TASK-31250: register at once so Appearance and the palette can
            # offer the theme without a restart.
            self.app.register_theme(
                create_theme_from_dict(
                    theme_name, {**self.current_theme_data, "dark": self.is_dark_theme}
                )
            )

            self.app.notify(f"Theme '{theme_name}' saved", severity="success")
            self.is_modified = False
            self._loaded_user_theme = theme_name

            tree = self.query_one("#settings-theme-tree", Tree)
            user_node = None
            for node in tree.root.children:
                if str(node.label) == "Your themes":
                    user_node = node
                    break

            if user_node:
                theme_exists = any(
                    str(child.label) == theme_name for child in user_node.children
                )
                if not theme_exists:
                    user_node.add_leaf(theme_name, data="user")
        except Exception as e:
            logger.error(f"Failed to save theme: {e}")
            self.app.notify(f"Failed to save theme: {e}", severity="error")

    @on(Button.Pressed, "#settings-theme-set-default")
    def on_set_launch_default(self) -> None:
        """Make the current saved theme the startup theme (TASK-31250)."""
        name = self._require_theme_name()
        if name is None:
            return
        saved = (self.custom_themes_path / f"{name}.toml").exists()
        if not saved and not self._is_catalog_theme(name):
            self.app.notify(
                "Save the theme first, then set it as the launch default",
                severity="warning",
            )
            return
        from ..config import save_setting_to_cli_config

        # PR #2375 review #7: the config write reports success as a bool.
        if not save_setting_to_cli_config("general", "default_theme", name):
            self.app.notify(
                "Could not save the launch default; check the config file",
                severity="error",
            )
            return
        self.app.notify(f"'{name}' will load at the next launch", severity="success")

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
        user_theme_path = self.custom_themes_path / f"{self.current_theme_name}.toml"
        if user_theme_path.exists():
            self.load_user_theme(self.current_theme_name)
        elif self._is_catalog_theme(self.current_theme_name):
            self.load_theme(self.current_theme_name)
        else:
            # TASK-31251: a renamed, never-saved theme has nothing to go back
            # to; say so instead of claiming a reset happened.
            self.app.notify(
                f"No saved version of '{self.current_theme_name}' to reset to",
                severity="warning",
            )
            return
        self.app.notify("Theme reset to original values", severity="information")

    def _is_catalog_theme(self, name: str) -> bool:
        """True for Textual built-ins and shipped ALL_THEMES names."""
        return name in ("textual-dark", "textual-light") or any(
            getattr(theme, "name", None) == name for theme in ALL_THEMES
        )

    @on(Button.Pressed, "#settings-theme-new")
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

        TASK-31257: the tree hint promises "from the current palette" and
        Clone already works that way; the hardcoded blue set is only the
        fallback for an editor that has nothing loaded yet. The dark flag is
        kept as well.
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
        self.is_modified = True

        self.app.notify("Creating new theme", severity="information")

    @on(Button.Pressed, "#settings-theme-clone")
    def on_clone_theme(self) -> None:
        """Clone the current theme."""
        new_name = f"{self.current_theme_name}_copy"

        name_input = self.query_one("#settings-theme-name", Input)
        name_input.value = new_name
        name_input.disabled = False
        name_input.focus()

        self.current_theme_name = new_name
        self._loaded_user_theme = None
        self.is_modified = True

        self.app.notify(f"Cloned theme as '{new_name}'", severity="information")

    @on(Button.Pressed, "#settings-theme-delete")
    def on_delete_theme(self) -> None:
        """Delete the current user theme."""
        built_in_names = {"textual-dark", "textual-light"}
        shipped_names = {t.name for t in ALL_THEMES if hasattr(t, "name")}

        if self._require_theme_name() is None:
            return

        # File existence decides: anything saved in the user themes directory
        # is a user theme and deletable, even when its name shadows a shipped
        # catalog theme. The built-in/shipped guard only applies when no user
        # file exists for the name.
        theme_path = self.custom_themes_path / f"{self.current_theme_name}.toml"
        if not theme_path.exists():
            if self.current_theme_name in built_in_names:
                self.app.notify(
                    f"'{self.current_theme_name}' is a built-in theme and cannot be deleted",
                    severity="warning",
                )
            elif self.current_theme_name in shipped_names:
                self.app.notify(
                    f"'{self.current_theme_name}' is a shipped theme and cannot be deleted",
                    severity="warning",
                )
            else:
                self.app.notify(
                    f"No saved custom theme named '{self.current_theme_name}'",
                    severity="warning",
                )
            return

        # task-1367: unlinking a user theme file is irreversible -- confirm
        # first, capturing the name so a theme switch while the dialog is up
        # cannot delete the wrong file.
        theme_name = self.current_theme_name

        async def _confirmed_delete() -> None:
            self._delete_user_theme(theme_path, theme_name)

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

    def _delete_user_theme(self, theme_path: Path, theme_name: str) -> None:
        """Unlink a user theme file and reset the editor (post-confirmation)."""
        try:
            theme_path.unlink()
            self.app.notify(
                f"Deleted theme '{theme_name}'", severity="success"
            )

            tree = self.query_one("#settings-theme-tree", Tree)
            for node in tree.root.children:
                if str(node.label) == "Your themes":
                    for child in node.children:
                        if str(child.label) == theme_name:
                            child.remove()
                            break
                    break

            # PR #2375 review #9: drop the runtime registration so Appearance
            # and the palette stop offering the deleted theme; a user file that
            # shadowed a shipped theme hands the shipped registration back.
            shipped = next(
                (t for t in ALL_THEMES if getattr(t, "name", None) == theme_name), None
            )
            if shipped is not None:
                self.app.register_theme(shipped)
            else:
                self.app.unregister_theme(theme_name)

            from ..config import get_cli_setting, save_setting_to_cli_config

            if str(get_cli_setting("general", "default_theme", "textual-dark")) == theme_name:
                if save_setting_to_cli_config("general", "default_theme", "textual-dark"):
                    self.app.notify(
                        "Launch default reset to textual-dark", severity="information"
                    )
                else:
                    self.app.notify(
                        "Could not reset the launch default; check the config file",
                        severity="error",
                    )

            self.load_theme("textual-dark")
        except Exception as e:
            logger.error(f"Failed to delete theme '{theme_name}': {e}")
            self.app.notify(f"Failed to delete theme: {e}", severity="error")

    @on(Button.Pressed, "#settings-theme-export")
    def on_export_theme(self) -> None:
        """Export the current theme."""
        name = self._require_theme_name()
        if name is None:
            return

        export_path = Path.home() / "Downloads" / f"{name}_theme.toml"

        theme_data = {
            "theme": {
                "name": self.current_theme_name,
                "dark": self.is_dark_theme,
            },
            "colors": self.current_theme_data,
        }

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
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                toml.dump(theme_data, f)

            self.app.notify(f"Theme exported to: {export_path}", severity="success")
        except Exception as e:
            logger.error(f"Failed to export theme: {e}")
            self.app.notify(f"Failed to export theme: {e}", severity="error")

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

    def _populate_theme_tree(self) -> None:
        """Populate the theme tree with built-in, custom, and user themes."""
        tree = self.query_one("#settings-theme-tree", Tree)
        tree.root.remove_children()

        # TASK-31256: the user's own themes first and open; the two Textual
        # built-ins; then the 58 shipped themes collapsed so they do not push
        # everything else out of the 12-row box. The root is expanded so the
        # box is never a collapsed line over ten blank rows (task-1585).
        user_node = tree.root.add("Your themes", expand=True)
        self._load_user_themes(user_node)

        builtin_node = tree.root.add("Built-in", expand=True)
        builtin_node.add_leaf("textual-dark", data="catalog")
        builtin_node.add_leaf("textual-light", data="catalog")

        shipped_node = tree.root.add("Shipped themes", expand=False)
        for theme in ALL_THEMES:
            if hasattr(theme, "name"):
                shipped_node.add_leaf(theme.name, data="catalog")

        tree.root.expand()

    def on_show(self) -> None:
        """Refresh the theme tree when the widget becomes visible."""
        self._populate_theme_tree()
