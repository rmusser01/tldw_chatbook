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
from textual.events import Click
from textual.message import Message
from textual.reactive import reactive
from textual.theme import Theme
from textual.widgets import Button, Input, Static, Switch, Tree

from ..css.Themes.themes import ALL_THEMES, create_theme_from_dict


class SettingsThemeEditor(Vertical):
    """Theme editor styled for the Settings screen."""

    class ThemeModifiedStatus(Message):
        """Message sent when the theme editor's modified state changes."""

        def __init__(self, is_modified: bool) -> None:
            self.is_modified = is_modified
            super().__init__()

    current_theme_name = reactive("textual-dark")
    current_theme_data: reactive[dict[str, str]] = reactive({}, layout=False)
    is_dark_theme = reactive(True)
    is_modified = reactive(False)

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
        self.custom_themes_path = Path.home() / ".config" / "tldw_cli" / "themes"
        self.custom_themes_path.mkdir(parents=True, exist_ok=True)
        self.color_inputs: dict[str, Input] = {}
        self.color_swatches: dict[str, Static] = {}
        self.last_focused_color_input: str | None = None

    def compose(self) -> ComposeResult:
        # Title is rendered by SettingsScreen._render_detail_pane()
        with Vertical(id="settings-theme-card", classes="settings-focus-card"):
            yield from self._compose_library_section()
            yield from self._compose_palette_section()
            yield from self._compose_actions_section()
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
            yield Switch(value=True, id="settings-theme-dark-mode")
        with Horizontal(classes="settings-action-row"):
            yield Button("New", id="settings-theme-new", variant="primary")
            yield Button("Clone", id="settings-theme-clone")
            yield Button("Delete", id="settings-theme-delete", variant="error")
            yield Button("Export", id="settings-theme-export")
        yield Tree("Themes", id="settings-theme-tree")

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
                    yield swatch

    def _compose_actions_section(self) -> ComposeResult:
        yield Static("Actions", classes="destination-section")
        with Horizontal(classes="settings-action-row"):
            yield Button("Apply", id="settings-theme-apply", variant="primary")
            yield Button("Save", id="settings-theme-save", variant="success")
            yield Button("Reset", id="settings-theme-reset", variant="warning")
            yield Button("Generate from Primary", id="settings-theme-generate", variant="primary")

    def _compose_preview_section(self) -> ComposeResult:
        yield Static("Live Preview", classes="destination-section")
        with Horizontal(classes="settings-preview-grid"):
            with Vertical(classes="preview-column"):
                yield Static("Buttons", classes="destination-section")
                yield Button("Default", classes="preview-button")
                yield Button("Primary", variant="primary", classes="preview-button")
                yield Button("Success", variant="success", classes="preview-button")
            with Vertical(classes="preview-column"):
                yield Static("Surfaces", classes="destination-section")
                yield Static("Surface", classes="preview-surface-demo")
                yield Static("Panel", classes="preview-panel-demo")
                yield Static("Boost", classes="preview-boost-demo")

    def on_mount(self) -> None:
        """Initialize the theme editor."""
        for color_name in self.BASE_COLORS:
            self.color_inputs[color_name] = self.query_one(
                f"#settings-theme-color-{color_name}", Input
            )
            self.color_swatches[color_name] = self.query_one(
                f"#settings-theme-swatch-{color_name}", Static
            )

        self.load_theme(self.app.theme)

        if "primary" in self.color_inputs:
            self.color_inputs["primary"].add_class("selected")
            self.last_focused_color_input = "primary"

    def watch_is_modified(self, is_modified: bool) -> None:
        """Notify parent screen when modified state changes."""
        self.post_message(SettingsThemeEditor.ThemeModifiedStatus(is_modified))

    def _load_user_themes(self, parent_node) -> None:
        """Load user-created themes from the themes directory."""
        for theme_file in self.custom_themes_path.glob("*.toml"):
            try:
                with open(theme_file, "r", encoding="utf-8") as f:
                    theme_data = toml.load(f)
                theme_name = theme_data.get("theme", {}).get("name", theme_file.stem)
                parent_node.add_leaf(f"user:{theme_name}")
            except Exception as e:
                logger.error(f"Failed to load user theme {theme_file}: {e}")

    @on(Tree.NodeSelected)
    def on_theme_selected(self, event: Tree.NodeSelected) -> None:
        """Handle theme selection from the tree."""
        if not event.node.children:
            theme_name = str(event.node.label)
            if theme_name.startswith("user:"):
                theme_name = theme_name[5:]
                self.load_user_theme(theme_name)
            else:
                self.load_theme(theme_name)

    def load_theme(self, theme_name: str) -> None:
        """Load a theme for editing."""
        self.current_theme_name = theme_name

        name_input = self.query_one("#settings-theme-name", Input)
        name_input.value = theme_name
        name_input.disabled = theme_name in ["textual-dark", "textual-light"]

        if theme_name in ["textual-dark", "textual-light"]:
            self.app.theme = theme_name
            self.current_theme_data = self._extract_current_theme_colors()
            self.is_dark_theme = theme_name == "textual-dark"
        else:
            theme = next(
                (t for t in ALL_THEMES if hasattr(t, "name") and t.name == theme_name),
                None,
            )
            if theme:
                self.current_theme_data = self._extract_theme_colors(theme)
                self.is_dark_theme = getattr(theme, "dark", True)

        self._update_color_inputs()
        self._update_dark_mode_switch()
        self.is_modified = False

    def load_user_theme(self, theme_name: str) -> None:
        """Load a user-created theme from file."""
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
                self._update_dark_mode_switch()
                self.is_modified = False

            except Exception as e:
                logger.error(f"Failed to load user theme {theme_name}: {e}")
                self.app.notify(f"Failed to load theme: {e}", severity="error")

    def _extract_current_theme_colors(self) -> dict[str, str]:
        """Extract colors from the current app theme."""
        if self.app.theme == "textual-dark":
            return {
                "primary": "#004578",
                "secondary": "#0178D4",
                "accent": "#FFD700",
                "background": "#0C0C0C",
                "surface": "#1A1A1A",
                "panel": "#1A1A1A",
                "foreground": "#E0E0E0",
                "success": "#4EBF71",
                "warning": "#FFA62B",
                "error": "#BA3C5B",
            }
        if self.app.theme == "textual-light":
            return {
                "primary": "#004578",
                "secondary": "#0178D4",
                "accent": "#FFD700",
                "background": "#F5F5F5",
                "surface": "#FFFFFF",
                "panel": "#EEEEEE",
                "foreground": "#333333",
                "success": "#228B22",
                "warning": "#FFA500",
                "error": "#DC143C",
            }
        return {color_name: "#808080" for color_name in self.BASE_COLORS}

    def _extract_theme_colors(self, theme: Theme) -> dict[str, str]:
        """Extract colors from a Theme object."""
        colors: dict[str, str] = {}
        color_mappings = {
            "primary": "primary",
            "secondary": "secondary",
            "accent": "accent",
            "background": "background",
            "surface": "surface",
            "panel": "panel",
            "foreground": "foreground",
            "success": "success",
            "warning": "warning",
            "error": "error",
        }

        for our_name, theme_attr in color_mappings.items():
            color_value = getattr(theme, theme_attr, None)
            if color_value:
                if isinstance(color_value, Color):
                    colors[our_name] = color_value.hex
                elif isinstance(color_value, str):
                    try:
                        parsed_color = Color.parse(color_value)
                        colors[our_name] = parsed_color.hex
                    except Exception:
                        colors[our_name] = "#808080"
                else:
                    colors[our_name] = "#808080"
            else:
                is_dark = getattr(theme, "dark", True)
                if is_dark:
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
                else:
                    defaults = {
                        "primary": "#0066CC",
                        "secondary": "#004499",
                        "accent": "#FF9900",
                        "background": "#FFFFFF",
                        "surface": "#F5F5F5",
                        "panel": "#EEEEEE",
                        "foreground": "#000000",
                        "success": "#228B22",
                        "warning": "#FFA500",
                        "error": "#DC143C",
                    }
                colors[our_name] = defaults.get(our_name, "#808080")

        return colors

    def _update_color_inputs(self) -> None:
        """Update color input fields with current theme data."""
        for color_name, color_value in self.current_theme_data.items():
            if color_name in self.color_inputs:
                self.color_inputs[color_name].value = color_value
                self._update_color_swatch(color_name, color_value)

    def _update_dark_mode_switch(self) -> None:
        """Update the dark mode switch."""
        switch = self.query_one("#settings-theme-dark-mode", Switch)
        switch.value = self.is_dark_theme

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

    def watch_focused(self, focused) -> None:
        """Watch for focus changes to track color input selection."""
        if (
            focused
            and isinstance(focused, Input)
            and focused.id
            and focused.id.startswith("settings-theme-color-")
        ):
            for input_widget in self.color_inputs.values():
                input_widget.remove_class("selected")
            focused.add_class("selected")
            self.last_focused_color_input = focused.id[len("settings-theme-color-") :]

    @on(Input.Changed)
    def on_color_input_changed(self, event: Input.Changed) -> None:
        """Handle color input changes."""
        if event.input.id and event.input.id.startswith("settings-theme-color-"):
            color_name = event.input.id[len("settings-theme-color-") :]
            color_value = event.value.strip()

            if color_value:
                if self._validate_color_input(color_value):
                    self._update_color_swatch(color_name, color_value)
                    self.current_theme_data[color_name] = color_value
                    self.is_modified = True
                    event.input.remove_class("invalid-color")
                else:
                    event.input.add_class("invalid-color")
                    self._update_color_swatch(color_name, "#000000")

    @on(Switch.Changed, "#settings-theme-dark-mode")
    def on_dark_mode_changed(self, event: Switch.Changed) -> None:
        """Handle dark mode switch changes."""
        self.is_dark_theme = event.value
        self.is_modified = True

    @on(Button.Pressed, "#settings-theme-apply")
    def on_apply_theme(self) -> None:
        """Apply the current theme to the app."""
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

        if theme_name in ["textual-dark", "textual-light"]:
            self.app.notify("Cannot overwrite built-in themes", severity="warning")
            return

        theme_data = {
            "theme": {"name": theme_name, "dark": self.is_dark_theme},
            "colors": self.current_theme_data,
        }
        theme_path = self.custom_themes_path / f"{theme_name}.toml"

        try:
            with open(theme_path, "w", encoding="utf-8") as f:
                toml.dump(theme_data, f)

            self.app.notify(f"Theme '{theme_name}' saved", severity="success")
            self.is_modified = False

            tree = self.query_one("#settings-theme-tree", Tree)
            user_node = None
            for node in tree.root.children:
                if str(node.label) == "User Themes":
                    user_node = node
                    break

            if user_node:
                theme_exists = any(
                    str(child.label) == f"user:{theme_name}"
                    for child in user_node.children
                )
                if not theme_exists:
                    user_node.add_leaf(f"user:{theme_name}")
        except Exception as e:
            logger.error(f"Failed to save theme: {e}")
            self.app.notify(f"Failed to save theme: {e}", severity="error")

    @on(Button.Pressed, "#settings-theme-reset")
    def on_reset_theme(self) -> None:
        """Reset theme to original values."""
        self.load_theme(self.current_theme_name)
        self.app.notify("Theme reset to original values", severity="information")

    @on(Button.Pressed, "#settings-theme-new")
    def on_new_theme(self) -> None:
        """Create a new theme."""
        self.current_theme_name = "new_theme"
        self.current_theme_data = {
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
        self.is_dark_theme = True

        name_input = self.query_one("#settings-theme-name", Input)
        name_input.value = "new_theme"
        name_input.disabled = False
        name_input.focus()

        self._update_color_inputs()
        self._update_dark_mode_switch()
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
        self.is_modified = True

        self.app.notify(f"Cloned theme as '{new_name}'", severity="information")

    @on(Button.Pressed, "#settings-theme-delete")
    def on_delete_theme(self) -> None:
        """Delete the current user theme."""
        built_in_names = {"textual-dark", "textual-light"}
        custom_names = {t.name for t in ALL_THEMES if hasattr(t, "name")}

        if (
            self.current_theme_name.startswith("user:")
            or self.current_theme_name in built_in_names
            or self.current_theme_name in custom_names
        ):
            self.app.notify("Cannot delete built-in or custom themes", severity="warning")
            return

        theme_path = self.custom_themes_path / f"{self.current_theme_name}.toml"
        if theme_path.exists():
            try:
                theme_path.unlink()
                self.app.notify(
                    f"Deleted theme '{self.current_theme_name}'", severity="success"
                )

                tree = self.query_one("#settings-theme-tree", Tree)
                for node in tree.root.children:
                    if str(node.label) == "User Themes":
                        for child in node.children:
                            if str(child.label) == f"user:{self.current_theme_name}":
                                child.remove()
                                break
                        break

                self.load_theme("textual-dark")
            except Exception as e:
                logger.error(f"Failed to delete theme: {e}")
                self.app.notify(f"Failed to delete theme: {e}", severity="error")

    @on(Button.Pressed, "#settings-theme-export")
    def on_export_theme(self) -> None:
        """Export the current theme."""
        export_path = Path.home() / "Downloads" / f"{self.current_theme_name}_theme.toml"

        theme_data = {
            "theme": {
                "name": self.current_theme_name,
                "dark": self.is_dark_theme,
            },
            "colors": self.current_theme_data,
        }

        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                toml.dump(theme_data, f)

            self.app.notify(f"Theme exported to: {export_path}", severity="success")
        except Exception as e:
            logger.error(f"Failed to export theme: {e}")
            self.app.notify(f"Failed to export theme: {e}", severity="error")

    @on(Click, ".color-preset-swatch")
    def on_preset_color_clicked(self, event: Click) -> None:
        """Handle clicks on color preset swatches."""
        color = str(event.control.styles.background)

        target = self.last_focused_color_input or "primary"
        if target not in self.color_inputs:
            target = "primary"

        self.color_inputs[target].value = color
        self._update_color_swatch(target, color)
        self.current_theme_data[target] = color
        self.is_modified = True
        self.color_inputs[target].add_class("selected")

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

            self.app.notify("Theme generated from primary color!", severity="success")
        except Exception as e:
            logger.error(f"Failed to generate theme: {e}")
            self.app.notify(f"Failed to generate theme: {e}", severity="error")

    def _generate_theme_from_primary(self, primary: Color) -> dict[str, str]:
        """Generate a complete theme based on a primary color."""
        h, s, l = primary.hsl

        return {
            "primary": primary.hex,
            "secondary": self._adjust_color(h, s * 0.8, l * 0.8),
            "accent": self._adjust_color((h + 180) % 360, s, l),
            "background": self._adjust_color(
                h, s * 0.1, 0.08 if self.is_dark_theme else 0.95
            ),
            "surface": self._adjust_color(
                h, s * 0.1, 0.12 if self.is_dark_theme else 0.92
            ),
            "panel": self._adjust_color(
                h, s * 0.1, 0.10 if self.is_dark_theme else 0.94
            ),
            "foreground": "#FFFFFF" if self.is_dark_theme else "#000000",
            "success": self._adjust_color(120, 0.7, 0.4),
            "warning": self._adjust_color(45, 0.9, 0.5),
            "error": self._adjust_color(0, 0.8, 0.5),
        }

    def _adjust_color(self, h: float, s: float, l: float) -> str:
        """Create a color from HSL values."""
        try:
            h = h % 360
            s = max(0, min(1, s))
            l = max(0, min(1, l))

            c = (1 - abs(2 * l - 1)) * s
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = l - c / 2

            if h < 60:
                r, g, b = c, x, 0
            elif h < 120:
                r, g, b = x, c, 0
            elif h < 180:
                r, g, b = 0, c, x
            elif h < 240:
                r, g, b = 0, x, c
            elif h < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            r = int((r + m) * 255)
            g = int((g + m) * 255)
            b = int((b + m) * 255)

            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return "#808080"

    def _populate_theme_tree(self) -> None:
        """Populate the theme tree with built-in, custom, and user themes."""
        tree = self.query_one("#settings-theme-tree", Tree)
        tree.root.remove_children()

        builtin_node = tree.root.add("Built-in Themes", expand=True)
        builtin_node.add_leaf("textual-dark")
        builtin_node.add_leaf("textual-light")

        custom_node = tree.root.add("Custom Themes", expand=True)
        for theme in ALL_THEMES:
            if hasattr(theme, "name"):
                custom_node.add_leaf(theme.name)

        user_node = tree.root.add("User Themes", expand=True)
        self._load_user_themes(user_node)

    def on_show(self) -> None:
        """Refresh the theme tree when the widget becomes visible."""
        self._populate_theme_tree()
