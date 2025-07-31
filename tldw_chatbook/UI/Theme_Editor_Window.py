# tldw_chatbook/UI/Theme_Editor_Window.py
"""
Theme Editor Window for creating and modifying Textual themes.
Allows users to customize colors, preview changes, and save custom themes.
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from pathlib import Path
import json
import toml
from functools import partial

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll, ScrollableContainer
from textual.widgets import (
    Static, Button, Input, Label, Tree, Switch,
    TabbedContent, TabPane, Collapsible
)
from textual.reactive import reactive
from textual.theme import Theme
from textual.color import Color
from textual.message import Message
from textual.binding import Binding

from loguru import logger

# Import all available themes
from ..css.Themes.themes import ALL_THEMES, create_theme_from_dict

if TYPE_CHECKING:
    from ..app import TldwCli


class ThemeModifiedMessage(Message):
    """Message sent when theme is modified."""
    
    def __init__(self, theme_name: str, theme_data: Dict[str, Any]) -> None:
        self.theme_name = theme_name
        self.theme_data = theme_data
        super().__init__()


class ThemeEditorView(VerticalScroll):
    """Main theme editor interface."""
    
    DEFAULT_CSS = """
    ThemeEditorView {
        height: 100%;
        scrollbar-gutter: stable;
    }
    
    .theme-list-section {
        height: auto;
        min-height: 20;
        max-height: 30;
        background: $boost;
        padding: 1;
        border-bottom: thick $background;
        overflow-y: auto;
    }
    
    .color-editor-section {
        height: auto;
        min-height: 30;
        padding: 1 2;
        border-bottom: thick $background;
    }
    
    .actions-section {
        height: auto;
        min-height: 8;
        padding: 1 2;
        background: $boost;
        border-bottom: thick $background;
        layout: vertical;
        align: center middle;
    }
    
    .preview-section {
        height: auto;
        min-height: 15;
        background: $surface;
        padding: 1;
        margin-bottom: 2;
    }
    
    .section-title {
        text-style: bold;
        margin: 1 0 2 0;
        color: $primary;
        text-align: center;
        border-bottom: solid $primary;
        padding-bottom: 1;
    }
    
    .color-input-group {
        height: 5;
        margin-bottom: 1;
        width: 100%;
    }
    
    .color-input-row {
        layout: horizontal;
        height: 4;
        align: center middle;
        width: 100%;
    }
    
    .color-label {
        width: 18;
        text-align: right;
        margin-right: 2;
        padding-right: 1;
    }
    
    .color-input {
        width: 16;
        min-width: 12;
        height: 3;
        padding: 0 1;
    }
    
    .color-input.invalid-color {
        border: thick $error;
        color: $error;
    }
    
    .color-input.selected {
        border: thick $accent;
    }
    
    .theme-name-input {
        width: 40;
        min-width: 30;
        height: 3;
        padding: 0 1;
    }
    
    .color-swatch {
        width: 14;
        height: 4;
        margin-left: 2;
        border: solid $primary;
        content-align: center middle;
        padding: 0;
        text-align: center;
        text-style: bold;
    }
    
    .theme-actions {
        layout: horizontal;
        height: auto;
        margin-top: 0;
        align: center middle;
        width: 100%;
    }
    
    .theme-actions Button {
        margin: 0 2;
        min-width: 20;
        height: 3;
    }
    
    .actions-section .section-title {
        margin: 0 0 1 0;
        border-bottom: none;
        padding-bottom: 0;
    }
    
    .theme-list-container {
        layout: horizontal;
        height: 100%;
    }
    
    .theme-tree-wrapper {
        width: 40%;
        min-width: 30;
        padding-right: 1;
    }
    
    .theme-info-wrapper {
        width: 60%;
        padding-left: 1;
    }
    
    #theme-tree {
        height: 100%;
    }
    
    .color-editor-container {
        layout: horizontal;
        height: 100%;
        width: 100%;
    }
    
    .color-inputs-wrapper {
        width: 70%;
        layout: horizontal;
        padding: 0 2;
    }
    
    .color-inputs-column {
        width: 50%;
        padding: 0 2;
        margin-right: 1;
    }
    
    .theme-actions-wrapper {
        width: 30%;
        padding: 0 1;
        overflow-y: auto;
    }
    
    .preview-container {
        layout: grid;
        grid-size: 4 1;
        grid-columns: 1fr 1fr 1fr 1fr;
        height: auto;
        width: 100%;
        overflow-x: auto;
    }
    
    .preview-component {
        padding: 0 1;
        min-width: 20;
        layout: vertical;
        height: auto;
    }
    
    .preview-label {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    
    .preview-button {
        margin: 0 0 1 0;
        width: 100%;
    }
    
    .preview-input {
        margin-bottom: 1;
        width: 100%;
    }
    
    .color-presets-container {
        margin-bottom: 1;
        max-height: 25;
        overflow-y: auto;
    }
    
    .color-preset-button {
        margin: 0 1 0 0;
        border: solid $panel;
        min-width: 4;
        width: 4;
        height: 3;
    }
    
    .color-preset-button:hover {
        border: thick $primary;
    }
    
    .preset-row {
        layout: horizontal;
        height: 4;
        margin-bottom: 1;
        align: center middle;
        width: 100%;
    }
    
    .preset-label {
        width: 8;
        text-align: right;
        margin-right: 1;
    }
    
    .preview-surface-demo {
        background: $surface;
        padding: 1;
        margin: 0 0 1 0;
        height: 3;
        width: 100%;
    }
    
    .preview-panel-demo {
        background: $panel;
        padding: 1;
        margin: 0 0 1 0;
        height: 3;
        width: 100%;
    }
    
    .preview-boost-demo {
        background: $boost;
        padding: 1;
        margin: 0 0 1 0;
        height: 3;
        width: 100%;
    }
    
    .text-primary {
        color: $primary;
    }
    
    .text-error {
        color: $error;
    }
    """
    
    # Base theme colors to edit
    BASE_COLORS = [
        "primary", "secondary", "accent",
        "background", "surface", "panel",
        "foreground",
        "success", "warning", "error"
    ]
    
    # Preset color palettes for quick selection
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
    
    current_theme_name = reactive("textual-dark")
    current_theme_data = reactive({})
    is_modified = reactive(False)
    is_dark_theme = reactive(True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_themes_path = Path.home() / ".config" / "tldw_cli" / "themes"
        self.custom_themes_path.mkdir(parents=True, exist_ok=True)
        self.color_inputs: Dict[str, Input] = {}
        self.color_swatches: Dict[str, Static] = {}
        self.last_focused_color_input: Optional[str] = None
        
    def compose(self) -> ComposeResult:
        """Compose the theme editor layout."""
        # Top section - Theme list
        with Container(classes="theme-list-section"):
            yield Label("Theme Library", classes="section-title")
            
            with Container(classes="theme-list-container"):
                # Theme tree on the left
                with Container(classes="theme-tree-wrapper"):
                    tree = Tree("Themes", id="theme-tree")
                    tree.root.expand()
                    
                    # Add built-in themes
                    builtin_node = tree.root.add("Built-in Themes", expand=True)
                    builtin_node.add_leaf("textual-dark")
                    builtin_node.add_leaf("textual-light")
                    
                    # Add custom themes from ALL_THEMES
                    custom_node = tree.root.add("Custom Themes", expand=True)
                    for theme in ALL_THEMES:
                        if hasattr(theme, 'name'):
                            custom_node.add_leaf(theme.name)
                    
                    # Add user themes
                    user_node = tree.root.add("User Themes", expand=True)
                    self._load_user_themes(user_node)
                    
                    yield tree
                
                # Theme info and actions on the right
                with Container(classes="theme-info-wrapper"):
                    # Theme name input
                    with Container(classes="color-input-group"):
                        with Horizontal(classes="color-input-row"):
                            yield Label("Theme Name:", classes="color-label")
                            yield Input(
                                placeholder="Enter theme name",
                                id="theme-name-input",
                                classes="theme-name-input",
                                disabled=True
                            )
                    
                    # Dark/Light mode switch
                    with Container(classes="color-input-group"):
                        with Horizontal(classes="color-input-row"):
                            yield Label("Dark Theme:", classes="color-label")
                            yield Switch(value=True, id="dark-mode-switch")
                    
                    # Theme actions
                    with Container(classes="theme-actions"):
                        yield Button("New", id="new-theme", variant="primary")
                        yield Button("Clone", id="clone-theme")
                        yield Button("Delete", id="delete-theme", variant="error")
                        yield Button("Export", id="export-theme")
        
        # Middle section - Color editor
        with Container(classes="color-editor-section"):
            yield Label("Color Editor", classes="section-title")
            
            with Container(classes="color-editor-container"):
                # Color inputs split into two columns
                with Container(classes="color-inputs-wrapper"):
                    # First column of colors - Primary colors
                    with Container(classes="color-inputs-column"):
                        primary_colors = ["primary", "secondary", "accent", "foreground", "background"]
                        for color_name in primary_colors:
                            with Container(classes="color-input-group"):
                                with Horizontal(classes="color-input-row"):
                                    label = Label(f"{color_name.title()}:", classes="color-label")
                                    yield label
                                    color_input = Input(
                                        placeholder="#RRGGBB",
                                        id=f"color-{color_name}",
                                        classes="color-input",
                                        max_length=7,
                                        tooltip=f"Enter hex color for {color_name}"
                                    )
                                    yield color_input
                                    swatch = Static("", id=f"swatch-{color_name}", classes="color-swatch")
                                    yield swatch
                    
                    # Second column of colors - Surface and status colors
                    with Container(classes="color-inputs-column"):
                        surface_colors = ["surface", "panel", "success", "warning", "error"]
                        for color_name in surface_colors:
                            with Container(classes="color-input-group"):
                                with Horizontal(classes="color-input-row"):
                                    label = Label(f"{color_name.title()}:", classes="color-label")
                                    yield label
                                    color_input = Input(
                                        placeholder="#RRGGBB",
                                        id=f"color-{color_name}",
                                        classes="color-input",
                                        max_length=7,
                                        tooltip=f"Enter hex color for {color_name}"
                                    )
                                    yield color_input
                                    swatch = Static("", id=f"swatch-{color_name}", classes="color-swatch")
                                    yield swatch
            
                # Actions and presets on the right
                with Container(classes="theme-actions-wrapper"):
                    # Color presets
                    yield Label("Color Presets", classes="section-title")
                    with VerticalScroll(classes="color-presets-container"):
                        for palette_name, colors in self.COLOR_PRESETS.items():
                            with Container(classes="preset-row"):
                                yield Label(f"{palette_name}:", classes="preset-label")
                                for i, color in enumerate(colors[:5]):
                                    button = Button(
                                        "",
                                        id=f"preset-{palette_name}-{i}",
                                        classes="color-preset-button"
                                    )
                                    button.styles.background = color
                                    yield button
                    
                    # Remove actions from here - they'll be moved below
        
        # Actions section - between color editor and preview
        with Container(classes="actions-section"):
            yield Label("Actions", classes="section-title")
            with Container(classes="theme-actions"):
                yield Button("Apply", id="apply-theme", variant="primary")
                yield Button("Save", id="save-theme", variant="success")
                yield Button("Reset", id="reset-theme", variant="warning")
                yield Button("Generate from Primary", id="generate-theme", variant="primary")
        
        # Bottom section - Live preview
        with Container(classes="preview-section"):
            yield Label("Live Preview", classes="section-title")
            
            with Container(classes="preview-container"):
                # Preview components arranged in grid
                with Vertical(classes="preview-component"):
                    yield Label("Buttons", classes="preview-label")
                    yield Button("Default", classes="preview-button")
                    yield Button("Primary", variant="primary", classes="preview-button")
                    yield Button("Success", variant="success", classes="preview-button")
                    yield Button("Warning", variant="warning", classes="preview-button")
                    yield Button("Error", variant="error", classes="preview-button")
                
                with Vertical(classes="preview-component"):
                    yield Label("Text & Labels", classes="preview-label")
                    yield Static("Normal text on background")
                    yield Static("[dim]Dimmed text variant[/dim]")
                    yield Static("[bold]Bold emphasis text[/bold]")
                    yield Static("Primary accent", classes="text-primary")
                    yield Static("Error message", classes="text-error")
                
                with Vertical(classes="preview-component"):
                    yield Label("Input Fields", classes="preview-label")
                    yield Input(placeholder="Enter text...", classes="preview-input")
                    yield Input(value="Focused input", classes="preview-input")
                    yield Input(value="Disabled", disabled=True, classes="preview-input")
                
                with Vertical(classes="preview-component"):
                    yield Label("Surfaces", classes="preview-label")
                    yield Static("Surface background", classes="preview-surface-demo")
                    yield Static("Panel background", classes="preview-panel-demo")
                    yield Static("Boost background", classes="preview-boost-demo")
    
    def on_mount(self) -> None:
        """Initialize the theme editor."""
        # Store references to color inputs and swatches
        for color_name in self.BASE_COLORS:
            self.color_inputs[color_name] = self.query_one(f"#color-{color_name}", Input)
            self.color_swatches[color_name] = self.query_one(f"#swatch-{color_name}", Static)
        
        # Load the current theme
        self.load_theme(self.app.theme)
        
        # Select primary color input by default
        if "primary" in self.color_inputs:
            self.color_inputs["primary"].add_class("selected")
            self.last_focused_color_input = "primary"
    
    def _load_user_themes(self, parent_node) -> None:
        """Load user-created themes from the themes directory."""
        for theme_file in self.custom_themes_path.glob("*.toml"):
            try:
                with open(theme_file, 'r') as f:
                    theme_data = toml.load(f)
                theme_name = theme_data.get('theme', {}).get('name', theme_file.stem)
                parent_node.add_leaf(f"user:{theme_name}")
            except Exception as e:
                logger.error(f"Failed to load user theme {theme_file}: {e}")
    
    @on(Tree.NodeSelected)
    def on_theme_selected(self, event: Tree.NodeSelected) -> None:
        """Handle theme selection from the tree."""
        # Check if it's a leaf node (no children)
        if not event.node.children:
            theme_name = str(event.node.label)
            if theme_name.startswith("user:"):
                theme_name = theme_name[5:]  # Remove "user:" prefix
                self.load_user_theme(theme_name)
            else:
                self.load_theme(theme_name)
    
    def load_theme(self, theme_name: str) -> None:
        """Load a theme for editing."""
        self.current_theme_name = theme_name
        
        # Update theme name input
        name_input = self.query_one("#theme-name-input", Input)
        name_input.value = theme_name
        name_input.disabled = theme_name in ["textual-dark", "textual-light"]
        
        # Get theme data
        if theme_name in ["textual-dark", "textual-light"]:
            # Built-in themes - extract colors from current app theme
            self.app.theme = theme_name
            theme_colors = self._extract_current_theme_colors()
            self.current_theme_data = theme_colors
            self.is_dark_theme = theme_name == "textual-dark"
        else:
            # Custom theme from ALL_THEMES
            theme = next((t for t in ALL_THEMES if hasattr(t, 'name') and t.name == theme_name), None)
            if theme:
                self.current_theme_data = self._extract_theme_colors(theme)
                self.is_dark_theme = getattr(theme, 'dark', True)
        
        # Update UI
        self._update_color_inputs()
        self._update_dark_mode_switch()
        self.is_modified = False
    
    def load_user_theme(self, theme_name: str) -> None:
        """Load a user-created theme from file."""
        theme_path = self.custom_themes_path / f"{theme_name}.toml"
        if theme_path.exists():
            try:
                with open(theme_path, 'r') as f:
                    theme_data = toml.load(f)
                
                self.current_theme_name = theme_name
                self.current_theme_data = theme_data.get('colors', {})
                self.is_dark_theme = theme_data.get('theme', {}).get('dark', True)
                
                # Update UI
                name_input = self.query_one("#theme-name-input", Input)
                name_input.value = theme_name
                name_input.disabled = False
                
                self._update_color_inputs()
                self._update_dark_mode_switch()
                self.is_modified = False
                
            except Exception as e:
                logger.error(f"Failed to load user theme {theme_name}: {e}")
                self.app.notify(f"Failed to load theme: {e}", severity="error")
    
    def _extract_current_theme_colors(self) -> Dict[str, str]:
        """Extract colors from the current app theme."""
        colors = {}
        
        # Get the current theme from the app
        if self.app.theme == "textual-dark":
            # Default dark theme colors
            colors = {
                "primary": "#004578",
                "secondary": "#0178D4", 
                "accent": "#FFD700",
                "background": "#0C0C0C",
                "surface": "#1A1A1A",
                "panel": "#1A1A1A",
                "foreground": "#E0E0E0",
                "success": "#4EBF71",
                "warning": "#FFA62B",
                "error": "#BA3C5B"
            }
        elif self.app.theme == "textual-light":
            # Default light theme colors  
            colors = {
                "primary": "#004578",
                "secondary": "#0178D4",
                "accent": "#FFD700", 
                "background": "#F5F5F5",
                "surface": "#FFFFFF",
                "panel": "#EEEEEE",
                "foreground": "#333333",
                "success": "#228B22",
                "warning": "#FFA500",
                "error": "#DC143C"
            }
        else:
            # For custom themes, try to extract from current styles
            # This is a fallback - ideally we'd extract from the theme object
            for color_name in self.BASE_COLORS:
                colors[color_name] = "#808080"  # Gray as fallback
                
        return colors
    
    def _extract_theme_colors(self, theme: Theme) -> Dict[str, str]:
        """Extract colors from a Theme object."""
        colors = {}
        
        # Map of theme attributes to our color names
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
            "error": "error"
        }
        
        for our_name, theme_attr in color_mappings.items():
            color_value = getattr(theme, theme_attr, None)
            if color_value:
                if isinstance(color_value, Color):
                    colors[our_name] = color_value.hex
                elif isinstance(color_value, str):
                    try:
                        # Ensure it's a valid color string
                        parsed_color = Color.parse(color_value)
                        colors[our_name] = parsed_color.hex
                    except:
                        colors[our_name] = "#808080"  # Gray fallback
                else:
                    colors[our_name] = "#808080"
            else:
                # Provide sensible defaults based on theme darkness
                is_dark = getattr(theme, 'dark', True)
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
                        "error": "#FF0000"
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
                        "error": "#DC143C"
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
        switch = self.query_one("#dark-mode-switch", Switch)
        switch.value = self.is_dark_theme
    
    def _update_color_swatch(self, color_name: str, color_value: str) -> None:
        """Update a color swatch preview."""
        if color_name in self.color_swatches:
            try:
                # Validate color
                parsed_color = Color.parse(color_value)
                self.color_swatches[color_name].styles.background = color_value
                # Update the swatch text to show the hex value
                self.color_swatches[color_name].update(color_value.upper())
                # Set text color based on background brightness
                if parsed_color.brightness > 0.5:
                    self.color_swatches[color_name].styles.color = "black"
                else:
                    self.color_swatches[color_name].styles.color = "white"
            except:
                self.color_swatches[color_name].styles.background = "#808080"
                self.color_swatches[color_name].update("Invalid")
                self.color_swatches[color_name].styles.color = "white"
    
    def _validate_color_input(self, color_value: str) -> bool:
        """Validate a color input value."""
        try:
            # Check basic format
            if not color_value.startswith("#"):
                return False
            
            # Remove # and check length
            hex_part = color_value[1:]
            if len(hex_part) not in [3, 6]:
                return False
            
            # Try to parse with Textual's Color
            Color.parse(color_value)
            return True
        except:
            return False
    
    def watch_focused(self, focused) -> None:
        """Watch for focus changes to track color input selection."""
        if focused and isinstance(focused, Input) and focused.id and focused.id.startswith("color-"):
            # Remove selection from all inputs
            for input_widget in self.color_inputs.values():
                input_widget.remove_class("selected")
            
            # Add selection to focused input
            focused.add_class("selected")
            self.last_focused_color_input = focused.id[6:]  # Remove "color-" prefix
    
    @on(Input.Changed)
    def on_color_input_changed(self, event: Input.Changed) -> None:
        """Handle color input changes."""
        if event.input.id and event.input.id.startswith("color-"):
            color_name = event.input.id[6:]  # Remove "color-" prefix
            color_value = event.value.strip()
            
            # Validate and update
            if color_value:
                if self._validate_color_input(color_value):
                    # Valid color - update swatch and data
                    self._update_color_swatch(color_name, color_value)
                    self.current_theme_data[color_name] = color_value
                    self.is_modified = True
                    event.input.remove_class("invalid-color")
                else:
                    # Invalid color - mark input
                    event.input.add_class("invalid-color")
                    self._update_color_swatch(color_name, "#000000")
    
    @on(Switch.Changed, "#dark-mode-switch")
    def on_dark_mode_changed(self, event: Switch.Changed) -> None:
        """Handle dark mode switch changes."""
        self.is_dark_theme = event.value
        self.is_modified = True
    
    @on(Button.Pressed, "#apply-theme")
    def on_apply_theme(self) -> None:
        """Apply the current theme to the app."""
        try:
            # Create a new theme from the current data
            theme_dict = {
                **self.current_theme_data,
                "dark": self.is_dark_theme
            }
            
            theme = create_theme_from_dict(
                name=f"custom_{self.current_theme_name}",
                theme_dict=theme_dict
            )
            
            # Register and apply the theme
            self.app.register_theme(theme)
            self.app.theme = theme.name
            
            self.app.notify(f"Theme '{self.current_theme_name}' applied", severity="information")
            
        except Exception as e:
            logger.error(f"Failed to apply theme: {e}")
            self.app.notify(f"Failed to apply theme: {e}", severity="error")
    
    @on(Button.Pressed, "#save-theme")
    def on_save_theme(self) -> None:
        """Save the current theme."""
        theme_name = self.query_one("#theme-name-input", Input).value.strip()
        
        if not theme_name:
            self.app.notify("Please enter a theme name", severity="warning")
            return
        
        if theme_name in ["textual-dark", "textual-light"]:
            self.app.notify("Cannot overwrite built-in themes", severity="warning")
            return
        
        # Save to file
        theme_data = {
            "theme": {
                "name": theme_name,
                "dark": self.is_dark_theme
            },
            "colors": self.current_theme_data
        }
        
        theme_path = self.custom_themes_path / f"{theme_name}.toml"
        
        try:
            with open(theme_path, 'w') as f:
                toml.dump(theme_data, f)
            
            self.app.notify(f"Theme '{theme_name}' saved", severity="success")
            self.is_modified = False
            
            # Update the tree to show the new theme
            tree = self.query_one("#theme-tree", Tree)
            user_node = None
            for node in tree.root.children:
                if node.label == "User Themes":
                    user_node = node
                    break
            
            if user_node:
                # Check if theme already exists in tree
                theme_exists = False
                for child in user_node.children:
                    if child.label == f"user:{theme_name}":
                        theme_exists = True
                        break
                
                if not theme_exists:
                    user_node.add_leaf(f"user:{theme_name}")
            
        except Exception as e:
            logger.error(f"Failed to save theme: {e}")
            self.app.notify(f"Failed to save theme: {e}", severity="error")
    
    @on(Button.Pressed, "#reset-theme")
    def on_reset_theme(self) -> None:
        """Reset theme to original values."""
        self.load_theme(self.current_theme_name)
        self.app.notify("Theme reset to original values", severity="information")
    
    @on(Button.Pressed, "#new-theme")
    def on_new_theme(self) -> None:
        """Create a new theme."""
        # Start with default dark theme colors
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
            "error": "#FF0000"
        }
        self.is_dark_theme = True
        
        # Update UI
        name_input = self.query_one("#theme-name-input", Input)
        name_input.value = "new_theme"
        name_input.disabled = False
        name_input.focus()
        
        self._update_color_inputs()
        self._update_dark_mode_switch()
        self.is_modified = True
        
        self.app.notify("Creating new theme", severity="information")
    
    @on(Button.Pressed, "#clone-theme")
    def on_clone_theme(self) -> None:
        """Clone the current theme."""
        new_name = f"{self.current_theme_name}_copy"
        
        # Update theme name
        name_input = self.query_one("#theme-name-input", Input)
        name_input.value = new_name
        name_input.disabled = False
        name_input.focus()
        
        self.current_theme_name = new_name
        self.is_modified = True
        
        self.app.notify(f"Cloned theme as '{new_name}'", severity="information")
    
    @on(Button.Pressed, "#delete-theme")
    def on_delete_theme(self) -> None:
        """Delete the current user theme."""
        if not self.current_theme_name.startswith("user:") and \
           self.current_theme_name not in ["textual-dark", "textual-light"] and \
           self.current_theme_name not in [t.name for t in ALL_THEMES if hasattr(t, 'name')]:
            
            theme_path = self.custom_themes_path / f"{self.current_theme_name}.toml"
            if theme_path.exists():
                try:
                    theme_path.unlink()
                    self.app.notify(f"Deleted theme '{self.current_theme_name}'", severity="success")
                    
                    # Remove from tree
                    tree = self.query_one("#theme-tree", Tree)
                    for node in tree.root.children:
                        if node.label == "User Themes":
                            for child in node.children:
                                if child.label == f"user:{self.current_theme_name}":
                                    child.remove()
                                    break
                            break
                    
                    # Load default theme
                    self.load_theme("textual-dark")
                    
                except Exception as e:
                    logger.error(f"Failed to delete theme: {e}")
                    self.app.notify(f"Failed to delete theme: {e}", severity="error")
        else:
            self.app.notify("Cannot delete built-in or custom themes", severity="warning")
    
    @on(Button.Pressed)
    def on_preset_color_clicked(self, event: Button.Pressed) -> None:
        """Handle clicks on color preset buttons."""
        if event.button.id and event.button.id.startswith("preset-"):
            # Get the color from the button's background style
            if event.button.styles.background:
                color = str(event.button.styles.background)
                
                # Apply to the last focused color input
                if self.last_focused_color_input and self.last_focused_color_input in self.color_inputs:
                    self.color_inputs[self.last_focused_color_input].value = color
                    self._update_color_swatch(self.last_focused_color_input, color)
                    self.current_theme_data[self.last_focused_color_input] = color
                    self.is_modified = True
                    
                    # Keep the input selected
                    self.color_inputs[self.last_focused_color_input].add_class("selected")
                else:
                    # If no input was focused, default to primary
                    self.last_focused_color_input = "primary"
                    self.color_inputs["primary"].value = color
                    self._update_color_swatch("primary", color)
                    self.current_theme_data["primary"] = color
                    self.is_modified = True
                    self.color_inputs["primary"].add_class("selected")
                    self.app.notify("Applied to primary color. Click a color input to select it.", severity="information")
    
    @on(Button.Pressed, "#export-theme")
    def on_export_theme(self) -> None:
        """Export the current theme."""
        # For now, just show the path where it would be saved
        export_path = Path.home() / "Downloads" / f"{self.current_theme_name}_theme.toml"
        
        theme_data = {
            "theme": {
                "name": self.current_theme_name,
                "dark": self.is_dark_theme
            },
            "colors": self.current_theme_data
        }
        
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'w') as f:
                toml.dump(theme_data, f)
            
            self.app.notify(f"Theme exported to: {export_path}", severity="success")
        except Exception as e:
            logger.error(f"Failed to export theme: {e}")
            self.app.notify(f"Failed to export theme: {e}", severity="error")
    
    @on(Button.Pressed, "#generate-theme")
    def on_generate_theme(self) -> None:
        """Generate a complete theme based on the primary color."""
        primary_color = self.current_theme_data.get("primary", "#0099FF")
        
        try:
            # Parse the primary color
            primary = Color.parse(primary_color)
            
            # Generate a harmonious color scheme
            generated_theme = self._generate_theme_from_primary(primary)
            
            # Update all color inputs
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
    
    def _generate_theme_from_primary(self, primary: Color) -> Dict[str, str]:
        """Generate a complete theme based on a primary color."""
        # Get HSL values for manipulation
        h, s, l = primary.hsl
        
        # Generate harmonious colors
        theme = {
            "primary": primary.hex,
            "secondary": self._adjust_color(h, s * 0.8, l * 0.8),  # Darker, less saturated
            "accent": self._adjust_color((h + 180) % 360, s, l),  # Complementary color
            "background": self._adjust_color(h, s * 0.1, 0.08 if self.is_dark_theme else 0.95),
            "surface": self._adjust_color(h, s * 0.1, 0.12 if self.is_dark_theme else 0.92),
            "panel": self._adjust_color(h, s * 0.1, 0.10 if self.is_dark_theme else 0.94),
            "foreground": "#FFFFFF" if self.is_dark_theme else "#000000",
            "success": self._adjust_color(120, 0.7, 0.4),  # Green
            "warning": self._adjust_color(45, 0.9, 0.5),   # Orange/Yellow
            "error": self._adjust_color(0, 0.8, 0.5),      # Red
        }
        
        return theme
    
    def _adjust_color(self, h: float, s: float, l: float) -> str:
        """Create a color from HSL values."""
        try:
            # Ensure values are in valid ranges
            h = h % 360
            s = max(0, min(1, s))
            l = max(0, min(1, l))
            
            # Convert HSL to RGB
            # This is a simplified conversion - for production, use a proper color library
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
        except:
            return "#808080"  # Fallback to gray