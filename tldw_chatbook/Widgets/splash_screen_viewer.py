"""Splash screen viewer widget for browsing and previewing available splash screens."""

import asyncio
from typing import Optional, Dict, Any, List
import time

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Button, Select, Label, OptionList, Input
from textual.widgets.option_list import Option
from textual.reactive import reactive
from textual.message import Message
from textual.worker import Worker, WorkerState

from loguru import logger

from ..config import get_cli_setting, save_setting_to_cli_config
from ..Utils.splash_animations import BaseEffect
from .splash_screen import SplashScreen


class SplashCardInfo:
    """Container for splash card information."""
    
    def __init__(self, name: str, data: Dict[str, Any]):
        self.name = name
        self.data = data
        self.type = data.get("type", "static")
        self.effect = data.get("effect")
        self.title = data.get("title", name)
        self.description = self._get_description()
    
    def _get_description(self) -> str:
        """Generate a description based on card data."""
        if self.type == "static":
            return "Static ASCII art display"
        elif self.effect:
            return f"Animated with {self.effect.replace('_', ' ').title()} effect"
        return "Custom splash screen"


class SplashPreviewWidget(Container):
    """Widget for previewing splash screen animations."""
    
    DEFAULT_CLASSES = "splash-preview"
    
    # Reactive attributes
    is_playing: reactive[bool] = reactive(False)
    current_card: reactive[Optional[str]] = reactive(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.animation_timer = None
        self.effect_instance = None
        self.card_data = None
        
    def compose(self) -> ComposeResult:
        """Compose the preview widget."""
        yield Static(
            "Select a splash screen to preview",
            id="preview-content",
            classes="preview-content"
        )
    
    def preview_card(self, card_name: str, card_data: Dict[str, Any]) -> None:
        """Start previewing a splash card."""
        logger.debug(f"Starting preview for card: {card_name}")
        
        # Stop any existing preview
        self.stop_preview()
        
        self.current_card = card_name
        self.card_data = card_data
        self.is_playing = True
        
        # Get the preview content widget
        content_widget = self.query_one("#preview-content", Static)
        
        if card_data["type"] == "static":
            # For static cards, just display the content
            content = card_data.get("content", "")
            content_widget.update(content)
        else:
            # For animated cards, initialize the effect
            self._initialize_effect(content_widget)
    
    def _initialize_effect(self, content_widget: Static) -> None:
        """Initialize animation effect for the card."""
        if not self.card_data:
            return
        
        effect_name = self.card_data.get("effect")
        if not effect_name:
            return
        
        # Import the effect class dynamically
        try:
            from ..Utils import splash_animations
            effect_class = getattr(splash_animations, f"{effect_name.title().replace('_', '')}Effect", None)
            
            if effect_class:
                # Get terminal size for the preview area
                width = content_widget.size.width or 80
                height = content_widget.size.height or 24
                
                # Create effect instance with card data
                # Note: Some effects may need adjustment for preview size
                self.effect_instance = effect_class(width, height, self.card_data)
                
                # Start animation timer
                animation_speed = self.card_data.get("animation_speed", 0.1)
                self.animation_timer = self.set_interval(
                    animation_speed,
                    self._update_animation,
                    name="preview_animation"
                )
                
                # Initial render
                self._update_animation()
        except Exception as e:
            logger.error(f"Failed to initialize effect {effect_name}: {e}")
            content_widget.update(f"Error: Could not load {effect_name} effect")
    
    def _update_animation(self) -> None:
        """Update the animation frame."""
        if not self.effect_instance or not self.is_playing:
            return
        
        try:
            # Update the effect
            self.effect_instance.update()
            
            # Get the rendered frame
            frame = self.effect_instance.render()
            
            # Update the display
            content_widget = self.query_one("#preview-content", Static)
            content_widget.update(frame)
        except Exception as e:
            logger.error(f"Animation update error: {e}")
    
    def stop_preview(self) -> None:
        """Stop the current preview."""
        self.is_playing = False
        
        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None
        
        self.effect_instance = None
        self.current_card = None


class SplashScreenViewer(Container):
    """Main splash screen viewer widget."""
    
    DEFAULT_CLASSES = "splash-viewer"
    
    # Available splash cards (same as in splash_screen.py)
    BUILT_IN_CARDS = [
        "default", "blueprint", "matrix", "glitch", "retro", "tech_pulse",
        "code_scroll", "minimal_fade", "arcade_high_score", "digital_rain",
        "loading_bar", "starfield", "terminal_boot", "glitch_reveal",
        "ascii_morph", "game_of_life", "scrolling_credits", "spotlight_reveal",
        "sound_bars", "raindrops_pond", "pixel_zoom", "text_explosion",
        "old_film", "maze_generator", "dwarf_fortress", "neural_network",
        "quantum_particles", "ascii_wave", "binary_matrix", "constellation_map",
        "typewriter_news", "dna_sequence", "circuit_trace", "plasma_field",
        "ascii_fire", "rubiks_cube", "data_stream", "fractal_zoom",
        "ascii_spinner", "hacker_terminal"
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.splash_cards: Dict[str, SplashCardInfo] = {}
        self.current_config = {
            "enabled": True,
            "duration": 2.0,
            "card_selection": "random",
            "active_cards": ["default"],
            "animation_speed": 1.0,
        }
        self._load_splash_cards()
        self._load_current_config()
    
    def _load_splash_cards(self) -> None:
        """Load all available splash cards."""
        # Create a temporary splash screen instance to access card data
        temp_splash = SplashScreen(duration=0, show_progress=False)
        
        logger.info(f"Loading {len(self.BUILT_IN_CARDS)} built-in splash cards")
        
        for card_name in self.BUILT_IN_CARDS:
            try:
                card_data = temp_splash._load_card(card_name)
                self.splash_cards[card_name] = SplashCardInfo(card_name, card_data)
                logger.debug(f"Loaded card: {card_name}")
            except Exception as e:
                logger.error(f"Failed to load card {card_name}: {e}")
        
        logger.info(f"Successfully loaded {len(self.splash_cards)} splash cards")
    
    def _load_current_config(self) -> None:
        """Load current splash screen configuration."""
        self.current_config = {
            "enabled": get_cli_setting("splash_screen.enabled", True),
            "duration": get_cli_setting("splash_screen.duration", 2.0),
            "card_selection": get_cli_setting("splash_screen.card_selection", "random"),
            "active_cards": get_cli_setting("splash_screen.active_cards", ["default"]),
            "animation_speed": get_cli_setting("splash_screen.effects.animation_speed", 1.0),
        }
    
    def compose(self) -> ComposeResult:
        """Compose the viewer interface."""
        # Create a vertical layout for the entire viewer
        yield Label("🎨 Splash Screen Gallery", classes="viewer-header")
        
        # Main horizontal split
        with Horizontal():
            # Left panel - Card list
            with Vertical(classes="card-list-panel"):
                yield Label("Available Splash Screens", classes="section-header")
                yield Static("Click to select • Double-click to play", classes="help-text")
                
                # Build options list
                options = []
                if not self.splash_cards:
                    # Try loading cards again if empty
                    self._load_splash_cards()
                
                for card_name, card_info in self.splash_cards.items():
                    option_text = f"{card_info.title} [{card_info.type}]"
                    options.append(Option(option_text, id=card_name))
                
                if not options:
                    options.append(Option("No splash cards available", id="none"))
                
                yield OptionList(*options, id="card-list", classes="card-list")
            
            # Right panel - Preview and info
            with Vertical(classes="preview-panel"):
                yield Label("Preview", classes="section-header")
                yield SplashPreviewWidget(id="preview-widget", classes="preview-widget")
                
                yield Label("Card Information", classes="section-header")
                yield Static("Select a card to view details", id="card-info", classes="card-info")
        
        # Settings section (outside the horizontal layout)
        with Vertical(classes="settings-section"):
            yield Label("Splash Screen Settings", classes="section-header")
            
            with Horizontal(classes="setting-row"):
                yield Label("Selection Mode:", classes="setting-label")
                # Ensure we have a valid value for the Select widget
                card_selection = self.current_config.get("card_selection", "random")
                if card_selection is None:
                    card_selection = "random"
                
                # Build options list
                options = [("random", "Random"), ("sequential", "Sequential")]
                options.extend([(name, name.title()) for name in self.splash_cards.keys()])
                
                # Validate that the selected value is in the options
                valid_values = [opt[0] for opt in options]
                if card_selection not in valid_values:
                    card_selection = "random"  # Default to random if invalid
                
                # Create Select widget without initial value, then set it
                select_widget = Select(
                    options,
                    id="selection-mode",
                    classes="setting-control"
                )
                yield select_widget
            
            with Horizontal(classes="setting-row"):
                yield Label("Duration (seconds):", classes="setting-label")
                # Ensure we have a valid duration value
                duration = self.current_config.get("duration", 2.0)
                if duration is None:
                    duration = 2.0
                yield Input(
                    value=str(duration),
                    placeholder="2.0",
                    id="duration-input",
                    classes="setting-control"
                )
                yield Label("(0.5 - 5.0)", id="duration-help", classes="value-label")
            
            with Horizontal(classes="setting-row"):
                yield Label("Animation Speed:", classes="setting-label")
                # Ensure we have a valid animation speed value
                animation_speed = self.current_config.get("animation_speed", 1.0)
                if animation_speed is None:
                    animation_speed = 1.0
                yield Input(
                    value=str(animation_speed),
                    placeholder="1.0",
                    id="speed-input",
                    classes="setting-control"
                )
                yield Label("(0.5x - 2.0x)", id="speed-help", classes="value-label")
            
            # Active cards selection
            yield Label("Active Cards (for random selection):", classes="section-header")
            with VerticalScroll(classes="active-cards-container"):
                for card_name in self.splash_cards:
                    checkbox_id = f"active-{card_name}"
                    # Safely check if card is in active_cards
                    if self.current_config:
                        active_cards = self.current_config.get("active_cards", ["default"])
                    else:
                        active_cards = ["default"]
                    
                    # Ensure active_cards is not None
                    if active_cards is None:
                        active_cards = ["default"]
                        
                    checked = card_name in active_cards
                    yield Button(
                        f"{'☑' if checked else '☐'} {card_name}",
                        id=checkbox_id,
                        classes="checkbox-button"
                    )
            
            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("Preview Selected", id="preview-button", variant="primary")
                yield Button("Play Selected", id="play-selected-button", variant="warning")
                yield Button("Test Random/Sequential", id="test-button")
                yield Button("Save Settings", id="save-button", variant="success")
    
    def on_mount(self) -> None:
        """Set initial values after widget is mounted."""
        # Set the initial value for the selection mode
        try:
            select_widget = self.query_one("#selection-mode", Select)
            card_selection = self.current_config.get("card_selection", "random")
            if card_selection and card_selection in [opt[0] for opt in select_widget._options]:
                select_widget.value = card_selection
        except Exception as e:
            logger.error(f"Failed to set initial select value: {e}")
    
    @on(OptionList.OptionHighlighted)
    def on_card_selected(self, event: OptionList.OptionHighlighted) -> None:
        """Handle card selection from the list."""
        if not event.option_id:
            return
        
        card_name = event.option_id
        if card_name not in self.splash_cards:
            return
        
        card_info = self.splash_cards[card_name]
        
        # Update card info display
        info_widget = self.query_one("#card-info", Static)
        info_text = f"""Name: {card_info.title}
Type: {card_info.type}
Effect: {card_info.effect or 'None'}
Description: {card_info.description}"""
        info_widget.update(info_text)
        
        # Start preview
        preview_widget = self.query_one("#preview-widget", SplashPreviewWidget)
        preview_widget.preview_card(card_name, card_info.data)
    
    @on(OptionList.OptionSelected)
    def on_card_double_clicked(self, event: OptionList.OptionSelected) -> None:
        """Handle double-click on a card to play it."""
        if event.option_id:
            # Play the selected card
            self.run_worker(self._play_card_worker(event.option_id))
    
    @work(exclusive=True)
    async def _play_card_worker(self, card_id: str) -> None:
        """Worker to play a specific splash screen card."""
        if card_id not in self.splash_cards:
            return
            
        # Get duration and animation speed from settings
        try:
            duration = float(self.query_one("#duration-input", Input).value)
            duration = max(0.5, min(5.0, duration))
        except ValueError:
            duration = 2.0
            
        try:
            animation_speed = float(self.query_one("#speed-input", Input).value)
            animation_speed = max(0.5, min(2.0, animation_speed))
        except ValueError:
            animation_speed = 1.0
        
        # Create config for the specific card
        test_config = {
            "enabled": True,
            "duration": duration,
            "card_selection": card_id,
            "active_cards": [card_id],
            "effects": {
                "animation_speed": animation_speed
            }
        }
        
        # Show notification
        card_info = self.splash_cards[card_id]
        self.app.notify(f"Playing '{card_info.title}' splash screen...", severity="information")
        
        # Create splash screen with test config
        splash = SplashScreen(
            duration=duration,
            show_progress=True,
            config=test_config
        )
        
        # Create overlay container
        from textual.containers import Container
        overlay = Container(splash, id="splash-overlay")
        overlay.styles.layer = "overlay"
        overlay.styles.width = "100%"
        overlay.styles.height = "100%"
        overlay.styles.align = ("center", "middle")
        
        # Mount overlay
        await self.app.mount(overlay)
        
        # Wait for splash screen to complete
        await asyncio.sleep(duration)
        
        # Remove overlay
        await overlay.remove()
    
    @on(Button.Pressed, "#preview-button")
    def on_preview_button(self, event: Button.Pressed) -> None:
        """Handle preview button press."""
        # Get selected card from option list
        card_list = self.query_one("#card-list", OptionList)
        if card_list.highlighted is not None:
            option_id = card_list.get_option_at_index(card_list.highlighted).id
            if option_id and option_id in self.splash_cards:
                card_info = self.splash_cards[option_id]
                preview_widget = self.query_one("#preview-widget", SplashPreviewWidget)
                preview_widget.preview_card(option_id, card_info.data)
    
    @on(Button.Pressed, "#play-selected-button")
    def on_play_selected_button(self, event: Button.Pressed) -> None:
        """Play the currently selected splash screen."""
        # Get selected card from option list
        card_list = self.query_one("#card-list", OptionList)
        if card_list.highlighted is None:
            self.app.notify("Please select a splash screen first", severity="warning")
            return
            
        option_id = card_list.get_option_at_index(card_list.highlighted).id
        if not option_id or option_id not in self.splash_cards:
            self.app.notify("Invalid selection", severity="error")
            return
        
        # Play the selected card using the shared worker
        self.run_worker(self._play_card_worker(option_id))
    
    @on(Button.Pressed, "#test-button")
    @work(exclusive=True)
    async def on_test_button(self, event: Button.Pressed) -> None:
        """Handle test button press - show actual splash screen."""
        # Get current settings
        try:
            duration = float(self.query_one("#duration-input", Input).value)
            duration = max(0.5, min(5.0, duration))  # Clamp to valid range
        except ValueError:
            duration = 2.0
            
        selection_mode = self.query_one("#selection-mode", Select).value
        
        try:
            animation_speed = float(self.query_one("#speed-input", Input).value)
            animation_speed = max(0.5, min(2.0, animation_speed))  # Clamp to valid range
        except ValueError:
            animation_speed = 1.0
        
        # Create and mount a temporary splash screen
        test_config = {
            "enabled": True,
            "duration": duration,
            "card_selection": selection_mode,
            "active_cards": self._get_active_cards(),
            "effects": {
                "animation_speed": animation_speed
            }
        }
        
        # Show notification
        self.app.notify("Showing splash screen preview...", severity="information")
        
        # Create splash screen with test config
        splash = SplashScreen(
            duration=duration,
            show_progress=True,
            config=test_config
        )
        
        # Create overlay container
        from textual.containers import Container
        overlay = Container(splash, id="splash-overlay")
        overlay.styles.layer = "overlay"
        overlay.styles.width = "100%"
        overlay.styles.height = "100%"
        overlay.styles.align = ("center", "middle")
        
        # Mount overlay
        await self.app.mount(overlay)
        splash.focus()
        
        # Wait for it to complete
        await asyncio.sleep(duration + 0.5)
        
        # Remove overlay
        await overlay.remove()
    
    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input value changes for validation feedback."""
        if event.input.id == "duration-input":
            try:
                value = float(event.value)
                if 0.5 <= value <= 5.0:
                    event.input.styles.border = ("solid", "green")
                else:
                    event.input.styles.border = ("solid", "yellow")
            except ValueError:
                event.input.styles.border = ("solid", "red")
        elif event.input.id == "speed-input":
            try:
                value = float(event.value)
                if 0.5 <= value <= 2.0:
                    event.input.styles.border = ("solid", "green")
                else:
                    event.input.styles.border = ("solid", "yellow")
            except ValueError:
                event.input.styles.border = ("solid", "red")
    
    @on(Button.Pressed)
    def on_checkbox_toggle(self, event: Button.Pressed) -> None:
        """Handle checkbox button presses."""
        if event.button.id and event.button.id.startswith("active-"):
            # Toggle checkbox state
            button = event.button
            current_text = str(button.label)
            if current_text.startswith("☑"):
                button.label = "☐" + current_text[1:]
            else:
                button.label = "☑" + current_text[1:]
    
    def _get_active_cards(self) -> List[str]:
        """Get list of currently active cards."""
        active_cards = []
        for card_name in self.splash_cards:
            checkbox_id = f"active-{card_name}"
            try:
                button = self.query_one(f"#{checkbox_id}", Button)
                if button.label.startswith("☑"):
                    active_cards.append(card_name)
            except:
                pass
        return active_cards
    
    @on(Button.Pressed, "#save-button")
    def on_save_settings(self, event: Button.Pressed) -> None:
        """Save the current settings."""
        # Gather all settings
        try:
            duration = float(self.query_one("#duration-input", Input).value)
            duration = max(0.5, min(5.0, duration))
        except ValueError:
            duration = self.current_config["duration"]
            
        try:
            animation_speed = float(self.query_one("#speed-input", Input).value)
            animation_speed = max(0.5, min(2.0, animation_speed))
        except ValueError:
            animation_speed = self.current_config["animation_speed"]
            
        new_config = {
            "splash_screen.enabled": self.current_config["enabled"],
            "splash_screen.duration": duration,
            "splash_screen.card_selection": self.query_one("#selection-mode", Select).value,
            "splash_screen.active_cards": self._get_active_cards(),
            "splash_screen.effects.animation_speed": animation_speed,
        }
        
        # Save each setting
        for key, value in new_config.items():
            # Split the key to get section and setting name
            parts = key.split(".", 1)
            if len(parts) == 2:
                section, setting = parts
                # Handle nested settings like splash_screen.effects.animation_speed
                if "." in setting:
                    # For nested settings, we need to handle them differently
                    save_setting_to_cli_config(section, setting, value)
                else:
                    save_setting_to_cli_config(section, setting, value)
            else:
                # Single level setting
                save_setting_to_cli_config("general", key, value)
        
        # Update current config
        self._load_current_config()
        
        # Show success notification
        self.app.notify("Splash screen settings saved!", severity="information")