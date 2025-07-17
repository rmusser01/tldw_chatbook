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
            "Preview area - select a card to preview",
            id="preview-content",
            classes="preview-content"
        )
    
    def preview_card(self, card_name: str, card_data: Dict[str, Any]) -> None:
        """Start previewing a splash card."""
        self.current_card = card_name
        self.card_data = card_data
        
        # Update preview content
        preview_content = self.query_one("#preview-content", Static)
        
        if card_data.get("type") == "animated":
            # For animated cards, show a placeholder
            preview_content.update(
                f"[bold]{card_data.get('title', card_name)}[/bold]\n\n"
                f"[dim]Animated splash screen\n"
                f"Effect: {card_data.get('effect', 'Unknown')}\n\n"
                f"Click 'Play Selected' to see the animation[/dim]"
            )
        else:
            # For static cards, show the actual content
            content = card_data.get("art", ["No preview available"])
            if isinstance(content, list):
                content = "\n".join(content)
            preview_content.update(content)
    
    def stop_preview(self) -> None:
        """Stop the current preview."""
        self.is_playing = False
        if self.animation_timer:
            self.animation_timer.cancel()
            self.animation_timer = None


class SplashScreenViewer(Container):
    """Main viewer widget for browsing splash screens."""
    
    DEFAULT_CLASSES = "splash-viewer"
    
    # Built-in cards list
    BUILT_IN_CARDS = [
        "default", "blueprint", "matrix", "glitch", "retro", "tech_pulse",
        "code_scroll", "minimal_fade", "arcade_high_score", "digital_rain",
        "neon_city", "wave_pattern", "pixel_art", "terminal_boot", "cyber_grid",
        "starfield", "circuit_board", "hologram", "vaporwave", "ascii_banner",
        "loading_bars", "particle_system", "plasma_effect", "fire_effect",
        "water_ripple", "rainbow_wave", "binary_rain", "hexagon_grid",
        "fractal_tree", "spiral_loader", "pulse_rings", "data_flow",
        "neural_network", "quantum_particles", "laser_grid", "constellation",
        "ascii_art_tldw", "game_of_life", "wireframe_globe", "zen_garden",
        "ascii_fire", "rubiks_cube", "data_stream", "fractal_zoom",
        "ascii_spinner", "hacker_terminal"
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.splash_cards: Dict[str, SplashCardInfo] = {}
        self._load_splash_cards()
    
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
    
    def compose(self) -> ComposeResult:
        """Compose the viewer interface."""
        # Header
        yield Label("🎨 Splash Screen Gallery", classes="viewer-header")
        yield Static("Select a splash screen below to preview it. Double-click to play.", classes="help-text")
        
        # Build the list of available splash screens
        if not self.splash_cards:
            self._load_splash_cards()
        
        # Create options for the list
        options = []
        for card_name, card_info in self.splash_cards.items():
            option_text = f"{card_info.title} [{card_info.type}]"
            options.append(Option(option_text, id=card_name))
        
        if not options:
            options.append(Option("No splash screens found", id="none"))
        
        # Main list of splash screens
        yield OptionList(*options, id="card-list", classes="card-list")
        
        # Card information display
        yield Static("Select a card to see details", id="card-info", classes="card-info")
        
        # Action buttons
        with Horizontal(classes="action-buttons"):
            yield Button("Play Selected", id="play-selected-button", variant="primary")
            yield Button("Close", id="close-button", variant="default")
    
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
        info_text = f"""[bold]{card_info.title}[/bold]
Type: {card_info.type}  |  Effect: {card_info.effect or 'None'}
{card_info.description}

[dim]Double-click or press "Play Selected" to view this splash screen[/dim]"""
        info_widget.update(info_text)
    
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
            
        # Use default duration and animation speed
        duration = 3.0  # Default 3 seconds for preview
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
    
    @on(Button.Pressed, "#close-button")
    def on_close_button(self, event: Button.Pressed) -> None:
        """Handle close button press."""
        # Try to dismiss the parent screen (modal)
        try:
            if hasattr(self.app.screen, 'dismiss'):
                self.app.screen.dismiss()
        except Exception as e:
            logger.error(f"Failed to close: {e}")