# splash_screen.py
# Customizable splash screen widget for tldw_chatbook startup
# Supports static and animated splash screens with Call of Duty-style "calling cards"

import asyncio
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple

from textual.app import ComposeResult
from textual.containers import Container, Center, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static, ProgressBar, Label
from textual.timer import Timer
from textual import events
from textual.message import Message
from rich.text import Text
from rich.align import Align
from rich.console import Console
from rich.style import Style

from loguru import logger

from ..Utils.Splash_Strings import splashscreen_message_selection
from ..config import get_cli_setting
from ..Utils.Splash import get_ascii_art, get_splash_card_config

# Import the registration system and load all effects
from ..Utils.Splash_Screens import load_all_effects, get_effect_class
from ..Utils.Splash_Screens.card_definitions import get_all_card_definitions

# Load all effects on module import
load_all_effects()


class SplashScreen(Container):
    """Customizable splash screen widget with animation support."""
    
    # Set the default classes to ensure proper styling
    DEFAULT_CLASSES = "splash-screen"
    
    # Reactive attributes
    progress: reactive[float] = reactive(0.0)
    progress_text: reactive[str] = reactive("Initializing...")
    is_active: reactive[bool] = reactive(True)
    current_frame: reactive[int] = reactive(0)
    
    # Default splash screen content if nothing is configured
    DEFAULT_SPLASH = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  ████████╗██╗     ██████╗ ██╗    ██╗                             ║
║  ╚══██╔══╝██║     ██╔══██╗██║    ██║                             ║
║     ██║   ██║     ██║  ██║██║ █╗ ██║                             ║
║     ██║   ██║     ██║  ██║██║███╗██║                             ║
║     ██║   ███████╗██████╔╝╚███╔███╔╝                             ║
║     ╚═╝   ╚══════╝╚═════╝  ╚══╝╚══╝                              ║
║                                                                  ║
║           too long; didn't watch                                 ║
║                  chatbook                                        ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    def __init__(
        self,
        *,
        card_name: Optional[str] = None,
        duration: float = 1.5,
        skip_on_keypress: bool = True,
        show_progress: bool = True,
        **kwargs
    ) -> None:
        """Initialize the splash screen.
        
        Args:
            card_name: Name of the splash card to use (None for random)
            duration: How long to display the splash screen
            skip_on_keypress: Whether to allow skipping with a keypress
            show_progress: Whether to show progress bar
        """
        # Ensure we have proper classes set
        if 'classes' not in kwargs:
            kwargs['classes'] = self.DEFAULT_CLASSES
        super().__init__(**kwargs)
        
        # Load configuration
        self.config = self._load_splash_config()
        
        # Override with parameters if provided
        self.duration = duration
        self.skip_on_keypress = skip_on_keypress
        self.show_progress = show_progress
        
        # Animation state
        self.start_time = time.time()
        self.animation_timer: Optional[Timer] = None
        self.fade_timer: Optional[Timer] = None
        self.auto_close_timer: Optional[Timer] = None
        self.effect_handler: Optional[Any] = None
        
        # Select splash card
        self.card_name = card_name or self._select_card()
        self.card_data = self._load_card(self.card_name)
        
        # Skip state
        self._skip_requested = False
        
        # SplashScreen initialized
    
    def _load_splash_config(self) -> Dict[str, Any]:
        """Load splash screen configuration from settings."""
        default_config = {
            "enabled": True,
            "duration": 2.5,
            "skip_on_keypress": True,
            "card_selection": "random",
            "show_progress": True,
            "fade_in_duration": 0.3,
            "fade_out_duration": 0.2,
            "animation_speed": 1.0,
            "active_cards": [
                "default", "matrix", "glitch", "retro", # Original
                "tech_pulse", "code_scroll", "minimal_fade", "blueprint", "arcade_high_score", # Batch 1
                "digital_rain", "loading_bar", "starfield", "terminal_boot", "glitch_reveal", # Batch 2
                "ascii_morph", "game_of_life", "scrolling_credits", "spotlight_reveal", "sound_bars", # Batch 3
                "raindrops_pond", "pixel_zoom", "text_explosion", "old_film", "maze_generator", # Batch 4 ("Crazy")
                "dwarf_fortress", # Batch 5 ("Fantasy")
                "waveforms", "neural_net", "quantum", "ascii_wave", "binary_matrix", # Batch 6 (Tech)
                "constellation", "news_ticker", "dna_sequence", "circuit_trace", "plasma_field", # Batch 7
                "ascii_fire", "rubiks_cube", "data_stream", "fractal_zoom", "spinner", # Batch 8
                "hacker_terminal", "cyberpunk_glitch", "ascii_mandala", "holographic", "quantum_tunnel", # Batch 9
                "chaotic_typewriter", "spy_vs_spy", "phonebooths", "emoji_face", "custom_logo", # Batch 10
                "aquarium", "bookshelf", "train_journey", "clock_mechanism", "weather", # Batch 11
                "music_visualizer", "origami", "ant_colony", "neon_sign", "zen_garden", # Batch 12
                "doom_fire", "pacman", "space_invaders", "tetris", "character_select", # Gaming 1
                "achievement_unlocked", "versus_screen", "world_map", "level_up", "retro_gaming_intro", # Gaming 2
                "psychedelic_mandala", "lava_lamp", "kaleidoscope", "deep_dream", # Psychedelic 1
                "trippy_tunnel", "melting_screen", "shroom_vision", "hypno_swirl", "electric_sheep" # Psychedelic 2
            ]
        }
        
        # Try to load from config
        try:
            config = get_cli_setting("splash_screen", default_config)
            
            # Merge with defaults for any missing keys
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            
            return config
        except Exception as e:
            logger.error(f"Failed to load splash screen config: {e}")
            return default_config
    
    def _select_card(self) -> str:
        """Select which splash card to use based on config."""
        card_selection = self.config.get("card_selection", "random")
        
        if card_selection == "random":
            # Pick random from active cards
            active_cards = self.config.get("active_cards", ["default"])
            return random.choice(active_cards)
        elif card_selection == "sequential":
            # TODO: Implement sequential selection with state tracking
            active_cards = self.config.get("active_cards", ["default"])
            return active_cards[0]
        else:
            # Specific card name
            return card_selection
    
    def _load_card(self, card_name: str) -> Dict[str, Any]:
        """Load card data from configuration."""
        # Start with predefined cards
        cards = self._get_predefined_cards()
        
        # Note: get_splash_card_config() requires a name parameter,
        # so we can't bulk load custom cards this way anymore
        
        # Return the requested card or default
        if card_name in cards:
            return cards[card_name]
        else:
            logger.warning(f"Splash card '{card_name}' not found, using default")
            return cards.get("default", {
                "type": "static",
                "content": self.DEFAULT_SPLASH,
                "style": "bold white on black"
            })
    
    def _get_predefined_cards(self) -> Dict[str, Dict[str, Any]]:
        """Get all predefined splash cards."""
        cards = get_all_card_definitions()
        
        # Handle custom image special case
        if "custom_image" in cards:
            custom_image_path = get_cli_setting("splash_screen.custom_image_path", None, "")
            if custom_image_path:
                cards["custom_image"]["image_path"] = custom_image_path
        
        return cards
    
    def compose(self) -> ComposeResult:
        """Compose the splash screen layout."""
        # Directly yield the display widget without containers
        yield Static("", id="splash-display", classes="splash-display")
        
        # Progress bar (if enabled)
        if self.show_progress:
            yield ProgressBar(
                total=100,
                show_eta=False,
                show_percentage=True,
                id="splash-progress",
                classes="splash-progress"
            )
            yield Label(
                self.progress_text,
                id="splash-progress-text",
                classes="splash-progress-text"
            )
    
    async def on_mount(self) -> None:
        """Handle mount event."""
        logger.info(f"Splash screen mounted - duration: {self.duration}s, card: {self.card_name}")
        # Start the splash screen
        await self._start_splash()
        
        # Schedule auto-close
        if self.duration > 0:
            self.auto_close_timer = self.set_timer(
                self.duration,
                self._request_close
            )
            logger.info(f"Splash screen auto-close timer set for {self.duration}s")
    
    async def _start_splash(self) -> None:
        """Start the splash screen display."""
        # Check card type
        card_type = self.card_data.get("type", "static")
        logger.info(f"Starting splash screen - type: {card_type}, card: {self.card_name}")
        
        if card_type == "static":
            # Display static content
            content = self.card_data.get("content", self.DEFAULT_SPLASH)
            style = self.card_data.get("style", "bold white on black")
            logger.info(f"Setting static content: {len(content)} chars with style: {style}")
            
            display = self.query_one("#splash-display", Static)
            # Try updating with plain string instead of Text object
            display.update(content)
            
        elif card_type == "animated":
            # Start animation
            self._start_animation()
    
    def _start_animation(self) -> None:
        """Start animation based on card type and effect."""
        effect_type = self.card_data.get("effect")
        logger.debug(f"Starting animation with effect: {effect_type}")
        
        if effect_type:
            # Get the effect class from the registry
            effect_class = get_effect_class(effect_type)
            
            if effect_class:
                # Get terminal size
                width, height = self._get_terminal_size()
                logger.debug(f"Terminal size: {width}x{height}")
                
                # Create effect handler with card data as kwargs
                try:
                    self.effect_handler = effect_class(
                        self,
                        width=width,
                        height=height,
                        **self.card_data
                    )
                    
                    # Start animation timer
                    self.animation_timer = self.set_interval(
                        self.card_data.get("animation_speed", 0.05),
                        self._update_animation
                    )
                    logger.debug(f"Animation started successfully for {effect_type}")
                except Exception as e:
                    logger.error(f"Failed to create effect {effect_type}: {e}")
                    self._display_static_fallback()
            else:
                logger.warning(f"Unknown effect type: {effect_type}")
                # Fall back to static display
                self._display_static_fallback()
        else:
            # No effect specified
            self._display_static_fallback()
    
    def _display_static_fallback(self) -> None:
        """Display static content as fallback."""
        content = self.card_data.get("content", self.DEFAULT_SPLASH)
        style = self.card_data.get("style", "bold white on black")
        logger.info(f"Displaying static fallback with {len(content)} chars")
        
        display = self.query_one("#splash-display", Static)
        # Use plain string for fallback too
        display.update(content)
    
    def _get_terminal_size(self) -> Tuple[int, int]:
        """Get the current terminal size."""
        try:
            # Try to get from display widget
            display = self.query_one("#splash-display", Static)
            width, height = display.size.width, display.size.height
            # Ensure we have valid dimensions
            if width > 0 and height > 0:
                return width, height
        except:
            pass
        
        # Try to get app size
        if hasattr(self, 'app') and self.app:
            width = self.app.size.width
            height = self.app.size.height
            if width > 0 and height > 0:
                return width, height
        
        # Fallback to defaults
        return 80, 24
    
    def _update_animation(self) -> None:
        """Update animation frame."""
        if self.effect_handler:
            try:
                # Get next frame from effect
                frame_content = self.effect_handler.update()
                
                if frame_content:
                    # Update display
                    display = self.query_one("#splash-display", Static)
                    display.update(frame_content)
                
                self.current_frame += 1
            except Exception as e:
                logger.error(f"Error updating animation frame: {e}")
                # Stop animation and show static fallback
                if self.animation_timer:
                    self.animation_timer.stop()
                self._display_static_fallback()
    
    def update_progress(self, value: float, text: str = "") -> None:
        """Update progress bar and text.
        
        Args:
            value: Progress value (0-100)
            text: Progress text to display
        """
        self.progress = value
        if text:
            self.progress_text = text
        
        # Update progress bar if it exists
        try:
            progress_bar = self.query_one("#splash-progress", ProgressBar)
            progress_bar.update(progress=value)
            
            if text:
                progress_text = self.query_one("#splash-progress-text", Label)
                progress_text.update(text)
        except:
            pass
    
    async def on_key(self, event: events.Key) -> None:
        """Handle key press events."""
        if self.skip_on_keypress and not self._skip_requested:
            event.stop()
            event.prevent_default()
            self._request_close()
    
    def _request_close(self) -> None:
        """Request the splash screen to close."""
        if self._skip_requested:
            return
            
        self._skip_requested = True
        logger.info("Splash screen close requested")
        
        # Cancel timers
        if self.animation_timer:
            self.animation_timer.stop()
        if self.auto_close_timer:
            self.auto_close_timer.stop()
        
        # Start fade out
        self._start_fade_out()
    
    def _start_fade_out(self) -> None:
        """Start fade out animation."""
        # For now, just close immediately
        # TODO: Implement actual fade animation
        self.close()
    
    def close(self) -> None:
        """Close the splash screen."""
        self.is_active = False
        logger.info("Splash screen closing")
        
        # Stop any running timers
        if self.animation_timer:
            self.animation_timer.stop()
        if self.fade_timer:
            self.fade_timer.stop()
        if self.auto_close_timer:
            self.auto_close_timer.stop()
        
        # Post close event
        self.post_message(self.Closed())
        logger.info("Splash screen Closed message posted")
    
    class Closed(Message):
        """Message sent when splash screen closes."""
        pass


# For backward compatibility
class SplashScreenClosed(events.Event):
    """Event fired when splash screen closes."""
    pass