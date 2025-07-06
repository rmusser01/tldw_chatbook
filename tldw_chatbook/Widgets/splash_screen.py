# splash_screen.py
# Customizable splash screen widget for tldw_chatbook startup
# Supports static and animated splash screens with Call of Duty-style "calling cards"

import asyncio
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

from textual.app import ComposeResult
from textual.containers import Container, Center, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static, ProgressBar, Label
from textual.timer import Timer
from textual import events
from rich.text import Text
from rich.align import Align
from rich.console import Console
from rich.style import Style

from loguru import logger

from ..Utils.Splash_Strings import splashscreen_message_selection
from ..config import get_cli_setting
from ..Utils.Splash import get_ascii_art, get_splash_card_config
from ..Utils.splash_animations import (
    MatrixRainEffect,
    GlitchEffect,
    TypewriterEffect,
    FadeEffect,
    RetroTerminalEffect
)

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
║           Too Long; Didn't Watch aka chatbook                    ║
║                                                                  ║
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
        
        logger.debug(f"SplashScreen initialized with card: {self.card_name}")
    
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
            "active_cards": ["default", "matrix", "glitch", "retro"]
        }
        
        # Get config from settings
        splash_config = get_cli_setting("splash_screen", None, {})
        
        # Merge with defaults
        config = {**default_config, **splash_config}
        
        # Get effects config
        effects_config = get_cli_setting("splash_screen.effects", None, {})
        config.update(effects_config)
        
        return config
    
    def _select_card(self) -> str:
        """Select which splash card to display based on configuration."""
        selection_mode = self.config.get("card_selection", "random")
        active_cards = self.config.get("active_cards", ["default"])
        
        if selection_mode == "random" and active_cards:
            return random.choice(active_cards)
        elif selection_mode == "sequential":
            # TODO: Implement sequential selection with persistence
            return active_cards[0] if active_cards else "default"
        elif selection_mode in active_cards:
            # Specific card selected
            return selection_mode
        else:
            return "default"
    
    def _load_card(self, card_name: str) -> Dict[str, Any]:
        """Load splash card data from file or return built-in card."""
        # Check for custom card file
        card_dir = Path.home() / ".config" / "tldw_cli" / "splash_cards"
        card_file = card_dir / f"{card_name}.toml"
        
        if card_file.exists():
            # TODO: Load custom card from TOML file
            pass
        
        # Check if it's a classic ASCII art card
        if card_name in ["classic", "compact", "minimal"]:
            return get_splash_card_config(card_name)
        
        # Return built-in cards
        built_in_cards = {
            "default": {
                "type": "static",
                "content": self.DEFAULT_SPLASH,
                "style": "bold white on rgb(0,0,0)",
                "effect": None
            },
            "matrix": {
                "type": "animated",
                "effect": "matrix_rain",
                "title": "tldw chatbook",
                "subtitle": (f"Loading user interface...{splashscreen_message_selection}"),
                "style": "bold green on black",
                "animation_speed": 0.05
            },
            "glitch": {
                "type": "animated", 
                "effect": "glitch",
                "content": self.DEFAULT_SPLASH,
                "style": "bold white on black",
                "glitch_chars": "!@#$%^&*()_+-=[]{}|;:,.<>?",
                "animation_speed": 0.1
            },
            "retro": {
                "type": "animated",
                "effect": "retro_terminal",
                "content": self.DEFAULT_SPLASH,
                "style": "bold green on black",
                "scanline_speed": 0.02,
                "phosphor_glow": True
            }
        }

        card_selection = random.choice(list(built_in_cards.keys()))
        return built_in_cards.get(card_name, built_in_cards[card_selection])
    
    def compose(self) -> ComposeResult:
        """Compose the splash screen layout."""
        logger.debug(f"Composing splash screen with card: {self.card_name}")
        
        with Center(id="splash-center"):
            with Vertical(id="splash-content"):
                # Main content area
                initial_content = self._render_initial_content()
                logger.debug(f"Initial content length: {len(initial_content)}")
                
                yield Static(
                    initial_content,
                    id="splash-main",
                    classes="splash-main"
                )
                
                # Progress section
                if self.show_progress:
                    yield Label(
                        self.progress_text,
                        id="splash-progress-text",
                        classes="splash-progress-text"
                    )
                    yield ProgressBar(
                        total=100,
                        id="splash-progress-bar",
                        classes="splash-progress-bar"
                    )
    
    def _render_initial_content(self) -> str:
        """Render the initial content based on card type."""
        if self.card_data["type"] == "static":
            return self.card_data.get("content", "Loading...")
        else:
            # For animated cards, return empty initially
            return ""
    
    def on_mount(self) -> None:
        """Start animations when mounted."""
        logger.debug(f"SplashScreen mounted, starting animations. Card: {self.card_name}, Duration: {self.duration}")
        logger.info(f"Splash screen is now visible with card: {self.card_name}")
        
        # Apply fade-in effect
        fade_in_duration = self.config.get("fade_in_duration", 0.3)
        if fade_in_duration > 0:
            self.styles.opacity = 0.0
            self.styles.animate("opacity", 1.0, duration=fade_in_duration)
        else:
            self.styles.opacity = 1.0
        
        # Start card-specific animations
        self._start_card_animation()
        
        # Schedule auto-close
        logger.debug(f"Scheduling auto-close in {self.duration} seconds")
        self.auto_close_timer = self.set_timer(self.duration, self._request_close)
    
    def _start_card_animation(self) -> None:
        """Start animation based on card type and effect."""
        effect_type = self.card_data.get("effect")
        
        if effect_type == "matrix_rain":
            self.effect_handler = MatrixRainEffect(
                self,
                title=self.card_data.get("title", "TLDW CLI"),
                subtitle=self.card_data.get("subtitle", ""),
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "glitch":
            self.effect_handler = GlitchEffect(
                self,
                content=self.card_data.get("content", self.DEFAULT_SPLASH),
                glitch_chars=self.card_data.get("glitch_chars", "!@#$%^&*"),
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "retro_terminal":
            self.effect_handler = RetroTerminalEffect(
                self,
                content=self.card_data.get("content", self.DEFAULT_SPLASH),
                scanline_speed=self.card_data.get("scanline_speed", 0.02),
                phosphor_glow=self.card_data.get("phosphor_glow", True)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("scanline_speed", 0.02),
                self._update_animation
            )
    
    def _update_animation(self) -> None:
        """Update animation frame."""
        if self.effect_handler and hasattr(self.effect_handler, "update"):
            content = self.effect_handler.update()
            if content:
                try:
                    main_widget = self.query_one("#splash-main", Static)
                    main_widget.update(content)
                except Exception as e:
                    logger.error(f"Error updating splash animation: {e}")
    
    def update_progress(self, value: float, text: Optional[str] = None) -> None:
        """Update progress indicator.
        
        Args:
            value: Progress value between 0.0 and 1.0
            text: Optional progress text
        """
        self.progress = max(0.0, min(1.0, value))
        
        if text:
            self.progress_text = text
        
        if self.show_progress:
            try:
                progress_bar = self.query_one("#splash-progress-bar", ProgressBar)
                progress_bar.update(progress=self.progress * 100)
                
                if text:
                    progress_label = self.query_one("#splash-progress-text", Label)
                    progress_label.update(text)
            except:
                pass  # Widgets might not be ready yet
    
    async def on_key(self, event: events.Key) -> None:
        """Handle keypress to skip splash screen."""
        if self.skip_on_keypress and self.is_active:
            logger.debug(f"Key pressed during splash: {event.key}, requesting skip")
            self._request_close()
            event.stop()
    
    def _request_close(self) -> None:
        """Request splash screen to close."""
        if self._skip_requested:
            return
            
        self._skip_requested = True
        logger.debug("Splash screen close requested")
        
        # Stop animations
        if self.animation_timer:
            self.animation_timer.stop()
        
        # Fade out
        fade_out_duration = self.config.get("fade_out_duration", 0.2)
        if fade_out_duration > 0:
            self.styles.animate(
                "opacity",
                0.0,
                duration=fade_out_duration,
                on_complete=self._finish_close
            )
        else:
            self._finish_close()
    
    def _finish_close(self) -> None:
        """Complete the splash screen close."""
        self.is_active = False
        logger.debug("Splash screen closing")
        
        # Notify parent app that splash is done
        self.post_message(SplashScreenClosed())
        
        # Don't remove immediately - let the app handle removal
        # self.remove()


class SplashScreenClosed(events.Event):
    """Event fired when splash screen closes."""
    pass