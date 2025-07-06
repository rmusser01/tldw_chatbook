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
    RetroTerminalEffect,
    PulseEffect,
    CodeScrollEffect,
    BlinkEffect,
    DigitalRainEffect,
    LoadingBarEffect,
    StarfieldEffect,
    TerminalBootEffect,
    GlitchRevealEffect
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
            "active_cards": [
                "default", "matrix", "glitch", "retro",
                "tech_pulse", "code_scroll", "minimal_fade", "blueprint", "arcade_high_score", # Previous batch
                "digital_rain", "loading_bar", "starfield", "terminal_boot", "glitch_reveal" # Current batch
            ]
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
            },
            "tech_pulse": {
                "type": "animated",
                "effect": "pulse",
                "content": get_ascii_art("tech_pulse"),
                "style": "bold white on black", # Base style, effect will override color
                "pulse_speed": 0.5, # Cycles per second
                "min_brightness": 80,
                "max_brightness": 200,
                "color": (100, 180, 255) # A light blue
            },
            "code_scroll": {
                "type": "animated",
                "effect": "code_scroll",
                "title": "TLDW CHATBOOK",
                "subtitle": f"{splashscreen_message_selection}",
                "style": "white on black", # General style, effect handles specifics
                "scroll_speed": 0.1,
                "num_code_lines": 18, # Adjusted for typical 24 line height
                "code_line_style": "dim blue",
                "title_style": "bold yellow",
                "subtitle_style": "green"
            },
            "minimal_fade": {
                "type": "animated",
                "effect": "typewriter", # Using typewriter for a slow reveal
                "content": get_ascii_art("minimal_fade"),
                "style": "white on black",
                "animation_speed": 0.08, # Controls typewriter speed
            },
            "blueprint": {
                "type": "static",
                "content": get_ascii_art("blueprint"),
                "style": "bold cyan on rgb(0,0,30)", # Cyan on dark blue
                "effect": None
            },
            "arcade_high_score": {
                "type": "animated",
                "effect": "blink",
                "content": get_ascii_art("arcade_high_score"),
                "style": "bold yellow on rgb(0,0,70)", # Yellow on dark blue
                "animation_speed": 0.1, # Interval for the animation timer
                "blink_speed": 0.5,     # How long each blink state (on/off) lasts
                "blink_targets": ["LOADING...", "PRESS ANY KEY TO START!"],
                "blink_style_off": "dim", # How targets look when "off"
            },
            "digital_rain": {
                "type": "animated",
                "effect": "digital_rain",
                "title": "TLDW CHATBOOK v2.0",
                "subtitle": f"Enhancing neural pathways... {splashscreen_message_selection}",
                "style": "white on black", # Base, effect controls most styling
                "animation_speed": 0.05, # Interval for animation timer
                "base_chars": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "highlight_chars": "!@#$%^*()-+=[]{};:,.<>/?",
                "base_color": "dim green",
                "highlight_color": "bold green",
                "title_style": "bold magenta",
                "subtitle_style": "cyan",
                "highlight_chance": 0.05
            },
            "loading_bar": {
                "type": "animated",
                "effect": "loading_bar",
                "content": get_ascii_art("loading_bar_frame"), # Base frame
                "style": "white on black", # General style for text
                "animation_speed": 0.1, # How often effect's update() is called
                "fill_char": "█",
                "bar_style": "bold green",
                "text_above": "SYSTEM INITIALIZATION SEQUENCE",
                # text_below will use SplashScreen.progress_text by default from effect
                # or "text_below_template": "{progress:.0f}% Synced" can be set here
            },
            "starfield": {
                "type": "animated",
                "effect": "starfield",
                "title": "Hyperdrive Initializing...",
                "style": "black on black", # Background, stars provide visuals
                "animation_speed": 0.04, # Faster updates for smoother stars
                "num_stars": 200,
                "warp_factor": 0.25,
                "max_depth": 40.0,
                "star_chars": ["·", ".", "*", "+"],
                "star_styles": ["dim white", "white", "bold white", "bold yellow"],
                "title_style": "bold cyan"
            },
            "terminal_boot": {
                "type": "animated",
                "effect": "terminal_boot",
                "style": "green on black", # Default style for lines
                "animation_speed": 0.03, # Interval for animation timer (effects internal timing)
                "cursor": "▋",
                "boot_sequence": [
                    {"text": "TLDW BIOS v4.2.1 initializing...", "type_speed": 0.02, "pause_after": 0.3, "style": "bold white"},
                    {"text": "Memory Test: 65536 KB OK", "delay_before": 0.1, "type_speed": 0.01, "pause_after": 0.2},
                    {"text": "Detecting CPU Type: Quantum Entangled Processor", "type_speed": 0.02, "pause_after": 0.1},
                    {"text": "Initializing USB Controllers ... Done.", "type_speed": 0.015, "pause_after": 0.2},
                    {"text": "Loading TL-DOS...", "delay_before": 0.3, "type_speed": 0.04, "pause_after": 0.3, "style": "yellow"},
                    {"text": "Starting services:", "type_speed": 0.02, "pause_after": 0.1},
                    {"text": "  Network Stack .............. [OK]", "delay_before": 0.2, "type_speed": 0.01, "style": "dim green"},
                    {"text": "  AI Core Diagnostics ........ [OK]", "type_speed": 0.01, "style": "dim green"},
                    {"text": "  Sarcasm Module ............. [ENABLED]", "type_speed": 0.01, "style": "dim green"},
                    {"text": f"Welcome to TLDW Chatbook - {splashscreen_message_selection}", "delay_before": 0.5, "type_speed": 0.03, "style": "bold cyan"}
                ]
            },
            "glitch_reveal": {
                "type": "animated",
                "effect": "glitch_reveal",
                "content": get_ascii_art("app_logo_clear"), # The final, clear logo
                "style": "bold white on black", # Base style for the final clear logo
                "animation_speed": 0.05, # Interval for animation timer
                "duration": 2.5, # Total duration for the reveal effect itself
                "glitch_chars": "!@#$%^&*▓▒░█",
                "start_intensity": 0.9,
                "end_intensity": 0.0
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
        elif effect_type == "pulse":
            self.effect_handler = PulseEffect(
                self,
                content=self.card_data.get("content", ""),
                pulse_speed=self.card_data.get("pulse_speed", 0.5),
                min_brightness=self.card_data.get("min_brightness", 100),
                max_brightness=self.card_data.get("max_brightness", 255),
                color=self.card_data.get("color", (255, 255, 255))
            )
            # Use a reasonable interval for smooth animation, effect handles timing by elapsed time
            self.animation_timer = self.set_interval(0.05, self._update_animation)
        elif effect_type == "code_scroll":
            # Get width and height from the splash screen main area if possible, else default
            main_widget = self.query_one("#splash-main", Static)
            width, height = main_widget.content_size

            self.effect_handler = CodeScrollEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                subtitle=self.card_data.get("subtitle", splashscreen_message_selection),
                width=width if width > 0 else 80,
                height=height if height > 0 else 24,
                scroll_speed=self.card_data.get("scroll_speed", 0.1),
                num_code_lines=self.card_data.get("num_code_lines", 15),
                code_line_style=self.card_data.get("code_line_style", "dim cyan"),
                title_style=self.card_data.get("title_style", "bold white"),
                subtitle_style=self.card_data.get("subtitle_style", "white")
            )
            # Interval should align with or be faster than effect's internal scroll_speed
            self.animation_timer = self.set_interval(
                min(0.1, self.card_data.get("scroll_speed", 0.1)),
                self._update_animation
            )
        elif effect_type == "blink":
            self.effect_handler = BlinkEffect(
                self,
                content=self.card_data.get("content", ""),
                blink_speed=self.card_data.get("blink_speed", 0.5),
                blink_targets=self.card_data.get("blink_targets", []),
                blink_style_off=self.card_data.get("blink_style_off", "dim")
                # blink_style_on could be added if needed
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1), # Animation timer interval
                self._update_animation
            )
        elif effect_type == "digital_rain":
            main_widget = self.query_one("#splash-main", Static)
            width, height = main_widget.content_size
            self.effect_handler = DigitalRainEffect(
                self,
                title=self.card_data.get("title", "Digital Rain"),
                subtitle=self.card_data.get("subtitle", splashscreen_message_selection),
                width=width if width > 0 else 80,
                height=height if height > 0 else 24,
                speed=self.card_data.get("animation_speed", 0.05),
                base_chars=self.card_data.get("base_chars", "abc0123"),
                highlight_chars=self.card_data.get("highlight_chars", "!@#$"),
                base_color=self.card_data.get("base_color", "dim green"),
                highlight_color=self.card_data.get("highlight_color", "bold green"),
                title_style=self.card_data.get("title_style", "bold white"),
                subtitle_style=self.card_data.get("subtitle_style", "white"),
                highlight_chance=self.card_data.get("highlight_chance", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "loading_bar":
            main_widget = self.query_one("#splash-main", Static)
            width, _ = main_widget.content_size # Only width needed for centering text
            self.effect_handler = LoadingBarEffect(
                self, # Pass SplashScreen instance as parent
                bar_frame_content=self.card_data.get("content", "[----------]"),
                fill_char=self.card_data.get("fill_char", "#"),
                bar_style=self.card_data.get("bar_style", "bold green"),
                text_above=self.card_data.get("text_above", "LOADING..."),
                text_below=self.card_data.get("text_below_template", ""), # Effect defaults to progress_text
                text_style=self.card_data.get("text_style", "white"),
                width=width if width > 0 else 80
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "starfield":
            main_widget = self.query_one("#splash-main", Static)
            width, height = main_widget.content_size
            self.effect_handler = StarfieldEffect(
                self,
                title=self.card_data.get("title", "Warping..."),
                num_stars=self.card_data.get("num_stars", 150),
                warp_factor=self.card_data.get("warp_factor", 0.2),
                max_depth=self.card_data.get("max_depth", 50.0),
                star_chars=self.card_data.get("star_chars", ["."]),
                star_styles=self.card_data.get("star_styles", ["white"]),
                width=width if width > 0 else 80,
                height=height if height > 0 else 24,
                title_style=self.card_data.get("title_style", "bold yellow")
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "terminal_boot":
            main_widget = self.query_one("#splash-main", Static)
            width, height = main_widget.content_size
            self.effect_handler = TerminalBootEffect(
                self,
                boot_sequence=self.card_data.get("boot_sequence", [{"text": "Booting..."}]),
                width=width if width > 0 else 80,
                height=height if height > 0 else 24,
                cursor=self.card_data.get("cursor", "_")
            )
            # The effect's internal timing is driven by elapsed time,
            # so timer interval just needs to be frequent enough for smooth typing.
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.03),
                self._update_animation
            )
        elif effect_type == "glitch_reveal":
            self.effect_handler = GlitchRevealEffect(
                self,
                content=self.card_data.get("content", "REVEAL!"),
                duration=self.card_data.get("duration", 2.0),
                glitch_chars=self.card_data.get("glitch_chars", "!@#$%^&*"),
                start_intensity=self.card_data.get("start_intensity", 0.8),
                end_intensity=self.card_data.get("end_intensity", 0.0)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
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