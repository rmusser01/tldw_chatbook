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
    GlitchRevealEffect,
    AsciiMorphEffect,
    GameOfLifeEffect,
    ScrollingCreditsEffect,
    SpotlightEffect,
    SoundBarsEffect,
    RaindropsEffect,
    PixelZoomEffect,
    TextExplosionEffect,
    OldFilmEffect,
    MazeGeneratorEffect,
    MiningEffect,
    NeuralNetworkEffect,
    QuantumParticlesEffect,
    ASCIIWaveEffect,
    BinaryMatrixEffect,
    ConstellationMapEffect,
    TypewriterNewsEffect,
    DNASequenceEffect,
    CircuitTraceEffect,
    PlasmaFieldEffect,
    ASCIIFireEffect,
    RubiksCubeEffect,
    DataStreamEffect,
    FractalZoomEffect,
    ASCIISpinnerEffect,
    HackerTerminalEffect,
    CyberpunkGlitchEffect,
    ASCIIMandalaEffect,
    HolographicInterfaceEffect,
    QuantumTunnelEffect,
    ChaoticTypewriterEffect,
    SpyVsSpyEffect,
    PhoneboothsDialingEffect,
    EmojiFaceEffect,
    CustomImageEffect,
    ASCIIAquariumEffect,
    BookshelfBrowserEffect,
    TrainJourneyEffect,
    ClockMechanismEffect,
    WeatherSystemEffect,
    MusicVisualizerEffect,
    OrigamiFoldingEffect,
    AntColonyEffect,
    NeonSignFlickerEffect,
    ZenGardenEffect
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
                "default", "matrix", "glitch", "retro", # Original
                "tech_pulse", "code_scroll", "minimal_fade", "blueprint", "arcade_high_score", # Batch 1
                "digital_rain", "loading_bar", "starfield", "terminal_boot", "glitch_reveal", # Batch 2
                "ascii_morph", "game_of_life", "scrolling_credits", "spotlight_reveal", "sound_bars", # Batch 3
                "raindrops_pond", "pixel_zoom", "text_explosion", "old_film", "maze_generator", # Batch 4 ("Crazy")
                "dwarf_fortress", # Batch 5 ("Fantasy")
                # New 15 animated effects
                "neural_network", "quantum_particles", "ascii_wave", "binary_matrix", "constellation_map",
                "typewriter_news", "dna_sequence", "circuit_trace", "plasma_field", "ascii_fire",
                "rubiks_cube", "data_stream", "fractal_zoom", "ascii_spinner", "hacker_terminal",
                # Latest additions
                "spy_vs_spy", "phonebooths", "emoji_face"
            ]
        }

        # Get config from settings
        splash_config = get_cli_setting("splash_screen", None, {})
        if not splash_config:
            logger.debug("No splash screen config found in settings, using defaults")
            splash_config = {}

        # Merge with defaults, ensuring we don't override active_cards if it's empty
        if "active_cards" in splash_config and not splash_config["active_cards"]:
            splash_config.pop("active_cards")

        config = {**default_config, **splash_config}

        # Double check the critical settings
        if not config["active_cards"]:
            config["active_cards"] = default_config["active_cards"]
            logger.warning("Active cards list was empty, restored defaults")

        if config["card_selection"] not in ["random", "sequential"] and config["card_selection"] not in config[
            "active_cards"]:
            config["card_selection"] = "random"
            logger.warning("Invalid card_selection setting, defaulting to random")

        # Get effects config
        effects_config = get_cli_setting("splash_screen.effects", None, {})
        if effects_config:
            config.update(effects_config)

        logger.debug(f"Loaded splash config: {config}")
        return config

    def _select_card(self) -> str:
        """Select which splash card to display based on configuration."""
        selection_mode = self.config.get("card_selection", "random")
        active_cards = self.config.get("active_cards", ["default"])

        logger.debug(f"Splash card selection mode: {selection_mode}")
        logger.debug(f"Available active cards: {active_cards}")

        if not active_cards:
            logger.warning("No active cards available, falling back to default")
            return "default"

        if selection_mode == "random":
            # Make a random choice from active cards
            selected = random.choice(active_cards) if active_cards else "default"
            logger.debug(f"Randomly selected card: {selected}")
            return selected
        elif selection_mode == "sequential":
            # TODO: Implement sequential selection with persistence
            selected = active_cards[0] if active_cards else "default"
            logger.debug(f"Sequential selection, chose: {selected}")
            return selected
        elif selection_mode in active_cards:
            # Specific card selected
            logger.debug(f"Specific card selected: {selection_mode}")
            return selection_mode
        else:
            logger.debug("Invalid selection mode, falling back to default")
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
            },
            "ascii_morph": {
                "type": "animated",
                "effect": "ascii_morph",
                "style": "bold white on black",
                "animation_speed": 0.05, # Interval for animation timer
                "duration": 3.0, # Duration of the morph itself
                "start_art_name": "morph_art_start", # Key for get_ascii_art
                "end_art_name": "morph_art_end",     # Key for get_ascii_art
                "morph_style": "dissolve" # "dissolve", "random_pixel", "wipe_left_to_right"
            },
            "game_of_life": {
                "type": "animated",
                "effect": "game_of_life",
                "title": "Cellular Automata Initializing...",
                "style": "black on black", # Background, effect handles cell colors
                "animation_speed": 0.1,  # Timer for the effect's update method
                "gol_update_interval": 0.15, # How often GoL simulation steps
                "grid_width": 50,
                "grid_height": 18,
                "cell_alive_char": "■", # "█"
                "cell_dead_char": " ",
                "alive_style": "bold blue",
                "dead_style": "on black", # So spaces take background color
                "initial_pattern": "glider", # "random", "glider"
                "title_style": "bold yellow"
            },
            "scrolling_credits": {
                "type": "animated",
                "effect": "scrolling_credits",
                "title": "TLDW Chatbook", # This title is part of the scrolling credits
                "style": "white on black", # Overall background for the splash screen
                "animation_speed": 0.03, # How often the effect's update() is called
                "scroll_speed": 2.0, # Lines per second for credits scroll
                "line_spacing": 1,
                "title_style": "bold magenta",
                "role_style": "bold cyan",
                "name_style": "green",
                "line_style": "white",
                "credits_list": [
                    {"role": "Lead Developer", "name": "Jules AI"},
                    {"role": "ASCII Art Design", "name": "The Byte Smiths"},
                    {"role": "Animation Engine", "name": "Temporal Mechanics Inc."},
                    {"line": ""},
                    {"line": "Special Thanks To:"},
                    {"line": "All the Electrons"},
                    {"line": "The Coffee Machine"},
                    {"line": ""},
                    {"line": f"And You! (User: {splashscreen_message_selection})"}
                ]
            },
            "spotlight_reveal": {
                "type": "animated",
                "effect": "spotlight",
                "background_art_name": "spotlight_background", # Key for get_ascii_art
                "style": "dim black on black", # Base style for hidden areas
                "animation_speed": 0.05,
                "spotlight_radius": 7,
                "movement_speed": 15.0, # Cells per second for path calculation
                "path_type": "lissajous", # "lissajous", "random_walk", "circle"
                "visible_style": "bold yellow on black", # Style within spotlight
                "hidden_style": "rgb(30,30,30) on black" # Very dim for outside spotlight
            },
            "sound_bars": {
                "type": "animated",
                "effect": "sound_bars",
                "title": "Frequency Analysis Engaged",
                "style": "black on black", # Background, bars provide color
                "animation_speed": 0.05, # Timer for effect's update method
                "num_bars": 25,
                "bar_char_filled": "┃", # Alternatives: █ ▓ ▒ ░ ║ ┃
                "bar_char_empty": " ",
                "bar_styles": ["bold blue", "bold magenta", "bold cyan", "bold green", "bold yellow", "bold red", "bold white"],
                "title_style": "bold white",
                "update_speed": 0.05 # How fast bars change their heights
            },
            "raindrops_pond": {
                "type": "animated",
                "effect": "raindrops", # Effect name needs to be consistent
                "title": "TLDW Reflections",
                "style": "dim blue on rgb(10,20,40)", # Base water color
                "animation_speed": 0.05,
                "spawn_rate": 2.0, # Drops per second
                "ripple_chars": ["·", "o", "O", "()"],
                "ripple_styles": ["blue", "cyan", "bold cyan", "dim blue"],
                "max_concurrent_ripples": 20,
                "base_water_char": "~",
                "water_style": "dim blue",
                "title_style": "italic bold white on rgb(20,40,80)"
            },
            "pixel_zoom": {
                "type": "animated",
                "effect": "pixel_zoom",
                "target_art_name": "pixel_art_target", # Key for get_ascii_art
                "style": "bold white on black", # Style for the final resolved art
                "animation_speed": 0.05,
                "duration": 3.0, # Duration of the zoom/pixelation effect
                "max_pixel_size": 10,
                "effect_type": "resolve" # "resolve" or "pixelate"
            },
            "text_explosion": {
                "type": "animated",
                "effect": "text_explosion",
                "text_to_animate": "T . L . D . W", # Text for the effect
                "style": "black on black", # Background, chars provide color
                "animation_speed": 0.03,
                "duration": 2.0, # Duration of explosion/implosion
                "effect_direction": "implode", # "explode" or "implode"
                "char_style": "bold orange",
                "particle_spread": 40.0
            },
            "old_film": {
                "type": "animated",
                "effect": "old_film",
                 # Frames can be provided as a list of strings (each a full ASCII art)
                 # or list of names to be fetched by get_ascii_art.
                 # For simplicity, using one generic frame by name here.
                "frames_art_names": ["film_generic_frame"], # Could be ["frame1_name", "frame2_name"]
                "style": "white on black", # This is the Textual widget style, effect applies its own base_style
                "animation_speed": 0.1, # How often effect updates
                "frame_duration": 0.8, # How long each film frame shows
                "shake_intensity": 1,
                "grain_density": 0.07,
                "grain_chars": ".:·'",
                "film_base_style": "sandy_brown on black" # Style applied by the effect to content
            },
            "maze_generator": {
                "type": "animated",
                "effect": "maze_generator",
                "title": "Constructing Reality Tunnels...",
                "style": "black on black", # Background, maze provides visuals
                "animation_speed": 0.01, # Timer for effect's update (can be fast)
                "maze_width": 79, # Odd number for character grid
                "maze_height": 21, # Odd number
                "wall_char": "▓", # Alternatives: █ ░ ▒ ▓ ╬ ═ ║ ╔ ╗ ╚ ╝
                "path_char": " ",
                "cursor_char": "❖",
                "wall_style": "bold dim blue",
                "path_style": "on rgb(10,10,10)", # Dark background for path
                "cursor_style": "bold yellow",
                "title_style": "bold white",
                "generation_speed": 0.005 # Delay between maze generation steps (fast)
            },
            "dwarf_fortress": {
                "type": "animated",
                "effect": "mining",
                "content": get_ascii_art("dwarf_fortress"),
                "style": "rgb(139,69,19) on black", # Brown stone color
                "animation_speed": 0.08,
                "dig_speed": 0.6, # How fast the mining progresses
            },
            # New Animation Effects
            "neural_network": {
                "type": "animated",
                "effect": "neural_network",
                "title": "TLDW Chatbook",
                "subtitle": splashscreen_message_selection,
                "animation_speed": 0.1
            },
            "quantum_particles": {
                "type": "animated",
                "effect": "quantum_particles",
                "title": "TLDW Chatbook",
                "subtitle": "Quantum Computing Interface",
                "animation_speed": 0.05
            },
            "ascii_wave": {
                "type": "animated",
                "effect": "ascii_wave",
                "title": "TLDW Chatbook",
                "subtitle": "Riding the Wave of AI",
                "animation_speed": 0.1
            },
            "binary_matrix": {
                "type": "animated",
                "effect": "binary_matrix",
                "title": "TLDW",
                "animation_speed": 0.05
            },
            "constellation_map": {
                "type": "animated",
                "effect": "constellation_map",
                "title": "TLDW Chatbook",
                "animation_speed": 0.1
            },
            "typewriter_news": {
                "type": "animated",
                "effect": "typewriter_news",
                "animation_speed": 0.05
            },
            "dna_sequence": {
                "type": "animated",
                "effect": "dna_sequence",
                "title": "TLDW Chatbook",
                "animation_speed": 0.05
            },
            "circuit_trace": {
                "type": "animated",
                "effect": "circuit_trace",
                "title": "TLDW Chatbook",
                "animation_speed": 0.02
            },
            "plasma_field": {
                "type": "animated",
                "effect": "plasma_field",
                "title": "TLDW Chatbook",
                "animation_speed": 0.05
            },
            "ascii_fire": {
                "type": "animated",
                "effect": "ascii_fire",
                "title": "TLDW Chatbook",
                "animation_speed": 0.05
            },
            "rubiks_cube": {
                "type": "animated",
                "effect": "rubiks_cube",
                "title": "TLDW",
                "animation_speed": 0.5
            },
            "data_stream": {
                "type": "animated",
                "effect": "data_stream",
                "title": "TLDW Chatbook",
                "animation_speed": 0.02
            },
            "fractal_zoom": {
                "type": "animated",
                "effect": "fractal_zoom",
                "title": "TLDW Chatbook",
                "animation_speed": 0.05
            },
            "ascii_spinner": {
                "type": "animated",
                "effect": "ascii_spinner",
                "title": "Loading TLDW Chatbook",
                "animation_speed": 0.1
            },
            "hacker_terminal": {
                "type": "animated",
                "effect": "hacker_terminal",
                "title": "TLDW Chatbook",
                "animation_speed": 0.05
            },
            "cyberpunk_glitch": {
                "type": "animated",
                "effect": "cyberpunk_glitch",
                "title": "tldw chatbook",
                "subtitle": splashscreen_message_selection,
                "style": "black on black",
                "animation_speed": 0.05
            },
            "ascii_mandala": {
                "type": "animated",
                "effect": "ascii_mandala",
                "title": "tldw chatbook",
                "subtitle": splashscreen_message_selection,
                "style": "black on black",
                "animation_speed": 0.05
            },
            "holographic_interface": {
                "type": "animated",
                "effect": "holographic_interface",
                "title": "tldw chatbook",
                "subtitle": splashscreen_message_selection,
                "style": "black on black",
                "animation_speed": 0.05
            },
            "quantum_tunnel": {
                "type": "animated",
                "effect": "quantum_tunnel",
                "title": "tldw chatbook",
                "subtitle": splashscreen_message_selection,
                "style": "black on black",
                "animation_speed": 0.05
            },
            "chaotic_typewriter": {
                "type": "animated",
                "effect": "chaotic_typewriter",
                "title": "tldw chatbook",
                "subtitle": splashscreen_message_selection,
                "style": "black on black",
                "animation_speed": 0.03
            },
            "spy_vs_spy": {
                "type": "animated",
                "effect": "spy_vs_spy",
                "style": "white on black",
                "animation_speed": 0.1
            },
            "phonebooths": {
                "type": "animated",
                "effect": "phonebooths_dialing",
                "style": "white on black",
                "animation_speed": 0.1
            },
            "emoji_face": {
                "type": "animated",
                "effect": "emoji_face",
                "style": "white on black",
                "animation_speed": 0.05
            },
            "custom_image": {
                "type": "animated",
                "effect": "custom_image",
                "image_path": "",  # Will be overridden by config
                "style": "white on black",
                "animation_speed": 0.1
            },
            "ascii_aquarium": {
                "type": "animated",
                "effect": "ascii_aquarium",
                "style": "white on black",
                "animation_speed": 0.1
            },
            "bookshelf_browser": {
                "type": "animated",
                "effect": "bookshelf_browser",
                "style": "white on black",
                "animation_speed": 0.1
            },
            "train_journey": {
                "type": "animated",
                "effect": "train_journey",
                "style": "white on black",
                "animation_speed": 0.1
            },
            "clock_mechanism": {
                "type": "animated",
                "effect": "clock_mechanism",
                "style": "white on black",
                "animation_speed": 0.1
            },
            "weather_system": {
                "type": "animated",
                "effect": "weather_system",
                "style": "white on black",
                "animation_speed": 0.1
            },
            "music_visualizer": {
                "type": "animated",
                "effect": "music_visualizer",
                "style": "white on black",
                "animation_speed": 0.1
            },
            "origami_folding": {
                "type": "animated",
                "effect": "origami_folding",
                "style": "white on black",
                "animation_speed": 0.1
            },
            "ant_colony": {
                "type": "animated",
                "effect": "ant_colony",
                "style": "white on black",
                "animation_speed": 0.1
            },
            "neon_sign_flicker": {
                "type": "animated",
                "effect": "neon_sign_flicker",
                "style": "white on black",
                "animation_speed": 0.1
            },
            "zen_garden": {
                "type": "animated",
                "effect": "zen_garden",
                "style": "white on black",
                "animation_speed": 0.05
            }
        }

        # Check if custom image is requested and path is provided
        if card_name == "custom_image":
            custom_image_path = get_cli_setting("splash_screen.custom_image_path", None, "")
            if custom_image_path:
                built_in_cards["custom_image"]["image_path"] = custom_image_path
            else:
                # Fallback to default if no image path provided
                return built_in_cards["default"]

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
    
    def _get_terminal_size(self) -> Tuple[int, int]:
        """Get the terminal size for animations."""
        # Try to get the app's size first
        if hasattr(self, 'app') and self.app:
            width = self.app.size.width
            height = self.app.size.height
            # Account for progress bar if shown
            if self.show_progress:
                height = max(1, height - 4)  # Reserve space for progress text and bar
            return width, height
        
        # Fallback to content size
        try:
            main_widget = self.query_one("#splash-main", Static)
            width, height = main_widget.content_size
            if width > 0 and height > 0:
                return width, height
        except:
            pass
        
        # Default fallback
        return 80, 24
    
    def _start_card_animation(self) -> None:
        """Start animation based on card type and effect."""
        effect_type = self.card_data.get("effect")
        
        if effect_type == "matrix_rain":
            width, height = self._get_terminal_size()
            self.effect_handler = MatrixRainEffect(
                self,
                title=self.card_data.get("title", "TLDW CLI"),
                subtitle=self.card_data.get("subtitle", ""),
                width=width,
                height=height,
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
            width, height = self._get_terminal_size()
            self.effect_handler = CodeScrollEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                subtitle=self.card_data.get("subtitle", splashscreen_message_selection),
                width=width,
                height=height,
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
            width, height = self._get_terminal_size()
            self.effect_handler = DigitalRainEffect(
                self,
                title=self.card_data.get("title", "Digital Rain"),
                subtitle=self.card_data.get("subtitle", splashscreen_message_selection),
                width=width,
                height=height,
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
            width, _ = self._get_terminal_size() # Only width needed for centering text
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
            width, height = self._get_terminal_size()
            self.effect_handler = StarfieldEffect(
                self,
                title=self.card_data.get("title", "Warping..."),
                num_stars=self.card_data.get("num_stars", 150),
                warp_factor=self.card_data.get("warp_factor", 0.2),
                max_depth=self.card_data.get("max_depth", 50.0),
                star_chars=self.card_data.get("star_chars", ["."]),
                star_styles=self.card_data.get("star_styles", ["white"]),
                width=width,
                height=height,
                title_style=self.card_data.get("title_style", "bold yellow")
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "terminal_boot":
            width, height = self._get_terminal_size()
            self.effect_handler = TerminalBootEffect(
                self,
                boot_sequence=self.card_data.get("boot_sequence", [{"text": "Booting..."}]),
                width=width,
                height=height,
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
        elif effect_type == "ascii_morph":
            start_art = get_ascii_art(self.card_data.get("start_art_name", "default"))
            end_art = get_ascii_art(self.card_data.get("end_art_name", "default"))
            self.effect_handler = AsciiMorphEffect(
                self,
                start_content=start_art,
                end_content=end_art,
                duration=self.card_data.get("duration", 2.0),
                morph_style=self.card_data.get("morph_style", "dissolve")
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "game_of_life":
            display_width, display_height = self._get_terminal_size()
            self.effect_handler = GameOfLifeEffect(
                self,
                title=self.card_data.get("title", "Evolving..."),
                width=self.card_data.get("grid_width", 40),
                height=self.card_data.get("grid_height", 20),
                cell_alive_char=self.card_data.get("cell_alive_char", "█"),
                cell_dead_char=self.card_data.get("cell_dead_char", " "),
                alive_style=self.card_data.get("alive_style", "bold green"),
                dead_style=self.card_data.get("dead_style", "on black"),
                initial_pattern=self.card_data.get("initial_pattern", "random"),
                update_interval=self.card_data.get("gol_update_interval", 0.2),
                title_style=self.card_data.get("title_style", "bold white"),
                display_width=display_width if display_width > 0 else 80,
                display_height=display_height if display_height > 0 else 24
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "scrolling_credits":
            display_width, display_height = self._get_terminal_size()
            self.effect_handler = ScrollingCreditsEffect(
                self,
                title=self.card_data.get("title", "Credits"),
                credits_list=self.card_data.get("credits_list", []),
                scroll_speed=self.card_data.get("scroll_speed", 1.0),
                line_spacing=self.card_data.get("line_spacing", 1),
                width=display_width if display_width > 0 else 80,
                height=display_height if display_height > 0 else 24,
                title_style=self.card_data.get("title_style", "bold yellow"),
                role_style=self.card_data.get("role_style", "bold white"),
                name_style=self.card_data.get("name_style", "white"),
                line_style=self.card_data.get("line_style", "white")
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.03),
                self._update_animation
            )
        elif effect_type == "spotlight":
            display_width, display_height = self._get_terminal_size()
            background_art = get_ascii_art(self.card_data.get("background_art_name", "default"))
            self.effect_handler = SpotlightEffect(
                self,
                background_content=background_art,
                spotlight_radius=self.card_data.get("spotlight_radius", 5),
                movement_speed=self.card_data.get("movement_speed", 10.0),
                path_type=self.card_data.get("path_type", "lissajous"),
                visible_style=self.card_data.get("visible_style", "bold white"),
                hidden_style=self.card_data.get("hidden_style", "dim black on black"),
                width=display_width if display_width > 0 else 80,
                height=display_height if display_height > 0 else 24
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "sound_bars":
            display_width, display_height = self._get_terminal_size()
            self.effect_handler = SoundBarsEffect(
                self,
                title=self.card_data.get("title", "Visualizing..."),
                num_bars=self.card_data.get("num_bars", 15),
                max_bar_height=self.card_data.get("max_bar_height"), # None will use default calc
                bar_char_filled=self.card_data.get("bar_char_filled", "█"),
                bar_char_empty=self.card_data.get("bar_char_empty", " "),
                bar_styles=self.card_data.get("bar_styles", ["bold green"]),
                width=display_width if display_width > 0 else 80,
                height=display_height if display_height > 0 else 24,
                title_style=self.card_data.get("title_style", "bold white"),
                update_speed=self.card_data.get("update_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "raindrops": # Matches card_data effect name
            display_width, display_height = self._get_terminal_size()
            self.effect_handler = RaindropsEffect(
                self,
                title=self.card_data.get("title", "Rainy Day"),
                width=display_width if display_width > 0 else 80,
                height=display_height if display_height > 0 else 24,
                spawn_rate=self.card_data.get("spawn_rate", 1.0),
                ripple_chars=self.card_data.get("ripple_chars", ["."]),
                ripple_styles=self.card_data.get("ripple_styles", ["blue"]),
                max_concurrent_ripples=self.card_data.get("max_concurrent_ripples", 10),
                base_water_char=self.card_data.get("base_water_char","~"),
                water_style=self.card_data.get("water_style", "dim blue"),
                title_style=self.card_data.get("title_style", "bold white")
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "pixel_zoom":
            target_art = get_ascii_art(self.card_data.get("target_art_name", "default"))
            self.effect_handler = PixelZoomEffect(
                self,
                target_content=target_art,
                duration=self.card_data.get("duration", 2.5),
                max_pixel_size=self.card_data.get("max_pixel_size", 8),
                effect_type=self.card_data.get("effect_type", "resolve")
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "text_explosion":
            display_width, display_height = self._get_terminal_size()
            self.effect_handler = TextExplosionEffect(
                self,
                text=self.card_data.get("text_to_animate", "BOOM!"),
                effect_type=self.card_data.get("effect_direction", "explode"),
                duration=self.card_data.get("duration", 1.5),
                width=display_width if display_width > 0 else 80,
                height=display_height if display_height > 0 else 24,
                char_style=self.card_data.get("char_style", "bold red"),
                particle_spread=self.card_data.get("particle_spread", 30.0)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.03),
                self._update_animation
            )
        elif effect_type == "old_film":
            display_width, display_height = self._get_terminal_size()

            frame_names = self.card_data.get("frames_art_names", ["film_generic_frame"])
            frames_content_list = [get_ascii_art(name) for name in frame_names]
            if not frames_content_list: # Fallback if names don't resolve
                frames_content_list = [get_ascii_art("film_generic_frame")]

            self.effect_handler = OldFilmEffect(
                self,
                frames_content=frames_content_list,
                frame_duration=self.card_data.get("frame_duration", 0.5),
                shake_intensity=self.card_data.get("shake_intensity", 1),
                grain_density=self.card_data.get("grain_density", 0.05),
                grain_chars=self.card_data.get("grain_chars", ".:'"),
                base_style=self.card_data.get("film_base_style", "white on black"),
                width=display_width if display_width > 0 else 80,
                height=display_height if display_height > 0 else 24
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "maze_generator":
            display_width, display_height = self._get_terminal_size()
            self.effect_handler = MazeGeneratorEffect(
                self,
                title=self.card_data.get("title", "Generating..."),
                maze_width=self.card_data.get("maze_width", 39),
                maze_height=self.card_data.get("maze_height", 19),
                wall_char=self.card_data.get("wall_char", "█"),
                path_char=self.card_data.get("path_char", " "),
                cursor_char=self.card_data.get("cursor_char", "░"),
                wall_style=self.card_data.get("wall_style", "bold blue"),
                path_style=self.card_data.get("path_style", "on black"),
                cursor_style=self.card_data.get("cursor_style", "bold yellow"),
                title_style=self.card_data.get("title_style", "bold white"),
                generation_speed=self.card_data.get("generation_speed", 0.01),
                display_width=display_width if display_width > 0 else 80,
                display_height=display_height if display_height > 0 else 24
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.01), # Fast updates for smooth generation
                self._update_animation
            )
        elif effect_type == "mining":
            display_width, display_height = self._get_terminal_size()
            self.effect_handler = MiningEffect(
                self,
                content=self.card_data.get("content", self.DEFAULT_SPLASH),
                width=display_width if display_width > 0 else 80,
                height=display_height if display_height > 0 else 24,
                dig_speed=self.card_data.get("dig_speed", 0.8)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.08),
                self._update_animation
            )
        elif effect_type == "neural_network":
            width, height = self._get_terminal_size()
            self.effect_handler = NeuralNetworkEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                subtitle=self.card_data.get("subtitle", ""),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "quantum_particles":
            width, height = self._get_terminal_size()
            self.effect_handler = QuantumParticlesEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                subtitle=self.card_data.get("subtitle", ""),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "ascii_wave":
            width, height = self._get_terminal_size()
            self.effect_handler = ASCIIWaveEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                subtitle=self.card_data.get("subtitle", ""),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "binary_matrix":
            width, height = self._get_terminal_size()
            self.effect_handler = BinaryMatrixEffect(
                self,
                title=self.card_data.get("title", "TLDW"),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "constellation_map":
            width, height = self._get_terminal_size()
            self.effect_handler = ConstellationMapEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "typewriter_news":
            width, height = self._get_terminal_size()
            self.effect_handler = TypewriterNewsEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "dna_sequence":
            width, height = self._get_terminal_size()
            self.effect_handler = DNASequenceEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "circuit_trace":
            width, height = self._get_terminal_size()
            self.effect_handler = CircuitTraceEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.02)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.02),
                self._update_animation
            )
        elif effect_type == "plasma_field":
            width, height = self._get_terminal_size()
            self.effect_handler = PlasmaFieldEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "ascii_fire":
            width, height = self._get_terminal_size()
            self.effect_handler = ASCIIFireEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "rubiks_cube":
            width, height = self._get_terminal_size()
            self.effect_handler = RubiksCubeEffect(
                self,
                title=self.card_data.get("title", "TLDW"),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.5)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.5),
                self._update_animation
            )
        elif effect_type == "data_stream":
            width, height = self._get_terminal_size()
            self.effect_handler = DataStreamEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.02)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.02),
                self._update_animation
            )
        elif effect_type == "fractal_zoom":
            width, height = self._get_terminal_size()
            self.effect_handler = FractalZoomEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "ascii_spinner":
            width, height = self._get_terminal_size()
            self.effect_handler = ASCIISpinnerEffect(
                self,
                title=self.card_data.get("title", "Loading TLDW Chatbook"),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "hacker_terminal":
            width, height = self._get_terminal_size()
            self.effect_handler = HackerTerminalEffect(
                self,
                title=self.card_data.get("title", "TLDW Chatbook"),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "cyberpunk_glitch":
            width, height = self._get_terminal_size()
            self.effect_handler = CyberpunkGlitchEffect(
                self,
                title=self.card_data.get("title", "tldw chatbook"),
                subtitle=self.card_data.get("subtitle", ""),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "ascii_mandala":
            width, height = self._get_terminal_size()
            self.effect_handler = ASCIIMandalaEffect(
                self,
                title=self.card_data.get("title", "tldw chatbook"),
                subtitle=self.card_data.get("subtitle", ""),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "holographic_interface":
            width, height = self._get_terminal_size()
            self.effect_handler = HolographicInterfaceEffect(
                self,
                title=self.card_data.get("title", "tldw chatbook"),
                subtitle=self.card_data.get("subtitle", ""),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "quantum_tunnel":
            width, height = self._get_terminal_size()
            self.effect_handler = QuantumTunnelEffect(
                self,
                title=self.card_data.get("title", "tldw chatbook"),
                subtitle=self.card_data.get("subtitle", ""),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "chaotic_typewriter":
            width, height = self._get_terminal_size()
            self.effect_handler = ChaoticTypewriterEffect(
                self,
                title=self.card_data.get("title", "tldw chatbook"),
                subtitle=self.card_data.get("subtitle", ""),
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.03)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.03),
                self._update_animation
            )
        elif effect_type == "spy_vs_spy":
            width, height = self._get_terminal_size()
            self.effect_handler = SpyVsSpyEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "phonebooths_dialing":
            width, height = self._get_terminal_size()
            self.effect_handler = PhoneboothsDialingEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "emoji_face":
            width, height = self._get_terminal_size()
            self.effect_handler = EmojiFaceEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )
        elif effect_type == "custom_image":
            width, height = self._get_terminal_size()
            image_path = self.card_data.get("image_path", "")
            self.effect_handler = CustomImageEffect(
                self,
                image_path=image_path,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "ascii_aquarium":
            width, height = self._get_terminal_size()
            self.effect_handler = ASCIIAquariumEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "bookshelf_browser":
            width, height = self._get_terminal_size()
            self.effect_handler = BookshelfBrowserEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "train_journey":
            width, height = self._get_terminal_size()
            self.effect_handler = TrainJourneyEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "clock_mechanism":
            width, height = self._get_terminal_size()
            self.effect_handler = ClockMechanismEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "weather_system":
            width, height = self._get_terminal_size()
            self.effect_handler = WeatherSystemEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "music_visualizer":
            width, height = self._get_terminal_size()
            self.effect_handler = MusicVisualizerEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "origami_folding":
            width, height = self._get_terminal_size()
            self.effect_handler = OrigamiFoldingEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "ant_colony":
            width, height = self._get_terminal_size()
            self.effect_handler = AntColonyEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "neon_sign_flicker":
            width, height = self._get_terminal_size()
            self.effect_handler = NeonSignFlickerEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.1)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.1),
                self._update_animation
            )
        elif effect_type == "zen_garden":
            width, height = self._get_terminal_size()
            self.effect_handler = ZenGardenEffect(
                self,
                width=width,
                height=height,
                speed=self.card_data.get("animation_speed", 0.05)
            )
            self.animation_timer = self.set_interval(
                self.card_data.get("animation_speed", 0.05),
                self._update_animation
            )

    def _update_animation(self) -> None:
        """Update animation frame."""
        if self.effect_handler and hasattr(self.effect_handler, "update"):
            # Calculate elapsed time since start
            elapsed_time = time.time() - self.start_time
            
            # Check if the update method expects elapsed_time parameter
            import inspect
            sig = inspect.signature(self.effect_handler.update)
            # Check if there are any parameters besides 'self'
            params = list(sig.parameters.keys())
            if 'self' in params:
                params.remove('self')
            
            if len(params) > 0:  # Has parameters besides 'self'
                content = self.effect_handler.update(elapsed_time)
            else:
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