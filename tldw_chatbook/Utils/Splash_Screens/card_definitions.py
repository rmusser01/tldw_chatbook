"""Predefined splash card definitions."""

from ..Splash_Strings import splashscreen_message_selection
from ..Splash import get_ascii_art


def get_all_card_definitions():
    """Get all predefined splash card definitions."""
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
    
    return {
        "default": {
            "type": "static",
            "content": DEFAULT_SPLASH,
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
            "content": DEFAULT_SPLASH,
            "style": "bold white on black",
            "glitch_chars": "!@#$%^&*()_+-=[]{}|;:,.<>?",
            "animation_speed": 0.1
        },
        "retro": {
            "type": "animated",
            "effect": "retro_terminal",
            "content": DEFAULT_SPLASH,
            "style": "bold green on black",
            "scanline_speed": 0.02,
            "phosphor_glow": True
        },
        "tech_pulse": {
            "type": "animated",
            "effect": "pulse",
            "content": get_ascii_art("tech_pulse"),
            "style": "bold white on black",
            "pulse_speed": 0.5,
            "min_brightness": 80,
            "max_brightness": 200,
            "color": (100, 180, 255)
        },
        "code_scroll": {
            "type": "animated",
            "effect": "code_scroll",
            "title": "TLDW CHATBOOK",
            "subtitle": f"{splashscreen_message_selection}",
            "style": "white on black",
            "scroll_speed": 0.1,
            "num_code_lines": 18,
            "code_line_style": "dim blue",
            "title_style": "bold yellow",
            "subtitle_style": "green"
        },
        "minimal_fade": {
            "type": "animated",
            "effect": "typewriter",
            "content": get_ascii_art("minimal_fade"),
            "style": "white on black",
            "animation_speed": 0.08,
        },
        "blueprint": {
            "type": "static",
            "content": get_ascii_art("blueprint"),
            "style": "bold cyan on rgb(0,0,30)",
            "effect": None
        },
        "arcade_high_score": {
            "type": "animated",
            "effect": "blink",
            "content": get_ascii_art("arcade_high_score"),
            "style": "bold yellow on rgb(0,0,70)",
            "animation_speed": 0.1,
            "blink_speed": 0.5,
            "blink_targets": ["LOADING...", "PRESS ANY KEY TO START!"],
            "blink_style_off": "dim",
        },
        "digital_rain": {
            "type": "animated",
            "effect": "digital_rain",
            "title": "TLDW CHATBOOK v2.0",
            "subtitle": f"Enhancing neural pathways... {splashscreen_message_selection}",
            "style": "white on black",
            "animation_speed": 0.05,
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
            "content": get_ascii_art("loading_bar_frame"),
            "style": "white on black",
            "animation_speed": 0.1,
            "fill_char": "█",
            "bar_style": "bold green",
            "text_above": "SYSTEM INITIALIZATION SEQUENCE",
        },
        "starfield": {
            "type": "animated",
            "effect": "starfield",
            "title": "Hyperdrive Initializing...",
            "style": "black on black",
            "animation_speed": 0.04,
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
            "style": "green on black",
            "animation_speed": 0.03,
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
            "content": get_ascii_art("app_logo_clear"),
            "style": "bold white on black",
            "animation_speed": 0.05,
            "duration": 2.5,
            "glitch_chars": "!@#$%^&*▓▒░█",
            "start_intensity": 0.9,
            "end_intensity": 0.0
        },
        "ascii_morph": {
            "type": "animated",
            "effect": "ascii_morph",
            "style": "bold white on black",
            "animation_speed": 0.05,
            "duration": 3.0,
            "start_art_name": "morph_art_start",
            "end_art_name": "morph_art_end",
            "morph_style": "dissolve"
        },
        "game_of_life": {
            "type": "animated",
            "effect": "game_of_life",
            "title": "Cellular Automata Initializing...",
            "style": "black on black",
            "animation_speed": 0.1,
            "gol_update_interval": 0.15,
            "grid_width": 50,
            "grid_height": 18,
            "cell_alive_char": "■",
            "cell_dead_char": " ",
            "alive_style": "bold blue",
            "dead_style": "on black",
            "initial_pattern": "glider",
            "title_style": "bold yellow"
        },
        "scrolling_credits": {
            "type": "animated",
            "effect": "scrolling_credits",
            "title": "TLDW Chatbook",
            "style": "white on black",
            "animation_speed": 0.03,
            "scroll_speed": 2.0,
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
            "background_art_name": "spotlight_background",
            "style": "dim black on black",
            "animation_speed": 0.05,
            "spotlight_radius": 7,
            "movement_speed": 15.0,
            "path_type": "lissajous",
            "visible_style": "bold yellow on black",
            "hidden_style": "rgb(30,30,30) on black"
        },
        "sound_bars": {
            "type": "animated",
            "effect": "sound_bars",
            "title": "Frequency Analysis Engaged",
            "style": "black on black",
            "animation_speed": 0.05,
            "num_bars": 25,
            "bar_char_filled": "┃",
            "bar_char_empty": " ",
            "bar_styles": ["bold blue", "bold magenta", "bold cyan", "bold green", "bold yellow", "bold red", "bold white"],
            "title_style": "bold white",
            "update_speed": 0.05
        },
        "raindrops_pond": {
            "type": "animated",
            "effect": "raindrops",
            "title": "TLDW Reflections",
            "style": "dim blue on rgb(10,20,40)",
            "animation_speed": 0.05,
            "spawn_rate": 2.0,
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
            "target_art_name": "pixel_art_target",
            "style": "bold white on black",
            "animation_speed": 0.05,
            "duration": 3.0,
            "max_pixel_size": 10,
            "effect_type": "resolve"
        },
        "text_explosion": {
            "type": "animated",
            "effect": "text_explosion",
            "text_to_animate": "T . L . D . W",
            "style": "black on black",
            "animation_speed": 0.03,
            "duration": 2.0,
            "effect_direction": "implode",
            "char_style": "bold orange",
            "particle_spread": 40.0
        },
        "old_film": {
            "type": "animated",
            "effect": "old_film",
            "frames_art_names": ["film_generic_frame"],
            "style": "white on black",
            "animation_speed": 0.1,
            "frame_duration": 0.8,
            "shake_intensity": 1,
            "grain_density": 0.07,
            "grain_chars": ".:·'",
            "film_base_style": "sandy_brown on black"
        },
        "maze_generator": {
            "type": "animated",
            "effect": "maze_generator",
            "title": "Constructing Reality Tunnels...",
            "style": "black on black",
            "animation_speed": 0.01,
            "maze_width": 79,
            "maze_height": 21,
            "wall_char": "▓",
            "path_char": " ",
            "cursor_char": "❖",
            "wall_style": "bold dim blue",
            "path_style": "on rgb(10,10,10)",
            "cursor_style": "bold yellow",
            "title_style": "bold white",
            "generation_speed": 0.005
        },
        "dwarf_fortress": {
            "type": "animated",
            "effect": "mining",
            "content": get_ascii_art("dwarf_fortress"),
            "style": "rgb(139,69,19) on black",
            "animation_speed": 0.08,
            "dig_speed": 0.6,
        },
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
            "image_path": "",
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
        },
        "psychedelic_mandala": {
            "type": "animated",
            "effect": "psychedelic_mandala",
            "style": "black on black",
            "animation_speed": 0.05
        },
        "lava_lamp": {
            "type": "animated",
            "effect": "lava_lamp",
            "style": "black on black",
            "animation_speed": 0.05
        },
        "kaleidoscope": {
            "type": "animated",
            "effect": "kaleidoscope",
            "style": "black on black",
            "animation_speed": 0.05
        },
        "deep_dream": {
            "type": "animated",
            "effect": "deep_dream",
            "style": "white on black",
            "animation_speed": 0.1
        },
        "trippy_tunnel": {
            "type": "animated",
            "effect": "trippy_tunnel",
            "style": "black on black",
            "animation_speed": 0.05
        },
        "melting_screen": {
            "type": "animated",
            "effect": "melting_screen",
            "style": "white on black",
            "animation_speed": 0.05
        },
        "shroom_vision": {
            "type": "animated",
            "effect": "shroom_vision",
            "style": "white on black",
            "animation_speed": 0.05
        },
        "hypno_swirl": {
            "type": "animated",
            "effect": "hypno_swirl",
            "style": "white on black",
            "animation_speed": 0.05
        },
        "electric_sheep": {
            "type": "animated",
            "effect": "electric_sheep",
            "style": "black on black",
            "animation_speed": 0.05
        },
        "doom_fire": {
            "type": "animated",
            "effect": "doom_fire",
            "style": "black on black",
            "animation_speed": 0.05
        },
        "pacman": {
            "type": "animated",
            "effect": "pacman",
            "style": "black on black",
            "animation_speed": 0.1
        },
        "space_invaders": {
            "type": "animated",
            "effect": "space_invaders",
            "style": "black on black",
            "animation_speed": 0.1
        },
        "tetris": {
            "type": "animated",
            "effect": "tetris",
            "style": "black on black",
            "animation_speed": 0.1
        },
        "character_select": {
            "type": "animated",
            "effect": "character_select",
            "style": "white on blue",
            "animation_speed": 0.1
        },
        "achievement_unlocked": {
            "type": "animated",
            "effect": "achievement_unlocked",
            "style": "white on black",
            "animation_speed": 0.05
        },
        "versus_screen": {
            "type": "animated",
            "effect": "versus_screen",
            "style": "white on black",
            "animation_speed": 0.05
        },
        "world_map": {
            "type": "animated",
            "effect": "world_map",
            "style": "white on black",
            "animation_speed": 0.1
        },
        "level_up": {
            "type": "animated",
            "effect": "level_up",
            "style": "white on black",
            "animation_speed": 0.05
        },
        "retro_gaming_intro": {
            "type": "animated",
            "effect": "retro_gaming_intro",
            "style": "white on black",
            "animation_speed": 0.05
        }
    }