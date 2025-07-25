# Splash.py
# Description: This file holds functions relating to the splashscreens for tldw_cli.
# Updated to work with the new splash screen system as an additional ASCII art provider.
#
# Imports
import os
import time
from typing import Optional, Dict, Any
#
# 3rd-party Libraries
#
# Local Imports
#
#######################################################################################################################
#
# Constants:

TLDW_ASCII_ART = r"""_____  _          ________  _    _                                 
|_   _|| |        / /|  _  \| |  | | _                              
  | |  | |       / / | | | || |  | |(_)                             
  | |  | |      / /  | | | || |/\| |                                
  | |  | |____ / /   | |/ / \  /\  / _                              
  \_/  \_____//_/    |___/   \/  \/ (_)                             


 _                   _                                              
| |                 | |                                             
| |_   ___    ___   | |  ___   _ __    __ _                         
| __| / _ \  / _ \  | | / _ \ | '_ \  / _` |                        
| |_ | (_) || (_) | | || (_) || | | || (_| | _                      
 \__| \___/  \___/  |_| \___/ |_| |_| \__, |( )                     
                                       __/ ||/                      
                                      |___/                         
     _  _      _         _  _                      _          _     
    | |(_)    | |       ( )| |                    | |        | |    
  __| | _   __| | _ __  |/ | |_  __      __  __ _ | |_   ___ | |__  
 / _` || | / _` || '_ \    | __| \ \ /\ / / / _` || __| / __|| '_ \ 
| (_| || || (_| || | | |   | |_   \ V  V / | (_| || |_ | (__ | | | |
 \__,_||_| \__,_||_| |_|    \__|   \_/\_/   \__,_| \__| \___||_| |_|
"""

# Additional ASCII art variations
TLDW_ASCII_COMPACT = r"""
╔══════════════════════════════════════════════════════════════════╗
║  ████████╗██╗     ██████╗ ██╗    ██╗                           ║
║  ╚══██╔══╝██║     ██╔══██╗██║    ██║                           ║
║     ██║   ██║     ██║  ██║██║ █╗ ██║                           ║
║     ██║   ██║     ██║  ██║██║███╗██║                           ║
║     ██║   ███████╗██████╔╝╚███╔███╔╝                           ║
║     ╚═╝   ╚══════╝╚═════╝  ╚══╝╚══╝                            ║
║                                                                  ║
║        Too Long; Didn't Watch aka ChatBook               ║
╚══════════════════════════════════════════════════════════════════╝
"""

TLDW_ASCII_MINIMAL = r"""
    ┌─┐┬  ┌┬┐┬ ┬   ┌─┐┬ ┬┌─┐┌┬┐┌┐ ┌─┐┌─┐┬┌─
    │  │   ││││││   │  ├─┤├─┤ │ ├┴┐│ ││ │├┴┐
    └─┘┴─┘─┴┘└┴┘   └─┘┴ ┴┴ ┴ ┴ └─┘└─┘└─┘┴ ┴
           tldw chatbook
"""

TECH_PULSE_ASCII = r"""
      .--''''''--.
    .'            '.
   /   O      O   \
  :                :
  |                |
  :    --------    :
   \   O      O   /
    '.            .'
      '--......--'
"""

MINIMAL_FADE_ASCII = r"""


          tldw

      chatbook AI Taking over....




"""

BLUEPRINT_ASCII = r"""
+----------------------------------------------------------------------+
| PROJECT: tldw chatbook AI                                            |
| REV: 1.0                                   DATE: 2083-04-15          |
+----------------------------------------------------------------------+
| MODULE: Splash Screen Interface                                      |
| DESIGNER: J. Doe (AI)                                                |
+----------------------------------------------------------------------+
|                                                                      |
|   +---------------------+        +---------------------+             |
|   |   Input Processor   |------->|   Core Logic Unit   |             |
|   | (ASCII Art Engine)  |        | (Animation Control) |             |
|   +---------------------+        +---------------------+             |
|         ^      |                         ^      |                    |
|         |      |                         |      |                    |
|         |      v                         |      v                    |
|   +---------------------+        +---------------------+             |
|   |  Resource Loader    |<-------|   Display Renderer  |             |
|   | (Cards & Config)  |        | (Rich Text Output)  |             |
|   +---------------------+        +---------------------+             |
|                                                                      |
|                                                                      |
|   LOADING SYSTEMS...                                                 |
+----------------------------------------------------------------------+
| STATUS: INITIALIZING ALL COMPONENTS                                  |
+----------------------------------------------------------------------+
"""

ARCADE_HIGH_SCORE_ASCII = r"""
+----------------------------------------+
|         **** HIGH SCORES ****         |
+----------------------------------------+
|                                        |
| RANK  NAME                SCORE        |
|                                        |
|  1ST  tldw_chatbook       31,687,490   |
|  2ND  DABEST              445,893      |
|  3RD  YANDER3             800,085      |
|  4TH  OH!                 233,333      |
|  5TH  ABRMS               134,870      |
|  6TH  MICHUL              79,420       |
|  7TH  SKY                 63,420       |
|  8TH  DEREK               61,220       |
|  9TH  URMOM               60,520       |
|                                        |
|                                        |
|       PRESS ANY KEY TO START!          |
|                                        |
+----------------------------------------+
"""

DWARF_FORTRESS_ASCII = r"""
████████████████████████████████████████████████████████████████████████████████
██████▓▓▓▓▓▓██████████▓▓▓▓▓▓████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████▓▓▓▓▓▓██████████
██████▓░░░░▓██████████▓░░░░▓████▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓████▓░░░░▓██████████
██████▓░██░▓██████████▓░██░▓████▓░████████████████████████░▓████▓░██░▓██████████
██████▓░██░▓██████████▓░██░▓████▓░████████████████████████░▓████▓░██░▓██████████
██████▓░██░▓▓▓▓▓▓▓▓▓▓▓▓░██░▓▓▓▓▓░████████████████████████░▓▓▓▓▓░██░▓██████████
██████▓░██░░░░░░░░░░░░░░██░░░░░░░████████████████████████░░░░░░░██░▓██████████
██████▓░████████████████████████████████████████████████████████░▓██████████
██████▓░████████████████████████████████████████████████████████░▓██████████
██████▓░██████████████████████████   tldw   ██████████████████████░▓██████████
██████▓░███████████████████████████████████████████████████████████░▓██████████
██████▓░████████████████████████   chatbook  ███████████████████░▓██████████
██████▓░████████████████████████████████████████████████████████░▓██████████
██████▓░████████████████████████████████████████████████████████░▓██████████
██████▓░██░░░░░░░░░░░░░░██░░░░░░░████████████████████████░░░░░░░██░▓██████████
██████▓░██░▓▓▓▓▓▓▓▓▓▓▓▓░██░▓▓▓▓▓░████████████████████████░▓▓▓▓▓░██░▓██████████
██████▓░██░▓██████████▓░██░▓████▓░████████████████████████░▓████▓░██░▓██████████
██████▓░██░▓██████████▓░██░▓████▓░████████████████████████░▓████▓░██░▓██████████
██████▓░░░░▓██████████▓░░░░▓████▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓████▓░░░░▓██████████
██████▓▓▓▓▓▓██████████▓▓▓▓▓▓████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████▓▓▓▓▓▓██████████
████████████████████████████████████████████████████████████████████████████████
                Strike the earth! Loading fortress systems...
"""

LOADING_BAR_FRAME_ASCII = r"""
[--------------------]
"""

MORPH_ART_START_ASCII = r"""
   #####
  ##   ##
      ##
     ##
     ##

     ##
"""

MORPH_ART_END_ASCII = r"""
     ##
     ##
     ##
     ##
     ##

     ##
"""

# A few options for fill characters, card config can pick one
LOADING_BAR_FILL_CHARS = {
    "default": "#",
    "block": "█",
    "dots": ".",
    "arrow": ">"
}


#
# Functions:

def print_tldw_ascii():
    """Legacy function for backwards compatibility."""
    print(TLDW_ASCII_ART)
    time.sleep(2)
    return

def get_ascii_art(name: str = "default") -> str:
    """Get ASCII art by name for use in splash screens.
    
    Args:
        name: Name of the ASCII art variant to retrieve.
              Options: "default", "compact", "minimal"
    
    Returns:
        The requested ASCII art as a string.
    """
    ascii_arts = {
        "default": TLDW_ASCII_ART,
        "compact": TLDW_ASCII_COMPACT,
        "minimal": TLDW_ASCII_MINIMAL,
        "classic": TLDW_ASCII_ART,  # Alias
        "tech_pulse": TECH_PULSE_ASCII,
        "minimal_fade": MINIMAL_FADE_ASCII,
        "blueprint": BLUEPRINT_ASCII,
        "arcade_high_score": ARCADE_HIGH_SCORE_ASCII,
        "loading_bar_frame": LOADING_BAR_FRAME_ASCII,
        "app_logo_clear": TLDW_ASCII_COMPACT, # Using compact version as the clear logo
        "morph_art_start": MORPH_ART_START_ASCII,
        "morph_art_end": MORPH_ART_END_ASCII,
        "spotlight_background": TLDW_ASCII_ART, # Using the main logo as background
        "pixel_art_target": TLDW_ASCII_COMPACT, # Target for pixel/zoom effect
        "dwarf_fortress": DWARF_FORTRESS_ASCII
    }
    # Note: For OldFilmEffect, frames will be passed directly in card config
    # or specific frame names can be added here if desired.
    # For this example, let's assume frames are small and defined in card or a generic one is used.
    # Adding one generic frame that can be referenced if needed.
    ascii_arts["film_generic_frame"] = """
+----------------------+
|       TLDW PRESENTS  |
|                      |
|   A Feature Splash   |
|                      |
|      Starring...     |
|        YOU!          |
+----------------------+
"""
    return ascii_arts.get(name, TLDW_ASCII_ART)

def get_splash_card_config(name: str) -> Dict[str, Any]:
    """Get a splash card configuration using ASCII art from this module.
    
    Args:
        name: Name of the ASCII art variant to use.
    
    Returns:
        Dictionary configuration for use with the splash screen system.
    """
    ascii_art = get_ascii_art(name)
    
    return {
        "type": "static",
        "content": ascii_art,
        "style": "bold cyan on rgb(0,0,0)",
        "effect": None
    }






#
# End of Splash.py
########################################################################################################################
