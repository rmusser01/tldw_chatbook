# splash_animations.py
# This file has been refactored into a modular structure.
# All splash screen effects have been moved to Utils/Splash_Screens/
# 
# The effects are now organized by category:
# - classic/       Classic and basic effects (matrix rain, glitch, typewriter, etc.)
# - environmental/ Nature and environment effects (starfield, raindrops, fire, etc.)
# - tech/          Technology and sci-fi effects (digital rain, terminal boot, etc.)
# - gaming/        Gaming inspired effects (pacman, tetris, achievements, etc.)
# - psychedelic/   Trippy and psychedelic effects (lava lamp, kaleidoscope, etc.)
# - custom/        User-contributed effects
#
# To use the new system:
# from tldw_chatbook.Utils.Splash_Screens import load_all_effects, get_effect_class
#
# # Load all effects
# load_all_effects()
#
# # Get a specific effect class
# MatrixRainEffect = get_effect_class("matrix_rain")
#
# For backward compatibility, you can import the base class:
from tldw_chatbook.Utils.Splash_Screens.base_effect import BaseEffect

# Import common constants
ESCAPED_OPEN_BRACKET = r'\['
ESCAPED_CLOSE_BRACKET = r'\]'