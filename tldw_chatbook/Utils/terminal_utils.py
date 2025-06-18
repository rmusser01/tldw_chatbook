# tldw_chatbook/Utils/terminal_utils.py
# Description: Terminal capability detection utilities
#
# Imports
#
# Standard Library
import os
import sys
from typing import Dict, Literal

#
# Local Imports
#
#######################################################################################################################
#
# Functions:

def detect_terminal_capabilities() -> Dict[str, any]:
    """
    Detect terminal image support capabilities.
    
    Returns:
        Dictionary with:
        - sixel: Whether sixel graphics are supported
        - tgp: Whether terminal graphics protocol is supported
        - unicode: Whether unicode is supported (assumed True)
        - recommended_mode: Recommended rendering mode ('pixels', 'regular', or 'auto')
    """
    term = os.environ.get('TERM', '').lower()
    term_program = os.environ.get('TERM_PROGRAM', '').lower()
    
    capabilities = {
        'sixel': False,
        'tgp': False,
        'unicode': True,  # Assume unicode support by default
        'recommended_mode': 'pixels'  # Default to rich-pixels
    }
    
    # Check for specific terminals that support advanced graphics
    
    # Kitty terminal
    if 'kitty' in term or 'kitty' in term_program:
        capabilities['tgp'] = True
        capabilities['recommended_mode'] = 'regular'
    
    # WezTerm
    elif 'wezterm' in term:
        capabilities['tgp'] = True
        capabilities['sixel'] = True
        capabilities['recommended_mode'] = 'regular'
    
    # iTerm2
    elif 'iterm' in term_program or os.environ.get('ITERM_SESSION_ID'):
        capabilities['tgp'] = True
        capabilities['recommended_mode'] = 'regular'
    
    # Alacritty (limited support)
    elif 'alacritty' in term or 'alacritty' in term_program:
        # Alacritty has limited image support
        capabilities['recommended_mode'] = 'pixels'
    
    # XTerm with 256 colors (may support sixel)
    elif 'xterm' in term and '256color' in term:
        # Some xterm builds support sixel
        capabilities['sixel'] = True
        capabilities['recommended_mode'] = 'regular'
    
    # Konsole
    elif 'konsole' in term_program:
        capabilities['sixel'] = True
        capabilities['recommended_mode'] = 'regular'
    
    # GNOME Terminal / VTE-based terminals
    elif os.environ.get('VTE_VERSION'):
        # Most VTE-based terminals don't support advanced graphics
        capabilities['recommended_mode'] = 'pixels'
    
    # Windows Terminal
    elif 'windows-terminal' in term_program or os.environ.get('WT_SESSION'):
        # Windows Terminal has limited image support
        capabilities['recommended_mode'] = 'pixels'
    
    return capabilities


def get_image_render_mode(
    config_mode: str = "auto"
) -> Literal["pixels", "regular"]:
    """
    Determine the best image rendering mode based on terminal capabilities and config.
    
    Args:
        config_mode: Configuration mode ('auto', 'pixels', 'regular')
        
    Returns:
        Either 'pixels' or 'regular' rendering mode
    """
    if config_mode == "pixels":
        return "pixels"
    elif config_mode == "regular":
        return "regular"
    else:  # auto mode
        capabilities = detect_terminal_capabilities()
        # If terminal supports advanced graphics and textual-image is available
        if capabilities['recommended_mode'] == 'regular':
            # Check if textual-image is available
            try:
                import textual_image
                return "regular"
            except ImportError:
                # Fall back to pixels if textual-image not available
                return "pixels"
        return "pixels"


def is_image_support_available() -> bool:
    """
    Check if any form of image support is available.
    
    Returns:
        True if images can be displayed (either mode)
    """
    # Check for rich-pixels (should always be available if PIL is installed)
    try:
        import rich_pixels
        import PIL
        return True
    except ImportError:
        pass
    
    # Check for textual-image
    try:
        import textual_image
        return True
    except ImportError:
        pass
    
    return False

#
#
#######################################################################################################################