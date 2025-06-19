# config_image_addition.py
# Description: Image configuration additions for tldw_chatbook config.py
#
# This file shows the additions needed to config.py for image support
#
#######################################################################################################################
#
# Add to DEFAULT configurations section (around line 125 after DEFAULT_RAG_SEARCH_CONFIG):

DEFAULT_IMAGE_CONFIG = {
    "enabled": True,
    "default_render_mode": "auto",  # auto, pixels, regular
    "max_size_mb": 10,
    "auto_resize": True,
    "resize_max_dimension": 2048,
    "save_location": "~/Downloads",
    "supported_formats": [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"],
    "terminal_overrides": {
        "kitty": "regular",
        "wezterm": "regular",
        "iterm2": "regular",
        "xterm": "regular",
        "alacritty": "pixels",
        "default": "pixels"
    }
}

# Add to the load_settings() function where other defaults are loaded:
def load_image_settings(config_data: dict) -> dict:
    """Load image settings from config or use defaults."""
    image_config = config_data.get('chat', {}).get('images', {})
    
    # Merge with defaults
    final_config = DEFAULT_IMAGE_CONFIG.copy()
    
    if image_config:
        # Handle individual settings
        if 'enabled' in image_config:
            final_config['enabled'] = _get_typed_value(image_config, 'enabled', True, bool)
        
        if 'default_render_mode' in image_config:
            mode = image_config.get('default_render_mode', 'auto')
            if mode in ['auto', 'pixels', 'regular']:
                final_config['default_render_mode'] = mode
        
        if 'max_size_mb' in image_config:
            final_config['max_size_mb'] = _get_typed_value(image_config, 'max_size_mb', 10, float)
        
        if 'auto_resize' in image_config:
            final_config['auto_resize'] = _get_typed_value(image_config, 'auto_resize', True, bool)
        
        if 'resize_max_dimension' in image_config:
            final_config['resize_max_dimension'] = _get_typed_value(image_config, 'resize_max_dimension', 2048, int)
        
        if 'save_location' in image_config:
            final_config['save_location'] = image_config.get('save_location', '~/Downloads')
        
        if 'supported_formats' in image_config:
            formats = image_config.get('supported_formats', [])
            if isinstance(formats, list):
                final_config['supported_formats'] = formats
        
        if 'terminal_overrides' in image_config:
            overrides = image_config.get('terminal_overrides', {})
            if isinstance(overrides, dict):
                final_config['terminal_overrides'].update(overrides)
    
    return final_config

# Add this to the config.toml file structure:
"""
[chat.images]
enabled = true
default_render_mode = "auto"  # auto, pixels, regular
max_size_mb = 10.0
auto_resize = true
resize_max_dimension = 2048
save_location = "~/Downloads"
supported_formats = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]

[chat.images.terminal_overrides]
kitty = "regular"
wezterm = "regular"
iterm2 = "regular"
xterm = "regular"
alacritty = "pixels"
default = "pixels"
"""

#
#
#######################################################################################################################