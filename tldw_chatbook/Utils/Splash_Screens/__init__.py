"""
Splash screen effects module with auto-discovery.

This module automatically discovers and loads all effect classes
from subdirectories, making them available through a central registry.
"""

import importlib
from pathlib import Path

from loguru import logger

from .base_effect import BaseEffect, register_effect, EFFECTS_REGISTRY, get_effect_class, list_available_effects


# Export the main components
__all__ = [
    'BaseEffect',
    'register_effect',
    'EFFECTS_REGISTRY',
    'get_effect_class',
    'list_available_effects',
    'load_all_effects',
]

_effects_loaded = False


def load_all_effects(force: bool = False):
    """
    Automatically discover and load all effect modules from subdirectories.
    
    This function scans all subdirectories (classic, environmental, tech, etc.)
    and imports any Python modules found, which should register their effects
    using the @register_effect decorator.
    """
    global _effects_loaded
    if _effects_loaded and not force:
        return 0

    # Get the current package directory
    package_dir = Path(__file__).parent
    
    # List of subdirectories to scan for effects
    effect_categories = ['classic', 'environmental', 'tech', 'gaming', 'psychedelic', 'custom']
    
    loaded_count = 0
    
    for category in effect_categories:
        category_path = package_dir / category
        if not category_path.exists():
            logger.debug(f"Category directory '{category}' does not exist, skipping...")
            continue
            
        # Import all Python files in the category directory
        for file_path in category_path.glob('*.py'):
            if file_path.name.startswith('_'):
                continue  # Skip private modules
                
            module_name = file_path.stem
            full_module_name = f"{__package__}.{category}.{module_name}"
            
            try:
                logger.debug(f"Loading effect module: {full_module_name}")
                importlib.import_module(full_module_name)
                loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load effect module '{full_module_name}': {e}")
    
    _effects_loaded = True
    logger.info(f"Loaded {loaded_count} effect modules, {len(EFFECTS_REGISTRY)} effects registered")
    return loaded_count
