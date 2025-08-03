"""
Splash screen effects module with auto-discovery.

This module automatically discovers and loads all effect classes
from subdirectories, making them available through a central registry.
"""

import os
import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Type, List

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


def load_all_effects():
    """
    Automatically discover and load all effect modules from subdirectories.
    
    This function scans all subdirectories (classic, environmental, tech, etc.)
    and imports any Python modules found, which should register their effects
    using the @register_effect decorator.
    """
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
    
    logger.info(f"Loaded {loaded_count} effect modules, {len(EFFECTS_REGISTRY)} effects registered")
    return loaded_count


# Auto-load all effects when this module is imported
try:
    load_all_effects()
except Exception as e:
    logger.error(f"Failed to auto-load effects: {e}")
    # Don't fail the import if auto-loading fails