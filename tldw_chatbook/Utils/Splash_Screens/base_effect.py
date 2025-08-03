"""
Base class and utilities for splash screen animation effects.

This module provides the foundation for all splash screen effects,
including the base class, common utilities, and registration system.
"""

import time
from typing import List, Optional, Any, Dict
from dataclasses import dataclass
from rich.style import Style

from loguru import logger


# Constants for escaping Rich markup
ESCAPED_OPEN_BRACKET = r'\['
ESCAPED_CLOSE_BRACKET = r'\]'


@dataclass
class AnimationFrame:
    """Represents a single frame of animation."""
    content: str
    style: Optional[Style] = None
    duration: float = 0.1


class BaseEffect:
    """Base class for animation effects."""
    
    def __init__(self, parent_widget: Any, **kwargs):
        self.parent = parent_widget
        self.frame_count = 0
        self.start_time = time.time()
        
    def update(self) -> Optional[str]:
        """Update and return the next frame of animation."""
        self.frame_count += 1
        return None
    
    def reset(self) -> None:
        """Reset the animation to its initial state."""
        self.frame_count = 0
        self.start_time = time.time()
    
    def _grid_to_string(self, grid: List[List[str]], style_grid: List[List[Optional[str]]]) -> str:
        """Convert a grid and style grid to a styled string."""
        lines = []
        for y in range(len(grid)):
            line_parts = []
            current_style = None
            current_text = ""
            
            for x in range(len(grid[y])):
                char = grid[y][x]
                style = style_grid[y][x]
                
                if style != current_style:
                    if current_text:
                        if current_style:
                            line_parts.append(f"[{current_style}]{current_text}[/{current_style}]")
                        else:
                            line_parts.append(current_text)
                    current_text = char
                    current_style = style
                else:
                    current_text += char
            
            # Add the last part
            if current_text:
                if current_style:
                    line_parts.append(f"[{current_style}]{current_text}[/{current_style}]")
                else:
                    line_parts.append(current_text)
            
            lines.append("".join(line_parts))
        
        return "\n".join(lines)
    
    def _add_centered_text(self, grid: List[List[str]], style_grid: List[List[Optional[str]]], 
                          text: str, y: int, style: str) -> None:
        """Add centered text to a grid at the specified y position."""
        if 0 <= y < len(grid):
            x_start = (len(grid[0]) - len(text)) // 2
            for i, char in enumerate(text):
                x = x_start + i
                if 0 <= x < len(grid[0]):
                    grid[y][x] = char
                    style_grid[y][x] = style


# Effect registration system
EFFECTS_REGISTRY: Dict[str, type] = {}


def register_effect(name: str):
    """
    Decorator to register an effect class.
    
    Usage:
        @register_effect("matrix_rain")
        class MatrixRainEffect(BaseEffect):
            ...
    """
    def decorator(cls):
        if name in EFFECTS_REGISTRY:
            logger.warning(f"Effect '{name}' is already registered, overwriting...")
        EFFECTS_REGISTRY[name] = cls
        cls._effect_name = name  # Store the registration name on the class
        logger.debug(f"Registered effect: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_effect_class(name: str) -> Optional[type]:
    """Get an effect class by its registered name."""
    return EFFECTS_REGISTRY.get(name)


def list_available_effects() -> List[str]:
    """Get a list of all registered effect names."""
    return sorted(EFFECTS_REGISTRY.keys())