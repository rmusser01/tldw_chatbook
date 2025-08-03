"""PlasmaField splash screen effect."""
import time

from rich.color import Color
import math
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("plasma_field")
class PlasmaFieldEffect(BaseEffect):
    """Animated plasma field effect."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.05, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.time_offset = 0
        self.plasma_chars = [' ', '·', ':', '░', '▒', '▓', '█']
    
    def update(self) -> Optional[str]:
        """Update plasma field."""
        elapsed_time = time.time() - self.start_time
        self.time_offset += elapsed_time * 2
        
        # Return the rendered content
        return self.render()
    
    def render(self):
        """Render plasma field."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Generate plasma field
        for y in range(self.height):
            for x in range(self.width):
                # Calculate plasma value using multiple sine waves
                v1 = math.sin((x * 0.1) + self.time_offset)
                v2 = math.sin((y * 0.1) + self.time_offset * 1.3)
                v3 = math.sin(((x + y) * 0.05) + self.time_offset * 0.7)
                v4 = math.sin(math.sqrt((x - self.width/2)**2 + (y - self.height/2)**2) * 0.1 - self.time_offset)
                
                # Combine waves
                plasma_value = (v1 + v2 + v3 + v4) / 4
                normalized = (plasma_value + 1) / 2  # Normalize to 0-1
                
                # Select character based on plasma value
                char_index = int(normalized * (len(self.plasma_chars) - 1))
                grid[y][x] = self.plasma_chars[char_index]
                
                # Color based on plasma value
                if normalized < 0.25:
                    style_grid[y][x] = 'blue'
                elif normalized < 0.5:
                    style_grid[y][x] = 'cyan'
                elif normalized < 0.75:
                    style_grid[y][x] = 'magenta'
                else:
                    style_grid[y][x] = 'red'
        
        # Clear area for title
        title_y = self.height // 2
        title_area_height = 3
        for y in range(title_y - 1, title_y + title_area_height - 1):
            for x in range(self.width // 4, 3 * self.width // 4):
                if 0 <= y < self.height:
                    grid[y][x] = ' '
                    style_grid[y][x] = None
        
        # Add title
        self._add_centered_text(grid, style_grid, self.title, title_y, 'bold white')
        
        return self._grid_to_string(grid, style_grid)