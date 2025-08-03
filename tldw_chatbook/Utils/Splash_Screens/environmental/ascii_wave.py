"""ASCIIWave splash screen effect."""
import time

import math
import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("ascii_wave")
class ASCIIWaveEffect(BaseEffect):
    """Ocean waves animation with ASCII characters."""
    
    def __init__(self, parent, title="TLDW Chatbook", subtitle="", width=80, height=24, speed=0.1, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.subtitle = subtitle
        self.wave_offset = 0
        self.wave_chars = ['_', '-', '~', '≈', '~', '-', '_']
        self.foam_chars = ['·', '°', '*', '°', '·']
    
    def update(self) -> Optional[str]:
        """Update wave animation."""
        elapsed_time = time.time() - self.start_time
        self.wave_offset += elapsed_time * 5
        # Return the rendered content
        return self.render()
    
    def render(self):
        """Render ocean waves."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Calculate wave heights
        for x in range(self.width):
            # Primary wave
            wave1 = math.sin((x + self.wave_offset) * 0.1) * 3
            wave2 = math.sin((x + self.wave_offset * 0.7) * 0.15) * 2
            wave3 = math.sin((x + self.wave_offset * 1.3) * 0.05) * 4
            
            total_wave = wave1 + wave2 + wave3
            wave_height = int(self.height / 2 + total_wave)
            
            # Draw water column
            for y in range(self.height):
                if y > wave_height:
                    # Below water
                    depth = y - wave_height
                    if depth < len(self.wave_chars):
                        grid[y][x] = self.wave_chars[depth]
                        intensity = 255 - depth * 20
                        style_grid[y][x] = f'rgb({intensity//2},{intensity//2},{intensity})'
                    else:
                        grid[y][x] = '▓'
                        style_grid[y][x] = 'blue'
                elif y == wave_height:
                    # Wave crest
                    if random.random() < 0.3:
                        grid[y][x] = random.choice(self.foam_chars)
                        style_grid[y][x] = 'bold white'
                    else:
                        grid[y][x] = '≈'
                        style_grid[y][x] = 'bold cyan'
        
        # Add title in the sky
        self._add_centered_text(grid, style_grid, self.title, self.height // 4, 'bold white')
        if self.subtitle:
            self._add_centered_text(grid, style_grid, self.subtitle, self.height // 4 + 2, 'white')
        
        return self._grid_to_string(grid, style_grid)