"""ASCIIFire splash screen effect."""
import time

from rich.color import Color
import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("ascii_fire")
class ASCIIFireEffect(BaseEffect):
    """Realistic fire animation using ASCII characters."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.05, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.fire_chars = [' ', '.', ':', '^', '*', '†', '‡', '¥', '§']
        self.fire_grid = [[0 for _ in range(width)] for _ in range(height)]
        self.embers = []
    
    def update(self) -> Optional[str]:
        """Update fire animation."""
        elapsed_time = time.time() - self.start_time
        # Add new fire at bottom
        for x in range(self.width):
            if random.random() < 0.8:
                intensity = random.randint(6, 8)
                self.fire_grid[self.height - 1][x] = intensity
        
        # Propagate fire upwards
        new_grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        for y in range(self.height - 1):
            for x in range(self.width):
                # Get fire from below with some spreading
                below = self.fire_grid[y + 1][x]
                left = self.fire_grid[y + 1][x - 1] if x > 0 else 0
                right = self.fire_grid[y + 1][x + 1] if x < self.width - 1 else 0
                
                # Average with decay
                avg = (below * 0.97 + left * 0.01 + right * 0.01)
                new_grid[y][x] = max(0, avg - random.uniform(0, 0.5))
        
        # Copy bottom row
        new_grid[self.height - 1] = self.fire_grid[self.height - 1][:]
        self.fire_grid = new_grid
        
        # Create embers
        if random.random() < 0.1:
            self.embers.append({
                'x': random.randint(self.width // 3, 2 * self.width // 3),
                'y': self.height - 5,
                'vy': -random.uniform(0.5, 1.5),
                'life': 1.0
            })
        
        # Update embers
        for ember in self.embers[:]:
            ember['y'] += ember['vy']
            ember['vy'] += 0.1  # Gravity
            ember['life'] -= elapsed_time * 0.5
            
            if ember['life'] <= 0 or ember['y'] >= self.height:
                self.embers.remove(ember)
    
        # Return the rendered content
        return self.render()
        
    def render(self):
        """Render fire effect."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw fire
        for y in range(self.height):
            for x in range(self.width):
                intensity = self.fire_grid[y][x]
                if intensity > 0:
                    char_index = min(int(intensity), len(self.fire_chars) - 1)
                    grid[y][x] = self.fire_chars[char_index]
                    
                    # Color based on intensity
                    if intensity > 6:
                        style_grid[y][x] = 'bold white'
                    elif intensity > 4:
                        style_grid[y][x] = 'bold yellow'
                    elif intensity > 2:
                        style_grid[y][x] = 'red'
                    else:
                        style_grid[y][x] = 'dim red'
        
        # Draw embers
        for ember in self.embers:
            x, y = int(ember['x']), int(ember['y'])
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = '°'
                style_grid[y][x] = 'yellow' if ember['life'] > 0.5 else 'dim red'
        
        # Add title in the flames
        title_y = self.height // 3
        self._add_centered_text(grid, style_grid, self.title, title_y, 'bold white on red')
        
        return self._grid_to_string(grid, style_grid)