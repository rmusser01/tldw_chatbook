"""BinaryMatrix splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("binary_matrix")
class BinaryMatrixEffect(BaseEffect):
    """Binary rain effect with highlighting patterns."""
    
    def __init__(self, parent, title="TLDW", width=80, height=24, speed=0.05, **kwargs):
        super().__init__(parent, **kwargs)
        self.width = kwargs.get('width', width)
        self.height = kwargs.get('height', height)
        self.speed = kwargs.get('speed', speed)
        self.title = kwargs.get('title', title)
        self.columns = []
        self.highlight_pattern = "TLDW"
        self.highlight_positions = []
        
        # Initialize columns
        for x in range(self.width):
            self.columns.append({
                'chars': ['0', '1'] * self.height,
                'offset': random.randint(0, self.height),
                'speed': random.uniform(0.5, 2.0),
                'highlight': False
            })
    
    def update(self) -> Optional[str]:
        """Update binary rain."""
        elapsed_time = time.time() - self.start_time
        for col in self.columns:
            col['offset'] += col['speed']
            if col['offset'] >= self.height * 2:
                col['offset'] = 0
                col['speed'] = random.uniform(0.5, 2.0)
                # Randomly generate new binary sequence
                col['chars'] = [random.choice(['0', '1']) for _ in range(self.height * 2)]
        
        # Update highlight positions
        if random.random() < 0.05:
            self._create_highlight()
        
        # Return the rendered content
        return self.render()
    
    def _create_highlight(self):
        """Create a highlighted pattern in the binary rain."""
        start_x = random.randint(0, self.width - len(self.highlight_pattern))
        start_y = random.randint(0, self.height - 1)
        
        self.highlight_positions = []
        for i, char in enumerate(self.highlight_pattern):
            self.highlight_positions.append({
                'x': start_x + i,
                'y': start_y,
                'char': char,
                'life': 2.0
            })
    
    def render(self):
        """Render binary matrix rain."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw binary columns
        for x, col in enumerate(self.columns):
            offset = int(col['offset'])
            for y in range(self.height):
                char_idx = (y + offset) % len(col['chars'])
                char = col['chars'][char_idx]
                
                # Fade based on position
                distance_from_head = (offset - y) % self.height
                if distance_from_head < 3:
                    style = 'bold green'
                elif distance_from_head < 10:
                    style = 'green'
                else:
                    style = 'dim green'
                
                grid[y][x] = char
                style_grid[y][x] = style
        
        # Draw highlights
        for highlight in self.highlight_positions[:]:
            if highlight['life'] > 0:
                x, y = highlight['x'], highlight['y']
                if 0 <= x < self.width and 0 <= y < self.height:
                    grid[y][x] = highlight['char']
                    style_grid[y][x] = 'bold yellow' if highlight['life'] > 1 else 'yellow'
                highlight['life'] -= 0.05
            else:
                self.highlight_positions.remove(highlight)
        
        return self._grid_to_string(grid, style_grid)