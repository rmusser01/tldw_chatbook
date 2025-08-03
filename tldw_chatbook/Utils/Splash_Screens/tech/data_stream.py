"""DataStream splash screen effect."""
import time

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("data_stream")
class DataStreamEffect(BaseEffect):
    """Hexadecimal data streaming with hidden messages."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.02, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.data_lines = []
        self.decoded_message = "TERMINAL LANGUAGE DATA WATCHER"
        self.decode_progress = 0
        self.highlight_positions = []
        
        # Initialize data lines
        for _ in range(height):
            self.data_lines.append(self._generate_data_line())
    
    def _generate_data_line(self):
        """Generate a line of hex data."""
        hex_chars = '0123456789ABCDEF'
        line = []
        for _ in range(self.width // 3):
            line.append(random.choice(hex_chars) + random.choice(hex_chars))
        return line
    
    def update(self) -> Optional[str]:
        """Update data stream."""
        elapsed_time = time.time() - self.start_time
        # Scroll data
        if random.random() < 0.3:
            self.data_lines.pop(0)
            self.data_lines.append(self._generate_data_line())
        
        # Update decode progress
        self.decode_progress += elapsed_time * 0.2
        
        # Create highlight positions for decoded message
        if self.decode_progress > 0.3 and len(self.highlight_positions) < len(self.decoded_message):
            if random.random() < 0.1:
                self.highlight_positions.append({
                    'char': self.decoded_message[len(self.highlight_positions)],
                    'x': random.randint(0, self.width - 3),
                    'y': random.randint(0, self.height - 1)
                })
    
        # Return the rendered content
        return self.render()
        
    def render(self):
        """Render data stream."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw hex data
        for y, line in enumerate(self.data_lines):
            x = 0
            for hex_pair in line:
                if x + 2 < self.width:
                    grid[y][x] = hex_pair[0]
                    grid[y][x + 1] = hex_pair[1]
                    
                    # Random highlighting
                    if random.random() < 0.05:
                        style_grid[y][x] = style_grid[y][x + 1] = 'bold green'
                    else:
                        style_grid[y][x] = style_grid[y][x + 1] = 'dim cyan'
                    
                    x += 3  # Space between hex pairs
        
        # Draw decoded characters
        for pos in self.highlight_positions:
            x, y = pos['x'], pos['y']
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = pos['char']
                style_grid[y][x] = 'bold yellow'
        
        # Show full decoded message when complete
        if len(self.highlight_positions) >= len(self.decoded_message):
            msg_y = self.height // 2
            self._add_centered_text(grid, style_grid, self.decoded_message, msg_y - 1, 'bold white on green')
            self._add_centered_text(grid, style_grid, self.title, msg_y + 1, 'bold white')
        
        return self._grid_to_string(grid, style_grid)