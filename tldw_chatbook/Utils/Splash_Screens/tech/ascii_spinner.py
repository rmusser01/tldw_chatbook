"""ASCIISpinner splash screen effect."""
import time

from rich.color import Color
import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("ascii_spinner")
class ASCIISpinnerEffect(BaseEffect):
    """Multiple synchronized loading spinners."""
    
    def __init__(self, parent, title="Loading TLDW Chatbook", width=80, height=24, speed=0.1, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.spinners = []
        self.phase = 0
        
        # Define spinner types
        self.spinner_types = [
            {'frames': ['|', '/', '-', '\\'], 'name': 'classic'},
            {'frames': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'], 'name': 'braille'},
            {'frames': ['◐', '◓', '◑', '◒'], 'name': 'circle'},
            {'frames': ['◰', '◳', '◲', '◱'], 'name': 'square'},
            {'frames': ['▖', '▘', '▝', '▗'], 'name': 'dots'},
            {'frames': ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'], 'name': 'arrows'},
            {'frames': ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃', '▂'], 'name': 'bars'}
        ]
        
        # Create spinner grid
        spacing_x = self.width // 4
        spacing_y = self.height // 4
        
        for i in range(3):
            for j in range(3):
                if i * 3 + j < len(self.spinner_types):
                    self.spinners.append({
                        'x': spacing_x * (j + 1),
                        'y': spacing_y * (i + 1),
                        'type': self.spinner_types[i * 3 + j],
                        'phase_offset': random.uniform(0, 1)
                    })
    
    def update(self) -> Optional[str]:
        """Update spinner animations."""
        elapsed_time = time.time() - self.start_time
        self.phase += elapsed_time * 2
    
        # Return the rendered content
        return self.render()
        
    def render(self):
        """Render multiple spinners."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw spinners
        for spinner in self.spinners:
            x, y = spinner['x'], spinner['y']
            frames = spinner['type']['frames']
            
            # Calculate frame index with phase offset
            frame_index = int((self.phase + spinner['phase_offset'] * len(frames)) % len(frames))
            
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = frames[frame_index]
                
                # Color based on spinner type
                if spinner['type']['name'] == 'classic':
                    style_grid[y][x] = 'cyan'
                elif spinner['type']['name'] == 'braille':
                    style_grid[y][x] = 'green'
                elif spinner['type']['name'] == 'circle':
                    style_grid[y][x] = 'blue'
                elif spinner['type']['name'] == 'square':
                    style_grid[y][x] = 'magenta'
                elif spinner['type']['name'] == 'dots':
                    style_grid[y][x] = 'yellow'
                elif spinner['type']['name'] == 'arrows':
                    style_grid[y][x] = 'red'
                else:
                    style_grid[y][x] = 'white'
                
                # Add label
                label = spinner['type']['name']
                if x + len(label) + 2 < self.width:
                    for i, char in enumerate(label):
                        grid[y][x + 2 + i] = char
                        style_grid[y][x + 2 + i] = 'dim white'
        
        # Add title
        self._add_centered_text(grid, style_grid, self.title, self.height - 2, 'bold white')
        
        # Add synchronization indicator
        sync_y = 1
        sync_text = f"Sync: {int(self.phase % len(self.spinner_types[0]['frames']))} / {len(self.spinner_types[0]['frames'])}"
        self._add_centered_text(grid, style_grid, sync_text, sync_y, 'dim white')
        
        return self._grid_to_string(grid, style_grid)