"""DNASequence splash screen effect."""
import time

from rich.color import Color
import math
import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("dna_sequence")
class DNASequenceEffect(BaseEffect):
    """Enhanced DNA double helix with genetic code."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.05, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.rotation = 0
        self.base_pairs = ['A-T', 'T-A', 'G-C', 'C-G']
        self.mutation_chance = 0.01
        self.gene_sequence = "INTELLIGENCEAUGMENTED"
        self.reveal_progress = 0
    
    def update(self) -> Optional[str]:
        """Update DNA rotation and mutations."""
        elapsed_time = time.time() - self.start_time
        self.rotation += elapsed_time * 1.5
        self.reveal_progress = min(1.0, self.reveal_progress + elapsed_time * 0.3)
        
        # Random mutations
        if random.random() < self.mutation_chance:
            self.mutation_flash = 1.0
    
        # Return the rendered content
        return self.render()
        
    def render(self):
        """Render DNA double helix."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        center_x = self.width // 2
        helix_width = 20
        
        for y in range(self.height):
            # Calculate helix position
            angle = (y * 0.5 + self.rotation) % (2 * math.pi)
            left_offset = int(math.sin(angle) * helix_width)
            right_offset = int(math.sin(angle + math.pi) * helix_width)
            
            left_x = center_x + left_offset
            right_x = center_x + right_offset
            
            # Draw backbone
            if 0 <= left_x < self.width:
                grid[y][left_x] = '|'
                style_grid[y][left_x] = 'bold blue'
            
            if 0 <= right_x < self.width:
                grid[y][right_x] = '|'
                style_grid[y][right_x] = 'bold blue'
            
            # Draw base pairs when strands cross
            if abs(left_offset - right_offset) < 3:
                base_pair = random.choice(self.base_pairs)
                connection_start = min(left_x, right_x) + 1
                connection_end = max(left_x, right_x)
                
                if connection_end - connection_start > 2:
                    mid = (connection_start + connection_end) // 2
                    if 0 <= mid - 1 < self.width and 0 <= mid + 1 < self.width:
                        grid[y][mid - 1] = base_pair[0]
                        grid[y][mid] = '-'
                        grid[y][mid + 1] = base_pair[2]
                        
                        # Color based on base
                        color_map = {'A': 'red', 'T': 'green', 'G': 'yellow', 'C': 'cyan'}
                        style_grid[y][mid - 1] = color_map.get(base_pair[0], 'white')
                        style_grid[y][mid] = 'white'
                        style_grid[y][mid + 1] = color_map.get(base_pair[2], 'white')
        
        # Add gene sequence reveal
        if self.reveal_progress > 0.3:
            seq_y = self.height // 2
            seq_x = (self.width - len(self.gene_sequence)) // 2
            revealed_chars = int(self.reveal_progress * len(self.gene_sequence))
            
            for i in range(revealed_chars):
                if seq_x + i < self.width:
                    grid[seq_y][seq_x + i] = self.gene_sequence[i]
                    style_grid[seq_y][seq_x + i] = 'bold white'
        
        # Add title
        if self.reveal_progress > 0.7:
            self._add_centered_text(grid, style_grid, self.title, 2, 'bold white')
        
        return self._grid_to_string(grid, style_grid)