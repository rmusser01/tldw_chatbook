"""TetrisBlock splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple
from dataclasses import dataclass

from ..base_effect import BaseEffect, register_effect


@register_effect("tetris_block")
class TetrisBlockEffect(BaseEffect):
    """Tetris-style blocks fall from the top and stack up to form the title text."""
    
    @dataclass
    class Block:
        x: int
        y: float
        char: str
        target_y: int
        color: str
        falling: bool = True
        
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Building blocks...",
        width: int = 80,
        height: int = 24,
        fall_speed: float = 8.0,  # Blocks per second
        block_chars: str = "█",
        colors: List[str] = ["red", "yellow", "blue", "green", "magenta", "cyan"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.fall_speed = fall_speed
        self.block_chars = block_chars
        self.colors = colors
        self.title_style = title_style
        
        self.blocks: List[TetrisBlockEffect.Block] = []
        self.last_update_time = time.time()
        self.spawn_delay = 0.1
        self.time_since_spawn = 0.0
        self.title_positions = []
        
        self._calculate_title_positions()
        self.spawn_index = 0
        
    def _calculate_title_positions(self):
        """Calculate where each character of the title should be."""
        title_y = self.display_height // 2 - 2
        title_x = (self.display_width - len(self.title)) // 2
        
        for i, char in enumerate(self.title):
            if char != ' ':
                self.title_positions.append((title_x + i, title_y, char))
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Spawn new blocks
        self.time_since_spawn += delta_time
        if self.time_since_spawn >= self.spawn_delay and self.spawn_index < len(self.title_positions):
            x, y, char = self.title_positions[self.spawn_index]
            color = random.choice(self.colors)
            self.blocks.append(TetrisBlockEffect.Block(
                x=x, y=0, char=char, target_y=y, color=color
            ))
            self.spawn_index += 1
            self.time_since_spawn = 0.0
        
        # Update falling blocks
        for block in self.blocks:
            if block.falling:
                block.y += self.fall_speed * delta_time
                if block.y >= block.target_y:
                    block.y = block.target_y
                    block.falling = False
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw blocks
        for block in self.blocks:
            y = int(block.y)
            if 0 <= block.x < self.display_width and 0 <= y < self.display_height:
                grid[y][block.x] = block.char
                styles[y][block.x] = block.color if block.falling else self.title_style
        
        # Always show subtitle
        if self.subtitle:
            subtitle_y = self.display_height // 2 + 1
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)