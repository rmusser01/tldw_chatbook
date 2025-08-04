"""MorphingShape splash screen effect."""

import math
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("morphing_shape")
class MorphingShapeEffect(BaseEffect):
    """Geometric ASCII shapes that morph and transform into one another."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Transforming reality...",
        width: int = 80,
        height: int = 24,
        morph_speed: float = 0.5,  # Morphs per second
        shape_chars: str = "○□△▽◇◆",
        shape_colors: List[str] = ["red", "yellow", "blue", "green", "magenta", "cyan"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.morph_speed = morph_speed
        self.shape_chars = shape_chars
        self.shape_colors = shape_colors
        self.title_style = title_style
        
        self.morph_progress = 0.0
        self.current_shape = 0
        self.shapes = ['circle', 'square', 'triangle', 'diamond']
        self.last_update_time = time.time()
        
    def _draw_circle(self, center_x: int, center_y: int, radius: int, progress: float):
        """Draw a circle shape."""
        points = []
        num_points = int(50 * progress)
        for i in range(num_points):
            angle = (2 * math.pi * i) / 50
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle) * 0.5)
            points.append((x, y))
        return points
    
    def _draw_square(self, center_x: int, center_y: int, size: int, progress: float):
        """Draw a square shape."""
        points = []
        half_size = size // 2
        perimeter = size * 4
        points_to_draw = int(perimeter * progress)
        
        for i in range(points_to_draw):
            if i < size:  # Top edge
                points.append((center_x - half_size + i, center_y - half_size))
            elif i < size * 2:  # Right edge
                points.append((center_x + half_size, center_y - half_size + (i - size)))
            elif i < size * 3:  # Bottom edge
                points.append((center_x + half_size - (i - size * 2), center_y + half_size))
            else:  # Left edge
                points.append((center_x - half_size, center_y + half_size - (i - size * 3)))
        
        return points
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.morph_progress += self.morph_speed * delta_time
        
        if self.morph_progress >= 1.0:
            self.morph_progress = 0.0
            self.current_shape = (self.current_shape + 1) % len(self.shapes)
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        center_x = self.display_width // 2
        center_y = self.display_height // 2
        
        # Draw current shape
        shape_type = self.shapes[self.current_shape]
        if shape_type == 'circle':
            points = self._draw_circle(center_x, center_y, 10, self.morph_progress)
        elif shape_type == 'square':
            points = self._draw_square(center_x, center_y, 20, self.morph_progress)
        else:
            points = []
        
        # Draw points
        for x, y in points:
            if 0 <= x < self.display_width and 0 <= y < self.display_height:
                char_index = int(self.morph_progress * (len(self.shape_chars) - 1))
                grid[y][x] = self.shape_chars[char_index]
                color_index = self.current_shape % len(self.shape_colors)
                styles[y][x] = self.shape_colors[color_index]
        
        # Draw title (always visible)
        if self.title:
            title_y = center_y
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Draw subtitle
        if self.subtitle:
            subtitle_y = center_y + 3
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