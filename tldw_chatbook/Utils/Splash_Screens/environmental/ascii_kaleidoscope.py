"""ASCIIKaleidoscope splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("ascii_kaleidoscope")
class ASCIIKaleidoscopeEffect(BaseEffect):
    """Creates symmetrical, rotating kaleidoscope patterns using ASCII characters."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Infinite patterns...",
        width: int = 80,
        height: int = 24,
        rotation_speed: float = 0.3,
        num_mirrors: int = 6,
        pattern_chars: str = "◆◇○●□■▲▼",
        pattern_colors: List[str] = ["red", "yellow", "blue", "green", "magenta", "cyan", "white"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.rotation_speed = rotation_speed
        self.num_mirrors = num_mirrors
        self.pattern_chars = pattern_chars
        self.pattern_colors = pattern_colors
        self.title_style = title_style
        
        self.rotation = 0.0
        self.last_update_time = time.time()
        self.pattern_elements = []
        
        self._generate_pattern()
        
    def _generate_pattern(self):
        """Generate random pattern elements in one segment."""
        segment_angle = (2 * math.pi) / self.num_mirrors
        
        for _ in range(10):  # Number of elements per segment
            angle = random.uniform(0, segment_angle)
            radius = random.uniform(3, 15)
            char = random.choice(self.pattern_chars)
            color = random.choice(self.pattern_colors)
            
            self.pattern_elements.append({
                'angle': angle,
                'radius': radius,
                'char': char,
                'color': color
            })
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.rotation += self.rotation_speed * delta_time
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        center_x = self.display_width // 2
        center_y = self.display_height // 2
        
        # Draw kaleidoscope pattern
        for element in self.pattern_elements:
            for mirror in range(self.num_mirrors):
                # Calculate mirrored angle
                mirror_angle = (2 * math.pi * mirror) / self.num_mirrors
                angle = element['angle'] + mirror_angle + self.rotation
                
                # Calculate position
                x = int(center_x + element['radius'] * math.cos(angle))
                y = int(center_y + element['radius'] * math.sin(angle) * 0.5)
                
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    grid[y][x] = element['char']
                    styles[y][x] = element['color']
                
                # Add reflection
                if mirror % 2 == 0:
                    angle_reflected = -element['angle'] + mirror_angle + self.rotation
                    x = int(center_x + element['radius'] * math.cos(angle_reflected))
                    y = int(center_y + element['radius'] * math.sin(angle_reflected) * 0.5)
                    
                    if 0 <= x < self.display_width and 0 <= y < self.display_height:
                        grid[y][x] = element['char']
                        styles[y][x] = element['color']
        
        # Draw title in center
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