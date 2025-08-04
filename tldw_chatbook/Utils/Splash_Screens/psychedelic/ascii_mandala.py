"""ASCIIMandala splash screen effect."""

from rich.color import Color
import math
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("ascii_mandala")
class ASCIIMandalaEffect(BaseEffect):
    """Rotating mandala pattern that expands from center."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "",
        width: int = 80,
        height: int = 24,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.height = height
        
        self.center_x = width // 2
        self.center_y = height // 2
        self.max_radius = min(width // 2, height // 2) - 2
        self.rotation = 0
        self.expansion = 0
        
        # Mandala patterns
        self.patterns = [
            "◆◇◈◊",
            "●○◐◑◒◓",
            "▲▼◀▶",
            "★☆✦✧",
            "┼╬╋╪",
        ]
    
    def update(self) -> Optional[str]:
        """Update mandala effect."""
        elapsed = time.time() - self.start_time
        self.rotation = elapsed * 0.5  # Rotation speed
        self.expansion = min(1.0, elapsed / 2.0)  # Expand over 2 seconds
        
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw expanding mandala
        current_radius = int(self.max_radius * self.expansion)
        
        for radius in range(1, current_radius + 1):
            # Number of points on this circle
            points = max(8, radius * 2)
            pattern_idx = radius % len(self.patterns)
            pattern = self.patterns[pattern_idx]
            
            for i in range(points):
                angle = (2 * math.pi * i / points) + self.rotation
                x = int(self.center_x + radius * math.cos(angle))
                y = int(self.center_y + radius * math.sin(angle) * 0.5)  # Adjust for aspect ratio
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    char_idx = i % len(pattern)
                    grid[y][x] = pattern[char_idx]
                    
                    # Color based on radius
                    hue = (radius * 20 + elapsed * 50) % 360
                    r = int(128 + 127 * math.sin(math.radians(hue)))
                    g = int(128 + 127 * math.sin(math.radians(hue + 120)))
                    b = int(128 + 127 * math.sin(math.radians(hue + 240)))
                    style_grid[y][x] = f"rgb({r},{g},{b})"
        
        # Draw center point
        grid[self.center_y][self.center_x] = '✦'
        style_grid[self.center_y][self.center_x] = 'bold white'
        
        # Title in center after expansion
        if self.expansion > 0.8:
            title_progress = (self.expansion - 0.8) / 0.2
            title_len = int(len(self.title) * title_progress)
            title_to_show = self.title[:title_len]
            
            title_x = self.center_x - len(self.title) // 2
            for i, char in enumerate(title_to_show):
                if 0 <= title_x + i < self.width:
                    grid[self.center_y][title_x + i] = char
                    style_grid[self.center_y][title_x + i] = 'bold white'
        
        # Subtitle below
        if self.subtitle and elapsed > 2.5:
            subtitle_x = self.center_x - len(self.subtitle) // 2
            subtitle_y = self.center_y + 2
            for i, char in enumerate(self.subtitle):
                if 0 <= subtitle_x + i < self.width and subtitle_y < self.height:
                    grid[subtitle_y][subtitle_x + i] = char
                    style_grid[subtitle_y][subtitle_x + i] = 'cyan'
        
        # Convert to string
        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                char = grid[y][x]
                style = style_grid[y][x]
                if style:
                    line += f"[{style}]{char}[/]"
                else:
                    line += char
            lines.append(line)
        return '\n'.join(lines)