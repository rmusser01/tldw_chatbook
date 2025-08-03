"""PixelDissolve splash screen effect."""

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("pixel_dissolve")
class PixelDissolveEffect(BaseEffect):
    """The screen starts filled with random ASCII characters that gradually dissolve away."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Revealing clarity...",
        width: int = 80,
        height: int = 24,
        dissolve_rate: float = 0.02,  # Percentage per frame
        noise_chars: str = "█▓▒░╳╱╲┃━┏┓┗┛",
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.dissolve_rate = dissolve_rate
        self.noise_chars = noise_chars
        self.title_style = title_style
        
        # Initialize with all pixels as noise
        self.dissolved_pixels = set()
        self.total_pixels = width * height
        
    def update(self) -> Optional[str]:
        # Calculate how many pixels to dissolve this frame
        current_dissolved = len(self.dissolved_pixels)
        target_dissolved = min(self.total_pixels, 
                              current_dissolved + int(self.total_pixels * self.dissolve_rate))
        
        # Dissolve random pixels
        while len(self.dissolved_pixels) < target_dissolved:
            x = random.randint(0, self.display_width - 1)
            y = random.randint(0, self.display_height - 1)
            self.dissolved_pixels.add((x, y))
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Fill with noise or clear based on dissolved state
        for y in range(self.display_height):
            for x in range(self.display_width):
                if (x, y) not in self.dissolved_pixels:
                    grid[y][x] = random.choice(self.noise_chars)
                    styles[y][x] = random.choice(["dim white", "dim gray", "dim black"])
        
        # Always show title and subtitle (on top of noise)
        if self.title:
            title_y = self.display_height // 2 - 2
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
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