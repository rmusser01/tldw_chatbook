"""SpiralGalaxy splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("spiral_galaxy")
class SpiralGalaxyEffect(BaseEffect):
    """Creates a rotating spiral galaxy pattern with ASCII stars and cosmic dust."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Exploring the cosmos...",
        width: int = 80,
        height: int = 24,
        rotation_speed: float = 0.2,  # Radians per second
        num_arms: int = 3,
        star_chars: str = "·∙*✦✧★☆",
        star_colors: List[str] = ["white", "bright_white", "yellow", "cyan", "dim white"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.rotation_speed = rotation_speed
        self.num_arms = num_arms
        self.star_chars = star_chars
        self.star_colors = star_colors
        self.title_style = title_style
        
        self.rotation = 0.0
        self.last_update_time = time.time()
        self.stars = []
        
        self._generate_stars()
        
    def _generate_stars(self):
        """Generate stars in spiral pattern."""
        center_x = self.display_width / 2
        center_y = self.display_height / 2
        
        for i in range(200):  # Number of stars
            # Spiral parameters
            angle = random.uniform(0, 4 * math.pi)
            radius = random.uniform(0, min(center_x, center_y) - 5)
            
            # Add arm structure
            arm = random.randint(0, self.num_arms - 1)
            arm_angle = (2 * math.pi * arm) / self.num_arms
            
            # Logarithmic spiral
            spiral_angle = angle + arm_angle + (radius * 0.1)
            
            x = center_x + radius * math.cos(spiral_angle)
            y = center_y + radius * math.sin(spiral_angle) * 0.5  # Aspect ratio
            
            char = random.choice(self.star_chars)
            color = random.choice(self.star_colors)
            
            self.stars.append({
                'radius': radius,
                'angle': spiral_angle,
                'char': char,
                'color': color,
                'brightness': random.uniform(0.3, 1.0)
            })
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.rotation += self.rotation_speed * delta_time
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        center_x = self.display_width / 2
        center_y = self.display_height / 2
        
        # Draw stars
        for star in self.stars:
            # Rotate star
            angle = star['angle'] + self.rotation
            x = int(center_x + star['radius'] * math.cos(angle))
            y = int(center_y + star['radius'] * math.sin(angle) * 0.5)
            
            if 0 <= x < self.display_width and 0 <= y < self.display_height:
                # Twinkle effect
                if random.random() < star['brightness']:
                    grid[y][x] = star['char']
                    styles[y][x] = star['color']
        
        # Draw title emerging from galactic center
        if self.title:
            title_y = int(center_y)
            title_x = int((self.display_width - len(self.title)) / 2)
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Draw subtitle
        if self.subtitle:
            subtitle_y = int(center_y) + 3
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