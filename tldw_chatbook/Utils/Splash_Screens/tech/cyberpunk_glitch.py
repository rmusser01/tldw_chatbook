"""CyberpunkGlitch splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("cyberpunk_glitch")
class CyberpunkGlitchEffect(BaseEffect):
    """Futuristic cyberpunk-themed glitch effect with neon colors."""
    
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
        
        # Cyberpunk color palette
        self.neon_colors = [
            "rgb(255,0,255)",  # Magenta
            "rgb(0,255,255)",  # Cyan
            "rgb(255,0,128)",  # Hot pink
            "rgb(128,0,255)",  # Purple
            "rgb(0,255,128)",  # Neon green
        ]
        
        # Glitch parameters
        self.glitch_chars = "▓▒░█▄▀■□▪▫◊◈◆◇○●◐◑◒◓"
        self.corruption_level = 0.8
        self.reveal_progress = 0.0
        self.glitch_zones = []
        self.scanline_y = 0
        
        # Initialize glitch zones
        for _ in range(5):
            self.glitch_zones.append({
                'x': random.randint(0, width - 20),
                'y': random.randint(0, height - 5),
                'w': random.randint(10, 20),
                'h': random.randint(3, 5),
                'intensity': random.uniform(0.3, 1.0)
            })
    
    def update(self) -> Optional[str]:
        """Update cyberpunk glitch effect."""
        elapsed = time.time() - self.start_time
        self.reveal_progress = min(1.0, elapsed / 2.0)
        
        # Create display grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Add background digital noise
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < 0.05:
                    grid[y][x] = random.choice('01')
                    style_grid[y][x] = 'rgb(50,50,50)'
        
        # Update glitch zones
        for zone in self.glitch_zones:
            zone['x'] += random.randint(-2, 2)
            zone['y'] += random.randint(-1, 1)
            zone['x'] = max(0, min(self.width - zone['w'], zone['x']))
            zone['y'] = max(0, min(self.height - zone['h'], zone['y']))
            
            # Draw glitch zone
            if random.random() < zone['intensity']:
                for dy in range(zone['h']):
                    for dx in range(zone['w']):
                        y = zone['y'] + dy
                        x = zone['x'] + dx
                        if 0 <= y < self.height and 0 <= x < self.width:
                            grid[y][x] = random.choice(self.glitch_chars)
                            style_grid[y][x] = random.choice(self.neon_colors)
        
        # Moving scanline effect
        self.scanline_y = (self.scanline_y + 1) % self.height
        for x in range(self.width):
            if random.random() < 0.8:
                grid[self.scanline_y][x] = '─'
                style_grid[self.scanline_y][x] = 'rgb(0,255,255)'
        
        # Title reveal through glitch
        title_y = self.height // 2 - 2
        title_x = (self.width - len(self.title)) // 2
        
        for i, char in enumerate(self.title):
            if title_x + i < self.width:
                if random.random() < self.reveal_progress:
                    # Character is revealed
                    grid[title_y][title_x + i] = char
                    if random.random() < 0.9:  # Occasional glitch
                        style_grid[title_y][title_x + i] = 'bold rgb(255,0,255)'
                    else:
                        style_grid[title_y][title_x + i] = random.choice(self.neon_colors)
                else:
                    # Character is still corrupted
                    grid[title_y][title_x + i] = random.choice(self.glitch_chars)
                    style_grid[title_y][title_x + i] = random.choice(self.neon_colors)
        
        # Subtitle with glitch reveal
        if self.subtitle and elapsed > 1.0:
            subtitle_progress = min(1.0, (elapsed - 1.0) / 1.0)
            subtitle_y = self.height // 2
            subtitle_x = (self.width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                if subtitle_x + i < self.width:
                    if random.random() < subtitle_progress:
                        grid[subtitle_y][subtitle_x + i] = char
                        style_grid[subtitle_y][subtitle_x + i] = 'rgb(0,255,255)'
        
        # Add random glitch artifacts
        for _ in range(20):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            grid[y][x] = random.choice(self.glitch_chars)
            style_grid[y][x] = random.choice(self.neon_colors)
        
        # Convert to string
        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                char = grid[y][x]
                style = style_grid[y][x]
                if style:
                    line += f"[{style}]{char}[/{style.split()[0]}]"
                else:
                    line += char
            lines.append(line)
        return '\n'.join(lines)