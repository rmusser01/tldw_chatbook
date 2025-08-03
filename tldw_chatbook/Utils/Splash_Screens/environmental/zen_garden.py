"""ZenGarden splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("zen_garden")
class ZenGardenEffect(BaseEffect):
    """Peaceful zen garden with raked sand patterns."""
    
    def __init__(
        self,
        parent_widget: Any,
        width: int = 80,
        height: int = 24,
        speed: float = 0.05,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.width = width
        self.height = height
        self.speed = speed
        
        # Garden elements
        self.stones = [
            (20, 10, "◯"),
            (50, 8, "○"),
            (35, 15, "●"),
            (60, 12, "◉")
        ]
        
        # Sand patterns
        self.rake_progress = 0.0
        self.pattern_type = "circular"  # circular, horizontal, diagonal
        
        # Cherry blossom petals
        self.petals = []
        for _ in range(8):
            self.petals.append({
                'x': random.randint(10, self.width - 10),
                'y': random.randint(-5, 5),
                'speed': random.uniform(0.3, 0.6),
                'drift': random.uniform(-0.2, 0.2),
                'char': random.choice(['✿', '❀', '✾'])
            })
        
        # Water ripples
        self.ripple_center = (65, 10)
        self.ripple_radius = 0.0
        
    def create_sand_pattern(self, x, y, pattern_type):
        """Create sand pattern at given position."""
        if pattern_type == "circular":
            # Calculate distance from stones
            min_dist = float('inf')
            for sx, sy, _ in self.stones:
                dist = math.sqrt((x - sx)**2 + (y - sy)**2)
                min_dist = min(min_dist, dist)
            
            # Create circular pattern
            if int(min_dist) % 3 == 0:
                return '～'
            elif int(min_dist) % 3 == 1:
                return '∽'
            else:
                return ' '
        
        elif pattern_type == "horizontal":
            if y % 2 == 0:
                return '─' if x % 3 != 0 else ' '
            else:
                return ' '
        
        elif pattern_type == "diagonal":
            if (x + y) % 3 == 0:
                return '╱'
            elif (x - y) % 3 == 0:
                return '╲'
            else:
                return ' '
        
        return ' '
    
    def update(self) -> Optional[str]:
        """Update the zen garden animation."""
        elapsed = time.time() - self.start_time
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Update rake progress
        self.rake_progress = (elapsed * 0.2) % 1.0
        
        # Cycle through patterns
        pattern_cycle = int(elapsed / 5) % 3
        if pattern_cycle == 0:
            self.pattern_type = "circular"
        elif pattern_cycle == 1:
            self.pattern_type = "horizontal"
        else:
            self.pattern_type = "diagonal"
        
        # Draw title
        title = "TLDW CHATBOOK"
        subtitle = "Find Your Inner Peace"
        title_x = (self.width - len(title)) // 2
        subtitle_x = (self.width - len(subtitle)) // 2
        
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[2][title_x + i] = char
        
        for i, char in enumerate(subtitle):
            if 0 <= subtitle_x + i < self.width:
                grid[4][subtitle_x + i] = char
        
        # Draw sand patterns (with progressive reveal)
        sand_area = {
            'x_start': 10,
            'x_end': self.width - 10,
            'y_start': 7,
            'y_end': self.height - 4
        }
        
        for y in range(sand_area['y_start'], sand_area['y_end']):
            for x in range(sand_area['x_start'], sand_area['x_end']):
                # Progressive pattern reveal
                reveal_progress = (x - sand_area['x_start']) / (sand_area['x_end'] - sand_area['x_start'])
                if reveal_progress < self.rake_progress:
                    pattern = self.create_sand_pattern(x, y, self.pattern_type)
                    if pattern != ' ':
                        grid[y][x] = pattern
                else:
                    # Unraked sand
                    if random.random() < 0.1:
                        grid[y][x] = '·'
        
        # Draw stones
        for sx, sy, stone in self.stones:
            if 0 <= sx < self.width and 0 <= sy < self.height:
                grid[sy][sx] = stone
                # Stone shadows
                if sx + 1 < self.width and sy + 1 < self.height:
                    if grid[sy + 1][sx + 1] in [' ', '·']:
                        grid[sy + 1][sx + 1] = '░'
        
        # Draw water feature (small pond)
        pond_x, pond_y = self.ripple_center
        pond_radius = 4
        for dy in range(-pond_radius, pond_radius + 1):
            for dx in range(-pond_radius, pond_radius + 1):
                x, y = pond_x + dx, pond_y + dy
                if (0 <= x < self.width and 0 <= y < self.height and 
                    dx*dx + dy*dy <= pond_radius*pond_radius):
                    if dx*dx + dy*dy == pond_radius*pond_radius:
                        grid[y][x] = '○'
                    else:
                        grid[y][x] = '≈'
        
        # Animate water ripples
        self.ripple_radius = (elapsed * 2) % pond_radius
        if self.ripple_radius > 0:
            for angle in range(0, 360, 30):
                rx = pond_x + int(self.ripple_radius * math.cos(math.radians(angle)))
                ry = pond_y + int(self.ripple_radius * math.sin(math.radians(angle)) * 0.5)
                if (0 <= rx < self.width and 0 <= ry < self.height and 
                    grid[ry][rx] == '≈'):
                    grid[ry][rx] = '~'
        
        # Update and draw falling petals
        for petal in self.petals:
            # Update position
            petal['y'] += petal['speed'] * self.speed
            petal['x'] += petal['drift']
            
            # Reset if off screen
            if petal['y'] > self.height:
                petal['y'] = random.randint(-5, 0)
                petal['x'] = random.randint(10, self.width - 10)
            
            # Draw petal
            x, y = int(petal['x']), int(petal['y'])
            if 0 <= x < self.width and 0 <= y < self.height:
                if grid[y][x] == ' ':
                    grid[y][x] = petal['char']
        
        # Draw bamboo decoration
        bamboo_x = 8
        for y in range(8, self.height - 4):
            if bamboo_x < self.width:
                if y % 3 == 0:
                    grid[y][bamboo_x] = '╫'
                else:
                    grid[y][bamboo_x] = '║'
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if y == 2 and char != ' ':  # Title
                    line += f"[bold white]{char}[/bold white]"
                elif y == 4 and char != ' ':  # Subtitle
                    line += f"[dim cyan]{char}[/dim cyan]"
                elif char in '◯○●◉':  # Stones
                    line += f"[white]{char}[/white]"
                elif char == '░':  # Shadows
                    line += f"[dim black]{char}[/dim black]"
                elif char in '～∽─╱╲':  # Sand patterns
                    line += f"[dim yellow]{char}[/dim yellow]"
                elif char == '·':  # Unraked sand
                    line += f"[dim white]{char}[/dim white]"
                elif char in '≈~':  # Water
                    line += f"[blue]{char}[/blue]"
                elif char in '✿❀✾':  # Petals
                    line += f"[light_pink]{char}[/light_pink]"
                elif char in '║╫':  # Bamboo
                    line += f"[green]{char}[/green]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)