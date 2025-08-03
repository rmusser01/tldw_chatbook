"""ASCIIAquarium splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("ascii_aquarium")
class ASCIIAquariumEffect(BaseEffect):
    """Animated ASCII aquarium with swimming fish and bubbles."""
    
    def __init__(
        self,
        parent_widget: Any,
        width: int = 80,
        height: int = 24,
        speed: float = 0.1,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.width = width
        self.height = height
        self.speed = speed
        
        # Different fish types
        self.fish_types = [
            "><>",
            "<><",
            "><(((*>",
            "<*)))><",
            "≈°))<><",
            "><>°≈",
            "◉◉><>",
            "<><◉◉"
        ]
        
        # Initialize fish with random positions and speeds
        self.fish = []
        for i in range(12):
            self.fish.append({
                'type': random.choice(self.fish_types),
                'x': random.uniform(-10, self.width + 10),
                'y': random.randint(6, self.height - 4),
                'speed': random.uniform(0.5, 2.0),
                'direction': random.choice([-1, 1]),
                'color': random.choice(['cyan', 'blue', 'yellow', 'green'])
            })
        
        # Initialize bubbles
        self.bubbles = []
        for i in range(15):
            self.bubbles.append({
                'x': random.randint(10, self.width - 10),
                'y': random.uniform(self.height - 2, self.height + 5),
                'speed': random.uniform(0.3, 0.8),
                'size': random.choice(['°', 'o', 'O', '○'])
            })
        
        # Seaweed positions
        self.seaweed_positions = [15, 25, 35, 50, 65]
        
    def update(self) -> Optional[str]:
        """Update the aquarium animation."""
        elapsed = time.time() - self.start_time
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw title
        title = "TLDW CHATBOOK"
        subtitle = "Diving into conversations..."
        title_x = (self.width - len(title)) // 2
        subtitle_x = (self.width - len(subtitle)) // 2
        
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[1][title_x + i] = char
        
        for i, char in enumerate(subtitle):
            if 0 <= subtitle_x + i < self.width:
                grid[3][subtitle_x + i] = char
        
        # Draw water surface
        water_y = 5
        for x in range(self.width):
            wave = '~' if int(elapsed * 4 + x) % 3 == 0 else '≈'
            grid[water_y][x] = wave
        
        # Draw seaweed
        for pos in self.seaweed_positions:
            if 0 <= pos < self.width:
                height = random.randint(3, 6)
                for y in range(self.height - height, self.height - 1):
                    if y < self.height:
                        sway = int(math.sin(elapsed * 2 + pos) * 2)
                        x = pos + sway
                        if 0 <= x < self.width:
                            grid[y][x] = '|' if random.random() > 0.3 else ')'
        
        # Update and draw fish
        for fish in self.fish:
            # Update position
            fish['x'] += fish['speed'] * fish['direction'] * self.speed
            
            # Wrap around screen
            if fish['direction'] > 0 and fish['x'] > self.width + 10:
                fish['x'] = -10
                fish['y'] = random.randint(6, self.height - 4)
            elif fish['direction'] < 0 and fish['x'] < -10:
                fish['x'] = self.width + 10
                fish['y'] = random.randint(6, self.height - 4)
            
            # Draw fish
            fish_chars = fish['type'] if fish['direction'] > 0 else fish['type'][::-1]
            for i, char in enumerate(fish_chars):
                x = int(fish['x']) + i
                y = fish['y']
                if 0 <= x < self.width and 0 <= y < self.height:
                    grid[y][x] = char
        
        # Update and draw bubbles
        for bubble in self.bubbles:
            # Update position
            bubble['y'] -= bubble['speed'] * self.speed
            
            # Reset bubble at bottom
            if bubble['y'] < water_y + 1:
                bubble['y'] = self.height - 2
                bubble['x'] = random.randint(10, self.width - 10)
            
            # Draw bubble
            x = int(bubble['x'] + math.sin(elapsed * 3 + bubble['y']) * 2)
            y = int(bubble['y'])
            if 0 <= x < self.width and water_y < y < self.height:
                grid[y][x] = bubble['size']
        
        # Draw bottom
        for x in range(self.width):
            grid[self.height - 2][x] = '▓'
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if y == 1 and char != ' ':  # Title
                    line += f"[bold white]{char}[/bold white]"
                elif y == 3 and char != ' ':  # Subtitle
                    line += f"[dim cyan]{char}[/dim cyan]"
                elif y == water_y and char in '~≈':  # Water surface
                    line += f"[blue]{char}[/blue]"
                elif char in '><*()°':  # Fish
                    # Find which fish this belongs to
                    for fish in self.fish:
                        if y == fish['y'] and abs(x - fish['x']) < len(fish['type']):
                            line += f"[{fish['color']}]{char}[/{fish['color']}]"
                            break
                    else:
                        line += char
                elif char in 'oO○°' and y > water_y:  # Bubbles
                    line += f"[bright_cyan]{char}[/bright_cyan]"
                elif char in '|)' and y > self.height - 8:  # Seaweed
                    line += f"[green]{char}[/green]"
                elif char == '▓':  # Bottom
                    line += f"[dim yellow]{char}[/dim yellow]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)