"""AntColony splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("ant_colony")
class AntColonyEffect(BaseEffect):
    """Animated ant colony building tunnels."""
    
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
        
        # Ant representation
        self.ant_chars = ['∴', ':', '⁚']
        
        # Initialize ants
        self.ants = []
        for _ in range(30):
            self.ants.append({
                'x': random.randint(10, self.width - 10),
                'y': random.randint(10, self.height - 5),
                'vx': random.choice([-1, 0, 1]),
                'vy': random.choice([-1, 0, 1]),
                'carrying': random.random() < 0.3,
                'target': None
            })
        
        # Tunnel system
        self.tunnels = set()
        self.queen_chamber = (self.width // 2, self.height // 2)
        self.food_sources = [
            (15, 8), (65, 8), (40, 18)
        ]
        
        # Pheromone trails
        self.pheromones = {}
        
        # Message to build
        self.message = "TLDW"
        self.message_progress = 0.0
        
    def update(self) -> Optional[str]:
        """Update the ant colony animation."""
        elapsed = time.time() - self.start_time
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Update message building progress
        self.message_progress = min(1.0, elapsed / 4.0)
        
        # Draw title
        title = "TLDW CHATBOOK"
        subtitle = "Building Connections..."
        title_x = (self.width - len(title)) // 2
        subtitle_x = (self.width - len(subtitle)) // 2
        
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[2][title_x + i] = char
        
        for i, char in enumerate(subtitle):
            if 0 <= subtitle_x + i < self.width:
                grid[4][subtitle_x + i] = char
        
        # Draw queen chamber
        qx, qy = self.queen_chamber
        chamber_size = 3
        for dy in range(-chamber_size, chamber_size + 1):
            for dx in range(-chamber_size, chamber_size + 1):
                x, y = qx + dx, qy + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    if abs(dx) == chamber_size or abs(dy) == chamber_size:
                        grid[y][x] = '○'
                    elif dx == 0 and dy == 0:
                        grid[y][x] = '♛'  # Queen
        
        # Draw food sources
        for fx, fy in self.food_sources:
            if 0 <= fx < self.width and 0 <= fy < self.height:
                grid[fy][fx] = '●'
        
        # Update and draw ants
        for ant in self.ants:
            # Simple ant AI
            if not ant['target']:
                if ant['carrying']:
                    # Go to queen
                    ant['target'] = self.queen_chamber
                else:
                    # Go to random food source
                    ant['target'] = random.choice(self.food_sources)
            
            # Move towards target
            tx, ty = ant['target']
            if ant['x'] < tx:
                ant['vx'] = 1
            elif ant['x'] > tx:
                ant['vx'] = -1
            else:
                ant['vx'] = 0
            
            if ant['y'] < ty:
                ant['vy'] = 1
            elif ant['y'] > ty:
                ant['vy'] = -1
            else:
                ant['vy'] = 0
            
            # Update position
            ant['x'] += ant['vx']
            ant['y'] += ant['vy']
            
            # Add to tunnels
            self.tunnels.add((ant['x'], ant['y']))
            
            # Leave pheromone trail
            if ant['carrying']:
                self.pheromones[(ant['x'], ant['y'])] = 5
            
            # Check if reached target
            if (ant['x'], ant['y']) == ant['target']:
                ant['carrying'] = not ant['carrying']
                ant['target'] = None
            
            # Draw ant
            if 0 <= ant['x'] < self.width and 0 <= ant['y'] < self.height:
                ant_char = self.ant_chars[hash((ant['x'], ant['y'])) % len(self.ant_chars)]
                grid[ant['y']][ant['x']] = ant_char
        
        # Draw tunnels
        for (x, y) in self.tunnels:
            if 0 <= x < self.width and 0 <= y < self.height:
                if grid[y][x] == ' ':
                    grid[y][x] = '·'
        
        # Draw pheromone trails
        pheromones_to_remove = []
        for (x, y), strength in self.pheromones.items():
            if strength > 0:
                if 0 <= x < self.width and 0 <= y < self.height:
                    if grid[y][x] == ' ' or grid[y][x] == '·':
                        if strength > 3:
                            grid[y][x] = '▪'
                        else:
                            grid[y][x] = '·'
                self.pheromones[(x, y)] = strength - 0.1
            else:
                pheromones_to_remove.append((x, y))
        
        for pos in pheromones_to_remove:
            del self.pheromones[pos]
        
        # Build message with tunnels
        if self.message_progress > 0:
            message_y = 15
            message_x = (self.width - len(self.message) * 6) // 2
            
            # Simple block letters for TLDW
            letter_patterns = {
                'T': ["█████", "  █  ", "  █  "],
                'L': ["█    ", "█    ", "█████"],
                'D': ["████ ", "█   █", "████ "],
                'W': ["█   █", "█ █ █", "█████"]
            }
            
            for i, letter in enumerate(self.message):
                if letter in letter_patterns and random.random() < self.message_progress:
                    pattern = letter_patterns[letter]
                    for row_idx, row in enumerate(pattern):
                        for col_idx, char in enumerate(row):
                            x = message_x + i * 6 + col_idx
                            y = message_y + row_idx
                            if 0 <= x < self.width and 0 <= y < self.height:
                                if char == '█':
                                    grid[y][x] = '▓'
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if y == 2 and char != ' ':  # Title
                    line += f"[bold white]{char}[/bold white]"
                elif y == 4 and char != ' ':  # Subtitle
                    line += f"[dim cyan]{char}[/dim cyan]"
                elif char == '♛':  # Queen
                    line += f"[bold yellow]{char}[/bold yellow]"
                elif char == '○':  # Chamber walls
                    line += f"[yellow]{char}[/yellow]"
                elif char == '●':  # Food
                    line += f"[green]{char}[/green]"
                elif char in '∴:⁚':  # Ants
                    line += f"[red]{char}[/red]"
                elif char == '▪':  # Strong pheromone
                    line += f"[bright_magenta]{char}[/bright_magenta]"
                elif char == '·':  # Tunnel/weak pheromone
                    line += f"[dim white]{char}[/dim white]"
                elif char == '▓':  # Message blocks
                    line += f"[bold cyan]{char}[/bold cyan]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)