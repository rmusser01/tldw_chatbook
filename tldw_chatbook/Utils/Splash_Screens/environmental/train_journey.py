"""TrainJourney splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("train_journey")
class TrainJourneyEffect(BaseEffect):
    """Animated train journey with parallax scrolling landscape."""
    
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
        
        # Train ASCII art
        self.train = [
            "    ___    ",
            "   |   |   ",
            "  /|___|\  ",
            " /_|_O_|_\ ",
            "  o o o o  "
        ]
        
        # Landscape elements
        self.mountains = "▲▲▲  ▲▲  ▲▲▲▲  ▲  ▲▲▲"
        self.trees = "↟ ↟↟ ↟ ↟↟↟ ↟ ↟↟ ↟"
        self.clouds = ["☁", "☁☁", "☁☁☁"]
        
        # Animation offsets
        self.track_offset = 0
        self.landscape_offset = 0
        self.cloud_positions = [(10, 3), (30, 4), (50, 3), (70, 5)]
        
    def update(self) -> Optional[str]:
        """Update the train journey animation."""
        elapsed = time.time() - self.start_time
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Update offsets
        self.track_offset = int(elapsed * 20) % 4
        self.landscape_offset = int(elapsed * 5) % self.width
        
        # Draw sky and clouds
        for cloud_x, cloud_y in self.cloud_positions:
            cloud = random.choice(self.clouds)
            x = int((cloud_x - elapsed * 3) % self.width)
            for i, char in enumerate(cloud):
                if 0 <= x + i < self.width and 0 <= cloud_y < self.height:
                    grid[cloud_y][x + i] = char
        
        # Draw mountains (far background)
        mountain_y = 8
        for i in range(self.width):
            mountain_idx = (i + self.landscape_offset // 3) % len(self.mountains)
            if self.mountains[mountain_idx] != ' ':
                grid[mountain_y][i] = self.mountains[mountain_idx]
        
        # Draw trees (near background)
        tree_y = 12
        for i in range(self.width):
            tree_idx = (i + self.landscape_offset // 2) % len(self.trees)
            if self.trees[tree_idx] != ' ':
                grid[tree_y][i] = self.trees[tree_idx]
        
        # Draw train
        train_x = 20
        train_y = 14
        for i, line in enumerate(self.train):
            for j, char in enumerate(line):
                if 0 <= train_x + j < self.width and 0 <= train_y + i < self.height:
                    if char != ' ':
                        grid[train_y + i][train_x + j] = char
        
        # Draw smoke puffs
        smoke_x = train_x + 5
        smoke_y = train_y - 1
        smoke_phase = int(elapsed * 3) % 3
        if smoke_phase == 0:
            if 0 <= smoke_x < self.width and 0 <= smoke_y < self.height:
                grid[smoke_y][smoke_x] = 'o'
        elif smoke_phase == 1:
            if 0 <= smoke_x - 1 < self.width and 0 <= smoke_y - 1 < self.height:
                grid[smoke_y - 1][smoke_x - 1] = 'O'
        else:
            if 0 <= smoke_x - 2 < self.width and 0 <= smoke_y - 2 < self.height:
                grid[smoke_y - 2][smoke_x - 2] = '○'
        
        # Draw tracks
        track_y = train_y + len(self.train)
        track_pattern = "═╪═"
        for x in range(self.width):
            pattern_idx = (x + self.track_offset) % len(track_pattern)
            grid[track_y][x] = track_pattern[pattern_idx]
        
        # Draw station sign (appears later)
        if elapsed > 2.0:
            sign_x = int(self.width - (elapsed - 2.0) * 15)
            if 10 < sign_x < self.width - 20:
                sign = ["┌─────────────┐", "│ TLDW CHATBOOK │", "└─────────────┘"]
                sign_y = 9
                for i, line in enumerate(sign):
                    for j, char in enumerate(line):
                        if 0 <= sign_x + j < self.width and 0 <= sign_y + i < self.height:
                            grid[sign_y + i][sign_x + j] = char
        
        # Draw title
        title = "All Aboard!"
        title_x = (self.width - len(title)) // 2
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[2][title_x + i] = char
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if y == 2 and char != ' ':  # Title
                    line += f"[bold white]{char}[/bold white]"
                elif char == '☁':  # Clouds
                    line += f"[white]{char}[/white]"
                elif char == '▲':  # Mountains
                    line += f"[dim blue]{char}[/dim blue]"
                elif char == '↟':  # Trees
                    line += f"[green]{char}[/green]"
                elif char in 'oO○':  # Smoke
                    line += f"[dim white]{char}[/dim white]"
                elif char in '═╪':  # Tracks
                    line += f"[yellow]{char}[/yellow]"
                elif char in '┌─┐│└┘' or "TLDW CHATBOOK" in ''.join(row):  # Sign
                    if "TLDW CHATBOOK" in ''.join(row) and char.isalpha():
                        line += f"[bold cyan]{char}[/bold cyan]"
                    else:
                        line += f"[white]{char}[/white]"
                elif char in '|/_O\\o':  # Train
                    line += f"[bold red]{char}[/bold red]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)