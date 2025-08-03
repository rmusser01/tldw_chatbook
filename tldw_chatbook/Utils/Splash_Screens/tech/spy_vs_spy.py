"""SpyVsSpy splash screen effect."""

import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("spy_vs_spy")
class SpyVsSpyEffect(BaseEffect):
    """Animated Spy vs Spy confrontation effect."""
    
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
        
        # Spy ASCII art (simplified for animation)
        self.spy_white = [
            "   ▄▄▄   ",
            "  █░░░█  ",
            "  █▀▀▀█  ",
            " ▄█───█▄ ",
            " █ ▐▌ █ ",
            " █  ▌  █ ",
            " ▀█▄▄▄█▀ ",
            "   █ █   ",
            "  █   █  "
        ]
        
        self.spy_black = [
            "   ▄▄▄   ",
            "  █▓▓▓█  ",
            "  █▀▀▀█  ",
            " ▄█───█▄ ",
            " █ ▐▌ █ ",
            " █  ▌  █ ",
            " ▀█▄▄▄█▀ ",
            "   █ █   ",
            "  █   █  "
        ]
        
        # Animation states
        self.white_pos = 10
        self.black_pos = width - 20
        self.action_phase = 0
        self.explosion_frame = -1
        self.trap_x = width // 2
        
    def update(self) -> Optional[str]:
        """Update the spy animation."""
        elapsed = time.time() - self.start_time
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Calculate spy positions and actions
        phase = int(elapsed * 2) % 8
        
        # Move spies closer
        if phase < 3:
            self.white_pos = min(self.width // 2 - 15, self.white_pos + 1)
            self.black_pos = max(self.width // 2 + 5, self.black_pos - 1)
        
        # Draw title
        title = "SPY vs SPY"
        title_x = (self.width - len(title)) // 2
        title_y = 2
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[title_y][title_x + i] = char
        
        # Draw floor
        floor_y = self.height - 8
        for x in range(self.width):
            grid[floor_y][x] = '─'
        
        # Draw trap or explosion
        if phase == 4:
            # Set trap
            trap_art = ["╱╲", "╲╱"]
            for i, line in enumerate(trap_art):
                for j, char in enumerate(line):
                    if 0 <= self.trap_x + j < self.width and floor_y - 2 + i < self.height:
                        grid[floor_y - 2 + i][self.trap_x + j] = char
        elif phase >= 5 and phase <= 6:
            # Explosion
            explosion = ["  ╱▓╲  ", " ╱▓▓▓╲ ", "╱▓▓▓▓▓╲", "▓▓▓▓▓▓▓", " ▀▀▀▀▀ "]
            exp_x = self.trap_x - 3
            exp_y = floor_y - len(explosion)
            for i, line in enumerate(explosion):
                for j, char in enumerate(line):
                    if 0 <= exp_x + j < self.width and 0 <= exp_y + i < self.height:
                        grid[exp_y + i][exp_x + j] = char
        
        # Draw white spy
        spy_y = floor_y - len(self.spy_white)
        for i, line in enumerate(self.spy_white):
            for j, char in enumerate(line):
                if char != ' ' and 0 <= self.white_pos + j < self.width and 0 <= spy_y + i < self.height:
                    grid[spy_y + i][self.white_pos + j] = char
        
        # Draw black spy
        for i, line in enumerate(self.spy_black):
            for j, char in enumerate(line):
                if char != ' ' and 0 <= self.black_pos + j < self.width and 0 <= spy_y + i < self.height:
                    grid[spy_y + i][self.black_pos + j] = char
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if char == '▓':
                    line += f"[bold red]{char}[/bold red]"
                elif char in '░':
                    line += f"[white]{char}[/white]"
                elif y == title_y and char != ' ':
                    line += f"[bold yellow]{char}[/bold yellow]"
                elif char == '─':
                    line += f"[dim white]{char}[/dim white]"
                elif char in '╱╲╲╱':
                    line += f"[yellow]{char}[/yellow]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)