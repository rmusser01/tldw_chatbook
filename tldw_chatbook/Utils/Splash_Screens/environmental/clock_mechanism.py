"""ClockMechanism splash screen effect."""

import math
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("clock_mechanism")
class ClockMechanismEffect(BaseEffect):
    """Animated clock mechanism with gears and pendulum."""
    
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
        
        # Gear designs
        self.large_gear = [
            "   ╱═╲   ",
            "  ╱   ╲  ",
            " │  ●  │ ",
            "  ╲   ╱  ",
            "   ╲═╱   "
        ]
        
        self.small_gear = [
            " ╱╲ ",
            "│●│",
            " ╲╱ "
        ]
        
        # Gear positions and speeds
        self.gears = [
            {'x': 20, 'y': 8, 'size': 'large', 'speed': 1.0, 'rotation': 0},
            {'x': 35, 'y': 7, 'size': 'small', 'speed': -2.0, 'rotation': 0},
            {'x': 45, 'y': 10, 'size': 'small', 'speed': 2.5, 'rotation': 0},
            {'x': 55, 'y': 8, 'size': 'large', 'speed': -0.8, 'rotation': 0}
        ]
        
        # Pendulum state
        self.pendulum_angle = 0
        self.pendulum_center_x = self.width // 2
        self.pendulum_top_y = 15
        
    def rotate_gear(self, gear_pattern, angle):
        """Rotate gear pattern based on angle."""
        # Simple rotation by cycling characters
        rotation_chars = ['╱', '─', '╲', '│']
        rotated = []
        
        for line in gear_pattern:
            new_line = ""
            for char in line:
                if char in rotation_chars:
                    idx = rotation_chars.index(char)
                    new_idx = (idx + int(angle / 90)) % len(rotation_chars)
                    new_line += rotation_chars[new_idx]
                else:
                    new_line += char
            rotated.append(new_line)
        
        return rotated
    
    def update(self) -> Optional[str]:
        """Update the clock mechanism animation."""
        elapsed = time.time() - self.start_time
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw title
        title = "TLDW CHATBOOK"
        subtitle = "Time to Connect"
        title_x = (self.width - len(title)) // 2
        subtitle_x = (self.width - len(subtitle)) // 2
        
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[2][title_x + i] = char
        
        for i, char in enumerate(subtitle):
            if 0 <= subtitle_x + i < self.width:
                grid[4][subtitle_x + i] = char
        
        # Update and draw gears
        for gear in self.gears:
            # Update rotation
            gear['rotation'] = (gear['rotation'] + gear['speed'] * 10) % 360
            
            # Get gear pattern
            if gear['size'] == 'large':
                pattern = self.rotate_gear(self.large_gear, gear['rotation'])
            else:
                pattern = self.rotate_gear(self.small_gear, gear['rotation'])
            
            # Draw gear
            for i, line in enumerate(pattern):
                for j, char in enumerate(line):
                    x = gear['x'] + j
                    y = gear['y'] + i
                    if 0 <= x < self.width and 0 <= y < self.height and char != ' ':
                        grid[y][x] = char
        
        # Update and draw pendulum
        self.pendulum_angle = math.sin(elapsed * 2) * 0.5  # Swing angle in radians
        pendulum_length = 6
        
        # Draw pendulum rod
        for i in range(pendulum_length):
            x = self.pendulum_center_x + int(math.sin(self.pendulum_angle) * i)
            y = self.pendulum_top_y + i
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = '│'
        
        # Draw pendulum bob
        bob_x = self.pendulum_center_x + int(math.sin(self.pendulum_angle) * pendulum_length)
        bob_y = self.pendulum_top_y + pendulum_length
        if 0 <= bob_x - 1 < self.width - 2 and 0 <= bob_y < self.height:
            grid[bob_y][bob_x - 1] = '('
            grid[bob_y][bob_x] = '●'
            grid[bob_y][bob_x + 1] = ')'
        
        # Draw clock frame elements
        frame_chars = "⚙"
        for i in range(5):
            x = 10 + i * 15
            y = 6
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = frame_chars
        
        # Draw time display (showing animation progress)
        time_str = f"{int(elapsed % 60):02d}:{int((elapsed * 10) % 60):02d}"
        time_x = (self.width - len(time_str)) // 2
        time_y = self.height - 3
        for i, char in enumerate(time_str):
            if 0 <= time_x + i < self.width:
                grid[time_y][time_x + i] = char
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if y == 2 and char != ' ':  # Title
                    line += f"[bold white]{char}[/bold white]"
                elif y == 4 and char != ' ':  # Subtitle
                    line += f"[dim cyan]{char}[/dim cyan]"
                elif char == '●':  # Gear centers and pendulum bob
                    line += f"[bold yellow]{char}[/bold yellow]"
                elif char in '╱╲│─═':  # Gear teeth
                    line += f"[cyan]{char}[/cyan]"
                elif char in '()':  # Pendulum bob outline
                    line += f"[yellow]{char}[/yellow]"
                elif char == '⚙':  # Frame decorations
                    line += f"[dim white]{char}[/dim white]"
                elif char.isdigit() or char == ':':  # Time
                    line += f"[bold green]{char}[/bold green]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)