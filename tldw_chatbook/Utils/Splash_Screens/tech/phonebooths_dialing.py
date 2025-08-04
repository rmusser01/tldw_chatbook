"""PhoneboothsDialing splash screen effect."""

import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("phonebooths_dialing")
class PhoneboothsDialingEffect(BaseEffect):
    """Animated phonebooths dialing each other."""
    
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
        
        # Phone booth ASCII art
        self.phonebooth = [
            "┌─────┐",
            "│PHONE│",
            "├─────┤",
            "│ ╔═╗ │",
            "│ ║☎║ │",
            "│ ╚═╝ │",
            "│ [#] │",
            "└─────┘"
        ]
        
        self.pulse_position = 0
        self.dialing_phase = 0
        self.connection_established = False
        
    def update(self) -> Optional[str]:
        """Update the phonebooth animation."""
        elapsed = time.time() - self.start_time
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Title
        title = "TLDW CHATBOOK"
        subtitle = "Establishing Connection..."
        title_x = (self.width - len(title)) // 2
        subtitle_x = (self.width - len(subtitle)) // 2
        
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[1][title_x + i] = char
        
        for i, char in enumerate(subtitle):
            if 0 <= subtitle_x + i < self.width:
                grid[3][subtitle_x + i] = char
        
        # Draw phonebooths
        booth1_x = 10
        booth2_x = self.width - 20
        booth_y = 8
        
        # Draw first phonebooth
        for i, line in enumerate(self.phonebooth):
            for j, char in enumerate(line):
                if 0 <= booth1_x + j < self.width and 0 <= booth_y + i < self.height:
                    grid[booth_y + i][booth1_x + j] = char
        
        # Draw second phonebooth
        for i, line in enumerate(self.phonebooth):
            for j, char in enumerate(line):
                if 0 <= booth2_x + j < self.width and 0 <= booth_y + i < self.height:
                    grid[booth_y + i][booth2_x + j] = char
        
        # Draw wire connection
        wire_y = booth_y + len(self.phonebooth) - 1
        wire_start = booth1_x + len(self.phonebooth[0])
        wire_end = booth2_x
        
        for x in range(wire_start, wire_end):
            if 0 <= x < self.width:
                grid[wire_y][x] = '─'
        
        # Animate dialing pulses
        phase = int(elapsed * 4) % 10
        if phase < 5:  # Dialing phase
            # Send pulses from left to right
            pulse_x = wire_start + int((wire_end - wire_start) * (phase / 5))
            if 0 <= pulse_x < self.width:
                grid[wire_y][pulse_x] = '●'
                if pulse_x > 0:
                    grid[wire_y][pulse_x - 1] = '○'
                if pulse_x < self.width - 1:
                    grid[wire_y][pulse_x + 1] = '○'
        else:  # Ringing phase
            # Flash the receiving phone
            if int(elapsed * 8) % 2 == 0:
                # Highlight second booth
                for i in range(3, 6):
                    for j in range(1, 6):
                        if 0 <= booth2_x + j < self.width and 0 <= booth_y + i < self.height:
                            if grid[booth_y + i][booth2_x + j] in '☎':
                                grid[booth_y + i][booth2_x + j] = '☏'
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if char in '●○':
                    line += f"[bold yellow]{char}[/bold yellow]"
                elif char == '☏':
                    line += f"[bold red blink]{char}[/bold red blink]"
                elif char == '☎':
                    line += f"[cyan]{char}[/cyan]"
                elif y == 1 and char != ' ':  # Title
                    line += f"[bold white]{char}[/bold white]"
                elif y == 3 and char != ' ':  # Subtitle
                    line += f"[dim cyan]{char}[/dim cyan]"
                elif char in '┌─┐│├┤└┘╔═╗║╚╝':
                    line += f"[blue]{char}[/blue]"
                elif char == '#':
                    line += f"[dim white]{char}[/dim white]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)