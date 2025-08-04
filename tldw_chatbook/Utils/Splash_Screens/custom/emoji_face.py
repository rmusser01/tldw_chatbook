"""EmojiFace splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("emoji_face")
class EmojiFaceEffect(BaseEffect):
    """Animated emoji face transformation from blank to smirking."""
    
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
        
        # ASCII emoji faces
        self.blank_face = [
            "     ▄▄▄▄▄▄▄     ",
            "   ▄█████████▄   ",
            "  ███████████▌  ",
            " ▐████●███●████▌ ",
            " ████████████████ ",
            " ████████████████ ",
            " ▐██████████████▌ ",
            "  ███████████▌  ",
            "   ▀█████████▀   ",
            "     ▀▀▀▀▀▀▀     "
        ]
        
        self.smirk_face = [
            "     ▄▄▄▄▄▄▄     ",
            "   ▄█████████▄   ",
            "  ███████████▌  ",
            " ▐████●███●████▌ ",
            " ████████████████ ",
            " ████████████████ ",
            " ▐████╱╲████████▌ ",
            "  ███╱  ╲█████▌  ",
            "   ▀█████████▀   ",
            "     ▀▀▀▀▀▀▀     "
        ]
        
        self.transformation_progress = 0.0
        
    def update(self) -> Optional[str]:
        """Update the emoji transformation."""
        elapsed = time.time() - self.start_time
        
        # Calculate transformation progress
        self.transformation_progress = min(1.0, elapsed / 3.0)  # 3 seconds for full transformation
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Title
        title = "TLDW CHATBOOK"
        subtitle = "Loading personality module..." if self.transformation_progress < 1.0 else "Ready to chat! 😏"
        
        title_x = (self.width - len(title)) // 2
        subtitle_x = (self.width - len(subtitle)) // 2
        
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[2][title_x + i] = char
        
        for i, char in enumerate(subtitle):
            if 0 <= subtitle_x + i < self.width:
                grid[4][subtitle_x + i] = char
        
        # Draw face
        face_x = (self.width - len(self.blank_face[0])) // 2
        face_y = (self.height - len(self.blank_face)) // 2 + 2
        
        # Interpolate between blank and smirking face
        for i, (blank_line, smirk_line) in enumerate(zip(self.blank_face, self.smirk_face)):
            for j, (blank_char, smirk_char) in enumerate(zip(blank_line, smirk_line)):
                if 0 <= face_x + j < self.width and 0 <= face_y + i < self.height:
                    # Choose character based on progress
                    if blank_char == smirk_char:
                        grid[face_y + i][face_x + j] = blank_char
                    else:
                        # Transition effect
                        if self.transformation_progress < 0.5:
                            grid[face_y + i][face_x + j] = blank_char
                        elif self.transformation_progress < 0.7:
                            # Glitch effect during transition
                            if random.random() < 0.3:
                                grid[face_y + i][face_x + j] = random.choice(['/', '\\', '─', '│'])
                            else:
                                grid[face_y + i][face_x + j] = blank_char
                        else:
                            grid[face_y + i][face_x + j] = smirk_char
        
        # Add sparkles when transformation is complete
        if self.transformation_progress >= 1.0:
            sparkle_positions = [
                (face_x - 3, face_y + 2),
                (face_x + len(self.blank_face[0]) + 2, face_y + 3),
                (face_x - 2, face_y + 7),
                (face_x + len(self.blank_face[0]) + 1, face_y + 6)
            ]
            sparkle_chars = ['✦', '✧', '★', '✦']
            
            for (sx, sy), sc in zip(sparkle_positions, sparkle_chars):
                if 0 <= sx < self.width and 0 <= sy < self.height:
                    if int(elapsed * 4) % 2 == 0:
                        grid[sy][sx] = sc
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if char in '●':
                    line += f"[bold white]{char}[/bold white]"
                elif char in '▄▀█▌▐':
                    if self.transformation_progress < 1.0:
                        line += f"[yellow]{char}[/yellow]"
                    else:
                        line += f"[bold yellow]{char}[/bold yellow]"
                elif char in '╱╲':
                    line += f"[bold red]{char}[/bold red]"
                elif char in '✦✧★':
                    line += f"[bold cyan blink]{char}[/bold cyan blink]"
                elif y == 2 and char != ' ':  # Title
                    line += f"[bold white]{char}[/bold white]"
                elif y == 4 and char != ' ':  # Subtitle
                    if self.transformation_progress < 1.0:
                        line += f"[dim cyan]{char}[/dim cyan]"
                    else:
                        line += f"[bold green]{char}[/bold green]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)