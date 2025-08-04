"""OrigamiFolding splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("origami_folding")
class OrigamiFoldingEffect(BaseEffect):
    """Animated origami paper folding effect."""
    
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
        
        # Folding stages
        self.fold_stages = [
            # Stage 1: Flat square
            [
                "┌─────────────┐",
                "│             │",
                "│             │",
                "│             │",
                "│             │",
                "│             │",
                "│             │",
                "└─────────────┘"
            ],
            # Stage 2: First diagonal fold
            [
                "┌─────────────┐",
                "│╲            │",
                "│ ╲           │",
                "│  ╲          │",
                "│   ╲         │",
                "│    ╲        │",
                "│     ╲       │",
                "└──────╲──────┘"
            ],
            # Stage 3: Triangle
            [
                "      ╱╲      ",
                "     ╱  ╲     ",
                "    ╱    ╲    ",
                "   ╱      ╲   ",
                "  ╱        ╲  ",
                " ╱          ╲ ",
                "╱            ╲",
                "──────────────"
            ],
            # Stage 4: Bird base
            [
                "     ╱│╲     ",
                "    ╱ │ ╲    ",
                "   ╱  │  ╲   ",
                "  ╱   │   ╲  ",
                " │    │    │ ",
                " │    │    │ ",
                "  ╲   │   ╱  ",
                "   ╲──┴──╱   "
            ],
            # Stage 5: Crane
            [
                "      ∧      ",
                "     ╱│╲     ",
                "    ╱ │ ╲    ",
                "   <  ●  >   ",
                "    ╲ │ ╱    ",
                "     ╲│╱     ",
                "      │      ",
                "     ╱ ╲     "
            ]
        ]
        
        self.current_stage = 0
        self.stage_progress = 0.0
        self.fold_complete = False
        
    def update(self) -> Optional[str]:
        """Update the origami folding animation."""
        elapsed = time.time() - self.start_time
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Calculate folding progress
        stage_duration = 0.8
        self.current_stage = min(int(elapsed / stage_duration), len(self.fold_stages) - 1)
        self.stage_progress = (elapsed % stage_duration) / stage_duration
        
        if self.current_stage == len(self.fold_stages) - 1:
            self.fold_complete = True
        
        # Draw title
        title = "TLDW CHATBOOK"
        if self.fold_complete:
            subtitle = "Your Story Takes Flight"
        else:
            subtitle = f"Folding... Step {self.current_stage + 1}/{len(self.fold_stages)}"
        
        title_x = (self.width - len(title)) // 2
        subtitle_x = (self.width - len(subtitle)) // 2
        
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[2][title_x + i] = char
        
        for i, char in enumerate(subtitle):
            if 0 <= subtitle_x + i < self.width:
                grid[4][subtitle_x + i] = char
        
        # Draw current folding stage
        paper_x = (self.width - 15) // 2
        paper_y = 8
        
        current_pattern = self.fold_stages[self.current_stage]
        
        # Draw with transition effect
        for i, line in enumerate(current_pattern):
            for j, char in enumerate(line):
                x = paper_x + j
                y = paper_y + i
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Show fold lines progressively
                    if char in '╱╲│─' and self.current_stage > 0:
                        if random.random() < self.stage_progress:
                            grid[y][x] = char
                        elif self.current_stage > 0:
                            # Show previous stage character
                            prev_pattern = self.fold_stages[self.current_stage - 1]
                            if i < len(prev_pattern) and j < len(prev_pattern[i]):
                                prev_char = prev_pattern[i][j]
                                if prev_char != ' ':
                                    grid[y][x] = prev_char
                    else:
                        grid[y][x] = char
        
        # Add fold line indicators
        if not self.fold_complete and self.stage_progress < 0.5:
            fold_indicator = "• • • • •"
            fold_x = (self.width - len(fold_indicator)) // 2
            fold_y = paper_y + len(current_pattern) + 2
            for i, char in enumerate(fold_indicator):
                if 0 <= fold_x + i < self.width and fold_y < self.height:
                    grid[fold_y][fold_x + i] = char
        
        # Add floating effect for completed crane
        if self.fold_complete:
            float_offset = int(math.sin(elapsed * 2) * 2)
            # Clear original position
            for i in range(len(current_pattern)):
                for j in range(len(current_pattern[0])):
                    if paper_y + i < self.height and paper_x + j < self.width:
                        grid[paper_y + i][paper_x + j] = ' '
            
            # Redraw at floating position
            for i, line in enumerate(current_pattern):
                for j, char in enumerate(line):
                    x = paper_x + j
                    y = paper_y + i - float_offset
                    if 0 <= x < self.width and 0 <= y < self.height and char != ' ':
                        grid[y][x] = char
            
            # Add motion lines
            if float_offset > 0:
                for j in range(len(current_pattern[0])):
                    y = paper_y + len(current_pattern) - float_offset
                    x = paper_x + j
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if random.random() < 0.3:
                            grid[y][x] = '·'
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if y == 2 and char != ' ':  # Title
                    line += f"[bold white]{char}[/bold white]"
                elif y == 4 and char != ' ':  # Subtitle
                    line += f"[dim cyan]{char}[/dim cyan]"
                elif char in '┌─┐│└┘':  # Paper outline
                    line += f"[white]{char}[/white]"
                elif char in '╱╲':  # Fold lines
                    line += f"[yellow]{char}[/yellow]"
                elif char == '●':  # Crane eye
                    line += f"[bold red]{char}[/bold red]"
                elif char in '∧<>':  # Crane details
                    line += f"[bold cyan]{char}[/bold cyan]"
                elif char == '·':  # Fold indicators or motion
                    line += f"[dim white]{char}[/dim white]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)