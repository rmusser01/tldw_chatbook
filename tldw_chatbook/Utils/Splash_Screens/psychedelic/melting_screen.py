"""MeltingScreen splash screen effect."""

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("melting_screen")
class MeltingScreenEffect(BaseEffect):
    """An effect where the screen content appears to melt and drip downwards."""
    def __init__(self, parent_widget: Any, content: str = "TLDW MELTDOWN", **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        # Place initial content
        x_start = (self.width - len(content)) // 2
        y_start = self.height // 4
        for i, char in enumerate(content):
            self.grid[y_start][x_start + i] = char
            self.styles[y_start][x_start+i] = f"rgb({random.randint(100,255)},{random.randint(100,255)},{random.randint(100,255)})"

    def update(self) -> Optional[str]:
        for y in range(self.height - 2, -1, -1):
            for x in range(self.width):
                if self.grid[y][x] != ' ' and self.grid[y+1][x] == ' ':
                    if random.random() < 0.2:
                        self.grid[y+1][x] = self.grid[y][x]
                        self.styles[y+1][x] = self.styles[y][x]
                        self.grid[y][x] = ' '
                        self.styles[y][x] = None
        return self._grid_to_string(self.grid, self.styles)