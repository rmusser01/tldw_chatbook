"""DeepDream splash screen effect."""

from tldw_chatbook.Utils.Splash import get_ascii_art
import math
import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("deep_dream")
class DeepDreamEffect(BaseEffect):
    """Simulates a 'deep dream' effect with recursive, layered patterns."""
    def __init__(self, parent_widget: Any, content: str = "", **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.base_content = get_ascii_art("default")
        self.lines = self.base_content.splitlines()
        self.height = len(self.lines)
        self.width = max(len(line) for line in self.lines) if self.lines else 0
        self.time = 0
        self.overlay_chars = "()∙○●◎"
        self.overlay_colors = ["rgb(255,0,255)", "rgb(0,255,0)", "rgb(255,255,0)"]

    def update(self) -> Optional[str]:
        self.time += 0.1
        grid = [list(line.ljust(self.width)) for line in self.lines]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        for y in range(self.height):
            for x in range(self.width):
                if grid[y][x] != ' ':
                    # Add recursive patterns based on time and position
                    v = math.sin(x * 0.2 + self.time) + math.cos(y * 0.2 + self.time)
                    if v > 1.2:
                        grid[y][x] = random.choice(self.overlay_chars)
                        styles[y][x] = random.choice(self.overlay_colors)
        return self._grid_to_string(grid, styles)