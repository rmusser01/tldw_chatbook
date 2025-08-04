"""HypnoSwirl splash screen effect."""

import math
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("hypno_swirl")
class HypnoSwirlEffect(BaseEffect):
    """A hypnotic, swirling pattern."""
    def __init__(self, parent_widget: Any, title: str = "tldw", **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.title = title
        self.time = 0
        self.chars = ".:*#@"

    def update(self) -> Optional[str]:
        self.time += 0.1
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]
        center_x, center_y = self.width / 2, self.height / 2

        for y in range(self.height):
            for x in range(self.width):
                dx, dy = x - center_x, (y - center_y) * 2
                angle = math.atan2(dy, dx)
                dist = math.sqrt(dx**2 + dy**2)

                v = math.sin(angle * 5 + dist * 0.5 - self.time * 3)
                if v > 0.5:
                    char_index = min(len(self.chars) - 1, int((v - 0.5) * 5))
                    grid[y][x] = self.chars[char_index]
                    hue = int((angle * 180 / math.pi)) % 360
                    styles[y][x] = f"hsv({hue},1,1)"

        self._add_centered_text(grid, styles, self.title, self.height // 2, 'bold black')
        return self._grid_to_string(grid, styles)