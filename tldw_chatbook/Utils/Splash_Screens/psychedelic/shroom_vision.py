"""ShroomVision splash screen effect."""

import math
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("shroom_vision")
class ShroomVisionEffect(BaseEffect):
    """Simulates a 'mushroom vision' effect with distorted, breathing visuals."""
    def __init__(self, parent_widget: Any, title: str = "tldw chatbook", **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.title = title
        self.time = 0

    def update(self) -> Optional[str]:
        self.time += 0.05
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]
        center_x, center_y = self.width / 2, self.height / 2

        for y in range(self.height):
            for x in range(self.width):
                dx, dy = x - center_x, y - center_y
                dist = math.sqrt(dx**2 + dy**2)
                angle = math.atan2(dy, dx)

                # Breathing effect
                dist_factor = math.sin(dist * 0.5 - self.time * 2) * 2

                new_x = int(center_x + (dx + dist_factor * math.cos(angle)))
                new_y = int(center_y + (dy + dist_factor * math.sin(angle)) * 0.5)

                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    title_x_start = (self.width - len(self.title)) // 2
                    if new_y == self.height // 2 and title_x_start <= new_x < title_x_start + len(self.title):
                         char_index = new_x - title_x_start
                         grid[y][x] = self.title[char_index]
                         hue = int((dist * 10 + self.time * 50)) % 360
                         r = int(128 + 127 * math.sin(math.radians(hue)))
                         g = int(128 + 127 * math.sin(math.radians(hue + 120)))
                         b = int(128 + 127 * math.sin(math.radians(hue + 240)))
                         styles[y][x] = f"bold rgb({r},{g},{b})"
        return self._grid_to_string(grid, styles)