"""TrippyTunnel splash screen effect."""

import math
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("trippy_tunnel")
class TrippyTunnelEffect(BaseEffect):
    """A perspective tunnel effect with shifting, vibrant colors."""
    def __init__(self, parent_widget: Any, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.center_x = self.width / 2
        self.center_y = self.height / 2
        self.time = 0
        self.chars = ".:*#@"

    def update(self) -> Optional[str]:
        self.time += 0.1
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        for y in range(self.height):
            for x in range(self.width):
                dx = x - self.center_x
                dy = (y - self.center_y) * 2
                dist = math.sqrt(dx**2 + dy**2)
                angle = math.atan2(dy, dx)

                if dist > 0:
                    radius = 50 / (dist + 1)
                    v = math.sin(radius - self.time * 2) + math.sin(angle * 3 + self.time)
                    if v > 0.5:
                        char_index = min(len(self.chars) - 1, int((v - 0.5) * 5))
                        grid[y][x] = self.chars[char_index]
                        hue = int((angle * 180 / math.pi + self.time * 100)) % 360
                        r = int(128 + 127 * math.sin(math.radians(hue)))
                        g = int(128 + 127 * math.sin(math.radians(hue + 120)))
                        b = int(128 + 127 * math.sin(math.radians(hue + 240)))
                        styles[y][x] = f"rgb({r},{g},{b})"
        return self._grid_to_string(grid, styles)