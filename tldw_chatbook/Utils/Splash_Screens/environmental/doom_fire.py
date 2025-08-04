"""DoomFire splash screen effect."""

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("doom_fire")
class DoomFireEffect(BaseEffect):
    """A classic 'Doom' style fire effect."""

    def __init__(self, parent_widget: Any, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.fire_grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.palette = [
            (0, 0, 0), (7, 7, 7), (31, 7, 7), (47, 15, 7), (71, 15, 7),
            (87, 23, 7), (103, 31, 7), (119, 31, 7), (143, 39, 7), (159, 47, 7),
            (175, 55, 7), (191, 55, 7), (199, 63, 7), (207, 71, 7), (223, 79, 7),
            (223, 87, 7), (223, 87, 7), (215, 95, 7), (207, 103, 15), (199, 111, 15),
            (191, 119, 15), (183, 127, 15), (175, 135, 23), (167, 143, 23),
            (159, 151, 23), (159, 159, 31), (159, 167, 39), (159, 175, 47),
            (159, 183, 55), (159, 191, 55), (159, 199, 63), (167, 207, 71),
            (175, 215, 79), (183, 223, 87), (191, 231, 95), (199, 239, 103),
            (207, 247, 111), (215, 255, 119)
        ]
        self.chars = " ....,,,;;;+++!!!iii|||$$$@@@"

    def update(self) -> Optional[str]:
        for x in range(self.width):
            self.fire_grid[self.height - 1][x] = random.randint(0, len(self.palette) - 1)

        for y in range(self.height - 1):
            for x in range(self.width):
                src_y = y + 1
                rand_x = (x + random.randint(-1, 1) + self.width) % self.width
                new_val = self.fire_grid[src_y][rand_x] - random.randint(0, 1)
                self.fire_grid[y][x] = max(0, new_val)

        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        for y in range(self.height):
            for x in range(self.width):
                val = self.fire_grid[y][x]
                if val > 0:
                    char_index = min(len(self.chars) - 1, val)
                    grid[y][x] = self.chars[char_index]
                    r, g, b = self.palette[val]
                    styles[y][x] = f"rgb({r},{g},{b})"

        return self._grid_to_string(grid, styles)