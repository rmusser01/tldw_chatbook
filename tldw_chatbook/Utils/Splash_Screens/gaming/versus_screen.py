"""VersusScreen splash screen effect."""

from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("versus_screen")
class VersusScreenEffect(BaseEffect):
    """A 'VS' screen with two characters facing off."""
    def __init__(self, parent_widget: Any, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.p1_x = -10
        self.p2_x = self.width - 1  # Start at the last valid position
        self.vs_scale = 0

    def update(self) -> Optional[str]:
        if self.p1_x < self.width // 4: self.p1_x += 2
        if self.p2_x > self.width * 3 // 4 - 10: self.p2_x -= 2
        if self.p1_x >= self.width // 4 and self.vs_scale < 5: self.vs_scale += 1

        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        p1_art = ['(ง •̀_•́)ง']
        p2_art = ['(ง`_´)ง']

        for i, line in enumerate(p1_art):
            for j, char in enumerate(line):
                y = self.height//2 + i
                x = self.p1_x + j
                if 0 <= y < self.height and 0 <= x < self.width:
                    grid[y][x] = char

        for i, line in enumerate(p2_art):
            for j, char in enumerate(line):
                y = self.height//2 + i
                x = self.p2_x + j
                if 0 <= y < self.height and 0 <= x < self.width:
                    grid[y][x] = char

        if self.vs_scale > 0:
            vs_art = ["V","S"]
            vs_x = self.width//2
            vs_y = self.height//2
            for i, line in enumerate(vs_art):
                for j, char in enumerate(line):
                    for s in range(self.vs_scale):
                        y = vs_y + i*self.vs_scale - s
                        x = vs_x - s + j*self.vs_scale
                        if 0 <= y < self.height and 0 <= x < self.width:
                            grid[y][x] = char
                            styles[y][x] = "bold red"

        return self._grid_to_string(grid, styles)