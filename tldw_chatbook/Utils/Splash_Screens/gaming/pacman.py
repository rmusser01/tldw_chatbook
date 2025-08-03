"""Pacman splash screen effect."""

from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("pacman")
class PacmanEffect(BaseEffect):
    """An animation of Pac-Man moving across the screen."""
    def __init__(self, parent_widget: Any, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.pacman_x = 1
        self.pacman_y = self.height // 2
        self.dots = set((x, self.pacman_y) for x in range(1, self.width - 1))
        self.ghosts = [
            {'x': self.width // 2, 'y': self.height // 2, 'char': 'ᗝ', 'color': 'red'},
            {'x': self.width // 2 + 5, 'y': self.height // 2, 'char': 'ᗝ', 'color': 'cyan'},
        ]

    def update(self) -> Optional[str]:
        if (self.pacman_x, self.pacman_y) in self.dots:
            self.dots.remove((self.pacman_x, self.pacman_y))

        self.pacman_x += 1
        if self.pacman_x >= self.width -1:
            self.pacman_x = 1

        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        for x, y in self.dots:
            grid[y][x] = '.'
            styles[y][x] = 'yellow'

        pac_char = 'ᗧ' if self.frame_count % 2 == 0 else 'ᗣ'
        grid[self.pacman_y][self.pacman_x] = pac_char
        styles[self.pacman_y][self.pacman_x] = 'bold yellow'

        for ghost in self.ghosts:
            grid[ghost['y']][ghost['x']] = ghost['char']
            styles[ghost['y']][ghost['x']] = f"bold {ghost['color']}"

        return self._grid_to_string(grid, styles)