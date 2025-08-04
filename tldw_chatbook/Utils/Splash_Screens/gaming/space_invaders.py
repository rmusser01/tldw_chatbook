"""SpaceInvaders splash screen effect."""

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("space_invaders")
class SpaceInvadersEffect(BaseEffect):
    """A recreation of the classic Space Invaders game."""
    def __init__(self, parent_widget: Any, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.invaders = []
        self.invader_dir = 1
        self.invader_y = 2
        for y in range(4):
            for x in range(8):
                self.invaders.append({'x': x * 4, 'y': y * 2 + self.invader_y})
        self.player_x = self.width // 2
        self.laser = None

    def update(self) -> Optional[str]:
        if self.frame_count % 10 == 0:
            move_down = False
            for invader in self.invaders:
                invader['x'] += self.invader_dir
                if invader['x'] <= 0 or invader['x'] >= self.width - 2:
                    move_down = True
            if move_down:
                self.invader_dir *= -1
                for invader in self.invaders:
                    invader['y'] += 1

        if self.laser:
            self.laser['y'] -= 1
            if self.laser['y'] < 0:
                self.laser = None
        elif random.random() < 0.1:
            self.laser = {'x': self.player_x, 'y': self.height - 2}

        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        for invader in self.invaders:
            grid[invader['y']][invader['x']] = '▞'
            styles[invader['y']][invader['x']] = 'bold green'

        grid[self.height - 2][self.player_x] = '▲'
        styles[self.height - 2][self.player_x] = 'bold white'

        if self.laser:
            grid[self.laser['y']][self.laser['x']] = '|'
            styles[self.laser['y']][self.laser['x']] = 'bold red'

        return self._grid_to_string(grid, styles)