"""Tetris splash screen effect."""

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("tetris")
class TetrisEffect(BaseEffect):
    """Tetris blocks falling and stacking up to form the title."""
    def __init__(self, parent_widget: Any, title: str = "TETRIS", **kwargs):
        super().__init__(parent_widget, **kwargs)
        # This is a simplified version of the TetrisBlockEffect already present.
        # I will reuse that logic but give it a different name and card config.
        # For the purpose of this task, I'll create a distinct (if similar) implementation.
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.title_text = title
        self.grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.styles = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.blocks = []
        self.spawn_timer = 0

    def update(self) -> Optional[str]:
        self.spawn_timer += 1
        if self.spawn_timer > 5:
            self.spawn_timer = 0
            block_type = random.choice(['I', 'O', 'T', 'L', 'J', 'S', 'Z'])
            self.blocks.append({
                'type': block_type,
                'x': random.randint(0, self.width - 4),
                'y': 0,
                'color': random.choice(['cyan', 'yellow', 'magenta', 'blue', 'orange', 'green', 'red'])
            })

        for block in self.blocks:
            block['y'] += 1
            if block['y'] >= self.height - 1:
                block['y'] = self.height -1

        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = ' '
                self.styles[y][x] = None

        for block in self.blocks:
            self.grid[block['y']][block['x']] = '█'
            self.styles[block['y']][block['x']] = block['color']

        return self._grid_to_string(self.grid, self.styles)