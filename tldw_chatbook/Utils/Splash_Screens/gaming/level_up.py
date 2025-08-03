"""LevelUp splash screen effect."""

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("level_up")
class LevelUpEffect(BaseEffect):
    """Text that 'levels up' with a flash of light and particle effects."""
    def __init__(self, parent_widget: Any, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.level = 1
        self.progress = 0
        self.particles = []

    def update(self) -> Optional[str]:
        self.progress += 1
        if self.progress >= 100:
            self.progress = 0
            self.level += 1
            for _ in range(20):
                self.particles.append({
                    'x': self.width / 2, 'y': self.height / 2,
                    'vx': random.uniform(-2,2), 'vy': random.uniform(-1,1),
                    'life': 20
                })

        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        text = f"LEVEL {self.level}"
        x_start = (self.width - len(text))//2
        y_start = self.height//2
        for i, char in enumerate(text):
            grid[y_start][x_start+i] = char
            styles[y_start][x_start+i] = 'bold yellow'

        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] > 0:
                px, py = int(p['x']), int(p['y'])
                if 0 <= px < self.width and 0 <= py < self.height:
                    grid[py][px] = '*'
                    styles[py][px] = 'yellow'

        self.particles = [p for p in self.particles if p['life'] > 0]
        return self._grid_to_string(grid, styles)