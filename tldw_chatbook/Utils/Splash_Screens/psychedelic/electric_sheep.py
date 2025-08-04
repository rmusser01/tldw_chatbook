"""ElectricSheep splash screen effect."""

import math
import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("electric_sheep")
class ElectricSheepEffect(BaseEffect):
    """Abstract, evolving patterns reminiscent of the 'Electric Sheep' screensaver."""
    def __init__(self, parent_widget: Any, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.time = 0
        self.particles = []
        for _ in range(30):
            self.particles.append({
                'x': random.uniform(0, self.width),
                'y': random.uniform(0, self.height),
                'vx': random.uniform(-1, 1),
                'vy': random.uniform(-1, 1),
                'life': random.uniform(20, 50),
                'color': f"rgb({random.randint(50,255)},{random.randint(50,255)},{random.randint(50,255)})"
            })

    def update(self) -> Optional[str]:
        self.time += 0.05
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vx'] += math.sin(p['y'] * 0.1 + self.time) * 0.1
            p['vy'] += math.cos(p['x'] * 0.1 + self.time) * 0.1
            p['life'] -= 1

            if p['life'] <= 0 or not (0 <= p['x'] < self.width and 0 <= p['y'] < self.height):
                p['x'] = random.uniform(0, self.width)
                p['y'] = random.uniform(0, self.height)
                p['vx'] = random.uniform(-1, 1)
                p['vy'] = random.uniform(-1, 1)
                p['life'] = random.uniform(20, 50)

            x, y = int(p['x']), int(p['y'])
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = '•'
                styles[y][x] = p['color']

        return self._grid_to_string(grid, styles)