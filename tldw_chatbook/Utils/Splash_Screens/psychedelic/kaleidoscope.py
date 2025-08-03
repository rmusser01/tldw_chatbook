"""Kaleidoscope splash screen effect."""

import math
import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("kaleidoscope")
class KaleidoscopeEffect(BaseEffect):
    """Symmetrical, mirrored patterns that shift and rotate."""

    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Reflecting reality...",
        width: int = 80,
        height: int = 24,
        num_segments: int = 6,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.num_segments = num_segments
        self.time = 0
        self.center_x = self.display_width / 2
        self.center_y = self.display_height / 2
        self.pattern_chars = "▚▞▙▟▛▜▝▘"
        self.particles = []

        for _ in range(20):
            self.particles.append({
                'x': random.uniform(0, self.display_width / self.num_segments),
                'y': random.uniform(0, self.display_height / 2),
                'vx': random.uniform(-1, 1),
                'vy': random.uniform(-1, 1),
                'char': random.choice(self.pattern_chars),
                'color': f"rgb({random.randint(50,255)},{random.randint(50,255)},{random.randint(50,255)})"
            })

    def update(self) -> Optional[str]:
        self.time += 0.05
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]

        for p in self.particles:
            p['x'] += p['vx'] * 0.1
            p['y'] += p['vy'] * 0.1
            if p['x'] < 0 or p['x'] > self.display_width: p['vx'] *= -1
            if p['y'] < 0 or p['y'] > self.display_height: p['vy'] *= -1

            for i in range(self.num_segments):
                angle = (2 * math.pi / self.num_segments) * i + self.time * 0.2

                # Original point
                rotated_x = p['x'] * math.cos(angle) - p['y'] * math.sin(angle)
                rotated_y = p['x'] * math.sin(angle) + p['y'] * math.cos(angle)

                draw_x, draw_y = int(self.center_x + rotated_x), int(self.center_y + rotated_y * 0.5)
                if 0 <= draw_x < self.display_width and 0 <= draw_y < self.display_height:
                    grid[draw_y][draw_x] = p['char']
                    styles[draw_y][draw_x] = p['color']

                # Mirrored point
                rotated_x_m = p['x'] * math.cos(angle) + p['y'] * math.sin(angle)
                rotated_y_m = p['x'] * math.sin(angle) - p['y'] * math.cos(angle)

                draw_x_m, draw_y_m = int(self.center_x + rotated_x_m), int(self.center_y + rotated_y_m * 0.5)
                if 0 <= draw_x_m < self.display_width and 0 <= draw_y_m < self.display_height:
                    grid[draw_y_m][draw_x_m] = p['char']
                    styles[draw_y_m][draw_x_m] = p['color']

        self._add_centered_text(grid, styles, self.title, self.display_height // 2, 'bold white on black')
        return self._grid_to_string(grid, styles)