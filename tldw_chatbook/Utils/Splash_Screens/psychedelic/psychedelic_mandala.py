"""PsychedelicMandala splash screen effect."""

import math
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("psychedelic_mandala")
class PsychedelicMandalaEffect(BaseEffect):
    """A rotating, colorful mandala that expands from the center."""

    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Expanding consciousness...",
        width: int = 80,
        height: int = 24,
        rotation_speed: float = 0.4,
        num_segments: int = 8,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.rotation_speed = rotation_speed
        self.num_segments = num_segments
        self.time = 0
        self.center_x = self.display_width / 2
        self.center_y = self.display_height / 2
        self.max_radius = min(self.center_x, self.center_y)
        self.pattern_chars = "◆◇◈○●◐◑◒◓"

    def update(self) -> Optional[str]:
        self.time += 0.05
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]

        for y in range(self.display_height):
            for x in range(self.display_width):
                dx = x - self.center_x
                dy = (y - self.center_y) * 2  # Correct for aspect ratio
                radius = math.sqrt(dx**2 + dy**2)
                angle = math.atan2(dy, dx)

                if radius < self.max_radius:
                    # Psychedelic pattern generation
                    v1 = math.sin(radius * 0.5 - self.time * 2)
                    v2 = math.sin(angle * self.num_segments + self.time * self.rotation_speed)
                    v3 = math.sin((angle + radius * 0.1) * 4 + self.time)
                    combined_value = v1 + v2 + v3

                    if combined_value > 1.5:
                        char_index = int(abs(combined_value * 5)) % len(self.pattern_chars)
                        grid[y][x] = self.pattern_chars[char_index]

                        # Psychedelic color calculation
                        hue = int((angle * 180 / math.pi + self.time * 50)) % 360
                        r = int(128 + 127 * math.sin(math.radians(hue)))
                        g = int(128 + 127 * math.sin(math.radians(hue + 120)))
                        b = int(128 + 127 * math.sin(math.radians(hue + 240)))
                        styles[y][x] = f"rgb({r},{g},{b})"

        self._add_centered_text(grid, styles, self.title, self.display_height // 2 - 1, 'bold white on black')
        self._add_centered_text(grid, styles, self.subtitle, self.display_height // 2 + 1, 'bold white on black')

        return self._grid_to_string(grid, styles)