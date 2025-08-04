"""LavaLamp splash screen effect."""

from rich.color import Color
import random
from typing import Optional, Any, List, Tuple
from dataclasses import dataclass

from ..base_effect import BaseEffect, register_effect


@register_effect("lava_lamp")
class LavaLampEffect(BaseEffect):
    """Morphing, colored blobs that rise and fall, mimicking a lava lamp."""

    @dataclass
    class Blob:
        x: float
        y: float
        vx: float
        vy: float
        radius: float
        color: str

    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Go with the flow...",
        width: int = 80,
        height: int = 24,
        num_blobs: int = 5,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.num_blobs = num_blobs
        self.blobs: List[LavaLampEffect.Blob] = []
        self.chars = ".:-=+*#%@"
        self.colors = ["magenta", "cyan", "yellow", "green", "red"]

        for i in range(self.num_blobs):
            self.blobs.append(self.Blob(
                x=random.uniform(self.display_width * 0.2, self.display_width * 0.8),
                y=random.uniform(0, self.display_height),
                vx=random.uniform(-0.5, 0.5),
                vy=random.uniform(-0.3, 0.3),
                radius=random.uniform(4, 8),
                color=self.colors[i % len(self.colors)]
            ))

    def update(self) -> Optional[str]:
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]

        for blob in self.blobs:
            blob.x += blob.vx
            blob.y += blob.vy
            blob.vx += random.uniform(-0.1, 0.1)
            blob.vy += random.uniform(-0.05, 0.05) - (blob.y - self.display_height/2) * 0.001

            if blob.x < blob.radius or blob.x > self.display_width - blob.radius: blob.vx *= -1
            if blob.y < blob.radius or blob.y > self.display_height - blob.radius: blob.vy *= -1

            blob.vx = max(-1, min(1, blob.vx))
            blob.vy = max(-0.5, min(0.5, blob.vy))

        for y in range(self.display_height):
            for x in range(self.display_width):
                energy = 0.0
                for blob in self.blobs:
                    dist_sq = (x - blob.x)**2 + ((y - blob.y)*2)**2
                    energy += blob.radius**2 / (dist_sq + 1e-6)

                if energy > 0.5:
                    char_index = min(len(self.chars) - 1, int(energy * 2))
                    grid[y][x] = self.chars[char_index]

                    # Mix colors of nearby blobs
                    r, g, b = 0, 0, 0
                    total_influence = 0
                    for blob in self.blobs:
                        dist_sq = (x - blob.x)**2 + ((y - blob.y)*2)**2
                        influence = 1 / (dist_sq + 1)
                        # This is a simplification; rich colors are named, not RGB.
                        # I'll just pick the color of the most influential blob.
                        if influence > total_influence:
                            total_influence = influence
                            c = Color.parse(blob.color).get_truecolor()
                            r,g,b = c[0], c[1], c[2]

                    styles[y][x] = f"rgb({r},{g},{b})"

        self._add_centered_text(grid, styles, self.title, 4, 'bold white on black')
        self._add_centered_text(grid, styles, self.subtitle, self.display_height - 4, 'bold white on black')
        return self._grid_to_string(grid, styles)