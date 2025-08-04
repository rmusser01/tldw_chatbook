"""Starfield splash screen effect."""

import math
import random
from typing import Optional, Any, List, Tuple
from dataclasses import dataclass

from ..base_effect import BaseEffect, register_effect, ESCAPED_OPEN_BRACKET


@register_effect("starfield")
class StarfieldEffect(BaseEffect):
    """Simulates a starfield warp effect."""

    @dataclass
    class Star:
        x: float # Current screen x
        y: float # Current screen y
        z: float # Depth (distance from viewer, max_depth is furthest)
        # For warp effect, stars also need a fixed trajectory from center
        angle: float # Angle of trajectory from center
        initial_speed_factor: float # Base speed factor for this star

    def __init__(
        self,
        parent_widget: Any,
        title: str = "WARP SPEED ENGAGED",
        num_stars: int = 150,
        warp_factor: float = 0.2, # Controls how fast z decreases and thus apparent speed
        max_depth: float = 50.0, # Furthest z value
        star_chars: List[str] = list("·.*+"), # Smallest to largest/brightest
        star_styles: List[str] = ["dim white", "white", "bold white", "bold yellow"],
        width: int = 80,
        height: int = 24,
        title_style: str = "bold cyan on black",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.num_stars = num_stars
        self.warp_factor = warp_factor
        self.max_depth = max_depth
        self.star_chars = star_chars
        self.star_styles = star_styles
        self.width = width
        self.height = height
        self.center_x = width / 2.0
        self.center_y = height / 2.0
        self.title_style = title_style
        self.stars: List[StarfieldEffect.Star] = []
        for _ in range(self.num_stars):
            self.stars.append(self._spawn_star(is_initial_spawn=True))

    def _spawn_star(self, is_initial_spawn: bool = False) -> Star:
        angle = random.uniform(0, 2 * math.pi)
        initial_speed_factor = random.uniform(0.2, 1.0) # How fast it moves from center

        z = self.max_depth # Always spawn at max depth for this warp effect

        return StarfieldEffect.Star(
            x=self.center_x,
            y=self.center_y,
            z=z,
            angle=angle,
            initial_speed_factor=initial_speed_factor
        )

    def update(self) -> Optional[str]:
        styled_chars_on_grid: Dict[Tuple[int, int], Tuple[str, str]] = {}

        for i in range(len(self.stars)):
            star = self.stars[i]
            star.z -= self.warp_factor

            if star.z <= 0:
                self.stars[i] = self._spawn_star()
                continue

            radius_on_screen = star.initial_speed_factor * (self.max_depth - star.z) * (self.width / (self.max_depth * 10.0))


            star.x = self.center_x + math.cos(star.angle) * radius_on_screen
            # Adjust y movement based on aspect ratio if terminal cells aren't square
            # Assuming roughly 2:1 height:width for characters, so y movement appears slower
            star.y = self.center_y + math.sin(star.angle) * radius_on_screen * 0.5

            z_ratio = star.z / self.max_depth

            char_idx = 0
            if z_ratio < 0.25: char_idx = 3
            elif z_ratio < 0.50: char_idx = 2
            elif z_ratio < 0.75: char_idx = 1
            else: char_idx = 0

            char_idx = min(char_idx, len(self.star_chars) - 1)
            style_idx = min(char_idx, len(self.star_styles) -1)

            star_char = self.star_chars[char_idx]
            star_style = self.star_styles[style_idx]

            ix, iy = int(star.x), int(star.y)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                styled_chars_on_grid[(ix, iy)] = (star_char, star_style)

        output_lines = []
        for r_idx in range(self.height):
            line_segments = []
            for c_idx in range(self.width):
                if (c_idx, r_idx) in styled_chars_on_grid:
                    char, style = styled_chars_on_grid[(c_idx, r_idx)]
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(' ')
            output_lines.append("".join(line_segments))

        if self.title:
            title_y = self.height // 2
            title_x_start = (self.width - len(self.title)) // 2

            if 0 <= title_y < self.height:
                title_segments = []
                current_title_char_idx = 0
                for c_idx in range(self.width):
                    is_title_char = title_x_start <= c_idx < title_x_start + len(self.title)
                    if is_title_char:
                        char_to_draw = self.title[current_title_char_idx].replace('[', r'\[')
                        title_segments.append(f"[{self.title_style}]{char_to_draw}[/{self.title_style}]")
                        current_title_char_idx +=1
                    else:
                        if (c_idx, title_y) in styled_chars_on_grid: # Star is behind title char
                            char, style = styled_chars_on_grid[(c_idx, title_y)]
                            escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                            title_segments.append(f"[{style}]{escaped_char}[/{style}]")
                        else: # Empty space behind title char
                            title_segments.append(' ')
                output_lines[title_y] = "".join(title_segments)

        return "\n".join(output_lines)