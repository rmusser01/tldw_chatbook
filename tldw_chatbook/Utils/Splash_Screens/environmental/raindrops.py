"""Raindrops splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple
from dataclasses import dataclass

from ..base_effect import BaseEffect, register_effect, ESCAPED_OPEN_BRACKET


@register_effect("raindrops")
class RaindropsEffect(BaseEffect):
    """Simulates raindrops creating ripples on a pond surface."""

    @dataclass
    class Ripple:
        cx: int # Center x
        cy: int # Center y
        radius: float = 0.0
        max_radius: int = 5
        current_char_index: int = 0
        speed: float = 1.0 # Radius increase per second
        life: float = 2.0 # Lifespan in seconds
        alive_time: float = 0.0
        style: str = "blue"

    def __init__(
        self,
        parent_widget: Any,
        title: str = "Aqua Reflections",
        width: int = 80,
        height: int = 24,
        spawn_rate: float = 0.5, # Average drops per second
        ripple_chars: List[str] = list("·oO()"), # Smallest to largest, then fades maybe
        ripple_styles: List[str] = ["blue", "cyan", "dim blue"],
        max_concurrent_ripples: int = 15,
        base_water_char: str = "~",
        water_style: str = "dim blue",
        title_style: str = "bold white on blue",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.display_width = width
        self.display_height = height
        self.spawn_rate = spawn_rate
        self.ripple_chars = ripple_chars
        self.ripple_styles = ripple_styles
        self.max_concurrent_ripples = max_concurrent_ripples
        self.base_water_char = base_water_char
        self.water_style = water_style
        self.title_style = title_style

        self.ripples: List[RaindropsEffect.Ripple] = []
        self.time_since_last_spawn = 0.0
        self.time_at_last_frame = time.time()

    def _spawn_ripple(self):
        if len(self.ripples) < self.max_concurrent_ripples:
            cx = random.randint(0, self.display_width -1)
            cy = random.randint(0, self.display_height -1)
            max_r = random.randint(3, 8)
            speed = random.uniform(3.0, 6.0) # Faster ripples
            life = random.uniform(0.8, 1.5)   # Shorter lifespan
            style = random.choice(self.ripple_styles)
            self.ripples.append(RaindropsEffect.Ripple(cx=cx, cy=cy, max_radius=max_r, speed=speed, life=life, style=style))

    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.time_at_last_frame
        self.time_at_last_frame = current_time

        # Spawn new ripples
        self.time_since_last_spawn += delta_time
        if self.time_since_last_spawn * self.spawn_rate >= 1.0:
            self._spawn_ripple()
            self.time_since_last_spawn = 0.0
            # Could spawn multiple if spawn_rate is high and delta_time was large
            while random.random() < (self.time_since_last_spawn * self.spawn_rate) -1:
                 self._spawn_ripple()
                 self.time_since_last_spawn -= 1.0/self.spawn_rate


        # Update and filter ripples
        active_ripples = []
        for ripple in self.ripples:
            ripple.alive_time += delta_time
            if ripple.alive_time < ripple.life:
                ripple.radius += ripple.speed * delta_time
                # Determine current ripple character based on radius progression
                # Progress through chars as radius grows, then maybe fade
                char_progress = (ripple.radius / ripple.max_radius) * (len(self.ripple_chars) -1)
                ripple.current_char_index = min(len(self.ripple_chars)-1, int(char_progress))
                active_ripples.append(ripple)
        self.ripples = active_ripples

        # Render to a grid first, handling overlaps (newer/smaller ripples on top conceptually)
        # For simplicity, let's not handle complex overlaps perfectly. Last drawn wins.
        # Initialize with base water pattern
        char_grid = [[(self.base_water_char, self.water_style) for _ in range(self.display_width)] for _ in range(self.display_height)]

        for ripple in sorted(self.ripples, key=lambda r: r.radius, reverse=True): # Draw larger (older) ripples first
            char_to_use = self.ripple_chars[ripple.current_char_index]
            # Could also fade style based on ripple.life vs ripple.alive_time

            # Draw the circle (approximate)
            # This is a simple way to draw a circle on a grid. More advanced algorithms exist.
            for angle_deg in range(0, 360, 10): # Draw points on the circle
                angle_rad = math.radians(angle_deg)
                # For character aspect ratio, y movement might need scaling if cells aren't square
                # Assume x_scale = 1, y_scale = 0.5 (chars are twice as tall as wide)
                # So, for a visually circular ripple, the "y radius" in grid cells is smaller.
                # Let's draw actual grid circles for now.
                x = int(ripple.cx + ripple.radius * math.cos(angle_rad))
                y = int(ripple.cy + ripple.radius * math.sin(angle_rad) * 0.6) # Y correction for char aspect

                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    char_grid[y][x] = (char_to_use, ripple.style)

        # Convert char_grid to styled output lines
        styled_output_lines = []
        for r_idx in range(self.display_height):
            line_segments = []
            for c_idx in range(self.display_width):
                char, style = char_grid[r_idx][c_idx]
                escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                line_segments.append(f"[{style}]{escaped_char}[/{style}]")
            styled_output_lines.append("".join(line_segments))

        # Overlay title (centered)
        if self.title:
            title_y = self.display_height // 2
            title_x_start = (self.display_width - len(self.title)) // 2
            if 0 <= title_y < self.display_height:
                # Reconstruct the title line to overlay on top of ripples
                title_line_segments = []
                current_title_char_idx = 0
                for c_idx in range(self.display_width):
                    is_title_char_pos = title_x_start <= c_idx < title_x_start + len(self.title)
                    if is_title_char_pos:
                        char_to_draw = self.title[current_title_char_idx].replace('[', r'\[')
                        title_line_segments.append(f"[{self.title_style}]{char_to_draw}[/{self.title_style}]")
                        current_title_char_idx +=1
                    else: # Use the already determined char from char_grid (ripple or water)
                        char, style = char_grid[title_y][c_idx]
                        escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                        title_line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                styled_output_lines[title_y] = "".join(title_line_segments)

        return "\n".join(styled_output_lines)