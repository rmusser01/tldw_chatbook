"""Spotlight splash screen effect."""

from rich.style import Style
import math
import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect, ESCAPED_OPEN_BRACKET


@register_effect("spotlight")
class SpotlightEffect(BaseEffect):
    """Moves a 'spotlight' over content, revealing parts of it."""

    def __init__(
        self,
        parent_widget: Any,
        background_content: str,
        spotlight_radius: int = 5,
        movement_speed: float = 10.0, # Pixels (grid cells) per second
        path_type: str = "lissajous", # "lissajous", "random_walk", "circle"
        visible_style: str = "bold white", # Style of text inside spotlight
        hidden_style: str = "dim black on black", # Style of text outside spotlight (very dim)
        width: int = 80, # display width
        height: int = 24, # display height
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.lines = background_content.splitlines()
        # Normalize background content dimensions
        self.content_height = len(self.lines)
        self.content_width = max(len(line) for line in self.lines) if self.lines else 0

        self.padded_lines = []
        for i in range(self.content_height):
            line = self.lines[i]
            self.padded_lines.append(line + ' ' * (self.content_width - len(line)))

        self.spotlight_radius = spotlight_radius
        self.spotlight_radius_sq = spotlight_radius ** 2
        self.movement_speed = movement_speed # Cells per second
        self.path_type = path_type
        self.visible_style = visible_style
        self.hidden_style = hidden_style
        self.display_width = width # Max width of the rendered output
        self.display_height = height # Max height

        self.spotlight_x = float(self.content_width // 2)
        self.spotlight_y = float(self.content_height // 2)

        # Path specific parameters
        self.time_elapsed_for_path = 0
        if self.path_type == "random_walk":
            self.vx = random.uniform(-self.movement_speed, self.movement_speed)
            self.vy = random.uniform(-self.movement_speed, self.movement_speed)

        self.time_at_last_frame = time.time()


    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.time_at_last_frame
        self.time_at_last_frame = current_time
        self.time_elapsed_for_path += delta_time

        # Update spotlight position
        if self.path_type == "lissajous":
            # Simple Lissajous curve for movement
            # Adjust frequencies (e.g., 0.1, 0.07) and phase for different paths
            self.spotlight_x = (self.content_width / 2) + (self.content_width / 2 - self.spotlight_radius) * math.sin(self.time_elapsed_for_path * 0.15)
            self.spotlight_y = (self.content_height / 2) + (self.content_height / 2 - self.spotlight_radius) * math.cos(self.time_elapsed_for_path * 0.1)
        elif self.path_type == "circle":
            radius = min(self.content_width, self.content_height) / 2 - self.spotlight_radius
            self.spotlight_x = (self.content_width / 2) + radius * math.cos(self.time_elapsed_for_path * self.movement_speed * 0.02) # speed factor
            self.spotlight_y = (self.content_height / 2) + radius * math.sin(self.time_elapsed_for_path * self.movement_speed * 0.02)
        elif self.path_type == "random_walk":
            self.spotlight_x += self.vx * delta_time
            self.spotlight_y += self.vy * delta_time
            # Boundary checks and bounce / change direction
            if not (0 <= self.spotlight_x < self.content_width):
                self.vx *= -1
                self.spotlight_x = max(0, min(self.spotlight_x, self.content_width -1))
            if not (0 <= self.spotlight_y < self.content_height):
                self.vy *= -1
                self.spotlight_y = max(0, min(self.spotlight_y, self.content_height -1))
            # Occasionally change direction randomly
            if random.random() < 0.01: # 1% chance per frame
                 self.vx = random.uniform(-self.movement_speed, self.movement_speed) * 0.1 # Slower random walk
                 self.vy = random.uniform(-self.movement_speed, self.movement_speed) * 0.1


        # Render the content with spotlight effect
        output_lines = []
        # Determine rendering bounds based on display_height/width and content_height/width
        render_height = min(self.display_height, self.content_height)

        # Center the content if smaller than display area
        content_start_row = (self.display_height - render_height) // 2

        for r_disp in range(self.display_height):
            if content_start_row <= r_disp < content_start_row + render_height:
                r_content = r_disp - content_start_row # Index in self.padded_lines
                line = self.padded_lines[r_content]
                styled_line_segments = []

                content_start_col = (self.display_width - self.content_width) // 2

                for c_disp in range(self.display_width):
                    if content_start_col <= c_disp < content_start_col + self.content_width:
                        c_content = c_disp - content_start_col # Index in line
                        char = line[c_content]
                        escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)

                        # Check distance from spotlight center
                        # Adjust for character aspect ratio if desired (y distances count more)
                        dist_sq = (c_content - self.spotlight_x)**2 + ((r_content - self.spotlight_y)*2)**2 # Y weighted

                        if dist_sq <= self.spotlight_radius_sq:
                            styled_line_segments.append(f"[{self.visible_style}]{escaped_char}[/{self.visible_style}]")
                        else:
                            # Optional: fade effect at edges of spotlight
                            # For now, simple binary visible/hidden
                            styled_line_segments.append(f"[{self.hidden_style}]{escaped_char}[/{self.hidden_style}]")
                    else: # Outside content width, but within display width (padding)
                        styled_line_segments.append(f"[{self.hidden_style}] [/{self.hidden_style}]")
                output_lines.append("".join(styled_line_segments))
            else: # Outside content height (padding)
                output_lines.append(f"[{self.hidden_style}]{' ' * self.display_width}[/{self.hidden_style}]")

        return "\n".join(output_lines)