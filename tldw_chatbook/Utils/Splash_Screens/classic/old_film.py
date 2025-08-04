"""OldFilm splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("old_film")
class OldFilmEffect(BaseEffect):
    """Simulates an old film projector effect with shaky frames and film grain."""

    def __init__(
        self,
        parent_widget: Any,
        frames_content: List[str], # List of ASCII art strings, each a frame
        frame_duration: float = 0.5, # How long each frame stays before switching
        shake_intensity: int = 1, # Max character offset for shake (0 for no shake)
        grain_density: float = 0.05, # Chance for a character to be a grain speck
        grain_chars: str = ".:'",
        base_style: str = "sepia", # e.g., "sepia", "grayscale", or just "white on black"
        # Projector beam not implemented in this version for simplicity
        width: int = 80,
        height: int = 24,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.frames = [frame.splitlines() for frame in frames_content]
        if not self.frames: # Ensure there's at least one frame
            self.frames = [["Error: No frames provided".center(width)]]

        # Normalize all frames to consistent dimensions
        self.frame_height = max(len(f) for f in self.frames)
        self.frame_width = max(max(len(line) for line in f) if f else 0 for f in self.frames)

        padded_frames = []
        for frame_idx, frame_data in enumerate(self.frames):
            current_padded_frame = []
            for i in range(self.frame_height):
                line = frame_data[i] if i < len(frame_data) else ""
                current_padded_frame.append(line + ' ' * (self.frame_width - len(line)))
            padded_frames.append(current_padded_frame)
        self.frames = padded_frames

        self.frame_duration = frame_duration
        self.shake_intensity = shake_intensity
        self.grain_density = grain_density
        self.grain_chars = grain_chars
        self.base_style = base_style # This style will be applied to the frame content
        self.display_width = width
        self.display_height = height

        self.current_frame_index = 0
        self.time_on_current_frame = 0.0
        self.time_at_last_frame_render = time.time()

    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.time_at_last_frame_render
        self.time_at_last_frame_render = current_time
        self.time_on_current_frame += delta_time

        if self.time_on_current_frame >= self.frame_duration:
            self.current_frame_index = (self.current_frame_index + 1) % len(self.frames)
            self.time_on_current_frame = 0.0

        current_frame_data = self.frames[self.current_frame_index]

        # Apply shake
        dx, dy = 0, 0
        if self.shake_intensity > 0:
            dx = random.randint(-self.shake_intensity, self.shake_intensity)
            dy = random.randint(-self.shake_intensity, self.shake_intensity)

        # Prepare display grid (chars only first)
        # Output grid matches display_width, display_height
        # Frame content is centered within this.

        output_grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]

        frame_start_row = (self.display_height - self.frame_height) // 2 + dy
        frame_start_col = (self.display_width - self.frame_width) // 2 + dx

        for r_frame in range(self.frame_height):
            for c_frame in range(self.frame_width):
                r_disp, c_disp = frame_start_row + r_frame, frame_start_col + c_frame
                if 0 <= r_disp < self.display_height and 0 <= c_disp < self.display_width:
                    char_to_draw = current_frame_data[r_frame][c_frame]

                    # Apply film grain
                    if random.random() < self.grain_density:
                        char_to_draw = random.choice(self.grain_chars)

                    output_grid[r_disp][c_disp] = char_to_draw

        # Convert to styled lines
        styled_output_lines = []
        for r_idx in range(self.display_height):
            line_str = "".join(output_grid[r_idx]).replace('[',r'\[')
            # Apply base style to the whole line (simpler than per-char if base_style is uniform)
            styled_output_lines.append(f"[{self.base_style}]{line_str}[/{self.base_style}]")

        return "\n".join(styled_output_lines)