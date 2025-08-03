"""DigitalRain splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("digital_rain")
class DigitalRainEffect(BaseEffect):
    """Digital rain effect with varied characters and color options."""

    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Interface Loading...",
        width: int = 80,
        height: int = 24,
        speed: float = 0.05, # Interval for updates
        base_chars: str = "abcdefghijklmnopqrstuvwxyz0123456789",
        highlight_chars: str = "!@#$%^&*()_+=-{}[]|:;<>,.?/~",
        base_color: str = "dim green", # Rich style for base rain
        highlight_color: str = "bold green", # Rich style for highlighted chars
        title_style: str = "bold white",
        subtitle_style: str = "white",
        highlight_chance: float = 0.1, # Chance for a character to be a highlight_char
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.height = height
        self.speed = speed # Not directly used in update logic timing, but for animation timer

        self.all_chars = base_chars + highlight_chars
        self.base_chars = base_chars
        self.highlight_chars = highlight_chars
        self.base_color = base_color
        self.highlight_color = highlight_color
        self.title_style = title_style
        self.subtitle_style = subtitle_style
        self.highlight_chance = highlight_chance

        self.columns: List[List[Tuple[str, str]]] = [] # char, style
        self.column_speeds: List[float] = [] # How many frames until this column updates
        self.column_next_update: List[int] = [] # Frame counter for next update

        self._init_columns()

        self.title_reveal_progress = 0.0
        self.subtitle_reveal_progress = 0.0

    def _init_columns(self) -> None:
        self.columns = []
        self.column_speeds = []
        self.column_next_update = []

        for _ in range(self.width):
            column = []
            # Initial column population (sparse)
            for _ in range(random.randint(self.height // 4, self.height // 2)):
                char = random.choice(self.all_chars)
                style = self.highlight_color if char in self.highlight_chars or random.random() < self.highlight_chance else self.base_color
                column.append((char, style))
            # Pad with spaces to height
            column.extend([(' ', self.base_color)] * (self.height - len(column)))
            random.shuffle(column) # Mix them up initially

            self.columns.append(column)
            self.column_speeds.append(random.randint(1, 5)) # Update every 1-5 frames
            self.column_next_update.append(0)


    def update(self) -> Optional[str]:
        self.frame_count +=1 # Manually increment as BaseEffect's update is overridden

        # Update columns that are due
        for col_idx in range(self.width):
            if self.frame_count >= self.column_next_update[col_idx]:
                # Shift column down
                last_char_tuple = self.columns[col_idx].pop()

                # New char at top
                new_char = random.choice(self.all_chars)
                new_style = self.highlight_color if new_char in self.highlight_chars or random.random() < self.highlight_chance else self.base_color
                self.columns[col_idx].insert(0, (new_char, new_style))

                self.column_next_update[col_idx] = self.frame_count + self.column_speeds[col_idx]

        # Prepare render grid (list of lists of styled characters)
        render_grid: List[List[str]] = [] # Each string is already Rich-escaped and styled

        for r_idx in range(self.height):
            line_segments = []
            for c_idx in range(self.width):
                char, style = self.columns[c_idx][r_idx]
                escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                line_segments.append(f"[{style}]{escaped_char}[/{style}]")
            render_grid.append(line_segments)

        # Title and Subtitle Reveal (similar to MatrixRainEffect)
        elapsed = time.time() - self.start_time
        if elapsed > 0.5: # Start revealing title
            self.title_reveal_progress = min(1.0, (elapsed - 0.5) / 1.5) # Slower reveal
            title_len_to_show = int(len(self.title) * self.title_reveal_progress)

            title_row = self.height // 2 - 2
            title_start_col = (self.width - len(self.title)) // 2

            if 0 <= title_row < self.height:
                for i in range(len(self.title)):
                    if title_start_col + i < self.width:
                        if i < title_len_to_show:
                            char_to_draw = self.title[i].replace('[', r'\[')
                            render_grid[title_row][title_start_col + i] = f"[{self.title_style}]{char_to_draw}[/{self.title_style}]"
                        else: # Keep rain char but make it almost invisible or a background color
                             render_grid[title_row][title_start_col + i] = "[on black] [/on black]"


        if elapsed > 1.0: # Start revealing subtitle
            self.subtitle_reveal_progress = min(1.0, (elapsed - 1.0) / 1.5)
            subtitle_len_to_show = int(len(self.subtitle) * self.subtitle_reveal_progress)

            subtitle_row = self.height // 2
            subtitle_start_col = (self.width - len(self.subtitle)) // 2

            if 0 <= subtitle_row < self.height:
                for i in range(len(self.subtitle)):
                     if subtitle_start_col + i < self.width:
                        if i < subtitle_len_to_show:
                            char_to_draw = self.subtitle[i].replace('[', r'\[')
                            render_grid[subtitle_row][subtitle_start_col + i] = f"[{self.subtitle_style}]{char_to_draw}[/{self.subtitle_style}]"
                        else:
                            render_grid[subtitle_row][subtitle_start_col + i] = "[on black] [/on black]"

        final_lines = ["".join(line_segments) for line_segments in render_grid]
        return "\n".join(final_lines)