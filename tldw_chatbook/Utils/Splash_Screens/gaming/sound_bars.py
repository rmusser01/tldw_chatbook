"""SoundBars splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("sound_bars")
class SoundBarsEffect(BaseEffect):
    """Simulates abstract sound visualizer bars."""

    def __init__(
        self,
        parent_widget: Any,
        title: str = "Audio Core Calibrating...",
        num_bars: int = 15,
        max_bar_height: Optional[int] = None, # If None, calculated from display_height
        bar_char_filled: str = "█",
        bar_char_empty: str = " ", # Usually not visible if styled with background
        bar_styles: List[str] = ["bold blue", "bold magenta", "bold cyan", "bold green", "bold yellow", "bold red"],
        width: int = 80, # display width
        height: int = 24, # display height
        title_style: str = "bold white",
        update_speed: float = 0.05, # How fast bars change height
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.num_bars = num_bars
        self.display_width = width
        self.display_height = height

        # Title takes ~1 line + 1 for spacing, rest for bars
        self.title_area_height = 2 if self.title else 0
        self.max_bar_height = max_bar_height if max_bar_height is not None else self.display_height - self.title_area_height -1 # -1 for base line
        if self.max_bar_height <=0: self.max_bar_height = 1

        self.bar_char_filled = bar_char_filled[0]
        self.bar_char_empty = bar_char_empty[0]
        self.bar_styles = bar_styles
        self.title_style = title_style
        self.update_speed = update_speed # Interval for changing bar heights

        self.bar_heights = [random.randint(1, self.max_bar_height) for _ in range(self.num_bars)]
        self.bar_targets = list(self.bar_heights) # Target heights for smooth transition
        self.bar_colors = [random.choice(self.bar_styles) for _ in range(self.num_bars)]

        self._last_bar_update_time = time.time()

    def _update_bar_heights(self):
        """Update target heights and smoothly move current heights."""
        for i in range(self.num_bars):
            # Chance to pick a new target height
            if random.random() < 0.2 or self.bar_heights[i] == self.bar_targets[i]:
                self.bar_targets[i] = random.randint(1, self.max_bar_height)
                self.bar_colors[i] = random.choice(self.bar_styles) # Change color too
            # Move towards target
            if self.bar_heights[i] < self.bar_targets[i]:
                self.bar_heights[i] = min(self.bar_targets[i], self.bar_heights[i] + 1) # Step of 1 for simplicity
            elif self.bar_heights[i] > self.bar_targets[i]:
                 self.bar_heights[i] = max(self.bar_targets[i], self.bar_heights[i] -1)


    def update(self) -> Optional[str]:
        current_time = time.time()
        if current_time - self._last_bar_update_time >= self.update_speed:
            self._update_bar_heights()
            self._last_bar_update_time = current_time

        output_lines = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styled_output_lines = [""] * self.display_height

        # Render title if present
        title_start_row = 0
        if self.title:
            title_x_start = (self.display_width - len(self.title)) // 2
            for c, char_val in enumerate(self.title):
                if title_x_start + c < self.display_width:
                    output_lines[title_start_row][title_x_start + c] = (char_val.replace('[',r'\['), self.title_style)
            title_start_row += 1 # Move down for next line (e.g. spacing or bars)
            if self.title_area_height > 1: # if spacing was reserved
                 title_start_row += (self.title_area_height -1)


        # Render bars
        # Calculate bar width and spacing (simple equal spacing)
        bar_display_area_width = self.display_width
        total_bar_chars_width = self.num_bars # Assuming each bar is 1 char wide

        # If we want wider bars, e.g. 2 chars wide:
        # bar_char_width = 2
        # total_bar_chars_width = self.num_bars * bar_char_width
        # For simplicity, 1 char wide bars.

        spacing = (bar_display_area_width - total_bar_chars_width) // (self.num_bars + 1)
        if spacing < 0: spacing = 0 # Bars might overlap if too many

        current_c = spacing # Start position for first bar

        for i in range(self.num_bars):
            if current_c >= self.display_width: break # No more space for bars

            bar_h = self.bar_heights[i]
            bar_style_to_use = self.bar_colors[i]

            for r in range(self.max_bar_height):
                # Bars are drawn from bottom up
                row_idx_on_display = (title_start_row + self.max_bar_height -1) - r
                if row_idx_on_display < title_start_row : continue # Don't draw into title area from bottom
                if row_idx_on_display >= self.display_height : continue


                if r < bar_h : # This part of bar is filled
                    output_lines[row_idx_on_display][current_c] = (self.bar_char_filled, bar_style_to_use)
                else: # This part is empty (above the current bar height)
                    output_lines[row_idx_on_display][current_c] = (self.bar_char_empty, "default") # or specific empty style

            current_c += 1 # Next char column for this bar (if multi-char wide)
            current_c += spacing # Move to start of next bar

        # Convert the character grid to styled lines
        for r_idx in range(self.display_height):
            line_segments = []
            for c_idx in range(self.display_width):
                cell = output_lines[r_idx][c_idx]
                if isinstance(cell, tuple):
                    char, style = cell
                    line_segments.append(f"[{style}]{char}[/{style}]")
                else: # Space
                    line_segments.append(' ') # Default background
            styled_output_lines[r_idx] = "".join(line_segments)

        return "\n".join(styled_output_lines)