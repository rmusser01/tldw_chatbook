"""AsciiMorph splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from rich.markup import escape

from ..base_effect import BaseEffect, register_effect


@register_effect("ascii_morph")
class AsciiMorphEffect(BaseEffect):
    """Smoothly morphs one ASCII art into another."""

    def __init__(
        self,
        parent_widget: Any,
        start_content: str,
        end_content: str,
        duration: float = 2.0, # Total duration of the morph
        morph_style: str = "dissolve", # "dissolve", "random_pixel", "wipe_left_to_right"
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.start_lines = start_content.splitlines()
        self.end_lines = end_content.splitlines()
        self.duration = duration
        self.morph_style = morph_style

        # Normalize line lengths and line counts for consistent morphing
        self.height = max(len(self.start_lines), len(self.end_lines))
        self.width = 0
        for line in self.start_lines + self.end_lines:
            if len(line) > self.width:
                self.width = len(line)

        self.start_lines = self._pad_art(self.start_lines)
        self.end_lines = self._pad_art(self.end_lines)

        # For 'dissolve' or 'random_pixel', precompute all character positions
        self.all_positions = []
        if self.morph_style in ["dissolve", "random_pixel"]:
            for r in range(self.height):
                for c in range(self.width):
                    if self.start_lines[r][c] != self.end_lines[r][c]:
                        self.all_positions.append((r, c))
            if self.morph_style == "dissolve": # Shuffle for dissolve
                random.shuffle(self.all_positions)

        self.current_art_chars = [list(line) for line in self.start_lines]

    def _pad_art(self, art_lines: List[str]) -> List[str]:
        """Pads ASCII art to consistent width and height."""
        padded_art = []
        for i in range(self.height):
            if i < len(art_lines):
                line = art_lines[i]
                padded_art.append(line + ' ' * (self.width - len(line)))
            else:
                padded_art.append(' ' * self.width)
        return padded_art

    def update(self) -> Optional[str]:
        elapsed_time = time.time() - self.start_time
        progress = min(1.0, elapsed_time / self.duration)

        if progress >= 1.0:
            return escape("\n".join(self.end_lines))

        if self.morph_style == "dissolve" or self.morph_style == "random_pixel":
            num_chars_to_change = int(progress * len(self.all_positions))
            for i in range(num_chars_to_change):
                if i < len(self.all_positions):
                    r, c = self.all_positions[i]
                    if self.morph_style == "dissolve":
                         # For dissolve, directly set to end char
                        self.current_art_chars[r][c] = self.end_lines[r][c]
                    elif self.morph_style == "random_pixel":
                        # For random_pixel, set to a random char during transition, then final
                        # This needs another state or to be driven by progress.
                        # Simpler: if not fully progressed, pick start or end based on sub-progress for that pixel
                        if random.random() < progress: # As progress increases, more chance to be end_char
                           self.current_art_chars[r][c] = self.end_lines[r][c]
                        else:
                           self.current_art_chars[r][c] = self.start_lines[r][c] # Or a random char

            # For random_pixel, we should re-evaluate all pixels each frame based on progress
            if self.morph_style == "random_pixel":
                 for r_idx in range(self.height):
                    for c_idx in range(self.width):
                        if self.start_lines[r_idx][c_idx] != self.end_lines[r_idx][c_idx]:
                            if random.random() < progress:
                                self.current_art_chars[r_idx][c_idx] = self.end_lines[r_idx][c_idx]
                            else:
                                # Optionally, insert a random "transition" character
                                # self.current_art_chars[r_idx][c_idx] = random.choice(".:-=+*#%@")
                                self.current_art_chars[r_idx][c_idx] = self.start_lines[r_idx][c_idx]
                        else:
                            self.current_art_chars[r_idx][c_idx] = self.start_lines[r_idx][c_idx]


        elif self.morph_style == "wipe_left_to_right":
            wipe_column = int(progress * self.width)
            for r in range(self.height):
                for c in range(self.width):
                    if c < wipe_column:
                        self.current_art_chars[r][c] = self.end_lines[r][c]
                    else:
                        self.current_art_chars[r][c] = self.start_lines[r][c]

        # Default or fallback: simple crossfade (alpha blending not possible with chars)
        # So, stick to one of the above, or make dissolve the default.
        # If morph_style is not recognized, it will effectively be stuck on start_art or do random_pixel if all_positions was populated.
        # Let's ensure 'dissolve' is the default if style is unknown.
        else: # Fallback or if morph_style == "dissolve" initially
            num_chars_to_change = int(progress * len(self.all_positions))
            for i in range(num_chars_to_change):
                if i < len(self.all_positions):
                    r, c = self.all_positions[i]
                    self.current_art_chars[r][c] = self.end_lines[r][c]


        return "\n".join("".join(row) for row in self.current_art_chars).replace('[',r'\[')