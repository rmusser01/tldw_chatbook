"""PixelZoom splash screen effect."""

import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("pixel_zoom")
class PixelZoomEffect(BaseEffect):
    """Starts with a pixelated (blocky) version of an ASCII art and resolves to clear."""

    def __init__(
        self,
        parent_widget: Any,
        target_content: str, # The clear, final ASCII art
        duration: float = 2.5, # Total duration of the effect
        max_pixel_size: int = 8, # Max block size for pixelation (e.g., 8x8 chars become one block)
        effect_type: str = "resolve", # "resolve" (pixelated to clear) or "pixelate" (clear to pixelated)
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.target_lines = target_content.splitlines()
        self.duration = duration
        self.max_pixel_size = max(1, max_pixel_size) # Must be at least 1
        self.effect_type = effect_type

        # Normalize target content dimensions
        self.content_height = len(self.target_lines)
        self.content_width = max(len(line) for line in self.target_lines) if self.target_lines else 0

        self.padded_target_lines = []
        if self.content_height > 0 and self.content_width > 0:
            for i in range(self.content_height):
                line = self.target_lines[i] if i < len(self.target_lines) else ""
                self.padded_target_lines.append(line + ' ' * (self.content_width - len(line)))
        else: # Handle empty target_content
            self.content_height = 1
            self.content_width = 1
            self.padded_target_lines = [" "]


    def _get_block_char(self, r_start: int, c_start: int, pixel_size: int) -> str:
        """Determines the representative character for a block."""
        if not self.padded_target_lines: return " "

        char_counts = {}
        num_chars_in_block = 0
        for r_offset in range(pixel_size):
            for c_offset in range(pixel_size):
                r, c = r_start + r_offset, c_start + c_offset
                if 0 <= r < self.content_height and 0 <= c < self.content_width:
                    char = self.padded_target_lines[r][c]
                    if char != ' ': # Ignore spaces for dominant char, or include if you want space to dominate
                        char_counts[char] = char_counts.get(char, 0) + 1
                        num_chars_in_block +=1

        if not char_counts: # Block is all spaces or out of bounds
            # Check the top-left char of the block in target art for a hint
            if 0 <= r_start < self.content_height and 0 <= c_start < self.content_width:
                 return self.padded_target_lines[r_start][c_start] # Could be a space
            return " "

        # Return the most frequent character in the block
        dominant_char = max(char_counts, key=char_counts.get)
        return dominant_char

    def update(self) -> Optional[str]:
        if not self.padded_target_lines : return " "

        elapsed_time = time.time() - self.start_time
        progress = min(1.0, elapsed_time / self.duration)

        current_pixel_size = 1
        if self.effect_type == "resolve":
            # Pixel size decreases from max_pixel_size to 1
            # Using (1-progress) for size factor, so at progress=0, factor=1 (max size)
            # and at progress=1, factor=0 (min size = 1)
            size_factor = 1.0 - progress
            current_pixel_size = 1 + int(size_factor * (self.max_pixel_size - 1))
        elif self.effect_type == "pixelate":
            # Pixel size increases from 1 to max_pixel_size
            size_factor = progress
            current_pixel_size = 1 + int(size_factor * (self.max_pixel_size - 1))

        current_pixel_size = max(1, current_pixel_size) # Ensure it's at least 1

        if current_pixel_size == 1 and self.effect_type == "resolve":
            return "\n".join(self.padded_target_lines).replace('[',r'\[')
        if current_pixel_size == self.max_pixel_size and self.effect_type == "pixelate" and progress >=1.0:
            # Final pixelated state, render it once more and then could be static
             pass # Let it render below

        output_art_chars = [[' ' for _ in range(self.content_width)] for _ in range(self.content_height)]

        for r_block_start in range(0, self.content_height, current_pixel_size):
            for c_block_start in range(0, self.content_width, current_pixel_size):
                block_char = self._get_block_char(r_block_start, c_block_start, current_pixel_size)
                for r_offset in range(current_pixel_size):
                    for c_offset in range(current_pixel_size):
                        r, c = r_block_start + r_offset, c_block_start + c_offset
                        if 0 <= r < self.content_height and 0 <= c < self.content_width:
                            output_art_chars[r][c] = block_char

        # The card's base style will apply. No specific styling here unless needed.
        return "\n".join("".join(row) for row in output_art_chars).replace('[',r'\[')