"""GameOfLife splash screen effect."""

from rich.style import Style
import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("game_of_life")
class GameOfLifeEffect(BaseEffect):
    """Simulates Conway's Game of Life."""

    def __init__(
        self,
        parent_widget: Any,
        title: str = "Evolving Systems...",
        width: int = 40, # Grid width for GoL (actual output width might be larger for title)
        height: int = 20, # Grid height for GoL
        cell_alive_char: str = "█",
        cell_dead_char: str = " ",
        alive_style: str = "bold green",
        dead_style: str = "black", # Effectively background
        initial_pattern: str = "random", # "random", or specific pattern names like "glider"
        update_interval: float = 0.2, # How often GoL updates, distinct from animation_speed
        title_style: str = "bold white",
        display_width: int = 80, # Total width for splash screen display
        display_height: int = 24, # Total height for splash screen display
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.grid_width = width
        self.grid_height = height
        self.cell_alive_char = cell_alive_char[0]
        self.cell_dead_char = cell_dead_char[0]
        self.alive_style = alive_style
        self.dead_style = dead_style # Usually background color
        self.initial_pattern = initial_pattern
        self.title_style = title_style
        self.display_width = display_width
        self.display_height = display_height

        self.grid = [[(random.choice([0,1]) if initial_pattern == "random" else 0) for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        if self.initial_pattern == "glider":
            # A common GoL pattern
            if self.grid_width >= 3 and self.grid_height >= 3:
                self.grid[0][1] = 1
                self.grid[1][2] = 1
                self.grid[2][0] = 1
                self.grid[2][1] = 1
                self.grid[2][2] = 1
        # Can add more predefined patterns here

        self._last_gol_update_time = time.time()
        self.gol_update_interval = update_interval


    def _count_neighbors(self, r: int, c: int) -> int:
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nr, nc = r + i, c + j
                # Toroidal grid (wraps around)
                nr = nr % self.grid_height
                nc = nc % self.grid_width
                count += self.grid[nr][nc]
        return count

    def _update_grid(self):
        new_grid = [[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                neighbors = self._count_neighbors(r, c)
                if self.grid[r][c] == 1: # Alive
                    if neighbors < 2 or neighbors > 3:
                        new_grid[r][c] = 0 # Dies
                    else:
                        new_grid[r][c] = 1 # Lives
                else: # Dead
                    if neighbors == 3:
                        new_grid[r][c] = 1 # Becomes alive
        self.grid = new_grid

    def update(self) -> Optional[str]:
        current_time = time.time()
        if current_time - self._last_gol_update_time >= self.gol_update_interval:
            self._update_grid()
            self._last_gol_update_time = current_time

        # Render the grid and title to the full display_width/height
        output_display = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styled_output_lines = [""] * self.display_height

        # Center the GoL grid within the display area
        grid_start_r = (self.display_height - self.grid_height) // 2
        grid_start_c = (self.display_width - self.grid_width) // 2

        for r_grid in range(self.grid_height):
            for c_grid in range(self.grid_width):
                r_disp, c_disp = grid_start_r + r_grid, grid_start_c + c_grid
                if 0 <= r_disp < self.display_height and 0 <= c_disp < self.display_width:
                    is_alive = self.grid[r_grid][c_grid] == 1
                    char_to_draw = self.cell_alive_char if is_alive else self.cell_dead_char
                    style_to_use = self.alive_style if is_alive else self.dead_style
                    # Store as (char, style) tuple for later Rich formatting
                    output_display[r_disp][c_disp] = (char_to_draw, style_to_use)

        # Convert output_display (which has tuples or spaces) to styled lines
        for r_idx in range(self.display_height):
            line_segments = []
            for c_idx in range(self.display_width):
                cell_content = output_display[r_idx][c_idx]
                if isinstance(cell_content, tuple):
                    char, style = cell_content
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else: # It's a space from initialization
                    line_segments.append(f"[{self.dead_style}] [/{self.dead_style}]") # Styled background space
            styled_output_lines[r_idx] = "".join(line_segments)


        # Overlay title (centered, typically above or below GoL grid)
        if self.title:
            title_y = grid_start_r - 2 if grid_start_r > 1 else self.display_height -1 # Place above or at bottom
            title_x_start = (self.display_width - len(self.title)) // 2

            if 0 <= title_y < self.display_height:
                # Construct title line, preserving background cells not covered by title
                title_line_segments = []
                title_char_idx = 0
                for c_idx in range(self.display_width):
                    is_title_char_pos = title_x_start <= c_idx < title_x_start + len(self.title)
                    if is_title_char_pos:
                        char = self.title[title_char_idx].replace('[',r'\[')
                        title_line_segments.append(f"[{self.title_style}]{char}[/{self.title_style}]")
                        title_char_idx += 1
                    else: # Part of the line not covered by title, use existing content
                        cell_content = output_display[title_y][c_idx]
                        if isinstance(cell_content, tuple):
                            char, style = cell_content
                            escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                            title_line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                        else:
                            title_line_segments.append(f"[{self.dead_style}] [/{self.dead_style}]")
                styled_output_lines[title_y] = "".join(title_line_segments)

        return "\n".join(styled_output_lines)