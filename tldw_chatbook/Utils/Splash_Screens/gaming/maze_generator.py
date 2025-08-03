"""MazeGenerator splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("maze_generator")
class MazeGeneratorEffect(BaseEffect):
    """Animates the generation of a random maze using Depth First Search."""

    CELL_PATH_N = 1
    CELL_PATH_E = 2
    CELL_PATH_S = 4
    CELL_PATH_W = 8
    CELL_VISITED = 16

    def __init__(
        self,
        parent_widget: Any,
        title: str = "Generating Labyrinth...",
        maze_width: int = 39, # Grid cells (must be odd for typical wall representation)
        maze_height: int = 19, # Grid cells (must be odd)
        wall_char: str = "█",
        path_char: str = " ",
        cursor_char: str = "░", # Char for the current cell being processed
        wall_style: str = "bold blue",
        path_style: str = "on black", # Path is often just background
        cursor_style: str = "bold yellow",
        title_style: str = "bold white",
        generation_speed: float = 0.01, # Delay between steps of generation
        display_width: int = 80, # Total splash screen width
        display_height: int = 24,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        # Ensure maze dimensions are odd for typical cell/wall structure
        self.maze_cols = maze_width if maze_width % 2 != 0 else maze_width -1
        self.maze_rows = maze_height if maze_height % 2 != 0 else maze_height -1
        if self.maze_cols < 3: self.maze_cols = 3
        if self.maze_rows < 3: self.maze_rows = 3

        self.wall_char = wall_char
        self.path_char = path_char
        self.cursor_char = cursor_char
        self.wall_style = wall_style
        self.path_style = path_style
        self.cursor_style = cursor_style
        self.title_style = title_style
        self.generation_speed = generation_speed # Interval for maze generation steps
        self.display_width = display_width
        self.display_height = display_height

        # Maze grid: stores bitmasks for paths and visited status
        self.maze_grid = [[0 for _ in range(self.maze_cols)] for _ in range(self.maze_rows)]
        self.stack = [] # For DFS algorithm

        # Start DFS from a random cell (must be an actual cell, not a wall position)
        # In our grid, (0,0) is a cell.
        self.current_cx = random.randrange(0, self.maze_cols, 2) # Ensure starting on a "cell" column if we consider walls
        self.current_cy = random.randrange(0, self.maze_rows, 2) # Ensure starting on a "cell" row
        # Simpler: map to a conceptual grid of cells (width/2, height/2) then map back to drawing grid
        # Let's use a cell-based grid for logic, then render to character grid.
        self.logic_cols = (self.maze_cols +1) // 2
        self.logic_rows = (self.maze_rows +1) // 2
        self.logic_grid = [[0 for _ in range(self.logic_cols)] for _ in range(self.logic_rows)]

        self.current_logic_x = self.current_cx // 2
        self.current_logic_y = self.current_cy // 2
        self.logic_grid[self.current_logic_y][self.current_logic_x] = self.CELL_VISITED
        self.stack.append((self.current_logic_x, self.current_logic_y))

        self.is_generating = True
        self._last_gen_step_time = time.time()

    def _generation_step_dfs(self):
        if not self.stack:
            self.is_generating = False
            return

        x, y = self.stack[-1] # Current cell
        self.current_logic_x, self.current_logic_y = x,y # For cursor drawing

        neighbors = []
        # Check North
        if y > 0 and self.logic_grid[y-1][x] & self.CELL_VISITED == 0: neighbors.append(('N', x, y-1))
        # Check East
        if x < self.logic_cols - 1 and self.logic_grid[y][x+1] & self.CELL_VISITED == 0: neighbors.append(('E', x+1, y))
        # Check South
        if y < self.logic_rows - 1 and self.logic_grid[y+1][x] & self.CELL_VISITED == 0: neighbors.append(('S', x, y+1))
        # Check West
        if x > 0 and self.logic_grid[y][x-1] & self.CELL_VISITED == 0: neighbors.append(('W', x-1, y))

        if neighbors:
            direction, nx, ny = random.choice(neighbors)
            if direction == 'N':
                self.logic_grid[y][x] |= self.CELL_PATH_N
                self.logic_grid[ny][nx] |= self.CELL_PATH_S
            elif direction == 'E':
                self.logic_grid[y][x] |= self.CELL_PATH_E
                self.logic_grid[ny][nx] |= self.CELL_PATH_W
            elif direction == 'S':
                self.logic_grid[y][x] |= self.CELL_PATH_S
                self.logic_grid[ny][nx] |= self.CELL_PATH_N
            elif direction == 'W':
                self.logic_grid[y][x] |= self.CELL_PATH_W
                self.logic_grid[ny][nx] |= self.CELL_PATH_E

            self.logic_grid[ny][nx] |= self.CELL_VISITED
            self.stack.append((nx, ny))
        else:
            self.stack.pop() # Backtrack

    def update(self) -> Optional[str]:
        current_time = time.time()
        if self.is_generating and (current_time - self._last_gen_step_time >= self.generation_speed):
            self._generation_step_dfs()
            self._last_gen_step_time = current_time

        # Render the maze to the display grid (display_width x display_height)
        # The maze itself is self.maze_cols x self.maze_rows characters
        output_grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styled_output_lines = [""] * self.display_height

        maze_start_row = (self.display_height - self.maze_rows) // 2
        maze_start_col = (self.display_width - self.maze_cols) // 2

        for r_draw in range(self.maze_rows):
            for c_draw in range(self.maze_cols):
                r_disp = maze_start_row + r_draw
                c_disp = maze_start_col + c_draw

                if not (0 <= r_disp < self.display_height and 0 <= c_disp < self.display_width):
                    continue # Skip drawing outside display boundary

                char_to_draw = self.wall_char
                style_to_use = self.wall_style

                # Convert draw coords to logic grid cell coords and wall/path determination
                logic_x, logic_y = c_draw // 2, r_draw // 2

                is_wall_row = r_draw % 2 == 1
                is_wall_col = c_draw % 2 == 1

                if not is_wall_row and not is_wall_col: # Cell center
                    char_to_draw = self.path_char
                    style_to_use = self.path_style
                    if self.is_generating and logic_x == self.current_logic_x and logic_y == self.current_logic_y:
                        char_to_draw = self.cursor_char
                        style_to_use = self.cursor_style

                elif not is_wall_row and is_wall_col: # Horizontal wall/path between cells (y, x) and (y, x+1)
                    if logic_x < self.logic_cols -1 and \
                       (self.logic_grid[logic_y][logic_x] & self.CELL_PATH_E or \
                        self.logic_grid[logic_y][logic_x+1] & self.CELL_PATH_W):
                        char_to_draw = self.path_char
                        style_to_use = self.path_style

                elif is_wall_row and not is_wall_col: # Vertical wall/path between cells (y,x) and (y+1,x)
                     if logic_y < self.logic_rows -1 and \
                       (self.logic_grid[logic_y][logic_x] & self.CELL_PATH_S or \
                        self.logic_grid[logic_y+1][logic_x] & self.CELL_PATH_N):
                        char_to_draw = self.path_char
                        style_to_use = self.path_style
                # else it's a wall intersection, keep wall_char

                output_grid[r_disp][c_disp] = (char_to_draw, style_to_use)

        # Convert to styled lines
        for r_idx in range(self.display_height):
            line_segments = []
            for c_idx in range(self.display_width):
                cell = output_grid[r_idx][c_idx]
                if isinstance(cell, tuple):
                    char, style = cell
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else: # Space, apply default path style or background
                    line_segments.append(f"[{self.path_style}] [/{self.path_style}]")
            styled_output_lines[r_idx] = "".join(line_segments)

        # Overlay title
        if self.title:
            title_y = maze_start_row - 2 if maze_start_row > 1 else self.display_height - 1
            if not self.is_generating: title_y = self.display_height // 2 # Center title when done

            title_x_start = (self.display_width - len(self.title)) // 2
            if 0 <= title_y < self.display_height:
                # Simplified title overlay for now: assumes it replaces the line
                title_line_str = self.title.center(self.display_width).replace('[',r'\[')
                styled_output_lines[title_y] = f"[{self.title_style}]{title_line_str}[/{self.title_style}]"

        return "\n".join(styled_output_lines)