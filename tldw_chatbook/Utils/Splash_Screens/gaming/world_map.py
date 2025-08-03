"""WorldMap splash screen effect."""

from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("world_map")
class WorldMapEffect(BaseEffect):
    """An ASCII world map with a blinking cursor moving between locations."""
    def __init__(self, parent_widget: Any, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.map_data = [
            "       ~~~~~      ",
            "  ..   ~~~~~   .. ",
            " ...    ~~~    ... ",
            "........ ... ......"
        ]
        self.locations = [(5,1), (15,2), (8,3)]
        self.cursor_pos_index = 0

    def update(self) -> Optional[str]:
        if self.frame_count % 20 == 0:
            self.cursor_pos_index = (self.cursor_pos_index + 1) % len(self.locations)

        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        for y, line in enumerate(self.map_data):
            for x, char in enumerate(line):
                grid[y+5][x+10] = char
                if char == '~': styles[y+5][x+10] = 'blue'
                else: styles[y+5][x+10] = 'green'

        cursor_x, cursor_y = self.locations[self.cursor_pos_index]
        if self.frame_count % 10 < 5:
             grid[cursor_y+5][cursor_x+10] = 'X'
             styles[cursor_y+5][cursor_x+10] = 'bold red'

        return self._grid_to_string(grid, styles)