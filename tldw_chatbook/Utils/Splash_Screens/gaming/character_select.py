"""CharacterSelect splash screen effect."""

from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("character_select")
class CharacterSelectEffect(BaseEffect):
    """A character selection screen with portraits and a selector."""
    def __init__(self, parent_widget: Any, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.characters = [
            {'name': 'JULES', 'portrait': ['(⌐■_■)', ' ']},
            {'name': 'CLAUDE', 'portrait': ['(·_·)', ' ']},
            {'name': 'GEMINI', 'portrait': ['(o.o)', ' ']},
        ]
        self.selected_char = 0
        self.selector_pos = 0

    def update(self) -> Optional[str]:
        if self.frame_count % 15 == 0:
            self.selected_char = (self.selected_char + 1) % len(self.characters)

        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        for i, char_data in enumerate(self.characters):
            x_pos = (self.width // (len(self.characters) + 1)) * (i + 1)
            for j, line in enumerate(char_data['portrait']):
                for k, char in enumerate(line):
                    grid[self.height // 2 + j][x_pos - len(line)//2 + k] = char
            name = char_data['name']
            for k, char in enumerate(name):
                 grid[self.height // 2 + 3][x_pos - len(name)//2 + k] = char

        selector_x = (self.width // (len(self.characters) + 1)) * (self.selected_char + 1)
        grid[self.height // 2 - 2][selector_x] = '▼'
        styles[self.height // 2 - 2][selector_x] = 'bold red'

        return self._grid_to_string(grid, styles)