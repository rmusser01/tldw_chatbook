"""RetroGamingIntro splash screen effect."""

from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("retro_gaming_intro")
class RetroGamingIntroEffect(BaseEffect):
    """A tribute to classic 8-bit and 16-bit game intros."""
    def __init__(self, parent_widget: Any, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.scroll_y = self.height
        self.text = ["A JULES PRODUCTION", "", "PRESENTING", "", "TLDW CHATBOOK"]

    def update(self) -> Optional[str]:
        self.scroll_y -= 0.5
        if self.scroll_y < -len(self.text):
            self.scroll_y = self.height

        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        for i, line in enumerate(self.text):
            y_pos = int(self.scroll_y + i)
            x_pos = (self.width - len(line)) // 2
            if 0 <= y_pos < self.height:
                for j, char in enumerate(line):
                    grid[y_pos][x_pos+j] = char
                    styles[y_pos][x_pos+j] = 'bold blue'

        return self._grid_to_string(grid, styles)