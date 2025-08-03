"""AchievementUnlocked splash screen effect."""

from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("achievement_unlocked")
class AchievementUnlockedEffect(BaseEffect):
    """An 'Achievement Unlocked' notification that slides in and out."""
    def __init__(self, parent_widget: Any, **kwargs):
        super().__init__(parent_widget, **kwargs)
        self.width = kwargs.get('width', 80)
        self.height = kwargs.get('height', 24)
        self.achievement = "Achievement Unlocked: New Splash Screens!"
        self.y_pos = -3
        self.state = 'in'

    def update(self) -> Optional[str]:
        if self.state == 'in':
            self.y_pos += 1
            if self.y_pos >= self.height // 2:
                self.y_pos = self.height // 2
                if self.frame_count > 100:
                    self.state = 'out'
        elif self.state == 'out':
            self.y_pos -=1

        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]

        box_width = len(self.achievement) + 4
        box_x = (self.width - box_width) // 2

        if self.y_pos > -3 and self.y_pos < self.height - 2:  # Ensure we have space for 3 rows
            for i in range(box_width):
                if box_x + i < self.width and self.y_pos >= 0:
                    grid[self.y_pos][box_x + i] = '─'
                if box_x + i < self.width and self.y_pos + 2 < self.height:
                    grid[self.y_pos+2][box_x + i] = '─'
            
            if self.y_pos + 1 < self.height:
                for i in range(len(self.achievement)):
                    if box_x + 2 + i < self.width:
                        grid[self.y_pos+1][box_x+2+i] = self.achievement[i]

                if box_x < self.width:
                    grid[self.y_pos+1][box_x] = '🏆'
                    styles[self.y_pos+1][box_x] = 'bold yellow'

        return self._grid_to_string(grid, styles)