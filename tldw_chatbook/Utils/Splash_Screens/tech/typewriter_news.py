"""TypewriterNews splash screen effect."""
import time

from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("typewriter_news")
class TypewriterNewsEffect(BaseEffect):
    """Old newspaper typewriter effect with breaking news."""
    
    def __init__(self, parent, width=80, height=24, speed=0.05, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.typed_chars = 0
        self.paper_lines = []
        self.cursor_blink = 0
        self.carriage_return_sound = False
        
        # News content
        self.headline = "BREAKING: TLDW CHATBOOK LAUNCHES!"
        self.subheadline = "Revolutionary AI Assistant Takes Terminal By Storm"
        self.dateline = "Terminal City - " + time.strftime("%B %d, %Y")
        self.article = [
            "In a stunning development today, the highly anticipated",
            "TLDW Chatbook has been released to the public. This",
            "groundbreaking terminal-based AI assistant promises to",
            "revolutionize how users interact with language models.",
            "",
            "Early reports indicate unprecedented user satisfaction",
            "with the innovative ASCII-based interface and powerful",
            "conversation management features.",
            "",
            '"This changes everything," said one beta tester.',
        ]
    
    def update(self) -> Optional[str]:
        """Update typewriter animation."""
        elapsed_time = time.time() - self.start_time
        # Type characters
        self.typed_chars += elapsed_time * 30  # Characters per second
        
        # Cursor blink
        self.cursor_blink = (self.cursor_blink + elapsed_time * 3) % 1.0
        
        # Check for carriage return
        if int(self.typed_chars) > 0 and int(self.typed_chars) % 60 == 0:
            self.carriage_return_sound = True
        else:
            self.carriage_return_sound = False
    
    def render(self):
        """Render typewriter news effect."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw paper background
        for y in range(2, self.height - 2):
            for x in range(5, self.width - 5):
                grid[y][x] = ' '
                style_grid[y][x] = 'on rgb(245,245,220)'  # Light beige
        
        # Draw paper edges
        for y in range(2, self.height - 2):
            grid[y][4] = '│'
            grid[y][self.width - 5] = '│'
            style_grid[y][4] = style_grid[y][self.width - 5] = 'black'
        
        # Type content
        current_line = 4
        chars_typed = int(self.typed_chars)
        
        # Type headline
        if chars_typed > 0:
            headline_typed = self.headline[:min(chars_typed, len(self.headline))]
            self._add_centered_text(grid, style_grid, headline_typed, current_line, 'bold black on rgb(245,245,220)')
            chars_typed -= len(self.headline)
            current_line += 2
        
        # Type subheadline
        if chars_typed > 0:
            subheadline_typed = self.subheadline[:min(chars_typed, len(self.subheadline))]
            self._add_centered_text(grid, style_grid, subheadline_typed, current_line, 'black on rgb(245,245,220)')
            chars_typed -= len(self.subheadline)
            current_line += 2
        
        # Type dateline
        if chars_typed > 0:
            dateline_typed = self.dateline[:min(chars_typed, len(self.dateline))]
            self._add_text_at(grid, style_grid, dateline_typed, 8, current_line, 'italic black on rgb(245,245,220)')
            chars_typed -= len(self.dateline)
            current_line += 2
        
        # Type article
        for line in self.article:
            if chars_typed > 0 and current_line < self.height - 4:
                line_typed = line[:min(chars_typed, len(line))]
                self._add_text_at(grid, style_grid, line_typed, 8, current_line, 'black on rgb(245,245,220)')
                chars_typed -= len(line)
                current_line += 1
        
        # Draw cursor
        if self.cursor_blink > 0.5 and chars_typed >= 0:
            cursor_pos = min(int(self.typed_chars), sum(len(line) for line in [self.headline, self.subheadline, self.dateline] + self.article))
            # Find cursor position (simplified)
            if current_line < self.height - 4:
                grid[current_line][min(8 + (cursor_pos % 50), self.width - 6)] = '█'
                style_grid[current_line][min(8 + (cursor_pos % 50), self.width - 6)] = 'black on rgb(245,245,220)'
        
        # Add typewriter sound effect
        if self.carriage_return_sound:
            self._add_text_at(grid, style_grid, "DING!", 2, 2, 'bold red')
        
        return self._grid_to_string(grid, style_grid)
    
    def _add_text_at(self, grid, style_grid, text, x, y, style):
        """Add text at specific position."""
        for i, char in enumerate(text):
            if x + i < len(grid[0]):
                grid[y][x + i] = char
                style_grid[y][x + i] = style