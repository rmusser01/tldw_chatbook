"""BookshelfBrowser splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("bookshelf_browser")
class BookshelfBrowserEffect(BaseEffect):
    """Animated bookshelf with sliding books and page flipping."""
    
    def __init__(
        self,
        parent_widget: Any,
        width: int = 80,
        height: int = 24,
        speed: float = 0.1,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.width = width
        self.height = height
        self.speed = speed
        
        # Book titles for spines
        self.book_titles = [
            "PYTHON", "CHAT", "AI", "DATA", "CODE", "LEARN",
            "DOCS", "API", "TEXT", "BOOK", "READ", "WIKI"
        ]
        
        # Book animation state
        self.selected_book = -1
        self.book_open_progress = 0.0
        self.page_flip_progress = 0.0
        self.title_reveal_progress = 0.0
        
    def update(self) -> Optional[str]:
        """Update the bookshelf animation."""
        elapsed = time.time() - self.start_time
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Animation phases
        if elapsed < 1.0:
            # Phase 1: Show bookshelf
            phase = "shelf"
        elif elapsed < 2.0:
            # Phase 2: Select and pull out book
            phase = "select"
            self.selected_book = 5  # Middle book
            self.book_open_progress = (elapsed - 1.0)
        elif elapsed < 3.5:
            # Phase 3: Open book and flip pages
            phase = "flip"
            self.page_flip_progress = (elapsed - 2.0) / 1.5
        else:
            # Phase 4: Reveal title
            phase = "reveal"
            self.title_reveal_progress = (elapsed - 3.5) / 0.5
        
        # Draw bookshelf frame
        shelf_y = 8
        shelf_width = 60
        shelf_x = (self.width - shelf_width) // 2
        
        # Top shelf
        for x in range(shelf_x, shelf_x + shelf_width):
            if x < self.width:
                grid[shelf_y - 1][x] = '═'
        
        # Draw books
        book_width = 5
        for i, title in enumerate(self.book_titles[:10]):
            book_x = shelf_x + 2 + i * 6
            
            # Skip selected book if it's being pulled out
            if phase in ["select", "flip", "reveal"] and i == self.selected_book:
                continue
            
            # Draw book spine
            for y in range(shelf_y, shelf_y + 8):
                if y < self.height and book_x < self.width - book_width:
                    grid[y][book_x] = '│'
                    grid[y][book_x + book_width] = '│'
                    
                    # Title on spine (vertical)
                    if y - shelf_y < len(title):
                        char_idx = y - shelf_y
                        if char_idx < len(title):
                            for x_off in range(1, book_width):
                                if book_x + x_off < self.width:
                                    grid[y][book_x + x_off] = title[char_idx]
        
        # Bottom shelf
        for x in range(shelf_x, shelf_x + shelf_width):
            if x < self.width and shelf_y + 8 < self.height:
                grid[shelf_y + 8][x] = '═'
        
        # Draw selected book animation
        if phase in ["select", "flip", "reveal"]:
            # Calculate book position
            book_start_x = shelf_x + 2 + self.selected_book * 6
            book_pull_distance = int(self.book_open_progress * 15)
            book_center_x = book_start_x - book_pull_distance
            book_center_y = shelf_y + 2
            
            if phase == "flip" or phase == "reveal":
                # Draw open book
                book_art = [
                    "     ╭─────────────╮     ",
                    "    ╱               ╲    ",
                    "   │                 │   ",
                    "   │                 │   ",
                    "   │                 │   ",
                    "   │                 │   ",
                    "   │                 │   ",
                    "   ╰─────────────────╯   "
                ]
                
                # Draw book
                for i, line in enumerate(book_art):
                    for j, char in enumerate(line):
                        x = book_center_x + j
                        y = book_center_y + i
                        if 0 <= x < self.width and 0 <= y < self.height and char != ' ':
                            grid[y][x] = char
                
                # Draw page content or title
                if phase == "reveal" and self.title_reveal_progress > 0.3:
                    # Show title
                    title_lines = ["TLDW", "CHATBOOK"]
                    for i, line in enumerate(title_lines):
                        line_x = book_center_x + (25 - len(line)) // 2
                        line_y = book_center_y + 2 + i * 2
                        for j, char in enumerate(line):
                            if 0 <= line_x + j < self.width and 0 <= line_y < self.height:
                                if random.random() < self.title_reveal_progress:
                                    grid[line_y][line_x + j] = char
                else:
                    # Show flipping pages
                    flip_chars = ['/', '|', '\\', '─']
                    flip_idx = int(self.page_flip_progress * 4) % 4
                    flip_x = book_center_x + 12
                    flip_y = book_center_y + 4
                    if 0 <= flip_x < self.width and 0 <= flip_y < self.height:
                        grid[flip_y][flip_x] = flip_chars[flip_idx]
        
        # Draw header
        if phase == "shelf":
            header = "Welcome to the Library"
        elif phase == "select":
            header = "Selecting a book..."
        elif phase == "flip":
            header = "Opening pages..."
        else:
            header = "Your adventure begins!"
        
        header_x = (self.width - len(header)) // 2
        for i, char in enumerate(header):
            if 0 <= header_x + i < self.width:
                grid[2][header_x + i] = char
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if y == 2 and char != ' ':  # Header
                    line += f"[bold yellow]{char}[/bold yellow]"
                elif char in '═│':  # Shelf
                    line += f"[dim white]{char}[/dim white]"
                elif char in '╱╲╮╯╰─':  # Book outline
                    line += f"[white]{char}[/white]"
                elif char in 'TLDWCHATBOOK' and phase == "reveal":  # Title
                    line += f"[bold cyan]{char}[/bold cyan]"
                elif char.isalpha():  # Book spines
                    line += f"[dim yellow]{char}[/dim yellow]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)