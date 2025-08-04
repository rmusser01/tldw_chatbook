"""ScrollingCredits splash screen effect."""

import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("scrolling_credits")
class ScrollingCreditsEffect(BaseEffect):
    """Simulates scrolling credits, like at the end of a movie."""

    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        credits_list: List[Dict[str, str]] = None, # Each dict: {"role": "Concept", "name": "The Universe"} or just {"line": "Some text"}
        scroll_speed: float = 1.0, # Lines per second (can be fractional)
        line_spacing: int = 1, # Number of blank lines between credit entries
        width: int = 80,
        height: int = 24,
        title_style: str = "bold yellow",
        role_style: str = "bold white",
        name_style: str = "white",
        line_style: str = "white", # For single line credits
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.overall_title = title # Title for the splash screen itself, displayed statically
        self.credits_list = credits_list if credits_list else [{"line": "Loading..."}]
        self.scroll_speed = scroll_speed # This will be used to calculate fractional line shifts
        self.line_spacing = line_spacing
        self.display_width = width
        self.display_height = height
        self.title_style = title_style
        self.role_style = role_style
        self.name_style = name_style
        self.line_style = line_style

        self.formatted_credit_lines: List[str] = []
        self._format_credits()

        self.current_scroll_offset = float(self.display_height) # Start with credits off-screen at the bottom
        self.time_at_last_frame = time.time()


    def _format_credits(self):
        """Pre-formats credit entries into Rich-styled strings."""
        self.formatted_credit_lines.append(f"[{self.title_style}]{self.overall_title.center(self.display_width)}[/{self.title_style}]")
        self.formatted_credit_lines.append("") # Blank line after title

        for item in self.credits_list:
            if "line" in item:
                # Single line entry
                text = item["line"].replace('[', r'\[')
                self.formatted_credit_lines.append(f"[{self.line_style}]{text.center(self.display_width)}[/{self.line_style}]")
            elif "role" in item and "name" in item:
                # Role: Name format
                role_text = item["role"].replace('[', r'\[')
                name_text = item["name"].replace('[', r'\[')
                # Simple centered alignment for now
                # More complex alignment (role left, name right) is harder with fixed width and Rich.
                # For now, centered role, then centered name on next line or combined.
                # Let's do: Role (centered), Name (centered below it)
                self.formatted_credit_lines.append(f"[{self.role_style}]{role_text.center(self.display_width)}[/{self.role_style}]")
                self.formatted_credit_lines.append(f"[{self.name_style}]{name_text.center(self.display_width)}[/{self.name_style}]")

            for _ in range(self.line_spacing):
                self.formatted_credit_lines.append("") # Add blank lines for spacing

        # Add some padding at the end so last credit scrolls fully off
        for _ in range(self.display_height // 2):
            self.formatted_credit_lines.append("")


    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.time_at_last_frame
        self.time_at_last_frame = current_time

        self.current_scroll_offset -= delta_time * self.scroll_speed

        # Determine which lines are visible
        output_lines = []
        start_line_index = int(self.current_scroll_offset)

        for i in range(self.display_height):
            current_line_to_fetch = start_line_index + i
            if 0 <= current_line_to_fetch < len(self.formatted_credit_lines):
                output_lines.append(self.formatted_credit_lines[current_line_to_fetch])
            else:
                output_lines.append(' ' * self.display_width) # Blank line

        # Reset scroll if all credits have passed
        # Total height of credits content: len(self.formatted_credit_lines)
        # Resets when the top of the credits (index 0) has scrolled past the top of the screen (offset becomes negative enough)
        if self.current_scroll_offset < -len(self.formatted_credit_lines):
             self.current_scroll_offset = float(self.display_height) # Reset to start from bottom again

        return "\n".join(output_lines)