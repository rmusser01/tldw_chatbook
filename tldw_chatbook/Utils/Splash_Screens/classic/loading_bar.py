"""LoadingBar splash screen effect."""

from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect, ESCAPED_OPEN_BRACKET


@register_effect("loading_bar")
class LoadingBarEffect(BaseEffect):
    """Displays an ASCII loading bar that fills based on SplashScreen's progress."""

    def __init__(
        self,
        parent_widget: Any, # This will be the SplashScreen instance
        bar_frame_content: str, # ASCII for the empty bar e.g., "[----------]"
        fill_char: str = "#",
        bar_style: str = "bold green",
        text_above: str = "LOADING MODULES...",
        text_below: str = "{progress:.0f}% Complete", # Format string for progress
        text_style: str = "white",
        width: int = 80, # Target width for centering
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.bar_frame_content = bar_frame_content.strip()
        self.fill_char = fill_char[0] if fill_char else "#" # Ensure single char
        self.bar_style = bar_style
        self.text_above = text_above
        self.text_below_template = text_below
        self.text_style = text_style
        self.width = width

        # Try to determine bar width from frame_content (excluding brackets/ends)
        self.bar_interior_width = len(self.bar_frame_content) - 2 # Assuming frame is like [---]
        if self.bar_interior_width <= 0:
            self.bar_interior_width = 20 # Default if frame is unusual

    def update(self) -> Optional[str]:
        # Access progress from the parent SplashScreen widget
        # The parent SplashScreen widget has a reactive 'progress' attribute (0.0 to 1.0)
        current_progress = self.parent.progress if hasattr(self.parent, 'progress') else 0.0

        num_filled = int(current_progress * self.bar_interior_width)
        num_empty = self.bar_interior_width - num_filled

        # Construct the bar
        # Assuming bar_frame_content is like "[--------------------]"
        bar_start = self.bar_frame_content[0]
        bar_end = self.bar_frame_content[-1]

        filled_part = self.fill_char * num_filled
        empty_part = self.bar_frame_content[1+num_filled : 1+num_filled+num_empty] # Get actual empty chars from frame

        # Ensure the bar is always the correct total interior width
        current_bar_interior = filled_part + empty_part
        if len(current_bar_interior) < self.bar_interior_width:
            current_bar_interior += self.bar_frame_content[1+len(current_bar_interior)] * (self.bar_interior_width - len(current_bar_interior))
        elif len(current_bar_interior) > self.bar_interior_width:
            current_bar_interior = current_bar_interior[:self.bar_interior_width]


        styled_bar = f"[{self.bar_style}]{bar_start}{current_bar_interior}{bar_end}[/{self.bar_style}]"

        # Format text below with current progress
        # The parent also has 'progress_text' which might be more descriptive
        progress_percentage_text = self.parent.progress_text if hasattr(self.parent, 'progress_text') and self.parent.progress_text else ""

        # Use the template if available, otherwise use the SplashScreen's progress_text
        if "{progress}" in self.text_below_template:
             text_below_formatted = self.text_below_template.format(progress=current_progress * 100)
        else:
            text_below_formatted = progress_percentage_text if progress_percentage_text else f"{current_progress*100:.0f}%"


        # Centering text and bar (approximate)
        output_lines = []
        if self.text_above:
            pad_above = (self.width - len(self.text_above)) // 2
            escaped_text = self.text_above.replace('[', ESCAPED_OPEN_BRACKET)
            output_lines.append(f"[{self.text_style}]{' ' * pad_above}{escaped_text}{' ' * pad_above}[/{self.text_style}]")
        else:
            output_lines.append("") # Keep spacing consistent

        pad_bar = (self.width - len(self.bar_frame_content)) // 2
        output_lines.append(f"{' ' * pad_bar}{styled_bar}")

        if text_below_formatted:
            pad_below = (self.width - len(text_below_formatted)) // 2
            escaped_text_below = text_below_formatted.replace('[', ESCAPED_OPEN_BRACKET)
            output_lines.append(f"[{self.text_style}]{' ' * pad_below}{escaped_text_below}{' ' * pad_below}[/{self.text_style}]")
        else:
            output_lines.append("")

        # Add some blank lines for spacing if needed, assuming height of ~5-7 lines for this effect
        while len(output_lines) < 5: # Assuming a small vertical footprint
            output_lines.insert(0, "") # Add blank lines at the top for centering
            if len(output_lines) >=5: break
            output_lines.append("") # Add blank lines at the bottom

        return '\n'.join(output_lines[:self.parent.height if hasattr(self.parent, 'height') else 7])