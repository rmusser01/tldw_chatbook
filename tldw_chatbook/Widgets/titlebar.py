from textual.widgets import Static
from rich.text import Text


class TitleBar(Static):
    """A one-line decorative title bar with emoji art."""
    DEFAULT_ART = "[b]tldw chatbook[/b]"

    def __init__(self) -> None:
        super().__init__(Text.from_markup(self.DEFAULT_ART), id="app-titlebar") # Use Text.from_markup

    def update_title(self, new_title: str) -> None:
        """Updates the content of the title bar."""
        # You might want to preserve some of the original art or define a new format
        # For example, just updating the main text part:
        # updated_art = f"✨🤖  [b]{new_title}[/b]  📝🚀"
        # self.update(Text.from_markup(updated_art))
        # Or simply replace the whole thing if that's intended:
        self.update(Text.from_markup(new_title)) # Ensure new_title is also markup-compatible or plain

    def reset_title(self) -> None:
        """Resets the title to the default art."""
        self.update(Text.from_markup(self.DEFAULT_ART))