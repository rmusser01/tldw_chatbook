# tldw_chatbook/Widgets/AppFooterStatus.py
#
# Imports
#
# 3rd-party Libraries
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static
#
# Local Imports
#
########################################################################################################################
#
# AppFooterStatus

class AppFooterStatus(Widget):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._key_quit = Static("Ctrl+Q (quit) / Ctrl+P or Cmd+P (palette menu)", id="footer-key-quit")
        self._word_count_display: Static = Static("", id="footer-word-count")
        self._db_status_display: Static = Static("", id="internal-db-size-indicator")

    def compose(self) -> ComposeResult:
        yield self._key_quit
        yield Static(id="footer-spacer") # This will push items to the right
        yield self._word_count_display # Word count display
        yield self._db_status_display # This is the existing DB size display

    def update_db_sizes_display(self, status_string: str) -> None:
        try:
            self._db_status_display.update(status_string)
        except Exception as e:
            # If the app is shutting down, the widget might be gone
            # In a real scenario, you'd use self.log from the widget
            print(f"Error updating AppFooterStatus display: {e}")
    
    def update_word_count(self, word_count: int) -> None:
        """Update the word count display in the footer."""
        try:
            if word_count > 0:
                self._word_count_display.update(f"Words: {word_count} | ")
            else:
                self._word_count_display.update("")
        except Exception as e:
            print(f"Error updating word count display: {e}")

#
# End of AppFooterStatus.py
########################################################################################################################
