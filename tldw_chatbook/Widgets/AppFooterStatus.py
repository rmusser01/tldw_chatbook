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
from ..UI.Navigation.shortcut_context import ShortcutAction, ShortcutContext
#
########################################################################################################################
#
# AppFooterStatus

class AppFooterStatus(Widget):
    DEFAULT_SHORTCUT_TEXT = "Ctrl+Q quit | Ctrl+P palette"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._shortcut_text = self.DEFAULT_SHORTCUT_TEXT
        #: Source of the active shortcut context (e.g. "personas"); ``None``
        #: when the default shortcuts are shown.
        self._shortcut_source: str | None = None
        self._shortcut_display = Static(self._shortcut_text, id="footer-key-quit")
        self._word_count_display: Static = Static("", id="footer-word-count")
        self._token_count_display: Static = Static("Tokens: -- | ", id="footer-token-count")
        self._db_status_display: Static = Static("", id="internal-db-size-indicator")

    def compose(self) -> ComposeResult:
        yield self._shortcut_display
        yield Static(id="footer-spacer") # This will push items to the right
        yield self._word_count_display # Word count display
        yield self._token_count_display # Token count display
        yield self._db_status_display # This is the existing DB size display

    @property
    def shortcut_text(self) -> str:
        return self._shortcut_text

    def _set_shortcut_text(self, text: str) -> None:
        self._shortcut_text = text
        self._shortcut_display.update(text)

    def set_shortcut_context(self, context: ShortcutContext) -> None:
        text = context.render() or self.DEFAULT_SHORTCUT_TEXT
        self._shortcut_source = context.source
        self._set_shortcut_text(text)

    def set_workbench_shortcuts(
        self,
        *,
        source: str,
        shortcuts: tuple[tuple[str, str], ...],
    ) -> None:
        """Render Workbench shortcut hints through the footer context model."""
        context = ShortcutContext(
            source=source,
            actions=tuple(ShortcutAction(key, label) for key, label in shortcuts),
        )
        self.set_shortcut_context(context)

    def clear_shortcut_context(self, source: str | None = None) -> None:
        """Reset the footer to the default shortcuts.

        Textual's ``switch_screen`` mounts the incoming screen before
        unmounting the outgoing one, so an unmount-time clear can race a
        just-registered context. Passing ``source`` makes the clear a no-op
        unless that source still owns the context; calling with no argument
        clears unconditionally (backward compatible).
        """
        if source is not None and source != self._shortcut_source:
            return
        self._shortcut_source = None
        self._set_shortcut_text(self.DEFAULT_SHORTCUT_TEXT)

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
    
    def update_token_count(self, display_text: str) -> None:
        """Update the token count display in the footer."""
        try:
            if display_text:
                self._token_count_display.update(f"{display_text} | ")
            else:
                self._token_count_display.update("")
        except Exception as e:
            print(f"Error updating token count display: {e}")

#
# End of AppFooterStatus.py
########################################################################################################################
