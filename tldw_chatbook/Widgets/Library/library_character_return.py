"""Native, route-owned return action for unavailable Character deep links."""

from collections.abc import Callable

from textual import on
from textual.containers import Horizontal
from textual.widgets import Button


class LibraryCharacterReturn(Horizontal):
    """One compact action; route admission and navigation stay with its owner."""

    def __init__(self, return_to_origin: Callable[[], None]) -> None:
        self._return_to_origin = return_to_origin
        super().__init__(id="library-character-return")
        self.styles.height = 1

    def compose(self):
        button = Button(
            "Back to Console", id="library-character-back-console", compact=True
        )
        button.styles.height = 1
        button.styles.min_height = 1
        button.styles.width = "auto"
        yield button

    @on(Button.Pressed, "#library-character-back-console")
    def return_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._return_to_origin()
