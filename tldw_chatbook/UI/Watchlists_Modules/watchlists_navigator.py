from __future__ import annotations

from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button


class SectionSelected(Message):
    """Posted when the user selects a section in the watchlists navigator."""

    def __init__(self, section_id: str) -> None:
        self.section_id = section_id
        super().__init__()


class WatchlistsNavigator(Vertical):
    """Left-rail section list for the watchlists screen."""

    SECTIONS = [
        ("overview", "Overview"),
        ("sources", "Sources"),
        ("items", "Items"),
        ("runs", "Runs"),
        ("rules", "Rules"),
    ]

    def compose(self):
        for section_id, label in self.SECTIONS:
            yield Button(label, id=f"nav-{section_id}", classes="watchlists-nav-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        section_id = str(event.button.id).replace("nav-", "")
        self.post_message(SectionSelected(section_id))
