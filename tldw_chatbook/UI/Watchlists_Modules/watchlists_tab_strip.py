"""One-row section tab strip for the Watchlists centre.

Replaces the left-rail section navigator: the rail now holds the watchlist
tree, which is what the user actually navigates by. The message type and
shape are unchanged from the navigator so the screen's existing handler
keeps working.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button


SECTIONS: tuple[tuple[str, str], ...] = (
    ("overview", "Overview"),
    ("sources", "Sources"),
    ("items", "Items"),
    ("runs", "Runs"),
    ("rules", "Rules"),
    ("notifications", "Notifications"),
)


class SectionSelected(Message):
    """Posted when the user selects a section."""

    def __init__(self, section_id: str) -> None:
        self.section_id = section_id
        super().__init__()


class WatchlistsTabStrip(Horizontal):
    """Compact one-row tab strip across the top of the centre stack."""

    def __init__(self, active_section: str = "overview", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_class("watchlists-tab-strip")
        self.active_section = active_section
        self.styles.height = 1
        self.styles.min_height = 1

    def compose(self) -> ComposeResult:
        for section_id, label in SECTIONS:
            tab = Button(
                label,
                id=f"wl-tab-{section_id}",
                compact=True,
                tooltip=f"Open the {label} section",
            )
            tab.add_class("watchlists-tab")
            if section_id == self.active_section:
                tab.add_class("is-active")
            yield tab

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        prefix = "wl-tab-"
        if not button_id.startswith(prefix):
            return
        event.stop()
        self.post_message(SectionSelected(button_id[len(prefix):]))
