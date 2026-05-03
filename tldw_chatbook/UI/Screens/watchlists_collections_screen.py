"""Watchlists+Collections destination shell."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


class WatchlistsCollectionsScreen(BaseAppScreen):
    """Monitored sources and curated reading/content collections."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "watchlists_collections", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="watchlists-collections-shell"):
            yield Static(
                "Watchlists+Collections",
                id="watchlists-collections-title",
                classes="ds-destination-header",
            )
            yield Static(
                "Monitored sources and curated reading/content collections.",
                id="watchlists-collections-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="watchlists-collections-sections", classes="ds-panel"):
                yield Static("Watchlists | Collections | Feed status | Open in Console")
