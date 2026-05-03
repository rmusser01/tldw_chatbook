"""W+C destination shell."""

from textual.app import ComposeResult
from textual import on
from textual.containers import Vertical
from textual.widgets import Button, Static

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen


class WatchlistsCollectionsScreen(BaseAppScreen):
    """Monitored sources and curated reading/content collections."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "watchlists_collections", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="watchlists-collections-shell"):
            yield Static(
                "W+C",
                id="watchlists-collections-title",
                classes="ds-destination-header",
            )
            yield Static(
                "Monitored sources and curated reading/content collections.",
                id="watchlists-collections-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="watchlists-collections-sections", classes="ds-panel"):
                yield Static("Watchlists", classes="destination-section")
                yield Static(
                    "Monitored sources, filters, jobs, runs, outputs, templates, alerts, telemetry, retry/backoff."
                )
                yield Static("Collections", classes="destination-section")
                yield Static(
                    "Reading/content items, highlights, saved searches, archive state, note links, templates, feeds, import/export."
                )
                yield Button(
                    "Open current Watchlists",
                    id="wc-open-watchlists",
                    tooltip="Open the current watchlist/subscription surface.",
                )
                yield Static(
                    "Console follow is unavailable until watchlist and collection live-work payloads are wired.",
                    id="watchlists-console-unavailable",
                )
                yield Button(
                    "Console follow unavailable",
                    id="watchlists-follow-in-console",
                    disabled=True,
                    tooltip="Unavailable until W+C can pass actionable live-work context to Console.",
                )

    @on(Button.Pressed, "#wc-open-watchlists")
    def open_watchlists(self) -> None:
        self.post_message(NavigateToScreen("subscriptions"))
