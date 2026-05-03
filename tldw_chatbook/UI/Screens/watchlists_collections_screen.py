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

    def _latest_console_follow_item(self):
        adapter = getattr(self.app_instance, "home_active_work_adapter", None)
        build_dashboard_input = getattr(adapter, "build_dashboard_input", None)
        if not callable(build_dashboard_input):
            return None
        try:
            dashboard_input = build_dashboard_input(
                providers_models={},
                has_recent_work=False,
            )
        except Exception:
            return None
        for item in tuple(getattr(dashboard_input, "active_work_items", ()) or ()):
            if (
                getattr(item, "source", None) == "W+C"
                and bool(getattr(item, "console_available", False))
                and getattr(item, "item_id", None)
            ):
                return item
        return None

    def compose_content(self) -> ComposeResult:
        latest_console_item = self._latest_console_follow_item()
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
                if latest_console_item is not None:
                    yield Static(
                        (
                            "Console can follow latest W+C run: "
                            f"{latest_console_item.title} ({latest_console_item.status})."
                        ),
                        id="watchlists-console-available",
                    )
                    yield Button(
                        f"Follow {latest_console_item.title} in Console",
                        id="watchlists-follow-in-console",
                        tooltip="Open the latest active W+C run in Console.",
                    )
                else:
                    yield Static(
                        "No active W+C run is available for Console follow.",
                        id="watchlists-console-unavailable",
                    )
                    yield Button(
                        "Console follow unavailable",
                        id="watchlists-follow-in-console",
                        disabled=True,
                        tooltip="Unavailable until W+C has an active run with Console context.",
                    )

    @on(Button.Pressed, "#wc-open-watchlists")
    def open_watchlists(self) -> None:
        self.post_message(NavigateToScreen("subscriptions"))

    @on(Button.Pressed, "#watchlists-follow-in-console")
    def follow_latest_watchlist_run_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        latest_console_item = self._latest_console_follow_item()
        if latest_console_item is None:
            self.app_instance.notify(
                "No active W+C run is available for Console follow.",
                severity="warning",
            )
            return
        open_in_console = getattr(self.app_instance, "open_active_home_item_in_console", None)
        if not callable(open_in_console):
            self.app_instance.notify(
                "Console follow is unavailable for W+C in this runtime.",
                severity="warning",
            )
            return
        open_in_console(
            target_id=latest_console_item.item_id,
            target_route="chat",
        )
