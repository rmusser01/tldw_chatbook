"""Overview pane for the watchlists screen."""

from __future__ import annotations

from textual.containers import Grid, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Static

from ...Widgets.recompose_capture_guard import RecomposeCaptureGuard


class OverviewPane(RecomposeCaptureGuard, Vertical):
    """Dashboard cards and recent failed runs for watchlists."""

    data = reactive({}, recompose=True)
    #: TASK-998. How many watchlists exist, so the first-run panel can tell a
    #: user who has already made one that their remaining step is a source.
    #: Screen-seeded like every other reactive on these panes -- the pane has
    #: no service of its own and `data` only counts sources, items and runs.
    watchlist_count = reactive(0, recompose=True)

    _CARD_IDS = {
        "total_sources": "overview-total-sources",
        "active_sources": "overview-active-sources",
        "sources_in_error": "overview-sources-in-error",
        "total_items": "overview-total-items",
        "new_items": "overview-new-items",
        "latest_run_status": "overview-latest-run-status",
        "active_alert_rules": "overview-active-alert-rules",
    }

    def _card_value(self, key: str, label: str) -> str:
        value = self.data.get(key, "-")
        return f"{label}\n{value}"

    #: The three answers this region can give about a profile (TASK-1020).
    #: `LOADING` is the one that was missing: `data` starts as `{}` and is
    #: filled by a worker, and on the server backend that worker is a network
    #: call. Collapsing "not loaded yet" into either resolved answer is wrong
    #: in one direction or the other -- into `EMPTY` it flashes first-run copy
    #: at every user with sources on every visit, into `POPULATED` it delays
    #: the first-run guidance a brand-new user needs for as long as the
    #: request takes.
    LOADING = "loading"
    EMPTY = "empty"
    POPULATED = "populated"

    @staticmethod
    def profile_state(data: dict) -> str:
        """Which of the three states `data` describes.

        The single definition of that question (Qodo #3 on PR #1017, extended
        by TASK-1020). It lived here and, copied, on
        `WatchlistsCollectionsScreen`; two copies of a predicate that decides
        what the Overview region and the Inspector each say is a drift waiting
        to happen, and the two disagreeing is exactly the confusing state
        TASK-998 set out to remove.

        `total_sources` being PRESENT is what separates loading from the two
        resolved answers -- an absent key means "not loaded yet", which is not
        the same answer as "loaded, and empty". `_refresh_overview_data`'s
        failure branch publishes `total_sources: 0`, so a failed or timed-out
        load resolves to `EMPTY` rather than sticking on `LOADING` forever.

        Args:
            data: The overview payload, as published to `overview_data`.

        Returns:
            One of `LOADING`, `EMPTY`, `POPULATED`.
        """
        if "total_sources" not in data:
            return OverviewPane.LOADING
        has_anything = any(
            (
                data.get("total_sources"),
                data.get("total_items"),
                data.get("active_alert_rules"),
                data.get("failed_runs"),
            )
        )
        return OverviewPane.POPULATED if has_anything else OverviewPane.EMPTY

    @staticmethod
    def profile_is_empty(data: dict) -> bool:
        """Whether `data` says this profile has nothing in Watchlists yet.

        Kept as the narrow "is this first run" question now that
        `profile_state` answers the wider one; both resolved answers are
        unchanged, and an unloaded payload is still not empty.

        Args:
            data: The overview payload, as published to `overview_data`.

        Returns:
            True only when the payload has loaded and reports nothing.
        """
        return OverviewPane.profile_state(data) == OverviewPane.EMPTY

    def _first_run_body(self) -> str:
        """What to do next, phrased for what the user has actually done.

        Two variants because the UAT's journey ended exactly between them: the
        user created a watchlist and then had nowhere to go. Telling someone
        who already has one to "create a watchlist" is the same dead end this
        task exists to remove, one step further along.
        """
        if self.watchlist_count:
            return (
                "Your watchlists have no sources yet. Open Sources above and "
                "press New Source to add a feed, or Import OPML to bring a set "
                "of feeds over from another reader.\n\n"
                "Runs, items, rules and notifications fill in once a source "
                "has been checked."
            )
        return (
            "A watchlist is a folder of feeds. Watchlists checks them on a "
            "schedule and collects whatever is new.\n\n"
            "1. Press New in the rail on the left to create a watchlist.\n"
            "2. Open Sources above and press New Source to add a feed to it, "
            "or Import OPML to bring a set of feeds over from another reader."
            "\n\n"
            "Runs, items, rules and notifications fill in once a source has "
            "been checked."
        )

    def compose(self):
        # TASK-998. Seven bordered cards reading "-" and an empty failed-runs
        # table were the largest region on a new user's first screen, and the
        # first thing they saw. Chrome around data that does not exist is not
        # a neutral placeholder: it is the screen's whole first impression
        # spent saying nothing. On an empty profile the cards and the table
        # are replaced -- not merely blanked -- by copy that names the two
        # controls that actually do something (`New` in the rail,
        # `New Source` under Sources). Every populated state is untouched.
        state = self.profile_state(self.data)
        if state == self.LOADING:
            # TASK-1020. Neither the cards nor the first-run copy: both would
            # be a claim about a profile nothing has reported on yet. One line
            # rather than a skeleton, because the local backend resolves in
            # milliseconds and a shimmering placeholder would be the flash
            # this task exists to remove.
            yield Static(
                "Loading watchlist activity...",
                id="overview-loading",
                classes="watchlists-loading-state",
            )
            return

        if state == self.EMPTY:
            with Vertical(id="overview-first-run"):
                yield Static(
                    "Nothing is being watched yet.",
                    id="overview-first-run-title",
                    classes="watchlists-first-run-title",
                )
                yield Static(
                    self._first_run_body(),
                    id="overview-first-run-body",
                    classes="watchlists-first-run-body",
                )
            return

        with Grid(id="watchlists-overview-grid"):
            yield Static(
                self._card_value("total_sources", "Total sources"),
                id=self._CARD_IDS["total_sources"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("active_sources", "Active sources"),
                id=self._CARD_IDS["active_sources"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("sources_in_error", "Sources in error"),
                id=self._CARD_IDS["sources_in_error"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("total_items", "Total items"),
                id=self._CARD_IDS["total_items"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("new_items", "New items"),
                id=self._CARD_IDS["new_items"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("latest_run_status", "Latest run status"),
                id=self._CARD_IDS["latest_run_status"],
                classes="overview-card",
            )
            yield Static(
                self._card_value("active_alert_rules", "Active alert rules"),
                id=self._CARD_IDS["active_alert_rules"],
                classes="overview-card",
            )

        yield Static("Recent failed runs", classes="pane-title")
        table = DataTable(id="overview-failed-runs")
        table.add_columns("Source", "Status", "Error")
        for run in self.data.get("failed_runs", []):
            table.add_row(
                run.get("source_title", ""),
                run.get("status", ""),
                run.get("error_msg", ""),
            )
        yield table
