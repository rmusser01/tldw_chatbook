"""An entirely-skipped Check Now says so instead of "0 found, 0 new" — task-16838.

The per-(subscription, url) in-flight guard
(`LocalWatchlistsService._check_url_guarded`) makes a manual Check Now that
lands while a scheduled check of the same source is mid-flight complete as a
`skipped` run rather than double-checking. Before this test's fix, that run's
toast read "Check complete: X — 0 found, 0 new." — which tells the user their
page was checked and unchanged when it was not checked at all, the same
false-clean shape TASK-1090 removed for failures.

The run shape consumed here — dispositions all zero except `skipped` — is
exactly what the guard produces, pinned at the service level in
`Tests/Subscriptions/test_watchlist_check_in_flight_guard.py`
(`manual_run["stats"]["dispositions"] == {..., "skipped": 1}`); this test
closes the chain by pinning what the screen tells the user about it.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.full_app_destination_context import (
    FullAppDestinationContext as DestinationHarness,
)
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane


class Notified:
    """Capture what the app told the user, since the toast itself is transient."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, message, *args, severity: str = "information", **kwargs) -> None:
        self.calls.append((str(message), severity))

    @property
    def warnings(self) -> list[str]:
        return [message for message, severity in self.calls if severity == "warning"]


#: The zero-filled counter shape `_disposition_counts` writes, with one skip —
#: the run the guard's losing entrant records.
_ALL_SKIPPED_STATS = {
    "dispositions": {
        "changed": 0,
        "unchanged": 0,
        "withheld": 0,
        "baseline": 0,
        "rebaselined": 0,
        "error": 0,
        "skipped": 1,
    }
}


async def _open_sources(pilot, host):
    screen = host.screen_stack[-1]
    screen.active_section = "sources"
    await pilot.pause(0.3)
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    for _ in range(40):
        await pilot.pause()
        if pane.sources:
            break
    return screen, pane


@pytest.mark.asyncio
async def test_an_entirely_skipped_check_now_reports_the_skip_not_a_clean_zero():
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    db.add_subscription(
        name="Summit Route", type="url", source="https://summitroute.com/blog"
    )
    notified = Notified()
    app.notify = notified

    async def already_checking(subscription):
        # What `_default_run_executor` returns when the in-flight guard
        # skipped the source's only URL: no items, one skipped disposition.
        return {"items": [], "stats": dict(_ALL_SKIPPED_STATS)}

    app.local_watchlists_service.run_executor = already_checking

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen, pane = await _open_sources(pilot, host)
        assert pane.sources, "the seeded source must reach the Sources pane"

        pane.select_source_by_id(str(pane.sources[0]["id"]))
        await pilot.pause(0.2)
        pane.query_one("#sources-check-now-button", Button).press()
        for _ in range(60):
            await pilot.pause()
            if notified.warnings:
                break

        skip_warnings = [
            message for message in notified.warnings if "Check skipped" in message
        ]
        assert skip_warnings, (
            "an entirely-skipped check must be reported as skipped; "
            f"got: {notified.calls!r}"
        )
        assert "already running" in skip_warnings[0], (
            "the report must say WHY it skipped -- a concurrent check owns "
            "the result the user is about to get"
        )
        assert not any(
            "Check complete" in message for message, _severity in notified.calls
        ), (
            "'Check complete — 0 found, 0 new' over a run that checked "
            "nothing is the false-clean reading this toast exists to remove"
        )
