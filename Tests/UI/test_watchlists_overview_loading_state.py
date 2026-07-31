"""The Overview region has three states, not two — TASK-1020.

`OverviewPane.profile_is_empty` returned `False` while `overview_data` was
missing `total_sources`, so first-run guidance could not appear until the
overview worker landed. On the **server** backend `get_overview_data` is a
network call, so a brand-new user sat looking at the non-first-run UI for as
long as that request took.

Removing the guard is worse, which is why Qodo's finding #4 on PR #1017 was
deliberately not fixed there: `overview_data` starts `{}`, every key reads
falsy, and **every** user — including one with hundreds of sources — would get
a flash of first-run copy on every visit.

The region has three states: *loading*, *empty*, *populated*. While the
request is in flight it shows neither the cards nor the first-run copy, and
the Inspector's own first-run text follows the same predicate so the two
regions cannot disagree.
"""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.full_app_destination_context import (
    FullAppDestinationContext as DestinationHarness,
    active_destination_screen as _active_destination_screen,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
from tldw_chatbook.UI.Watchlists_Modules.overview_pane import OverviewPane


class SlowWatchlistsScopeService:
    """A backend that has not answered yet — what `server` looks like on a
    cold network. Holds every overview query open until `release` is set."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.watch_items: tuple = ()

    async def list_watch_items(self, **kwargs):
        await self.release.wait()
        return list(self.watch_items)


def test_profile_state_names_all_three_states():
    assert OverviewPane.profile_state({}) == "loading", (
        "an empty payload means the worker has not landed, which is not the "
        "same answer as 'loaded, and empty'"
    )
    assert OverviewPane.profile_state({"total_sources": 0}) == "empty"
    assert OverviewPane.profile_state({"total_sources": 3}) == "populated"
    # The old two-state predicate stays exactly as it was for both resolved
    # states, so nothing that asks "is this a first-run profile" changes.
    assert OverviewPane.profile_is_empty({}) is False
    assert OverviewPane.profile_is_empty({"total_sources": 0}) is True
    assert OverviewPane.profile_is_empty({"total_sources": 3}) is False


@pytest.mark.asyncio
async def test_the_overview_shows_a_loading_state_while_the_request_is_in_flight():
    """AC#1 and AC#3, with a deliberately slow backend (AC#5)."""
    app = _build_test_app()
    service = SlowWatchlistsScopeService()
    app.watchlist_scope_service = service

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.3)
        screen = _active_destination_screen(host)
        assert screen.active_section == "overview"
        overview = screen.query_one("#watchlists-overview-pane", OverviewPane)

        assert overview.query("#overview-loading"), (
            "while the overview request is in flight the region must say so"
        )
        assert not overview.query(".overview-card"), (
            "the cards must not be rendered over data that has not arrived"
        )
        assert not overview.query("#overview-first-run"), (
            "first-run copy must not be claimed before the load resolves"
        )

        service.release.set()
        overview = await _wait_for_overview_first_run(screen, pilot)

        assert overview.query("#overview-first-run"), (
            "a brand-new user must get the guidance as soon as the load lands"
        )
        assert not overview.query("#overview-loading")


async def _wait_for_overview_first_run(screen, pilot) -> OverviewPane:
    for _ in range(120):
        await pilot.pause()
        overview = screen.query_one("#watchlists-overview-pane", OverviewPane)
        if overview.query("#overview-first-run"):
            return overview
    return screen.query_one("#watchlists-overview-pane", OverviewPane)


@pytest.mark.asyncio
async def test_the_inspector_follows_the_same_three_states():
    """AC#4: the two regions answer one question, so they cannot disagree."""
    app = _build_test_app()
    service = SlowWatchlistsScopeService()
    app.watchlist_scope_service = service

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.3)
        screen = _active_destination_screen(host)
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)

        empty_state = inspector.query_one("#inspector-empty-state")
        text = str(empty_state.renderable)
        assert "Select a source" not in text, (
            "before the load resolves the Inspector must not name five things "
            f"that may not exist; it said {text!r}"
        )
        assert not inspector.query("#inspector-first-run-hint"), (
            "...and must not claim first-run either"
        )

        service.release.set()
        for _ in range(120):
            await pilot.pause()
            inspector = screen.query_one(
                "#watchlists-entity-inspector", InspectorPane
            )
            if inspector.query("#inspector-first-run-hint"):
                break

        assert inspector.query("#inspector-first-run-hint"), (
            "once the load resolves empty, the Inspector shows first-run text"
        )


@pytest.mark.asyncio
async def test_a_user_with_sources_never_flashes_first_run_copy():
    """AC#2, the regression the guard was protecting: sample every frame
    from mount until the data lands and assert first-run copy never appears."""
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    db.add_subscription(
        name="Summit Route", type="rss", source="https://summitroute.com/blog/feed.xml"
    )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        seen_first_run = False
        seen_cards = False
        for _ in range(120):
            await pilot.pause()
            try:
                overview = screen.query_one("#watchlists-overview-pane", OverviewPane)
            except Exception:
                continue
            if overview.query("#overview-first-run"):
                seen_first_run = True
            if overview.query(".overview-card"):
                seen_cards = True
                break

        assert seen_cards, "the populated cards must eventually appear"
        assert not seen_first_run, (
            "a profile with a source must never show first-run copy, not even "
            "for one frame"
        )
