"""Production-shaped state tests for stale-while-refreshing Artifacts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Button, Collapsible, DataTable, Static
from textual.worker import NoActiveWorker, WorkerCancelled, get_current_worker

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.UI.Screens import watchlists_collections_screen as screen_module
from tldw_chatbook.UI.Watchlists_Modules.artifacts_pane import (
    ArtifactsPane,
    BriefingSelected,
    ExportBriefingRequested,
    ExportFeedRequested,
    InspectArtifactRecoveryRequested,
    KeepBriefingRequested,
    KeptBriefingsRequested,
    RefreshBriefingsRequested,
    ScriptSelected,
    ServeFeedRequested,
    StopFeedServerRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope

pytestmark = pytest.mark.ui


COMPLETE_BRIEFING = {
    "id": 41,
    "status": "complete",
    "body_markdown": "## Last good briefing\n\nDurable body [item 7].",
    "created_at": "2026-08-28T08:00:00+00:00",
    "window_start": "2026-08-27T08:00:00+00:00",
    "window_end": "2026-08-28T08:00:00+00:00",
    "item_count": 1,
    "featured_count": 0,
    "overflow_count": 0,
}


class ArtifactsStateHarness(ConsolidatedCSSApp):
    CSS_PATH = [str(sheet) for sheet in APP_STYLESHEETS]

    def __init__(self, *, populated: bool = True) -> None:
        super().__init__()
        self.populated = populated
        self.messages: list[tuple[str, object]] = []

    def compose(self) -> ComposeResult:
        pane = ArtifactsPane()
        pane.set_reactive(ArtifactsPane.can_generate, True)
        pane.set_reactive(ArtifactsPane.scope_label, "Briefings for Threat Intel")
        pane.set_reactive(ArtifactsPane.briefing_cadence_seconds, 86_400)
        if self.populated:
            pane.set_reactive(ArtifactsPane.briefings, [dict(COMPLETE_BRIEFING)])
            pane.set_reactive(ArtifactsPane.selected_briefing, dict(COMPLETE_BRIEFING))
            pane.set_reactive(
                ArtifactsPane.citations,
                [{"item_id": 7, "label": Text("[item 7] Source"), "available": True}],
            )
        yield pane

    def on_refresh_briefings_requested(self, _event: RefreshBriefingsRequested) -> None:
        self.messages.append(("retry", None))

    def on_inspect_artifact_recovery_requested(
        self, event: InspectArtifactRecoveryRequested
    ) -> None:
        self.messages.append(("inspect", event.destination))

    def on_export_briefing_requested(self, _event: ExportBriefingRequested) -> None:
        self.messages.append(("export", None))

    def on_keep_briefing_requested(self, _event: KeepBriefingRequested) -> None:
        self.messages.append(("keep", None))

    def on_kept_briefings_requested(self, _event: KeptBriefingsRequested) -> None:
        self.messages.append(("kept", None))

    def on_export_feed_requested(self, _event: ExportFeedRequested) -> None:
        self.messages.append(("export-feed", None))

    def on_serve_feed_requested(self, _event: ServeFeedRequested) -> None:
        self.messages.append(("serve", None))

    def on_stop_feed_server_requested(self, _event: StopFeedServerRequested) -> None:
        self.messages.append(("stop-serving", None))


class ArtifactsScreenHarness(DestinationHarness):
    CSS_PATH = [str(sheet) for sheet in APP_STYLESHEETS]


class _EventuallyVisibleBriefingsDB:
    """Delegate all DB work while hiding one accepted receipt until retry."""

    def __init__(self, delegate, hidden_id: int) -> None:
        self._delegate = delegate
        self.hidden_id = hidden_id
        self.visible = False

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def get_briefing(self, briefing_id: int):
        if not self.visible and briefing_id == self.hidden_id:
            return None
        return self._delegate.get_briefing(briefing_id)

    def list_briefings(self, watchlist_id: int):
        if self.visible:
            return self._delegate.list_briefings(watchlist_id)
        return [
            row
            for row in self._delegate.list_briefings(watchlist_id)
            if row["id"] != self.hidden_id
        ]


async def _settle_briefings_load(host, pilot) -> None:
    """Wait out the Artifacts loader group; a newer load may cancel an older one."""
    for worker in [w for w in host.workers if w.group == "wl-briefings-load"]:
        try:
            await worker.wait()
        except WorkerCancelled:
            pass
    await pilot.pause()


#: Pane messages whose screen handler stops them and queues an Artifacts load.
_LOAD_REQUESTS = (RefreshBriefingsRequested, BriefingSelected, ScriptSelected)


def _track_load_requests(pane) -> list:
    """Record each load-requesting message the pane posts (TASK-39058)."""
    posted: list = []
    post = pane.post_message

    def record(message):
        if isinstance(message, _LOAD_REQUESTS):
            posted.append(message)
        return post(message)

    pane.post_message = record
    return posted


def _track_every_pane_load_request(monkeypatch) -> list:
    """Record load-requesting messages from every ArtifactsPane, from mount on.

    For tests whose setup moves the selection itself: the load that setup
    queues must be recorded too, and the pane does not exist before the run.
    """
    posted: list = []
    post = ArtifactsPane.post_message

    def record(self, message):
        if isinstance(message, _LOAD_REQUESTS):
            posted.append(message)
        return post(self, message)

    monkeypatch.setattr(ArtifactsPane, "post_message", record)
    return posted


async def _drain_briefings_loads(host, pilot, posted, *, after=None) -> None:
    """Wait for every Artifacts load the tracked messages queued (TASK-39058).

    A load that moves the selection posts ``BriefingSelected``, and the
    screen's handler stops it and queues one more load. So this waits until
    each tracked message was handled (``stop()`` is the handler's first line)
    and no loader worker is left. ``after`` names a message that must have
    been posted first, such as the one a button press sends.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 30  # a backstop for a hang, not a settle
    while True:
        await _settle_briefings_load(host, pilot)
        loading = any(
            worker.group == "wl-briefings-load" and not worker.is_finished
            for worker in host.workers
        )
        if (
            not loading
            and (after is None or any(isinstance(m, after) for m in posted))
            and all(message._stop_propagation for message in posted)
        ):
            return
        assert loop.time() < deadline, "Artifacts loads did not settle"


def _seed_complete_briefing(app) -> tuple[int, int]:
    watchlist_id = app.watchlist_bundle_service.create("Threat Intel")["id"]
    db = app.watchlist_bundle_service.db
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(
        briefing_id,
        status="complete",
        body_markdown="## Last good briefing\n\nDurable body [item 7].",
        item_count=1,
    )
    return watchlist_id, briefing_id


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 42), (220, 52)])
async def test_refresh_and_failure_keep_last_good_artifact_content_mounted(size):
    """Refresh/failure overlays must not replace table, body, or citations."""
    app = ArtifactsStateHarness()
    async with app.run_test(size=size) as pilot:
        pane = app.query_one(ArtifactsPane)
        table = pane.query_one("#artifacts-table", DataTable)
        detail = pane.query_one("#artifacts-detail", Static)
        citations = pane.query_one("#artifacts-citations-table", DataTable)

        pane.set_view_state("refreshing", "Refreshing briefings…")
        await pilot.pause()
        assert pane.query_one("#artifacts-table", DataTable) is table
        assert pane.query_one("#artifacts-detail", Static) is detail
        assert pane.query_one("#artifacts-citations-table", DataTable) is citations
        assert pane.selected_briefing["id"] == 41
        assert citations.row_count == 1
        assert "Refreshing briefings" in str(
            pane.query_one("#artifacts-state-message", Static).render()
        )

        pane.set_view_state(
            "failed",
            "Briefings could not be refreshed. Last good content is still shown.",
        )
        await pilot.pause()
        assert pane.query_one("#artifacts-table", DataTable) is table
        assert pane.selected_briefing["body_markdown"].startswith("## Last good")
        retry = pane.query_one("#artifacts-retry-button", Button)
        assert retry.display is True
        retry.press()
        await pilot.pause()
        assert app.messages == [("retry", None)]


@pytest.mark.asyncio
async def test_storage_mismatch_is_distinct_and_offers_retry_and_runs_target():
    app = ArtifactsStateHarness()
    async with app.run_test(size=(160, 42)) as pilot:
        pane = app.query_one(ArtifactsPane)
        pane.set_view_state(
            "storage_mismatch",
            "Briefing saved, but this view could not reload it.",
        )
        await pilot.pause()

        state = str(pane.query_one("#artifacts-state-message", Static).render())
        assert "Briefing saved, but this view could not reload it" in state
        assert pane.query_one("#artifacts-retry-button", Button).display is True
        inspect = pane.query_one("#artifacts-inspect-runs-button", Button)
        assert inspect.display is True
        inspect.press()
        await pilot.pause()
        assert app.messages == [("inspect", "runs")]


@pytest.mark.asyncio
async def test_empty_artifacts_foreground_generate_every_24_hours_and_more():
    app = ArtifactsStateHarness(populated=False)
    async with app.run_test(size=(160, 42)):
        pane = app.query_one(ArtifactsPane)
        generate = pane.query_one("#artifacts-generate-button", Button)
        cadence = pane.query_one("#artifacts-cadence-select")
        more = pane.query_one("#artifacts-more-actions", Collapsible)

        assert str(generate.label) == "Generate briefing"
        assert cadence.value == 86_400
        assert more.title == "More briefing actions"
        assert more.collapsed is True
        assert generate.region.right <= pane.region.right
        assert cadence.region.right <= pane.region.right


@pytest.mark.asyncio
async def test_downstream_disclosure_actions_survive_live_state_updates():
    """Re-arming an action must not replace or re-collapse its disclosure."""
    app = ArtifactsStateHarness()
    async with app.run_test(size=(220, 52)) as pilot:
        pane = app.query_one(ArtifactsPane)
        pane.chachanotes_available = True
        pane.has_audio_episodes = True
        await pilot.pause()
        more = pane.query_one("#artifacts-more-actions", Collapsible)
        more.collapsed = False
        await pilot.pause()
        buttons = {
            selector: pane.query_one(selector, Button)
            for selector in (
                "#artifacts-export-button",
                "#artifacts-keep-button",
                "#artifacts-kept-briefings-button",
                "#artifacts-export-feed-button",
                "#artifacts-serve-feed-button",
            )
        }
        assert not pane.query("#artifacts-stop-feed-button")

        pane.can_serve_feed = True
        await pilot.pause()
        assert pane.query_one("#artifacts-more-actions", Collapsible) is more
        assert more.collapsed is False
        for selector, button in buttons.items():
            assert pane.query_one(selector, Button) is button

        for selector in (
            "#artifacts-export-button",
            "#artifacts-keep-button",
            "#artifacts-kept-briefings-button",
            "#artifacts-export-feed-button",
            "#artifacts-serve-feed-button",
        ):
            buttons[selector].press()
        await pilot.pause()
        assert app.messages == [
            ("export", None),
            ("keep", None),
            ("kept", None),
            ("export-feed", None),
            ("serve", None),
        ]

        pane.feed_server_running = True
        pane.feed_server_url = "http://127.0.0.1:8000/feed.xml"
        await pilot.pause()
        assert pane.query_one("#artifacts-more-actions", Collapsible) is more
        assert more.collapsed is False
        assert buttons["#artifacts-serve-feed-button"].disabled is True
        stop = pane.query_one("#artifacts-stop-feed-button", Button)
        stop.press()
        await pilot.pause()
        assert app.messages[-1] == ("stop-serving", None)

        pane.feed_server_running = False
        await pilot.pause()
        assert not pane.query("#artifacts-stop-feed-button")
        for selector, button in buttons.items():
            assert pane.query_one(selector, Button) is button

        pane.automation_receipt = "Every 24 hours · queue reload acknowledged"
        await pilot.pause()
        recomposed_more = pane.query_one("#artifacts-more-actions", Collapsible)
        assert recomposed_more is not more
        recomposed_more.collapsed = False
        await pilot.pause()
        recomposed_buttons = [
            pane.query_one(selector, Button)
            for selector in (
                "#artifacts-export-button",
                "#artifacts-keep-button",
                "#artifacts-kept-briefings-button",
                "#artifacts-export-feed-button",
                "#artifacts-serve-feed-button",
            )
        ]
        assert all(not button.disabled for button in recomposed_buttons)
        for button in recomposed_buttons:
            button.press()
        await pilot.pause()
        assert app.messages[-5:] == [
            ("export", None),
            ("keep", None),
            ("kept", None),
            ("export-feed", None),
            ("serve", None),
        ]


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_refresh_failure_preserves_screen_owned_last_good_content(monkeypatch):
    """A failed database refresh must not turn the pane into an empty state."""
    app_instance = _build_test_app()
    watchlist_id, briefing_id = _seed_complete_briefing(app_instance)
    host = ArtifactsScreenHarness(app_instance, "watchlists_collections")
    posted = _track_every_pane_load_request(monkeypatch)

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        screen.active_section = "artifacts"
        await pilot.pause()
        await host.workers.wait_for_complete()
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        selected = next(
            row for row in screen._loaded_briefings if row["id"] == briefing_id
        )
        screen._selected_briefing = selected
        screen._loaded_citations = [
            {"item_id": 7, "label": Text("[item 7] Source"), "available": True}
        ]
        screen._apply_briefing_state_to_pane()
        # TASK-39058: the selection move queues a load; let it finish first.
        await _drain_briefings_loads(host, pilot, posted)
        table = pane.query_one("#artifacts-table", DataTable)

        def fail_refresh(_watchlist_id):
            raise RuntimeError("private storage detail")

        monkeypatch.setattr(
            app_instance.watchlist_bundle_service.db,
            "list_briefings",
            fail_refresh,
        )
        await screen._load_briefings()
        await _drain_briefings_loads(host, pilot, posted)

        assert pane.query_one("#artifacts-table", DataTable) is table
        assert table.row_count == 1
        assert pane.selected_briefing["id"] == briefing_id
        assert pane.citations[0]["item_id"] == 7
        assert pane.view_state == "failed"
        assert "private storage detail" not in str(
            pane.query_one("#artifacts-state-message", Static).render()
        )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_generation_failure_preserves_last_good_content_and_has_own_state(
    monkeypatch,
):
    """Generation failure is distinct from a list-refresh failure."""
    app_instance = _build_test_app()
    watchlist_id, briefing_id = _seed_complete_briefing(app_instance)
    host = ArtifactsScreenHarness(app_instance, "watchlists_collections")
    posted = _track_every_pane_load_request(monkeypatch)

    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        screen.active_section = "artifacts"
        await pilot.pause()
        await host.workers.wait_for_complete()
        screen._selected_briefing = next(
            row for row in screen._loaded_briefings if row["id"] == briefing_id
        )
        screen._apply_briefing_state_to_pane()
        # TASK-39058: the selection move queues a load; let it finish first.
        await _drain_briefings_loads(host, pilot, posted)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-table", DataTable)

        async def fail_generation(*_args, **_kwargs):
            raise RuntimeError("provider internals")

        monkeypatch.setattr(screen_module, "generate_briefing", fail_generation)
        await screen._generate_briefing(
            app_instance.watchlist_bundle_service.db,
            watchlist_id,
            None,
        )
        # The failure repaints through a queued load (generation_failed=True).
        await _drain_briefings_loads(host, pilot, posted)

        assert pane.query_one("#artifacts-table", DataTable) is table
        assert pane.selected_briefing["id"] == briefing_id
        assert pane.view_state == "failed"
        assert (
            "generation failed"
            in str(pane.query_one("#artifacts-state-message", Static).render()).lower()
        )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_failed_generation_receipt_does_not_replace_readable_selection(
    monkeypatch,
):
    """A durable failed row belongs in history, not in the reading surface."""
    app_instance = _build_test_app()
    watchlist_id, briefing_id = _seed_complete_briefing(app_instance)
    host = ArtifactsScreenHarness(app_instance, "watchlists_collections")
    posted = _track_every_pane_load_request(monkeypatch)

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        screen.active_section = "artifacts"
        await pilot.pause()
        await host.workers.wait_for_complete()
        screen._selected_briefing = next(
            row for row in screen._loaded_briefings if row["id"] == briefing_id
        )
        screen._loaded_citations = [
            {"item_id": 7, "label": Text("[item 7] Source"), "available": True}
        ]
        screen._apply_briefing_state_to_pane()
        # TASK-39058: the selection move queues a load; let it finish first.
        await _drain_briefings_loads(host, pilot, posted)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)

        async def return_failed_receipt(db, selected_watchlist_id, **_kwargs):
            failed_id = db.insert_briefing(selected_watchlist_id, status="failed")
            return db.get_briefing(failed_id)

        monkeypatch.setattr(
            screen_module,
            "generate_briefing",
            return_failed_receipt,
        )
        await screen._generate_briefing(
            app_instance.watchlist_bundle_service.db,
            watchlist_id,
            None,
        )
        await _drain_briefings_loads(host, pilot, posted)

        assert len(pane.briefings) == 2
        assert pane.selected_briefing["id"] == briefing_id
        assert pane.selected_briefing["body_markdown"].startswith("## Last good")
        assert pane.citations[0]["item_id"] == 7
        assert pane.view_state == "failed"


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_durable_receipt_mismatch_retains_last_good_content(monkeypatch):
    """Only a durable expected ID missing from a successful reload is mismatch."""
    app_instance = _build_test_app()
    watchlist_id, briefing_id = _seed_complete_briefing(app_instance)
    host = ArtifactsScreenHarness(app_instance, "watchlists_collections")
    posted = _track_every_pane_load_request(monkeypatch)

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        screen.active_section = "artifacts"
        await pilot.pause()
        await host.workers.wait_for_complete()
        screen._selected_briefing = next(
            row for row in screen._loaded_briefings if row["id"] == briefing_id
        )
        screen._apply_briefing_state_to_pane()
        # TASK-39058: the selection move queues a load; let it finish first.
        await _drain_briefings_loads(host, pilot, posted)
        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        table = pane.query_one("#artifacts-table", DataTable)

        await screen._load_briefings(
            select_briefing_id=briefing_id + 10_000,
            expect_durable_receipt=True,
        )
        await _drain_briefings_loads(host, pilot, posted)

        assert pane.query_one("#artifacts-table", DataTable) is table
        assert pane.selected_briefing["id"] == briefing_id
        assert pane.view_state == "storage_mismatch"
        assert pane.query_one("#artifacts-retry-button", Button).display is True
        assert pane.query_one("#artifacts-inspect-runs-button", Button).display is True


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 42), (220, 52)])
@pytest.mark.parametrize("coordinated", [False, True])
async def test_accepted_missing_receipt_retries_its_exact_id(
    monkeypatch,
    size,
    coordinated,
):
    """Accepted local/coordinated IDs stay the mismatch target until visible."""
    app_instance = _build_test_app()
    watchlist_id, last_good_id = _seed_complete_briefing(app_instance)
    db = app_instance.watchlist_bundle_service.db
    accepted_id = db.insert_briefing(watchlist_id)
    db.update_briefing(
        accepted_id,
        status="complete",
        body_markdown="## Eventually visible briefing",
    )
    hidden_db = _EventuallyVisibleBriefingsDB(db, accepted_id)
    host = ArtifactsScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=size) as pilot:
        screen = host.screen_stack[-1]
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        screen.active_section = "artifacts"
        await pilot.pause()
        await host.workers.wait_for_complete()
        screen._selected_briefing = next(
            row for row in screen._loaded_briefings if row["id"] == last_good_id
        )
        screen._apply_briefing_state_to_pane()
        await pilot.pause()
        screen._briefings_db = lambda: hidden_db
        notifications: list[str] = []
        screen._notify_watchlists = lambda message, *_args, **_kwargs: (
            notifications.append(message)
        )

        if coordinated:

            class Coordinator:
                async def accept_briefing(self, _watchlist_id, _preset_id):
                    return {"id": accepted_id}

            await screen._follow_coordinated_briefing(
                hidden_db,
                watchlist_id,
                None,
                Coordinator(),
            )
        else:

            async def accepted_local_receipt(_db, _watchlist_id, **_kwargs):
                return db.get_briefing(accepted_id)

            monkeypatch.setattr(
                screen_module,
                "generate_briefing",
                accepted_local_receipt,
            )
            await screen._generate_briefing(hidden_db, watchlist_id, None)
        # Both paths repaint through the loader group (TASK-38990).
        await _settle_briefings_load(host, pilot)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.view_state == "storage_mismatch"
        assert pane.selected_briefing["id"] == last_good_id
        assert pane.query_one("#artifacts-retry-button", Button).display is True
        assert pane.query_one("#artifacts-inspect-runs-button", Button).display is True
        assert all(
            "Nothing new was started" not in message for message in notifications
        )

        hidden_db.visible = True
        posted = _track_load_requests(pane)
        pane.query_one("#artifacts-retry-button", Button).press()
        await _drain_briefings_loads(
            host, pilot, posted, after=RefreshBriefingsRequested
        )

        assert pane.view_state == "idle"
        assert pane.selected_briefing["id"] == accepted_id


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_coordinated_briefing_follow_releases_ui_when_receipt_stays_generating(
    monkeypatch,
):
    """The coordinator owns long work; the screen follower remains bounded."""
    app_instance = _build_test_app()
    watchlist_id, _briefing_id = _seed_complete_briefing(app_instance)
    accepted_id = 912
    host = ArtifactsScreenHarness(app_instance, "watchlists_collections")

    class NeverTerminalDB:
        @staticmethod
        def get_briefing(briefing_id):
            return {"id": briefing_id, "status": "generating"}

    class Coordinator:
        @staticmethod
        async def accept_briefing(_watchlist_id, _preset_id):
            return {"id": accepted_id}

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        screen.active_section = "artifacts"
        await pilot.pause()
        await host.workers.wait_for_complete()
        reload_groups: list[str | None] = []

        async def record_reload_group(**_kwargs):
            try:
                reload_groups.append(get_current_worker().group)
            except NoActiveWorker:
                reload_groups.append(None)

        reload_briefings = AsyncMock(side_effect=record_reload_group)
        screen._load_briefings = reload_briefings
        screen._briefing_in_flight = True
        screen._briefing_in_flight_watchlist_id = watchlist_id
        monkeypatch.setattr(
            screen_module,
            "_COORDINATED_BRIEFING_FOLLOW_TIMEOUT_SECONDS",
            0.01,
            raising=False,
        )

        await asyncio.wait_for(
            screen._follow_coordinated_briefing(
                NeverTerminalDB(),
                watchlist_id,
                None,
                Coordinator(),
            ),
            timeout=0.25,
        )

        assert screen._briefing_in_flight is False
        assert screen._briefing_in_flight_watchlist_id is None
        assert screen._durable_briefing_reload_target == (watchlist_id, accepted_id)
        await _settle_briefings_load(host, pilot)
        reload_briefings.assert_awaited_once_with(
            select_briefing_id=accepted_id,
            expect_durable_receipt=True,
        )
        # TASK-38990: the reload runs in the Artifacts loader group, so a newer
        # load supersedes it instead of racing it.
        assert reload_groups == ["wl-briefings-load"]


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_coordinated_failed_receipt_shows_the_generation_failed_state(
    monkeypatch,
):
    """TASK-38990 review: a failed coordinated receipt reaches the failure banner.

    The coordinated follower hands `generation_failed` to the loader group's
    refresh, which raises the banner after its own load lands.
    """
    app_instance = _build_test_app()
    watchlist_id, briefing_id = _seed_complete_briefing(app_instance)
    db = app_instance.watchlist_bundle_service.db
    failed_id = db.insert_briefing(watchlist_id, status="failed")
    host = ArtifactsScreenHarness(app_instance, "watchlists_collections")

    class Coordinator:
        async def accept_briefing(self, _watchlist_id, _preset_id):
            return {"id": failed_id}

    posted = _track_every_pane_load_request(monkeypatch)
    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        screen.active_section = "artifacts"
        await pilot.pause()
        await host.workers.wait_for_complete()
        screen._selected_briefing = next(
            row for row in screen._loaded_briefings if row["id"] == briefing_id
        )
        screen._apply_briefing_state_to_pane()
        # TASK-39058: the selection move queues a load; let it finish first.
        await _drain_briefings_loads(host, pilot, posted)

        await screen._follow_coordinated_briefing(
            db, watchlist_id, None, Coordinator()
        )
        await _drain_briefings_loads(host, pilot, posted)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert pane.view_state == "failed"
        assert (
            "Briefing generation failed. Last good content is still shown."
            in str(pane.query_one("#artifacts-state-message", Static).render())
        )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("coordinated", [False, True])
async def test_generation_completion_does_not_publish_into_a_new_watchlist_scope(
    monkeypatch,
    coordinated,
):
    """An old operation receipt cannot become a mismatch in a newly opened scope."""
    app_instance = _build_test_app()
    first_watchlist_id, _first_briefing_id = _seed_complete_briefing(app_instance)
    db = app_instance.watchlist_bundle_service.db
    second_watchlist_id = app_instance.watchlist_bundle_service.create("Second scope")[
        "id"
    ]
    second_briefing_id = db.insert_briefing(second_watchlist_id)
    db.update_briefing(
        second_briefing_id,
        status="complete",
        body_markdown="## Second scope briefing",
    )
    host = ArtifactsScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.tree_scope = TreeScope(
            kind="watchlist",
            watchlist_id=first_watchlist_id,
        )
        screen.active_section = "artifacts"
        await pilot.pause()
        await host.workers.wait_for_complete()
        entered = asyncio.Event()
        release = asyncio.Event()

        if coordinated:
            generated_id = db.insert_briefing(first_watchlist_id)

            class Coordinator:
                async def accept_briefing(self, _watchlist_id, _preset_id):
                    entered.set()
                    return {"id": generated_id}

            task = asyncio.create_task(
                screen._follow_coordinated_briefing(
                    db,
                    first_watchlist_id,
                    None,
                    Coordinator(),
                )
            )
        else:

            async def delayed_generation(db_handle, watchlist_id, **_kwargs):
                entered.set()
                await release.wait()
                generated_id = db_handle.insert_briefing(watchlist_id)
                db_handle.update_briefing(
                    generated_id,
                    status="complete",
                    body_markdown="## First scope generated",
                )
                return db_handle.get_briefing(generated_id)

            monkeypatch.setattr(
                screen_module,
                "generate_briefing",
                delayed_generation,
            )
            task = asyncio.create_task(
                screen._generate_briefing(db, first_watchlist_id, None)
            )

        await entered.wait()
        posted = _track_load_requests(
            screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        )
        screen._request_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=second_watchlist_id)
        )
        await screen._load_briefings(select_briefing_id=second_briefing_id)
        if coordinated:
            db.update_briefing(
                generated_id,
                status="complete",
                body_markdown="## First scope generated",
            )
        else:
            release.set()
        await task
        # The scope change queued its own load (TASK-39058); wait for it.
        await _drain_briefings_loads(host, pilot, posted)

        assert screen.tree_scope.watchlist_id == second_watchlist_id
        assert screen._selected_briefing["id"] == second_briefing_id
        assert screen._artifacts_view_state == "idle"


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_scope_change_does_not_suppress_generation_cancellation(monkeypatch):
    """Skipping an old-scope repaint must still propagate worker cancellation."""
    app_instance = _build_test_app()
    first_watchlist_id, _briefing_id = _seed_complete_briefing(app_instance)
    second_watchlist_id = app_instance.watchlist_bundle_service.create(
        "Cancellation destination"
    )["id"]
    db = app_instance.watchlist_bundle_service.db
    host = ArtifactsScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.tree_scope = TreeScope(
            kind="watchlist",
            watchlist_id=first_watchlist_id,
        )
        screen.active_section = "artifacts"
        await pilot.pause()
        await host.workers.wait_for_complete()
        entered = asyncio.Event()

        async def wait_forever(*_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(screen_module, "generate_briefing", wait_forever)
        task = asyncio.create_task(
            screen._generate_briefing(db, first_watchlist_id, None)
        )
        await entered.wait()
        screen._request_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=second_watchlist_id)
        )
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_citation_resolution_failure_retains_last_good_citations(monkeypatch):
    """A resolver outage is not evidence that every cited item was deleted."""
    app_instance = _build_test_app()
    watchlist_id, briefing_id = _seed_complete_briefing(app_instance)
    host = ArtifactsScreenHarness(app_instance, "watchlists_collections")
    posted = _track_every_pane_load_request(monkeypatch)

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=watchlist_id)
        screen.active_section = "artifacts"
        await pilot.pause()
        await host.workers.wait_for_complete()
        screen._selected_briefing = next(
            row for row in screen._loaded_briefings if row["id"] == briefing_id
        )
        screen._loaded_citations = [
            {"item_id": 7, "label": Text("[item 7] Last good"), "available": True}
        ]
        screen._citation_item_lookup = {7: {"id": "local:item:7"}}
        screen._apply_briefing_state_to_pane()
        # Drain the initial section-load messages before establishing the
        # exact last-good snapshot this refresh owns (TASK-39058: every
        # queued load, not one pause).
        await _drain_briefings_loads(host, pilot, posted)
        screen._loaded_citations = [
            {"item_id": 7, "label": Text("[item 7] Last good"), "available": True}
        ]
        screen._citation_item_lookup = {7: {"id": "local:item:7"}}
        screen.query_one(
            "#watchlists-artifacts-pane", ArtifactsPane
        ).citations = screen._loaded_citations
        failed_projection_id = app_instance.watchlist_bundle_service.db.insert_briefing(
            watchlist_id
        )
        app_instance.watchlist_bundle_service.db.update_briefing(
            failed_projection_id,
            status="complete",
            body_markdown="## New body whose citation lookup fails [item 7].",
        )

        def fail_citations(_item_ids):
            raise RuntimeError("private resolver detail")

        monkeypatch.setattr(
            app_instance.watchlist_bundle_service.db,
            "get_subscription_items_by_ids",
            fail_citations,
        )
        await screen._load_briefings(select_briefing_id=failed_projection_id)
        await _drain_briefings_loads(host, pilot, posted)

        pane = screen.query_one("#watchlists-artifacts-pane", ArtifactsPane)
        assert screen._loaded_citations[0]["available"] is True
        assert len(pane.briefings) == 1
        assert pane.selected_briefing["id"] == briefing_id
        assert pane.citations[0]["available"] is True
        assert "Last good" in str(pane.citations[0]["label"])
        assert pane.view_state == "failed"
        assert "private resolver detail" not in str(
            pane.query_one("#artifacts-state-message", Static).render()
        )
