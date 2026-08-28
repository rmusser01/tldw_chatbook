import pytest
from loguru import logger
from unittest.mock import AsyncMock

from tldw_chatbook.UI.Watchlists_Modules.watchlists_backend_controller import WatchlistsBackendController
from tldw_chatbook.Subscriptions.watchlist_item_page import (
    WatchlistItemPage,
)


class FakeScopeService:
    def __init__(self):
        self.calls = []

    def create_form_source_types(self, *, runtime_backend):
        self.calls.append(("create_form_source_types", runtime_backend))
        if runtime_backend == "server":
            return ("rss", "site", "forum")
        return ("rss", "atom", "url")

    async def list_watch_items(self, *, runtime_backend, **kwargs):
        return [{"id": 1, "title": "Source"}]

    async def create_watch_item(self, *, runtime_backend, payload):
        return {"id": 1, **payload}


@pytest.mark.parametrize(
    ("runtime_backend", "expected_backend", "expected_types"),
    [
        (None, "local", ("rss", "atom", "url")),
        (" SERVER ", "server", ("rss", "site", "forum")),
    ],
)
def test_create_form_source_types_routes_to_normalized_backend(
    runtime_backend, expected_backend, expected_types
):
    scope = FakeScopeService()
    ctrl = WatchlistsBackendController(
        app_instance=None,
        scope_service=scope,
        server_service=None,
    )

    assert (
        ctrl.create_form_source_types(runtime_backend=runtime_backend) == expected_types
    )
    assert scope.calls == [("create_form_source_types", expected_backend)]


@pytest.mark.parametrize(
    ("runtime_backend", "expected_types"),
    [
        ("local", ("rss", "atom", "url")),
        ("server", ("rss", "site", "forum")),
    ],
)
def test_create_form_source_types_degrades_when_capability_is_absent(
    runtime_backend, expected_types
):
    ctrl = WatchlistsBackendController(
        app_instance=None,
        scope_service=object(),
        server_service=None,
    )

    assert ctrl.create_form_source_types(runtime_backend=runtime_backend) == expected_types


def test_create_form_source_types_fallback_does_not_log_exception_payload():
    secret = "https://user:token@example.test/private/feed"

    class FailingScopeService:
        def create_form_source_types(self, *, runtime_backend):
            raise RuntimeError(secret)

    records = []
    sink_id = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        ctrl = WatchlistsBackendController(
            app_instance=None,
            scope_service=FailingScopeService(),
            server_service=None,
        )

        assert ctrl.create_form_source_types(runtime_backend="local") == (
            "rss",
            "atom",
            "url",
        )
    finally:
        logger.remove(sink_id)

    fallback = next(
        record
        for record in records
        if "Watchlists create-form source types unavailable" in record["message"]
    )
    assert fallback["exception"] is None
    assert "RuntimeError" in fallback["message"]
    assert secret not in fallback["message"]


def test_controller_normalizes_backend():
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=FakeScopeService(), server_service=None)
    assert ctrl._normalize_backend("server") == "server"
    assert ctrl._normalize_backend(None) == "local"


@pytest.mark.asyncio
async def test_list_sources_routes_to_scope_service():
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=FakeScopeService(), server_service=None)
    items = await ctrl.list_sources(runtime_backend="local")
    assert len(items) == 1
    assert items[0]["title"] == "Source"


@pytest.mark.asyncio
async def test_create_source_routes_to_scope_service():
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=FakeScopeService(), server_service=None)
    result = await ctrl.create_source(payload={"name": "New"})
    assert result["name"] == "New"


@pytest.mark.asyncio
async def test_preview_source_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.preview_source = AsyncMock(return_value={"items": ["a"], "log_text": "ok"})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.preview_source(runtime_backend="local", source_config={"url": "http://example.com/feed"})

    scope_service.preview_source.assert_awaited_once_with(
        runtime_backend="local", source_config={"url": "http://example.com/feed"}
    )
    assert result == {"items": ["a"], "log_text": "ok"}


@pytest.mark.asyncio
async def test_check_now_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.check_now = AsyncMock(return_value={"run_id": "42", "status": "queued"})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.check_now(runtime_backend="local", source_id="1")

    scope_service.check_now.assert_awaited_once_with(runtime_backend="local", source_id="1")
    assert result["run_id"] == "42"


@pytest.mark.asyncio
async def test_import_opml_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.import_opml = AsyncMock(return_value={"created": 2, "sources": [{"id": 1}, {"id": 2}]})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.import_opml(runtime_backend="local", xml_text="<opml></opml>")

    scope_service.import_opml.assert_awaited_once_with(runtime_backend="local", xml_text="<opml></opml>")
    assert result["created"] == 2
    assert len(result["sources"]) == 2


@pytest.mark.asyncio
async def test_export_opml_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.export_opml = AsyncMock(return_value="<opml></opml>")
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.export_opml(runtime_backend="local")

    scope_service.export_opml.assert_awaited_once_with(runtime_backend="local")
    assert result == "<opml></opml>"


@pytest.mark.asyncio
async def test_list_items_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.list_items = AsyncMock(return_value=[{"id": "local:watchlist_item:1", "title": "Post"}])
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.list_items(runtime_backend="local")

    scope_service.list_items.assert_awaited_once_with(runtime_backend="local")
    assert len(result) == 1
    assert result[0]["title"] == "Post"


@pytest.mark.asyncio
async def test_list_reader_items_page_preserves_typed_page():
    scope_service = AsyncMock()
    page = WatchlistItemPage(
        items=({"id": "local:watchlist_item:1"},),
        has_more=False,
        snapshot_max_item_id=1,
        snapshot_count=1,
        next_cursor=None,
    )
    scope_service.list_reader_items_page = AsyncMock(return_value=page)
    ctrl = WatchlistsBackendController(
        app_instance=None, scope_service=scope_service, server_service=None
    )

    result = await ctrl.list_reader_items_page(
        runtime_backend=" LOCAL ", statuses=["new", "reviewed"]
    )

    scope_service.list_reader_items_page.assert_awaited_once_with(
        runtime_backend="local", statuses=["new", "reviewed"]
    )
    assert result is page


@pytest.mark.asyncio
async def test_count_reader_item_arrivals_preserves_integer():
    scope_service = AsyncMock()
    scope_service.count_reader_item_arrivals = AsyncMock(return_value=3)
    ctrl = WatchlistsBackendController(
        app_instance=None, scope_service=scope_service, server_service=None
    )

    result = await ctrl.count_reader_item_arrivals(
        runtime_backend="local", snapshot_max_item_id=42
    )

    scope_service.count_reader_item_arrivals.assert_awaited_once_with(
        runtime_backend="local", snapshot_max_item_id=42
    )
    assert result == 3


@pytest.mark.asyncio
async def test_cancel_run_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.cancel_run = AsyncMock(return_value={"run_id": "42", "status": "cancelled"})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.cancel_run(runtime_backend="local", run_id="42")

    scope_service.cancel_run.assert_awaited_once_with(runtime_backend="local", run_id="42")
    assert result["run_id"] == "42"
    assert result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_save_alert_rule_create_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.save_alert_rule = AsyncMock(return_value={"rule_id": "7", "name": "New Rule"})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.save_alert_rule(runtime_backend="local", payload={"name": "New Rule"})

    scope_service.save_alert_rule.assert_awaited_once_with(
        runtime_backend="local", payload={"name": "New Rule"}
    )
    assert result["rule_id"] == "7"


@pytest.mark.asyncio
async def test_save_alert_rule_update_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.save_alert_rule = AsyncMock(return_value={"rule_id": "7", "name": "Updated Rule"})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.save_alert_rule(runtime_backend="local", payload={"id": "7", "name": "Updated Rule"})

    scope_service.save_alert_rule.assert_awaited_once_with(
        runtime_backend="local", payload={"id": "7", "name": "Updated Rule"}
    )
    assert result["name"] == "Updated Rule"


@pytest.mark.asyncio
async def test_overview_counts_paused_sources_as_in_error(monkeypatch):
    """task-2050 Qodo: the Overview's `sources_in_error` card must not DROP
    when a failing source finally trips auto-pause. A paused source's
    `status_summary` reads "paused" (not "error (N)"), so a startswith("error")
    predicate alone would exclude exactly the most-broken sources — the same
    regression the Sources pane's Error filter bucket fixed. Reds if the
    paused arm is removed from the count.
    """
    ctrl = WatchlistsBackendController(
        app_instance=None, scope_service=FakeScopeService(), server_service=None
    )

    async def fake_list_sources(**kwargs):
        return [
            {"id": "s1", "status_summary": "active", "active": True},
            {"id": "s2", "status_summary": "error (3)", "active": True},
            {"id": "s3", "status_summary": "paused", "active": False},
        ]

    async def fake_safe_list(*args, **kwargs):
        return []

    monkeypatch.setattr(ctrl, "list_sources", fake_list_sources)
    monkeypatch.setattr(ctrl, "_safe_list", fake_safe_list)

    data = await ctrl.get_overview_data(runtime_backend="local")

    assert data["sources_in_error"] == 2, (
        "erroring + paused must both count; a paused source failed past the "
        "threshold and is the most broken state there is"
    )
    assert data["total_sources"] == 3


@pytest.mark.asyncio
async def test_latest_run_status_is_none_not_the_string_unavailable_with_no_runs(
    monkeypatch,
):
    """TASK-2313, AC#2: UAT -- "Latest run status: unavailable" read as a
    fault, not a state, for a watchlist that has simply never run. The
    controller must hand back `None` (the same "nothing recorded" sentinel
    every OTHER missing value in this dict uses), not the string
    "unavailable" -- both `OverviewPane._card_value`'s "-" default and
    `WatchlistsCollectionsScreen._latest_run_status_text`'s "no runs yet"
    copy key off `None`/falsy, not off a specific magic string.
    """
    ctrl = WatchlistsBackendController(
        app_instance=None, scope_service=FakeScopeService(), server_service=None
    )

    async def fake_list_sources(**kwargs):
        return []

    async def fake_safe_list(*args, **kwargs):
        return []

    monkeypatch.setattr(ctrl, "list_sources", fake_list_sources)
    monkeypatch.setattr(ctrl, "_safe_list", fake_safe_list)

    data = await ctrl.get_overview_data(runtime_backend="local")

    assert data["latest_run_status"] is None, (
        f"expected None with no runs, got {data['latest_run_status']!r}"
    )


@pytest.mark.asyncio
async def test_an_unwired_scope_service_reports_its_own_status_not_no_runs_yet():
    """UAT batch-5 review, finding I1: the degraded-state dict (no
    `scope_service` at all) must NOT hand back the same `None` a healthy,
    simply-unrun watchlist gets -- "the feature isn't wired up" is a
    different condition from "this watchlist is fine and hasn't checked
    yet," and collapsing them (an earlier version of the TASK-2313 fix did
    exactly this) reintroduces the dishonesty this UAT programme removes,
    just one level down. Reverting `NOT_CONFIGURED_STATUS` back to `None`
    is the mutation this test pins: with the sentinel gone, this assertion
    fails identically to the healthy-zero-runs case, which is the bug.
    """
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=None, server_service=None)

    data = await ctrl.get_overview_data(runtime_backend="local")

    assert data["latest_run_status"] == WatchlistsBackendController.NOT_CONFIGURED_STATUS
    assert data["latest_run_status"] is not None, (
        "an unwired scope_service must not read identically to a healthy "
        "watchlist that simply has zero runs"
    )


@pytest.mark.asyncio
async def test_check_all_checks_each_source_and_soft_fails():
    """TASK-3791 plan task 5: refresh-all iterates the eligible ids in
    order, a failing source never stops the batch, and the result carries
    the per-source outcome for the aggregate toast."""
    ctrl = WatchlistsBackendController(
        app_instance=None, scope_service=AsyncMock(), server_service=None
    )
    seen: list[str] = []

    async def _check(*, runtime_backend=None, source_id):
        seen.append(source_id)
        if source_id == "2":
            raise RuntimeError("boom")
        return {"status": "completed"}

    ctrl.check_now = _check

    result = await ctrl.check_all(runtime_backend="local", source_ids=["1", "2", "3"])

    assert seen == ["1", "2", "3"], "one failure must not stop the batch"
    assert result["checked"] == 2
    assert result["failed"] == ["2"]


@pytest.mark.asyncio
async def test_check_all_with_no_sources_is_a_noop():
    ctrl = WatchlistsBackendController(
        app_instance=None, scope_service=AsyncMock(), server_service=None
    )
    ctrl.check_now = AsyncMock(side_effect=AssertionError("must not be called"))

    result = await ctrl.check_all(runtime_backend="local", source_ids=[])

    assert result == {"checked": 0, "failed": []}
