"""A failed watchlist check must be reported and must leave a trace — TASK-1090.

`_check_now_source` wrapped the whole fetch in `except Exception`, logged at
**debug**, and showed a transient toast. That is the swallow that hid
TASK-1100: `Check now` raised `ValueError` on every single press, the entire
feature was dead, and the only evidence was a debug line nobody reads and a
toast that had gone before anyone looked. Three UAT runs and a full test suite
reported the screen as working while it fetched nothing.

Worse than the swallow, and found while fixing it: a check that *ran* and
failed did not raise at all. `LocalWatchlistsService.execute_run` catches the
fetch error, records a `failed` run and returns it — so the screen's `try`
succeeded and it told the user **"Check now started."** over a feed that had
just 404'd.

A fetch is the one Watchlists operation that routinely fails for ordinary
reasons — the feed moved, the host is down, the XML is malformed, the network
is out — so it is exactly the operation that must report.
"""

from __future__ import annotations

import pytest
from loguru import logger
from rich.text import Text
from textual.widgets import Button, DataTable

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane


class Notified:
    """Capture what the app told the user, since the toast itself is transient."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, message, *args, severity: str = "information", **kwargs) -> None:
        self.calls.append((str(message), severity))

    @property
    def errors(self) -> list[str]:
        return [message for message, severity in self.calls if severity == "error"]


def _seed_source(app, *, name: str = "Summit Route") -> int:
    db = app.local_watchlists_service._db()
    return db.add_subscription(
        name=name, type="rss", source="https://summitroute.com/blog/feed.xml"
    )


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
async def test_a_failed_fetch_is_reported_as_a_failure_and_leaves_a_trace():
    """AC#1, AC#2, AC#4: the durable half.

    The run executor raises the way a dead host does. The service's own
    failure path then has to write `subscriptions.last_error`, record a
    `failed` run, and the screen has to say so instead of "Check now started."
    """
    app = _build_test_app()
    source_id = _seed_source(app)
    notified = Notified()
    app.notify = notified

    async def dead_host(subscription):
        raise ConnectionError("Name or service not known: summitroute.com")

    app.local_watchlists_service.run_executor = dead_host

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
            if notified.calls:
                break

        assert notified.errors, (
            "a check that failed must be reported as a failure; it used to "
            f"report success. Got: {notified.calls!r}"
        )
        assert "summitroute.com" in " ".join(notified.errors), (
            "the report must name the reason, not just say it failed"
        )

    db = app.local_watchlists_service._db()
    row = db.get_subscription(source_id)
    assert row["last_error"], "a failed check must write subscriptions.last_error"

    runs = await app.local_watchlists_service.list_runs(source_id=source_id)
    assert runs, "a failed check must leave a run behind, not vanish"
    assert runs[0]["status"] == "failed"
    assert runs[0]["error_msg"], "the recorded run must carry the error"


@pytest.mark.asyncio
async def test_the_sources_status_column_shows_the_failure_after_the_toast_has_gone():
    """AC#1 (the surfaced half) and AC#2.

    `_source_row_cells` read `source.get("status")`, a key no normalizer
    emits -- `normalize_local_subscription_row` publishes `status_summary`.
    So the Status column read `-` whatever had happened, and `Last scraped`
    read `-` even straight after a successful check.
    """
    app = _build_test_app()
    source_id = _seed_source(app)
    app.notify = Notified()

    async def dead_host(subscription):
        raise ConnectionError("connection refused")

    app.local_watchlists_service.run_executor = dead_host

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen, pane = await _open_sources(pilot, host)
        pane.select_source_by_id(str(pane.sources[0]["id"]))
        await pilot.pause(0.2)
        pane.query_one("#sources-check-now-button", Button).press()

        # The pane recomposes while the source list reloads, so the table is
        # briefly absent; poll rather than grabbing a handle once.
        status_cell = ""
        for _ in range(120):
            await pilot.pause()
            try:
                table = screen.query_one("#sources-table", DataTable)
                if not table.row_count:
                    continue
                status_cell = str(table.get_cell_at((0, 2)))
            except Exception:
                continue
            if "error" in status_cell.lower():
                break

        assert "error" in status_cell.lower(), (
            "after a failed check the Sources table must say so in its Status "
            f"column; it read {status_cell!r}"
        )


@pytest.mark.asyncio
async def test_an_unexpected_exception_in_the_fetch_path_logs_above_debug():
    """AC#3: the level that hid TASK-1100 for three UAT runs."""
    app = _build_test_app()
    _seed_source(app)
    app.notify = Notified()

    records: list[tuple[str, str]] = []
    sink_id = logger.add(
        lambda message: records.append(
            (message.record["level"].name, message.record["message"])
        ),
        level="DEBUG",
    )

    host = DestinationHarness(app, "watchlists_collections")
    try:
        async with host.run_test(size=(180, 50)) as pilot:
            await pilot.pause(0.2)
            screen, pane = await _open_sources(pilot, host)

            async def boom(**kwargs):
                raise ValueError("invalid literal for int() with base 10")

            screen._controller.check_now = boom
            pane.select_source_by_id(str(pane.sources[0]["id"]))
            await pilot.pause(0.2)
            records.clear()
            pane.query_one("#sources-check-now-button", Button).press()
            for _ in range(60):
                await pilot.pause()
                if any("check" in message.lower() for _, message in records):
                    break
    finally:
        logger.remove(sink_id)

    check_records = [
        (level, message)
        for level, message in records
        if "invalid literal" in message or "check" in message.lower()
    ]
    assert check_records, "the failure must be logged at all"
    assert any(
        level in ("WARNING", "ERROR", "CRITICAL") for level, _ in check_records
    ), (
        "an unexpected exception in the fetch path must log at warning or "
        f"above, not debug. Got: {check_records!r}"
    )


@pytest.mark.asyncio
async def test_a_check_that_fails_before_execution_still_records_a_run():
    """AC#4's remaining gap.

    `execute_run` records its own failure, but anything that goes wrong
    *around* it -- the namespaced-id `ValueError` of TASK-1100, a bad run id,
    a service fault -- left the queued run sitting at `queued` forever with no
    error on it and nothing else written anywhere.
    """
    app = _build_test_app()
    source_id = _seed_source(app)
    service = app.local_watchlists_service

    async def broken_execute(run_id):
        raise ValueError("invalid literal for int() with base 10: 'local'")

    service.execute_run = broken_execute

    with pytest.raises(ValueError):
        await app.watchlist_scope_service.check_now(
            runtime_backend="local", source_id=f"local:subscription:{source_id}"
        )

    runs = await service.list_runs(source_id=source_id)
    assert runs, "the launched run must not disappear when execution fails"
    assert runs[0]["status"] == "failed", (
        f"a run whose execution raised must be recorded failed, not "
        f"{runs[0]['status']!r}"
    )
    assert "invalid literal" in str(runs[0]["error_msg"] or ""), (
        "the recorded run must carry the error that stopped it"
    )
    row = service._db().get_subscription(source_id)
    assert row["last_error"], (
        "a check that failed before it fetched anything must still mark the "
        "source as errored"
    )


def test_source_row_cells_render_the_normalizer_status_summary():
    """The column mapping, at the unit level (AC#1).

    Pinned separately because the pane's own tests all feed a synthetic
    `status`/`last_scraped` shape that the real normalizer never produces,
    which is how a column reading `-` for every source in every state stayed
    green.
    """
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_local_subscription_row,
    )

    source = normalize_local_subscription_row(
        {
            "id": 1,
            "name": "Summit Route",
            "type": "rss",
            "source": "https://summitroute.com/blog/feed.xml",
            "is_active": 1,
            "last_error": "connection refused",
            "error_count": 2,
            "last_checked": "2026-07-28T09:00:00+00:00",
        }
    )
    cells = SourcesPane._source_row_cells(source, False)
    assert isinstance(cells[2], Text)
    assert "error" in cells[2].plain.lower(), (
        f"Status column rendered {cells[2].plain!r} for an errored source"
    )
    assert cells[3].plain == "2026-07-28T09:00:00+00:00", (
        f"Last scraped column rendered {cells[3].plain!r}"
    )


# TASK-1090 AC#6. Every Watchlists action a user can *press*. A swallowed
# failure in one of these means the button did nothing and nothing durable
# said so -- which is exactly how `Check now` stayed dead through three UAT
# runs and a green suite. Load/refresh coroutines are deliberately NOT in this
# list: they are background reads whose failure is already visible as an empty
# region plus a "Failed to load ..." toast, and promoting them would make an
# offline session log a wall of warnings.
USER_INITIATED_MUTATIONS = (
    "_start_tree_write",
    "_run_tree_write",
    "_create_source",
    "_cancel_run",
    "_rerun_run",
    "_preview_source",
    "_check_now_source",
    "_on_opml_import_complete",
    "_export_opml",
    "_update_item_status",
    "_save_rule",
    "_delete_source",
    "_delete_run",
    "_delete_rule",
    "_delete_item",
)


def _method_source(module_source: str, name: str) -> str:
    """The body of one method, from its `def` to the next same-indent `def`."""
    import re

    match = re.search(rf"^(\s*)(?:async )?def {re.escape(name)}\(", module_source, re.M)
    assert match, f"{name} is not defined on the Watchlists screen any more"
    indent = match.group(1)
    rest = module_source[match.end():]
    end = re.search(rf"^{indent}(?:async )?def |^{indent}@", rest, re.M)
    return rest[: end.start()] if end else rest


@pytest.mark.parametrize("method_name", USER_INITIATED_MUTATIONS)
def test_user_initiated_actions_do_not_swallow_failures_into_debug(method_name):
    """AC#6, as a contract rather than a one-time audit."""
    from pathlib import Path

    screen_source = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "UI"
        / "Screens"
        / "watchlists_collections_screen.py"
    ).read_text(encoding="utf-8")

    body = _method_source(screen_source, method_name)
    assert ".debug(" not in body, (
        f"{method_name} logs a failure at debug. It is behind a control the "
        "user pressed, so a swallowed failure there is invisible: the action "
        "appears to do nothing. Log at warning or above."
    )
