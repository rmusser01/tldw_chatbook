from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.briefing_service import (
    accept_briefing,
    execute_accepted_briefing,
)
from tldw_chatbook.Subscriptions.local_watchlists_service import (
    LocalWatchlistsService,
)
from tldw_chatbook.Subscriptions.startup_reconcile import (
    capture_prior_process_boundary,
)
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.Subscriptions.watchlists_operation_coordinator import (
    WatchlistsOperationCoordinator,
)


def _source(db: SubscriptionsDB, number: int) -> int:
    return db.add_subscription(
        name=f"Feed {number}",
        type="rss",
        source=f"https://example.com/{number}.xml",
    )


@pytest.mark.asyncio
async def test_source_acceptance_validates_the_whole_batch_before_writing(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    valid_id = _source(db, 1)

    with pytest.raises(KeyError, match="Subscription not found"):
        await service.accept_source_checks([valid_id, valid_id + 999])

    rows = db.conn.execute("SELECT id FROM local_watchlist_runs").fetchall()
    assert rows == []


@pytest.mark.asyncio
async def test_source_acceptance_returns_the_database_winner_without_reexecution(
    tmp_path,
):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source_id = _source(db, 1)

    first = (await service.accept_source_checks([source_id]))[0]
    second = (await service.accept_source_checks([source_id]))[0]

    assert first["run_id"] == second["run_id"]
    assert first["_claim_acquired"] is True
    assert second["_claim_acquired"] is False
    assert db.conn.execute("SELECT COUNT(*) FROM local_watchlist_runs").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_briefing_acceptance_is_durable_before_exact_execution(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    watchlist_id = WatchlistBundleService(db).create("Threat intel")["id"]

    accepted = await accept_briefing(db, watchlist_id, preset_id=None)
    duplicate = await accept_briefing(db, watchlist_id, preset_id=None)

    assert accepted["id"] == duplicate["id"]
    assert accepted["_claim_acquired"] is True
    assert duplicate["_claim_acquired"] is False
    assert db.get_briefing(accepted["id"])["status"] == "generating"

    finished = await execute_accepted_briefing(
        db,
        accepted["id"],
        chat=lambda **_kwargs: {"message": {"content": "unused"}},
    )
    assert finished["id"] == accepted["id"]
    assert finished["status"] == "empty"


@pytest.mark.asyncio
async def test_coordinator_caps_checks_at_four_and_keeps_strong_receipt_tasks(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    entered = 0
    peak = 0
    four_running = asyncio.Event()
    release = asyncio.Event()

    async def executor(_subscription):
        nonlocal entered, peak
        entered += 1
        peak = max(peak, entered)
        if entered == 4:
            four_running.set()
        await release.wait()
        entered -= 1
        return {"items": []}

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=executor)
    source_ids = [_source(db, number) for number in range(1, 7)]
    coordinator = WatchlistsOperationCoordinator(
        local_service=service,
        briefing_db=db,
    )

    receipts = await coordinator.accept_checks(source_ids)
    await asyncio.wait_for(four_running.wait(), timeout=2)

    assert peak == 4
    assert len(coordinator.active_receipt_ids) == 6
    assert [receipt["run_id"] for receipt in receipts] == list(range(1, 7))

    release.set()
    await coordinator.wait_idle(timeout=2)
    assert coordinator.active_receipt_ids == ()
    assert {
        row["status"]
        for row in db.conn.execute("SELECT status FROM local_watchlist_runs")
    } == {"completed"}


@pytest.mark.asyncio
async def test_duplicate_coordinator_submission_reuses_one_task_and_receipt(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def executor(_subscription):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"items": []}

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=executor)
    source_id = _source(db, 1)
    coordinator = WatchlistsOperationCoordinator(
        local_service=service,
        briefing_db=db,
    )

    first = (await coordinator.accept_checks([source_id]))[0]
    await asyncio.wait_for(started.wait(), timeout=2)
    second = (await coordinator.accept_checks([source_id]))[0]

    assert first["run_id"] == second["run_id"]
    assert calls == 1
    assert coordinator.active_receipt_ids == (
        f"local:watchlist_run:{first['run_id']}",
    )

    release.set()
    await coordinator.wait_idle(timeout=2)


@pytest.mark.asyncio
async def test_shutdown_stops_acceptance_and_terminalizes_cancelled_receipts(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    started = asyncio.Event()

    async def executor(_subscription):
        started.set()
        await asyncio.Event().wait()

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=executor)
    first_id = _source(db, 1)
    second_id = _source(db, 2)
    coordinator = WatchlistsOperationCoordinator(
        local_service=service,
        briefing_db=db,
    )

    receipt = (await coordinator.accept_checks([first_id]))[0]
    await asyncio.wait_for(started.wait(), timeout=2)
    await coordinator.shutdown(timeout=1)

    assert (await service.get_run(receipt["run_id"]))["status"] == "cancelled"
    with pytest.raises(RuntimeError, match="shutting down"):
        await coordinator.accept_checks([second_id])


@pytest.mark.asyncio
async def test_coordinator_scrubs_unexpected_check_failures(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    secret = "https://example.com/?token=top-secret"

    async def executor(_subscription):
        raise RuntimeError(secret)

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=executor)
    source_id = _source(db, 1)
    coordinator = WatchlistsOperationCoordinator(
        local_service=service,
        briefing_db=db,
    )

    receipt = (await coordinator.accept_checks([source_id]))[0]
    await coordinator.wait_idle(timeout=2)
    row = await service.get_run(receipt["run_id"])

    assert row["status"] == "failed"
    assert row["error_msg"] == "Watchlists source check failed. Try again."
    assert secret not in str(row)


@pytest.mark.asyncio
async def test_coordinator_reconciles_receipts_stranded_before_startup(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source_id = _source(db, 1)
    receipt = (await service.accept_source_checks([source_id]))[0]
    boundary = capture_prior_process_boundary(db)
    coordinator = WatchlistsOperationCoordinator(
        local_service=service,
        briefing_db=db,
    )

    reconciled = await coordinator.reconcile_startup(boundary)

    assert reconciled["runs"] == 1
    assert (await service.get_run(receipt["run_id"]))["status"] == "failed"
