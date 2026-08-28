from __future__ import annotations

import asyncio
import time

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
import tldw_chatbook.Subscriptions.watchlists_operation_coordinator as coordinator_module


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


@pytest.mark.asyncio
async def test_shutdown_timeout_stops_waiting_for_cancellation_ignoring_task(
    tmp_path,
):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    started = asyncio.Event()
    release = asyncio.Event()

    async def executor(_subscription):
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        return {"items": []}

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=executor)
    source_id = _source(db, 1)
    coordinator = WatchlistsOperationCoordinator(
        local_service=service,
        briefing_db=db,
    )
    receipt = (await coordinator.accept_checks([source_id]))[0]
    operation_id = f"local:watchlist_run:{receipt['run_id']}"
    await asyncio.wait_for(started.wait(), timeout=2)

    before = time.monotonic()
    shutdown_task = asyncio.create_task(coordinator.shutdown(timeout=0.02))
    done, _pending = await asyncio.wait({shutdown_task}, timeout=0.2)
    elapsed = time.monotonic() - before

    if not done:
        release.set()
        await asyncio.wait_for(shutdown_task, timeout=1)
        pytest.fail("shutdown waited beyond its wall-clock timeout")
    await shutdown_task
    assert elapsed < 0.2
    assert operation_id in coordinator.active_receipt_ids
    assert (await service.get_run(receipt["run_id"]))["status"] == "cancelled"

    release.set()
    await coordinator.wait_idle(timeout=2)
    assert operation_id not in coordinator.active_receipt_ids


@pytest.mark.asyncio
async def test_check_ownership_survives_failed_terminal_write_and_duplicate_retries(
    tmp_path, monkeypatch
):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    execute_calls = 0

    async def executor(_subscription):
        nonlocal execute_calls
        execute_calls += 1
        raise RuntimeError("secret execution failure")

    service = LocalWatchlistsService(db_factory=lambda: db, run_executor=executor)
    source_id = _source(db, 1)
    real_record_failure = service.record_run_failure
    failure_calls = 0

    async def broken_record_failure(*_args, **_kwargs):
        nonlocal failure_calls
        failure_calls += 1
        raise RuntimeError("terminal storage unavailable")

    monkeypatch.setattr(service, "record_run_failure", broken_record_failure)
    coordinator = WatchlistsOperationCoordinator(
        local_service=service,
        briefing_db=db,
    )

    first = (await coordinator.accept_checks([source_id]))[0]
    operation_id = f"local:watchlist_run:{first['run_id']}"
    await coordinator.wait_idle(timeout=2)

    assert operation_id in coordinator.active_receipt_ids
    assert (await service.get_run(first["run_id"]))["status"] == "running"
    assert failure_calls >= 2

    monkeypatch.setattr(service, "record_run_failure", real_record_failure)
    duplicate = (await coordinator.accept_checks([source_id]))[0]
    await coordinator.wait_idle(timeout=2)

    assert duplicate["run_id"] == first["run_id"]
    assert execute_calls == 1
    assert (await service.get_run(first["run_id"]))["status"] == "failed"
    assert operation_id not in coordinator.active_receipt_ids


@pytest.mark.asyncio
async def test_briefing_duplicate_retries_terminal_write_without_provider_replay(
    tmp_path,
    monkeypatch,
):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    watchlist_id = WatchlistBundleService(db).create("Threat intel")["id"]
    provider_side_effects = 0

    async def broken_execute(*_args, **_kwargs):
        nonlocal provider_side_effects
        provider_side_effects += 1
        raise RuntimeError("secret model failure")

    monkeypatch.setattr(
        coordinator_module,
        "execute_accepted_briefing",
        broken_execute,
    )
    real_transition = db.transition_briefing
    failure_calls = 0

    def broken_transition(briefing_id, *, status, **kwargs):
        nonlocal failure_calls
        if status == "failed":
            failure_calls += 1
            raise RuntimeError("terminal storage unavailable")
        return real_transition(briefing_id, status=status, **kwargs)

    monkeypatch.setattr(db, "transition_briefing", broken_transition)
    coordinator = WatchlistsOperationCoordinator(
        local_service=LocalWatchlistsService(db_factory=lambda: db),
        briefing_db=db,
    )

    first = await coordinator.accept_briefing(watchlist_id)
    operation_id = f"local:briefing:{first['id']}"
    await coordinator.wait_idle(timeout=2)

    assert operation_id in coordinator.active_receipt_ids
    assert db.get_briefing(first["id"])["status"] == "generating"
    assert failure_calls >= 2

    monkeypatch.setattr(db, "transition_briefing", real_transition)
    duplicate = await coordinator.accept_briefing(watchlist_id)
    await coordinator.wait_idle(timeout=2)

    assert duplicate["id"] == first["id"]
    assert provider_side_effects == 1
    assert db.get_briefing(first["id"])["status"] == "failed"
    assert operation_id not in coordinator.active_receipt_ids


@pytest.mark.asyncio
async def test_ownerless_duplicate_does_not_interrupt_live_other_coordinator_briefing(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "subscriptions.db"
    loser_db = SubscriptionsDB(db_path, "loser")
    loser_boundary = capture_prior_process_boundary(loser_db)
    winner_db = SubscriptionsDB(db_path, "winner")
    watchlist_id = WatchlistBundleService(winner_db).create("Threat intel")["id"]
    provider_entered = asyncio.Event()
    provider_release = asyncio.Event()
    provider_side_effects = 0
    publish_succeeded: list[bool] = []

    async def provider_path(db, briefing_id, **_kwargs):
        nonlocal provider_side_effects
        provider_side_effects += 1
        provider_entered.set()
        await provider_release.wait()
        published = await asyncio.to_thread(
            db.transition_briefing,
            briefing_id,
            status="empty",
        )
        publish_succeeded.append(published is not None)
        return published

    monkeypatch.setattr(
        coordinator_module,
        "execute_accepted_briefing",
        provider_path,
    )
    winner = WatchlistsOperationCoordinator(
        local_service=LocalWatchlistsService(db_factory=lambda: winner_db),
        briefing_db=winner_db,
    )
    loser = WatchlistsOperationCoordinator(
        local_service=LocalWatchlistsService(db_factory=lambda: loser_db),
        briefing_db=loser_db,
    )

    first = await winner.accept_briefing(watchlist_id)
    operation_id = f"local:briefing:{first['id']}"
    await asyncio.wait_for(provider_entered.wait(), timeout=2)

    try:
        reconciled = await loser.reconcile_startup(loser_boundary)
        duplicate = await loser.accept_briefing(watchlist_id)
        await loser.wait_idle(timeout=2)

        assert reconciled["briefings"] == 0
        assert duplicate["id"] == first["id"]
        assert provider_side_effects == 1
        assert loser_db.get_briefing(first["id"])["status"] == "generating"
        assert loser_db.get_briefing(first["id"])["error"] is None
        assert loser.active_receipt_ids == ()
        assert winner.active_receipt_ids == (operation_id,)
    finally:
        provider_release.set()
        await winner.wait_idle(timeout=2)

    assert provider_side_effects == 1
    assert publish_succeeded == [True]
    assert winner_db.get_briefing(first["id"])["status"] == "empty"
    assert winner.active_receipt_ids == ()


@pytest.mark.asyncio
async def test_ownerless_queued_check_waits_for_startup_boundary_reconciliation(
    tmp_path,
):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    source_id = _source(db, 1)
    service = LocalWatchlistsService(
        db_factory=lambda: db,
        run_executor=lambda _source: asyncio.sleep(0, result={"items": []}),
    )
    check = (await service.accept_source_checks([source_id]))[0]
    boundary = capture_prior_process_boundary(db)
    coordinator = WatchlistsOperationCoordinator(
        local_service=service,
        briefing_db=db,
    )

    duplicate = (await coordinator.accept_checks([source_id]))[0]
    await coordinator.wait_idle(timeout=2)

    assert duplicate["run_id"] == check["run_id"]
    assert (await service.get_run(check["run_id"]))["status"] == "queued"
    assert coordinator.active_receipt_ids == ()

    reconciled = await coordinator.reconcile_startup(boundary)

    assert reconciled["runs"] == 1
    assert (await service.get_run(check["run_id"]))["status"] == "failed"


@pytest.mark.asyncio
async def test_startup_boundary_terminalizes_process_orphan_briefing(tmp_path):
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    watchlist_id = WatchlistBundleService(db).create("Threat intel")["id"]
    orphan = await accept_briefing(db, watchlist_id, preset_id=None)
    boundary = capture_prior_process_boundary(db)
    coordinator = WatchlistsOperationCoordinator(
        local_service=LocalWatchlistsService(db_factory=lambda: db),
        briefing_db=db,
    )

    reconciled = await coordinator.reconcile_startup(boundary)

    assert reconciled["briefings"] == 1
    assert db.get_briefing(orphan["id"])["status"] == "failed"
    assert db.get_briefing(orphan["id"])["error"] == "interrupted"
    assert coordinator.active_receipt_ids == ()


@pytest.mark.asyncio
async def test_ownerless_duplicate_does_not_interrupt_live_other_coordinator_check(
    tmp_path,
):
    db_path = tmp_path / "subscriptions.db"
    loser_db = SubscriptionsDB(db_path, "loser")
    winner_db = SubscriptionsDB(db_path, "winner")
    source_id = _source(winner_db, 1)
    winner_entered = asyncio.Event()
    winner_release = asyncio.Event()
    winner_effects = 0
    loser_effects = 0

    async def winner_executor(_subscription):
        nonlocal winner_effects
        winner_effects += 1
        winner_entered.set()
        await winner_release.wait()
        return {"items": []}

    async def loser_executor(_subscription):
        nonlocal loser_effects
        loser_effects += 1
        return {"items": []}

    winner_service = LocalWatchlistsService(
        db_factory=lambda: winner_db,
        run_executor=winner_executor,
    )
    loser_service = LocalWatchlistsService(
        db_factory=lambda: loser_db,
        run_executor=loser_executor,
    )
    winner = WatchlistsOperationCoordinator(
        local_service=winner_service,
        briefing_db=winner_db,
    )
    loser = WatchlistsOperationCoordinator(
        local_service=loser_service,
        briefing_db=loser_db,
    )

    first = (await winner.accept_checks([source_id]))[0]
    operation_id = f"local:watchlist_run:{first['run_id']}"
    await asyncio.wait_for(winner_entered.wait(), timeout=2)

    try:
        duplicate = (await loser.accept_checks([source_id]))[0]
        await loser.wait_idle(timeout=2)

        assert duplicate["run_id"] == first["run_id"]
        assert winner_effects == 1
        assert loser_effects == 0
        assert (await loser_service.get_run(first["run_id"]))["status"] == "running"
        assert loser.active_receipt_ids == ()
        assert winner.active_receipt_ids == (operation_id,)
    finally:
        winner_release.set()
        await winner.wait_idle(timeout=2)

    assert winner_effects == 1
    assert loser_effects == 0
    assert (await winner_service.get_run(first["run_id"]))["status"] == "completed"
    assert winner.active_receipt_ids == ()
