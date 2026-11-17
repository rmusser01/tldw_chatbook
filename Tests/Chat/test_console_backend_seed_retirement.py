"""Original fleet seed results cannot stand in for physical retirement."""

import asyncio
import copy
import inspect
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_stock_owned_db_callback_retains_console_seed_owner_keyword(request):
    from tldw_chatbook.DB.base_db import run_owned_db_call

    supplied = object()

    def operation(*, _console_seed_owner):
        return _console_seed_owner

    assert (
        await run_owned_db_call(object(), operation, _console_seed_owner=supplied)
        is supplied
    )


async def _until(predicate):
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(0.01)
    assert predicate(), "Original seed did not reach the selected boundary"


@asynccontextmanager
async def _backend():
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import (
        open_console_backend_profile,
    )

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(
        expected_config_path=path, expected_data_directory=directory
    )
    backend = ConsoleExecutionBackend(
        profile, profile_epoch="profile", service_epoch="seed-receipt"
    )
    try:
        backend.start()
        session = backend.store.create_session(title="Saved seed receipt")
        assert backend.store.persist_session_if_needed(session.id)
        lifecycle = backend.presentation.view_lifecycle
        checkpoint = lifecycle.admit("actual-seed-client", 1, "mount")
        seed = lifecycle._seeds[lifecycle._attachment]
        yield backend, lifecycle, checkpoint, seed
    finally:
        if backend.presentation.view_lifecycle._drain_error is None:
            await backend.aclose()
        else:
            with pytest.raises(RuntimeError, match="retirement is unproven"):
                await backend.aclose()


@private_profile_test
@pytest.mark.asyncio
async def test_saved_seed_retains_original_callback_and_exact_positive_receipt(
    request, monkeypatch
):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator
    from tldw_chatbook.Chat.console_backend_seed_retirement import (
        ConsoleBackendSeedRead,
    )
    from tldw_chatbook.Chat.console_backend_view_lifecycle import _seed_retired
    from tldw_chatbook.DB.base_db import run_owned_db_call

    original_invoke = next(
        code
        for code in run_owned_db_call.__code__.co_consts
        if inspect.iscode(code) and code.co_name == "invoke"
    )
    callers = []
    async with _backend() as (backend, lifecycle, checkpoint, seed):

        def entered(frame):
            if frame.f_locals["self"] is backend.controller._fleet_wake:
                callers.append(frame.f_back.f_code)

        with _observe(
            ConsoleFleetWakeCoordinator.seed_from_marks.__code__,
            sys.monitoring.events.PY_START,
            entered,
        ):
            await lifecycle.prepare(checkpoint)
        receipt = seed.receipt
        assert callers == [original_invoke]
        assert seed.task.result() == 0 and receipt.confirmed()
        assert receipt.handoff._producer is receipt.producer
        assert receipt.observation.connections
        assert all(
            item[0] is receipt.database for item in receipt.observation.connections
        )
        assert not copy.copy(receipt).confirmed()
        with monkeypatch.context() as changed:
            changed.setattr(ConsoleBackendSeedRead, "confirmed", lambda self: True)
            assert not _seed_retired(receipt)
        assert backend.controller._interrupt_host.view_visible
        await lifecycle.drain()
        await lifecycle.drain()
        assert receipt.confirmed()


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["query", "close"])
async def test_optional_seed_error_keeps_independent_sticky_native_outcome(
    request, failure
):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.store_kernel import SQLiteStoreKernel

    async with _backend() as (backend, lifecycle, checkpoint, seed):
        database = backend.controller._fleet_wake._runs_db()
        loop = asyncio.get_running_loop()
        previous_executor = loop._default_executor
        executor = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        failed = []

        def fail_original(frame):
            if frame.f_locals.get("self") is not database or failed:
                return
            failed.append(True)
            if failure == "query":
                # An actual query error after the original getter enrolled its
                # real connection; no owner or source definition is replaced.
                with database.connection() as connection:
                    connection.execute(
                        "SELECT absent_seed_receipt_column FROM agent_runs"
                    )
            raise OSError("Original seed connection close failed")

        code = (
            AgentRunsDB.pending_wake_conversation_ids
            if failure == "query"
            else SQLiteStoreKernel.close
        ).__code__
        try:
            with _observe(code, sys.monitoring.events.PY_START, fail_original):
                if failure == "query":
                    await lifecycle.prepare(checkpoint)
                else:
                    with pytest.raises(
                        RuntimeError, match="native retirement is unproven"
                    ):
                        await lifecycle.prepare(checkpoint)
            receipt = seed.receipt
            assert failed and seed.task.result() == 0  # Original optional behavior.
            assert receipt.observation.connections
            if failure == "query":
                assert isinstance(receipt.producer.exception(), sqlite3.OperationalError)
            else:
                # The stock DB context catches close errors. Its optional
                # result can succeed while the observed connection stays live.
                assert receipt.producer.result() == 0
            assert receipt.confirmed() is (failure == "query")
            if failure == "close":
                assert not backend.controller._interrupt_host.view_visible
                assert not receipt.observation.confirmed()
                # The exact executor thread performs the original later close.
                await loop.run_in_executor(None, database.close)
                assert receipt.observation.confirmed()
                assert not receipt.confirmed()  # Later cleanup cannot rewrite it.
                with pytest.raises(RuntimeError, match="retirement is unproven"):
                    await lifecycle.drain()
                with pytest.raises(RuntimeError, match="retirement is unproven"):
                    await lifecycle.drain()
        finally:
            await loop.run_in_executor(None, database.close)
            executor.shutdown(wait=True)
            loop._default_executor = previous_executor


@private_profile_test
@pytest.mark.asyncio
async def test_saved_seed_waits_for_actual_executor_callback_tail(request):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import (
        run_preparation_read,
    )

    code = next(
        item
        for item in run_preparation_read.__code__.co_consts
        if inspect.iscode(item) and item.co_name == "invoke"
    )
    entered, release = threading.Event(), threading.Event()
    async with _backend() as (backend, lifecycle, checkpoint, seed):

        def hold_tail(frame):
            read = frame.f_locals.get("read")
            if seed.receipt is not None and read is seed.receipt.handoff:
                entered.set()
                assert release.wait(10), "Original seed executor tail was not released"

        with _observe(code, sys.monitoring.events.PY_RETURN, hold_tail):
            task = asyncio.create_task(lifecycle.prepare(checkpoint))
            try:
                await _until(entered.is_set)
                receipt = seed.receipt
                assert receipt.handoff.callback_returned
                assert receipt.observation.closed
                assert not receipt.producer.done() and not receipt.confirmed()
                assert (
                    not task.done()
                    and not backend.controller._interrupt_host.view_visible
                )
                release.set()
                await task
                assert (
                    receipt.confirmed()
                    and backend.controller._interrupt_host.view_visible
                )
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["queued", "running"])
async def test_saved_seed_cancel_retains_actual_producer_until_retired(
    request, boundary
):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator

    entered, release = threading.Event(), threading.Event()
    async with _backend() as (backend, lifecycle, checkpoint, seed):
        loop = asyncio.get_running_loop()
        previous_executor = loop._default_executor
        executor = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        occupied = executor.submit(release.wait, 10) if boundary == "queued" else None

        def hold_body(frame):
            if (
                boundary == "running"
                and frame.f_locals["self"] is backend.controller._fleet_wake
            ):
                entered.set()
                assert release.wait(10), "Original running seed was not released"

        with _observe(
            ConsoleFleetWakeCoordinator.seed_from_marks.__code__,
            sys.monitoring.events.PY_START,
            hold_body,
        ):
            task = asyncio.create_task(lifecycle.prepare(checkpoint))
            try:
                if boundary == "queued":
                    await _until(
                        lambda: seed.receipt is not None and bool(seed.receipt.reads)
                    )
                    (read,) = seed.receipt.reads
                    assert read._producer is not None and not read._producer.done()
                    assert seed.receipt.handoff is None
                else:
                    await _until(entered.is_set)
                receipt = seed.receipt
                seed.task.cancel()  # The original producer's own task is cancelled.
                await asyncio.sleep(0)
                seed.task.cancel()
                await asyncio.sleep(0)
                assert (
                    not seed.task.done() and not task.done() and not receipt.confirmed()
                )
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert seed.task.cancelled() and receipt.confirmed()
                assert not backend.controller._interrupt_host.view_visible
                await lifecycle.drain()
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                if occupied is not None:
                    occupied.result(10)
                executor.shutdown(wait=True)
                loop._default_executor = previous_executor
