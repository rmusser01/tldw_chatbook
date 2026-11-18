"""Display workers keep foreign custody separate without weakening refusal."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from Tests.private_profile import private_profile_test


def _backend():
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
        profile, profile_epoch="profile", service_epoch="display-worker"
    )
    backend.start()
    return backend


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed_worker", ["default", "display"])
async def test_display_worker_separates_foreign_custody_but_still_refuses_own_borrow(
    request, borrowed_worker,
):
    from tldw_chatbook.Chat.console_display_phases import ConsoleDisplayPhaseRefused
    from tldw_chatbook.Chat.console_execution_protocol import (
        ConsoleWireMessage,
        WireIdentity,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    backend = _backend()
    loop = asyncio.get_running_loop()
    previous_default = loop._default_executor
    default = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-foreign-custody"
    )
    loop.set_default_executor(default)
    display = backend.runtime._display_phase_executor.executor()
    database = backend.store.persistence.db
    selected = default if borrowed_worker == "default" else display
    connection = None
    receipt = None
    try:
        session = backend.store.create_session(
            title="Actual display custody",
            assistant_kind="generic",
            settings=ConsoleSessionSettings(
                provider="openai", model=None, source="user"
            ),
        )

        def open_borrowed():
            connection = database.get_connection()
            assert database._connection_quiescence.is_registered(connection)
            assert not connection.in_transaction
            return connection, threading.current_thread()

        connection, owner = await loop.run_in_executor(selected, open_borrowed)
        assert await loop.run_in_executor(display, threading.current_thread) is not (
            await loop.run_in_executor(default, threading.current_thread)
        )
        receipt = backend.dispatch(
            ConsoleWireMessage(
                WireIdentity(
                    "profile",
                    "display-worker",
                    1,
                    "actual-display",
                    session.id,
                    session.incarnation_id,
                    backend.controls.source_revision,
                ),
                "command",
                "ReadProjection",
                {},
            )
        )
        if borrowed_worker == "default":
            await asyncio.shield(receipt.applied)
        else:
            with pytest.raises(ConsoleDisplayPhaseRefused) as refused:
                await asyncio.shield(receipt.applied)
            error = refused.value
            while error.__cause__ is not None:
                error = error.__cause__
            assert str(error) == "Fresh phase found original borrowed custody"
        assert await asyncio.shield(receipt.retired) is True

        def unchanged_borrow():
            assert threading.current_thread() is owner
            assert database._local.conn is connection
            assert database._connection_quiescence.is_registered(connection)
            assert connection.execute("SELECT 1").fetchone()[0] == 1

        await loop.run_in_executor(selected, unchanged_borrow)
    finally:
        if receipt is not None:
            await asyncio.shield(receipt.retired)
        if connection is not None:
            await loop.run_in_executor(selected, database.close_connection)
        # Restore the loop's original executor before runtime shutdown needs it.
        loop._default_executor = previous_default
        try:
            await backend.aclose()
        finally:
            default.shutdown(wait=True)
        assert all(not thread.is_alive() for thread in display._threads)


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "source"])
async def test_display_worker_close_joins_actual_work_despite_failure(request, failure):
    backend = _backend()
    owner = backend.runtime._display_phase_executor
    pool = owner.executor()
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    closing = None

    def held():
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10)

    work = loop.run_in_executor(pool, held)
    try:
        async with asyncio.timeout(10):
            await entered.wait()
        if failure == "source":
            owner.executor = lambda: None
        closing = asyncio.create_task(owner.aclose())
        await asyncio.sleep(0)
        assert owner._closed and not closing.done()
        with pytest.raises(RuntimeError):
            pool.submit(lambda: None)
        if failure == "cancel":
            closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done() and not work.done()
        # The owning loop has progressed while the original worker is held.
        release.set()
        await work
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
            await closing
        assert all(not thread.is_alive() for thread in pool._threads)
        if failure == "source":
            del owner.executor
        await owner.aclose()
    finally:
        release.set()
        await work
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        if "executor" in vars(owner):
            del owner.executor
        await backend.aclose()
