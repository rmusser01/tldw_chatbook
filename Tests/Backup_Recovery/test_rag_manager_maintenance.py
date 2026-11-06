"""Live manager visibility and experiment intake use real private sources."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import gc, sys, threading, weakref
from pathlib import Path
from types import SimpleNamespace
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Backup_Recovery.rag_definition_participant import participant, retained_issues
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ExperimentConfig, get_profile_manager, reset_profile_manager_cache
route = sys.argv[1]
manager = get_profile_manager()
app = SimpleNamespace(_rag_service=None)
def refused(call):
    try:
        call()
    except RecoveryRequired:
        return
    except TypeError as error:
        raise AssertionError('paused operation entered original serialization') from error
    raise AssertionError('operation did not refuse')
if route == 'census':
    alternate = get_profile_manager(Path.home()/'alternate')
    assert any(i.reason == 'alternate_source_unqualified' for i in retained_issues(app)), 'uncached manager invisible'
    reset_profile_manager_cache()
    newer = get_profile_manager()
    assert {manager, newer, alternate} <= set(ConfigProfileManager.live_instances())
    reference = weakref.ref(alternate)
    del alternate
    gc.collect()
    assert reference() is None, 'census retains owner'
    assert not retained_issues(app)
elif route == 'constructor':
    participant._maintenance_close_admission()
    destination = Path.home()/'not-created'
    try:
        refused(lambda: ConfigProfileManager(destination))
        assert not destination.exists()
    finally:
        participant._maintenance_resume()
    reopened = ConfigProfileManager(destination)
    assert destination.is_dir()
elif route == 'active':
    experiment = ExperimentConfig(save_results=False)
    manager._current_experiment = experiment
    manager._experiment_results[experiment.experiment_id] = []
    refused(participant._maintenance_close_admission)
    manager.record_experiment_result('hybrid_basic', 'private query', {'search_latency': 1})
    assert manager._experiment_results[experiment.experiment_id][0]['query'] == 'private query'
    assert manager.end_experiment()['total_queries'] == 1
    participant._maintenance_close_admission()
    participant._maintenance_resume()
elif route == 'transition':
    # No model service constructor: exact real methods on its initialized
    # experiment-only fields exercise the wrapper without opening engines.
    from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
    service = object.__new__(EnhancedRAGServiceV2)
    service.profile_manager = manager
    service._current_experiment = None
    entered, release = threading.Event(), threading.Event()
    original = manager.start_experiment
    def blocked(config):
        entered.set()
        assert release.wait(5)
        return original(config)
    manager.start_experiment = blocked
    errors = []
    def run():
        try:
            service.start_experiment(ExperimentConfig(save_results=False))
        except TypeError as error:
            errors.append(error)  # Existing Path JSON serialization failure.
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5)
        refused(participant._maintenance_close_admission)
    finally:
        release.set()
        worker.join(5)
        participant._maintenance_resume()
    assert not worker.is_alive()
    assert len(errors) == 1
    refused(participant._maintenance_close_admission)
    service.end_experiment()
    participant._maintenance_close_admission()
    participant._maintenance_resume()
elif route == 'service_refusal':
    from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
    service = object.__new__(EnhancedRAGServiceV2)
    service.profile_manager = manager
    service._current_experiment = None
    participant._maintenance_close_admission()
    try:
        refused(lambda: service.start_experiment(ExperimentConfig()))
        assert service._current_experiment is None
        assert manager._current_experiment is None
    finally:
        participant._maintenance_resume()
elif route == 'nested_refusal':
    @participant.operation
    def accepted():
        participant._maintenance_close_admission()
        try:
            refused(lambda: manager.start_experiment(ExperimentConfig()))
            assert manager._current_experiment is None
        finally:
            participant._maintenance_resume()
    accepted()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    [
        "census",
        "constructor",
        "active",
        "transition",
        "service_refusal",
        "nested_refusal",
    ],
)
def test_manager_maintenance_boundaries(tmp_path, route):
    _run(tmp_path, route, "success", script=_SCRIPT)


_CLEANUP_SCRIPT = r"""
import asyncio, inspect, sys

async def join_owned(task):
    outcomes = await asyncio.gather(task, return_exceptions=True)
    for error in outcomes:
        if isinstance(error, BaseException) and not isinstance(error, asyncio.CancelledError):
            raise error

async def cleanup(primary, actions):
    errors = []
    for action in actions:
        try:
            result = action()
            if inspect.isawaitable(result):
                await result
        except BaseException as error:
            errors.append(error)
    if errors:
        target = primary if primary is not None else errors[0]
        for error in errors:
            if error is not target:
                target.add_note('cleanup_error:' + type(error).__name__)
        if primary is None:
            raise target
"""


_SQL_POOL_SCRIPT = (
    _CLEANUP_SCRIPT
    + r"""
import asyncio, sqlite3, threading, time
from pathlib import Path
from Tests import network_guard
network_guard.install()
import sys
for name in ('sounddevice', 'pyaudio'):
    sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.runtime_maintenance import RuntimeMaintenance
from Tests.RAG_Search.test_keyword_leg_db_resolution import _make_service
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Backup_Recovery.rag_definition_participant import participant
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import retire_current_thread_caches
from tldw_chatbook.RAG_Search.simplified import db_connection_pool as pools

async def main():
    app = TldwCli()
    app.console_runtime.ensure_chat_store()
    await app.console_runtime._attention_refresh_task
    path = Path.home() / 'media.db'
    seed = MediaDatabase(db_path=str(path), client_id='idle-pool-control')
    seed.add_media_with_keywords(title='Numbat Notes', media_type='document',
                                content='numbat native pool maintenance', keywords=['numbat'])
    seed.close_connection()
    service = _make_service(Path.home(), media_db_path=path)
    runtime = None
    producer = None
    token = None
    token_released = False
    try:
        rows = await service._keyword_search('numbat', top_k=5)
        assert rows, 'real media search never reached its native rows'
        assert str(path) in pools._connection_pools
        pool = pools._connection_pools[str(path)]
        executor = pools.pool_executor(pool)
        loop = asyncio.get_running_loop()
        connection = await loop.run_in_executor(executor, pool._get_thread_connection)
        owner = await loop.run_in_executor(executor, threading.current_thread)
        assert owner is not threading.current_thread()
        sqlite3.Connection.in_transaction.__get__(connection)
        deadline = time.monotonic()+30
        if sys.argv[1] in ('accepted_projection', 'cancel_before_start'):
            from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant as projection
            token = projection.reserve()
            async def accepted_later():
                nonlocal token_released
                try:
                    with projection.accepted(token):
                        while not projection._closed:
                            assert time.monotonic() < deadline, 'later producer stage never reached'
                            await asyncio.sleep(.01)
                        assert not pools.maintenance._closed
                        assert pools._connection_pools[str(path)] is pool
                        sqlite3.Connection.in_transaction.__get__(connection)
                        assert await service._keyword_search('numbat', top_k=5)
                finally:
                    projection.release(token)
                    token_released = True
            producer = asyncio.create_task(accepted_later())
            if sys.argv[1] == 'cancel_before_start':
                raise RuntimeError('controlled before producer first step')
        runtime = RuntimeMaintenance(app)
        await runtime.settle_producers(deadline)
        if producer is not None:
            await producer
        runtime.retire_local_caches()
        assert runtime.pause.drain(time.monotonic()+.02), 'completed RAG pool still holds native storage'
        with_closed = False
        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            with_closed = True
        assert with_closed, 'native pool handle was not physically retired'
        await runtime.resume()
        runtime = None
        assert await service._keyword_search('numbat', top_k=5)
        assert pools._connection_pools[str(path)] is not pool
    finally:
        primary = sys.exception()
        actions = []
        if producer is not None:
            if not producer.done():
                producer.cancel()
            actions.append(lambda: join_owned(producer))
        def release_reservation():
            if token is not None and not token_released:
                projection.release(token)
        actions.append(release_reservation)
        if runtime is not None:
            actions.append(runtime.resume)
        actions.extend((service.close, app._shutdown_app_owned_lifecycles))
        await cleanup(primary, actions)
        if sys.argv[1] == 'cancel_before_start':
            assert token not in projection._tokens
            import pytest
            with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
                sqlite3.Connection.in_transaction.__get__(connection)
try:
    asyncio.run(main())
except RuntimeError as error:
    assert sys.argv[1] == 'cancel_before_start'
    assert str(error) == 'controlled before producer first step'
    assert not getattr(error, '__notes__', [])
print('retired and reopened')
"""
)


@pytest.mark.parametrize(
    "route", ["idle_keyword_pool", "accepted_projection", "cancel_before_start"]
)
def test_completed_keyword_pool_retires_before_native_storage_pause(tmp_path, route):
    """A real App settles its actual producers before the local native-lease drain."""
    _run(tmp_path, route, "success", script=_SQL_POOL_SCRIPT)


_POOL_LIFECYCLE_SCRIPT = (
    _CLEANUP_SCRIPT
    + r"""
import asyncio, sqlite3, threading, time
from pathlib import Path
from Tests import network_guard
network_guard.install()
import sys
for name in ('sounddevice', 'pyaudio'):
    sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery.runtime_maintenance import RuntimeMaintenance
from Tests.RAG_Search.test_keyword_leg_db_resolution import _make_service
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Backup_Recovery.rag_definition_participant import participant
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import retire_current_thread_caches
from tldw_chatbook.RAG_Search.simplified import db_connection_pool as pools

import pytest

async def main():
    mode = sys.argv[1]
    path = Path.home() / 'lifecycle-media.db'
    seed = MediaDatabase(db_path=str(path), client_id='pool-lifecycle-control')
    seed.add_media_with_keywords(title='Numbat Notes', media_type='document',
                                content='numbat pool lifecycle', keywords=['numbat'])
    seed.close_connection()
    service = _make_service(Path.home(), media_db_path=path)
    release = threading.Event()
    waiter = resume = None
    pool = original_close = None
    try:
        assert await service._keyword_search('numbat', top_k=5)
        pool = pools._connection_pools[str(path)]
        executor = pools.pool_executor(pool)
        loop = asyncio.get_running_loop()
        connection = await loop.run_in_executor(executor, pool._get_thread_connection)
        entered = asyncio.Event()
        original_close = pool.close_connection
        def controlled_close():
            loop.call_soon_threadsafe(entered.set)
            if mode in ('deadline', 'cancel_waiter'):
                assert release.wait(5), 'native close was never released'
            elif mode == 'close_error':
                raise RuntimeError('controlled pool close failure')
            else:
                return None
            return original_close()
        pool.close_connection = controlled_close
        pools.maintenance._maintenance_close_admission()
        deadline = time.monotonic() + (.02 if mode == 'deadline' else 2)
        waiter = asyncio.create_task(pools.maintenance._maintenance_drain(deadline))
        await asyncio.wait_for(entered.wait(), 2)
        if mode == 'cancel_waiter':
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            waiter = None
        elif mode == 'deadline':
            done, _ = await asyncio.wait({waiter}, timeout=.2)
            assert waiter in done and waiter.result() is False
        else:
            with pytest.raises(RuntimeError, match='close failure|close refused'):
                await waiter
        waiter = None
        assert pools._connection_pools[str(path)] is pool
        assert pools.pool_executor(pool) is executor
        sqlite3.Connection.in_transaction.__get__(connection)
        with pytest.raises(RuntimeError, match='paused'):
            pools.get_connection_pool(str(path))
        pool.close_connection = original_close
        resume = asyncio.create_task(pools.maintenance._maintenance_resume())
        if mode in ('deadline', 'cancel_waiter'):
            await asyncio.sleep(.01)
            assert not resume.done() and pools.maintenance._closed
        release.set()
        await resume
        if mode in ('close_error', 'close_refusal'):
            assert pools.get_connection_pool(str(path)) is pool
            assert await service._keyword_search('numbat', top_k=5) == []
            assert pools._connection_pools[str(path)] is pool
            pools.close_all_pools()
        with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
            sqlite3.Connection.in_transaction.__get__(connection)
        assert await service._keyword_search('numbat', top_k=5)
        replacement = pools._connection_pools[str(path)]
        assert replacement is not pool
        replacement_handle = await loop.run_in_executor(pools.pool_executor(replacement), replacement._get_thread_connection)
        pools.close_all_pools()
        with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
            sqlite3.Connection.in_transaction.__get__(replacement_handle)
    finally:
        release.set()
        if pool is not None and original_close is not None:
            pool.close_connection = original_close
        actions = []
        if waiter is not None:
            actions.append(lambda: join_owned(waiter))
        if resume is not None:
            actions.append(lambda: join_owned(resume))
        actions.extend((pools.maintenance._maintenance_resume, service.close))
        await cleanup(sys.exception(), actions)
asyncio.run(main())
print('retired and reopened')
"""
)


@pytest.mark.parametrize(
    "route", ["deadline", "cancel_waiter", "close_error", "close_refusal"]
)
def test_pool_retirement_retains_native_ownership_and_reopens(tmp_path, route):
    _run(tmp_path, route, "success", script=_POOL_LIFECYCLE_SCRIPT)


_POOL_CONSTRUCTION_SCRIPT = (
    _CLEANUP_SCRIPT
    + r"""
import asyncio, sqlite3, threading, time
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.RAG_Search.simplified import db_connection_pool as pools

async def main():
    path = str(Path.home() / 'constructing-media.db')
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    actual_database = pools.MediaDatabase
    native = []
    builder = resume = None
    def held_constructor(*args, **kwargs):
        db = actual_database(*args, **kwargs)
        native.append(db._get_thread_connection())
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), 'native constructor was never released'
        return db
    pools.MediaDatabase = held_constructor
    try:
        builder = asyncio.create_task(asyncio.to_thread(pools.get_connection_pool, path))
        await asyncio.wait_for(entered.wait(), 2)
        assert not pools._connection_pools
        assert pools._connection_pools_lock.locked()
        sqlite3.Connection.in_transaction.__get__(native[0])
        pools.maintenance._maintenance_close_admission()
        assert await pools.maintenance._maintenance_drain(time.monotonic() + .02) is False
        assert pools.maintenance._retirement is not None
        resume = asyncio.create_task(pools.maintenance._maintenance_resume())
        await asyncio.sleep(.01)
        assert not resume.done() and pools.maintenance._closed
        release.set()
        await builder
        await resume
        assert not pools._connection_pools
        import pytest
        with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
            sqlite3.Connection.in_transaction.__get__(native[0])
        replacement = await asyncio.to_thread(pools.get_connection_pool, path)
        assert replacement is not None
    finally:
        release.set()
        pools.MediaDatabase = actual_database
        actions = []
        if builder is not None:
            actions.append(lambda: join_owned(builder))
        if resume is not None:
            actions.append(lambda: join_owned(resume))
        actions.extend((pools.maintenance._maintenance_resume,
                        lambda: asyncio.to_thread(pools.close_all_pools)))
        await cleanup(sys.exception(), actions)
asyncio.run(main())
print('retired and reopened')
"""
)


def test_pool_retirement_waits_for_unpublished_native_constructor(tmp_path):
    _run(tmp_path, "constructing", "success", script=_POOL_CONSTRUCTION_SCRIPT)


def test_pool_control_cleanup_attempts_all_owners_and_preserves_primary(tmp_path):
    script = (
        _CLEANUP_SCRIPT
        + r"""
async def main():
    seen = []
    def first():
        seen.append('first')
        raise RuntimeError('private cleanup detail')
    async def last():
        seen.append('last')
    primary = AssertionError('original assertion')
    await cleanup(primary, [first, last])
    assert seen == ['first', 'last']
    assert primary.__notes__ == ['cleanup_error:RuntimeError']
    seen.clear()
    try:
        await cleanup(None, [first, last])
    except RuntimeError:
        pass
    else:
        raise AssertionError('unreported cleanup failure')
    assert seen == ['first', 'last']
asyncio.run(main())
print('retired and reopened')
"""
    )
    _run(tmp_path, "cleanup", "success", script=script)
