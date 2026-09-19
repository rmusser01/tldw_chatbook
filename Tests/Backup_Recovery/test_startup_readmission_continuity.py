"""Native startup continuity; runtime coverage issuance is tested separately."""

import asyncio
import json
import os
import threading
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery.test_bootstrap import local_scope  # noqa: F401
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import bind_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_native_startup_completion_waits_without_timer_polling(
    local_scope, monkeypatch, cancel  # noqa: F811 - imported pytest fixture.
):
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    errors, sleeps = [], []
    original_sleep = asyncio.sleep

    async def observed_sleep(delay, *args, **kwargs):
        sleeps.append(delay)
        return await original_sleep(delay, *args, **kwargs)

    with retired_bound_startup(local_scope) as (pause, _, _, authority):
        def hold_native_gate():
            try:
                with authority.maintenance(("profile",), 5):
                    loop.call_soon_threadsafe(entered.set)
                    assert release.wait(5)
            except BaseException as error:  # noqa: BLE001 - relay worker failure to test.
                errors.append(error)
                loop.call_soon_threadsafe(entered.set)

        worker = threading.Thread(target=hold_native_gate)
        worker.start()
        try:
            async with asyncio.timeout(5):
                await entered.wait()
            assert not errors
            monkeypatch.setattr(asyncio, "sleep", observed_sleep)
            loop.call_later(0.08, release.set)
            if cancel:
                loop.call_later(0.02, asyncio.current_task().cancel)
                with pytest.raises(asyncio.CancelledError):
                    await pause.reacquire_startup()
            else:
                await pause.reacquire_startup()
            assert release.is_set() and not pause._startup_retired
            assert pause._startup_source[0] in storage._startups
            assert not sleeps, "native completion must not depend on timer polling"
            assert not errors
        finally:
            release.set()
            worker.join(5)
            if pause._startup_thread is not None:
                pause._startup_thread.join(5)


@contextmanager
def retired_bound_startup(scope):
    root, config, data, authority = scope
    bind_profile(root, config, ("profile",), root / "admission")
    storage.admit_startup()
    key = (os.getpid(), str(root))
    lease = storage._startups[key]
    hold = storage._holds[lease._key]
    pause = storage._begin_local_pause()
    # Set up the post-retirement state using actual admitted scope, not a fake
    # authority or a mocked _scope. App coverage issuance has separate tests.
    pause._startup_source = (key, config, hold.names, hold.authority._identity)
    pause._startup_roots = tuple(bootstrap._records(root)[1][0]["roots"])
    pause._startup_retired = True
    pause._startup_thread = None
    pause._startup_error = None
    storage._startups.pop(key).close()
    try:
        yield pause, config, data, authority
    finally:
        _cleanup(pause, key)


def _cleanup(pause, key):
    lease = storage._startups.pop(key, None)
    if lease is not None:
        lease.close()
    # A refused unit-level readmission intentionally leaves the local gate shut.
    pause._startup_retired = False
    if storage._pause is pause:
        pause.resume()


@pytest.mark.asyncio
async def test_config_edit_keeps_native_bound_startup(local_scope):  # noqa: F811
    with retired_bound_startup(local_scope) as (pause, config, data, _):
        before = bootstrap._records(bootstrap.default_bootstrap_root())[1]
        config.write_text("scope1\n# ordinary preference save\n")
        await pause.reacquire_startup()
        key = pause._startup_source[0]
        assert storage._holds[storage._startups[key]._key].names == ("profile",)
        assert bootstrap._records(bootstrap.default_bootstrap_root())[1] == before
        pause.resume()
        with storage.acquire_storage(data / "allowed.sqlite"):
            pass
        with pytest.raises(
            bootstrap.RecoveryRequired, match="storage_scope_not_enrolled"
        ):
            storage.acquire_storage(data.parent / "outside.sqlite")


@pytest.mark.asyncio
async def test_changed_mapping_refuses_readmission(local_scope):  # noqa: F811
    with retired_bound_startup(local_scope) as (pause, config, data, _):
        root = bootstrap.default_bootstrap_root()
        record_path = next(root.glob("profile-*.json"))
        record = json.loads(record_path.read_text())
        record["roots"].append(str(data.parent / "other"))
        record_path.write_text(json.dumps(record))
        config.write_text("scope1\n# ordinary preference save\n")
        with pytest.raises(
            bootstrap.RecoveryRequired, match="startup_reacquisition_failed"
        ):
            await pause.reacquire_startup()
        assert pause._startup_source[0] not in storage._startups
