"""Native startup continuity; runtime coverage issuance is tested separately."""

import json
import os
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery.test_bootstrap import local_scope  # noqa: F401
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import bind_profile


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
