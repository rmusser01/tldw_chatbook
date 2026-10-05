"""Evidence-only native ownership controls; root installs and runs serially."""

import json

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


pytestmark = pytest.mark.bootstrap_profile


_SCRIPT = r"""
import json, os, shutil, sqlite3, sys, threading, time
from io import BufferedRandom
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route = sys.argv[1]


def physically_closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return True
    return False


def main():
    root = Path(os.environ['XDG_DATA_HOME']).absolute()
    selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    selector.write_text('[general]\nusers_name="factory-owner-control"\n[paths]\ndata_dir="'
                        + root.as_posix() + '"\n', encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook import config, app_service_wiring
    from tldw_chatbook.Backup_Recovery import participants, storage_admission as storage
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from Tests.UI import app_factory
    from tldw_chatbook import app as app_module
    from tldw_chatbook.Utils.instance_lock import InstanceLockStatus, acquire_profile_instance_lock

    borrowed = WorkspaceDB(config.get_user_data_dir() / 'borrowed-factory-control.sqlite')
    borrowed_connection = borrowed._held_connection()
    with storage._lock:
        startup_before = dict(storage._startups)
    assert startup_before and all(vars(lease).get('_key') is not None
                                  for lease in startup_before.values())
    borrowed_lock_directory = config.get_user_data_dir() / 'borrowed-instance-control'
    borrowed_lock_directory.mkdir(mode=0o700)
    borrowed_lock = acquire_profile_instance_lock(borrowed_lock_directory)
    assert type(borrowed_lock) is InstanceLockStatus and borrowed_lock.acquired
    assert type(borrowed_lock.handle) is BufferedRandom and not borrowed_lock.handle.closed
    borrowed_collection = None
    constructor_patch = None
    lock_constructor_patch = None
    if route == 'borrowed_constructor':
        borrowed_collection = LibraryCollectionsDB(config.get_user_data_dir() / 'borrowed-collections.sqlite')
        borrowed_collection_connection = borrowed_collection._held_connection()
        constructor_patch = patch.object(app_service_wiring, 'LibraryCollectionsDB',
                                        return_value=borrowed_collection)
        constructor_patch.start()
    if route == 'borrowed_lock_constructor':
        lock_constructor_patch = patch.object(app_module, 'acquire_profile_instance_lock',
                                             return_value=borrowed_lock)
        lock_constructor_patch.start()
    try:
        app = app_factory._build_test_app()
    finally:
        if constructor_patch is not None:
            constructor_patch.stop()
        if lock_constructor_patch is not None:
            lock_constructor_patch.stop()
    owned_directory = app_factory._created_dirs[-1]
    assert owned_directory.is_absolute() and owned_directory.exists()
    assert borrowed.db_path.parent != owned_directory
    with storage._lock:
        assert storage._startups == startup_before, 'App added a new startup owner; ownership setup unqualified'
        assert all(not Path(lease._key[1]).is_relative_to(owned_directory)
                   for lease in startup_before.values()), 'startup owner belongs to owned sandbox'
    owned_lock = app._instance_lock_status
    assert type(owned_lock) is InstanceLockStatus
    if route == 'borrowed_lock_constructor':
        assert owned_lock is borrowed_lock
        original_owned_lock_handle = None
    else:
        original_owned_lock_handle = owned_lock.handle
        assert type(original_owned_lock_handle) is BufferedRandom
        assert Path(original_owned_lock_handle.name) == owned_directory / '.instance.lock'
        assert not original_owned_lock_handle.closed
    if route == 'replaced_lock_field':
        app._instance_lock_status = borrowed_lock
    owners = [app.local_workspace_db, app.subscriptions_db]
    assert type(owners[0]) is WorkspaceDB and type(owners[1]) is SubscriptionsDB
    if route != 'borrowed_constructor':
        assert type(app.local_library_collections_db) is LibraryCollectionsDB
        owners.append(app.local_library_collections_db)
    original_owners = tuple(owners)
    connections, leases = [], []
    with storage._lock:
        for owner in original_owners:
            assert owner.db_path.parent == owned_directory
            participant = owner._maintenance_participant
            assert participant.repository() is owner and participant.path == owner.db_path
            actual = tuple(participant.connections.items())
            assert len(actual) == 1
            connection, lease = actual[0]
            assert lease.resource_thread is threading.current_thread()
            assert lease in storage._live_leases
            assert not physically_closed(connection)
            connections.append(connection)
            leases.append(lease)
    if route == 'replaced_field':
        app.local_workspace_db = borrowed
    worker = None
    try:
        if route == 'closed_empty':
            # Establish the exact original accepted closed-owner state; do not
            # clear participant tables or fake a successful native close.
            for owner in original_owners:
                participant = participants._repository_participant(owner)
                participant.close_admission()
                type(owner).close(owner)
                assert participant.drain(time.monotonic() + 1.0)
                assert participant.closed and not participant.connections
            assert all(physically_closed(item) for item in connections)
        if route == 'foreign_worker':
            worker = ThreadPoolExecutor(max_workers=1)
            foreign_connection = worker.submit(original_owners[0]._held_connection).result(timeout=5)
            assert not physically_closed(foreign_connection)
            try:
                app_factory.drain_created_dirs()
            except RuntimeError as error:
                assert str(error) in {'test_factory_database_not_retired',
                                      'test_factory_directory_has_live_storage'}
            else:
                raise AssertionError('factory deleted directory despite exact foreign live handle')
            assert owned_directory.exists() and owned_directory in app_factory._created_dirs
            assert not physically_closed(foreign_connection)
            worker.submit(original_owners[0].close).result(timeout=5)
            assert physically_closed(foreign_connection)
        # Keep all original app/owner/connection/lease objects strongly alive.
        # No finalizer or registry-count-only retirement can satisfy this check.
        observed_at_delete = []
        original_rmtree = app_factory.shutil.rmtree

        def observed_delete(path, *args, **kwargs):
            if Path(path) == owned_directory:
                observed_at_delete.append(
                    all(physically_closed(item) for item in connections)
                    and (original_owned_lock_handle is None or original_owned_lock_handle.closed))
            return original_rmtree(path, *args, **kwargs)

        with patch.object(app_factory.shutil, 'rmtree', observed_delete):
            app_factory.drain_created_dirs()
        assert observed_at_delete == [True], 'directory removal began with original native database/lock handles open'
        assert all(physically_closed(item) for item in connections), 'original constructor handles remain open'
        with storage._lock:
            assert all(lease not in storage._live_leases for lease in leases)
            assert storage._startups == startup_before
            assert all(lease in storage._live_leases for lease in startup_before.values())
        assert not physically_closed(borrowed_connection), 'borrowed owner was closed'
        assert not borrowed_lock.handle.closed, 'borrowed advisory lock was closed'
        assert original_owned_lock_handle is None or original_owned_lock_handle.closed
        if borrowed_collection is not None:
            assert not physically_closed(borrowed_collection_connection), 'borrowed constructor owner was closed'
        assert not owned_directory.exists()
        receipt = {'route': route, 'physical_owned_handles_closed': len(connections),
                   'closed_before_directory_removal': True, 'borrowed_owner_live': True,
                   'exact_startup_owner_retained': True, 'borrowed_advisory_lock_live': True,
                   'exact_owned_advisory_lock_closed': original_owned_lock_handle is None or original_owned_lock_handle.closed,
                   'app_strongly_retained': app is not None}
        (selector.parent.parent / 'factory-owned-retirement.json').write_text(
            json.dumps(receipt), encoding='utf-8')
    finally:
        if worker is not None:
            worker.shutdown(wait=True)
        # Cleanup belongs only to these explicitly-created control owners.
        for owner in original_owners:
            type(owner).close(owner)
        assert all(physically_closed(item) for item in connections)
        if original_owned_lock_handle is not None:
            assert Path(original_owned_lock_handle.name) == owned_directory / '.instance.lock'
            BufferedRandom.close(original_owned_lock_handle)
            assert original_owned_lock_handle.closed
        app_factory.drain_active_service_patches()
        if owned_directory.exists():
            assert all(owner.db_path.parent == owned_directory for owner in original_owners)
            shutil.rmtree(owned_directory)
        if owned_directory in app_factory._created_dirs:
            app_factory._created_dirs.remove(owned_directory)
        if hasattr(app_factory, '_created_databases'):
            app_factory._created_databases.pop(owned_directory, None)
        if hasattr(app_factory, '_created_instance_locks'):
            app_factory._created_instance_locks.pop(owned_directory, None)
        borrowed.close()
        if borrowed_collection is not None:
            borrowed_collection.close()
        borrowed_lock.handle.close()


with user_fixture_default_owner():
    main()
print('retired and reopened')  # Existing runner's unchanged successful-child marker.
"""


@pytest.mark.parametrize(
    "route",
    [
        "retained",
        "replaced_field",
        "borrowed_constructor",
        "closed_empty",
        "foreign_worker",
        "replaced_lock_field",
        "borrowed_lock_constructor",
    ],
)
def test_factory_owns_physical_constructor_retirement_before_directory_removal(
    tmp_path, route
):
    _run(tmp_path, route, "ownership", script=_SCRIPT)
    receipt = json.loads(
        (tmp_path / "factory-owned-retirement.json").read_text(encoding="utf-8")
    )
    assert receipt["route"] == route
    assert receipt["closed_before_directory_removal"]
    assert receipt["borrowed_owner_live"] and receipt["exact_startup_owner_retained"]
    assert (
        receipt["borrowed_advisory_lock_live"]
        and receipt["exact_owned_advisory_lock_closed"]
    )
