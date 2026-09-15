"""Actual model store execution observes paired restoration activation."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio
import os
from pathlib import Path
import sys
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.activation import bind_activation, ActivationStore
from tldw_chatbook.Backup_Recovery.control_records import admission_authority, register_pending
route, state = sys.argv[1:]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
data = selector.parent.parent / 'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="' + str(data) + '"\n')
selector.chmod(0o600)
from Tests.Model_Artifacts.test_service import descriptor, artifact_file
from tldw_chatbook.Model_Artifacts.service import ModelArtifactService, ArtifactStateError
from tldw_chatbook.Model_Artifacts.acquisition import ArtifactAcquisitionService, AcquisitionError
store = ModelArtifactService(data / 'custom-models')
payload = b'small local model fixture'
item = descriptor(files=(artifact_file(payload),))
source = data / 'source'
source.mkdir()
(source / item.files[0].path).write_bytes(payload)
store.install(item, source)
store.activate(item.reference)
root = bootstrap.default_bootstrap_root()
# This source-only process has finished config startup before simulating restore.
from tldw_chatbook.Backup_Recovery import storage_admission as storage
storage._startups.pop((os.getpid(), str(root))).close()
authority = admission_authority(root)
authority.register('profile', (selector.parent, data))
control = selector.parent.parent / 'operation'
control.mkdir(mode=0o700)
if state != 'ordinary':
    register_pending(root, 'restore', ('profile',), control, (selector,))
    with authority.maintenance(('profile',), 2) as session:
        bind_activation(root, 'restore', selector, 'generation',
                        ('config', 'models.artifacts'), session=session)
    (root / ('pending-' + bootstrap._key('restore') + '.json')).unlink()
    if state in ('approved', 'shared'):
        for owner in ('config', 'models.artifacts'):
            ActivationStore(control / 'activation').approve('generation', owner)
    elif state == 'config_only':
        ActivationStore(control / 'activation').approve('generation', 'config')
    if state == 'shared':
        other = selector.parent.parent / 'other.toml'
        other.write_text('[general]\nusers_name="other"\n')
        other.chmod(0o600)
        os.environ['TLDW_CONFIG_PATH'] = str(other)

assert store.list_installed(), 'local inspection must remain available'
before = {str(p.relative_to(store._root)): p.read_bytes()
          for p in store._root.rglob('*') if p.is_file()}
effects = []
def prevent_execution(event, arguments):
    if event in ('subprocess.Popen', 'os.posix_spawn', 'os.system'):
        effects.append('process')
        raise AssertionError('unexpected model process')
sys.addaudithook(prevent_execution)
denied = state not in ('ordinary', 'approved')
if route in ('preflight', 'provision'):
    from dataclasses import replace
    from tldw_chatbook.Model_Artifacts import ArtifactRef
    missing = replace(item, reference=ArtifactRef('uninstalled', 'revision', 'int8'))
    class Catalog:
        def descriptor(self, reference):
            assert reference == missing.reference
            return missing
    class BoundaryReached(Exception): pass
    def client_factory():
        effects.append('network-client')
        raise BoundaryReached()
    acq = ArtifactAcquisitionService(store, client_factory=client_factory,
                                     free_bytes_probe=lambda path: 10**12)
    catalog = Catalog()
    consent = acq._aggregate_closure(missing.reference, catalog)[1].grant()
    async def call():
        if route == 'preflight':
            return await acq.preflight(missing.reference, catalog)
        return await acq.provision(missing.reference, consent, catalog)
    try:
        asyncio.run(call())
    except AcquisitionError:
        assert denied
    except BoundaryReached:
        assert not denied
    else:
        raise AssertionError('expected acquisition boundary')
else:
    try:
        if route == 'install':
            value = store.install(item, source)
        else:
            value = getattr(store, route)(item.reference)
    except ArtifactStateError:
        assert denied
    else:
        if route.startswith('acquire'):
            assert value.handle is not None
            value.close()
        assert not denied, 'inactive model execution was accepted'
if denied:
    assert not effects, effects
    after = {str(p.relative_to(store._root)): p.read_bytes()
             for p in store._root.rglob('*') if p.is_file()}
    assert after == before, 'denial changed artifact state'
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "route",
    [
        "install",
        "activate",
        "acquire",
        "acquire_installed_root",
        "preflight",
        "provision",
    ],
)
@pytest.mark.parametrize(
    "state", ["inactive", "ordinary", "approved", "shared", "config_only"]
)
def test_model_execution_uses_actual_root_and_paired_approval(tmp_path, route, state):
    _run(tmp_path, route, state, script=_SCRIPT)


@pytest.mark.parametrize(
    "route",
    [
        "install",
        "activate",
        "acquire",
        "acquire_installed_root",
        "preflight",
        "provision",
    ],
)
@pytest.mark.parametrize("state", ["ordinary", "inactive", "approved", "shared"])
def test_native_unqualified_execution_requires_positively_ordinary_state(
    tmp_path, route, state
):
    script = _SCRIPT.replace(
        "assert store.list_installed()",
        "storage.qualified_for = lambda *args: (False, 'native_unqualified')\nassert store.list_installed()",
    ).replace(
        "denied = state not in ('ordinary', 'approved')", "denied = state != 'ordinary'"
    )
    _run(tmp_path, route, state, script=script)


_RETENTION_SCRIPT = (
    _SCRIPT.split("assert store.list_installed")[0]
    + r"""
import threading
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout

def assert_held():
    try:
        with authority.maintenance(('profile',), .03):
            raise AssertionError('live model work lost native admission')
    except AdmissionTimeout:
        pass

if route == 'handle':
    handle = store.acquire_installed_root(item.reference)
    try:
        assert_held()
        assert handle.handle is not None
    finally:
        handle.close()
elif route == 'provision_cancel':
    from dataclasses import replace
    from tldw_chatbook.Model_Artifacts import ArtifactRef
    entered = threading.Event()
    release = threading.Event()
    original = store._download_stage_for
    def blocked(*args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            return original(*args, **kwargs)
        entered.set()
        assert release.wait(3), 'bounded test release missing'
        return original(*args, **kwargs)
    missing = replace(item, reference=ArtifactRef('uninstalled', 'revision', 'int8'))
    class Catalog:
        def descriptor(self, reference):
            assert reference == missing.reference
            return missing
    def network_sentry():
        raise AssertionError('cancelled native staging reached network')
    acq = ArtifactAcquisitionService(store, client_factory=network_sentry,
                                     free_bytes_probe=lambda path: 10**12)
    catalog = Catalog()
    consent = acq._aggregate_closure(missing.reference, catalog)[1].grant()
    store._download_stage_for = blocked
    async def main():
        waiter = asyncio.create_task(acq.provision(missing.reference, consent, catalog))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            waiter.cancel()
            await asyncio.sleep(0)
            assert not waiter.done(), 'native worker must settle before cancellation'
            assert_held()
        finally:
            release.set()
            try:
                await asyncio.wait_for(waiter, 2)
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('cancellation was lost')
    asyncio.run(main())
else:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = store._copy_payload
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3), 'bounded test release missing'
        return original(*args, **kwargs)
    store._copy_payload = blocked
    errors = []
    def native():
        try:
            store.install(item, source)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()
    async def main():
        waiter = asyncio.create_task(asyncio.to_thread(native))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            waiter.cancel()
            try:
                await waiter
            except asyncio.CancelledError:
                pass
            assert not finished.is_set()
            assert_held()
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 2)
    asyncio.run(main())
    assert not errors, errors
with authority.maintenance(('profile',), 1):
    pass
assert store.list_installed()
assert (source / item.files[0].path).read_bytes() == payload
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["handle", "native_cancel", "provision_cancel"])
def test_approved_model_native_admission_survives_waiter_and_handle_lifetimes(
    tmp_path, route
):
    _run(tmp_path, route, "approved", script=_RETENTION_SCRIPT)


_PROOF_SCRIPT = (
    _SCRIPT.split("assert store.list_installed")[0]
    + r"""
from tldw_chatbook.Backup_Recovery.activation import execution_scope
owners = ('config', 'models.artifacts')
lease = storage.acquire_storage(store._root)
pause = storage._begin_local_pause()
try:
    with execution_scope(owners, store._root, retained=lease) as allowed:
        assert allowed
    assert lease.execution_scope()[0] == root, 'borrower closed original token'
    with execution_scope(owners, data / 'different', retained=lease) as allowed:
        assert not allowed
    fake = storage.StorageLease(None)
    try:
        with execution_scope(owners, store._root, retained=fake) as allowed:
            assert not allowed, 'unminted token became ordinary proof'
    finally:
        fake.close()
    original_pid = os.getpid
    try:
        storage.os.getpid = lambda: original_pid() + 1
        with execution_scope(owners, store._root, retained=lease) as allowed:
            assert not allowed
    finally:
        storage.os.getpid = original_pid
    other = selector.parent.parent / 'different.toml'
    other.write_text('[general]\n')
    try:
        os.environ['TLDW_CONFIG_PATH'] = str(other)
        with execution_scope(owners, store._root, retained=lease) as allowed:
            assert not allowed
    finally:
        os.environ['TLDW_CONFIG_PATH'] = str(selector)
finally:
    pause.resume()
    lease.close()
with execution_scope(owners, store._root, retained=lease) as allowed:
    assert not allowed, 'retired token authorized execution'
print('retired and reopened')
"""
)


def test_retained_execution_requires_exact_live_acquisition_selection(tmp_path):
    _run(tmp_path, "proof", "approved", script=_PROOF_SCRIPT)
