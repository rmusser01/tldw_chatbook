"""Each activation consumer reads its own fresh controls once."""

from textwrap import indent

import pytest

from Tests.Backup_Recovery.test_activation_agents import _SCRIPT as _AGENT_SETUP
from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SETUP = _AGENT_SETUP.split("effects=[]")[0]
_COUNT = (
    _SETUP
    + r"""
import errno
import inspect
import json
import threading
from tldw_chatbook.Backup_Recovery import activation, generation_witnesses
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.Backup_Recovery.control_records import UNBOUND_NAMESPACE

path = db.db_path
before_leases = set(storage._live_leases)
before_pending = set(storage._pending_acquisitions)
lease = storage.acquire_storage(path)
native_root, names = lease.execution_context(path)
assert native_root == root
assert (UNBOUND_NAMESPACE if state == 'ordinary' else 'profile') in names
hold = storage._holds[lease._key]
assert hold.ready.is_set() and not hold.stop.is_set() and hold.error is None
profiles = bootstrap._control_records(root)[1]
expected = [row['activation'] for row in profiles if row['selector'] == str(selector)]
assert bool(expected) == (state == 'approved')
reader = activation.execution_scope if route == 'execution' else generation_witnesses._paired_witnesses
reader_code = inspect.unwrap(reader).__code__
context_code = storage.StorageLease.execution_context.__code__
originals = (bootstrap._control_records, bootstrap._registry)
labels = {function.__code__: label for function, label in zip(originals, ('controls', 'registry'), strict=True)}
counts = dict.fromkeys(labels.values(), 0)
owner_thread = threading.get_ident()
errors = []

def started(code, _offset):
    if threading.get_ident() != owner_thread:
        return
    frame = sys._getframe(1)
    try:
        ancestors = set()
        while frame is not None:
            ancestors.add(frame.f_code)
            frame = frame.f_back
        if reader_code in ancestors and context_code not in ancestors:
            counts[labels[code]] += 1
    except Exception as error:
        errors.append(type(error).__name__)
    finally:
        del frame

tool = next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
sys.monitoring.use_tool_id(tool, 'activation-control-observation')
old_callback = sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
old_masks = {code: sys.monitoring.get_local_events(tool, code) for code in labels}
assert old_callback is None and not any(old_masks.values())
assert sys.monitoring.get_events(tool) == 0
try:
    try:
        for code in labels:
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        if route == 'execution':
            with reader(('config', 'db.agent_runs'), path, retained=lease) as allowed:
                assert allowed, 'approved original execution refused'
                with path.open('rb') as payload:
                    descriptor = payload.fileno()
                    assert payload.read(16) == b'SQLite format 3\x00'
                assert payload.closed
                try:
                    os.fstat(descriptor)
                except OSError as error:
                    assert error.errno == errno.EBADF
                else:
                    raise AssertionError('payload descriptor not retired')
        else:
            assert reader(path, root, names, selector) == expected
            with path.open('rb') as payload:
                assert payload.read(16) == b'SQLite format 3\x00'
            assert payload.closed
    finally:
        for code, mask in old_masks.items():
            sys.monitoring.set_local_events(tool, code, mask)
        assert sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, old_callback) is started
        assert sys.monitoring.get_events(tool) == 0
        sys.monitoring.free_tool_id(tool)
    assert not errors, errors
    assert (bootstrap._control_records, bootstrap._registry) == originals
    assert lease.execution_context(path) == (native_root, names)
    assert set(storage._live_leases) == before_leases | {lease}
    assert set(storage._pending_acquisitions) == before_pending
    try:
        with authority.maintenance(names, .02):
            raise AssertionError('consumer lost actual native exclusion')
    except AdmissionTimeout:
        pass
finally:
    lease.close()
assert set(storage._live_leases) == before_leases
assert set(storage._pending_acquisitions) == before_pending
assert hold.stop.is_set() and not hold.thread.is_alive() and hold.error is None
with authority.maintenance(names, 2):
    pass
assert not blocked_attempts()
print(json.dumps({'consumer': route, 'state': state, 'counts': counts,
                  'payload_and_physical_retirement': True}))
# Last: the original source must fail only this duplicate-read oracle.
expected_counts = {'controls': 3, 'registry': 2} if route == 'execution' else {'controls': 1, 'registry': 1}
assert counts == expected_counts, (route, state, counts)
print('retired and reopened')
"""
)
_COUNT = (
    "from Tests.windows_private_fixture_runner import user_fixture_default_owner\n"
    "with user_fixture_default_owner():\n" + indent(_COUNT, "    ")
)


@pytest.mark.parametrize("state", ["ordinary", "approved"])
def test_execution_assembles_fresh_controls_once_per_owner(tmp_path, state):
    _run(tmp_path, "execution", state, script=_COUNT)


@pytest.mark.parametrize("state", ["ordinary", "approved"])
def test_paired_witnesses_assemble_fresh_controls_once(tmp_path, state):
    _run(tmp_path, "witness", state, script=_COUNT)
