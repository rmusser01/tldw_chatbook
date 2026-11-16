"""Fresh Windows evidence may reuse a settled native acquisition verdict."""

import os
from textwrap import indent

import pytest

from Tests.Backup_Recovery.test_activation_agents import _SCRIPT as _AGENT_SETUP
from Tests.Backup_Recovery.test_home_citation_retirement import _run


pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows evidence")

_SETUP = _AGENT_SETUP.split("effects=[]")[0]
_COUNT = _SETUP + r"""
import inspect
import json
import threading
import time
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout

target = data / 'settled-evidence-source.txt'
with storage.acquire_storage(target):
    target.write_text('original native payload', encoding='utf-8')

before_leases = set(storage._live_leases)
before_pending = set(storage._pending_acquisitions)
keeper = storage.acquire_storage(target)
assert keeper._key is not None
hold = storage._holds[keeper._key]
assert hold.ready.is_set() and not hold.stop.is_set() and hold.error is None
assert storage._EVIDENCE_REUSE
assert storage._EVIDENCE_SETTLE_NS == 1_000_000_000

original_acquire = storage._acquire_storage
acquire_code = original_acquire.__code__
selected = {
    'records': bootstrap._records,
    'registry': bootstrap._registry,
    'scope': storage._scope,
    'observation': storage._Evidence.observe,
    'fresh_batch': storage.os.stat_many_for_admission,
}
codes = {name: inspect.unwrap(function).__code__ for name, function in selected.items()}
assert len(set(codes.values())) == len(codes)
assert all(code.co_filename.replace('\\', '/').endswith(
    '/tldw_chatbook/' + (
        'Utils/windows_files.py' if name == 'fresh_batch'
        else 'Backup_Recovery/bootstrap.py' if name in ('records', 'registry')
        else 'Backup_Recovery/storage_admission.py'
    )
) for name, code in codes.items())
labels = {code: name for name, code in codes.items()}
counts = dict.fromkeys(selected, 0)
observer_errors = []
owner_thread = threading.get_ident()

def started(code, _offset):
    if threading.get_ident() != owner_thread:
        return
    frame = sys._getframe(1)
    try:
        assert frame.f_code is code
        under_acquire = False
        under_observation = False
        depth = 0
        while frame is not None and depth < 64:
            under_acquire |= frame.f_code is acquire_code
            under_observation |= frame.f_code is codes['observation']
            frame = frame.f_back
            depth += 1
        label = labels[code]
        if under_acquire and (label != 'fresh_batch' or under_observation):
            counts[label] += 1
    except Exception as error:
        observer_errors.append(type(error).__name__)
    finally:
        del frame

tool = None
measured = None
try:
    # Preserve the production settling rule. These are ordinary full acquisitions,
    # never fabricated evidence or a test replacement of a product authority.
    time.sleep(storage._EVIDENCE_SETTLE_NS / 1_000_000_000 + .05)
    for _ in range(3):
        with storage.acquire_storage(target) as warm:
            assert warm._key == keeper._key
            warm_root, namespaces = warm.execution_context(target)
            assert warm_root == root and 'profile' in namespaces

    tool = next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
    sys.monitoring.use_tool_id(tool, 'windows-acquisition-evidence')
    previous_callback = sys.monitoring.register_callback(
        tool, sys.monitoring.events.PY_START, started
    )
    previous_masks = {
        code: sys.monitoring.get_local_events(tool, code) for code in codes.values()
    }
    assert previous_callback is None and not any(previous_masks.values())
    assert sys.monitoring.get_events(tool) == 0
    try:
        for code in codes.values():
            sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
        with storage.acquire_storage(target) as measured:
            assert measured._key == keeper._key
            assert storage._holds[measured._key] is hold
            actual_root, namespaces = measured.execution_context(target)
            assert actual_root == root and 'profile' in namespaces
            assert set(storage._pending_acquisitions) == before_pending
            assert target.read_text(encoding='utf-8') == 'original native payload'
            try:
                with authority.maintenance(('profile',), .02):
                    raise AssertionError('warm reuse lost actual native exclusion')
            except AdmissionTimeout:
                pass
    finally:
        for code, mask in previous_masks.items():
            sys.monitoring.set_local_events(tool, code, mask)
        assert sys.monitoring.register_callback(
            tool, sys.monitoring.events.PY_START, previous_callback
        ) is started
        assert sys.monitoring.get_events(tool) == 0
        sys.monitoring.free_tool_id(tool)
        tool = None
finally:
    keeper.close()

assert not observer_errors, observer_errors
assert storage._acquire_storage is original_acquire
assert storage._acquire_storage.__code__ is acquire_code
assert bootstrap._records is selected['records']
assert bootstrap._registry is selected['registry']
assert storage._scope is selected['scope']
assert storage._Evidence.observe is selected['observation']
assert storage.os.stat_many_for_admission is selected['fresh_batch']
assert measured is not None and measured not in storage._live_leases
assert set(storage._live_leases) == before_leases
assert set(storage._pending_acquisitions) == before_pending
assert hold.stop.is_set() and not hold.thread.is_alive()
with authority.maintenance(('profile',), 2):
    pass
assert not blocked_attempts()
print(json.dumps({'counts': counts, 'same_native_hold': True, 'payload': True,
                  'physical_exclusion': True, 'retired': True}))
# Safety and physical retirement are established before the intended RED.
assert {name: counts[name] for name in ('records', 'registry', 'scope')} == {
    'records': 0, 'registry': 0, 'scope': 0,
}, ('settled acquisition re-derived unchanged control evidence', counts)
assert counts['observation'] > 0 and counts['fresh_batch'] > 0, counts
print('retired and reopened')
"""
_COUNT = (
    "from Tests.windows_private_fixture_runner import user_fixture_default_owner\n"
    "with user_fixture_default_owner():\n" + indent(_COUNT, "    ")
)


def test_settled_acquisition_reuses_only_fresh_windows_evidence(tmp_path):
    _run(tmp_path, "reuse", "approved", script=_COUNT)
