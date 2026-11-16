"""One execution entry shares facts without reusing another owner's decision."""

from textwrap import indent

import pytest

from Tests.Backup_Recovery.test_activation_agents import _SCRIPT as _AGENT_SETUP
from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SETUP = _AGENT_SETUP.split("effects=[]")[0]
_SCRIPT = (
    _SETUP
    + r"""
import inspect
import json
import threading
import time
from tldw_chatbook.Backup_Recovery import activation
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout

path = db.db_path
lease = storage.acquire_storage(path)
scope = activation.execution_scope
scope_code = inspect.unwrap(scope).__code__
context_code = storage.StorageLease.execution_context.__code__
readers = {
    bootstrap._control_records: 'controls',
    bootstrap._registry: 'registry',
}
codes = {function.__code__: name for function, name in readers.items()}
allowed_reader = activation.ActivationStore.allowed
allowed_code = allowed_reader.__code__
startup_reader = bootstrap.startup_permission
startup_code = startup_reader.__code__
custom_calls = []

def refuse_startup(config_selector, bootstrap_root, /):
    return False, 'custom_startup_refusal'

custom_code = refuse_startup.__code__
counts = {
    'gate': {'controls': 0, 'registry': 0},
    'lease_context': {'controls': 0, 'registry': 0},
}
owner_thread = threading.get_ident()
mutating = False
mutated = False
approvals = 0
tool = next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
sys.monitoring.use_tool_id(tool, 'execution-activation-reads')

def ancestry():
    frame = sys._getframe(2)
    result = set()
    try:
        while frame is not None:
            result.add(frame.f_code)
            frame = frame.f_back
        return result
    finally:
        del frame

def started(code, _offset):
    if threading.get_ident() != owner_thread or mutating:
        return
    parents = ancestry()
    if scope_code in parents:
        if code is custom_code:
            frame = sys._getframe(1)
            custom_calls.append((
                frame.f_locals['config_selector'] == selector,
                frame.f_locals['bootstrap_root'] == root,
            ))
            return
        category = 'lease_context' if context_code in parents else 'gate'
        counts[category][codes[code]] += 1

def returned(code, _offset, value):
    global mutating, mutated, approvals
    if threading.get_ident() != owner_thread or scope_code not in ancestry():
        return
    approvals += 1
    if route not in ('pending', 'pair_loss') or mutated or value is not True:
        return
    mutating = True
    try:
        if route == 'pending':
            register_pending(root, 'next-restore', ('profile',), control, (selector,))
        elif route == 'pair_loss':
            (root / ('activation-' + bootstrap._key(str(selector)) + '.json')).unlink()
        else:
            raise AssertionError(route)
        mutated = True
    finally:
        mutating = False

try:
    sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
    sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, returned)
    for code in codes:
        sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_START)
    sys.monitoring.set_local_events(tool, custom_code, sys.monitoring.events.PY_START)
    sys.monitoring.set_local_events(tool, allowed_code, sys.monitoring.events.PY_RETURN)
    try:
        if route == 'custom_binding':
            bootstrap.startup_permission = refuse_startup
        elif route == 'custom_code':
            startup_reader.__code__ = custom_code
        owners = ('',) if route == 'invalid_owner' else ('config', 'db.agent_runs')
        began = time.perf_counter()
        with scope(owners, path, retained=lease) as allowed:
            permitted = allowed
        elapsed = time.perf_counter() - began
    finally:
        bootstrap.startup_permission = startup_reader
        startup_reader.__code__ = startup_code
        for code in (*codes, allowed_code, custom_code):
            sys.monitoring.set_local_events(tool, code, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, None)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, None)
        sys.monitoring.free_tool_id(tool)

    assert activation.execution_scope is scope
    assert activation.ActivationStore.allowed is allowed_reader
    assert all(getattr(bootstrap, function.__name__) is function for function in readers)
    assert lease.execution_context(path)[0] == root, 'borrower retired its caller lease'
    try:
        with authority.maintenance(('profile',), .02):
            raise AssertionError('execution lost native exclusion')
    except AdmissionTimeout:
        pass
finally:
    lease.close()

assert lease not in storage._live_leases
with authority.maintenance(('profile',), 2):
    pass
assert not blocked_attempts()
print(json.dumps({'counts': counts, 'gate_seconds': elapsed, 'approvals': approvals}))
if route == 'count':
    assert permitted, 'original execution gate refused an approved fixture'
    assert counts['gate'] == {'controls': 3, 'registry': 2}, counts
elif route in ('custom_binding', 'custom_code', 'invalid_owner'):
    assert not permitted and approvals == 0
    assert counts['gate'] == {'controls': 1, 'registry': 0}, counts
    assert custom_calls == ([] if route == 'invalid_owner' else [(True, True)])
else:
    assert mutated and approvals == 1, 'mutation must follow only the first approval'
    assert not permitted, 'next owner reused facts from before pending or paired-history loss'
print('retired and reopened')
"""
)
_SCRIPT = (
    "from Tests.windows_private_fixture_runner import user_fixture_default_owner\n"
    "with user_fixture_default_owner():\n" + indent(_SCRIPT, "    ")
)


@pytest.mark.parametrize("state", ["ordinary", "approved"])
def test_execution_reads_one_control_set_per_owner(tmp_path, state):
    _run(tmp_path, "count", state, script=_SCRIPT)


@pytest.mark.parametrize("change", ["pending", "pair_loss"])
def test_execution_rereads_controls_for_the_next_owner(tmp_path, change):
    _run(tmp_path, change, "approved", script=_SCRIPT)


@pytest.mark.parametrize("replacement", ["binding", "code"])
def test_execution_preserves_custom_startup_refusal(tmp_path, replacement):
    _run(tmp_path, "custom_" + replacement, "ordinary", script=_SCRIPT)


def test_execution_refuses_invalid_owner_before_activation_reads(tmp_path):
    _run(tmp_path, "invalid_owner", "ordinary", script=_SCRIPT)
