"""A finite execution source set owns one actual native acquisition."""

from textwrap import indent

import pytest

from Tests.Backup_Recovery.test_activation_agents import _SCRIPT as _AGENT_SETUP
from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SETUP = _AGENT_SETUP.split("effects=[]")[0]
_COUNT = _SETUP + r"""
import inspect
import json
import threading
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.Backup_Recovery.admission_runtime import RecoveryAdmissionGuard

notes = data / 'execution-notes.db'
workspaces = data / 'execution-workspaces.db'
history = data / 'execution-history'
for selected in (notes, workspaces, history):
    with storage.acquire_storage(selected):
        if selected is history:
            selected.mkdir(mode=0o700)
        else:
            selected.write_bytes(b'private fixture source')
declared = (
    ('config', selector),
    ('db.agent_runs', Path(db.db_path)),
    ('db.chachanotes.primary', notes),
    ('db.workspaces', workspaces),
    ('agents.history', history),
)
assert len({path for _, path in declared}) == 5
assert all(path.exists() for _, path in declared)

class Refused(PermissionError):
    pass

guard = RecoveryAdmissionGuard(
    'execution-source-group-test',
    error=lambda: Refused('execution_group_refused'),
    sources=lambda service: declared,
    owners=lambda owner: ('config', owner),
    check_states=True,
)
original = storage._acquire_storage
original_code = original.__code__
execution = RecoveryAdmissionGuard.execution
execution_code = inspect.unwrap(execution).__code__
owner_thread = threading.get_ident()
acquisitions = 0
observer_errors = []

def started(code, _offset):
    global acquisitions
    if threading.get_ident() != owner_thread:
        return
    frame = sys._getframe(1)
    try:
        assert frame.f_code is original_code
        while frame is not None:
            if frame.f_code is execution_code:
                acquisitions += 1
                break
            frame = frame.f_back
    except Exception as error:
        observer_errors.append(type(error).__name__)
    finally:
        del frame

before_leases = set(storage._live_leases)
before_pending = set(storage._pending_acquisitions)
tool = next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
sys.monitoring.use_tool_id(tool, 'execution-source-groups')
previous = sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, started)
previous_mask = sys.monitoring.get_local_events(tool, original_code)
assert previous_mask == 0 and sys.monitoring.get_events(tool) == 0
owned_leases = ()
payload = history / 'payload.txt'
try:
    sys.monitoring.set_local_events(tool, original_code, sys.monitoring.events.PY_START)
    with guard.execution():
        accepted = guard.context.get()
        owned_leases = tuple(set(accepted.leases.values()))
        assert set(accepted.leases) == {path for _, path in declared}
        for _, path in declared:
            lease = accepted.leases[path]
            assert type(lease) is storage.StorageLease and lease._key is not None
            admitted_root, namespaces = lease.execution_context(path)
            assert admitted_root == root and 'profile' in namespaces
        assert set(storage._pending_acquisitions) == before_pending
        payload.write_text('all declared sources approved', encoding='utf-8')
        try:
            with authority.maintenance(('profile',), .02):
                raise AssertionError('payload lost actual native exclusion')
        except AdmissionTimeout:
            pass
    assert guard.context.get() is None and not accepted.live
finally:
    sys.monitoring.set_local_events(tool, original_code, previous_mask)
    assert sys.monitoring.register_callback(tool, sys.monitoring.events.PY_START, previous) is started
    assert sys.monitoring.get_events(tool) == 0
    sys.monitoring.free_tool_id(tool)

assert storage._acquire_storage is original and original.__code__ is original_code
assert RecoveryAdmissionGuard.execution is execution
assert not observer_errors, observer_errors
assert payload.read_text(encoding='utf-8') == 'all declared sources approved'
assert owned_leases and all(lease not in storage._live_leases for lease in owned_leases)
assert set(storage._live_leases) == before_leases
assert set(storage._pending_acquisitions) == before_pending
with authority.maintenance(('profile',), 2):
    pass
assert not blocked_attempts()
print(json.dumps({'original_acquisitions': acquisitions, 'owned_tokens': len(owned_leases), 'payload': True, 'physical_exclusion': True, 'retired': True}))
# The count is deliberately last: an earlier fixture/ownership failure is not RED.
assert acquisitions == 1, ('declared execution sources reacquired independently', acquisitions)
assert len(owned_leases) == 1
print('retired and reopened')
"""
_COUNT = (
    "from Tests.windows_private_fixture_runner import user_fixture_default_owner\n"
    "with user_fixture_default_owner():\n" + indent(_COUNT, "    ")
)


def test_declared_execution_sources_share_one_original_native_acquisition(tmp_path):
    _run(tmp_path, "count", "approved", script=_COUNT)


_GROUP_SETUP = _SETUP + r"""
import asyncio
import inspect
import threading
import time
from contextlib import contextmanager
from tldw_chatbook.Backup_Recovery import activation
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
from tldw_chatbook.Backup_Recovery.admission_runtime import RecoveryAdmissionGuard

notes = data / 'execution-notes.db'
workspaces = data / 'execution-workspaces.db'
history = data / 'execution-history'
for selected in (notes, workspaces, history):
    with storage.acquire_storage(selected):
        if selected is history:
            selected.mkdir(mode=0o700)
        else:
            selected.write_bytes(b'private fixture source')
declared = (
    ('config', selector),
    ('db.agent_runs', Path(db.db_path)),
    ('db.chachanotes.primary', notes),
    ('db.workspaces', workspaces),
    ('agents.history', history),
)
paths = tuple(path for _, path in declared)

class Refused(PermissionError):
    pass

guard = RecoveryAdmissionGuard(
    'execution-source-group-test',
    error=lambda: Refused('execution_group_refused'),
    sources=lambda service: declared,
    owners=lambda owner: ('config', owner),
    check_states=True,
)
before_leases = set(storage._live_leases)
before_pending = set(storage._pending_acquisitions)
original_acquire = storage._acquire_storage
original_scope = activation.execution_scope
original_close = storage.StorageLease.close
scope_code = inspect.unwrap(original_scope).__code__
assert scope_code.co_name == 'execution_scope'
assert scope_code.co_filename.endswith('activation.py')

@contextmanager
def observe(code, event, callback):
    tool = next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
    sys.monitoring.use_tool_id(tool, 'execution-group-control')
    previous = sys.monitoring.register_callback(tool, event, callback)
    mask = sys.monitoring.get_local_events(tool, code)
    assert mask == 0 and sys.monitoring.get_events(tool) == 0
    try:
        sys.monitoring.set_local_events(tool, code, event)
        yield
    finally:
        sys.monitoring.set_local_events(tool, code, mask)
        assert sys.monitoring.register_callback(tool, event, previous) is callback
        assert sys.monitoring.get_events(tool) == 0
        sys.monitoring.free_tool_id(tool)

def excluded():
    try:
        with authority.maintenance(('profile',), .02):
            raise AssertionError('actual native exclusion was lost')
    except AdmissionTimeout:
        pass

def retired():
    assert storage._acquire_storage is original_acquire
    assert activation.execution_scope is original_scope
    assert storage.StorageLease.close is original_close
    assert set(storage._live_leases) == before_leases
    assert set(storage._pending_acquisitions) == before_pending
    with authority.maintenance(('profile',), 2):
        pass
    assert not blocked_attempts()
    print('retired and reopened')
"""


def _group_script(body):
    return (
        "from Tests.windows_private_fixture_runner import user_fixture_default_owner\n"
        "with user_fixture_default_owner():\n"
        + indent(_GROUP_SETUP + body, "    ")
    )


_MEMBERS = r"""
from Tests.Backup_Recovery.path_aliases import create_directory_alias
attempt = storage._Acquisition()
lease = storage._acquire_execution_storage(paths, attempt)
attempt.close()
assert type(lease._execution_members) is tuple
assert len(lease._execution_members) == len(paths)
try:
    for path in paths:
        admitted_root, namespaces = lease.execution_context(path)
        assert admitted_root == root and 'profile' in namespaces
    excluded()
    if route == 'undeclared':
        changed = data / 'not-declared.db'
    elif route == 'member':
        original_history = data / 'original-history'
        history.rename(original_history)
        changed_history = data / 'changed-history'
        changed_history.mkdir(mode=0o700)
        create_directory_alias(history, changed_history)
        changed = history
    elif route == 'selector':
        changed = notes
        other_selector = selector.parent / 'other-config.toml'
        other_selector.write_bytes(selector.read_bytes())
        other_selector.chmod(0o600)
        os.environ['TLDW_CONFIG_PATH'] = str(other_selector)
    else:
        raise AssertionError(route)
    try:
        lease.execution_context(changed)
    except bootstrap.RecoveryRequired as error:
        assert str(error) == 'execution_selection_changed'
    else:
        raise AssertionError('changed or undeclared member borrowed group authority')
    assert lease in storage._live_leases
    excluded()
finally:
    os.environ['TLDW_CONFIG_PATH'] = str(selector)
    if route == 'member':
        if os.name == 'nt':
            history.rmdir()
        else:
            history.unlink()
        original_history.rename(history)
    lease.close()
for path in paths:
    try:
        lease.execution_context(path)
    except bootstrap.RecoveryRequired:
        pass
    else:
        raise AssertionError('retired group remained borrowable')
retired()
"""


@pytest.mark.parametrize("change", ["undeclared", "member", "selector"])
def test_execution_group_refuses_undeclared_changed_and_retired_members(tmp_path, change):
    _run(tmp_path, change, "approved", script=_group_script(_MEMBERS))


_OUTSIDE = r"""
outside = base / 'outside-group' / 'source.db'
attempt = storage._Acquisition()
try:
    try:
        storage._acquire_execution_storage((*paths, outside), attempt)
    except bootstrap.RecoveryRequired as error:
        assert str(error) == 'storage_scope_not_enrolled'
    else:
        raise AssertionError('unenrolled member admitted')
    assert attempt in storage._pending_acquisitions
    assert not outside.exists() and not outside.parent.exists()
    assert set(storage._live_leases) == before_leases
finally:
    attempt.close()
retired()
"""


def test_execution_group_requires_every_member_to_be_enrolled(tmp_path):
    _run(tmp_path, "outside", "approved", script=_group_script(_OUTSIDE))


_ATTEMPT = r"""
made = []
if route == 'retired':
    attempt = storage._Acquisition()
    attempt.close()
elif route == 'thread':
    worker = threading.Thread(target=lambda: made.append(storage._Acquisition()))
    worker.start()
    worker.join(5)
    assert not worker.is_alive() and len(made) == 1
    attempt = made[0]
elif route == 'task':
    async def make():
        made.append(storage._Acquisition())
    asyncio.run(make())
    assert len(made) == 1
    attempt = made[0]
elif route == 'subclass':
    class OtherAcquisition(storage._Acquisition):
        pass
    attempt = OtherAcquisition()
else:
    raise AssertionError(route)
calls = []
def started(_code, _offset):
    calls.append(threading.get_ident())
try:
    with observe(original_acquire.__code__, sys.monitoring.events.PY_START, started):
        try:
            storage._acquire_execution_storage(paths, attempt)
        except bootstrap.RecoveryRequired as error:
            assert str(error) == 'acquisition_provenance_invalid'
        else:
            raise AssertionError('foreign or retired attempt issued authority')
    assert calls == [], 'invalid attempt reached original native acquisition'
finally:
    attempt.close()
retired()
"""


@pytest.mark.parametrize("kind", ["retired", "thread", "task", "subclass"])
def test_execution_group_requires_exact_live_caller_attempt(tmp_path, kind):
    _run(tmp_path, kind, "approved", script=_group_script(_ATTEMPT))


_RETIRED_ATTEMPT = r"""
from Tests.Backup_Recovery.path_aliases import create_directory_alias
if route == 'selector-after':
    other_selector = selector.parent / 'other-config.toml'
    other_selector.write_bytes(selector.read_bytes())
    other_selector.chmod(0o600)
elif route == 'member-after':
    changed_history = data / 'changed-history'
    changed_history.mkdir(mode=0o700)
    original_history = data / 'original-history'
attempt = storage._Acquisition()
tokens = []
closes = []
def acquired(_code, _offset, value):
    frame = sys._getframe(1)
    if frame.f_locals['attempt'] is attempt:
        assert type(value) is storage.StorageLease and value._key is not None
        assert value._execution_members == ()
        tokens.append(value)
        if route == 'retired-after':
            attempt.close()
        elif route == 'member-after':
            history.rename(original_history)
            create_directory_alias(history, changed_history)
        elif route == 'selector-after':
            os.environ['TLDW_CONFIG_PATH'] = str(other_selector)
        else:
            raise AssertionError(route)
def closed(_code, _offset):
    token = sys._getframe(1).f_locals['self']
    if token in tokens:
        closes.append(token)
try:
    with observe(original_close.__code__, sys.monitoring.events.PY_START, closed):
        with observe(original_acquire.__code__, sys.monitoring.events.PY_RETURN, acquired):
            try:
                storage._acquire_execution_storage(paths, attempt)
            except bootstrap.RecoveryRequired as error:
                assert str(error) == (
                    'acquisition_provenance_invalid' if route == 'retired-after'
                    else 'execution_selection_changed'
                )
            else:
                raise AssertionError('changed source or attempt survived member issuance')
finally:
    attempt.close()
    os.environ['TLDW_CONFIG_PATH'] = str(selector)
    if route == 'member-after' and tokens:
        if os.name == 'nt':
            history.rmdir()
        else:
            history.unlink()
        original_history.rename(history)
assert len(tokens) == 1 and closes == tokens
assert tokens[0]._execution_members == ()
assert tokens[0] not in storage._live_leases
retired()
"""


@pytest.mark.parametrize("change", ["retired-after", "member-after", "selector-after"])
def test_execution_group_rechecks_sources_and_attempt_after_actual_acquisition(tmp_path, change):
    _run(tmp_path, change, "approved", script=_group_script(_RETIRED_ATTEMPT))


_BOUNDARY = r"""
pause = None
approved = []
tokens = []
changed = False
def change():
    global pause, changed
    assert not changed
    pending = set(storage._pending_acquisitions) - before_pending
    assert len(pending) == 1, 'group preflight retired before source approval'
    assert len(set(storage._live_leases) - before_leases) == 1
    excluded()
    if route.startswith('pause'):
        pause = storage._begin_local_pause()
    elif route == 'pending':
        register_pending(root, 'next-restore', ('profile',), control, (selector,))
    elif route == 'pair-loss':
        (root / ('activation-' + bootstrap._key(str(selector)) + '.json')).unlink()
    else:
        raise AssertionError(route)
    changed = True
def approved_source(_code, _offset, value):
    frame = sys._getframe(1)
    path = frame.f_locals['path']
    if path not in paths or value is not True:
        return
    approved.append(path)
    tokens.append(frame.f_locals['retained'])
    if not changed and (
        (route in ('pause-first', 'pending', 'pair-loss') and len(approved) == 1)
        or (route == 'pause-last' and len(approved) == len(paths))
    ):
        change()
def finalize(_observed, _leases):
    assert approved == list(paths)
    if route == 'pause-finalize':
        change()
    return True
guard.finalize = finalize
if route == 'pause-before':
    pause = storage._begin_local_pause()
try:
    with observe(scope_code, sys.monitoring.events.PY_YIELD, approved_source):
        try:
            with guard.execution():
                raise AssertionError('payload crossed a changed admission boundary')
        except Refused:
            pass
    assert guard.context.get() is None
    if route == 'pause-before':
        assert not approved and not changed
    else:
        assert changed and len(set(tokens)) == 1
        assert len(approved) == (len(paths) if route in ('pause-last', 'pause-finalize') else 1)
finally:
    if pause is not None:
        pause.resume()
retired()
"""


@pytest.mark.parametrize(
    "change",
    ["pause-before", "pause-first", "pause-last", "pause-finalize", "pending", "pair-loss"],
)
def test_grouped_source_approvals_recheck_changed_admission(tmp_path, change):
    _run(tmp_path, change, "approved", script=_group_script(_BOUNDARY))


_NESTED = r"""
closed = []
def closing(_code, _offset):
    closed.append(sys._getframe(1).f_locals['self'])
outer_members = (declared[2], declared[4])
with observe(original_close.__code__, sys.monitoring.events.PY_START, closing):
    with guard.execution(resolve=lambda service: outer_members):
        outer = guard.context.get()
        outer_tokens = set(outer.leases.values())
        assert len(outer_tokens) == 1
        with guard.execution():
            inner = guard.context.get()
            assert inner.parent is outer
            assert all(inner.leases[path] is outer.leases[path] for _, path in outer_members)
            inner_tokens = set(inner.leases.values()) - outer_tokens
            assert len(inner_tokens) == 1
            assert set(storage._pending_acquisitions) == before_pending
            excluded()
        assert not inner.live and guard.context.get() is outer and outer.live
        assert all(token not in storage._live_leases for token in inner_tokens)
        assert all(token in storage._live_leases for token in outer_tokens)
        errors = []
        transfers = []
        def work():
            try:
                with guard.execution(transfer=outer, resolve=lambda service: outer_members):
                    child = guard.context.get()
                    assert child.parent is outer
                    assert set(child.leases.values()) == outer_tokens
                    assert set(storage._pending_acquisitions) == before_pending
                    (history / 'worker-payload.txt').write_text('borrowed', encoding='utf-8')
                    transfers.append(child)
                assert guard.context.get() is None
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=work)
        worker.start()
        worker.join(10)
        assert not worker.is_alive() and not errors, errors
        assert len(transfers) == 1 and not transfers[0].live
        assert (history / 'worker-payload.txt').read_text(encoding='utf-8') == 'borrowed'
        assert all(token in storage._live_leases for token in outer_tokens)
        assert all(closed.count(token) == 0 for token in outer_tokens)
        excluded()
    assert not outer.live
assert all(closed.count(token) == 1 for token in outer_tokens | inner_tokens)
try:
    with guard.execution(transfer=outer):
        raise AssertionError('retired parent allowed a transferred execution')
except Refused:
    pass
retired()
"""


def test_nested_and_transferred_groups_borrow_and_retire_only_owned_tokens(tmp_path):
    _run(tmp_path, "nested-transfer", "approved", script=_group_script(_NESTED))


_REFUSAL = r"""
seen = []
approved = []
tokens = []
closed = []
def admit(_stack, owner, path, lease):
    seen.append((owner, path))
    tokens.append(lease)
    admitted_root, namespaces = lease.execution_context(path)
    assert admitted_root == root and 'profile' in namespaces
    assert len(set(storage._pending_acquisitions) - before_pending) == 1
    if route == 'owner' and len(seen) == 3:
        raise RuntimeError('controlled owner refusal')
    return False
def finalize(observed, leases):
    assert observed == declared and set(leases) == set(paths)
    assert approved == list(paths)
    if route == 'finalize-error':
        raise RuntimeError('controlled finalize refusal')
    assert route == 'finalize-false'
    return False
def yielded(_code, _offset, value):
    if value is True:
        approved.append(sys._getframe(1).f_locals['path'])
def closing(_code, _offset):
    token = sys._getframe(1).f_locals['self']
    if token in tokens:
        closed.append(token)
guard.admit_source = admit
guard.finalize = finalize
with observe(original_close.__code__, sys.monitoring.events.PY_START, closing):
    with observe(scope_code, sys.monitoring.events.PY_YIELD, yielded):
        try:
            with guard.execution():
                raise AssertionError('refused source/finalize reached payload')
        except Refused:
            pass
assert seen == list(declared[:3] if route == 'owner' else declared)
assert approved == list(paths[:2] if route == 'owner' else paths)
assert len(set(tokens)) == 1 and closed == [tokens[0]]
assert guard.context.get() is None
retired()
"""


@pytest.mark.parametrize("phase", ["owner", "finalize-false", "finalize-error"])
def test_failed_group_approval_or_finalize_retires_one_actual_token(tmp_path, phase):
    _run(tmp_path, phase, "approved", script=_group_script(_REFUSAL))


_UNCERTAIN = r"""
from tldw_chatbook.Backup_Recovery import admission
native = admission.os
native_close = native.close
lock_code = inspect.unwrap(admission.Admission._lock).__code__
target = []
attempted = []
holds = []
owned = []
def locked(_code, _offset, descriptor):
    frame = sys._getframe(1)
    if (
        frame.f_locals['self'].control_root == authority.control_root
        and frame.f_locals['name'] == admission.Admission._key('profile', 'lease')
        and frame.f_locals['mode'] == admission.fcntl.LOCK_SH
    ):
        target.append(descriptor)
def uncertain_close(descriptor):
    if descriptor in target:
        attempted.append(descriptor)
        raise OSError('controlled original native lease close uncertainty')
    return native_close(descriptor)
assert not storage._holds and not storage._retiring_holds
try:
    with observe(lock_code, sys.monitoring.events.PY_YIELD, locked):
        native.close = uncertain_close
        with guard.execution():
            accepted = guard.context.get()
            owned = list(set(accepted.leases.values()))
            assert len(owned) == 1
            holds.append(storage._holds[owned[0]._key])
            assert set(storage._pending_acquisitions) == before_pending
            (history / 'uncertain-payload.txt').write_text('completed', encoding='utf-8')
finally:
    native.close = native_close
assert len(target) == 1 and attempted == target
assert len(holds) == 1 and isinstance(holds[0].error, OSError)
assert holds[0] in storage._retiring_holds and not holds[0].thread.is_alive()
assert not storage._holds and not accepted.live
assert set(storage._live_leases) == before_leases
assert set(storage._pending_acquisitions) == before_pending
assert native.fstat(target[0])
assert (history / 'uncertain-payload.txt').read_text(encoding='utf-8') == 'completed'
pause = storage._begin_local_pause()
try:
    assert not pause.drain(time.monotonic() + .02)
    excluded()
finally:
    pause.resume()
    # Explicit fixture cleanup only, after physical exclusion and the retained
    # coordinator blocker are proved. The failed hold remains until child exit.
    native_close(target[0])
assert holds[0] in storage._retiring_holds
retired()
"""


def test_group_native_close_uncertainty_keeps_exclusion_and_retirement_blocker(tmp_path):
    _run(tmp_path, "uncertain-close", "approved", script=_group_script(_UNCERTAIN))
