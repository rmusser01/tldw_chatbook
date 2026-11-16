"""Native content invalidation and custody at acquisition-evidence boundaries."""

import os
from textwrap import indent

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_windows_acquisition_evidence import _SETUP
from Tests.Backup_Recovery.test_windows_evidence_content_prerequisites import _ADVERSARIAL


pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows evidence")

_WARM = r"""
import time
time.sleep(storage._EVIDENCE_SETTLE_NS / 1_000_000_000 + .05)
for _ in range(3):
    with storage.acquire_storage(target) as warm:
        assert warm._key == keeper._key
        assert 'profile' in warm.execution_context(target)[1]
assert hold.evidence[str(selector)].confirmed
"""
_WARM_ADVERSARIAL = _ADVERSARIAL.replace(
    "    registry = bootstrap._registry(root)\n",
    indent(_WARM, "    ") + "    registry = bootstrap._registry(root)\n",
    1,
)
assert _WARM_ADVERSARIAL != _ADVERSARIAL


@pytest.mark.parametrize("mutation", ["same-size-content", "new-pending-membership"])
def test_confirmed_evidence_refuses_fresh_content_and_membership_changes(tmp_path, mutation):
    _run(tmp_path, mutation, "approved", script=_WARM_ADVERSARIAL)


_COMMON = _SETUP + r"""
import ctypes
import errno
import inspect
import json
import threading
import time
from contextlib import contextmanager
from types import CodeType
from tldw_chatbook.Utils import windows_files as wf
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout

@contextmanager
def observe(function, event, callback):
    code = inspect.unwrap(function).__code__
    tool = next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
    sys.monitoring.use_tool_id(tool, 'windows-evidence-boundary')
    old_callback = sys.monitoring.register_callback(tool, event, callback)
    old_mask = sys.monitoring.get_local_events(tool, code)
    assert old_mask == 0 and sys.monitoring.get_events(tool) == 0
    try:
        sys.monitoring.set_local_events(tool, code, event)
        yield
    finally:
        sys.monitoring.set_local_events(tool, code, old_mask)
        assert sys.monitoring.register_callback(tool, event, old_callback) is callback
        assert sys.monitoring.get_events(tool) == 0
        sys.monitoring.free_tool_id(tool)

def has_ancestor(code):
    frame = sys._getframe(1)
    try:
        while frame is not None:
            if frame.f_code is code:
                return True
            frame = frame.f_back
        return False
    finally:
        del frame

target = data / 'evidence-lifetime-source.txt'
with storage.acquire_storage(target):
    target.write_text('actual native payload', encoding='utf-8')
before_leases = set(storage._live_leases)
before_pending = set(storage._pending_acquisitions)
keeper = storage.acquire_storage(target)
hold = storage._holds[keeper._key]
assert 'profile' in keeper.execution_context(target)[1]
native = wf._native()
""" + _WARM


def _script(body):
    return (
        "from Tests.windows_private_fixture_runner import user_fixture_default_owner\n"
        "with user_fixture_default_owner():\n" + indent(_COMMON + body, "    ")
    )


_CANCEL = r"""
acquire_code = storage._acquire_storage.__code__
note_code = storage._note_evidence.__code__
original_digest = native.content_digest
cancelled = []
unavailable = []

def limited_digest(handle, limit):
    result = original_digest(handle, limit)
    if route == 'unavailable' and has_ancestor(note_code):
        unavailable.append(True)
        raise OSError(errno.EFBIG, 'controlled optional evidence budget')
    return result

def completed(_code, _offset, _result):
    frame = sys._getframe(1)
    try:
        while frame is not None and frame.f_code is not acquire_code:
            frame = frame.f_back
        assert frame is not None
        attempt = frame.f_locals['attempt']
        assert attempt in storage._pending_acquisitions
        cancelled.append(attempt)
        attempt.cancel.set()
    finally:
        del frame

try:
    bootstrap.advance_admission_epoch()
    native.content_digest = limited_digest
    with observe(storage._note_evidence, sys.monitoring.events.PY_RETURN, completed):
        try:
            with storage.acquire_storage(target):
                raise AssertionError('post-observation cancellation reached payload')
        except bootstrap.RecoveryRequired as error:
            assert str(error) == 'storage_locally_paused', str(error)
    assert len(cancelled) == 1
    assert bool(unavailable) == (route == 'unavailable')
    assert set(storage._live_leases) == before_leases | {keeper}
    assert set(storage._pending_acquisitions) == before_pending
finally:
    native.content_digest = original_digest
    keeper.close()
assert set(storage._live_leases) == before_leases
with authority.maintenance(('profile',), 2):
    pass
assert not blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("observation", ["available", "unavailable"])
def test_full_acquisition_rechecks_cancel_after_optional_evidence(tmp_path, observation):
    _run(tmp_path, observation, "approved", script=_script(_CANCEL))


_EARLY_EOF = r"""
original_read = native.kernel.ReadFile
shortened = []
scope_calls = []
owner_thread = threading.get_ident()

def scope_started(_code, _offset):
    if threading.get_ident() == owner_thread:
        scope_calls.append(True)

def early_eof(handle, buffer, wanted, received, overlapped):
    okay = original_read(handle, buffer, wanted, received, overlapped)
    actual = ctypes.cast(received, ctypes.POINTER(wf._U32))
    if okay and actual.contents.value:
        shortened.append(actual.contents.value)
        actual.contents.value = 0
    return okay

try:
    bootstrap.advance_admission_epoch()
    native.kernel.ReadFile = early_eof
    # Every ordinary acquisition still reads its real complete control records.
    # Repeated short optional reads must never confirm a partial content proof.
    with observe(storage._scope, sys.monitoring.events.PY_START, scope_started):
        for _ in range(3):
            with storage.acquire_storage(target) as lease:
                assert lease._key == keeper._key
                assert 'profile' in lease.execution_context(target)[1]
                assert target.read_text(encoding='utf-8') == 'actual native payload'
    assert shortened
    assert len(scope_calls) == 6, ('short evidence skipped original scope checks', len(scope_calls))
    entry = hold.evidence.get(str(selector))
    assert (entry is None or not entry.confirmed
            or entry.epoch != bootstrap._admission_epoch), 'incomplete digest became reusable evidence'
    assert set(storage._live_leases) == before_leases | {keeper}
    assert set(storage._pending_acquisitions) == before_pending
finally:
    native.kernel.ReadFile = original_read
    keeper.close()
assert set(storage._live_leases) == before_leases
with authority.maintenance(('profile',), 2):
    pass
assert not blocked_attempts()
print('retired and reopened')
"""


def test_incomplete_native_content_read_never_confirms_evidence(tmp_path):
    _run(tmp_path, "early-eof", "approved", script=_script(_EARLY_EOF))


_UNCERTAIN = r"""
phase_code = {
    'candidate': storage._observe_candidates.__code__,
    'reuse': storage._reuse_evidence.__code__,
    'confirmation': storage._note_evidence.__code__,
}[route]
close_code = next(
    code for code in storage.os.stat_many_for_admission.__code__.co_consts
    if isinstance(code, CodeType) and code.co_name == 'close_handle'
)
original_close = native.kernel.CloseHandle
failed = []
attempts = []
retained = None
failure = None

def uncertain_close(handle):
    frame = sys._getframe(1)
    try:
        record = frame.f_locals.get('record') if frame.f_code is close_code else None
        if failed and record is failed[0]:
            attempts.append(record)
        if not failed and record is not None and has_ancestor(phase_code):
            assert type(record) is wf._AdmissionMetadataHandle
            assert record.handle == handle and record.identity is not None
            failed.append(record)
            attempts.append(record)
            if route == 'confirmation':
                raise OSError('controlled metadata close uncertainty')
            ctypes.set_last_error(5)
            return 0
        return original_close(handle)
    finally:
        del frame

try:
    if route != 'reuse':
        bootstrap.advance_admission_epoch()
    native.kernel.CloseHandle = uncertain_close
    try:
        with storage.acquire_storage(target):
            raise AssertionError('uncertain observation reached payload')
    except wf._AdmissionMetadataCloseError as error:
        failure = error
    finally:
        native.kernel.CloseHandle = original_close
    assert failure is not None and len(failed) == 1 and attempts == failed
    assert failure.failed_handles == tuple(failed)
    extra = set(storage._live_leases) - before_leases - {keeper}
    assert len(extra) == 1
    retained = extra.pop()
    assert retained._metadata_close_error is failure
    assert retained._key == keeper._key and not retained.resource_close_failed
    assert set(storage._pending_acquisitions) == before_pending
    assert native.info(failed[0].handle)
    keeper.close()
    assert hold.thread.is_alive() and not hold.stop.is_set()
    try:
        retained.close()
    except wf._AdmissionMetadataCloseError as error:
        assert error is failure
    else:
        raise AssertionError('uncertain observation released its native hold')
    assert retained in storage._live_leases and attempts == failed
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + .02)
        try:
            with authority.maintenance(('profile',), .02):
                raise AssertionError('failed metadata birth lost actual exclusion')
        except AdmissionTimeout:
            pass
    finally:
        pause.resume()
finally:
    native.kernel.CloseHandle = original_close
    if failed:
        # Isolated fixture cleanup only, after proving the product retained the
        # live birth and native exclusion. Product offers no retry/release here.
        assert original_close(failed[0].handle)
    if retained is not None:
        retained._metadata_close_error = None
        retained.close()
    keeper.close()
assert set(storage._live_leases) == before_leases
assert set(storage._pending_acquisitions) == before_pending
assert hold.stop.is_set() and not hold.thread.is_alive()
with authority.maintenance(('profile',), 2):
    pass
assert not blocked_attempts()
print(json.dumps({'phase': route, 'product_retained_exclusion': True,
                  'metadata_birth_and_token_released_only_by_fixture': True}))
print('retired and reopened')
"""


@pytest.mark.parametrize("phase", ["candidate", "reuse", "confirmation"])
def test_uncertain_metadata_close_retains_actual_observation_lease(tmp_path, phase):
    _run(tmp_path, phase, "approved", script=_script(_UNCERTAIN))
