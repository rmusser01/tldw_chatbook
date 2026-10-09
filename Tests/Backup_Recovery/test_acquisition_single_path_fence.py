"""One storage acquisition natively fences its operation path once (TASK-34601).

The bracket checks before and after the lease count keep every in-memory
provenance and lexical-path check; only the repeated native parent/resolution
walk is skipped after this attempt already proved the same path.
"""

import inspect
import sys

import pytest

from Tests.Backup_Recovery import test_participant_lifetimes as repository_fixtures
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.participants import _repository_participant
from tldw_chatbook.Notifications.event_state_repository import EventStateRepository

local_root = repository_fixtures.local_root


@pytest.fixture
def repository(tmp_path):
    owner = EventStateRepository(tmp_path / "events.sqlite")
    try:
        yield owner
    finally:
        owner.close()


class _CountingOS:
    """``storage.os`` proxy recording each native ``stat`` made by the fence.

    Portable: on POSIX ``os.stat`` is a C builtin a profile hook would only see
    as a ``c_call``; here every call through the module's ``os`` is counted.
    """

    def __init__(self, real, native_stats):
        self._real = real
        self._native_stats = native_stats

    def __getattr__(self, name):
        return getattr(self._real, name)

    def stat(self, *args, **kwargs):
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code is storage._Operation.check.__code__:
            self._native_stats.append(args[0] if args else kwargs.get("path"))
        return self._real.stat(*args, **kwargs)


def _observe_checks(monkeypatch, on_return=None):
    """Record (path, fence) per operation check and count native fence stats."""
    calls, native_stats = [], []
    monkeypatch.setattr(storage, "os", _CountingOS(storage.os, native_stats))
    code = storage._Operation.check.__code__

    def profile(frame, event, _arg):
        if frame.f_code is not code:
            return
        if event == "call":
            calls.append((frame.f_locals["path"], frame.f_locals.get("fence", True)))
        elif event == "return" and on_return is not None:
            on_return(calls)

    return calls, native_stats, profile


@pytest.mark.usefixtures("local_root")
def test_nested_acquisition_fences_its_path_once(repository, monkeypatch):
    participant = _repository_participant(repository)
    with participant.operation() as operation, monkeypatch.context() as patch:
        calls, native_stats, profile = _observe_checks(patch)
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            lease = storage.acquire_storage(operation.path)
        finally:
            sys.setprofile(previous)
        lease.close()
    path_checks = [fence for path, fence in calls if path is not None]
    assert len(path_checks) >= 3, calls  # every bracket still checks state
    assert len(native_stats) == 1, (native_stats, calls)  # one native fence


@pytest.mark.usefixtures("local_root")
def test_unfenced_brackets_still_refuse_a_changed_operation_path(
    repository, tmp_path, monkeypatch
):
    participant = _repository_participant(repository)
    original = repository.db_path
    retargeted = []

    def retarget(calls):
        # Only after the single native PATH fence returned: the repository's
        # issued path then changes, and a later state-only bracket must refuse.
        if not retargeted and calls[-1][0] is not None and calls[-1][1] is True:
            retargeted.append(len(calls))
            repository.db_path = tmp_path / "elsewhere.sqlite"

    with participant.operation() as operation, monkeypatch.context() as patch:
        calls, native_stats, profile = _observe_checks(patch, retarget)
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            with pytest.raises(RecoveryRequired, match="operation_path_outside_scope"):
                storage.acquire_storage(operation.path)
        finally:
            sys.setprofile(previous)
            repository.db_path = original
    assert retargeted, "the native path fence never completed"
    # Refused by an in-memory bracket after the fence, never by a second fence.
    assert len(native_stats) == 1, (native_stats, calls)
    assert not storage._pending_acquisitions
