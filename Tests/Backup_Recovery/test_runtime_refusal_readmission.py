"""Refused runtime attempts wait for actual native gates before reopening owners."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import asyncio, sys, threading
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.app import TldwCli
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.runtime_maintenance import RuntimeMaintenance, _bind
from tldw_chatbook.Chat.console_image_edit_operations import ImageEditOperationRegistry
from tldw_chatbook.Backup_Recovery.admission import AdmissionError
pause_mode, outcome = sys.argv[1:]
db = WorkspaceDB(Path.home() / 'workspace.db')
registry = ImageEditOperationRegistry()
app = object.__new__(TldwCli)
stop, finished = threading.Event(), threading.Event()
with storage._lock:
    hold = next(iter(storage._holds.values()))
def capturing():
    try:
        with hold.authority.maintenance(hold.names, 10, cancel=stop):
            raise AssertionError('retained native connection was ignored')
    except (InterruptedError, AdmissionError):
        pass
    finally:
        finished.set()
thread = threading.Thread(target=capturing)
thread.start()
async def main():
    async with asyncio.timeout(5):
        while not storage._local_pause_requested(): await asyncio.sleep(.01)
    ready = asyncio.Event()
    async def resume():
        runtime = RuntimeMaintenance(app)
        hook = _bind(registry, 'Chat.console_image_edit_operations', 'ImageEditOperationRegistry')
        hook.close(hook.owner)
        runtime.closed.append(hook)
        if pause_mode == 'local_pause':
            runtime.pause = storage._begin_local_pause()
        ready.set()
        await runtime.resume()
    waiting = asyncio.create_task(resume())
    await ready.wait()
    try:
        await asyncio.sleep(.1)
        assert not waiting.done(), 'refused runtime resumed while native gate remained closed'
        assert registry._maintenance_closed
        if outcome == 'cancel':
            waiting.cancel()
            await asyncio.sleep(.05)
            assert not waiting.done(), 'cancelled waiter abandoned native readmission'
            assert registry._maintenance_closed
        stop.set()
        try:
            await asyncio.wait_for(asyncio.shield(waiting), 5)
        except asyncio.CancelledError:
            assert outcome == 'cancel'
        assert not registry._maintenance_closed
        assert storage._pause is None
        assert db._held_connection().execute('SELECT 1').fetchone()[0] == 1
    finally:
        stop.set()
        await asyncio.to_thread(thread.join, 5)
        await asyncio.gather(waiting, return_exceptions=True)
        if storage._pause is not None:
            raise AssertionError('local refusal pause was abandoned')
        db.close()
asyncio.run(main())
assert finished.is_set() and not blocked_attempts()
print('retired and reopened')
'''


@pytest.mark.parametrize("pause_mode", ("before_pause", "local_pause"))
@pytest.mark.parametrize("outcome", ("normal", "cancel"))
def test_refusal_waits_for_native_readmission(tmp_path, pause_mode, outcome):
    _run(tmp_path, pause_mode, outcome, script=_SCRIPT)

_MULTI = r'''
import os, threading, time
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.admission import Admission, AdmissionCancelled
from tldw_chatbook.Backup_Recovery.storage_admission import _Hold
from tldw_chatbook.Backup_Recovery.runtime_maintenance import _readmit_native_holds
holds, cancellations, captures = [], [], []
errors = []
finished = threading.Event()
def capture(hold, cancel):
    try:
        with hold.authority.maintenance(hold.names, 10, cancel=cancel):
            errors.append('existing native holder was bypassed')
    except AdmissionCancelled:
        pass
def readmit():
    try:
        _readmit_native_holds(tuple(holds))
    except BaseException as error:
        errors.append(error)
    finally:
        finished.set()
try:
    for index in range(2):
        source = Path.home() / ('source-' + str(index))
        source.mkdir(mode=0o700)
        authority = Admission(Path.home() / ('authority-' + str(index)))
        authority.register('source', (source,))
        hold = _Hold(authority, ('source',), (os.getpid(), str(authority.control_root)))
        holds.append(hold)
        assert hold.ready.wait(5) and hold.error is None
        cancel = threading.Event()
        cancellations.append(cancel)
        thread = threading.Thread(target=capture, args=(hold, cancel))
        captures.append(thread)
        thread.start()
    deadline = time.monotonic() + 5
    while not all(h.authority.pause_requested(h.names) for h in holds):
        assert time.monotonic() < deadline
        time.sleep(.01)
    waiting = threading.Thread(target=readmit)
    waiting.start()
    assert not finished.wait(.1)
    cancellations[0].set()
    captures[0].join(5)
    assert not captures[0].is_alive()
    assert not finished.wait(.1), errors
    cancellations[1].set()
    assert finished.wait(5)
    waiting.join(5)
    assert not errors, errors
finally:
    for cancel in cancellations: cancel.set()
    for thread in captures: thread.join(5)
    for hold in holds:
        hold.stop.set()
        hold.thread.join(5)
assert not blocked_attempts()
print('retired and reopened')
'''


def test_readmission_waits_for_two_actual_native_authorities(tmp_path):
    _run(tmp_path, "two_authorities", "normal", script=_MULTI)
