"""Cold visual imports must not break unrelated private native readers."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_COLD_IMPORT = r"""
import importlib
import os
from pathlib import Path
import sys
import threading
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Utils import private_paths

name = 'tldw_chatbook.Backup_Recovery.' + sys.argv[1]
assert name not in sys.modules
entered = threading.Event()
release = threading.Event()
failures = []
def barrier(frame, event, arg):
    if (event == 'call' and frame.f_code.co_name == '<module>'
            and frame.f_globals.get('__name__') == name):
        entered.set()
        assert release.wait(10), 'cold import release timed out'
    return barrier
def importing():
    sys.settrace(barrier)
    try:
        importlib.import_module(name)
    except BaseException as error:
        failures.append(error)
    finally:
        sys.settrace(None)

root = Path.home()
payload = root / 'ordinary.txt'
payload.write_bytes(b'ordinary private bytes')
control = root / 'empty-bootstrap'
control.mkdir(mode=0o700)
worker = threading.Thread(target=importing)
worker.start()
observed = []
try:
    assert entered.wait(10), 'cold import was not reached'
    partial = sys.modules[name]
    assert '_local' not in vars(partial)
    def trace_exception(frame, event, arg):
        if (event == 'exception'
                and frame.f_code is private_paths._active_visual_source.__code__):
            observed.append((arg[0].__name__, str(arg[1])))
        return trace_exception
    sys.settrace(trace_exception)
    try:
        permission = bootstrap.startup_permission(
            root / 'config.toml', control)
    finally:
        sys.settrace(None)
    # Run the real descriptor reader too: startup_permission deliberately maps
    # its caught exception to a bounded reason, whereas this exposes the cause.
    try:
        with bootstrap.pinned_directory(root) as parent:
            native = os.stat(payload.name, dir_fd=parent).st_ino == payload.stat().st_ino
    except AttributeError as error:
        native = (type(error).__name__, str(error))
    assert permission == (True, 'startup_allowed') and native is True, (
        permission, native, observed)
    assert not observed
finally:
    release.set()
    worker.join(10)
assert not worker.is_alive() and not failures, failures
assert hasattr(sys.modules[name], '_local')
with bootstrap.pinned_directory(root) as parent:
    assert os.stat(payload.name, dir_fd=parent).st_ino == payload.stat().st_ino
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "module", ["persona_visual_participants", "visual_identity_participants"]
)
def test_private_readers_survive_actual_visual_cold_import(tmp_path, module):
    _run(tmp_path, module, "cold", script=_COLD_IMPORT)
