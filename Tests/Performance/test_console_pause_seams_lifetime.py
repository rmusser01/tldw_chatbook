"""Pure supported-monitoring regressions; no App, SQL or native handles."""

import collections
import gc
import os
import sys
import threading
from types import ModuleType
import weakref

import pytest

from Tests.Performance.console_native_pause_seams import PassivePauseProbeSeams


class _Observation:
    def __init__(self):
        self.phase = "pure"
        self.main = threading.get_ident()
        self.lock = threading.Lock()
        self.rows = collections.defaultdict(lambda: [0, 0.0, 0.0])
        self.slow = []

    def stack(self, frame):
        return "pure-source"

    def record(self, label, phase, elapsed, caller):
        self.rows[(phase, "main", label, caller)][0] += 1

    def config_record(self, *args):
        raise AssertionError("this lifetime control must not read config")


_SOURCES = {
    "config": """
def _config_file_posture(config_path): pass
def _settings_cache_hit(active_config_path): pass
def load_settings(force_reload=False): pass
def _invalidate_config_caches(): pass
def set_encryption_password(password): pass
def _set_session_encryption_password(password): pass
""",
    "Backup_Recovery.config_participants": """
from contextlib import contextmanager
@contextmanager
def operation(source, *, fail=False):
    if fail:
        raise RuntimeError('failed original generator entry')
    yield source
""",
    "Backup_Recovery.storage_admission": """
from contextlib import contextmanager
def _acquire_storage(path=None, *, fail=False):
    if fail:
        raise RuntimeError('failed original synchronous function')
def _scope(): pass
def _local_pause_requested(): pass
def _observe_candidates(): pass
def _reuse_evidence(): pass
class _Acquisition:
    @contextmanager
    def initializing(self, root, path):
        yield None
""",
    "DB.private_sqlite_process": """
class HelperLease:
    @classmethod
    def start(cls): pass
""",
    "Utils.windows_files": """
class _Native:
    def open_handle(self, path):
        raise RuntimeError('failed original selected receiver')
    def security(self): pass
    def ntfs(self): pass
    def _token_sid(self): pass
""",
}


class _Owner:
    pass


def _exceptional_argument(storage):
    owner = _Owner()
    reference = weakref.ref(owner)
    try:
        storage._acquire_storage(owner, fail=True)
    except RuntimeError:
        pass
    else:
        pytest.fail("the original synchronous function did not fail")
    return reference


def _exceptional_generator(life):
    owner = _Owner()
    reference = weakref.ref(owner)
    try:
        with life.operation(owner, fail=True):
            pytest.fail("the original failed-entry generator yielded")
    except RuntimeError:
        pass
    return reference


def _exceptional_receiver(windows):
    receiver, argument = windows._Native(), _Owner()
    references = weakref.ref(receiver), weakref.ref(argument)
    try:
        receiver.open_handle(argument)
    except RuntimeError:
        pass
    else:
        pytest.fail("the original receiver did not fail")
    return references


def test_passive_seams_release_exceptional_frames_and_context_owners(monkeypatch):
    """Unknown exits cannot retain owners while the observer is still active."""
    modules = {}
    for suffix, source in _SOURCES.items():
        name = "tldw_chatbook." + suffix
        module = ModuleType(name)
        module.__file__ = "<pure-source:" + name + ">"
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        monkeypatch.setitem(sys.modules, name, module)
        modules[suffix] = module
    monkeypatch.delitem(
        sys.modules, "tldw_chatbook.Utils.sensitive_paths", raising=False
    )
    observed = _Observation()
    observer = PassivePauseProbeSeams(observed)
    try:
        observer.start()
        assert sys.monitoring.get_events(observer.tool) == 0
        storage = modules["Backup_Recovery.storage_admission"]
        life = modules["Backup_Recovery.config_participants"]
        argument = _exceptional_argument(storage)
        source = _exceptional_generator(life)
        receivers = (
            _exceptional_receiver(modules["Utils.windows_files"])
            if os.name == "nt"
            else ()
        )
        gc.collect()
        assert argument() is None, "exceptional function retained its argument frame"
        assert source() is None, "failed-entry generator retained its source owner"
        assert all(reference() is None for reference in receivers)
        owner = _Owner()
        reference = weakref.ref(owner)
        with life.operation(owner) as current:
            assert current is owner
        del current, owner
        gc.collect()
        assert reference() is None, "normal yield/resume/return retained its source"
        assert observer.bindings_current()
    finally:
        if observer.active:
            observer.stop()
    receipt = observer.receipt()
    assert receipt["original_bindings_and_bodies_unchanged"]
    assert receipt["overflow"] == 0
    assert receipt["exceptional_timing_gaps_not_invented_as_returns"]
    assert sys.monitoring.get_events(observer.tool) == 0
    assert sys.monitoring.get_tool(observer.tool) is None
    assert all(
        sys.monitoring.get_local_events(observer.tool, code) == 0
        for code in observer.codes
    )
    counts = collections.Counter()
    for (_, _, label, _), row in observed.rows.items():
        counts[label] += row[0]
    assert (
        counts["tldw_chatbook.Backup_Recovery.storage_admission._acquire_storage"] == 1
    )
    assert (
        counts["tldw_chatbook.Backup_Recovery.config_participants.operation.enter"] == 2
    )
    assert (
        counts["tldw_chatbook.Backup_Recovery.config_participants.operation.exit"] == 1
    )
    if os.name == "nt":
        assert counts["_Native.open_handle"] == 1
