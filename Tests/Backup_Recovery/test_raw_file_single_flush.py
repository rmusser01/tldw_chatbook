"""Count real Windows barriers for an exact source-owned history descriptor."""

import errno
import inspect
import json
import os
import sys
import threading

import pytest

from Tests.Backup_Recovery.config_test_support import install_config_source
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414 - pytest fixture dependency
)
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Chat.prompt_history import PromptHistory

pytestmark = pytest.mark.skipif(os.name != "nt", reason="real Windows file barriers")


class _HistoryFlushObservation:
    """Local original-code events only; retain no frames or tracebacks."""

    def __init__(self, history):
        from tldw_chatbook.Utils import windows_files

        self.history = history
        self.windows = windows_files
        self.file_body = inspect.unwrap(raw._file)
        self.flush = windows_files.flush_file
        self.bindings = (
            (self.file_body, self.file_body.__code__),
            (raw.flush_file, raw.flush_file.__code__),
            (self.flush, self.flush.__code__),
            (windows_files.WindowsOS.fsync, windows_files.WindowsOS.fsync.__code__),
        )
        self.rows = []
        self.pending = {}
        self.flush_pending = {}
        self.errors = []
        self.events = 0
        self.tool = None
        self.retired = False
        self.registered = []
        self.installed = {}
        self.callbacks = (
            (sys.monitoring.events.PY_START, self.started),
            (sys.monitoring.events.PY_YIELD, self.yielded),
            (sys.monitoring.events.PY_RETURN, self.returned),
        )
        self.masks = {
            self.file_body.__code__: (
                sys.monitoring.events.PY_START
                | sys.monitoring.events.PY_YIELD
                | sys.monitoring.events.PY_RETURN
            ),
            self.flush.__code__: (
                sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN
            ),
        }

    def __enter__(self):
        monitor = sys.monitoring
        self.tool = next(
            (
                tool
                for tool in range(5, -1, -1)
                if tool != monitor.DEBUGGER_ID and monitor.get_tool(tool) is None
            ),
            None,
        )
        assert self.tool is not None, "no free local monitoring tool"
        monitor.use_tool_id(self.tool, "history-single-native-flush")
        try:
            assert monitor.get_events(self.tool) == 0
            for event, callback in self.callbacks:
                previous = monitor.register_callback(self.tool, event, callback)
                self.registered.append((event, callback))
                assert previous is None
            for code, mask in self.masks.items():
                assert monitor.get_local_events(self.tool, code) == 0
                monitor.set_local_events(self.tool, code, mask)
                self.installed[code] = mask
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _bounded(self):
        self.events += 1
        if self.events <= 128:
            return True
        if not self.errors:
            self.errors.append("event bound exceeded")
        return False

    def started(self, code, _offset):
        if not self._bounded():
            return
        try:
            frame = sys._getframe(1)
            thread = threading.get_ident()
            key = (thread, id(frame))
            if code is self.file_body.__code__:
                operation = frame.f_locals["operation"]
                state = raw._states[operation]
                if state.source is not self.history:
                    return
                assert len(self.rows) < 4
                assert state.participant is not None and state.pinned
                assert state.active and state.leases
                assert all(lease in storage._live_leases for lease in state.leases)
                row = {
                    "mode": frame.f_locals["mode"],
                    "thread": thread,
                    "operation": operation,
                    "state": state,
                    "leases": tuple(state.leases),
                    "flush_starts": 0,
                    "flush_returns": 0,
                    "yielded": False,
                    "returned": False,
                }
                self.rows.append(row)
                self.pending[key] = row
            else:
                fd = frame.f_locals["fd"]
                matches = [
                    row
                    for row in self.pending.values()
                    if row["thread"] == thread and row.get("fd") == fd
                ]
                if not matches:
                    return
                assert len(matches) == 1 and key not in self.flush_pending
                row = matches[0]
                assert row["yielded"] and not row["returned"]
                assert fd in row["state"].descriptors
                row["flush_starts"] += 1
                self.flush_pending[key] = row
        except Exception as error:
            self.errors.append(type(error).__name__)

    def yielded(self, code, _offset, value):
        if not self._bounded():
            return
        try:
            frame = sys._getframe(1)
            row = self.pending.get((threading.get_ident(), id(frame)))
            if row is None:
                return
            assert code is self.file_body.__code__ and not row["yielded"]
            fd = frame.f_locals["fd"]
            assert fd in row["state"].descriptors
            info = os.fstat(fd)
            row.update(
                fd=fd,
                identity=(info.st_dev, info.st_ino),
                wrapper=value,
                native=frame.f_locals["native"],
                yielded=True,
            )
        except Exception as error:
            self.errors.append(type(error).__name__)

    def returned(self, code, _offset, _value):
        if not self._bounded():
            return
        try:
            frame = sys._getframe(1)
            key = (threading.get_ident(), id(frame))
            if code is self.flush.__code__:
                row = self.flush_pending.pop(key, None)
                if row is not None:
                    row["flush_returns"] += 1
                return
            row = self.pending.pop(key, None)
            if row is None:
                return
            assert row["yielded"]
            assert row["wrapper"].closed and row["native"].closed
            assert row["fd"] not in row["state"].descriptors
            try:
                os.fstat(row["fd"])
            except OSError as error:
                assert error.errno == errno.EBADF
            else:
                raise AssertionError("source descriptor still open at original return")
            row["returned"] = True
        except Exception as error:
            self.errors.append(type(error).__name__)

    def __exit__(self, *_error):
        monitor = sys.monitoring
        if self.tool is None:
            return
        if monitor.get_tool(self.tool) != "history-single-native-flush":
            self.errors.append("monitor ownership changed")
            return
        if monitor.get_events(self.tool) != 0:
            self.errors.append("global mask changed")
        for code, mask in self.installed.items():
            if monitor.get_local_events(self.tool, code) != mask:
                self.errors.append("local mask changed")
            monitor.set_local_events(self.tool, code, 0)
        monitor.set_events(self.tool, 0)
        for event, callback in self.registered:
            if monitor.register_callback(self.tool, event, None) is not callback:
                self.errors.append("monitor callback changed")
        monitor.free_tool_id(self.tool)
        self.tool = None
        self.retired = True

    def assert_retired(self):
        assert self.retired and not self.errors, self.errors
        assert not self.pending and not self.flush_pending
        assert inspect.unwrap(raw._file) is self.file_body
        assert self.windows.flush_file is self.flush
        assert raw.flush_file is self.bindings[1][0]
        assert self.windows.WindowsOS.fsync is self.bindings[3][0]
        assert all(function.__code__ is code for function, code in self.bindings)
        for row in self.rows:
            assert row["returned"]
            state = row["state"]
            assert not state.uncertain
            assert not state.files and not state.descriptors and not state.pins
            assert row["operation"] not in raw._states
            assert row["operation"] not in storage._raw_operations
            assert all(lease not in storage._live_leases for lease in row["leases"])
            assert row["flush_starts"] == row["flush_returns"]


@pytest.mark.usefixtures("local_root")
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["append", "rewrite", "read"])
async def test_history_uses_one_native_file_barrier_per_write(mode, monkeypatch):
    """Original append/rewrite is RED at two, read stays zero; all IO is real."""
    install_config_source(monkeypatch)
    history = PromptHistory(max_entries=2)
    # Qualify a real loaded owner before observation, like a warm Console Send.
    await history.load()
    assert history._loaded and history.persistence_error is None
    if mode in {"rewrite", "read"}:
        assert await history.append("first")
    if mode == "rewrite":
        assert await history.append("second")
    if mode == "read":
        history = PromptHistory(max_entries=2)

    with _HistoryFlushObservation(history) as observed:
        if mode == "read":
            await history.load()
            assert history._loaded and history.persistence_error is None
        else:
            assert await history.append("third")
    observed.assert_retired()
    assert len(observed.rows) == 1
    row = observed.rows[0]
    expected = {"append": ["third"], "rewrite": ["second", "third"], "read": ["first"]}[
        mode
    ]
    persisted = [
        json.loads(line)
        for line in history.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["input"] for entry in persisted] == expected
    assert all(isinstance(entry["timestamp"], (int, float)) for entry in persisted)
    assert [entry["input"] for entry in history._entries] == expected
    published = history.path.stat()
    assert row["identity"] == (published.st_dev, published.st_ino)
    assert row["mode"] == {"append": "a", "rewrite": "w", "read": "r"}[mode]
    assert row["flush_starts"] == (0 if mode == "read" else 1), (
        "same source-owned descriptor reached the original native file barrier "
        f"{row['flush_starts']} times in {mode} mode"
    )
