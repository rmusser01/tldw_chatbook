"""Real monitoring retirement after acceptance validation fails."""

import contextlib
import sys
import threading
from collections import Counter
from types import SimpleNamespace

import pytest
import pytest_timeout

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver


def _observed_original_counter():
    return 7


@pytest.mark.parametrize("unexpected_global_events", [False, True])
def test_owned_monitoring_slots_retire_before_acceptance(unexpected_global_events):
    observer = OriginalStorageUnitObserver({}, {"active": True}, lambda unit: None)
    monitor = sys.monitoring
    tool = next(value for value in range(5, 0, -1) if monitor.get_tool(value) is None)
    monitor.use_tool_id(tool, "storage-observer-retirement-control")
    observer.tool = tool
    observer.active = observer.installed = True
    code = observer._pin(_observed_original_counter)
    observer.codes[code] = "storage_admissions"
    observer.slots.append(
        (
            sys.modules[__name__],
            "_observed_original_counter",
            _observed_original_counter,
        )
    )
    observer.registered = {
        monitor.events.PY_START: observer._start,
        monitor.events.PY_RETURN: observer._return,
    }
    try:
        for event, callback in observer.registered.items():
            assert monitor.register_callback(tool, event, callback) is None
        monitor.set_local_events(
            tool, code, monitor.events.PY_START | monitor.events.PY_RETURN
        )
        assert _observed_original_counter() == 7
        assert observer.started == observer.returned == 1
        mask = monitor.events.PY_START if unexpected_global_events else 0
        monitor.set_events(tool, mask)
        receipt = observer.close()
        assert receipt["original_source_current"]
        assert receipt["global_events"] == mask
        assert receipt["complete"] is (not unexpected_global_events)
        assert bool(receipt["invalid"]) is unexpected_global_events
        assert observer.active is False
        assert monitor.get_tool(tool) is None
        assert monitor.get_events(tool) == 0
        # Reclaim the exact freed slot to inspect the actual retired local mask
        # and callback table: free_tool_id alone does not clear these resources.
        monitor.use_tool_id(tool, "storage-observer-retirement-verification")
        assert monitor.get_local_events(tool, code) == 0
        for event in observer.registered:
            assert monitor.register_callback(tool, event, None) is None
    finally:
        if monitor.get_tool(tool) is not None:
            monitor.set_events(tool, 0)
            monitor.set_local_events(tool, code, 0)
            for event in observer.registered:
                monitor.register_callback(tool, event, None)
            monitor.free_tool_id(tool)


# Ordinary declared bodies give install() genuine module/source/contextlib pins.
@contextlib.contextmanager
def operation():
    yield None


def _acquire_storage():
    return True


class _SourceHelperLease:
    @classmethod
    def start(cls):
        return None


@contextlib.contextmanager
def _occupied_nondebugger_slots(slots):
    monitor = sys.monitoring
    owned = {}
    try:
        assert monitor.DEBUGGER_ID == 0
        assert all(monitor.get_tool(tool) is None for tool in range(6))
        assert not pytest_timeout.is_debugging()
        for tool in slots:
            assert tool != monitor.DEBUGGER_ID
            name = "storage-observer-capacity-control-" + str(tool)
            monitor.use_tool_id(tool, name)
            owned[tool] = name
            assert monitor.get_events(tool) == 0
        yield
    finally:
        changed = []
        for tool, name in owned.items():
            if monitor.get_tool(tool) != name:
                changed.append(tool)
                continue  # A changed owner is never adopted or freed.
            monitor.set_events(tool, 0)
            monitor.free_tool_id(tool)
        assert not changed
        assert all(monitor.get_tool(tool) is None for tool in owned)


def _assert_original_observer_retired(observer, tool):
    monitor = sys.monitoring
    assert not observer.active
    assert monitor.get_tool(tool) is None
    assert monitor.get_events(tool) == 0
    monitor.use_tool_id(tool, "storage-observer-capacity-retirement-verification")
    try:
        for code in (*observer.codes, *observer.context_codes):
            assert monitor.get_local_events(tool, code) == 0
        for event in observer.registered:
            assert monitor.register_callback(tool, event, None) is None
    finally:
        monitor.set_events(tool, 0)
        for code in (*observer.codes, *observer.context_codes):
            monitor.set_local_events(tool, code, 0)
        for event in observer.registered:
            monitor.register_callback(tool, event, None)
        monitor.free_tool_id(tool)


def test_original_install_refuses_exhausted_capacity_without_debugger_slot():
    observer = OriginalStorageUnitObserver(Counter(), lambda: True, lambda unit: None)
    module = sys.modules[__name__]
    monitor = sys.monitoring
    with _occupied_nondebugger_slots((5, 4, 3, 2, 1)):
        try:
            with pytest.raises(AssertionError):
                observer.install(module, module, _SourceHelperLease)
            assert observer.tool is None
            assert not observer.active and not observer.installed
            assert monitor.get_tool(monitor.DEBUGGER_ID) is None
            assert not pytest_timeout.is_debugging()
        finally:
            receipt = observer.close()
            if observer.tool is not None:
                _assert_original_observer_retired(observer, observer.tool)
        assert not receipt["complete"]
        assert receipt["original_source_current"]
        assert (
            receipt["hooks_retired_before_inactive"] and receipt["global_events"] == 0
        )
    assert monitor.get_tool(monitor.DEBUGGER_ID) is None
    assert not pytest_timeout.is_debugging()


class _TimeoutItem:
    def __init__(self):
        self.nodeid = "storage-observer-source-qualified-thread-timer"
        self.config = SimpleNamespace(
            _env_timeout=300.0,
            _env_timeout_method="thread",
            _env_timeout_func_only=False,
            _env_timeout_disable_debugger_detection=False,
        )

    def get_closest_marker(self, name):
        return None


def test_original_install_uses_only_free_slot_one_and_preserves_thread_timer():
    counts = Counter()
    observer = OriginalStorageUnitObserver(
        counts, lambda: True, lambda unit: counts.update((unit,))
    )
    module = sys.modules[__name__]
    monitor = sys.monitoring
    item, timer = _TimeoutItem(), None
    with _occupied_nondebugger_slots((5, 4, 3, 2)):
        try:
            observer.install(module, module, _SourceHelperLease)
            assert observer.tool == 1
            assert monitor.get_tool(monitor.DEBUGGER_ID) is None
            assert not pytest_timeout.is_debugging()
            with operation():
                assert _acquire_storage()
                assert _SourceHelperLease.start() is None
            assert observer.started == observer.returned == 3
            assert counts == {
                "config_admissions": 1,
                "storage_admissions": 1,
                "helper_spawns": 1,
            }
            for function in (
                pytest_timeout.is_debugging,
                pytest_timeout._get_item_settings,
                pytest_timeout.pytest_timeout_set_timer,
                pytest_timeout.pytest_timeout_cancel_timer,
                pytest_timeout.timeout_timer,
                threading.Timer.__init__,
                threading.Timer.run,
            ):
                observer._pin(function)
            settings = pytest_timeout._get_item_settings(item)
            assert settings == pytest_timeout.Settings(300.0, "thread", False, False)
            assert pytest_timeout.pytest_timeout_set_timer(item, settings) is True
            cancel = item.cancel_timeout
            observer._pin(cancel)
            assert cancel.__code__.co_freevars == ("timer",)
            timer = cancel.__closure__[0].cell_contents
            assert type(timer) is threading.Timer
            assert timer.interval == 300.0
            assert timer.function is pytest_timeout.timeout_timer
            assert timer.args == (item, settings)
            assert timer.is_alive() and not timer.finished.is_set()
            assert not pytest_timeout.is_debugging()
        finally:
            # Cancel joins the exact original Timer even if a later oracle fails.
            pytest_timeout.pytest_timeout_cancel_timer(item)
            receipt = observer.close()
            if observer.tool is not None:
                _assert_original_observer_retired(observer, observer.tool)
        assert timer is not None and not timer.is_alive() and timer.finished.is_set()
        assert receipt["complete"] and receipt["original_source_current"]
        assert (
            receipt["hooks_retired_before_inactive"] and receipt["global_events"] == 0
        )
    assert monitor.get_tool(monitor.DEBUGGER_ID) is None
    assert not pytest_timeout.is_debugging()
