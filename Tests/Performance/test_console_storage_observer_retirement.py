"""Real monitoring retirement after acceptance validation fails."""

import sys

import pytest

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver


def _observed_original_counter():
    return 7


@pytest.mark.parametrize("unexpected_global_events", [False, True])
def test_owned_monitoring_slots_retire_before_acceptance(unexpected_global_events):
    observer = OriginalStorageUnitObserver({}, {"active": True}, lambda unit: None)
    monitor = sys.monitoring
    tool = next(value for value in range(6) if monitor.get_tool(value) is None)
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
