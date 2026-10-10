"""One stock hold needs one fresh pause decision, not one per counted lease."""

from contextlib import contextmanager
import inspect
import os
import sys
from types import FunctionType

import pytest

from Tests.Backup_Recovery import test_raw_related_source_preparation as sources
from Tests.Backup_Recovery import test_admission as admission_controls
from Tests.Backup_Recovery.test_participants import _eventually
from tldw_chatbook.Backup_Recovery import bootstrap, config_participants
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.admission import Admission

configured_source = sources.configured_source
local_root = sources.local_root
launch = admission_controls.launch
line = admission_controls.line
release = admission_controls.release


@contextmanager
def _scope_pause_calls(source, callback):
    """Observe only the actual original source scope's pause-request calls."""
    scope = inspect.unwrap(raw._scope)
    scope_code = scope.__code__
    pause = Admission.pause_requested
    pause_code, pause_globals = pause.__code__, pause.__globals__
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event != "call" or frame.f_code is not pause_code:
            return
        parent = frame.f_back
        for _ in range(4):
            if parent is None:
                return
            if parent.f_code is scope_code:
                break
            parent = parent.f_back
        else:
            return
        if (
            parent.f_code is not scope_code
            or parent.f_locals.get("source") is not source
        ):
            return
        assert frame.f_globals is pause_globals
        state = parent.f_locals["state"]
        assert not state.active
        assert state.source is source
        authority = frame.f_locals["self"]
        names = frame.f_locals["namespaces"]
        matches = tuple(
            hold
            for hold in state.holds
            if hold is not None and hold.authority is authority and hold.names == names
        )
        assert matches and all(hold is matches[0] for hold in matches)
        assert all(lease in storage._live_leases for lease in state.leases)
        callback(state, matches[0])

    sys.setprofile(observe)
    try:
        yield
    finally:
        sys.setprofile(previous)
        assert raw._scope.__wrapped__ is scope and scope.__code__ is scope_code
        assert Admission.pause_requested is pause
        assert pause.__code__ is pause_code and pause.__globals__ is pause_globals


def _read(source):
    # This is the same finite original config-operation/public-reader shape as
    # readiness read_current, with no widget or timing-dependent background work.
    with config_participants.operation(source):
        return source.get_cli_setting("general", "users_name", "missing")


def test_original_config_read_probes_one_exact_hold_once(configured_source, request):
    source = configured_source
    before = source.get_cli_config_path().read_bytes()
    before_leases = set(storage._live_leases)
    calls = []
    with sources._original_scope_acquisitions(source) as observed:
        with _scope_pause_calls(
            source, lambda state, hold: calls.append((state, hold))
        ):
            result = _read(source)
    assert result == "retirement"
    assert source.get_cli_config_path().read_bytes() == before
    assert len(observed.entries) == 1
    operation, state, leases = observed.entries[0]
    assert state.route == "config" and state.source is source
    assert len(leases) == 2 and len(state.holds) == 2
    assert state.holds[0] is state.holds[1] and state.holds[0] is not None
    assert all(item[0] is state and item[1] is state.holds[0] for item in calls)
    assert set(storage._live_leases) == before_leases
    assert operation not in raw._states and operation not in storage._raw_operations
    request.node.user_properties.extend(
        [
            ("original_pause_calls", len(calls)),
            ("original_counted_leases", len(leases)),
            ("same_exact_stock_hold", True),
            ("all_new_leases_retired", True),
        ]
    )
    # On unchanged source the qualified two-lease case reaches two original
    # native calls. This assertion is the causal work-count RED target.
    assert len(calls) == 1, "one exact hold was probed once per counted lease"


@pytest.mark.parametrize("change", ["replacement", "body"])
def test_custom_pause_callback_keeps_original_per_lease_route(
    configured_source, monkeypatch, change
):
    source = configured_source
    original = Admission.pause_requested
    calls = []
    before_leases = set(storage._live_leases)
    if change == "replacement":

        def replacement(self, namespaces):
            calls.append((self, namespaces))
            return original(self, namespaces)

        monkeypatch.setattr(Admission, "pause_requested", replacement)
    else:
        # Retain the original callable object/globals/defaults while replacing
        # its body. A type-only or identity-only stock selector must decline.
        copied = FunctionType(
            original.__code__,
            original.__globals__,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        copied.__kwdefaults__ = original.__kwdefaults__
        monkeypatch.setitem(original.__globals__, "_test_pause_calls", calls)
        monkeypatch.setitem(original.__globals__, "_test_pause_original", copied)
        namespace = {}
        exec(
            "def substituted(self, namespaces):\n"
            "    _test_pause_calls.append((self, namespaces))\n"
            "    return _test_pause_original(self, namespaces)\n",
            namespace,
        )
        monkeypatch.setattr(original, "__code__", namespace["substituted"].__code__)
    with sources._original_scope_acquisitions(source) as observed:
        assert _read(source) == "retirement"
    assert len(observed.entries) == 1
    _operation, state, leases = observed.entries[0]
    assert len(leases) == 2 and state.holds[0] is state.holds[1]
    # The original body clone may also be used by non-scope admission work;
    # qualify only requests on this exact operation's retained authority/group.
    matching = [
        row
        for row in calls
        if row[0] is state.holds[0].authority and row[1] == state.holds[0].names
    ]
    assert len(matching) == 2
    assert set(storage._live_leases) == before_leases


def test_native_pause_between_config_acquisitions_refuses_before_body(
    configured_source, local_root, launch
):
    source = configured_source
    before_bytes = source.get_cli_config_path().read_bytes()
    cache, generation = source._CONFIG_CACHE, source._CONFIG_GENERATION
    before_leases = set(storage._live_leases)
    startup_key = (os.getpid(), str(local_root))
    owned_startup = storage._startups.get(startup_key)
    # The existing fixture created only this new pytest-owned root. Its startup
    # is not an ambient/default-profile owner and cannot be redirected later.
    assert local_root == source.get_cli_config_path().parent / "bootstrap"
    assert owned_startup in before_leases and owned_startup._key == startup_key
    acquire = storage.acquire_storage
    acquire_code, acquire_globals = acquire.__code__, acquire.__globals__
    scope_code = inspect.unwrap(raw._scope).__code__
    previous = sys.getprofile()
    children, captured, body = [], [], []

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if captured or event != "return" or frame.f_code is not acquire_code:
            return
        parent = frame.f_back
        if parent is None or parent.f_code is not scope_code:
            return
        if parent.f_locals.get("source") is not source:
            return
        assert frame.f_globals is acquire_globals
        assert type(result) is storage.StorageLease and result in storage._live_leases
        state = parent.f_locals["state"]
        assert not state.leases and not state.active
        hold = storage._holds[result._key]
        assert hold.ready.is_set() and not hold.stop.is_set() and hold.error is None
        captured.append((result, state, hold))
        child = launch(hold.authority.control_root, "maintenance", hold.names)
        children.append(child)
        # Existing 3s native-state wait. This observation cannot grant entry;
        # the original final scope probe must independently see the contention.
        _eventually(lambda: hold.authority.pause_requested(hold.names))

    sys.setprofile(observe)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            with config_participants.operation(source):
                body.append(True)
                source.get_cli_setting("general", "users_name", "missing")
    finally:
        sys.setprofile(previous)
        assert storage.acquire_storage is acquire and acquire.__code__ is acquire_code
        if captured:
            # A profiling setup failure can interrupt the function's RETURN
            # before raw._scope receives this exact already-created lease.
            returned = captured[0][0]
            if returned in storage._live_leases:
                returned.close()
        if children:
            # Only this fresh source fixture's captured startup can retain the
            # real maintenance waiter. Do not sweep other roots or owners.
            assert storage._startups.get(startup_key) is owned_startup
            owned_startup.close()
            storage._startups.pop(startup_key)
            before_leases.remove(owned_startup)
            assert line(children[0]) == "entered"
            release(children[0])
    assert captured and not body
    lease, state, _hold = captured[0]
    assert lease not in storage._live_leases
    assert all(item not in storage._live_leases for item in state.leases)
    assert not any(item is state for item in raw._states.values())
    assert set(storage._live_leases) == before_leases
    assert source._CONFIG_CACHE is cache and source._CONFIG_GENERATION == generation
    assert source.get_cli_config_path().read_bytes() == before_bytes
