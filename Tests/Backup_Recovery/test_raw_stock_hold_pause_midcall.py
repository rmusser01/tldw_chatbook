"""A same-operation callback replacement cannot borrow an earlier stock probe."""

import inspect
import sys

from Tests.Backup_Recovery import test_raw_related_source_preparation as sources
from tldw_chatbook.Backup_Recovery import config_participants
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.admission import Admission

configured_source = sources.configured_source
local_root = sources.local_root


def test_callback_changed_after_first_original_probe_keeps_second_call(
    configured_source, monkeypatch
):
    source = configured_source
    before_bytes = source.get_cli_config_path().read_bytes()
    before_leases = set(storage._live_leases)
    original = Admission.pause_requested
    code, namespace = original.__code__, original.__globals__
    scope = inspect.unwrap(raw._scope)
    scope_code = scope.__code__
    captures, changed_calls = [], []

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if captures or event != "return" or frame.f_code is not code:
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
        assert frame.f_globals is namespace and result is False
        state = parent.f_locals["state"]
        assert not state.active and state.source is source
        assert len(state.leases) == 2 and len(state.holds) == 2
        assert state.holds[0] is state.holds[1]
        hold = state.holds[0]
        authority = frame.f_locals["self"]
        assert authority is hold.authority
        assert frame.f_locals["namespaces"] is hold.names
        assert all(lease in storage._live_leases for lease in state.leases)
        captures.append((state, hold, authority))

        def changed(namespaces):
            changed_calls.append((authority, namespaces))
            return original(authority, namespaces)

        # The existing class/body remains stock. Only the exact issued receiver
        # gets a supported live callback override AFTER the first native RETURN.
        monkeypatch.setattr(authority, "pause_requested", changed)

    with sources._original_scope_acquisitions(source) as observed:
        previous = sys.getprofile()
        sys.setprofile(observe)
        try:
            with config_participants.operation(source):
                result = source.get_cli_setting("general", "users_name", "missing")
        finally:
            sys.setprofile(previous)
    assert result == "retirement" and len(captures) == 1
    state, hold, authority = captures[0]
    assert len(observed.entries) == 1 and observed.entries[0][1] is state
    assert changed_calls == [(authority, hold.names)]
    assert len(observed.entries[0][2]) == 2
    assert source.get_cli_config_path().read_bytes() == before_bytes
    assert set(storage._live_leases) == before_leases
    assert Admission.pause_requested is original and original.__code__ is code
    assert raw._scope.__wrapped__ is scope and scope.__code__ is scope_code
