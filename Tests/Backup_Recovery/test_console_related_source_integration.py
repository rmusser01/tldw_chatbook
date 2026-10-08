"""Changed acquisition callbacks retain source behavior and exact custody."""

import inspect
import json
import sys
from contextlib import contextmanager
from types import FunctionType

import pytest

from Tests.Backup_Recovery import test_raw_related_source_preparation as related_cases
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage

configured_source = related_cases.configured_source
local_root = related_cases.local_root
installed_source_case = related_cases.installed_source_case


def _assert_read_output(case, output):
    if case.kind == "permission":
        assert output is False
        assert json.loads(case.path.read_bytes())["kill_switch"] is False
    else:
        assert case.kind == "hook"
        assert output.ready and output.rows == () and output.store_path == case.path


def _assert_scope_retired(observed):
    assert observed.entries and observed.closes
    assert all(closed and removed for closed, removed in observed.closes)
    assert not observed.active_acquires and not observed.active_closes
    for operation, state, leases in observed.entries:
        assert not state.active and not state.uncertain
        assert not state.pins and not state.files and not state.descriptors
        assert operation not in raw._states
        assert operation not in storage._raw_operations
        assert all(lease not in storage._live_leases for lease in leases)


@pytest.mark.parametrize("installed_source_case", ["permission", "hook"], indirect=True)
def test_custom_acquisition_keeps_original_positional_member_calls(
    installed_source_case, monkeypatch
):
    case = installed_source_case
    original = storage.acquire_storage
    scope_code = inspect.unwrap(raw._scope).__code__
    calls, returned = [], []
    before = case.path.read_bytes()

    def custom(*args, **kwargs):
        parent = sys._getframe(1)
        selected = (
            parent.f_code is scope_code and parent.f_locals.get("source") is case.source
        )
        if selected:
            calls.append((args, dict(kwargs)))
        lease = original(*args, **kwargs)
        if selected:
            assert lease in storage._live_leases
            returned.append(lease)
        return lease

    with monkeypatch.context() as patch:
        patch.setattr(storage, "acquire_storage", custom)
        _assert_read_output(case, case.read())
    assert case.path.read_bytes() == before
    assert len(calls) == (3 if case.kind == "permission" else 4)
    assert all(len(args) == 1 and not kwargs for args, kwargs in calls)
    assert calls[0][0] == (case.path,)
    assert returned and all(lease not in storage._live_leases for lease in returned)
    assert not any(state.source is case.source for state in raw._states.values())
    assert storage.acquire_storage is original


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_changed_acquisition_body_keeps_original_positional_member_calls(
    installed_source_case, monkeypatch
):
    case = installed_source_case
    original = storage.acquire_storage
    original_code, defaults, keywords = (
        original.__code__,
        original.__defaults__,
        original.__kwdefaults__,
    )
    copy = FunctionType(
        original_code, original.__globals__, original.__name__, defaults
    )
    copy.__kwdefaults__ = keywords
    calls, returned = [], []
    scope_code = inspect.unwrap(raw._scope).__code__
    before = case.path.read_bytes()

    def record_call(parent, args, kwargs):
        selected = (
            parent.f_code is scope_code and parent.f_locals.get("source") is case.source
        )
        if selected:
            calls.append((args, dict(kwargs)))
        return selected

    def record_result(selected, lease):
        if selected:
            assert lease in storage._live_leases
            returned.append(lease)

    namespace = original.__globals__
    with monkeypatch.context() as patch:
        patch.setitem(namespace, "_related_integration_record_call", record_call)
        patch.setitem(namespace, "_related_integration_record_result", record_result)
        patch.setitem(namespace, "_related_integration_acquire_copy", copy)
        replacement = compile(
            "def changed(*args, **kwargs):\n"
            "    selected = _related_integration_record_call(sys._getframe(1), args, kwargs)\n"
            "    lease = _related_integration_acquire_copy(*args, **kwargs)\n"
            "    _related_integration_record_result(selected, lease)\n"
            "    return lease\n",
            "<changed-original-acquisition-body>",
            "exec",
        )
        functions = {}
        exec(replacement, namespace, functions)
        patch.setattr(original, "__code__", functions["changed"].__code__)
        assert storage.acquire_storage is original
        assert original.__code__ is not original_code
        _assert_read_output(case, case.read())
    assert original.__code__ is original_code and original.__defaults__ is defaults
    assert original.__kwdefaults__ is keywords
    assert case.path.read_bytes() == before
    assert len(calls) == 3
    assert all(len(args) == 1 and not kwargs for args, kwargs in calls)
    assert calls[0][0] == (case.path,)
    assert returned and all(lease not in storage._live_leases for lease in returned)
    assert not any(state.source is case.source for state in raw._states.values())


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_changed_acquisition_default_tuple_keeps_original_member_calls(
    installed_source_case, monkeypatch
):
    case = installed_source_case
    original = storage.acquire_storage
    defaults = original.__defaults__
    before = case.path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(original, "__defaults__", (case.path.parent / "unused-default",))
        assert (
            storage.acquire_storage is original
            and original.__defaults__ is not defaults
        )
        with related_cases._original_scope_acquisitions(case.source) as observed:
            output = case.read()
    _assert_read_output(case, output)
    assert case.path.read_bytes() == before
    assert len(observed.requests) == 3
    assert all(not related for _primary, related in observed.requests)
    _assert_scope_retired(observed)
    assert original.__defaults__ is defaults


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_changed_acquisition_keyword_default_keeps_original_member_calls(
    installed_source_case, monkeypatch
):
    case = installed_source_case
    original = storage.acquire_storage
    keywords = original.__kwdefaults__
    assert type(keywords) is dict and keywords["related_paths"] == ()  # noqa: E721 -- original default mapping identity.
    before = case.path.read_bytes()
    changed_related = (case.path.parent / "additional-default-member",)
    with monkeypatch.context() as patch:
        # Keep the same function, code and default mapping. The ordinary
        # positional calls must consume this new tuple, rather than replacing
        # it with explicit batch members.
        patch.setitem(keywords, "related_paths", changed_related)
        assert (
            storage.acquire_storage is original and original.__kwdefaults__ is keywords
        )
        with related_cases._original_scope_acquisitions(case.source) as observed:
            output = case.read()
    _assert_read_output(case, output)
    assert case.path.read_bytes() == before
    assert len(observed.requests) == 3
    assert all(related == changed_related for _primary, related in observed.requests)
    assert observed.requests[0][0] == case.path
    _assert_scope_retired(observed)
    assert original.__kwdefaults__ is keywords and keywords["related_paths"] == ()


@contextmanager
def _replace_after_original_acquisition(source, replacement, monkeypatch):
    """Change the callback only after a real original lease has been issued."""
    function = storage.acquire_storage
    code, scope_code = function.__code__, inspect.unwrap(raw._scope).__code__
    previous = sys.getprofile()
    returned, states = [], []

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if frame.f_code is code and event == "return" and not returned:
            parent = frame.f_back
            if (
                parent is None
                or parent.f_code is not scope_code
                or parent.f_locals.get("source") is not source
            ):
                return
            assert type(result) is storage.StorageLease
            assert result in storage._live_leases
            state = parent.f_locals["state"]
            assert not state.active and state.source is source
            assert parent.f_locals["operation"] in raw._states
            returned.append(result)
            states.append(state)
            monkeypatch.setattr(storage, "acquire_storage", replacement)

    sys.setprofile(observe)
    try:
        yield returned, states
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("installed_source_case", ["permission", "hook"], indirect=True)
def test_late_acquisition_replacement_refuses_and_retires_issued_lease(
    installed_source_case, monkeypatch
):
    case = installed_source_case
    original = storage.acquire_storage
    before = case.path.read_bytes()
    baseline = set(storage._live_leases)
    bodies, foreign = [], []

    def replacement(*args, **kwargs):
        foreign.append((args, kwargs))
        raise AssertionError("a replaced acquisition must not be invoked")

    with monkeypatch.context() as patch:
        with _replace_after_original_acquisition(
            case.source, replacement, patch
        ) as issued:
            with pytest.raises(bootstrap.RecoveryRequired, match="selection_changed"):
                with raw._scope(case.source, case.route, writing=True):
                    bodies.append("must not execute")
    returned, states = issued
    assert (
        len(returned) == len(states) == 1
    ), "actual original acquisition was not reached"
    assert not bodies and not foreign
    assert case.path.read_bytes() == before
    assert returned[0] not in storage._live_leases
    state = states[0]
    assert not state.active and not state.uncertain
    assert not state.pins and not state.files and not state.descriptors
    observer = state.mcp_observation_lease
    expected = returned if observer is None else [observer, *returned]
    assert state.leases == expected and len(state.holds) == len(expected)
    assert all(lease not in storage._live_leases for lease in expected)
    assert not any(current is state for current in raw._states.values())
    assert storage._live_leases == baseline
    assert storage.acquire_storage is original
