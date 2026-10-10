"""Nested MCP metadata never substitutes for the final live source check."""

import copy
import inspect
import sys
import threading
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery import test_raw_related_source_preparation as source_cases
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage

configured_source = source_cases.configured_source
local_root = source_cases.local_root
installed_source_case = source_cases.installed_source_case


@contextmanager
def _nested_observation(source, operation, mutation=None):
    """Observe original nested calls; mutate only at actual metadata return."""
    scope_code = inspect.unwrap(raw._scope).__code__
    check_code = raw._check.__code__
    lookup = getattr(raw, "_nested_installed_mcp_state", None)
    lookup_code = lookup.__code__ if lookup is not None else None
    if mutation is not None:
        assert lookup_code is not None, "the owned metadata lookup is not implemented"
    state = raw._states[operation]
    previous = sys.getprofile()
    checks, lookups = [], []

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        parent = frame.f_back
        if (
            frame.f_code is check_code
            and event == "call"
            and parent is not None
            and parent.f_code is scope_code
            and parent.f_locals.get("previous") is operation
            and parent.f_locals.get("source") is source
        ):
            checks.append((frame.f_locals["path"], frame.f_locals["writing"]))
        elif (
            lookup_code is not None
            and frame.f_code is lookup_code
            and event == "return"
            and result is state
            and frame.f_locals.get("previous") is operation
            and frame.f_locals.get("source") is source
            and not lookups
        ):
            assert state.active and state.participant is not None
            assert operation in storage._raw_operations
            assert all(lease in storage._live_leases for lease in state.leases)
            assert state.pins
            for fd in state.pins.values():
                raw.os.fstat(fd)
            lookups.append(state)
            if mutation is not None:
                mutation(state)

    sys.setprofile(observe)
    try:
        yield checks, lookups
    finally:
        sys.setprofile(previous)


def _assert_retired(observed):
    assert observed.entries and observed.closes
    assert all(closed and removed for closed, removed in observed.closes)
    assert not observed.active_acquires and not observed.active_closes
    for operation, state, leases in observed.entries:
        assert not state.active and not state.uncertain
        assert not state.pins and not state.files and not state.descriptors
        assert operation not in raw._states and operation not in storage._raw_operations
        assert all(lease not in storage._live_leases for lease in leases)


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("change", ["source", "config", "lease", "participant"])
def test_drift_after_original_metadata_lookup_refuses_nested_body(
    installed_source_case, configured_source, monkeypatch, change
):
    case = installed_source_case
    before = case.path.read_bytes()
    changed, bodies = [], []
    alternate = configured_source._get_effective_config_path().with_name(
        "alternate.toml"
    )
    if change == "config":
        alternate.write_bytes(
            configured_source._get_effective_config_path().read_bytes()
        )
        alternate.chmod(0o600)
    with source_cases._original_scope_acquisitions(case.source) as retirement:
        with raw._scope(case.source, case.route, writing=True) as operation:
            state = raw._states[operation]
            with monkeypatch.context() as patch:

                def mutate(actual):
                    assert actual is state
                    changed.append(change)
                    if change == "source":
                        patch.setattr(
                            case.source, "path", case.path.with_name("other.json")
                        )
                    elif change == "config":
                        patch.setenv("TLDW_CONFIG_PATH", str(alternate))
                    elif change == "lease":
                        actual.leases[0].close()
                    else:
                        patch.delitem(raw._source_participants, case.source)

                with _nested_observation(case.source, operation, mutate) as observed:
                    with pytest.raises(bootstrap.RecoveryRequired):
                        with raw._scope(case.source, case.route, writing=True):
                            bodies.append("must not execute")
                checks, lookups = observed
                assert lookups == [state] and checks == [(state.selected, True)]
            assert raw._local.operation is operation
    assert changed == [change] and not bodies
    assert case.path.read_bytes() == before
    _assert_retired(retirement)
    assert case.read() is False


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
def test_foreign_thread_and_unissued_copy_cannot_enter_nested_mcp(
    installed_source_case,
):
    case = installed_source_case
    bodies, errors = [], []
    before = case.path.read_bytes()
    with source_cases._original_scope_acquisitions(case.source) as observed:
        with raw._scope(case.source, case.route, writing=True) as operation:
            state = raw._states[operation]
            raw._local.operation = copy.copy(operation)
            try:
                with pytest.raises(bootstrap.RecoveryRequired):
                    with raw._scope(case.source, case.route, writing=True):
                        bodies.append("unissued copy")
            finally:
                raw._local.operation = operation

            def foreign():
                raw._local.operation = operation
                try:
                    with raw._scope(case.source, case.route, writing=True):
                        bodies.append("foreign actor")
                except BaseException as error:
                    errors.append(error)
                finally:
                    raw._local.operation = None

            worker = threading.Thread(target=foreign)
            worker.start()
            worker.join(3)
            assert not worker.is_alive()
            assert len(errors) == 1 and isinstance(
                errors[0], bootstrap.RecoveryRequired
            )
            assert state.active and raw._local.operation is operation
            assert all(lease in storage._live_leases for lease in state.leases)
            for fd in state.pins.values():
                raw.os.fstat(fd)
    assert not bodies and case.path.read_bytes() == before
    _assert_retired(observed)


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("kind", ["path", "custom-string-conversion"])
def test_explicit_selected_read_keeps_original_equality_and_two_checks(
    installed_source_case, kind
):
    case = installed_source_case
    path_calls, bodies = [], []
    with source_cases._original_scope_acquisitions(case.source) as retirement:
        with raw._scope(case.source, case.route, writing=True) as operation:
            selected = raw._states[operation].selected
            with _nested_observation(case.source, operation) as observed:
                checks, lookups = observed

                class SelectedPath:
                    def __str__(self):
                        path_calls.append(len(checks))
                        return str(selected)

                    def __fspath__(self):
                        raise AssertionError(
                            "lexical_path must preserve str conversion"
                        )

                requested = selected if kind == "path" else SelectedPath()
                with raw._scope(
                    case.source, case.route, writing=True, selected_read=requested
                ) as nested:
                    assert nested is operation
                    bodies.append(True)
            assert checks == [(None, False), (selected, True)] and not lookups
            assert path_calls == ([1] if kind == "custom-string-conversion" else [])
    assert bodies == [True]
    _assert_retired(retirement)


@pytest.mark.parametrize("installed_source_case", ["permission"], indirect=True)
@pytest.mark.parametrize("kind", ["config", "custom-mcp"])
def test_other_routes_keep_original_nested_checks(
    installed_source_case, configured_source, kind
):
    case = installed_source_case
    if kind == "config":
        source, route = configured_source, "config"
    else:

        class CustomPermission(type(case.source)):
            pass

        source, route = CustomPermission(case.path), case.route
    bodies = []
    baseline = set(storage._live_leases)
    with raw._scope(source, route, writing=True) as operation:
        state = raw._states[operation]
        pins, leases = tuple(state.pins.values()), tuple(state.leases)
        assert leases and all(lease in storage._live_leases for lease in leases)
        if kind == "custom-mcp":
            assert state.participant is None and not state.pinned and not pins
            assert state.identities
            for directory, identity in state.identities.items():
                info = raw.os.stat(directory)
                assert (info.st_dev, info.st_ino) == identity
        else:
            assert state.pinned and pins
            for fd in pins:
                raw.os.fstat(fd)
        with _nested_observation(source, operation) as observed:
            with raw._scope(source, route, writing=True) as nested:
                assert nested is operation
                bodies.append(True)
        checks, lookups = observed
        assert checks == [(None, False), (state.selected, True)] and not lookups
    assert bodies == [True]
    assert not state.active and not state.uncertain
    assert not state.pins and not state.files and not state.descriptors
    assert operation not in raw._states and operation not in storage._raw_operations
    assert all(lease not in storage._live_leases for lease in leases)
    assert storage._live_leases == baseline
    for fd in pins:
        with pytest.raises(OSError):
            raw.os.fstat(fd)
