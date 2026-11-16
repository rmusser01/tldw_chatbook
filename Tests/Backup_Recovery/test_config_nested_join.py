"""Nested installed config reads retain real native custody with one fresh check."""

import os
import sys
import threading
import tomllib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_raw_native_retirement as controls
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import config_participants as config_files
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout

configured_source = controls.configured_source
local_root = controls.local_root


@contextmanager
def _original_calls():
    """Observe original calls without replacing a validator or native owner."""
    acquire = storage._acquire_storage
    parents = raw._check_parent_pins
    codes = (acquire.__code__, parents.__code__)
    observed = SimpleNamespace(acquisitions=0, parents=0, counting_join=False)
    thread = threading.get_ident()
    tool = next(index for index in range(6) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "config-nested-original-checks")

    def started(code, _offset):
        if threading.get_ident() != thread:
            return
        if code is codes[0]:
            observed.acquisitions += 1
        elif observed.counting_join:
            observed.parents += 1

    event = sys.monitoring.events.PY_START
    previous = sys.monitoring.register_callback(tool, event, started)
    masks = {code: sys.monitoring.get_local_events(tool, code) for code in codes}
    assert sys.monitoring.get_events(tool) == 0 and not any(masks.values())
    try:
        for code in codes:
            sys.monitoring.set_local_events(tool, code, event)
        yield observed
    finally:
        for code, mask in masks.items():
            sys.monitoring.set_local_events(tool, code, mask)
        assert sys.monitoring.register_callback(tool, event, previous) is started
        assert sys.monitoring.get_events(tool) == 0
        sys.monitoring.free_tool_id(tool)
        assert storage._acquire_storage is acquire
        assert raw._check_parent_pins is parents


def _join(source, kind, *, selected_read=None):
    if kind == "config":
        return config_files.operation(source, target=selected_read)
    return raw.scope(source, "config", writing=True, selected_read=selected_read)


def _retire_startup(local_root):
    # Let the measured raw operation be the final actual native owner.
    key = (os.getpid(), str(local_root))
    with storage._lock:
        startup = storage._startups.pop(key, None)
    if startup is not None:
        startup.close()
    assert key not in storage._holds


def _live_owner(source, operation):
    state = raw._states[operation]
    assert state.active and state.participant is not None
    assert state.source is source and state.route == "config"
    assert raw.participant_identity(state.participant).owner == "config"
    leases = tuple(state.leases)
    holds = tuple(dict.fromkeys(state.holds))
    pins = tuple(state.pins.values())
    assert leases and all(lease in storage._live_leases for lease in leases)
    assert holds and all(hold is not None for hold in holds)
    assert pins and all(os.fstat(descriptor) for descriptor in pins)
    return SimpleNamespace(state=state, leases=leases, holds=holds, pins=pins)


def _excluded(owned):
    for hold in owned.holds:
        with pytest.raises(AdmissionTimeout):
            with hold.authority.maintenance(hold.names, 0.02):
                pytest.fail("live config custody lost native exclusion")


def _retired(operation, owned, previous):
    assert getattr(raw._local, "operation", None) is previous
    assert operation not in raw._states and operation not in storage._raw_operations
    assert all(lease not in storage._live_leases for lease in owned.leases)
    for descriptor in owned.pins:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    for hold in owned.holds:
        assert hold.stop.is_set() and not hold.thread.is_alive() and hold.error is None
        assert storage._holds.get(hold.key) is not hold
        with hold.authority.maintenance(hold.names, 2):
            pass


def _published_state(source):
    return {
        name: getattr(source, name)
        for name in (
            "_CONFIG_CACHE",
            "_CONFIG_CACHE_SOURCE",
            "_CONFIG_FILE_STAMP",
            "_CONFIG_STAT_CHECKED_MONOTONIC",
            "_SETTINGS_CACHE",
            "_SETTINGS_CACHE_SOURCE",
            "settings",
            "_CONFIG_GENERATION",
        )
    }


def _same_publication(source, before):
    for name, value in before.items():
        assert getattr(source, name) is value, name


def _read_payload(operation, selected, expected):
    with raw.file(operation, selected, "r") as stream:
        assert tomllib.loads(stream.read()) == expected


@pytest.mark.parametrize("kind", ["raw", "config"])
def test_nested_config_join_uses_one_fresh_parent_check(
    configured_source, local_root, kind, record_property
):
    source = configured_source
    selected = source.get_cli_config_path()
    before = selected.read_bytes()
    expected = tomllib.loads(before.decode("utf-8"))
    _retire_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    with _original_calls() as observed:
        with config_files.operation(source) as outer:
            owned = _live_owner(source, outer)
            acquisitions_before = observed.acquisitions
            observed.counting_join = True
            try:
                with _join(source, kind) as joined:
                    assert joined is outer
            finally:
                observed.counting_join = False
            assert observed.acquisitions == acquisitions_before
            assert raw._states[outer] is owned.state and owned.state.active
            assert tuple(owned.state.leases) == owned.leases
            # Exercise the original guarded reader/file/final checks separately
            # from the empty join whose redundant checks are being counted.
            assert tomllib.loads(source.read_cli_config_serialized()) == expected
            assert all(lease in storage._live_leases for lease in owned.leases)
            _excluded(owned)
        assert selected.read_bytes() == before
    _retired(outer, owned, previous)
    record_property("nested_parent_checks", observed.parents)
    record_property("actual_acquisitions", observed.acquisitions)
    # Last oracle: real payload, same custody and physical retirement precede RED.
    assert observed.parents == 1, observed.parents


@pytest.mark.parametrize("kind", ["raw", "config"])
@pytest.mark.parametrize("explicit", [False, True], ids=["implicit", "native-target"])
def test_changed_selector_refuses_nested_body_without_failure_bookkeeping(
    configured_source, local_root, tmp_path, monkeypatch, kind, explicit
):
    source = configured_source
    selected = source.get_cli_config_path()
    requested = selected if explicit else None
    before_bytes = selected.read_bytes()
    expected = tomllib.loads(before_bytes.decode("utf-8"))
    foreign = tmp_path / "foreign.toml"
    foreign_bytes = b'[general]\nusers_name="foreign"\n'
    foreign.write_bytes(foreign_bytes)
    foreign.chmod(0o600)
    _retire_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    with config_files.operation(source) as outer:
        owned = _live_owner(source, outer)
        before = _published_state(source)
        persistence_error = source._CONFIG_PERSISTENCE_ERROR
        assert not owned.state.config_failed
        with monkeypatch.context() as changed:
            changed.setenv("TLDW_CONFIG_PATH", str(foreign))
            with pytest.raises(
                bootstrap.RecoveryRequired, match="raw_source_selection_changed"
            ):
                with _join(source, kind, selected_read=requested):
                    pytest.fail("changed effective selector entered nested body")
        # The original fresh check precedes rollback capture. Admission refusal
        # cannot be mistaken for a body failure or damage the live outer owner.
        _same_publication(source, before)
        assert source._CONFIG_PERSISTENCE_ERROR is persistence_error
        assert not owned.state.config_failed and owned.state.active
        assert tuple(owned.state.leases) == owned.leases
        _read_payload(outer, selected, expected)
        _excluded(owned)
    _retired(outer, owned, previous)
    assert selected.read_bytes() == before_bytes
    assert foreign.read_bytes() == foreign_bytes


@pytest.mark.parametrize("kind", ["raw", "config"])
@pytest.mark.parametrize("explicit", [False, True], ids=["implicit", "native-target"])
def test_nested_config_join_preserves_read_only_refusal_and_bookkeeping(
    configured_source, local_root, kind, explicit
):
    source = configured_source
    selected = source.get_cli_config_path()
    requested = selected if explicit else None
    before_bytes = selected.read_bytes()
    expected = tomllib.loads(before_bytes.decode("utf-8"))
    _retire_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    with raw.scope(source, "config", writing=False) as outer:
        owned = _live_owner(source, outer)
        before = _published_state(source)
        persistence_error = source._CONFIG_PERSISTENCE_ERROR
        assert not owned.state.config_failed
        with pytest.raises(bootstrap.RecoveryRequired, match="raw_path_outside_scope"):
            with _join(source, kind, selected_read=requested):
                pytest.fail("read-only config custody became writable")
        assert not owned.state.writing
        _same_publication(source, before)
        if kind == "config":
            # The existing fallback captures state before its writable raw
            # entry, so this refusal retains its original failure bookkeeping.
            assert owned.state.config_failed
            assert source._CONFIG_PERSISTENCE_ERROR == "config_operation_failed"
        else:
            assert not owned.state.config_failed
            assert source._CONFIG_PERSISTENCE_ERROR is persistence_error
        with raw.scope(
            source, "config", writing=False, selected_read=requested
        ) as joined:
            assert joined is outer
            _read_payload(joined, selected, expected)
        _excluded(owned)
    _retired(outer, owned, previous)
    assert selected.read_bytes() == before_bytes


@pytest.mark.parametrize("kind", ["raw", "config"])
@pytest.mark.parametrize("matches", [True, False], ids=["same", "foreign"])
def test_explicit_native_config_selection_checks_once_or_refuses(
    configured_source, local_root, tmp_path, kind, matches, record_property
):
    source = configured_source
    selected = source.get_cli_config_path()
    before_bytes = selected.read_bytes()
    expected = tomllib.loads(before_bytes.decode("utf-8"))
    foreign = tmp_path / "foreign-read.toml"
    foreign_bytes = b'[general]\nusers_name="unselected"\n'
    foreign.write_bytes(foreign_bytes)
    foreign.chmod(0o600)
    requested = selected if matches else foreign
    _retire_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    with _original_calls() as observed:
        with config_files.operation(source) as outer:
            owned = _live_owner(source, outer)
            acquisitions_before = observed.acquisitions
            observed.counting_join = True
            try:
                if matches:
                    with _join(source, kind, selected_read=requested) as joined:
                        assert joined is outer
                else:
                    with pytest.raises(
                        bootstrap.RecoveryRequired, match="raw_source_selection_changed"
                    ):
                        with _join(source, kind, selected_read=requested):
                            pytest.fail("foreign explicit config selection entered")
            finally:
                observed.counting_join = False
            assert observed.acquisitions == acquisitions_before
            assert tuple(owned.state.leases) == owned.leases
            _read_payload(outer, selected, expected)
            _excluded(owned)
    _retired(outer, owned, previous)
    assert selected.read_bytes() == before_bytes
    assert foreign.read_bytes() == foreign_bytes
    record_property("explicit_selection_parent_checks", observed.parents)
    if matches:
        assert observed.parents == 1, observed.parents


class _CallbackPath(os.PathLike):
    def __init__(self, path):
        self.path = path
        self.coordinator_owned = []

    def __fspath__(self):
        self.coordinator_owned.append(storage._lock._is_owned())
        return os.fspath(self.path)

    def __str__(self):
        return self.__fspath__()

    def __eq__(self, other):
        pytest.fail("callback-bearing path equality used for config join")


@pytest.mark.parametrize("kind", ["raw", "config"])
@pytest.mark.parametrize("target_form", ["string", "callback", "relative"])
def test_other_explicit_config_targets_keep_original_join_checks(
    configured_source, local_root, monkeypatch, kind, target_form, record_property
):
    source = configured_source
    selected = source.get_cli_config_path()
    before_bytes = selected.read_bytes()
    expected = tomllib.loads(before_bytes.decode("utf-8"))
    if target_form == "string":
        requested = str(selected)
    elif target_form == "callback":
        requested = _CallbackPath(selected)
    else:
        monkeypatch.chdir(selected.parent)
        requested = Path(selected.name)
        assert not requested.is_absolute()
    _retire_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    with _original_calls() as observed:
        with config_files.operation(source) as outer:
            owned = _live_owner(source, outer)
            acquisitions_before = observed.acquisitions
            observed.counting_join = True
            try:
                with _join(source, kind, selected_read=requested) as joined:
                    assert joined is outer
            finally:
                observed.counting_join = False
            assert observed.acquisitions == acquisitions_before
            assert raw._states[outer] is owned.state and owned.state.active
            assert tuple(owned.state.leases) == owned.leases
            if target_form == "callback":
                assert requested.coordinator_owned
                assert not any(requested.coordinator_owned)
            _read_payload(outer, selected, expected)
            _excluded(owned)
    _retired(outer, owned, previous)
    assert selected.read_bytes() == before_bytes
    record_property("fallback_parent_checks", observed.parents)
    assert observed.parents == (3 if kind == "config" else 2), observed.parents


class _NestedBodyFailure(RuntimeError):
    pass


@pytest.mark.parametrize("explicit", [False, True], ids=["implicit", "native-target"])
def test_nested_config_body_failure_rolls_back_actual_cache_publication(
    configured_source, local_root, explicit
):
    source = configured_source
    selected = source.get_cli_config_path()
    requested = selected if explicit else None
    before_bytes = selected.read_bytes()
    expected = tomllib.loads(before_bytes.decode("utf-8"))
    _retire_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    before = _published_state(source)
    failure = _NestedBodyFailure("after actual config publication")
    with config_files.operation(source) as outer:
        owned = _live_owner(source, outer)
        with pytest.raises(_NestedBodyFailure) as raised:
            with config_files.operation(source, target=requested) as joined:
                assert joined is outer
                published = source._publish_runtime_config_unlocked(
                    expected, config_path=selected
                )
                assert (
                    published["general"]["users_name"]
                    == expected["general"]["users_name"]
                )
                assert source._CONFIG_CACHE is not before["_CONFIG_CACHE"]
                assert source._CONFIG_GENERATION == before["_CONFIG_GENERATION"] + 1
                raise failure
        assert raised.value is failure
        _same_publication(source, before)
        assert owned.state.config_failed and owned.state.active
        assert source._CONFIG_PERSISTENCE_ERROR == "config_operation_failed"
        _read_payload(outer, selected, expected)
        _excluded(owned)
    _retired(outer, owned, previous)
    _same_publication(source, before)
    assert source._CONFIG_PERSISTENCE_ERROR == "config_operation_failed"
    assert selected.read_bytes() == before_bytes


@pytest.mark.parametrize("kind", ["raw", "config"])
@pytest.mark.parametrize("explicit", [False, True], ids=["implicit", "native-target"])
def test_nested_config_selector_runs_outside_storage_coordinator(
    configured_source, local_root, monkeypatch, kind, explicit
):
    source = configured_source
    selected = source.get_cli_config_path()
    requested = selected if explicit else None
    before_bytes = selected.read_bytes()
    expected = tomllib.loads(before_bytes.decode("utf-8"))
    _retire_startup(local_root)
    previous = getattr(raw._local, "operation", None)
    selector = source._get_effective_config_path
    coordinator_owned = []

    def observe_selector():
        coordinator_owned.append(storage._lock._is_owned())
        return selector()

    with config_files.operation(source) as outer:
        owned = _live_owner(source, outer)
        with monkeypatch.context() as observer:
            observer.setattr(source, "_get_effective_config_path", observe_selector)
            with _join(source, kind, selected_read=requested) as joined:
                assert joined is outer
        assert coordinator_owned and not any(coordinator_owned)
        _read_payload(outer, selected, expected)
        _excluded(owned)
    _retired(outer, owned, previous)
    assert selected.read_bytes() == before_bytes
