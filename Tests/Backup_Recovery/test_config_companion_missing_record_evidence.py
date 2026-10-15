"""A completed absent profile lookup grants no config companion authority."""

import hashlib
import inspect
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import CodeType, FunctionType

import pytest

from Tests.Backup_Recovery import test_raw_related_source_preparation as sources
from Tests.Backup_Recovery.test_participants import _eventually
from tldw_chatbook.Backup_Recovery import (
    bootstrap,
    config_participants,
    control_records,
)
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage

configured_source = sources.configured_source
local_root = sources.local_root

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="ordinary derived-evidence reuse is POSIX-only"
)


def _read(source):
    with config_participants.operation(source):
        return source.get_cli_setting("general", "users_name", "missing")


def _warm_missing(source):
    """Leave the original settle margin intact and retain the real startup hold."""
    root = bootstrap.default_bootstrap_root()
    selected = source.get_cli_config_path()
    assert not any(
        row["selector"] == str(selected) for row in bootstrap._records(root)[1]
    )
    hold = storage._holds[(os.getpid(), str(root))]
    assert hold.names == (control_records.UNBOUND_NAMESPACE,)
    # Preparation only: let actual freshly written inputs reach the original
    # settle margin, rather than changing the margin, timestamps or admission.
    newest = max(path.stat().st_ctime_ns for path in root.rglob("*") if path.is_file())
    newest = max(
        newest, selected.stat().st_ctime_ns, root.stat().st_ctime_ns, time.time_ns()
    )
    _eventually(lambda: time.time_ns() - newest >= storage._EVIDENCE_SETTLE_NS)
    for _ in range(3):
        assert _read(source) == "retirement"
    assert storage._holds[hold.key] is hold and storage._hold_serving(hold)
    # Every mutation below must start from actual engaged absence reuse.
    # The original three reads, real settle margin and deadline stay intact.
    with sources._original_scope_acquisitions(source) as observed:
        with _companion_record_calls(source) as records:
            with _companion_reuses(source) as reuses:
                assert _read(source) == "retirement"
    assert not records and reuses == [True] and _cache_fact(hold, selected)
    assert len(observed.entries) == 1
    operation, state, leases = observed.entries[0]
    assert state.companion_guard is None and len(leases) == 2
    assert len(state.holds) == 2 and all(item is hold for item in state.holds)
    assert all(lease not in storage._live_leases for lease in leases)
    assert operation not in raw._states and operation not in storage._raw_operations
    return root, selected, hold


@contextmanager
def _companion_record_calls(source):
    """Observe the untouched _records body only in this exact source's guard."""
    records = bootstrap._records
    assert type(records) is FunctionType  # noqa: E721 - exact defining function.
    records_code, records_globals = records.__code__, records.__globals__
    defining_path = Path(bootstrap.__file__).resolve()
    defining_source = defining_path.read_bytes()
    assert hashlib.sha256(defining_source).hexdigest() == (
        "398ac4ad624e75979e15a57a49ea59c27928a2996ce3122cad000601064412aa"
    )
    assert bootstrap.__spec__.name == bootstrap.__name__
    assert Path(bootstrap.__spec__.origin).resolve() == defining_path
    assert Path(records_code.co_filename).resolve() == defining_path
    expected_code = next(
        item
        for item in compile(
            defining_source, str(defining_path), "exec", dont_inherit=True
        ).co_consts
        if type(item) is CodeType and item.co_name == "_records"  # noqa: E721
    )
    assert records_code == expected_code
    assert tuple((type(value), value) for value in records_code.co_consts) == tuple(
        (type(value), value) for value in expected_code.co_consts
    )
    assert (
        records_code.co_qualname,
        records_code.co_firstlineno,
        records_code.co_linetable,
        records_code.co_exceptiontable,
        records_code.co_stacksize,
    ) == (
        expected_code.co_qualname,
        expected_code.co_firstlineno,
        expected_code.co_linetable,
        expected_code.co_exceptiontable,
        expected_code.co_stacksize,
    )
    assert records.__defaults__ is None and records.__kwdefaults__ is None
    assert records.__closure__ is None
    assert records_globals is vars(bootstrap)
    guard = config_participants.companion_guard
    guard_code, guard_globals = guard.__code__, guard.__globals__
    rows = []
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event != "return" or frame.f_code is not records_code:
            return
        parent = frame.f_back
        if parent is None or parent.f_code is not guard_code:
            return
        state = raw._states.get(parent.f_locals["operation"])
        if state is None or state.source is not source:
            return
        assert frame.f_globals is records_globals and parent.f_globals is guard_globals
        assert not state.active and len(state.leases) == 1
        assert state.leases[0] in storage._live_leases
        assert type(result) is tuple and len(result) == 2  # noqa: E721 - original result.
        pending, profiles = result
        record = next(
            (row for row in profiles if row["selector"] == str(state.selected)), None
        )
        rows.append((bool(pending), record is None, state.holds[0].names))

    sys.setprofile(observe)
    try:
        yield rows
    finally:
        sys.setprofile(previous)
        assert bootstrap._records is records and records.__code__ is records_code
        assert records.__globals__ is records_globals
        assert records.__defaults__ is None and records.__kwdefaults__ is None
        assert records.__closure__ is None
        assert defining_path.read_bytes() == defining_source
        assert (
            config_participants.companion_guard is guard
            and guard.__code__ is guard_code
        )
        assert guard.__globals__ is guard_globals


@contextmanager
def _companion_reuses(source):
    """Observe a true original reuse return, for this exact guard and source."""
    reuse = storage._derived_reuse
    code, namespace = reuse.__code__, reuse.__globals__
    guard = config_participants.companion_guard
    guard_code = guard.__code__
    previous = sys.getprofile()
    rows = []

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        parent = frame.f_back
        if (
            event != "return" or frame.f_code is not code
            or frame.f_globals is not namespace or parent is None
            or parent.f_code is not guard_code
        ):
            return
        state = raw._states.get(parent.f_locals["operation"])
        if state is None or state.source is not source:
            return
        assert frame.f_locals["key"] == ("companions", str(state.config_anchor or state.selected))
        assert frame.f_locals["hold"] is state.holds[0]
        assert not state.active and len(state.leases) == 1
        assert state.leases[0] in storage._live_leases
        assert result[0] is True and result[1][0] is None
        rows.append(True)

    sys.setprofile(observe)
    try:
        yield rows
    finally:
        sys.setprofile(previous)
        assert storage._derived_reuse is reuse and reuse.__code__ is code
        assert reuse.__globals__ is namespace
        assert config_participants.companion_guard is guard and guard.__code__ is guard_code


def _cache_fact(hold, selected):
    item = hold.derived_evidence.get(("companions", str(selected)))
    return item is not None and item[0].confirmed


def test_warm_missing_profile_skips_records_but_keeps_two_actual_leases(
    configured_source, request
):
    source = configured_source
    _, selected, hold = _warm_missing(source)
    before = selected.read_bytes()
    before_leases = set(storage._live_leases)
    with sources._original_scope_acquisitions(source) as observed:
        with _companion_record_calls(source) as rows:
            assert _read(source) == "retirement"
    assert len(observed.entries) == 1
    operation, state, leases = observed.entries[0]
    assert state.companion_guard is None and len(leases) == 2
    assert len(state.holds) == 2 and all(item is hold for item in state.holds)
    assert set(storage._live_leases) == before_leases
    assert operation not in raw._states and operation not in storage._raw_operations
    assert selected.read_bytes() == before
    request.node.user_properties.extend(
        [
            ("original_companion_records", len(rows)),
            ("negative_metadata_confirmed", _cache_fact(hold, selected)),
            ("actual_fallback_leases", len(leases)),
            ("all_new_leases_retired", True),
        ]
    )
    assert not rows, "the same settled missing profile still re-reads control records"


def test_pending_publication_refuses_without_body_or_new_leases(
    configured_source, request
):
    source = configured_source
    root = bootstrap.default_bootstrap_root()
    selected = source.get_cli_config_path()
    authority = storage._holds[(os.getpid(), str(root))].authority
    data = source.get_user_data_dir()
    authority.register("profile", (selected, data))
    _, _, hold = _warm_missing(source)
    was_confirmed = _cache_fact(hold, selected)
    before, before_leases = selected.read_bytes(), set(storage._live_leases)
    control_records.register_pending(
        root, "companion-pending", ("profile",), root.parent / "control", (selected,)
    )
    body_entered = False
    with pytest.raises(bootstrap.RecoveryRequired):
        with config_participants.operation(source):
            body_entered = True
    assert not body_entered and selected.read_bytes() == before
    assert set(storage._live_leases) == before_leases
    assert not raw._states and not storage._raw_operations
    assert not storage._derived_reuse(
        hold,
        ("companions", str(selected)),
        storage._derived_before(hold, ("companions", str(selected))),
    )[0]
    request.node.user_properties.append(
        ("negative_metadata_before_pending", was_confirmed)
    )


def test_real_profile_enrollment_waits_for_old_hold_then_uses_new_companion_guard(
    configured_source, request
):
    source = configured_source
    root = bootstrap.default_bootstrap_root()
    selected = source.get_cli_config_path()
    key = (os.getpid(), str(root))
    # Preserve borrowed/process owners. This fixture owns only its exact root.
    with storage._lock:
        old = storage._holds[key]
        owned_startup = storage._startups[key]
        assert owned_startup._key == key
        assert {lease for lease in storage._live_leases if lease._key == key} == {owned_startup}
        foreign_leases = tuple((lease, lease._key) for lease in storage._live_leases if lease._key != key)
        foreign_holds = {held_key: holder for held_key, holder in storage._holds.items() if held_key != key}
        foreign_retiring = set(storage._retiring_holds)
    authority = old.authority
    authority.register("profile", (selected, source.get_user_data_dir()))
    _, _, old = _warm_missing(source)
    was_confirmed = _cache_fact(old, selected)
    with pytest.raises(
        bootstrap.RecoveryRequired, match="close_unenrolled_clients_and_restart"
    ):
        control_records.bind_profile(root, selected, ("profile",), root / "admission")
    assert not any(
        row["selector"] == str(selected) for row in bootstrap._records(root)[1]
    )
    # This fixture created only this exact tmp-root startup; do not drain or
    # sweep any borrowed/global owner. Enrollment itself requires retirement.
    startup = storage._startups.pop(old.key)
    assert storage._holds[startup._key] is old and old.count == 1
    startup.close()
    assert old.stop.is_set() and not old.thread.is_alive()
    assert old.key not in storage._holds and old not in storage._retiring_holds
    control_records.bind_profile(root, selected, ("profile",), root / "admission")
    with sources._original_scope_acquisitions(source) as observed:
        with config_participants.operation(source):
            assert source.get_cli_setting("general", "users_name", "missing") == "retirement"
            assert len(observed.entries) == 1
            operation, state, leases = observed.entries[0]
            assert state.companion_guard is not None and len(leases) == 1
            guard = state.companion_guard
    assert len(observed.entries) == 1
    assert state.companion_guard is None and not guard._exit_callbacks
    assert state.holds[0] is not old and state.holds[0].names == ("profile",)
    assert operation not in raw._states and operation not in storage._raw_operations
    # Positive retirement belongs to these actual fixture owners; no global sweep.
    current = state.holds[0]
    assert all(lease._key is None and lease not in storage._live_leases for lease in leases)
    assert current.key == key and current.count == 0
    assert current.stop.is_set() and not current.thread.is_alive()
    assert current not in storage._retiring_holds and storage._holds.get(key) is not current
    with storage._lock:
        assert storage._live_leases == {lease for lease, _ in foreign_leases}
        assert all(lease._key == held_key for lease, held_key in foreign_leases)
        assert storage._holds.keys() == foreign_holds.keys()
        assert all(storage._holds[held_key] is holder for held_key, holder in foreign_holds.items())
        assert storage._retiring_holds == foreign_retiring
    request.node.user_properties.extend([
        ("foreign_baseline_lease_ids", tuple(sorted(id(lease) for lease, _ in foreign_leases))),
        ("foreign_baseline_hold_ids", tuple(sorted(id(holder) for holder in foreign_holds.values()))),
        ("foreign_baseline_retiring_ids", tuple(sorted(id(holder) for holder in foreign_retiring))),
        ("fixture_owned_holders_retired", old.stop.is_set() and current.stop.is_set()
         and not old.thread.is_alive() and not current.thread.is_alive()),
    ])
    request.node.user_properties.append(
        ("negative_metadata_before_enrollment", was_confirmed)
    )


def test_valid_shaped_profile_insertion_after_first_acquire_keeps_scope_mismatch_refusal(
    configured_source, request
):
    """A real durable tamper cannot turn old unbound evidence into a companion guard."""
    source = configured_source
    root = bootstrap.default_bootstrap_root()
    selected = source.get_cli_config_path()
    authority = storage._holds[(os.getpid(), str(root))].authority
    data = source.get_user_data_dir()
    authority.register("profile", (selected, data))
    _, _, hold = _warm_missing(source)
    was_confirmed = _cache_fact(hold, selected)
    registry = bootstrap._registry(root)
    record = control_records._Profile.model_validate(
        {
            "version": 1,
            "selector": str(selected),
            "fingerprint": bootstrap._fingerprint(selected),
            "namespaces": ["profile"],
            "roots": registry["profile"]["roots"],
        }
    )
    scope_code = inspect.unwrap(raw._scope).__code__
    acquire = storage.acquire_storage
    acquire_code, acquire_globals = acquire.__code__, acquire.__globals__
    inserted = []
    before, before_leases = selected.read_bytes(), set(storage._live_leases)
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        parent = frame.f_back
        if (
            inserted
            or event != "return"
            or frame.f_code is not acquire_code
            or parent is None
            or parent.f_code is not scope_code
            or parent.f_locals.get("source") is not source
        ):
            return
        state = parent.f_locals["state"]
        assert not state.active and not state.leases
        assert frame.f_globals is acquire_globals and result in storage._live_leases
        assert storage._holds[result._key] is hold
        # Deliberate protocol damage, not a supported enrollment bypass: use the
        # original exclusive fixed-record writer, then require the full original
        # acquisition to refuse the mismatching namespace before any config IO.
        control_records._write(
            root,
            "profile-" + bootstrap._key(str(selected)) + ".json",
            record.model_dump_json(exclude_none=True).encode(),
        )
        inserted.append(True)

    body_entered = False
    sys.setprofile(observe)
    try:
        with pytest.raises(
            bootstrap.RecoveryRequired, match="storage_scope_not_enrolled"
        ):
            with config_participants.operation(source):
                body_entered = True
                source.get_cli_setting("general", "users_name", "missing")
    finally:
        sys.setprofile(previous)
    assert not body_entered
    assert inserted == [True] and selected.read_bytes() == before
    assert set(storage._live_leases) == before_leases
    assert not raw._states and not storage._raw_operations
    assert tuple(bootstrap._records(root)[1][0]["namespaces"]) != hold.names
    assert acquire.__code__ is acquire_code and acquire.__globals__ is acquire_globals
    request.node.user_properties.append(
        ("negative_metadata_before_scope_mutation", was_confirmed)
    )


@pytest.mark.parametrize("change", ["replacement", "body"])
def test_changed_records_callback_keeps_the_original_missing_profile_route(
    configured_source, monkeypatch, change
):
    source = configured_source
    _warm_missing(source)
    original = bootstrap._records
    copied = FunctionType(
        original.__code__,
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    copied.__kwdefaults__ = original.__kwdefaults__
    calls = []
    guard_code = config_participants.companion_guard.__code__

    def replacement(root):
        if sys._getframe(1).f_code is guard_code:
            calls.append(True)
        return copied(root)

    if change == "replacement":
        monkeypatch.setattr(bootstrap, "_records", replacement)
    else:
        namespace = {}
        exec(
            "def changed(root):\n"
            "    if _test_sys._getframe(1).f_code is _test_guard_code:\n"
            "        _test_companion_calls.append(True)\n"
            "    return _test_original_records(root)\n",
            namespace,
        )
        monkeypatch.setitem(original.__globals__, "_test_sys", sys)
        monkeypatch.setitem(original.__globals__, "_test_guard_code", guard_code)
        monkeypatch.setitem(original.__globals__, "_test_companion_calls", calls)
        monkeypatch.setitem(original.__globals__, "_test_original_records", copied)
        monkeypatch.setattr(original, "__code__", namespace["changed"].__code__)
    before_leases = set(storage._live_leases)
    with sources._original_scope_acquisitions(source) as observed:
        assert _read(source) == "retirement"
    assert calls == [True]
    assert len(observed.entries) == 1 and len(observed.entries[0][2]) == 2
    assert observed.entries[0][1].companion_guard is None
    assert set(storage._live_leases) == before_leases
    assert not raw._states and not storage._raw_operations
