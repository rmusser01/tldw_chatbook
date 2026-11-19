"""AC77 proposal controls; real ordinary owners, unchanged settle and deadlines."""

import os
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest

from Tests.Backup_Recovery import test_config_companion_missing_record_evidence as missing
from Tests.Backup_Recovery import test_raw_related_source_preparation as sources
from tldw_chatbook.Backup_Recovery import bootstrap, config_participants, control_records
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage

configured_source = sources.configured_source
local_root = sources.local_root
pytestmark = missing.pytestmark


def _verdict(source):
    try:
        with config_participants.operation(source):
            return "allowed", source.get_cli_setting("general", "users_name", "missing")
    except bootstrap.RecoveryRequired as error:
        return "refused", str(error)


@pytest.mark.parametrize("mutation", ["registry", "publication", "source"])
def test_engaged_absence_and_fresh_route_agree_after_mutation(
    configured_source, monkeypatch, mutation, request
):
    source = configured_source
    root = bootstrap.default_bootstrap_root()
    selected = source.get_cli_config_path()
    authority = storage._holds[(os.getpid(), str(root))].authority
    # This is the original six-control setup, not a companion enrollment.
    # The selected record remains absent and the current holder remains unbound.
    authority.register("profile", (selected, source.get_user_data_dir()))
    _, _, hold = missing._warm_missing(source)
    key = ("companions", str(selected))
    assert missing._cache_fact(hold, selected)
    assert hold.derived_evidence[key][1] == (None, None)
    before, leases_before = selected.read_bytes(), set(storage._live_leases)
    calls = []
    if mutation == "registry":
        foreign = selected.parent / "foreign-data"
        foreign.mkdir(mode=0o700)
        authority.register("foreign", (foreign,))
    elif mutation == "publication":
        control_records.register_pending(
            root, "companion-oracle", ("profile",), root.parent / "control", (selected,)
        )
    else:
        original = bootstrap._records
        guard_code = config_participants.companion_guard.__code__

        def changed(root):
            if sys._getframe(1).f_code is guard_code:
                calls.append(True)
            return original(root)

        monkeypatch.setattr(bootstrap, "_records", changed)
    # This assertion follows an observed true original reuse in _warm_missing.
    # Neither an empty cache nor a cold refusal can establish the prerequisite.
    if mutation != "source":
        assert not storage._derived_reuse(hold, key, storage._derived_before(hold, key))[0]
    enabled = _verdict(source)
    enabled_guard_calls = len(calls)
    with monkeypatch.context() as fresh:
        fresh.setattr(storage, "_EVIDENCE_REUSE", False)
        disabled = _verdict(source)
    assert enabled == disabled
    if mutation == "publication":
        assert enabled[0] == "refused"
    else:
        assert enabled == ("allowed", "retirement")
    if mutation == "source":
        assert enabled_guard_calls == 1 and len(calls) == 2
        assert not missing._cache_fact(hold, selected)
    assert selected.read_bytes() == before
    assert set(storage._live_leases) == leases_before
    assert not raw._states and not storage._raw_operations
    request.node.user_properties.extend([
        ("engaged_absence_before_mutation", True),
        ("reuse_enabled_verdict", repr(enabled)),
        ("reuse_disabled_verdict", repr(disabled)),
    ])


def test_control_reader_keyword_mutation_keeps_fresh_missing_route(
    configured_source, monkeypatch
):
    source = configured_source
    _, selected, hold = missing._warm_missing(source)
    before, leases_before = selected.read_bytes(), set(storage._live_leases)
    keywords = bootstrap._control_records.__kwdefaults__
    assert keywords == {"activation": True}
    # _records explicitly passes activation=False, so this changes eligibility,
    # not the original reader result. The mutation is after actual engagement.
    monkeypatch.setitem(keywords, "activation", False)
    with sources._original_scope_acquisitions(source) as observed:
        with missing._companion_record_calls(source) as records:
            assert missing._read(source) == "retirement"
    assert records == [(False, True, (control_records.UNBOUND_NAMESPACE,))]
    assert len(observed.entries) == 1 and len(observed.entries[0][2]) == 2
    assert observed.entries[0][1].companion_guard is None
    assert not missing._cache_fact(hold, selected)
    assert selected.read_bytes() == before and set(storage._live_leases) == leases_before
    assert not raw._states and not storage._raw_operations


def test_fixed_intent_absence_is_stamped_and_marker_appearance_invalidates(
    configured_source
):
    source = configured_source
    root, selected, hold = missing._warm_missing(source)
    evidence = storage._metadata_evidence(hold, (selected, selected.parent))
    assert evidence is not None
    intents = {
        child.parent / control_records._DIRECTORY_CREATION_INTENT
        for child in (root, *root.parents) if child != Path(root.anchor)
    }
    assert intents <= {path for path, _ in evidence.content}
    assert all(stamp is None for path, stamp in evidence.content if path in intents)
    assert all(
        stamp is not None or path == selected or path in intents
        for path, stamp in evidence.content
    )
    marker = root.parent / control_records._DIRECTORY_CREATION_INTENT
    assert not marker.exists()
    # Exact original exclusive fixed writer, in this fixture-owned tmp parent.
    control_records._write(root.parent, marker.name, b"bootstrap directory creation\n")
    try:
        key = ("companions", str(selected))
        assert not storage._derived_reuse(hold, key, storage._derived_before(hold, key))[0]
        assert storage._metadata_evidence(hold, (selected, selected.parent)) is None
        assert evidence.observe() != evidence.stamps()
    finally:
        marker.unlink()


def test_missing_registered_root_still_prevents_complete_metadata(configured_source):
    source = configured_source
    root = bootstrap.default_bootstrap_root()
    selected = source.get_cli_config_path()
    foreign = selected.parent / "registered-data"
    foreign.mkdir(mode=0o700)
    storage._holds[(os.getpid(), str(root))].authority.register("foreign", (foreign,))
    _, _, hold = missing._warm_missing(source)
    moved = foreign.with_name("registered-data-moved")
    foreign.rename(moved)
    try:
        key = ("companions", str(selected))
        assert not storage._derived_reuse(hold, key, storage._derived_before(hold, key))[0]
        assert storage._metadata_evidence(hold, (selected, selected.parent)) is None
    finally:
        moved.rename(foreign)


def test_epoch_invalidation_runs_fresh_and_publishes_after_temporary_close(
    configured_source
):
    source = configured_source
    _, selected, hold = missing._warm_missing(source)
    bootstrap.advance_admission_epoch()
    key = ("companions", str(selected))
    assert not storage._derived_reuse(hold, key, storage._derived_before(hold, key))[0]
    guard_code = config_participants.companion_guard.__code__
    close_code = ExitStack.close.__code__
    note = storage._note_derived
    note_code, note_globals = note.__code__, note.__globals__
    previous = sys.getprofile()
    closes, publications = [], []

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        # Observe only the two actual consumers, before reading parent locals.
        if not ((frame.f_code is close_code and event == "return")
                or (frame.f_code is note_code and event == "call")):
            return
        parent = frame.f_back
        if parent is None or parent.f_code is not guard_code:
            return
        state = raw._states.get(parent.f_locals["operation"])
        if state is None or state.source is not source:
            return
        guard = parent.f_locals["guard"]
        if frame.f_code is close_code and event == "return":
            assert frame.f_locals["self"] is guard and not guard._exit_callbacks
            closes.append(guard)
        if frame.f_code is note_code and event == "call":
            assert frame.f_globals is note_globals
            assert parent.f_locals["record"] is None and guard in closes
            assert not guard._exit_callbacks
            assert frame.f_locals["value"] == (None, None)
            assert frame.f_locals["entry"] is not None
            publications.append(guard)

    with missing._companion_record_calls(source) as records:
        records_observer = sys.getprofile()
        previous = records_observer
        sys.setprofile(observe)
        try:
            assert missing._read(source) == "retirement"
        finally:
            sys.setprofile(previous)
    assert records == [(False, True, (control_records.UNBOUND_NAMESPACE,))]
    assert len(publications) == 1 and closes == publications
    # Qualify subsequent real engagement without assuming a stronger epoch
    # reconfirmation rule than the unchanged generic helper implements.
    missing._warm_missing(source)
    assert storage._note_derived is note and note.__code__ is note_code
    assert note.__globals__ is note_globals
    assert not raw._states and not storage._raw_operations
