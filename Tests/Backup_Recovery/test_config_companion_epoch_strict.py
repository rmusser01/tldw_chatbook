"""Separate, unrun strict-epoch contract diagnostic; no semantics authorized."""
import sys
from contextlib import contextmanager

from Tests.Backup_Recovery import test_config_companion_missing_record_evidence as missing
from Tests.Backup_Recovery import test_raw_related_source_preparation as sources
from Tests.Backup_Recovery.test_config_companion_review_regressions import _actual_reuses
from tldw_chatbook.Backup_Recovery import bootstrap, config_participants
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage

configured_source = sources.configured_source
local_root = sources.local_root
pytestmark = missing.pytestmark


@contextmanager
def _actual_missing_notes(source, hold, key):
    function = storage._note_derived
    code, namespace = function.__code__, function.__globals__
    guard = config_participants.companion_guard
    guard_code, guard_globals = guard.__code__, guard.__globals__
    previous, rows = sys.getprofile(), []

    def observe(frame, event, result):
        if event == "return" and frame.f_code is code:
            parent = frame.f_back
            if (frame.f_globals is namespace and parent is not None
                and parent.f_code is guard_code and parent.f_globals is guard_globals
                and frame.f_locals.get("hold") is hold and frame.f_locals.get("key") == key):
                state = raw._states.get(parent.f_locals.get("operation"))
                if state is not None and state.source is source:
                    entry = frame.f_locals.get("entry")
                    before = frame.f_locals.get("before")
                    rows.append((type(entry) is storage._Evidence,
                                 None if entry is None else entry.epoch,
                                 None if entry is None else entry.confirmed,
                                 before[0], frame.f_locals.get("value") == (None, None)))
        if previous is not None:
            previous(frame, event, result)

    sys.setprofile(observe)
    try:
        yield rows
    finally:
        sys.setprofile(previous)
        assert storage._note_derived is function and function.__code__ is code
        assert function.__globals__ is namespace
        assert config_participants.companion_guard is guard and guard.__code__ is guard_code


def test_strict_epoch_requires_two_new_complete_observations_after_bare_advance(
    configured_source,
):
    source = configured_source
    _, selected, hold = missing._warm_missing(source)
    key = ("companions", str(selected))
    before_bytes, before_leases, before_count = selected.read_bytes(), set(storage._live_leases), hold.count
    old = hold.derived_evidence[key][0]
    assert old.confirmed and old.epoch == bootstrap._admission_epoch
    bootstrap.advance_admission_epoch()
    epoch = bootstrap._admission_epoch
    assert epoch == old.epoch + 1
    assert not storage._derived_reuse(hold, key, storage._derived_before(hold, key))[0]
    with missing._companion_record_calls(source) as records:
        with _actual_missing_notes(source, hold, key) as notes:
            assert missing._read(source) == "retirement"
            first = hold.derived_evidence[key][0]
            first_confirmed = first.confirmed
            assert missing._read(source) == "retirement"
            second = hold.derived_evidence[key][0]
            second_confirmed = second.confirmed
    # Always complete both real reads/cleanup before evaluating the stronger rule.
    with _actual_reuses(config_participants.companion_guard, hold, key) as reuses:
        with missing._companion_record_calls(source) as warm_records:
            assert missing._read(source) == "retirement"
    assert selected.read_bytes() == before_bytes and set(storage._live_leases) == before_leases
    assert hold.count == before_count and storage._holds.get(hold.key) is hold
    assert storage._hold_serving(hold) and hold.thread.is_alive()
    assert not raw._states and not storage._raw_operations
    assert bootstrap._admission_epoch == epoch
    assert first.epoch == second.epoch == epoch and first is not old
    assert not warm_records and reuses == [True]
    # This is deliberately separate from the unchanged weaker R2 epoch test.
    assert records == [(False, True, hold.names), (False, True, hold.names)]
    assert notes == [(True, epoch, False, epoch, True), (True, epoch, True, epoch, True)]
    assert not first_confirmed and second_confirmed and second is not first
