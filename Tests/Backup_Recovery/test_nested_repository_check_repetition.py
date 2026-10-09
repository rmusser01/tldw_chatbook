"""Same-path nesting keeps one real fresh source proof, with no authority reuse."""

import collections
import sys

import pytest

from Tests.Backup_Recovery import test_participant_lifetimes as repository_fixtures
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.participants import _repository_participant
from tldw_chatbook.Notifications.event_state_repository import EventStateRepository

local_root = repository_fixtures.local_root


@pytest.fixture
def repository(tmp_path):
    owner = EventStateRepository(tmp_path / "events.sqlite")
    try:
        yield owner
    finally:
        owner.close()


@pytest.mark.usefixtures("local_root")
def test_same_path_nested_owner_performs_one_fresh_native_proof(
    repository, record_property
):
    from tldw_chatbook.Utils import windows_files

    participant = _repository_participant(repository)
    codes = {storage._Operation.check.__code__: "full_path_checks"}
    if sys.platform == "win32":
        codes[windows_files._Native.open_handle.__code__] = "native_opens"
        codes[windows_files._Native.security.__code__] = "native_security"
    counts = collections.Counter()

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1

    with participant.operation() as outer:
        assert outer in storage._operations and outer.lease in storage._live_leases
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            with participant.operation() as nested:
                assert nested is outer
        finally:
            sys.setprofile(previous)
    for name, value in counts.items():
        record_property(name, value)
    if sys.platform == "win32":
        assert counts["native_opens"] > 0 and counts["native_security"] > 0
    assert counts["full_path_checks"] == 1


@pytest.mark.usefixtures("local_root")
def test_changed_nested_target_keeps_both_actual_path_proofs(repository):
    participant = _repository_participant(repository)
    paths = []

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code is storage._Operation.check.__code__:
            paths.append(frame.f_locals["path"])

    with participant.operation() as outer:
        selected = participant.path
        participant.path = selected.with_name("unselected.sqlite")
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            with pytest.raises(RecoveryRequired, match="operation_path_outside_scope"):
                with participant.operation():
                    pytest.fail("changed path inherited the prior operation")
        finally:
            sys.setprofile(previous)
            participant.path = selected
        assert paths == [outer.path, selected.with_name("unselected.sqlite")]
        storage._check_operation(outer, selected)


@pytest.mark.usefixtures("local_root")
def test_same_owner_continues_pause_while_distinct_owner_refuses(repository, tmp_path):
    participant = _repository_participant(repository)
    other = EventStateRepository(tmp_path / "other.sqlite")
    target = _repository_participant(other)
    pause = None
    try:
        with participant.operation() as outer:
            pause = storage._begin_local_pause()
            with participant.operation() as nested:
                assert nested is outer
                storage._check_operation(nested, repository.db_path)
            with pytest.raises(RecoveryRequired, match="storage_locally_paused"):
                with target.operation():
                    pytest.fail("independent owner inherited admitted work")
            assert storage._operation_local.operation is outer
            storage._check_operation(outer, repository.db_path)
        assert pause is not None
        assert outer not in storage._operations
    finally:
        if pause is not None:
            pause.resume()
        other.close()
