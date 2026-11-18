"""The repository borrower counts its warm lease under the checked coordinator lock."""

from contextlib import contextmanager
import sys

import pytest

from Tests.Backup_Recovery.test_agents_ordinary_activation_batch import _warm
from Tests.Backup_Recovery.test_agents_vector_admission import native_scope as native_scope_fixture
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.participants import repository_participant
from tldw_chatbook.Notifications.event_state_repository import EventStateRepository

native_scope = native_scope_fixture


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows lease and metadata recheck")
@pytest.mark.parametrize("maintenance", [False, True])
def test_warm_repository_lease_checks_maintenance_before_count_and_revalidates_afterward(
    native_scope, monkeypatch, maintenance,
):
    _, _, data, startup = native_scope
    repository = EventStateRepository(data / "events.sqlite")
    try:
        with repository.transaction() as connection:
            assert connection.execute("SELECT 1").fetchone()[0] == 1
        with repository_participant(repository).operation():
            path = repository.db_path
            _warm((path,))
            hold = storage._holds[startup._key]
            count = hold.count
            before = set(storage._live_leases)
            boundaries = []
            original = storage._reuse_evidence

            def observe_reuse(root, selector, selected, related, checked, selection):
                @contextmanager
                def observe_boundary(*, provenance=False):
                    with checked(provenance=provenance):
                        assert provenance
                        boundaries.append(hold.count - count)
                        if maintenance:
                            with monkeypatch.context() as change:
                                change.setattr(storage._local, "admitted", True, raising=False)
                                yield
                        else:
                            yield

                return original(root, selector, selected, related, observe_boundary, selection)

            monkeypatch.setattr(storage, "_reuse_evidence", observe_reuse)
            if maintenance:
                with pytest.raises(bootstrap.RecoveryRequired, match="maintenance_requires_owner_capability"):
                    with storage.acquire_storage(path):
                        pytest.fail("Maintenance admitted a new ordinary borrower")
                assert boundaries == [0]
            else:
                with storage.acquire_storage(path) as lease:
                    assert lease._warm_execution[0] is hold
                    assert hold.count == count + 1
                assert boundaries == [0, 1]
            assert hold.count == count
            assert set(storage._live_leases) == before
    finally:
        repository.close()
