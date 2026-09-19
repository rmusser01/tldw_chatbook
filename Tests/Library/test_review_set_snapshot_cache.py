"""The active review-set snapshot caches by service.revision (TASK-32804.4).

The footer/banner/progress readers each loaded the whole active set + a liveness
query per key resolution and render (tens of ms per keypress on a 500-item set).
They now share one _active_review_set_snapshot() keyed on service.revision, which
every ReviewSetService write bumps -- so within one revision the readers share a
single load, and a write re-loads once for all of them.
"""

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


class _Item:
    def __init__(self, backing):
        self.backing_media_id = backing


class _Set:
    items = [_Item(1), _Item(2), _Item(3)]


class _Service:
    def __init__(self):
        self.revision = 5
        self.get_calls = 0

    def get_active_review_set(self):
        self.get_calls += 1
        return _Set()


class _Stub:
    def __init__(self, service):
        self._service = service
        self.live_calls = 0

    def _review_set_service(self):
        return self._service

    def _review_set_live_ids(self, ids):
        self.live_calls += 1
        return set(ids)


def test_snapshot_loads_once_per_revision_and_reloads_on_bump():
    service = _Service()
    stub = _Stub(service)

    snap1 = LibraryScreen._active_review_set_snapshot(stub)
    snap2 = LibraryScreen._active_review_set_snapshot(stub)

    # Same revision -> one load shared across readers.
    assert snap1 is snap2
    assert service.get_calls == 1
    assert stub.live_calls == 1
    review_set, live_ids = snap1
    assert live_ids == {1, 2, 3}

    # A write bumps the revision -> exactly one reload.
    service.revision = 6
    snap3 = LibraryScreen._active_review_set_snapshot(stub)
    assert service.get_calls == 2
    assert stub.live_calls == 2
    assert snap3 is not snap1


def test_snapshot_is_none_when_no_service():
    class _NoService(_Stub):
        def _review_set_service(self):
            return None

    assert LibraryScreen._active_review_set_snapshot(_NoService(None)) is None
