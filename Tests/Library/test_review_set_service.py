"""ReviewSetService persistence contract (task-28240).

Uses a real in-memory LibraryCollectionsDB (one held connection per test, so
``:memory:`` persists across the service's transactions). A deterministic
id/clock keeps assertions stable. The Media-DB resolve is injected as an
``is_live`` predicate so these tests stay free of the Media DB.
"""

from __future__ import annotations

import itertools

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.review_set_service import ReviewSetService


def _service() -> ReviewSetService:
    db = LibraryCollectionsDB(":memory:")
    counter = itertools.count(1)
    return ReviewSetService(
        db,
        id_factory=lambda: f"set-{next(counter)}",
        now=lambda: "2026-09-02T00:00:00Z",
    )


_ALL_LIVE = lambda _backing_id: True  # noqa: E731


def test_schema_upgrades_to_v4():
    db = LibraryCollectionsDB(":memory:")
    assert db.get_schema_version() == 4


def test_create_pins_ordered_items_and_activates():
    svc = _service()
    set_id = svc.create_review_set(
        "Talks", origin="browse", items=[(10, "A"), (11, "B"), (12, "C")]
    )

    review_set = svc.get_review_set(set_id)
    assert [item.backing_media_id for item in review_set.items] == [10, 11, 12]
    assert [item.position for item in review_set.items] == [0, 1, 2]
    assert review_set.cursor == 0
    assert review_set.active is True
    assert review_set.completed_at is None
    assert all(item.done is False for item in review_set.items)


def test_create_rejects_empty_items():
    # task-28241 review: an empty set can't be navigated or completed, so it is
    # never created (and never deactivates the current active set).
    svc = _service()
    existing = svc.create_review_set("Keep", origin="browse", items=[(1, "a")])
    with pytest.raises(ValueError):
        svc.create_review_set("Empty", origin="browse", items=[])
    assert svc.get_active_review_set().set_id == existing  # unchanged


def test_create_rejects_unknown_origin_and_blank_name():
    svc = _service()
    with pytest.raises(ValueError):
        svc.create_review_set("X", origin="not-an-origin", items=[(1, "a")])
    with pytest.raises(ValueError):
        svc.create_review_set("   ", origin="browse", items=[(1, "a")])


def test_activate_unknown_id_does_not_clear_the_active_set():
    # task-28241 review: activating a stale/missing/dismissed id must not leave
    # the app with no active set.
    svc = _service()
    active = svc.create_review_set("A", origin="browse", items=[(1, "a")])

    svc.activate("does-not-exist")
    assert svc.get_active_review_set().set_id == active

    dismissed = svc.create_review_set("B", origin="browse", items=[(2, "b")])
    svc.activate(active)  # back to A
    svc.dismiss(dismissed)
    svc.activate(dismissed)  # dismissed -> no-op, A stays active
    assert svc.get_active_review_set().set_id == active


def test_list_review_sets_respects_a_limit():
    svc = _service()
    for index in range(3):
        svc.create_review_set(f"S{index}", origin="browse", items=[(index, "x")])
    assert len(svc.list_review_sets(limit=2)) == 2


def test_create_dedupes_by_backing_id_keeping_first():
    svc = _service()
    set_id = svc.create_review_set(
        "Dupes", origin="selection", items=[(10, "A"), (11, "B"), (10, "A-again")]
    )
    review_set = svc.get_review_set(set_id)
    assert [item.backing_media_id for item in review_set.items] == [10, 11]


def test_create_deactivates_the_previous_active():
    svc = _service()
    first = svc.create_review_set("A", origin="browse", items=[(1, "a")])
    second = svc.create_review_set("B", origin="browse", items=[(2, "b")])

    assert svc.get_review_set(first).active is False
    assert svc.get_review_set(second).active is True
    assert svc.get_active_review_set().set_id == second


def test_advance_persists_the_cursor():
    svc = _service()
    set_id = svc.create_review_set(
        "X", origin="browse", items=[(1, "a"), (2, "b"), (3, "c")]
    )

    assert svc.advance(set_id, step=1, is_live=_ALL_LIVE) == 1
    assert svc.get_review_set(set_id).cursor == 1
    assert svc.advance(set_id, step=1, is_live=_ALL_LIVE) == 2
    assert svc.get_review_set(set_id).cursor == 2


def test_advance_skips_a_tombstoned_item():
    svc = _service()
    set_id = svc.create_review_set(
        "X", origin="browse", items=[(1, "a"), (2, "b"), (3, "c")]
    )
    is_live = lambda backing_id: backing_id != 2  # noqa: E731

    # From position 0, forward past the dead middle item -> position 2.
    assert svc.advance(set_id, step=1, is_live=is_live) == 2
    assert svc.get_review_set(set_id).cursor == 2


def test_mark_done_then_completion_over_live_items():
    svc = _service()
    set_id = svc.create_review_set(
        "X", origin="browse", items=[(1, "a"), (2, "b")]
    )

    svc.mark_item_done(set_id, backing_media_id=1, done=True)
    assert svc.refresh_completion(set_id, is_live=_ALL_LIVE) is False
    assert svc.get_review_set(set_id).completed_at is None

    svc.mark_item_done(set_id, backing_media_id=2, done=True)
    assert svc.refresh_completion(set_id, is_live=_ALL_LIVE) is True
    assert svc.get_review_set(set_id).completed_at is not None


def test_completion_ignores_a_tombstoned_unreviewed_item():
    svc = _service()
    set_id = svc.create_review_set("X", origin="browse", items=[(1, "a"), (2, "b")])
    svc.mark_item_done(set_id, backing_media_id=1, done=True)

    # item 2 is not done, but it is deleted -> the live set is all done.
    is_live = lambda backing_id: backing_id != 2  # noqa: E731
    assert svc.refresh_completion(set_id, is_live=is_live) is True


def test_all_tombstoned_set_never_completes():
    svc = _service()
    set_id = svc.create_review_set("X", origin="browse", items=[(1, "a")])
    svc.mark_item_done(set_id, backing_media_id=1, done=True)

    assert svc.refresh_completion(set_id, is_live=lambda _b: False) is False
    assert svc.get_review_set(set_id).completed_at is None


def test_dismiss_soft_deletes_and_deactivates():
    svc = _service()
    set_id = svc.create_review_set("X", origin="browse", items=[(1, "a")])

    svc.dismiss(set_id)
    assert svc.get_active_review_set() is None
    assert svc.list_review_sets() == ()
    assert svc.get_review_set(set_id) is None  # dismissed sets are hidden


def test_reopen_clears_completion():
    svc = _service()
    set_id = svc.create_review_set("X", origin="browse", items=[(1, "a")])
    svc.mark_item_done(set_id, backing_media_id=1, done=True)
    svc.refresh_completion(set_id, is_live=_ALL_LIVE)
    assert svc.get_review_set(set_id).completed_at is not None

    svc.reopen(set_id)
    assert svc.get_review_set(set_id).completed_at is None
    assert svc.get_review_set(set_id).items[0].done is True  # marks kept
