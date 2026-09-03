"""Screen-side review-set walker (task-28241).

Drives ``LibraryScreen._walk_active_review_set`` with a SimpleNamespace fake so
the walk logic is exercised without a live app: a real ReviewSetService over a
tmp DB provides the set, an injected liveness set stands in for the Media DB,
and the per-item actuator is recorded rather than mounting a Reader.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.review_set_service import ReviewSetService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _service(tmp_path) -> ReviewSetService:
    db = LibraryCollectionsDB(tmp_path / "collections.db")
    counter = itertools.count(1)
    return ReviewSetService(
        db,
        id_factory=lambda: f"set-{next(counter)}",
        now=lambda: "2026-09-02T00:00:00Z",
    )


def _walker_fake(service: ReviewSetService, *, live_ids=None) -> SimpleNamespace:
    calls: list[tuple[str, str]] = []
    fake = SimpleNamespace(
        _review_set_service=lambda: service,
        _review_set_live_ids=(
            (lambda ids: {int(i) for i in ids})
            if live_ids is None
            else (lambda ids: set(live_ids))
        ),
        _select_library_media_reader_row=(
            lambda media_id, title, **_kwargs: calls.append((media_id, title))
        ),
    )
    fake._select_calls = calls
    return fake


def test_walk_forward_advances_marks_left_done_and_loads_next(tmp_path):
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B"), (12, "C")]
    )
    fake = _walker_fake(service)

    handled = LibraryScreen._walk_active_review_set(fake, 1)

    assert handled is True
    review_set = service.get_review_set(set_id)
    assert review_set.cursor == 1
    assert review_set.items[0].done is True  # the item we left
    assert review_set.items[1].done is False
    assert fake._select_calls == [("local:media:11", "B")]


def test_walk_back_moves_without_marking(tmp_path):
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B"), (12, "C")]
    )
    service.set_cursor(set_id, 2)
    fake = _walker_fake(service)

    LibraryScreen._walk_active_review_set(fake, -1)

    review_set = service.get_review_set(set_id)
    assert review_set.cursor == 1
    assert all(item.done is False for item in review_set.items)  # Prev never marks
    assert fake._select_calls == [("local:media:11", "B")]


def test_walk_forward_skips_a_tombstoned_target(tmp_path):
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B"), (12, "C")]
    )
    # id 11 is deleted -> forward from 0 lands on 12.
    fake = _walker_fake(service, live_ids={10, 12})

    LibraryScreen._walk_active_review_set(fake, 1)

    assert service.get_review_set(set_id).cursor == 2
    assert fake._select_calls == [("local:media:12", "C")]


def test_walk_forward_on_last_item_completes_without_reloading(tmp_path):
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B")]
    )
    # The user has already reviewed the first item and is on the last.
    service.mark_item_done(set_id, backing_media_id=10, done=True)
    service.set_cursor(set_id, 1)
    fake = _walker_fake(service)

    LibraryScreen._walk_active_review_set(fake, 1)

    review_set = service.get_review_set(set_id)
    assert review_set.cursor == 1  # clamped
    assert review_set.items[1].done is True  # completion gesture marks the last
    assert review_set.completed_at is not None  # every live item done
    assert fake._select_calls == []  # nothing new to load


def test_walk_returns_false_when_no_set_is_active(tmp_path):
    service = _service(tmp_path)  # no set created
    fake = _walker_fake(service)

    assert LibraryScreen._walk_active_review_set(fake, 1) is False
    assert fake._select_calls == []


def test_walk_returns_false_when_service_is_absent():
    fake = SimpleNamespace(_review_set_service=lambda: None, _select_calls=[])
    assert LibraryScreen._walk_active_review_set(fake, 1) is False


def test_active_review_set_progress_formats_the_line(tmp_path):
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B"), (12, "C")]
    )
    service.mark_item_done(set_id, backing_media_id=10, done=True)
    fake = _walker_fake(service)

    assert LibraryScreen._active_review_set_progress(fake) == "1 of 3 · 1 reviewed"


def test_active_review_set_progress_is_none_without_an_active_set(tmp_path):
    fake = _walker_fake(_service(tmp_path))
    assert LibraryScreen._active_review_set_progress(fake) is None


def test_exit_review_deactivates_but_keeps_the_set(tmp_path):
    service = _service(tmp_path)
    set_id = service.create_review_set("X", origin="browse", items=[(1, "a")])
    fake = _walker_fake(service)
    fake._library_media_item_traversal_active = lambda: True
    fake._sync_library_media_viewer_or_recompose = lambda: None

    LibraryScreen.action_library_media_exit_review(fake)

    assert service.get_active_review_set() is None
    assert [rs.set_id for rs in service.list_review_sets()] == [set_id]  # not deleted


def test_toggle_reviewed_marks_and_unmarks_the_loaded_item(tmp_path):
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B")]
    )
    fake = _walker_fake(service)
    fake._library_media_item_traversal_active = lambda: True
    fake._sync_library_media_viewer_or_recompose = lambda: None
    fake._library_media_reader_session = SimpleNamespace(loaded_backing_id=10)

    LibraryScreen.action_library_media_toggle_reviewed(fake)
    assert service.get_review_set(set_id).items[0].done is True

    LibraryScreen.action_library_media_toggle_reviewed(fake)
    assert service.get_review_set(set_id).items[0].done is False


def test_review_set_active_reflects_the_service(tmp_path):
    service = _service(tmp_path)
    fake = _walker_fake(service)
    assert LibraryScreen._review_set_active(fake) is False
    service.create_review_set("X", origin="browse", items=[(1, "a")])
    assert LibraryScreen._review_set_active(fake) is True
