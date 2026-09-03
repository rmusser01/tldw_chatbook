"""Screen-side review-set walker (task-28241).

Drives ``LibraryScreen._walk_active_review_set`` with a SimpleNamespace fake so
the walk logic is exercised without a live app: a real ReviewSetService over a
tmp DB provides the set, an injected liveness set stands in for the Media DB,
and the per-item actuator is recorded rather than mounting a Reader.
"""

from __future__ import annotations

import itertools
from types import MethodType, SimpleNamespace

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


def _walker_fake(
    service: ReviewSetService, *, live_ids=None, loaded=None
) -> SimpleNamespace:
    """Fake screen for the walker.

    ``loaded`` is the backing id the Reader is currently showing (Qodo #2333:
    the walk marks/advances from the DISPLAYED item, not the persisted cursor).
    """
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
        _library_media_reader_session=SimpleNamespace(loaded_backing_id=loaded),
    )
    fake._select_calls = calls
    return fake


def test_walk_forward_advances_marks_left_done_and_loads_next(tmp_path):
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B"), (12, "C")]
    )
    fake = _walker_fake(service, loaded=10)

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
    fake = _walker_fake(service, loaded=12)

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
    fake = _walker_fake(service, live_ids={10, 12}, loaded=10)

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
    fake = _walker_fake(service, loaded=11)

    LibraryScreen._walk_active_review_set(fake, 1)

    review_set = service.get_review_set(set_id)
    assert review_set.cursor == 1  # clamped
    assert review_set.items[1].done is True  # completion gesture marks the last
    assert review_set.completed_at is not None  # every live item done
    assert fake._select_calls == []  # nothing new to load


def test_walk_resumes_at_cursor_without_marking_when_reader_is_off_set(tmp_path):
    # Qodo #2333: if the Reader is showing a non-set item (fresh entry, or a
    # browse item), the first ] must NOT mark an unseen item -- it resumes the
    # set at its cursor.
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B"), (12, "C")]
    )
    service.set_cursor(set_id, 1)
    fake = _walker_fake(service, loaded=999)  # 999 is not in the set

    LibraryScreen._walk_active_review_set(fake, 1)

    review_set = service.get_review_set(set_id)
    assert all(item.done is False for item in review_set.items)  # nothing marked
    assert fake._select_calls == [("local:media:11", "B")]  # loaded the cursor item
    assert review_set.cursor == 1  # unchanged (already a live position)


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


def test_review_set_live_ids_filters_deleted_and_trashed(tmp_path):
    # Qodo #2333: exercise the REAL Media-DB liveness query (parameter binding,
    # deleted/is_trash filtering, an unknown id), not the injected fake.
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    media_db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="t")
    ids = []
    for index in range(3):
        media_id, _uuid, _msg = media_db.add_media_with_keywords(
            title=f"Item {index}",
            media_type="video",
            content=f"content {index}",
            url=f"http://example/{index}",
        )
        ids.append(media_id)
    media_db.soft_delete_media(ids[1])  # deleted = 1
    media_db.mark_as_trash(ids[2])  # is_trash = 1

    fake = SimpleNamespace(
        app_instance=SimpleNamespace(media_db=media_db),
        _REVIEW_SET_LIVENESS_BATCH=LibraryScreen._REVIEW_SET_LIVENESS_BATCH,
    )
    live = LibraryScreen._review_set_live_ids(fake, ids + [999_999])

    assert live == {ids[0]}  # only the live item; deleted/trashed/unknown excluded


def test_review_set_live_ids_treats_ids_live_when_media_db_absent():
    fake = SimpleNamespace(
        app_instance=SimpleNamespace(media_db=None),
        _REVIEW_SET_LIVENESS_BATCH=LibraryScreen._REVIEW_SET_LIVENESS_BATCH,
    )
    assert LibraryScreen._review_set_live_ids(fake, [1, 2, 3]) == {1, 2, 3}


def _entry_fake(service):
    opened: list[str] = []
    notices: list[tuple[str, str]] = []
    fake = SimpleNamespace(
        _review_set_service=lambda: service,
        _open_library_media_viewer=lambda media_id: opened.append(media_id),
        app_instance=SimpleNamespace(
            notify=lambda message, severity="information": notices.append(
                (message, severity)
            )
        ),
    )
    fake._notify_review_set = MethodType(
        LibraryScreen._notify_review_set, fake
    )
    fake._opened = opened
    fake._notices = notices
    return fake


def test_create_and_open_review_set_creates_activates_and_lands(tmp_path):
    service = _service(tmp_path)
    fake = _entry_fake(service)

    LibraryScreen._create_and_open_review_set(
        fake, "Talks", "browse", [(10, "A"), (11, "B")]
    )

    review_set = service.get_active_review_set()
    assert review_set is not None
    assert [item.backing_media_id for item in review_set.items] == [10, 11]
    assert fake._opened == ["local:media:10"]  # landed on the first item


def test_create_and_open_review_set_empty_notifies_and_makes_no_set(tmp_path):
    service = _service(tmp_path)
    fake = _entry_fake(service)

    LibraryScreen._create_and_open_review_set(fake, "X", "browse", [])

    assert service.get_active_review_set() is None
    assert fake._opened == []
    assert fake._notices and fake._notices[0][1] == "warning"


def test_create_and_open_review_set_warns_on_truncation(tmp_path):
    service = _service(tmp_path)
    fake = _entry_fake(service)

    LibraryScreen._create_and_open_review_set(
        fake, "X", "browse", [(index, f"t{index}") for index in range(600)]
    )

    review_set = service.get_active_review_set()
    assert len(review_set.items) == 500
    assert any(severity == "warning" for _msg, severity in fake._notices)


def test_review_these_name_from_scope():
    assert (
        LibraryScreen._review_these_name(
            SimpleNamespace(query="cats", media_type=None)
        )
        == 'Search: "cats"'
    )
    assert (
        LibraryScreen._review_these_name(
            SimpleNamespace(query="", media_type="video")
        )
        == "video items"
    )
    assert (
        LibraryScreen._review_these_name(
            SimpleNamespace(query="", media_type=None)
        )
        == "All media"
    )


def test_review_set_active_reflects_the_service(tmp_path):
    service = _service(tmp_path)
    fake = _walker_fake(service)
    assert LibraryScreen._review_set_active(fake) is False
    service.create_review_set("X", origin="browse", items=[(1, "a")])
    assert LibraryScreen._review_set_active(fake) is True
