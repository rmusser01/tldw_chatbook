"""Screen-side review-set walker (task-28241).

Drives ``LibraryScreen._walk_active_review_set`` with a SimpleNamespace fake so
the walk logic is exercised without a live app: a real ReviewSetService over a
tmp DB provides the set, an injected liveness set stands in for the Media DB,
and the per-item actuator is recorded rather than mounting a Reader.
"""

from __future__ import annotations

import itertools
from types import MethodType, SimpleNamespace

import pytest

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


def _worker_fake(service, *, search_media, scope=None):
    """Fake screen for the entry-point workers, wiring the real methods."""
    fake = _entry_fake(service)
    fake._library_media_browse_controller = SimpleNamespace(applied_scope=scope)
    fake.app_instance.media_reading_scope_service = SimpleNamespace(
        search_media=search_media
    )

    async def run_call(call, *args, isolate_in_worker=False, **kwargs):
        return await call(*args, **kwargs)

    fake._run_library_service_call = run_call
    for name in (
        "_collect_review_pairs_from_scope",
        "_order_selected_review_pairs",
        "_create_and_open_review_set",
        "_review_these_worker",
        "_review_selected_worker",
    ):
        setattr(fake, name, MethodType(getattr(LibraryScreen, name), fake))
    fake._review_these_name = LibraryScreen._review_these_name
    return fake


@pytest.mark.asyncio
async def test_review_these_worker_pages_the_whole_result_and_lands(tmp_path):
    service = _service(tmp_path)
    pages = {
        0: [
            {"backing_media_id": 10, "title": "A"},
            {"backing_media_id": 11, "title": "B"},
        ],
        2: [{"backing_media_id": 12, "title": "C"}],
    }

    async def search_media(*, offset=0, **_kwargs):
        return {"items": pages.get(offset, []), "total": 3}

    scope = SimpleNamespace(
        query="", media_type=None, sort_by="last_modified_desc", page_size=2
    )
    fake = _worker_fake(service, search_media=search_media, scope=scope)

    await fake._review_these_worker()

    review_set = service.get_active_review_set()
    assert [item.backing_media_id for item in review_set.items] == [10, 11, 12]
    assert review_set.origin == "browse"
    assert fake._opened == ["local:media:10"]


@pytest.mark.asyncio
async def test_review_selected_worker_orders_via_the_id_allowlist(tmp_path):
    service = _service(tmp_path)
    seen_kwargs = {}

    async def search_media(**kwargs):
        seen_kwargs.update(kwargs)
        # the service returns the allowlist in browse order (newest first).
        return {
            "items": [
                {"backing_media_id": 30, "title": "Z"},
                {"backing_media_id": 20, "title": "Y"},
            ],
            "total": 2,
        }

    fake = _worker_fake(service, search_media=search_media, scope=None)

    await fake._review_selected_worker((20, 30))

    review_set = service.get_active_review_set()
    assert [item.backing_media_id for item in review_set.items] == [30, 20]
    assert review_set.origin == "selection"
    assert seen_kwargs["id_allowlist"] == [20, 30]  # bounded allowlist passed


@pytest.mark.asyncio
async def test_review_these_worker_notifies_on_failure(tmp_path):
    service = _service(tmp_path)

    async def search_media(**_kwargs):
        raise RuntimeError("db exploded")

    scope = SimpleNamespace(
        query="", media_type=None, sort_by="last_modified_desc", page_size=2
    )
    fake = _worker_fake(service, search_media=search_media, scope=scope)

    await fake._review_these_worker()  # must not raise

    assert service.get_active_review_set() is None
    assert any(severity == "error" for _msg, severity in fake._notices)


def _picker_fake(service, *, decision, live_ids=None):
    """Fake screen for the set-picker worker (task-28243).

    ``decision`` is what the (stubbed) modal resolves to; the real service,
    row collector, and activation run against the tmp DB.
    """
    fake = _entry_fake(service)
    fake._review_set_live_ids = (
        (lambda ids: {int(i) for i in ids})
        if live_ids is None
        else (lambda ids: set(live_ids))
    )

    async def run_call(call, *args, isolate_in_worker=False, **kwargs):
        result = call(*args, **kwargs)
        return await result if hasattr(result, "__await__") else result

    fake._run_library_service_call = run_call
    pushed: list = []

    async def push(rows):
        pushed.append(rows)
        return decision

    fake._push_review_set_picker = push
    fake._pushed = pushed
    for name in (
        "_review_set_picker_worker",
        "_collect_review_set_picker_rows",
        "_activate_review_set",
    ):
        setattr(fake, name, MethodType(getattr(LibraryScreen, name), fake))
    return fake


@pytest.mark.asyncio
async def test_picker_worker_lists_rows_and_opens_the_chosen_set(tmp_path):
    service = _service(tmp_path)
    first = service.create_review_set("First", origin="browse", items=[(10, "A")])
    second = service.create_review_set(
        "Second", origin="browse", items=[(20, "B"), (21, "C")]
    )
    fake = _picker_fake(service, decision=("open", first))

    await fake._review_set_picker_worker()

    rows = fake._pushed[0]
    assert {row[0] for row in rows} == {first, second}
    by_id = {row[0]: row for row in rows}
    assert by_id[second][3] is True  # newest set was the active one
    assert by_id[first][2] == "1 of 1 · 0 reviewed"
    active = service.get_active_review_set()
    assert active is not None and active.set_id == first  # switched (one-active)
    assert fake._opened == ["local:media:10"]  # landed at its cursor


@pytest.mark.asyncio
async def test_picker_worker_open_reopens_a_completed_set(tmp_path):
    service = _service(tmp_path)
    done_set = service.create_review_set("Done", origin="browse", items=[(10, "A")])
    service.mark_item_done(done_set, 10, True)
    service.refresh_completion(done_set, lambda _id: True)
    service.create_review_set("Other", origin="browse", items=[(20, "B")])
    fake = _picker_fake(service, decision=("open", done_set))

    await fake._review_set_picker_worker()

    reopened = service.get_review_set(done_set)
    assert reopened.completed_at is None  # AC#2: reopened
    assert reopened.active is True
    assert fake._opened == ["local:media:10"]


@pytest.mark.asyncio
async def test_picker_worker_dismiss_soft_deletes_without_opening(tmp_path):
    service = _service(tmp_path)
    set_id = service.create_review_set("X", origin="browse", items=[(10, "A")])
    fake = _picker_fake(service, decision=("dismiss", set_id))

    await fake._review_set_picker_worker()

    assert service.list_review_sets() == ()
    assert fake._opened == []
    assert fake._notices  # the dismissal is confirmed


@pytest.mark.asyncio
async def test_picker_worker_cancel_changes_nothing(tmp_path):
    service = _service(tmp_path)
    set_id = service.create_review_set("X", origin="browse", items=[(10, "A")])
    fake = _picker_fake(service, decision=None)

    await fake._review_set_picker_worker()

    active = service.get_active_review_set()
    assert active is not None and active.set_id == set_id
    assert fake._opened == [] and fake._notices == []


@pytest.mark.asyncio
async def test_picker_worker_open_all_tombstoned_notifies_without_activating(
    tmp_path,
):
    service = _service(tmp_path)
    gone = service.create_review_set("Gone", origin="browse", items=[(10, "A")])
    keep = service.create_review_set("Keep", origin="browse", items=[(20, "B")])
    fake = _picker_fake(service, decision=("open", gone), live_ids={20})

    await fake._review_set_picker_worker()

    active = service.get_active_review_set()
    assert active is not None and active.set_id == keep  # unchanged
    assert fake._opened == []
    assert any(severity == "warning" for _msg, severity in fake._notices)


@pytest.mark.asyncio
async def test_picker_worker_failure_notifies(tmp_path):
    service = _service(tmp_path)
    fake = _picker_fake(service, decision=None)

    def boom(*_args, **_kwargs):
        raise RuntimeError("db exploded")

    fake._collect_review_set_picker_rows = boom

    await fake._review_set_picker_worker()  # must not raise

    assert any(severity == "error" for _msg, severity in fake._notices)


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
