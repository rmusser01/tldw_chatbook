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
    notices: list[tuple[str, str]] = []
    syncs: list[bool] = []
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
        _notify_review_set=(
            lambda message, severity="information": notices.append(
                (message, severity)
            )
        ),
        _sync_library_media_viewer_or_recompose=lambda: syncs.append(True),
    )
    fake._select_calls = calls
    fake._notices = notices
    fake._syncs = syncs
    for name in (
        "_walk_active_review_set_unguarded",
        "_toggle_reviewed_unguarded",
    ):
        setattr(fake, name, MethodType(getattr(LibraryScreen, name), fake))
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


# --- the finish line (task-30041, critique 2026-09-03) -----------------------


def test_check_action_enables_walk_keys_whenever_a_set_is_active(tmp_path):
    """]/[ must gate on the ACTIVE SET, not browse-row adjacency.

    Critique P1: at the last browse row _library_media_adjacent_row returns
    None, which disabled the documented final-] completion gesture at the
    binding layer (and misfired whenever set order diverged from browse
    order). With a set active the walk clamps safely, so the keys stay live.
    """
    fake = SimpleNamespace(
        _library_media_item_traversal_active=lambda: True,
        _review_set_active=lambda: True,
        _library_media_adjacent_row=lambda direction: None,  # last browse row
    )
    assert (
        LibraryScreen.check_action(fake, "library_media_next_item", ()) is True
    )
    assert (
        LibraryScreen.check_action(fake, "library_media_prev_item", ()) is True
    )

    fake._review_set_active = lambda: False
    assert (
        LibraryScreen.check_action(fake, "library_media_next_item", ()) is False
    )


def test_check_action_fails_closed_when_storage_errors(tmp_path):
    """A storage error in the binding gate degrades to browse, never crashes.

    Qodo #2346: check_action runs on every key resolution and footer render,
    so an exception here would crash-loop the screen. The gate fails closed
    (browse adjacency), and the explicit gesture paths carry the notices.
    """

    class _Boom:
        def get_active_review_set(self):
            raise RuntimeError("db gone")

    fake = SimpleNamespace(
        _library_media_item_traversal_active=lambda: True,
        _review_set_service=lambda: _Boom(),
        _library_media_adjacent_row=lambda direction: None,
    )
    fake._review_set_active = MethodType(LibraryScreen._review_set_active, fake)

    assert (
        LibraryScreen.check_action(fake, "library_media_next_item", ()) is False
    )


def test_walk_final_forward_syncs_the_footer_and_notifies_completion(tmp_path):
    """The completion gesture refreshes the chrome and gets a real moment.

    Critique P1: the clamp branch marked the last item and refreshed
    completion but skipped the viewer sync, so the footer stayed stale at
    exactly the finish; and completing a set had no acknowledgment at all.
    """
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B")]
    )
    service.mark_item_done(set_id, backing_media_id=10, done=True)
    service.set_cursor(set_id, 1)
    fake = _walker_fake(service, loaded=11)

    LibraryScreen._walk_active_review_set(fake, 1)

    review_set = service.get_review_set(set_id)
    assert review_set.completed_at is not None
    assert fake._syncs == [True]  # footer refreshed at the finish
    assert any("All 2 reviewed" in message for message, _sev in fake._notices)


def test_walk_completion_notice_fires_only_on_the_completing_walk(tmp_path):
    """A ] pressed after completion must not re-announce."""
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B")]
    )
    service.mark_item_done(set_id, backing_media_id=10, done=True)
    service.set_cursor(set_id, 1)
    fake = _walker_fake(service, loaded=11)

    LibraryScreen._walk_active_review_set(fake, 1)  # completes
    LibraryScreen._walk_active_review_set(fake, 1)  # idempotent extra press

    completion_notices = [
        message for message, _sev in fake._notices if "All 2 reviewed" in message
    ]
    assert len(completion_notices) == 1


# --- no silent failures (task-30042, critique 2026-09-03) --------------------


def test_walk_failure_notifies_and_consumes_instead_of_raising(tmp_path):
    """Storage exploding mid-walk surfaces an error, never a crash."""

    class _Boom:
        def get_active_review_set(self):
            raise RuntimeError("db gone")

    fake = _walker_fake(_service(tmp_path))
    fake._review_set_service = lambda: _Boom()

    handled = LibraryScreen._walk_active_review_set(fake, 1)

    assert handled is True  # consumed; no browse fallback double-action
    assert any(severity == "error" for _msg, severity in fake._notices)


def test_toggle_reviewed_failure_notifies(tmp_path):
    class _Boom:
        def get_active_review_set(self):
            raise RuntimeError("db gone")

    fake = _walker_fake(_service(tmp_path))
    fake._review_set_service = lambda: _Boom()
    fake._library_media_item_traversal_active = lambda: True

    LibraryScreen.action_library_media_toggle_reviewed(fake)  # must not raise

    assert any(severity == "error" for _msg, severity in fake._notices)


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
    fake._REVIEW_SET_STORAGE_UNAVAILABLE = (
        LibraryScreen._REVIEW_SET_STORAGE_UNAVAILABLE
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


def test_create_notifies_when_it_pauses_an_in_progress_active_set(tmp_path):
    """task-31238: creating a set silently deactivated the active one.

    A walk in progress just stopped being resumable-by-] with no
    acknowledgment. The create now names the paused set and its live
    progress, and points at the resume path.
    """
    service = _service(tmp_path)
    old = service.create_review_set(
        "Read later", origin="read_later", items=[(10, "A"), (11, "B")]
    )
    service.mark_item_done(old, 10, True)
    service.set_cursor(old, 1)
    fake = _entry_fake(service)
    fake._review_set_live_ids = lambda ids: {int(i) for i in ids}

    LibraryScreen._create_and_open_review_set(
        fake, "New", "browse", [(20, "C")]
    )

    assert any(
        "Paused" in message
        and "Read later" in message
        and "2 of 2 · 1 reviewed" in message
        for message, _severity in fake._notices
    )


def test_create_stays_quiet_when_no_active_set_was_displaced(tmp_path):
    """task-31238 AC#3: no pause notice when nothing was displaced."""
    fake = _entry_fake(_service(tmp_path))

    LibraryScreen._create_and_open_review_set(
        fake, "New", "browse", [(20, "C")]
    )

    assert not any("Paused" in message for message, _severity in fake._notices)


def test_create_stays_quiet_when_the_displaced_set_was_complete(tmp_path):
    """A finished set being displaced loses nothing — no pause notice."""
    service = _service(tmp_path)
    old = service.create_review_set("Done", origin="browse", items=[(10, "A")])
    service.mark_item_done(old, 10, True)
    service.refresh_completion(old, lambda _id: True)
    fake = _entry_fake(service)
    fake._review_set_live_ids = lambda ids: {int(i) for i in ids}

    LibraryScreen._create_and_open_review_set(
        fake, "New", "browse", [(20, "C")]
    )

    assert not any("Paused" in message for message, _severity in fake._notices)


def test_create_from_selection_exits_select_mode_before_landing(
    tmp_path, monkeypatch
):
    """task-31233: the Select-path create leaves select mode, then lands.

    Critique #3 P1: "Review selected" toasted "Reviewing 2 items." but left
    the user in select mode with a blank reader — the viewer open at the end
    of _create_and_open_review_set never surfaced because select mode was
    still armed (it is cleared only by Done/bulk-delete). The exit must come
    BEFORE the open, sync the CANVAS (the viewer open syncs only the viewer
    in place — live-verified stale select toolbar without it), and not
    announce a discard — the selection was consumed, not thrown away.
    """
    import tldw_chatbook.UI.Screens.library_screen as screen_module

    service = _service(tmp_path)
    fake = _entry_fake(service)
    fake._library_media_select_mode = True
    events: list[object] = []
    fake._exit_library_media_select_mode = (
        lambda *, announce_discard: events.append(("exit", announce_discard))
    )
    monkeypatch.setattr(
        screen_module,
        "_sync_library_canvas",
        lambda _screen, kind, **_kw: events.append(("canvas_sync", kind)),
    )
    fake._open_library_media_viewer = lambda media_id: events.append(
        ("open", media_id)
    )

    LibraryScreen._create_and_open_review_set(
        fake, "2 selected items", "selection", [(10, "A"), (11, "B")]
    )

    assert events == [
        ("exit", False),
        ("canvas_sync", "media"),
        ("open", "local:media:10"),
    ]


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
    syncs: list[bool] = []
    fake._sync_library_media_viewer_or_recompose = lambda: syncs.append(True)
    fake._syncs = syncs
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
    # task-31238: the detail label carries the created date after progress.
    assert by_id[first][2] == "1 of 1 · 0 reviewed · 2026-09-02"
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
async def test_picker_worker_open_persists_a_resolved_tombstoned_cursor(tmp_path):
    # Qodo #2337: the cursor sat on a tombstone; opening resolves to the next
    # live item AND persists that position, so a later restore of the deleted
    # item cannot yank a subsequent resume back to the stale cursor.
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B")]
    )
    service.create_review_set("Other", origin="browse", items=[(20, "C")])
    fake = _picker_fake(service, decision=("open", set_id), live_ids={11, 20})

    await fake._review_set_picker_worker()

    assert fake._opened == ["local:media:11"]
    assert service.get_review_set(set_id).cursor == 1  # persisted resolve


@pytest.mark.asyncio
async def test_picker_worker_dismiss_soft_deletes_and_arms_the_undo_receipt(
    tmp_path, monkeypatch
):
    import tldw_chatbook.UI.Screens.library_screen as screen_module

    service = _service(tmp_path)
    set_id = service.create_review_set("X", origin="browse", items=[(10, "A")])
    fake = _picker_fake(service, decision=("dismiss", set_id))
    canvas_syncs: list[str] = []
    monkeypatch.setattr(
        screen_module,
        "_sync_library_canvas",
        lambda _screen, kind, **_kw: canvas_syncs.append(kind),
    )

    await fake._review_set_picker_worker()

    assert service.list_review_sets() == ()
    assert fake._opened == []
    # task-31236: the confirmation is the undo receipt, not a toast -- a
    # one-click dismissal of a mid-walk set must be recoverable in place.
    # (set_id, name, was_active) so Undo can restore the activation too.
    assert fake._library_media_review_dismiss_receipt == (set_id, "X", True)
    # Dismissing the ACTIVE set must refresh the Reader chrome -- the footer
    # kept advertising "] next in set · 1 of 3" after a live dismissal
    # (live-verified 2026-09-02). The CANVAS sync is separate and required:
    # the viewer seam updates only the viewer in place, so the receipt row
    # never mounted without it (live-verified 2026-09-04).
    assert fake._syncs == [True]
    assert canvas_syncs == ["media"]


@pytest.mark.asyncio
async def test_dismiss_undo_worker_restores_the_set(tmp_path, monkeypatch):
    """task-31236 AC#2: Undo brings the set back — marks, cursor, active."""
    import tldw_chatbook.UI.Screens.library_screen as screen_module

    monkeypatch.setattr(
        screen_module, "_sync_library_canvas", lambda *_a, **_kw: None
    )
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B")]
    )
    service.mark_item_done(set_id, 10, True)
    service.set_cursor(set_id, 1)
    fake = _picker_fake(service, decision=("dismiss", set_id))
    await fake._review_set_picker_worker()
    assert service.list_review_sets() == ()

    fake._review_dismiss_undo_worker = MethodType(
        LibraryScreen._review_dismiss_undo_worker, fake
    )
    await fake._review_dismiss_undo_worker(
        fake._library_media_review_dismiss_receipt
    )

    restored = service.get_active_review_set()
    assert restored is not None and restored.set_id == set_id
    assert restored.cursor == 1
    assert [item.done for item in restored.items] == [True, False]
    assert fake._library_media_review_dismiss_receipt is None
    assert fake._syncs == [True, True]  # chrome refreshed after the restore


@pytest.mark.asyncio
async def test_dismiss_undo_worker_failure_notifies(tmp_path):
    service = _service(tmp_path)
    fake = _picker_fake(service, decision=None)
    fake._review_dismiss_undo_worker = MethodType(
        LibraryScreen._review_dismiss_undo_worker, fake
    )

    def boom(*_args, **_kwargs):
        raise RuntimeError("db gone")

    fake._review_set_service = boom

    await fake._review_dismiss_undo_worker(("set-1", "X", True))  # no raise

    assert any(severity == "error" for _msg, severity in fake._notices)


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
async def test_picker_worker_read_later_builds_a_set_in_saved_order(tmp_path):
    """The read-later decision pins the queue in saved order (task-28244).

    Ids come from ``list_read_it_later_media_ids`` (saved_at DESC); titles
    come from the bounded allowlist query, whose browse-sort order is then
    discarded in favor of the saved order.
    """
    service = _service(tmp_path)
    fake = _picker_fake(service, decision=("read_later", ""))
    fake.app_instance.media_db = SimpleNamespace(
        list_read_it_later_media_ids=lambda **_kwargs: [30, 10, 20]
    )

    async def search_media(**kwargs):
        assert kwargs["id_allowlist"] == [30, 10, 20]
        return {
            "items": [  # browse order differs from saved order on purpose
                {"backing_media_id": 10, "title": "A"},
                {"backing_media_id": 20, "title": "B"},
                {"backing_media_id": 30, "title": "C"},
            ],
            "total": 3,
        }

    fake.app_instance.media_reading_scope_service = SimpleNamespace(
        search_media=search_media
    )
    fake._library_media_browse_controller = SimpleNamespace(applied_scope=None)
    for name in (
        "_order_selected_review_pairs",
        "_review_read_later_pairs",
        "_create_and_open_review_set",
    ):
        setattr(fake, name, MethodType(getattr(LibraryScreen, name), fake))

    await fake._review_set_picker_worker()

    review_set = service.get_active_review_set()
    assert review_set is not None
    assert review_set.origin == "read_later"
    assert [item.backing_media_id for item in review_set.items] == [30, 10, 20]
    assert fake._opened == ["local:media:30"]


@pytest.mark.asyncio
async def test_picker_worker_read_later_empty_notifies_without_a_set(tmp_path):
    """An empty read-later queue notices instead of creating an empty set."""
    service = _service(tmp_path)
    fake = _picker_fake(service, decision=("read_later", ""))
    fake.app_instance.media_db = SimpleNamespace(
        list_read_it_later_media_ids=lambda **_kwargs: []
    )
    for name in (
        "_order_selected_review_pairs",
        "_review_read_later_pairs",
        "_create_and_open_review_set",
    ):
        setattr(fake, name, MethodType(getattr(LibraryScreen, name), fake))

    await fake._review_set_picker_worker()

    assert service.get_active_review_set() is None
    assert fake._opened == []
    assert any(severity == "warning" for _msg, severity in fake._notices)


def test_list_read_it_later_media_ids_honors_the_limit(tmp_path):
    """The real DB lister bounds its query when a limit is passed (Qodo #2340)."""
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    media_db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="t")
    ids = []
    for index in range(3):
        media_id, _uuid, _msg = media_db.add_media_with_keywords(
            title=f"Item {index}", media_type="document", content=f"c{index}"
        )
        ids.append(media_id)
        media_db.save_media_to_read_it_later(media_id)

    bounded = media_db.list_read_it_later_media_ids(limit=2)
    # Same-second saves tie-break by media_id DESC: the two newest rows.
    assert bounded == [ids[2], ids[1]]
    assert media_db.list_read_it_later_media_ids() == [ids[2], ids[1], ids[0]]


@pytest.mark.asyncio
async def test_read_later_pairs_run_against_the_real_media_db(tmp_path):
    """The worker seam reads a REAL Media DB's read-later order end to end.

    Qodo #2340: only the (remote-shaped) search service is stubbed -- the
    lister runs the real query with the worker's real kwargs, so a bad limit
    kwarg or a broken ordering would fail here, not just against a fake.
    """
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    media_db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="t")
    ids = []
    for index in range(3):
        media_id, _uuid, _msg = media_db.add_media_with_keywords(
            title=f"Doc {index}", media_type="document", content=f"c{index}"
        )
        ids.append(media_id)
    media_db.save_media_to_read_it_later(ids[0])
    media_db.save_media_to_read_it_later(ids[2])

    service = _service(tmp_path)
    fake = _picker_fake(service, decision=("read_later", ""))
    fake.app_instance.media_db = media_db

    async def search_media(**kwargs):
        titles = {ids[0]: "Doc 0", ids[2]: "Doc 2"}
        return {
            "items": [
                {"backing_media_id": bid, "title": titles[bid]}
                for bid in sorted(kwargs["id_allowlist"])  # browse order differs
            ],
            "total": len(kwargs["id_allowlist"]),
        }

    fake.app_instance.media_reading_scope_service = SimpleNamespace(
        search_media=search_media
    )
    fake._library_media_browse_controller = SimpleNamespace(applied_scope=None)
    for name in (
        "_order_selected_review_pairs",
        "_review_read_later_pairs",
        "_create_and_open_review_set",
    ):
        setattr(fake, name, MethodType(getattr(LibraryScreen, name), fake))

    await fake._review_set_picker_worker()

    review_set = service.get_active_review_set()
    # saved_at DESC with same-second tie-break by media_id DESC -> [ids2, ids0]
    assert [item.backing_media_id for item in review_set.items] == [
        ids[2],
        ids[0],
    ]
    assert fake._opened == [f"local:media:{ids[2]}"]


@pytest.mark.asyncio
async def test_picker_press_with_unavailable_storage_notifies(tmp_path):
    """Pressing Sets with no storage must say so, never silently no-op.

    task-30042 (critique P1 + user ruling: silent failure is never
    acceptable): a dead button is indistinguishable from the feature not
    existing.
    """
    fake = _picker_fake(_service(tmp_path), decision=None)
    fake._review_set_service = lambda: None

    await fake._review_set_picker_worker()

    assert any(
        "storage" in message.lower() and severity == "error"
        for message, severity in fake._notices
    )


def test_create_with_unavailable_storage_notifies(tmp_path):
    """A build that resolved items but has no storage must not vanish."""
    fake = _entry_fake(_service(tmp_path))
    fake._review_set_service = lambda: None

    LibraryScreen._create_and_open_review_set(
        fake, "Talks", "browse", [(10, "A")]
    )

    assert fake._opened == []
    assert any(
        "storage" in message.lower() and severity == "error"
        for message, severity in fake._notices
    )


@pytest.mark.asyncio
async def test_auto_resume_failure_notifies(tmp_path):
    """Auto-resume exceptions surface an error instead of the old silence."""
    service = _service(tmp_path)
    service.create_review_set("X", origin="browse", items=[(10, "A")])
    fake = _auto_resume_fake(service)

    def boom():
        raise RuntimeError("db gone")

    fake._resolve_active_review_set_landing = boom

    await fake._auto_resume_review_set_worker()  # must not raise

    assert any(severity == "error" for _msg, severity in fake._notices)


@pytest.mark.asyncio
async def test_auto_resume_with_unavailable_storage_warns_once(tmp_path):
    """Missing storage on media entry warns once per session, not per visit."""
    fake = _auto_resume_fake(_service(tmp_path))
    fake._review_set_service = lambda: None

    await fake._auto_resume_review_set_worker()
    await fake._auto_resume_review_set_worker()

    warnings = [
        message
        for message, severity in fake._notices
        if severity == "warning" and "storage" in message.lower()
    ]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_picker_worker_failure_notifies(tmp_path):
    service = _service(tmp_path)
    fake = _picker_fake(service, decision=None)

    def boom(*_args, **_kwargs):
        raise RuntimeError("db exploded")

    fake._collect_review_set_picker_rows = boom

    await fake._review_set_picker_worker()  # must not raise

    assert any(severity == "error" for _msg, severity in fake._notices)


def _auto_resume_fake(service, *, live_ids=None):
    """Fake screen for the auto-resume worker (task-28245)."""
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
    fake._library_selected_row_id = "browse-media"
    fake._library_media_view = "list"
    fake.is_current = True
    for name in (
        "_auto_resume_review_set_worker",
        "_resolve_active_review_set_landing",
    ):
        setattr(fake, name, MethodType(getattr(LibraryScreen, name), fake))
    return fake


@pytest.mark.asyncio
async def test_auto_resume_opens_the_cursor_item_on_every_entry(tmp_path):
    # task-31234 (SUPERSEDES task-28245's once-per-set AC — user ruling at
    # the critique #3 close): EVERY entry to the media area with an active
    # set lands on its cursor item. The once-per-set gate re-armed the
    # banner over whatever document was showing on re-entry — the frame
    # restored, the item not, and the first ] became a silent sync.
    service = _service(tmp_path)
    set_id = service.create_review_set(
        "X", origin="browse", items=[(10, "A"), (11, "B")]
    )
    service.set_cursor(set_id, 1)
    fake = _auto_resume_fake(service)

    await fake._auto_resume_review_set_worker()
    assert fake._opened == ["local:media:11"]

    await fake._auto_resume_review_set_worker()
    assert fake._opened == ["local:media:11", "local:media:11"]  # every entry


def test_auto_resume_runs_in_its_own_worker_group():
    """Auto-resume must not share the exclusive creation group (Qodo #2342).

    ``library_review_set`` is the exclusive group the entry-point build
    workers run in; if auto-resume joined it, clicking the Media rail while
    "Review these" was still paging would CANCEL the set creation.
    """

    async def noop():
        return None

    recorded: dict = {}

    def run_worker(coro, **kwargs):
        recorded.update(kwargs)
        coro.close()

    fake = SimpleNamespace(
        run_worker=run_worker, _auto_resume_review_set_worker=noop
    )
    LibraryScreen._maybe_auto_resume_review_set(fake)

    assert recorded["group"] != "library_review_set"
    assert recorded["exclusive"] is True  # still debounces its own re-entries


@pytest.mark.asyncio
async def test_auto_resume_is_a_no_op_without_an_active_set(tmp_path):
    fake = _auto_resume_fake(_service(tmp_path))
    await fake._auto_resume_review_set_worker()
    assert fake._opened == [] and fake._notices == []


@pytest.mark.asyncio
async def test_auto_resume_aborts_when_the_user_moved_away(tmp_path):
    # task-28245 AC#3: if the initial-tab switch (or the user) yanked the
    # screen off the media list before the resolve finished, do not open.
    service = _service(tmp_path)
    service.create_review_set("X", origin="browse", items=[(10, "A")])
    fake = _auto_resume_fake(service)
    fake._library_media_view = "viewer"

    await fake._auto_resume_review_set_worker()
    assert fake._opened == []


@pytest.mark.asyncio
async def test_auto_resume_skips_an_all_tombstoned_set_quietly(tmp_path):
    service = _service(tmp_path)
    service.create_review_set("X", origin="browse", items=[(10, "A")])
    fake = _auto_resume_fake(service, live_ids=set())

    await fake._auto_resume_review_set_worker()
    assert fake._opened == [] and fake._notices == []  # convenience, no nag


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
