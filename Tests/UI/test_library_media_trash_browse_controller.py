"""Non-visual source-owner contracts for exact Media Trash pages."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_media_state import MediaTrashScope
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
from tldw_chatbook.UI.Library_Modules import (
    library_media_trash_browse_controller as controller_module,
)
from tldw_chatbook.UI.Library_Modules.library_media_trash_browse_controller import (
    LibraryMediaTrashBrowseController,
)


def _item(media_id: int) -> dict[str, object]:
    return {
        "id": f"local:media:{media_id}",
        "backing_media_id": media_id,
        "title": f"Trash {media_id}",
        "media_type": "pdf",
        "trash_date": "2026-08-30T00:00:00+00:00",
    }


def _page(scope: MediaTrashScope, *, total: int) -> dict[str, object]:
    count = min(scope.page_size, max(total - scope.offset, 0))
    return {
        "items": [_item(scope.offset + index + 1) for index in range(count)],
        "total": total,
        "limit": scope.page_size,
        "offset": scope.offset,
        "types": ["pdf"],
    }


class _Screen:
    def __init__(self) -> None:
        self.pending: list[Awaitable[None]] = []
        self.worker_calls: list[dict[str, Any]] = []

    def run_worker(self, work: Awaitable[None], **kwargs: Any) -> Awaitable[None]:
        self.pending.append(work)
        self.worker_calls.append(kwargs)
        return work


class _Service:
    def __init__(
        self,
        *outcomes: object,
        by_query: dict[str, object] | None = None,
    ) -> None:
        self.outcomes = list(outcomes)
        self.by_query = dict(by_query or {})
        self.calls: list[dict[str, Any]] = []

    async def list_library_media_trash(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        query = str(kwargs["query"])
        outcome = (
            self.by_query[query] if query in self.by_query else self.outcomes.pop(0)
        )
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def _call(fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Any:
    assert kwargs.pop("isolate_in_worker") is True
    return await fn(**kwargs)


def _controller(
    *outcomes: object,
    by_query: dict[str, object] | None = None,
    active: Callable[[], bool] = lambda: True,
    sync: Callable[[str | None], None] = lambda _focus: None,
) -> tuple[LibraryMediaTrashBrowseController, _Screen, _Service]:
    screen = _Screen()
    service = _Service(*outcomes, by_query=by_query)
    controller = LibraryMediaTrashBrowseController(
        screen=screen,
        run_service_call=lambda: _call,
        media_service=lambda: service,
        sync_view=lambda: sync,
        request_is_active=active,
    )
    return controller, screen, service


@pytest.mark.asyncio
async def test_controller_canonicalizes_real_database_trash_titles(tmp_path):
    database = MediaDatabase(
        db_path=tmp_path / "canonical-trash-titles.sqlite",
        client_id="canonical-trash-titles",
    )
    try:
        padded_id, _uuid, _message = database.add_media_with_keywords(
            title="  padded title  ",
            content="padded",
            media_type="document",
            keywords=[],
        )
        blank_id, _uuid, _message = database.add_media_with_keywords(
            title="   ",
            content="blank",
            media_type="document",
            keywords=[],
        )
        assert database.mark_as_trash(padded_id)
        assert database.mark_as_trash(blank_id)
        service = MediaReadingScopeService(
            local_service=LocalMediaReadingService(database), server_service=None
        )
        screen = _Screen()
        controller = LibraryMediaTrashBrowseController(
            screen=screen,
            run_service_call=lambda: _call,
            media_service=lambda: service,
            sync_view=lambda: (lambda _focus: None),
            request_is_active=lambda: True,
        )

        controller.request(MediaTrashScope(), origin="entry", focus_identity=None)
        await screen.pending.pop()

        assert controller.state.error_copy == ""
        assert {
            item["backing_media_id"]: item["title"]
            for item in controller.state.retained_items
        } == {
            padded_id: "padded title",
            blank_id: "Untitled",
        }
        assert controller.retry(focus_identity=None) is not None
        await screen.pending.pop()
        assert controller.state.freshness == "fresh"
    finally:
        database.close_connection()


@pytest.mark.asyncio
async def test_controller_sends_exact_local_trash_scope_and_rejects_late_result():
    old_scope = MediaTrashScope(page=2)
    new_scope = MediaTrashScope(query="new")
    synced: list[str | None] = []
    controller, screen, service = _controller(
        by_query={
            "": _page(old_scope, total=21),
            "new": _page(new_scope, total=1),
        },
        sync=synced.append,
    )

    controller.request(
        old_scope,
        origin="next",
        focus_identity="#library-media-trash-next",
    )
    old = screen.pending.pop()
    controller.request(
        new_scope,
        origin="search",
        focus_identity="#library-media-trash-search",
    )
    new = screen.pending.pop()
    await new
    await old

    assert service.calls == [
        {
            "mode": "local",
            "query": "new",
            "media_type": None,
            "limit": 20,
            "offset": 0,
        },
        {
            "mode": "local",
            "query": "",
            "media_type": None,
            "limit": 20,
            "offset": 20,
        },
    ]
    assert controller.state.applied_result is not None
    assert controller.state.applied_result.scope == new_scope
    assert controller.state.selected_id == ""
    assert synced == [
        "#library-media-trash-next",
        "#library-media-trash-search",
        "#library-media-trash-search",
    ]


@pytest.mark.asyncio
async def test_retry_repeats_failed_filter_target_and_original_focus_origin():
    base = MediaTrashScope()
    failed = MediaTrashScope(query="failed")
    controller, screen, service = _controller(
        _page(base, total=1),
        RuntimeError("private-query-sentinel"),
        _page(failed, total=1),
    )
    controller.request(base, origin="entry", focus_identity=None)
    await screen.pending.pop()

    controller.request(failed, origin="search", focus_identity="search")
    await screen.pending.pop()
    assert controller.state.error_copy == "Filter not applied — showing All Trash."
    assert controller.state.failed_scope == failed
    assert controller.state.applied_result is not None
    assert controller.state.applied_result.scope == base

    controller.retry(focus_identity="retry")
    await screen.pending.pop()

    assert service.calls[-1]["query"] == "failed"
    assert service.calls[-1]["offset"] == 0
    assert controller.state.applied_result is not None
    assert controller.state.applied_result.scope == failed
    assert controller.state.selected_id == ""


@pytest.mark.asyncio
async def test_controller_clamps_once_to_authoritative_final_page():
    requested = MediaTrashScope(page=99)
    clamped = MediaTrashScope(page=3)
    controller, screen, service = _controller(
        _page(requested, total=45),
        _page(clamped, total=45),
    )

    controller.request(requested, origin="next", focus_identity="next")
    await screen.pending.pop()

    assert [call["offset"] for call in service.calls] == [1960, 40]
    assert controller.state.requested_scope == requested
    assert controller.state.applied_result is not None
    assert controller.state.applied_result.scope == clamped
    assert controller.pager.page_copy == "Page 3 of 3"


@pytest.mark.asyncio
async def test_second_shrink_marks_retained_page_stale_without_third_read():
    initial = MediaTrashScope(page=2)
    requested = MediaTrashScope(page=99)
    second_clamp = MediaTrashScope(page=3)
    controller, screen, service = _controller(
        _page(initial, total=40),
        _page(requested, total=45),
        _page(second_clamp, total=20),
    )
    controller.request(initial, origin="entry", focus_identity=None)
    await screen.pending.pop()
    retained = controller.state.retained_items

    controller.request(requested, origin="next", focus_identity="next")
    await screen.pending.pop()

    assert [call["offset"] for call in service.calls] == [20, 1960, 40]
    assert controller.state.retained_items is retained
    assert controller.state.freshness == "stale"
    assert controller.state.stale_copy == "Source changed again; try again."
    assert controller.pager.title_count is None
    assert controller.pager.range_copy == "List may be out of date"
    assert controller.pager.retry_visible is True


@pytest.mark.asyncio
async def test_invalidation_and_inactive_route_fence_late_local_completion():
    active = True
    synced: list[str | None] = []
    scope = MediaTrashScope()
    controller, screen, _service = _controller(
        _page(scope, total=1),
        active=lambda: active,
        sync=synced.append,
    )
    controller.request(scope, origin="entry", focus_identity="entry")
    late = screen.pending.pop()

    controller.invalidate()
    active = False
    await late

    assert controller.state.applied_result is None
    assert synced == ["entry"]
    assert controller.request(scope, origin="entry", focus_identity="entry") is None
    assert screen.pending == []


@pytest.mark.asyncio
async def test_failure_logging_is_metadata_only(monkeypatch: pytest.MonkeyPatch):
    sentinel = "private-query-sentinel"
    logged: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        controller_module.logger,
        "warning",
        lambda *args: logged.append(args),
    )
    controller, screen, _service = _controller(RuntimeError(sentinel))

    controller.request(
        MediaTrashScope(query=sentinel),
        origin="search",
        focus_identity=None,
    )
    await screen.pending.pop()

    rendered = " ".join(str(value) for call in logged for value in call)
    assert logged
    assert sentinel not in rendered
    assert "list_library_media_trash" in rendered
    assert "RuntimeError" in rendered


def test_controller_uses_trash_specific_exclusive_worker_group():
    controller, screen, _service = _controller(_page(MediaTrashScope(), total=1))

    controller.request(MediaTrashScope(), origin="entry", focus_identity=None)

    assert screen.worker_calls == [
        {"exclusive": True, "group": "library-media-trash-browse"}
    ]
    screen.pending.pop().close()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_controller_mutation_surface_preserves_failure_and_refreshes_commit():
    scope = MediaTrashScope()
    controller, screen, service = _controller(
        _page(scope, total=2),
        _page(scope, total=1),
    )
    controller.request(scope, origin="entry", focus_identity=None)
    await screen.pending.pop()
    controller.select("local:media:2")

    target = controller.open_delete_confirmation()
    assert target is not None
    failure_claim = controller.claim_mutation()
    assert failure_claim is not None
    assert failure_claim.target == target
    controller.finish_mutation_failure(failure_claim, "Could not delete this item.")
    assert controller.state.selected_id == target.stable_id
    assert controller.state.freshness == "fresh"

    assert controller.open_delete_confirmation() == target
    commit_claim = controller.claim_mutation()
    assert commit_claim is not None
    assert commit_claim.target == target
    controller.finish_mutation_commit(commit_claim, "Deleted 'Trash 2' permanently.")
    assert controller.state.freshness == "stale"
    assert controller.state.loading is True
    assert [item["id"] for item in controller.state.retained_items] == ["local:media:1"]

    controller.request_after_mutation(commit_claim, focus_identity="fallback")
    await screen.pending.pop()

    assert service.calls[-1]["offset"] == 0
    assert controller.state.freshness == "fresh"
    assert controller.state.committed_notice == "Deleted 'Trash 2' permanently."


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["commit", "failure"])
async def test_invalidated_mutation_completion_cannot_publish_or_refresh(outcome):
    scope = MediaTrashScope()
    synced: list[str | None] = []
    controller, screen, _service = _controller(
        _page(scope, total=1),
        _page(scope, total=0),
        sync=synced.append,
    )
    controller.request(scope, origin="entry", focus_identity=None)
    await screen.pending.pop()
    claim = controller.claim_mutation()
    assert claim is not None
    controller.invalidate()
    invalidated_state = controller.state
    synced.clear()

    if outcome == "commit":
        controller.finish_mutation_commit(claim, "Restored 'Trash 1'.")
    else:
        controller.finish_mutation_failure(claim, "Could not restore this item.")
    pending = controller.request_after_mutation(focus_identity="fallback")
    if pending is not None:
        pending.close()
        screen.pending.remove(pending)

    assert controller.state is invalidated_state
    assert synced == []
    assert pending is None
    assert screen.pending == []


@pytest.mark.asyncio
async def test_mutation_claim_keeps_captured_stable_identity_for_duplicate_titles():
    """Visible duplicate text cannot redirect an already captured mutation."""
    scope = MediaTrashScope()
    payload = _page(scope, total=2)
    payload["items"][0]["title"] = "Duplicate title"
    payload["items"][1]["title"] = "Duplicate title"
    controller, screen, _service = _controller(payload)
    controller.request(scope, origin="entry", focus_identity=None)
    await screen.pending.pop()

    controller.select("local:media:2")
    captured = controller.open_delete_confirmation()
    assert captured is not None
    assert captured.stable_id == "local:media:2"
    assert captured.backing_media_id == 2

    claimed = controller.claim_mutation()

    assert claimed is not None
    assert claimed.target == captured
    assert claimed.target.stable_id != "local:media:1"
    assert controller.state.mutation_pending is True
    assert controller.state.confirmation_target is None


@pytest.mark.asyncio
async def test_precommit_failure_retains_exact_fresh_page_and_action_authority():
    scope = MediaTrashScope(page=2)
    controller, screen, _service = _controller(_page(scope, total=40))
    controller.request(scope, origin="entry", focus_identity=None)
    await screen.pending.pop()
    controller.select("local:media:22")
    claim = controller.claim_mutation()
    assert claim is not None
    target = claim.target
    applied = controller.state.applied_result
    retained = controller.state.retained_items

    controller.finish_mutation_failure(claim, "Could not delete this item.")

    assert controller.state.applied_result is applied
    assert controller.state.retained_items is retained
    assert controller.state.selected_id == "local:media:22"
    assert controller.state.freshness == "fresh"
    assert controller.state.loading is False
    assert controller.state.mutation_pending is False
    assert controller.state.error_copy == "Could not delete this item."
    assert controller.pager.title_count == 40
    assert controller.pager.range_copy == "21-40 of 40"
    assert controller.pager.page_copy == "Page 2 of 2"


@pytest.mark.asyncio
async def test_committed_mutation_withdraws_exact_claims_before_refresh():
    scope = MediaTrashScope(page=2)
    controller, screen, _service = _controller(_page(scope, total=40))
    controller.request(scope, origin="entry", focus_identity=None)
    await screen.pending.pop()
    controller.select("local:media:22")
    claim = controller.claim_mutation()
    assert claim is not None
    target = claim.target

    controller.finish_mutation_commit(claim, "Restored 'Trash 22'.")

    assert controller.state.freshness == "stale"
    assert controller.state.loading is True
    assert controller.state.selected_id == ""
    assert "local:media:22" not in {
        str(item["id"]) for item in controller.state.retained_items
    }
    assert controller.pager.title_count is None
    assert controller.pager.range_copy == "List may be out of date"
    assert controller.pager.page_copy == ""
