from __future__ import annotations

import inspect

import pytest

from tldw_chatbook.Library.library_skills_state import (
    SkillBrowseScope,
    build_skill_browse_error,
    build_skill_browse_result,
)
from tldw_chatbook.UI.Library_Modules.library_skills_browse_controller import (
    LibrarySkillsBrowseController,
)


def _summary(name: str, *, blocked: bool = False) -> dict[str, object]:
    return {
        "name": name,
        "description": f"{name} description",
        "argument_hint": None,
        "user_invocable": True,
        "disable_model_invocation": False,
        "context": "inline",
        "trust_status": "quarantined_modified" if blocked else "trusted",
        "trust_blocked": blocked,
    }


class _Screen:
    def __init__(self) -> None:
        self.pending = None

    def run_worker(self, awaitable, **_kwargs):
        self.pending = awaitable
        return awaitable


async def _run_service(callable_obj, *args, isolate_in_worker=False, **kwargs):
    result = callable_obj(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


@pytest.mark.asyncio
async def test_controller_calls_explicit_local_page_and_exposes_exact_pager():
    calls: list[dict[str, object]] = []

    class Service:
        async def list_skills(self, **kwargs):
            calls.append(kwargs)
            return {
                "skills": [_summary(f"skill-{index:02d}") for index in range(20, 25)],
                "count": 5,
                "total": 25,
                "limit": 20,
                "offset": 20,
                "blocked_total": 1,
                "first_blocked_skill_name": "blocked-off-page",
            }

    screen = _Screen()
    synced = []
    controller = LibrarySkillsBrowseController(
        screen=screen,
        run_service_call=lambda: _run_service,
        skills_service=lambda: Service(),
        sync_view=lambda: lambda result, focus: synced.append((result, focus)),
        request_is_active=lambda: True,
    )

    controller.request(
        SkillBrowseScope(query="needle", sort="status", page=2), focus_identity="next"
    )
    await screen.pending

    assert calls == [
        {
            "mode": "local",
            "query": "needle",
            "sort": "status",
            "limit": 20,
            "offset": 20,
        }
    ]
    assert controller.pager.range_copy == "21-25 of 25"
    assert controller.pager.page_copy == "Page 2 of 2"
    assert controller.blocked_total == 1
    assert controller.first_blocked_skill_name == "blocked-off-page"
    assert synced[-1][1] == "next"


@pytest.mark.asyncio
async def test_controller_refetches_one_clamped_final_page():
    offsets: list[int] = []

    class Service:
        async def list_skills(self, **kwargs):
            offset = kwargs["offset"]
            offsets.append(offset)
            items = [_summary("skill-20")] if offset == 20 else []
            return {
                "skills": items,
                "count": len(items),
                "total": 21,
                "limit": 20,
                "offset": offset,
                "blocked_total": 0,
                "first_blocked_skill_name": None,
            }

    screen = _Screen()
    controller = LibrarySkillsBrowseController(
        screen=screen,
        run_service_call=lambda: _run_service,
        skills_service=lambda: Service(),
        sync_view=lambda: lambda *_args: None,
        request_is_active=lambda: True,
    )

    controller.request(SkillBrowseScope(page=4), focus_identity=None)
    await screen.pending

    assert offsets == [60, 20]
    assert controller.visible_result.page == 2
    assert controller.pager.range_copy == "21-21 of 21"


@pytest.mark.asyncio
async def test_controller_clamps_restored_high_page_to_first_page_when_source_empty():
    offsets: list[int] = []

    class Service:
        async def list_skills(self, **kwargs):
            offset = kwargs["offset"]
            offsets.append(offset)
            return {
                "skills": [],
                "count": 0,
                "total": 0,
                "limit": 20,
                "offset": offset,
                "blocked_total": 0,
                "first_blocked_skill_name": None,
            }

    screen = _Screen()
    controller = LibrarySkillsBrowseController(
        screen=screen,
        run_service_call=lambda: _run_service,
        skills_service=lambda: Service(),
        sync_view=lambda: lambda *_args: None,
        request_is_active=lambda: True,
    )

    controller.request(SkillBrowseScope(page=4), focus_identity=None)
    await screen.pending

    assert offsets == [60, 0]
    assert controller.visible_result.page == 1
    assert controller.visible_result.status == "empty"


@pytest.mark.asyncio
async def test_controller_reclamps_when_source_shrinks_during_clamp_fetch():
    offsets: list[int] = []
    totals = iter((21, 1, 1))

    class Service:
        async def list_skills(self, **kwargs):
            offset = kwargs["offset"]
            total = next(totals)
            offsets.append(offset)
            items = [_summary("alpha")] if offset == 0 and total == 1 else []
            return {
                "skills": items,
                "count": len(items),
                "total": total,
                "limit": 20,
                "offset": offset,
                "blocked_total": 0,
                "first_blocked_skill_name": None,
            }

    screen = _Screen()
    controller = LibrarySkillsBrowseController(
        screen=screen,
        run_service_call=lambda: _run_service,
        skills_service=lambda: Service(),
        sync_view=lambda: lambda *_args: None,
        request_is_active=lambda: True,
    )

    controller.request(SkillBrowseScope(page=4), focus_identity=None)
    await screen.pending

    assert offsets == [60, 20, 0]
    assert controller.visible_result.page == 1
    assert tuple(item["name"] for item in controller.visible_result.items) == ("alpha",)


def test_controller_rejects_late_generation_and_fences_inactive_route():
    active = True
    screen = _Screen()
    controller = LibrarySkillsBrowseController(
        screen=screen,
        run_service_call=lambda: _run_service,
        skills_service=lambda: None,
        sync_view=lambda: lambda *_args: None,
        request_is_active=lambda: active,
    )
    old = controller.result
    controller.begin(SkillBrowseScope(page=2))

    assert controller.apply(old, focus_identity=None) is False
    active = False
    assert controller.apply(controller.result, focus_identity=None) is False


def test_controller_stale_retention_suppresses_total_and_disables_boundaries():
    screen = _Screen()
    controller = LibrarySkillsBrowseController(
        screen=screen,
        run_service_call=lambda: _run_service,
        skills_service=lambda: None,
        sync_view=lambda: lambda *_args: None,
        request_is_active=lambda: True,
    )
    ready = build_skill_browse_result(
        SkillBrowseScope(),
        {
            "skills": [_summary("alpha")],
            "count": 1,
            "total": 1,
            "limit": 20,
            "offset": 0,
            "blocked_total": 0,
            "first_blocked_skill_name": None,
        },
    )
    controller.apply(ready, focus_identity=None)

    controller.retain_stale_items(
        (_summary("alpha"),),
        stale_copy="Saved, but the refreshed list could not be loaded.",
    )

    assert controller.pager.title_count is None
    assert controller.pager.previous_disabled is True
    assert controller.pager.next_disabled is True
    assert controller.pager.retry_visible is True


def test_failed_filter_retains_prior_rows_as_stale_and_inert():
    screen = _Screen()
    controller = LibrarySkillsBrowseController(
        screen=screen,
        run_service_call=lambda: _run_service,
        skills_service=lambda: None,
        sync_view=lambda: lambda *_args: None,
        request_is_active=lambda: True,
    )
    ready = build_skill_browse_result(
        SkillBrowseScope(),
        {
            "skills": [_summary("alpha")],
            "count": 1,
            "total": 1,
            "limit": 20,
            "offset": 0,
            "blocked_total": 0,
            "first_blocked_skill_name": None,
        },
    )
    controller.apply(ready, focus_identity=None)
    failed_scope = SkillBrowseScope(query="needle")
    request_token = controller.begin(failed_scope)
    failed = build_skill_browse_error(
        failed_scope,
        request_token=request_token,
        error="Couldn't load Skills.",
    )

    assert controller.apply(failed, focus_identity=None) is True
    assert controller.retained_items == (_summary("alpha"),)
    assert controller.pager.title_count is None
    assert controller.pager.previous_disabled is True
    assert controller.pager.next_disabled is True
    assert controller.pager.retry_visible is True
    assert controller.stale_copy == "Filter wasn't applied; showing previous results."
