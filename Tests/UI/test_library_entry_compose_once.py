from __future__ import annotations

import asyncio
import statistics
import time

import pytest

from tldw_chatbook.UI.Screens.library_screen import (
    LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS,
    LibraryEntryReconcileResult,
    LibraryScreen,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_SKILLS,
    LIBRARY_ROW_CREATE_STUDY,
)
from tldw_chatbook.Widgets.Library import (
    LibraryConversationsCanvas,
    LibraryLandingCanvas,
    LibraryLandingCanvasState,
    LibraryLandingRecentItem,
    LibraryStudyHandoffCanvas,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _compositor_text(screen: LibraryScreen) -> str:
    """Return only text actually painted in the current terminal frame."""
    return "\n".join(
        "".join(segment.text for segment in strip)
        for strip in screen._compositor.render_strips()
    )


def _assert_widget_text_is_painted(
    screen: LibraryScreen, selector: str, expected: str
) -> None:
    """Assert a widget and its literal label are inside the rendered viewport."""
    widget = screen.query_one(selector)
    viewport = screen.region
    assert viewport.contains_region(widget.region)
    lines = [
        "".join(segment.text for segment in strip)
        for strip in screen._compositor.render_strips()
    ]
    painted = "\n".join(
        line[widget.region.x : widget.region.right]
        for line in lines[widget.region.y : widget.region.bottom]
    )
    assert expected in painted, (
        f"{selector} region={widget.region!r} display={widget.display!r} "
        f"painted={painted!r} frame={_compositor_text(screen)!r}"
    )


def _apply_changed_snapshot(
    screen: LibraryScreen,
    *,
    conversations: tuple[dict[str, object], ...] | None = None,
    notes: tuple[dict[str, object], ...] | None = None,
    study_decks: int | None = None,
) -> bool:
    """Apply one literal changed snapshot through the production boundary."""
    records = dict(screen._local_source_records)
    counts = dict(screen._local_source_counts)
    if conversations is not None:
        records["conversations"] = conversations
        counts["conversations"] = len(conversations)
    if notes is not None:
        records["notes"] = notes
        counts["notes"] = len(notes)
    study_counts = dict(screen._library_study_counts)
    if study_decks is not None:
        study_counts["study_decks"] = study_decks
    return screen._apply_local_source_snapshot(
        records,
        counts,
        dict(screen._local_source_total_known),
        screen._library_lookup_error,
        screen._library_lookup_recovery_state,
        study_counts,
    )


@pytest.mark.asyncio
async def test_landing_snapshot_sync_retains_actions_focus_and_updates_recents():
    """Recomposing the inline landing branch would replace all three actions."""
    app = _build_test_app()
    conversations = _two_conversations()
    _seed_conversations(app, conversations[:1])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)
        import_button = screen.query_one("#library-hub-action-import")
        search_button = screen.query_one("#library-hub-action-search")
        new_note_button = screen.query_one("#library-hub-action-new-note")
        search_button.focus()
        await pilot.pause()

        changed = _apply_changed_snapshot(
            screen,
            conversations=(conversations[1], conversations[0]),
        )
        await pilot.pause()
        await pilot.pause()

        assert changed is True
        assert screen.query_one("#library-landing-canvas") is landing
        assert screen.query_one("#library-hub-action-import") is import_button
        assert screen.query_one("#library-hub-action-search") is search_button
        assert screen.query_one("#library-hub-action-new-note") is new_note_button
        assert screen.focused is search_button
        assert "Conversations (2)" in str(
            screen.query_one("#library-hub-counts").renderable
        )
        recent = screen.query_one("#library-hub-recent-conversations")
        assert getattr(recent, "record_id", "") == "chat-2"


@pytest.mark.asyncio
async def test_landing_deferred_recents_converge_on_latest_state(monkeypatch):
    """Capturing recents before the deferred await would mount stale rows."""
    app = _build_test_app()
    _seed_conversations(app, [])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)
        first = LibraryLandingCanvasState(
            purpose=landing.state.purpose,
            counts_line="Conversations (1)",
            recent_items=(
                LibraryLandingRecentItem(
                    "conversations", "stale", "Stale row", "Conversation"
                ),
            ),
        )
        latest = LibraryLandingCanvasState(
            purpose=landing.state.purpose,
            counts_line="Conversations (1)",
            recent_items=(
                LibraryLandingRecentItem(
                    "conversations", "latest", "Latest row", "Conversation"
                ),
            ),
        )

        recents_owner = landing.query_one("#library-hub-recents")
        original_remove = recents_owner.remove_children
        removal_started = asyncio.Event()
        release_removal = asyncio.Event()

        async def delayed_remove():
            removal_started.set()
            await release_removal.wait()
            await original_remove()

        monkeypatch.setattr(recents_owner, "remove_children", delayed_remove)
        landing.state = first
        replacement = asyncio.create_task(landing._replace_recent_rows())
        await removal_started.wait()
        landing.state = latest
        release_removal.set()
        await replacement

        recents = list(landing.query(".library-hub-recent"))
        assert [getattr(recent, "record_id", "") for recent in recents] == ["latest"]


@pytest.mark.asyncio
async def test_stale_landing_deferred_sync_performs_zero_dom_mutation():
    """A route-stale deferred replacement must leave mounted rows untouched."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations()[:1])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)
        recents_owner = landing.query_one("#library-hub-recents")
        children_before = tuple(recents_owner.children)
        assert len(children_before) == 1

        records = dict(screen._local_source_records)
        records["conversations"] = (
            {
                "title": "Newer conversation",
                "conversation_id": "chat-new",
                "message_count": 1,
                "updated_at": "2026-08-13T10:00:00Z",
            },
        )
        screen._local_source_records = records
        generation = screen._library_snapshot_state_generation + 1
        route_key = screen._library_entry_route_key()
        screen._library_snapshot_state_generation = generation
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, route_key)

        await screen._reconcile_library_entry_state(generation, route_key)
        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
        await pilot.pause()
        await pilot.pause()

        assert tuple(recents_owner.children) == children_before
        assert children_before[0].parent is recents_owner


@pytest.mark.asyncio
async def test_study_handoff_snapshot_sync_retains_open_action_and_paints_readiness():
    """A source/readiness change must patch the mounted handoff owner in place."""
    app = _build_test_app()
    _seed_conversations(app, [])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_CREATE_STUDY)
        await _wait_for_selector(screen, pilot, "#library-study-handoff-canvas")
        handoff = screen.query_one(
            "#library-study-handoff-canvas", LibraryStudyHandoffCanvas
        )
        open_button = screen.query_one("#library-open-study")
        assert "Import sources or create notes first" in _compositor_text(screen)

        changed = _apply_changed_snapshot(
            screen,
            notes=(
                {
                    "id": "note-1",
                    "title": "Retained source",
                    "content": "Body",
                    "last_modified": "2026-08-13T10:00:00Z",
                },
            ),
            study_decks=2,
        )
        await pilot.pause()
        await pilot.pause()

        painted = _compositor_text(screen)
        assert changed is True
        assert screen.query_one("#library-study-handoff-canvas") is handoff
        assert screen.query_one("#library-open-study") is open_button
        assert "Source snapshot is ready." in painted
        assert "Study decks (2)" in painted
        assert "Retained source" in painted


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(60, 20), (170, 48)])
@pytest.mark.parametrize("surface", ["landing", "handoff"])
async def test_retained_entry_actions_paint_before_and_after_sync(size, surface):
    """Existence is insufficient when compact geometry clips an entry action."""
    app = _build_test_app()
    _seed_conversations(app, [])
    screen = LibraryScreen(app)
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        if size[0] == 60 and surface == "landing":
            # Compact Library is a one-pane presentation. Expose its mounted
            # content stage so the render oracle measures the owner rather than
            # correctly-hidden children behind the entry rail.
            screen._library_notes_stage = "notes"
            screen._set_library_rail_collapsed(True)
            await pilot.pause()
        if surface == "handoff":
            await screen._select_library_rail_row(LIBRARY_ROW_CREATE_STUDY)
            await _wait_for_selector(screen, pilot, "#library-open-study")
            if size[0] == 60:
                screen._set_library_rail_collapsed(True)
                await pilot.pause()
            checks = (("#library-open-study", "Continue in Study"),)
        else:
            checks = (
                ("#library-hub-action-import", "Import…"),
                ("#library-hub-action-search", "Search"),
                ("#library-hub-action-new-note", "New note"),
            )
        for selector, expected in checks:
            _assert_widget_text_is_painted(screen, selector, expected)

        _apply_changed_snapshot(
            screen,
            notes=(
                {
                    "id": "note-geometry",
                    "title": "Geometry source",
                    "content": "Body",
                    "last_modified": "2026-08-13T10:00:00Z",
                },
            ),
            study_decks=2,
        )
        await pilot.pause()
        await pilot.pause()

        for selector, expected in checks:
            _assert_widget_text_is_painted(screen, selector, expected)


@pytest.mark.asyncio
async def test_warm_repeat_visit_composes_once_before_fresh_reconcile(monkeypatch):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    calls: list[LibraryScreen] = []
    original = LibraryScreen.compose_content

    def counted_compose(screen):
        calls.append(screen)
        yield from original(screen)

    monkeypatch.setattr(LibraryScreen, "compose_content", counted_compose)
    samples: list[float] = []
    revisits: list[LibraryScreen] = []
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        first = _active_library_screen(host)
        await _wait_for_library_shell(first, pilot)
        await host.pop_screen()
        await pilot.pause()
        for _ in range(5):
            revisit = LibraryScreen(app)
            revisits.append(revisit)
            started = time.perf_counter()
            await host.push_screen(revisit)
            await _wait_for_library_shell(revisit, pilot)
            samples.append((time.perf_counter() - started) * 1000)
            await host.pop_screen()
            await pilot.pause()
    print(
        f"warm_visit_median_ms={statistics.median(samples):.3f} "
        f"min_ms={min(samples):.3f} max_ms={max(samples):.3f} n={len(samples)}"
    )
    print(f"warm_visit_compose_counts={[calls.count(revisit) for revisit in revisits]}")
    assert all(calls.count(revisit) == 1 for revisit in revisits)


@pytest.mark.asyncio
async def test_library_source_snapshot_changed_reconciles_conversations_below_screen(
    monkeypatch,
):
    """Restoring either whole-screen refresh would replace captured owners."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        await screen.workers.wait_for_complete()
        await pilot.pause()

        rail = screen.query_one("#library-rail")
        canvas_host = screen.query_one("#library-canvas")
        canvas = screen.query_one(
            "#library-conversations-canvas", LibraryConversationsCanvas
        )
        refresh_calls: list[bool] = []
        recompose_calls: list[LibraryScreen] = []
        original_refresh = LibraryScreen.refresh
        original_recompose = LibraryScreen.recompose

        def recorded_refresh(active_screen, *regions, **kwargs):
            refresh_calls.append(bool(kwargs.get("recompose")))
            return original_refresh(active_screen, *regions, **kwargs)

        async def recorded_recompose(active_screen):
            recompose_calls.append(active_screen)
            return await original_recompose(active_screen)

        monkeypatch.setattr(LibraryScreen, "refresh", recorded_refresh)
        monkeypatch.setattr(LibraryScreen, "recompose", recorded_recompose)
        records = dict(screen._local_source_records)
        records["conversations"] = (
            *records["conversations"],
            {
                "title": "Incident review",
                "conversation_id": "chat-3",
                "message_count": 5,
                "updated_at": "2026-06-03T12:00:00Z",
            },
        )
        counts = dict(screen._local_source_counts)
        counts["conversations"] = 3

        changed = screen._apply_local_source_snapshot(
            records,
            counts,
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )
        await pilot.pause()
        await pilot.pause()

        assert changed is True
        assert _active_library_screen(host) is screen
        assert screen.query_one("#library-rail") is rail
        assert screen.query_one("#library-canvas") is canvas_host
        assert (
            screen.query_one(
                "#library-conversations-canvas", LibraryConversationsCanvas
            )
            is canvas
        )
        assert "Conversations (3)" in str(
            screen.query_one("#library-conversations-title").renderable
        )
        assert "(3)" in str(screen.query_one("#library-row-browse-conversations").label)
        assert True not in refresh_calls
        assert recompose_calls == []


@pytest.mark.asyncio
async def test_library_source_snapshot_changed_retains_conversation_row_focus():
    """Dropping the Conversations follow-up moves focus outside its canvas."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        await screen.workers.wait_for_complete()
        row = screen.query_one("#library-conversation-row-0")
        row.focus()
        await pilot.pause()
        assert getattr(screen.focused, "conversation_id", None) == "chat-2"

        records = dict(screen._local_source_records)
        records["conversations"] = (
            *records["conversations"],
            {
                "title": "Incident review",
                "conversation_id": "chat-3",
                "message_count": 5,
                "updated_at": "2026-06-03T12:00:00Z",
            },
        )
        counts = dict(screen._local_source_counts)
        counts["conversations"] = 3

        screen._apply_local_source_snapshot(
            records,
            counts,
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )
        await pilot.pause()
        await pilot.pause()

        focused = screen.focused
        assert getattr(focused, "conversation_id", None) == "chat-2"
        assert focused is not None and focused.disabled is False
        assert screen.query_one("#library-conversations-canvas") in (
            focused.ancestors_with_self
        )


@pytest.mark.asyncio
async def test_library_source_snapshot_equal_clean_refreshes_cache_without_dom_work(
    monkeypatch,
):
    """Removing the equality gate would call canvas or screen recomposition."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        await screen.workers.wait_for_complete()
        await pilot.pause()

        canvas = screen.query_one(
            "#library-conversations-canvas", LibraryConversationsCanvas
        )
        sync_calls: list[None] = []
        refresh_calls: list[bool] = []
        recompose_calls: list[LibraryScreen] = []
        original_sync_state = canvas.sync_state
        original_refresh = LibraryScreen.refresh
        original_recompose = LibraryScreen.recompose

        def recorded_sync_state(*args, **kwargs):
            sync_calls.append(None)
            return original_sync_state(*args, **kwargs)

        def recorded_refresh(active_screen, *regions, **kwargs):
            refresh_calls.append(bool(kwargs.get("recompose")))
            return original_refresh(active_screen, *regions, **kwargs)

        async def recorded_recompose(active_screen):
            recompose_calls.append(active_screen)
            return await original_recompose(active_screen)

        async def equal_snapshot():
            return (
                dict(screen._local_source_records),
                dict(screen._local_source_counts),
                dict(screen._local_source_total_known),
                screen._library_lookup_error,
                screen._library_lookup_recovery_state,
                dict(screen._library_study_counts),
            )

        monkeypatch.setattr(canvas, "sync_state", recorded_sync_state)
        monkeypatch.setattr(LibraryScreen, "refresh", recorded_refresh)
        monkeypatch.setattr(LibraryScreen, "recompose", recorded_recompose)
        monkeypatch.setattr(screen, "_list_local_source_snapshot", equal_snapshot)
        previous_stamp = time.monotonic() - 1.0
        app._library_source_snapshot_cache_stamp = previous_stamp

        screen._refresh_local_source_snapshot()
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert app._library_source_snapshot_cache_stamp > previous_stamp
        assert screen._library_snapshot_rendered_generation == (
            screen._library_snapshot_state_generation
        )
        assert screen._library_entry_reconcile_dirty is False
        assert sync_calls == []
        assert True not in refresh_calls
        assert recompose_calls == []


@pytest.mark.asyncio
async def test_library_source_snapshot_equal_dirty_repairs_with_targeted_sync(
    monkeypatch,
):
    """Skipping dirty equal state would leave the mounted canvas stale."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        await screen.workers.wait_for_complete()
        await pilot.pause()

        canvas = screen.query_one(
            "#library-conversations-canvas", LibraryConversationsCanvas
        )
        sync_calls: list[None] = []
        refresh_calls: list[bool] = []
        recompose_calls: list[LibraryScreen] = []
        original_sync_state = canvas.sync_state
        original_refresh = LibraryScreen.refresh
        original_recompose = LibraryScreen.recompose

        def recorded_sync_state(*args, **kwargs):
            sync_calls.append(None)
            return original_sync_state(*args, **kwargs)

        def recorded_refresh(active_screen, *regions, **kwargs):
            refresh_calls.append(bool(kwargs.get("recompose")))
            return original_refresh(active_screen, *regions, **kwargs)

        async def recorded_recompose(active_screen):
            recompose_calls.append(active_screen)
            return await original_recompose(active_screen)

        monkeypatch.setattr(canvas, "sync_state", recorded_sync_state)
        monkeypatch.setattr(LibraryScreen, "refresh", recorded_refresh)
        monkeypatch.setattr(LibraryScreen, "recompose", recorded_recompose)
        generation = screen._library_snapshot_state_generation
        screen._library_entry_reconcile_dirty = True

        changed = screen._apply_local_source_snapshot(
            dict(screen._local_source_records),
            dict(screen._local_source_counts),
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )
        await pilot.pause()
        await pilot.pause()

        assert changed is False
        assert sync_calls == [None]
        assert screen._library_snapshot_state_generation == generation
        assert screen._library_snapshot_rendered_generation == generation
        assert screen._library_entry_reconcile_dirty is False
        assert True not in refresh_calls
        assert recompose_calls == []


@pytest.mark.asyncio
async def test_library_source_snapshot_stale_route_clears_retry_markers():
    """Leaving the retry generation armed would skip the new route's retry."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        await screen.workers.wait_for_complete()
        await pilot.pause()

        generation = screen._library_snapshot_state_generation
        stale_route = screen._library_entry_route_key()
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, stale_route)
        screen._library_entry_reconcile_retry_generation = generation
        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA

        result = await screen._reconcile_library_entry_state(
            generation, stale_route
        )

        assert result is LibraryEntryReconcileResult.SUPERSEDED
        assert screen._library_entry_reconcile_pending is None
        assert screen._library_entry_reconcile_retry_generation is None


@pytest.mark.asyncio
async def test_library_source_snapshot_missing_skills_retries_then_equal_can_retry(
    monkeypatch,
):
    """Falling through a missing Skills selector would falsely mark it rendered."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS
        screen._library_skills_view = "list"
        generation = screen._library_snapshot_state_generation
        route_key = screen._library_entry_route_key()
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, route_key)
        queued: list[tuple[object, tuple[object, ...]]] = []

        def capture_call_later(callback, *args):
            queued.append((callback, args))

        monkeypatch.setattr(screen, "call_later", capture_call_later)

        first = await screen._reconcile_library_entry_state(generation, route_key)

        assert first is LibraryEntryReconcileResult.FAILED
        assert screen._library_entry_reconcile_dirty is True
        assert screen._library_entry_reconcile_pending == (generation, route_key)
        assert screen._library_entry_reconcile_retry_generation == generation
        assert len(queued) == 1

        callback, args = queued.pop()
        second = await callback(*args)

        assert second is LibraryEntryReconcileResult.FAILED
        assert screen._library_entry_reconcile_dirty is True
        assert screen._library_entry_reconcile_pending is None
        assert screen._library_entry_reconcile_retry_generation is None

        changed = screen._apply_local_source_snapshot(
            dict(screen._local_source_records),
            dict(screen._local_source_counts),
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )

        assert changed is False
        assert screen._library_entry_reconcile_pending == (generation, route_key)
        assert len(queued) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_site", ["rail", "header"])
async def test_library_source_snapshot_shell_exception_releases_retry_markers(
    monkeypatch, failure_site
):
    """An owned shell failure must not deduplicate the next equal repair."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        generation = screen._library_snapshot_state_generation
        route_key = screen._library_entry_route_key()
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, route_key)
        screen._library_entry_reconcile_retry_generation = generation
        queued: list[tuple[object, tuple[object, ...]]] = []

        def capture_call_later(callback, *args):
            queued.append((callback, args))

        def fail_shell_sync(*args, **kwargs):
            raise RuntimeError(f"forced {failure_site} sync failure")

        monkeypatch.setattr(screen, "call_later", capture_call_later)
        if failure_site == "rail":
            rail = screen.query_one("#library-rail")
            monkeypatch.setattr(rail, "sync_state", fail_shell_sync)
        else:
            header = screen.query_one("#library-header-line")
            header.update("stale header")
            monkeypatch.setattr(header, "update", fail_shell_sync)

        result = await screen._reconcile_library_entry_state(
            generation, route_key
        )

        assert result is LibraryEntryReconcileResult.FAILED
        assert screen._library_entry_reconcile_dirty is True
        assert screen._library_entry_reconcile_pending is None
        assert screen._library_entry_reconcile_retry_generation is None

        changed = screen._apply_local_source_snapshot(
            dict(screen._local_source_records),
            dict(screen._local_source_counts),
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )

        assert changed is False
        assert screen._library_entry_reconcile_pending == (generation, route_key)
        assert len(queued) == 1


def test_constructor_seeds_cached_snapshot_before_restore_state_wins_selection():
    """Changing pre-compose cache seeding to mount-time seeding would leave
    this fresh screen unloaded before its first composition.
    """
    app = _build_test_app()
    app._library_source_snapshot_cache = (
        {
            "notes": ({"id": "n1"},),
            "media": ({"id": "m1"},),
            "conversations": ({"id": "c1"},),
            "prompts": (None, ()),
            "skills": (None, {"available_skills": [], "blocked_skills": []}),
        },
        {"notes": 1, "media": 1, "conversations": 1},
        {"notes": True, "media": True, "conversations": True},
        None,
        None,
        {"study_decks": None, "flashcards_due": None, "quizzes": None},
    )
    app._library_source_snapshot_cache_stamp = time.monotonic()

    screen = LibraryScreen(app)

    assert screen._library_loaded is True
    assert screen._local_source_counts == {
        "notes": 1,
        "media": 1,
        "conversations": 1,
    }

    screen.restore_state(
        {
            "library_selected_row_id": LIBRARY_ROW_BROWSE_MEDIA,
            "selected_media_id": "m1",
            "library_media_view": "viewer",
        }
    )

    assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
    assert screen._selected_media_id == "m1"
    assert screen._library_media_view == "viewer"


def test_cache_seed_rejects_future_and_ttl_boundary_stamps():
    """Changing either cache-age guard would accept a future or expired seed."""
    app = _build_test_app()
    app._library_source_snapshot_cache = (
        {
            "notes": (),
            "media": (),
            "conversations": (),
            "prompts": (None, ()),
            "skills": (None, {"available_skills": [], "blocked_skills": []}),
        },
        {"notes": 0, "media": 0, "conversations": 0},
        {"notes": True, "media": True, "conversations": True},
        None,
        None,
        {"study_decks": None, "flashcards_due": None, "quizzes": None},
    )
    stamp = 100.0
    app._library_source_snapshot_cache_stamp = stamp
    screen = LibraryScreen(app)

    assert screen._seed_local_source_snapshot_from_cache(now=stamp - 0.1) is False
    assert (
        screen._seed_local_source_snapshot_from_cache(
            now=stamp + LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS
        )
        is False
    )
    assert screen._library_loaded is False

    assert (
        screen._seed_local_source_snapshot_from_cache(
            now=stamp + LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS - 0.1
        )
        is True
    )
    assert screen._library_loaded is True
