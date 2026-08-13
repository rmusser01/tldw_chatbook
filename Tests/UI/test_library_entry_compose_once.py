from __future__ import annotations

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
)
from tldw_chatbook.Widgets.Library import LibraryConversationsCanvas
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
