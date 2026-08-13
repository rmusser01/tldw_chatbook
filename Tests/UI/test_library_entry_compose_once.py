from __future__ import annotations

import statistics
import time

import pytest

from tldw_chatbook.UI.Screens.library_screen import (
    LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS,
    LibraryScreen,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
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
