"""Watchlists loads its five sections on every mount (task-34929 revert).

The revisit-TTL gate shipped in task-34929 was REVERTED after independent
review: this route is not reusable, so a fresh screen instance on a rapid
revisit skipped its loaders with empty data state and rendered a permanent
loading state. These tests pin the reverted contract.
"""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)

pytestmark = pytest.mark.bootstrap_profile


class _MountScreen(WatchlistsCollectionsScreen):
    """Bare shell: reactives and the app property become plain attributes."""

    active_section = "watchlists"
    runtime_backend = "local"
    app_instance = None

    def __init__(self) -> None:  # skip the heavy real screen init
        pass


def _screen():
    screen = _MountScreen()
    screen.app_instance = SimpleNamespace()
    calls = []

    def _record(*_args, **_kwargs):
        calls.append(None)

    screen._recompute_effective_layout = _record
    screen._refresh_local_wc_snapshot = _record
    screen._load_active_section_data = _record
    screen._load_tree_data = _record
    screen.set_timer = _record
    screen._refresh_overview_data = _record
    screen._resolve_daily_report_banner = _record
    return screen, calls


def test_every_mount_loads_the_sections():
    """Rapid revisits must not skip loads: each visit is a FRESH instance."""
    screen, calls = _screen()
    screen.on_mount()
    assert len(calls) >= 5, "layout pass plus the five loaders run on mount"
