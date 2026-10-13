"""Rapid Watchlists revisits skip the five mount-time section loads."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WATCHLISTS_REVISIT_SKIP_SECONDS,
    WatchlistsCollectionsScreen,
)

pytestmark = pytest.mark.bootstrap_profile


class _RevisitScreen(WatchlistsCollectionsScreen):
    """Bare shell: reactives and the app property become plain attributes."""

    active_section = "watchlists"
    runtime_backend = "local"
    app_instance = None

    def __init__(self) -> None:  # skip the heavy real screen init
        pass


def _screen(app_instance):
    screen = _RevisitScreen()
    screen.app_instance = app_instance
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


def test_first_visit_loads_everything():
    screen, calls = _screen(SimpleNamespace())
    screen.on_mount()
    assert len(calls) >= 5, "layout pass plus the five loaders must run"


def test_immediate_revisit_skips_the_loaders():
    import time

    app = SimpleNamespace(_watchlists_last_visit_monotonic=time.monotonic())
    screen, calls = _screen(app)
    screen.on_mount()
    assert len(calls) == 1, "only the layout pass runs; the loaders skip"


def test_revisit_after_the_ttl_loads_again():
    import time

    stale = time.monotonic() - (WATCHLISTS_REVISIT_SKIP_SECONDS + 1)
    app = SimpleNamespace(_watchlists_last_visit_monotonic=stale)
    screen, calls = _screen(app)
    screen.on_mount()
    assert len(calls) >= 5, "an old timestamp no longer gates the loads"
