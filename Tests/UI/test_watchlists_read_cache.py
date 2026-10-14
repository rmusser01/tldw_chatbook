"""Watchlists whole-screen reads hit a short-TTL app cache on rapid revisits.

TASK-34929 second attempt: the loaders always run and write state -- only
the underlying queries are cached, so a fresh screen instance can never be
stranded empty (the failure mode that reverted the first attempt).
"""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WATCHLISTS_READ_CACHE_TTL_SECONDS,
    WatchlistsCollectionsScreen,
)

pytestmark = pytest.mark.bootstrap_profile


class _CacheScreen(WatchlistsCollectionsScreen):
    """Bare shell: reactives/properties become plain attributes."""

    app_instance = None
    runtime_backend = "local"
    _controller = None

    def __init__(self) -> None:  # skip the heavy real screen init
        pass


def _screen(app, reads):
    screen = _CacheScreen()
    screen.app_instance = app

    async def reader():
        reads.append(None)
        return ("result",)

    screen._list_local_wc_snapshot = reader
    return screen


@pytest.mark.asyncio
async def test_snapshot_read_served_from_cache_within_ttl():
    import time

    app = SimpleNamespace(
        _wc_snapshot_read_cache=(time.monotonic(), ("cached",))
    )
    reads = []
    screen = _screen(app, reads)

    assert await screen._list_local_wc_snapshot_cached() == ("cached",)
    assert not reads, "a fresh cached entry must not re-query"


@pytest.mark.asyncio
async def test_snapshot_read_queries_after_ttl():
    import time

    stale = time.monotonic() - (WATCHLISTS_READ_CACHE_TTL_SECONDS + 1)
    app = SimpleNamespace(_wc_snapshot_read_cache=(stale, ("stale",)))
    reads = []
    screen = _screen(app, reads)

    assert await screen._list_local_wc_snapshot_cached() == ("result",)
    assert reads, "an expired entry must re-query"
    assert app._wc_snapshot_read_cache[1] == ("result",), "fresh result cached"


@pytest.mark.asyncio
async def test_overview_cache_is_backend_keyed():
    import time

    now = time.monotonic()
    app = SimpleNamespace(
        _wc_overview_read_cache=("local", now, {"total_sources": 3})
    )
    screen = _CacheScreen()
    screen.app_instance = app
    screen.runtime_backend = "server"

    async def overview(**kwargs):
        return {"total_sources": 9}

    screen._controller = SimpleNamespace(get_overview_data=overview)
    assert await screen._overview_data_cached() == {"total_sources": 9}, (
        "a backend switch must not serve the other backend's cache"
    )
    assert app._wc_overview_read_cache[0] == "server"
