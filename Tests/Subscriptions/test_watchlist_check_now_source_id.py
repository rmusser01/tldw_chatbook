"""Check now must accept the namespaced source id the UI actually holds.

Found in the 2026-07-28 live UAT with real feeds. `Check now` did nothing at
all: no run, no items, `last_checked` still NULL, and no error the user could
see. The scrape backend was fine -- driven directly with the integer id it
fetched a real feed and ingested 10 items in 268ms.

The break was one seam. `LocalWatchlistsService` returns a row carrying **both**
``"id": "local:subscription:1"`` (the namespaced display id, which is what every
UI path passes around) and ``"source_id": 1``. `local.launch_run` does
``int(source_id)``, so the namespaced form raises
``ValueError: invalid literal for int() with base 10: 'local:subscription:1'``,
which `_check_now_source` swallows into a debug log and a transient toast.

`WatchlistScopeService` already owns this translation for two other entity
types -- `_run_id_from_item_id` and `_rule_id_from_item_id`. Sources were the
type it did not cover.
"""
from __future__ import annotations

import inspect

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService
from tldw_chatbook.Subscriptions.watchlist_scope_service import WatchlistScopeService


async def _maybe(value):
    return await value if inspect.isawaitable(value) else value


@pytest.fixture
def services(tmp_path):
    db_path = tmp_path / "subs.db"
    local = LocalWatchlistsService(
        db_factory=lambda: SubscriptionsDB(str(db_path), "test")
    )
    return local, WatchlistScopeService(local_service=local, server_service=None)


@pytest.mark.asyncio
async def test_check_now_accepts_the_namespaced_source_id(services):
    """The id the screen holds is the namespaced one; it must work."""
    local, scope = services
    source = await _maybe(
        local.create_source(
            {"name": "Example", "type": "rss", "source": "https://example.invalid/f.xml"}
        )
    )
    assert source["id"] == f"local:subscription:{source['source_id']}", (
        "precondition: create_source returns a namespaced display id"
    )

    # Exactly what `_check_now_source` passes: `source.get("id")`.
    run = await _maybe(scope.check_now(source_id=source["id"], runtime_backend="local"))

    assert run is not None
    assert int(run["source_id"]) == int(source["source_id"]), (
        "check_now must resolve the namespaced id to the right subscription"
    )


@pytest.mark.asyncio
async def test_check_now_still_accepts_a_plain_integer_id(services):
    """Callers holding the integer keep working."""
    local, scope = services
    source = await _maybe(
        local.create_source(
            {"name": "Example", "type": "rss", "source": "https://example.invalid/f.xml"}
        )
    )
    run = await _maybe(
        scope.check_now(source_id=source["source_id"], runtime_backend="local")
    )
    assert int(run["source_id"]) == int(source["source_id"])
