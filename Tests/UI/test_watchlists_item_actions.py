"""A selected watchlist item must be typed as an item — TASK-1120.

The 2026-07-28 reading-path UAT fetched ten real articles from
`https://summitroute.com/blog/feed.xml`, clicked one in the Items table, and
the Inspector said:

    Selected: Lightsail object storage concerns - Part 2
    Type: source
               Preview
              Check now

`Preview` and `Check now` are *source* actions. `Mark reviewed`, `Ingest` and
`Ignore` — the actions the Inspector is built to offer for an item — never
appeared, so a fetched item could not be acted on at all.

`InspectorPane._entity_type` decided this from the entity's *shape*, and its
first shape test was `"source_type" in entity or "url" in entity`. Every item
`normalize_watchlist_item` produces carries both: `source_type` is the type of
the feed the item came from, and `url` is the article's own link. The item
tests (`item_id`, `source_name`) sat two branches below and were never
reached. Both normalizers also emit an explicit `entity_kind`, which is
unambiguous, so that is what decides now.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Subscriptions.watchlist_normalizers import (
    normalize_local_subscription_row,
    normalize_watchlist_alert_rule,
    normalize_watchlist_item,
    normalize_watchlist_run,
)
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsPane

# Whole-branch review (Important): without this, CI's `pytest -m unit` run
# DESELECTS this entire module, so every assertion here -- including the ones
# this branch changed when Task 5 deleted the Inspector's "Mark reviewed"
# button -- was only ever exercised locally.
pytestmark = pytest.mark.unit

# Exactly the row shape `SubscriptionsDB.get_new_items` returns for a scraped
# feed entry, put through the same normalizer the Items pane is fed from.
REAL_ITEM = normalize_watchlist_item(
    "local",
    {
        "id": 2,
        "subscription_id": 1,
        "subscription_name": "Summit Route",
        "subscription_type": "rss",
        "title": "Lightsail object storage concerns - Part 2",
        "url": "https://summitroute.com/blog/2024/lightsail-part-2/",
        "status": "new",
        "author": "Scott Piper",
        "created_at": "2026-07-28T09:00:00+00:00",
    },
)


def _inspector_text(inspector: InspectorPane, widget_id: str) -> str:
    return str(inspector.query_one(f"#{widget_id}", Static).renderable)


@pytest.mark.asyncio
async def test_selected_item_reports_type_item_and_offers_item_actions():
    """AC#1, AC#2, AC#5."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        screen.active_section = "items"
        await pilot.pause(0.2)

        pane = screen.query_one("#watchlists-items-pane", ItemsPane)
        pane.items = [dict(REAL_ITEM)]
        await pilot.pause(0.2)
        pane.select_item_by_id(str(REAL_ITEM["id"]))
        await pilot.pause(0.3)

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert "Lightsail object storage concerns" in _inspector_text(
            inspector, "inspector-entity-title"
        )
        assert _inspector_text(inspector, "inspector-entity-type") == "Type: item", (
            "a fetched watchlist item was reported as a source, so the actions "
            "beneath it were source actions (TASK-1120)"
        )

        assert inspector.query_one("#inspector-ingest-button", Button)
        assert inspector.query_one("#inspector-ignore-button", Button)
        # "Mark reviewed" was removed (Task 5 fix round 1, Important):
        # `selected_entity` is set by the same `ItemSelected` that now marks
        # an item read on open, so this button was only ever reachable on an
        # item already at "reviewed" -- dead in practice.
        assert not inspector.query("#inspector-mark-reviewed-button")
        assert not inspector.query("#inspector-preview-button"), (
            "Preview is a source action and must not be offered for an item"
        )
        assert not inspector.query("#inspector-check-now-button")


# Realistic normalizer output for every other selectable kind, so the
# discriminator cannot be fixed for items by breaking one of these (AC#4).
OTHER_ENTITIES = (
    (
        "source",
        normalize_local_subscription_row(
            {
                "id": 1,
                "name": "Summit Route",
                "type": "rss",
                "source": "https://summitroute.com/blog/feed.xml",
                "is_active": 1,
            }
        ),
    ),
    (
        "run",
        normalize_watchlist_run(
            "local",
            {
                "id": 3,
                "source_id": 1,
                "status": "completed",
                "started_at": "2026-07-28T09:00:00+00:00",
            },
        ),
    ),
    (
        "rule",
        normalize_watchlist_alert_rule(
            "local",
            {
                "id": 4,
                "source_id": 1,
                "name": "Nothing fetched",
                "condition_type": "no_items",
                "severity": "warning",
            },
        ),
    ),
    (
        "notification",
        {
            "id": 5,
            "entity_kind": "client_notification",
            "title": "Feed failed",
            "message": "boom",
            "category": "watchlist",
            "severity": "warning",
            "is_read": False,
        },
    ),
)


@pytest.mark.parametrize(
    "expected_type,entity", OTHER_ENTITIES, ids=[kind for kind, _ in OTHER_ENTITIES]
)
def test_every_other_selection_still_reports_its_own_type(expected_type, entity):
    """AC#4."""
    assert InspectorPane._entity_type(entity) == expected_type


def test_item_is_typed_from_its_kind_not_its_shape():
    """AC#1 at the unit level: `url` and `source_type` no longer win."""
    assert "url" in REAL_ITEM and "source_type" in REAL_ITEM, (
        "the fixture must keep the two keys that used to mistype it"
    )
    assert InspectorPane._entity_type(REAL_ITEM) == "item"


# `test_mark_reviewed_writes_the_new_status_to_the_database` (AC#3) lived
# here through TASK-1120, pressing `#inspector-mark-reviewed-button` and
# reading the "reviewed" status back from the database. That button was
# removed in the Phase D content-pane plan's Task 5, fix round 1 (Important):
# `selected_entity` is set by the same `ItemSelected` that now marks an item
# read on open, so the button was only ever reachable on an item already
# "reviewed" -- dead in practice. The same database-write assertion this
# test made (opening/marking an item moves its row to "reviewed") is now
# covered by `test_opening_an_item_marks_it_read` in
# `Tests/UI/test_watchlists_read_status.py`, exercising the path that
# actually writes that status today.
