"""Tests for the briefing presets/scripts DB foundation (spec #2 phase 2a, Task 1).

Presets are a named, reusable N-speaker roster (plus optional style notes
and a provider/model override); scripts are one cast run of a specific
`briefings` row, snapshotting the preset's roster and name at cast time so a
later preset edit or delete never changes the meaning of a script someone
already cast. `set_watchlist_briefing_settings` is the writer half of the
`watchlists.briefing_selection_mode` / `default_briefing_preset_id` columns
phase 1 added; `get_subscription_items_by_ids` is the chunked-lookup
counterpart to phase 1's unbounded-`NOT IN` fix in `briefing_selection`.

Same harness as `test_briefing_selection.py`: a real `SubscriptionsDB` on
`:memory:`, `WatchlistBundleService` for watchlist creation (there is no
`SubscriptionsDB.create_watchlist`).
"""

from unittest.mock import Mock

import pytest

from tldw_chatbook.DB import Subscriptions_DB as subscriptions_db_module
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


# --- Presets -----------------------------------------------------------------


def test_briefing_preset_round_trip_including_nulls():
    """A preset created with only the required fields reads back with the
    optional columns genuinely NULL, not empty strings or missing keys."""
    db = SubscriptionsDB(":memory:", "test")
    preset_id = db.insert_briefing_preset("Daily digest", roster_json='[{"name": "Host"}]')

    row = db.get_briefing_preset(preset_id)
    assert row["id"] == preset_id
    assert row["name"] == "Daily digest"
    assert row["roster_json"] == '[{"name": "Host"}]'
    assert row["style_notes"] is None
    assert row["provider"] is None
    assert row["model"] is None
    assert row["created_at"] is not None
    assert row["updated_at"] is not None


def test_briefing_preset_round_trip_with_every_field_set():
    db = SubscriptionsDB(":memory:", "test")
    preset_id = db.insert_briefing_preset(
        "Two-host banter",
        roster_json='[{"name": "Alex"}, {"name": "Sam"}]',
        style_notes="Keep it punchy.",
        provider="openai",
        model="gpt-4o",
    )

    row = db.get_briefing_preset(preset_id)
    assert row["name"] == "Two-host banter"
    assert row["style_notes"] == "Keep it punchy."
    assert row["provider"] == "openai"
    assert row["model"] == "gpt-4o"


def test_update_briefing_preset_rejects_unknown_field_but_accepts_a_valid_one():
    """Matches `update_briefing`'s allowlist pattern: a typo'd or renamed
    keyword must raise immediately rather than silently building a query
    against a column that was never meant to be settable this way."""
    db = SubscriptionsDB(":memory:", "test")
    preset_id = db.insert_briefing_preset("Original", roster_json="[]")

    with pytest.raises(ValueError, match="not_a_real_column"):
        db.update_briefing_preset(preset_id, not_a_real_column="oops")

    db.update_briefing_preset(preset_id, name="Renamed", style_notes="New notes")
    row = db.get_briefing_preset(preset_id)
    assert row["name"] == "Renamed"
    assert row["style_notes"] == "New notes"


def test_update_briefing_preset_also_enforces_sql_validation_not_just_the_allowlist(
    monkeypatch,
):
    """Qodo-round precedent (`test_update_briefing_also_enforces_sql_validation_
    not_just_the_allowlist`): the local allowlist and
    `sql_validation.validate_identifier` are two independent gates. Forces
    the divergence with a monkeypatch, since every real allowlisted column
    already passes `validate_identifier` on its own."""
    db = SubscriptionsDB(":memory:", "test")
    preset_id = db.insert_briefing_preset("Original", roster_json="[]")

    monkeypatch.setattr(
        subscriptions_db_module, "validate_identifier", lambda *a, **k: False
    )
    with pytest.raises(ValueError, match="name"):
        db.update_briefing_preset(preset_id, name="Blocked")


def test_list_briefing_presets_orders_by_name_asc():
    db = SubscriptionsDB(":memory:", "test")
    db.insert_briefing_preset("Zebra", roster_json="[]")
    db.insert_briefing_preset("Apple", roster_json="[]")
    db.insert_briefing_preset("Mango", roster_json="[]")

    listed = db.list_briefing_presets()
    assert [row["name"] for row in listed] == ["Apple", "Mango", "Zebra"]


def test_delete_briefing_preset_returns_false_for_missing_id():
    db = SubscriptionsDB(":memory:", "test")
    assert db.delete_briefing_preset(999999) is False


def test_delete_briefing_preset_hard_deletes_and_returns_true():
    db = SubscriptionsDB(":memory:", "test")
    preset_id = db.insert_briefing_preset("Doomed", roster_json="[]")

    assert db.delete_briefing_preset(preset_id) is True
    assert db.get_briefing_preset(preset_id) is None
    # Hard delete, not soft: the row is genuinely gone from the table.
    row_count = db.conn.execute(
        "SELECT COUNT(*) FROM briefing_presets WHERE id = ?", (preset_id,)
    ).fetchone()[0]
    assert row_count == 0


def test_delete_briefing_preset_leaves_existing_scripts_snapshot_intact():
    """Scripts snapshot their roster/preset name at cast time -- deleting the
    preset later must not touch (or orphan the meaning of) a script already
    cast from it. `preset_id` on the script is not a foreign key, so the
    delete succeeds and the script's own columns are untouched."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]
    briefing_id = db.insert_briefing(watchlist_id)
    preset_id = db.insert_briefing_preset("Doomed", roster_json='[{"name": "Host"}]')
    script_id = db.insert_briefing_script(
        briefing_id,
        preset_id=preset_id,
        preset_name="Doomed",
        roster_snapshot_json='[{"name": "Host"}]',
    )

    assert db.delete_briefing_preset(preset_id) is True

    script = db.get_briefing_script(script_id)
    assert script is not None
    assert script["preset_name"] == "Doomed"
    assert script["roster_snapshot_json"] == '[{"name": "Host"}]'
    assert script["preset_id"] == preset_id  # now a dangling back-reference


# --- Scripts -------------------------------------------------------------


def test_briefing_script_round_trip():
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]
    briefing_id = db.insert_briefing(watchlist_id)

    script_id = db.insert_briefing_script(
        briefing_id,
        preset_id=None,
        preset_name="Ad-hoc",
        roster_snapshot_json='[{"name": "Narrator"}]',
    )

    row = db.get_briefing_script(script_id)
    assert row["id"] == script_id
    assert row["briefing_id"] == briefing_id
    assert row["preset_id"] is None
    assert row["preset_name"] == "Ad-hoc"
    assert row["roster_snapshot_json"] == '[{"name": "Narrator"}]'
    assert row["status"] == "generating"
    assert row["turns_json"] is None
    assert row["error"] is None
    assert row["model_used"] is None


def test_briefing_script_cascades_on_briefing_delete():
    """`briefing_scripts.briefing_id` carries `ON DELETE CASCADE`. This DB
    overrides `_get_connection` to set `PRAGMA foreign_keys = ON` per
    connection (see the docstring there), so unlike a bare sqlite3
    connection this cascade is actually live, not merely declared."""
    db = SubscriptionsDB(":memory:", "test")
    # Confirm the precondition this test relies on rather than assuming it:
    # PRAGMA foreign_keys is per-connection and OFF by default in sqlite3.
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]
    briefing_id = db.insert_briefing(watchlist_id)
    script_id = db.insert_briefing_script(
        briefing_id,
        preset_id=None,
        preset_name="Ad-hoc",
        roster_snapshot_json="[]",
    )
    assert db.get_briefing_script(script_id) is not None

    with db.transaction() as conn:
        conn.execute("DELETE FROM briefings WHERE id = ?", (briefing_id,))

    assert db.get_briefing_script(script_id) is None


def test_list_briefing_scripts_returns_newest_first_by_identity():
    """Identities, not just a count -- a query that returns "three rows"
    without honoring recency cannot pass this by accident."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]
    briefing_id = db.insert_briefing(watchlist_id)

    first = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="p", roster_snapshot_json="[]"
    )
    second = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="p", roster_snapshot_json="[]"
    )
    third = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="p", roster_snapshot_json="[]"
    )

    listed = db.list_briefing_scripts(briefing_id)
    assert [row["id"] for row in listed] == [third, second, first]


def test_list_briefing_scripts_is_scoped_to_its_own_briefing():
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]
    briefing_a = db.insert_briefing(watchlist_id)
    briefing_b = db.insert_briefing(watchlist_id)

    script_a = db.insert_briefing_script(
        briefing_a, preset_id=None, preset_name="p", roster_snapshot_json="[]"
    )
    db.insert_briefing_script(
        briefing_b, preset_id=None, preset_name="p", roster_snapshot_json="[]"
    )

    listed = db.list_briefing_scripts(briefing_a)
    assert [row["id"] for row in listed] == [script_a]


def test_update_briefing_script_rejects_unknown_field_but_accepts_a_valid_one():
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]
    briefing_id = db.insert_briefing(watchlist_id)
    script_id = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="p", roster_snapshot_json="[]"
    )

    with pytest.raises(ValueError, match="not_a_real_column"):
        db.update_briefing_script(script_id, not_a_real_column="oops")

    db.update_briefing_script(
        script_id, status="complete", turns_json='[{"speaker": "Host", "text": "Hi"}]'
    )
    row = db.get_briefing_script(script_id)
    assert row["status"] == "complete"
    assert row["turns_json"] == '[{"speaker": "Host", "text": "Hi"}]'


def test_update_briefing_script_also_enforces_sql_validation_not_just_the_allowlist(
    monkeypatch,
):
    """The Step 5(a) mutation-check precedent, kept as a permanent
    regression test: dropping `validate_identifier` from
    `update_briefing_script` must not leave this divergence undetected."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]
    briefing_id = db.insert_briefing(watchlist_id)
    script_id = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="p", roster_snapshot_json="[]"
    )

    monkeypatch.setattr(
        subscriptions_db_module, "validate_identifier", lambda *a, **k: False
    )
    with pytest.raises(ValueError, match="status"):
        db.update_briefing_script(script_id, status="complete")


# --- set_watchlist_briefing_settings --------------------------------------


def test_set_watchlist_briefing_settings_rejects_an_invalid_mode_by_name():
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]

    with pytest.raises(ValueError, match="bogus_mode"):
        db.set_watchlist_briefing_settings(watchlist_id, selection_mode="bogus_mode")


def test_set_watchlist_briefing_settings_writes_selection_mode():
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]

    db.set_watchlist_briefing_settings(watchlist_id, selection_mode="curated")

    row = db.conn.execute(
        "SELECT briefing_selection_mode FROM watchlists WHERE id = ?", (watchlist_id,)
    ).fetchone()
    assert row["briefing_selection_mode"] == "curated"


def test_set_watchlist_briefing_settings_default_preset_id_unset_leaves_alone_and_none_clears():
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]
    preset_id = db.insert_briefing_preset("p", roster_json="[]")

    # Set it once.
    db.set_watchlist_briefing_settings(watchlist_id, default_preset_id=preset_id)
    row = db.conn.execute(
        "SELECT default_briefing_preset_id FROM watchlists WHERE id = ?", (watchlist_id,)
    ).fetchone()
    assert row["default_briefing_preset_id"] == preset_id

    # Calling again without the argument (the `_UNSET` sentinel default)
    # must leave it alone -- a call that only wants to change
    # `selection_mode` must not accidentally clear the preset.
    db.set_watchlist_briefing_settings(watchlist_id, selection_mode="auto")
    row = db.conn.execute(
        "SELECT default_briefing_preset_id, briefing_selection_mode FROM watchlists "
        "WHERE id = ?",
        (watchlist_id,),
    ).fetchone()
    assert row["default_briefing_preset_id"] == preset_id
    assert row["briefing_selection_mode"] == "auto"

    # Passing `None` explicitly clears it.
    db.set_watchlist_briefing_settings(watchlist_id, default_preset_id=None)
    row = db.conn.execute(
        "SELECT default_briefing_preset_id FROM watchlists WHERE id = ?", (watchlist_id,)
    ).fetchone()
    assert row["default_briefing_preset_id"] is None


def test_set_watchlist_briefing_settings_with_no_arguments_is_a_no_op():
    """Neither argument given -- the `updates` list stays empty and the
    method returns before touching the connection at all."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]

    before = db.conn.execute(
        "SELECT briefing_selection_mode, default_briefing_preset_id FROM watchlists "
        "WHERE id = ?",
        (watchlist_id,),
    ).fetchone()

    db.set_watchlist_briefing_settings(watchlist_id)

    after = db.conn.execute(
        "SELECT briefing_selection_mode, default_briefing_preset_id FROM watchlists "
        "WHERE id = ?",
        (watchlist_id,),
    ).fetchone()
    assert tuple(before) == tuple(after)


# --- get_subscription_items_by_ids ----------------------------------------


def _seed_items(db, count):
    """Insert `count` bare `subscription_items` rows; return their ids in
    insertion (== AUTOINCREMENT) order."""
    source_id = db.add_subscription(name="Feed", type="rss", source="https://feed.example/f")
    ids = []
    with db.transaction() as conn:
        for n in range(count):
            cursor = conn.execute(
                "INSERT INTO subscription_items (subscription_id, url, title) "
                "VALUES (?, ?, ?)",
                (source_id, f"https://feed.example/{n}", f"Item {n}"),
            )
            ids.append(cursor.lastrowid)
    return ids


def test_get_subscription_items_by_ids_returns_only_existing_rows_keyed_by_id():
    db = SubscriptionsDB(":memory:", "test")
    ids = _seed_items(db, 3)
    missing_id = max(ids) + 1000

    result = db.get_subscription_items_by_ids([ids[0], ids[2], missing_id])

    assert set(result.keys()) == {ids[0], ids[2]}
    assert result[ids[0]]["title"] == "Item 0"
    assert result[ids[2]]["title"] == "Item 2"
    assert missing_id not in result


def test_get_subscription_items_by_ids_empty_input_returns_empty_dict():
    db = SubscriptionsDB(":memory:", "test")
    assert db.get_subscription_items_by_ids([]) == {}


def test_get_subscription_items_by_ids_chunks_the_in_clause_at_500_params():
    """The Qodo NOT-IN lesson, DB-lookup half: a single statement bound with
    every id at once risks SQLite's host-parameter limit for a heavy user's
    briefing. Spies on the connection like `test_briefing_selection.py`'s
    `test_window_query_parameter_count_does_not_scale_with_queue_size`, and
    asserts every individual statement stays at or under 500 bound
    parameters even though the total id count is well past that.
    """
    db = SubscriptionsDB(":memory:", "test")
    total = 1200
    ids = _seed_items(db, total)
    assert len(ids) == total

    real_conn = db.conn  # materialise this thread's connection first
    spy_conn = Mock(wraps=real_conn)
    db._local.conn = spy_conn
    try:
        result = db.get_subscription_items_by_ids(ids)
    finally:
        db._local.conn = real_conn

    assert len(result) == total  # every seeded id was actually found

    param_counts = [
        len(call.args[1]) for call in spy_conn.execute.call_args_list if len(call.args) > 1
    ]
    assert param_counts, "get_subscription_items_by_ids must have executed at least one query"
    assert max(param_counts) <= 500, (
        f"a statement was bound with {max(param_counts)} parameters against a "
        f"{total}-id lookup -- must be chunked at <= 500 params per statement"
    )
    # More than one statement had to run, since 1200 ids can't fit in one
    # <=500-param chunk.
    assert len(param_counts) >= 3
