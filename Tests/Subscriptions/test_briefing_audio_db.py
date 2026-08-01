"""Tests for the `briefing_audio` DB foundation (spec #2 phase 2b, Task 1).

Audio is one synthesis run of a specific `briefing_scripts` row, turning a
cast script into a playable recording. `voice_snapshot_json` freezes the
voice assignment used for that render -- exactly like `roster_snapshot_json`
on `briefing_scripts` (phase 2a), it is write-once: not in
`update_briefing_audio`'s allowlist, so a synthesized artifact's provenance
can never be revised after the fact.

Same harness as `test_briefing_presets_db.py`'s script tests: a real
`SubscriptionsDB` on `:memory:`, `WatchlistBundleService` for watchlist
creation, `insert_briefing_script` for the parent script row every audio row
hangs off of.
"""

import pytest

from tldw_chatbook.DB import Subscriptions_DB as subscriptions_db_module
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


def _make_script(db: SubscriptionsDB) -> int:
    """Create a watchlist -> briefing -> briefing_scripts chain and return the script id."""
    watchlist_id = WatchlistBundleService(db).create(name="w")["id"]
    briefing_id = db.insert_briefing(watchlist_id)
    return db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="p", roster_snapshot_json="[]"
    )


def test_briefing_audio_round_trip_with_null_file_path_and_duration():
    """A row created with only the required fields reads back with the
    optional columns genuinely NULL, not empty strings or missing keys."""
    db = SubscriptionsDB(":memory:", "test")
    script_id = _make_script(db)

    audio_id = db.create_briefing_audio(
        script_id, voice_snapshot_json='[{"speaker": "Host", "voice": "alloy"}]'
    )

    row = db.get_briefing_audio(audio_id)
    assert row["id"] == audio_id
    assert row["script_id"] == script_id
    assert row["voice_snapshot_json"] == '[{"speaker": "Host", "voice": "alloy"}]'
    assert row["status"] == "generating"
    assert row["file_path"] is None
    assert row["duration_seconds"] is None
    assert row["turn_count"] is None
    assert row["error"] is None
    assert row["created_at"] is not None
    assert row["updated_at"] is not None


def test_create_briefing_audio_accepts_an_explicit_status():
    db = SubscriptionsDB(":memory:", "test")
    script_id = _make_script(db)

    audio_id = db.create_briefing_audio(
        script_id, voice_snapshot_json="[]", status="complete"
    )

    assert db.get_briefing_audio(audio_id)["status"] == "complete"


def test_get_briefing_audio_returns_none_for_missing_id():
    db = SubscriptionsDB(":memory:", "test")
    assert db.get_briefing_audio(999999) is None


def test_update_briefing_audio_rejects_unknown_field_but_accepts_valid_ones():
    db = SubscriptionsDB(":memory:", "test")
    script_id = _make_script(db)
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")

    with pytest.raises(ValueError, match="not_a_real_column"):
        db.update_briefing_audio(audio_id, not_a_real_column="oops")

    db.update_briefing_audio(
        audio_id,
        status="complete",
        file_path="/tmp/briefing.mp3",
        duration_seconds=42.5,
        turn_count=3,
    )
    row = db.get_briefing_audio(audio_id)
    assert row["status"] == "complete"
    assert row["file_path"] == "/tmp/briefing.mp3"
    assert row["duration_seconds"] == 42.5
    assert row["turn_count"] == 3


def test_update_briefing_audio_can_record_an_error():
    db = SubscriptionsDB(":memory:", "test")
    script_id = _make_script(db)
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")

    db.update_briefing_audio(audio_id, status="failed", error="provider said no")

    row = db.get_briefing_audio(audio_id)
    assert row["status"] == "failed"
    assert row["error"] == "provider said no"


def test_update_briefing_audio_also_enforces_sql_validation_not_just_the_allowlist(
    monkeypatch,
):
    """The Step 5(a) mutation-check precedent (`test_update_briefing_script_
    also_enforces_sql_validation_not_just_the_allowlist`), kept as a
    permanent regression test: dropping `validate_identifier` from
    `update_briefing_audio` must not leave this divergence undetected."""
    db = SubscriptionsDB(":memory:", "test")
    script_id = _make_script(db)
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")

    monkeypatch.setattr(
        subscriptions_db_module, "validate_identifier", lambda *a, **k: False
    )
    with pytest.raises(ValueError, match="status"):
        db.update_briefing_audio(audio_id, status="complete")


def test_update_briefing_audio_rejects_voice_snapshot_json_write_once():
    """`voice_snapshot_json` is deliberately NOT in the update allowlist --
    it is write-once, exactly as `roster_snapshot_json` is on
    `briefing_scripts` (see `_snapshot_roster`'s docstring in
    `briefing_cast.py`): an artifact's snapshot must not be revisable."""
    db = SubscriptionsDB(":memory:", "test")
    script_id = _make_script(db)
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")

    with pytest.raises(ValueError, match="voice_snapshot_json"):
        db.update_briefing_audio(audio_id, voice_snapshot_json='[{"changed": true}]')

    # Untouched.
    assert db.get_briefing_audio(audio_id)["voice_snapshot_json"] == "[]"


def test_briefing_audio_cascades_on_script_delete():
    """`briefing_audio.script_id` carries `ON DELETE CASCADE`. This DB
    overrides `_get_connection` to set `PRAGMA foreign_keys = ON` per
    connection, so unlike a bare sqlite3 connection this cascade is actually
    live, not merely declared -- confirmed here rather than assumed."""
    db = SubscriptionsDB(":memory:", "test")
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    script_id = _make_script(db)
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    assert db.get_briefing_audio(audio_id) is not None

    with db.transaction() as conn:
        conn.execute("DELETE FROM briefing_scripts WHERE id = ?", (script_id,))

    assert db.get_briefing_audio(audio_id) is None


def test_list_briefing_audio_returns_newest_first_by_identity():
    """Identities, not just a count -- a query that returns "three rows"
    without honoring recency cannot pass this by accident."""
    db = SubscriptionsDB(":memory:", "test")
    script_id = _make_script(db)

    first = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    second = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    third = db.create_briefing_audio(script_id, voice_snapshot_json="[]")

    listed = db.list_briefing_audio(script_id)
    assert [row["id"] for row in listed] == [third, second, first]


def test_list_briefing_audio_is_scoped_to_its_own_script():
    db = SubscriptionsDB(":memory:", "test")
    script_a = _make_script(db)
    script_b = _make_script(db)

    audio_a = db.create_briefing_audio(script_a, voice_snapshot_json="[]")
    db.create_briefing_audio(script_b, voice_snapshot_json="[]")

    listed = db.list_briefing_audio(script_a)
    assert [row["id"] for row in listed] == [audio_a]


def test_list_briefing_audio_limit_returns_exactly_limit_rows():
    """CLAUDE.md Performance Rules: DB listing results must be paginated.
    Seeding `limit + 5` rows and asking for `limit` must return exactly
    `limit` rows, not the whole set."""
    db = SubscriptionsDB(":memory:", "test")
    script_id = _make_script(db)
    limit = 3
    for _ in range(limit + 5):
        db.create_briefing_audio(script_id, voice_snapshot_json="[]")

    assert len(db.list_briefing_audio(script_id, limit=limit)) == limit


def test_list_briefing_audio_offset_pages_through_every_row_without_gaps_or_repeats():
    """Paging with `limit`/`offset` must walk the same newest-first ordering
    `test_list_briefing_audio_returns_newest_first_by_identity` pins,
    covering every row exactly once."""
    db = SubscriptionsDB(":memory:", "test")
    script_id = _make_script(db)
    ids = [
        db.create_briefing_audio(script_id, voice_snapshot_json="[]")
        for _ in range(7)
    ]
    expected_newest_first = list(reversed(ids))

    limit = 3
    seen: list[int] = []
    offset = 0
    while True:
        page = db.list_briefing_audio(script_id, limit=limit, offset=offset)
        if not page:
            break
        seen.extend(row["id"] for row in page)
        offset += limit

    assert seen == expected_newest_first
