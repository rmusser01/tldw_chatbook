"""Tests for `list_watchlist_audio_episodes` (spec #2 phase 3, Task 2).

One joined `SELECT` across `briefing_audio -> briefing_scripts ->
briefings`, scoped to a watchlist through the `briefings` row each script
belongs to (`briefing_audio` has no `watchlist_id` column of its own). An
"episode" is a `briefing_audio` row that is `status='complete'` AND has a
non-NULL `file_path` -- two independent reasons a row could otherwise be
excluded, seeded independently below so a single seed can't accidentally
pass for the wrong reason.

Tasks 3 (RSS feed generation) and 5 (the export button) quote this query's
column aliases verbatim, so this file also pins the exact output shape, not
just row counts.

Same harness as `test_briefing_audio_db.py`: a real `SubscriptionsDB` on
`:memory:`, `WatchlistBundleService` for watchlist creation,
`insert_briefing` / `insert_briefing_script` / `create_briefing_audio` /
`update_briefing_audio` for the rest of the chain.
"""

from unittest.mock import Mock

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


def _make_watchlist(db: SubscriptionsDB, name: str = "w") -> int:
    return WatchlistBundleService(db).create(name=name)["id"]


def _make_briefing(db: SubscriptionsDB, watchlist_id: int, *, created_at: str) -> int:
    """Create a `briefings` row and pin its `created_at` explicitly.

    `update_briefing`'s allowlist deliberately excludes `created_at` (it is
    the row's own creation stamp, not something a caller revises), so this
    reaches past it with a raw `UPDATE` -- inside `db.transaction()`,
    matching the pattern `test_briefing_audio_db.py`'s cascade test uses for
    the same reason (a direct statement the production allowlist won't
    permit). SQLite's `CURRENT_TIMESTAMP` default only has one-second
    resolution, and this suite must prove an ordering rule (newest briefing
    first) that a same-second default would make flaky.
    """
    briefing_id = db.insert_briefing(watchlist_id)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE briefings SET created_at = ? WHERE id = ?",
            (created_at, briefing_id),
        )
    return briefing_id


def _make_script(db: SubscriptionsDB, briefing_id: int, *, preset_name: str = "p") -> int:
    return db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name=preset_name, roster_snapshot_json="[]"
    )


def _make_complete_audio(
    db: SubscriptionsDB,
    script_id: int,
    *,
    file_path: str = "/tmp/episode.mp3",
    duration_seconds: float = 123.0,
    turn_count: int = 4,
) -> int:
    """Create a `briefing_audio` row that qualifies as a playable episode."""
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]", status="complete")
    db.update_briefing_audio(
        audio_id,
        file_path=file_path,
        duration_seconds=duration_seconds,
        turn_count=turn_count,
    )
    return audio_id


def _chain(
    db: SubscriptionsDB,
    watchlist_id: int,
    *,
    briefing_created_at: str,
    preset_name: str = "p",
    file_path: str = "/tmp/episode.mp3",
) -> tuple[int, int, int]:
    """Build one full watchlist -> briefing -> script -> complete-audio chain.

    Returns:
        `(briefing_id, script_id, audio_id)`.
    """
    briefing_id = _make_briefing(db, watchlist_id, created_at=briefing_created_at)
    script_id = _make_script(db, briefing_id, preset_name=preset_name)
    audio_id = _make_complete_audio(db, script_id, file_path=file_path)
    return briefing_id, script_id, audio_id


# --- happy path: newest briefing first, by identity ----------------------


def test_returns_audio_from_both_briefings_newest_briefing_first_by_identity():
    """Identities, not just a count -- see `list_briefing_audio`'s own
    `test_list_briefing_audio_returns_newest_first_by_identity` precedent."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)

    _older_briefing, _older_script, older_audio = _chain(
        db, watchlist_id, briefing_created_at="2026-01-01 00:00:00"
    )
    _newer_briefing, _newer_script, newer_audio = _chain(
        db, watchlist_id, briefing_created_at="2026-01-02 00:00:00"
    )

    result = db.list_watchlist_audio_episodes(watchlist_id)

    assert [row["audio_id"] for row in result] == [newer_audio, older_audio]


def test_orders_by_briefings_created_at_not_audio_created_at():
    """The interface deliberately differs from `list_briefing_audio`'s own
    `created_at` ordering: a podcast feed is ordered by episode (briefing)
    recency, not by when the audio happened to be rendered. Render the
    OLDER briefing's audio SECOND (i.e. its own row is "newer" by
    audio.created_at/id) and confirm the briefing's recency still wins."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)

    newer_briefing_id = _make_briefing(db, watchlist_id, created_at="2026-01-02 00:00:00")
    older_briefing_id = _make_briefing(db, watchlist_id, created_at="2026-01-01 00:00:00")

    newer_briefing_script = _make_script(db, newer_briefing_id)
    newer_briefing_audio = _make_complete_audio(db, newer_briefing_script)

    # Rendered AFTER the above, so it has a larger audio.id / later
    # audio.created_at -- yet its parent briefing is the OLDER one.
    older_briefing_script = _make_script(db, older_briefing_id)
    older_briefing_audio = _make_complete_audio(db, older_briefing_script)
    assert older_briefing_audio > newer_briefing_audio  # sanity: rendered later

    result = db.list_watchlist_audio_episodes(watchlist_id)

    assert [row["audio_id"] for row in result] == [newer_briefing_audio, older_briefing_audio]


def test_tiebreaks_same_briefing_multiple_audio_by_audio_id_desc():
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    briefing_id = _make_briefing(db, watchlist_id, created_at="2026-01-01 00:00:00")

    script_a = _make_script(db, briefing_id)
    audio_a = _make_complete_audio(db, script_a, file_path="/tmp/a.mp3")
    script_b = _make_script(db, briefing_id)
    audio_b = _make_complete_audio(db, script_b, file_path="/tmp/b.mp3")

    result = db.list_watchlist_audio_episodes(watchlist_id)

    assert [row["audio_id"] for row in result] == [audio_b, audio_a]


# --- the two exclusion predicates, seeded independently -------------------


def test_excludes_a_non_complete_audio_row_even_with_a_file_path():
    """Isolates the `status != 'complete'` predicate: this row has a real
    `file_path`, so only its status can be responsible for exclusion."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    briefing_id = _make_briefing(db, watchlist_id, created_at="2026-01-01 00:00:00")
    script_id = _make_script(db, briefing_id)

    failed_audio = db.create_briefing_audio(script_id, voice_snapshot_json="[]", status="failed")
    db.update_briefing_audio(failed_audio, file_path="/tmp/failed.mp3")

    assert db.list_watchlist_audio_episodes(watchlist_id) == []


def test_excludes_a_complete_audio_row_with_null_file_path():
    """Isolates the `file_path IS NOT NULL` predicate: this row is already
    `status='complete'`, so only the missing file can be responsible."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    briefing_id = _make_briefing(db, watchlist_id, created_at="2026-01-01 00:00:00")
    script_id = _make_script(db, briefing_id)

    # status='complete' but file_path is never set (stays NULL).
    db.create_briefing_audio(script_id, voice_snapshot_json="[]", status="complete")

    assert db.list_watchlist_audio_episodes(watchlist_id) == []


# --- scoping: identity, not count ------------------------------------------


def test_scoped_to_watchlist_by_identity_not_count():
    """The security-adjacent property: an export must never include another
    watchlist's audio. Seeding a SECOND watchlist with its own complete
    audio and asserting exact ids (not `len(...)`) catches a join bug that
    happens to preserve row count while returning the wrong rows."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_a = _make_watchlist(db, "watchlist a")
    watchlist_b = _make_watchlist(db, "watchlist b")

    _, _, audio_a = _chain(db, watchlist_a, briefing_created_at="2026-01-01 00:00:00")
    _chain(db, watchlist_b, briefing_created_at="2026-01-01 00:00:00")

    result = db.list_watchlist_audio_episodes(watchlist_a)

    assert [row["audio_id"] for row in result] == [audio_a]


def test_empty_watchlist_returns_empty_list():
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)

    assert db.list_watchlist_audio_episodes(watchlist_id) == []


# --- pagination -------------------------------------------------------------


def test_limit_returns_exactly_limit_rows():
    """CLAUDE.md Performance Rules: DB listing results must be paginated.
    Matches `list_briefing_audio`'s own
    `test_list_briefing_audio_limit_returns_exactly_limit_rows` shape."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    limit = 3
    for n in range(limit + 5):
        _chain(db, watchlist_id, briefing_created_at=f"2026-01-{n + 1:02d} 00:00:00")

    assert len(db.list_watchlist_audio_episodes(watchlist_id, limit=limit)) == limit


def test_offset_pages_through_every_row_without_gaps_or_repeats():
    """Matches `list_briefing_audio`'s own
    `test_list_briefing_audio_offset_pages_through_every_row_without_gaps_or_repeats`
    shape: walk every page and confirm the union covers every seeded row
    exactly once, in the same newest-first order the identity test above
    pins."""
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    audio_ids = [
        _chain(db, watchlist_id, briefing_created_at=f"2026-01-{n + 1:02d} 00:00:00")[2]
        for n in range(7)
    ]
    expected_newest_first = list(reversed(audio_ids))

    limit = 3
    seen: list[int] = []
    offset = 0
    while True:
        page = db.list_watchlist_audio_episodes(watchlist_id, limit=limit, offset=offset)
        if not page:
            break
        seen.extend(row["audio_id"] for row in page)
        offset += limit

    assert seen == expected_newest_first


def test_limit_and_offset_are_bound_as_real_sql_parameters():
    """Guards against a Python-side-slice reimplementation that would still
    pass the two pagination tests above but fetch every row from SQLite on
    every call. Spies on the real connection (`Mock(wraps=...)`, the
    technique `test_briefing_selection.py`'s
    `test_window_query_parameter_count_does_not_scale_with_queue_size` and
    `test_briefing_presets_db.py`'s chunking test use) and asserts the bound
    parameter tuple ends with the exact `(limit, offset)` values passed in.
    """
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    _chain(db, watchlist_id, briefing_created_at="2026-01-01 00:00:00")

    real_conn = db.conn  # materialise this thread's connection first
    spy_conn = Mock(wraps=real_conn)
    db._local.conn = spy_conn
    try:
        db.list_watchlist_audio_episodes(watchlist_id, limit=17, offset=5)
    finally:
        db._local.conn = real_conn

    bound_params = [
        call.args[1] for call in spy_conn.execute.call_args_list if len(call.args) > 1
    ]
    matching = [params for params in bound_params if tuple(params)[-2:] == (17, 5)]
    assert matching, (
        f"expected a statement bound with (..., 17, 5) as its trailing LIMIT/OFFSET "
        f"parameters; saw {bound_params}"
    )


# --- column aliases: the contract Tasks 3 and 5 quote verbatim ------------


def test_row_shape_carries_every_documented_alias_with_correct_values():
    db = SubscriptionsDB(":memory:", "test")
    watchlist_id = _make_watchlist(db)
    briefing_id = _make_briefing(db, watchlist_id, created_at="2026-01-01 00:00:00")
    db.update_briefing(
        briefing_id,
        status="complete",
        model_used="gpt-test",
        covers_from_ts="2026-01-01T00:00:00+00:00",
    )
    script_id = _make_script(db, briefing_id, preset_name="Two Host Debate")
    audio_id = _make_complete_audio(
        db, script_id, file_path="/tmp/episode.mp3", duration_seconds=321.5, turn_count=9
    )

    result = db.list_watchlist_audio_episodes(watchlist_id)

    assert len(result) == 1
    row = result[0]
    assert set(row.keys()) == {
        "audio_id",
        "script_id",
        "briefing_id",
        "file_path",
        "duration_seconds",
        "turn_count",
        "preset_name",
        "briefing_created_at",
        "briefing_status",
        "covers_from_ts",
        "model_used",
    }
    assert row["audio_id"] == audio_id
    assert row["script_id"] == script_id
    assert row["briefing_id"] == briefing_id
    assert row["file_path"] == "/tmp/episode.mp3"
    assert row["duration_seconds"] == 321.5
    assert row["turn_count"] == 9
    assert row["preset_name"] == "Two Host Debate"
    assert row["briefing_created_at"] == "2026-01-01 00:00:00"
    assert row["briefing_status"] == "complete"
    assert row["covers_from_ts"] == "2026-01-01T00:00:00+00:00"
    assert row["model_used"] == "gpt-test"
