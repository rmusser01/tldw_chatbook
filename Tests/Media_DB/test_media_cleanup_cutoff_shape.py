"""Media cleanup cutoffs must share the writer's timestamp shape (TASK-32803.3).

``last_modified``/``trash_date`` are written as ``%Y-%m-%dT%H:%M:%S.mmmZ`` but
the cleanup readers built their cutoff as ``%Y-%m-%d %H:%M:%S`` (space
separator). SQLite compares TEXT lexically and ``'T'`` (0x54) > ``' '`` (0x20),
so on the cutoff's calendar date every stored value sorts AFTER the cutoff and
the row is skipped. All cutoffs now derive from the one writer,
``MediaDatabase._utc_timestamp_str``.
"""

from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


def test_cutoff_and_stored_share_one_lexically_correct_shape():
    """The mechanism, gate-free: a row soft-deleted earlier on the cutoff's own
    calendar date must sort BEFORE the cutoff (so ``last_modified < cutoff``
    selects it), which the shared shape guarantees and the old space form did
    not."""
    stored = MediaDatabase._utc_timestamp_str(
        datetime(2026, 8, 19, 0, 0, 1, tzinfo=timezone.utc)
    )
    cutoff = MediaDatabase._utc_timestamp_str(
        datetime(2026, 8, 19, 2, 27, 11, tzinfo=timezone.utc)
    )
    assert stored == "2026-08-19T00:00:01.000Z"
    assert cutoff == "2026-08-19T02:27:11.000Z"
    # The row is older -> `last_modified < cutoff` must be True -> selected.
    assert stored < cutoff
    # The exact bug the fix removes: a space-separated cutoff sorted BEFORE
    # same-day stored values, so the same row looked NEWER than its cutoff.
    old_space_cutoff = "2026-08-19 02:27:11"
    assert not (stored < old_space_cutoff)


def test_millisecond_precision_is_always_present():
    """``[:-3]`` on ``%f`` keeps three digits even when microsecond is 0, so the
    shape is fixed-width and lexically stable."""
    assert MediaDatabase._utc_timestamp_str(
        datetime(2026, 1, 2, 3, 4, 5, 0, tzinfo=timezone.utc)
    ) == "2026-01-02T03:04:05.000Z"


def _make_db(tmp_path):
    try:
        return MediaDatabase(db_path=str(tmp_path / "cutoff.sqlite"), client_id="cutoff-test")
    except Exception as exc:  # noqa: BLE001
        # The ADR-126 admission gate raises RecoveryRequired in a clean
        # worktree, wrapped by _initialize_schema as DatabaseError. This
        # test's assertions run in CI where the gate is bound.
        names = {type(e).__name__ for e in (exc, exc.__cause__, exc.__context__) if e}
        if "RecoveryRequired" in names or "raw_source_selection_changed" in str(exc):
            pytest.skip("storage admission not bound in this worktree (runs in CI)")
        raise


def test_boundary_date_row_is_a_deletion_candidate(tmp_path):
    """Functional: a row soft-deleted just over the cutoff, on the cutoff's own
    calendar date, is returned by get_deletion_candidates and hard-deleted."""
    db = _make_db(tmp_path)
    try:
        media_id, _uuid, _msg = db.add_media_with_keywords(
            url="https://example.test/cutoff",
            title="Cutoff row",
            media_type="document",
            content="body",
            keywords=["k"],
        )
        # Soft-delete it with last_modified one second past a 30-day cutoff,
        # in the stored shape -- same calendar date as get_deletion_candidates'
        # internally-computed cutoff.
        stamp = db._utc_timestamp_str(
            datetime.now(timezone.utc) - timedelta(days=30, seconds=1)
        )
        with db.transaction() as conn:
            conn.execute(
                "UPDATE Media SET deleted = 1, last_modified = ? WHERE id = ?",
                (stamp, media_id),
            )

        candidates = db.get_deletion_candidates(days_old=30)
        assert any(row["id"] == media_id for row in candidates), (
            "a boundary-date row was skipped by the cleanup cutoff"
        )
        assert db.hard_delete_old_media(days_old=30) >= 1
    finally:
        db.close_connection()
