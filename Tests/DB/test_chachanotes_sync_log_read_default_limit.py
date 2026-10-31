"""task-34399 item 7: ``get_sync_log_entries`` default limit.

The reader's ``limit`` defaulted to ``None`` -- an unbounded read of a log
whose rows carry full-row JSON payloads. The default is now 500; explicit
limits (the only production caller passes one) and ``limit=None`` keep
their exact prior meaning.

Schema creation itself seeds one sync event (the default character card),
so every count below is taken relative to that baseline instead of
assuming change_id 1 is ours.
"""

import json

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

_TOTAL_ROWS = 510
_DEFAULT_LIMIT = 500


@pytest.fixture
def db(tmp_path):
    return CharactersRAGDB(tmp_path / "sync-limit.db", client_id="sync-test")


@pytest.fixture
def seeded(db):
    """Seed 510 sync_log rows directly (entity_id 'conv-*'; the
    notes-dispatch predicate cannot filter entity='conversations' rows)."""
    baseline_change_id = db.get_latest_sync_log_change_id()
    baseline_count = db.execute_query(
        "SELECT COUNT(*) AS cnt FROM sync_log"
    ).fetchone()["cnt"]
    with db.transaction() as cursor:
        for i in range(_TOTAL_ROWS):
            cursor.execute(
                "INSERT INTO sync_log (entity, entity_id, operation, timestamp,"
                " client_id, version, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "conversations",
                    f"conv-{i}",
                    "create",
                    "2026-01-01T00:00:00",
                    "sync-test",
                    1,
                    json.dumps({"i": i}),
                ),
            )
    return {
        "db": db,
        "baseline_change_id": baseline_change_id,
        "total": baseline_count + _TOTAL_ROWS,
    }


def _seeded_entries(entries):
    return [entry for entry in entries if entry["entity_id"].startswith("conv-")]


def test_default_read_is_bounded_at_500(seeded):
    entries = seeded["db"].get_sync_log_entries()

    assert len(entries) == _DEFAULT_LIMIT
    change_ids = [entry["change_id"] for entry in entries]
    assert change_ids == sorted(change_ids)
    assert change_ids[-1] == change_ids[0] + _DEFAULT_LIMIT - 1


def test_since_change_id_still_pages_past_the_default(seeded):
    cut = seeded["baseline_change_id"] + _DEFAULT_LIMIT
    entries = seeded["db"].get_sync_log_entries(since_change_id=cut)

    assert len(entries) == seeded["total"] - cut
    assert entries[0]["change_id"] == cut + 1


def test_explicit_limit_is_respected(seeded):
    entries = seeded["db"].get_sync_log_entries(limit=10)

    assert len(entries) == 10


def test_none_still_means_unbounded(seeded):
    entries = seeded["db"].get_sync_log_entries(limit=None)

    assert len(entries) == seeded["total"]
    assert len(_seeded_entries(entries)) == _TOTAL_ROWS


def test_payloads_still_decode(seeded):
    entries = _seeded_entries(seeded["db"].get_sync_log_entries(limit=None))

    assert len(entries) == _TOTAL_ROWS
    assert entries[0]["payload"] == {"i": 0}
    assert entries[-1]["payload"] == {"i": _TOTAL_ROWS - 1}
