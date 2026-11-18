"""task-34399 item 5: ``_persist_chunks`` batched insert parity.

``add_media_with_keywords`` used to run one INSERT per chunk inside the
media transaction; it now builds the row params and issues a single
``executemany`` (the shape ``process_unvectorized_chunks`` already uses).
These tests pin that the persisted rows and the per-chunk sync events are
byte-identical to the per-chunk loop's output: same columns, same index
consumption for invalid entries, same ``create`` events with the chunk
text still embedded in each payload.
"""

import json
from datetime import datetime

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase as Database


@pytest.fixture
def db():
    db = Database(db_path=":memory:", client_id="batch-client")
    yield db
    db.close_connection()


def _chunk(i):
    return {
        "text": f"chunk {i} body text",
        "start_char": i * 10,
        "end_char": i * 10 + 10,
        "chunk_type": "paragraph",
        "metadata": {"position": i},
    }


def _chunk_rows(db, media_id):
    return db.execute_query(
        "SELECT * FROM UnvectorizedMediaChunks WHERE media_id = ?"
        " ORDER BY chunk_index",
        (media_id,),
    ).fetchall()


def _chunk_events(db):
    return db.execute_query(
        "SELECT entity_uuid, operation, version, payload FROM sync_log"
        " WHERE entity = 'UnvectorizedMediaChunks' ORDER BY change_id"
    ).fetchall()


def test_chunks_persist_identically_with_one_batch(db):
    # Entry at index 3 is invalid and must consume its index, exactly as the
    # per-chunk loop did.
    chunks = [_chunk(0), _chunk(1), _chunk(2), {"text": None}, _chunk(4)]

    media_id, _media_uuid, _msg = db.add_media_with_keywords(
        title="Batch parity",
        media_type="article",
        content="parent content",
        keywords=["parity"],
        chunks=chunks,
    )

    rows = _chunk_rows(db, media_id)
    assert [row["chunk_index"] for row in rows] == [0, 1, 2, 4]
    for index, row in zip([0, 1, 2, 4], rows):
        assert row["chunk_text"] == chunks[index]["text"]
        assert row["start_char"] == chunks[index]["start_char"]
        assert row["end_char"] == chunks[index]["end_char"]
        assert row["chunk_type"] == "paragraph"
        assert row["is_processed"] == 0
        assert json.loads(row["metadata"]) == {"position": index}
        assert row["version"] == 1
        assert row["deleted"] == 0
        assert row["client_id"] == "batch-client"
        assert row["prev_version"] is None
        assert row["merge_parent_uuid"] is None
        assert row["uuid"], "every persisted chunk carries its uuid"


def test_sync_events_match_persisted_chunks_one_to_one(db):
    chunks = [_chunk(0), _chunk(1), _chunk(2)]

    media_id, _media_uuid, _msg = db.add_media_with_keywords(
        title="Batch events",
        media_type="article",
        content="parent content",
        keywords=["parity"],
        chunks=chunks,
    )

    events = _chunk_events(db)
    rows = _chunk_rows(db, media_id)
    assert len(events) == len(rows) == 3
    assert {event["entity_uuid"] for event in events} == {
        row["uuid"] for row in rows
    }
    for index, (event, row) in enumerate(zip(events, rows)):
        assert event["operation"] == "create"
        assert event["version"] == 1
        payload = json.loads(event["payload"])
        # Payload contents are unchanged from the per-chunk loop: the
        # caller's chunk dict spread in, plus the sync identity fields.
        assert payload["text"] == chunks[index]["text"]
        assert payload["metadata"] == {"position": index}
        assert payload["media_id"] == media_id
        assert payload["uuid"] == row["uuid"]
        assert payload["chunk_index"] == index
        assert payload["version"] == 1
        assert payload["client_id"] == "batch-client"
        assert payload["deleted"] == 0
        # The row column comes back parsed (PARSE_DECLTYPES); the payload
        # carries the same instant as the ISO string it was built from.
        assert datetime.fromisoformat(
            payload["creation_date"].replace("Z", "+00:00")
        ) == row["creation_date"]


def test_replace_existing_swaps_every_chunk_in_one_batch(db):
    first_chunks = [_chunk(0), _chunk(1), _chunk(2)]
    media_id, media_uuid, _msg = db.add_media_with_keywords(
        title="Batch replace",
        media_type="article",
        content="parent content",
        keywords=["parity"],
        chunks=first_chunks,
    )
    created = db.get_media_by_id(media_id)
    old_uuids = {row["uuid"] for row in _chunk_rows(db, media_id)}

    replacement = [{"text": "replacement A"}, {"text": "replacement B"}]
    returned_id, returned_uuid, _msg = db.add_media_with_keywords(
        title="Batch replace",
        media_type="article",
        content="parent content",
        keywords=["parity"],
        chunks=replacement,
        overwrite=True,
        url=created["url"],
    )
    assert (returned_id, returned_uuid) == (media_id, media_uuid)

    rows = _chunk_rows(db, media_id)
    assert [row["chunk_text"] for row in rows] == ["replacement A", "replacement B"]
    assert [row["chunk_index"] for row in rows] == [0, 1]
    assert {row["uuid"] for row in rows}.isdisjoint(old_uuids)


def test_empty_chunks_list_wipes_existing_chunks(db):
    media_id, _media_uuid, _msg = db.add_media_with_keywords(
        title="Batch wipe",
        media_type="article",
        content="parent content",
        keywords=["parity"],
        chunks=[_chunk(0), _chunk(1)],
    )
    assert len(_chunk_rows(db, media_id)) == 2

    created = db.get_media_by_id(media_id)
    db.add_media_with_keywords(
        title="Batch wipe",
        media_type="article",
        content="parent content",
        keywords=["parity"],
        chunks=[],
        overwrite=True,
        url=created["url"],
    )

    assert _chunk_rows(db, media_id) == []
