"""Task 2 (chunking-agent-tools): the backend chunk-read seam.

``LocalMediaReadingService.get_library_media_chunks`` -- the stored-row read
the fetch tool (Task 3) consumes. All Media-DB work here is REAL
(``tmp_path`` DBs): family filtering, budget-bounded neighbors, and the
item-level disambiguation signals (``families`` / ``engine_versions`` /
``media_version``) are observable DB behavior, not mock dance.

Spec anchors: design §4.2 (fetch contract), §8.10 (chunk_type families),
§8.12 (byte budget wins over context), §8.13 (pre-v6 unstamped rows
readable).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Media.local_media_reading_service import (
    PRIMARY_CHUNK_FAMILY,
    LocalMediaReadingService,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def media_db(tmp_path: Path) -> MediaDatabase:
    return MediaDatabase(tmp_path / "media.db", client_id="chunk-read-tests")


@pytest.fixture()
def service(media_db: MediaDatabase) -> LocalMediaReadingService:
    return LocalMediaReadingService(media_db)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _seed_media(db: MediaDatabase, content: str = "seed content") -> int:
    """One active media item, no chunk rows (``chunks=None`` skips the writer)."""
    media_id, _, _ = db.add_media_with_keywords(
        title="chunk-reads fixture",
        media_type="plaintext",
        content=content,
        url=f"https://example.test/{uuid.uuid4()}",
    )
    assert media_id is not None
    return int(media_id)


def _insert_chunk(
    db: MediaDatabase,
    media_id: int,
    chunk_index: int,
    text: str,
    *,
    chunk_type: str | None = None,
    engine_version: str | None = None,
    metadata: str | None = None,
    start_char: int | None = None,
    end_char: int | None = None,
) -> None:
    """Direct row insert -- the only way to land typed families / stamps /
    raw metadata JSON deterministically (flat ingest writes NULL types)."""
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO UnvectorizedMediaChunks (
                media_id, chunk_text, chunk_index, start_char, end_char,
                chunk_type, metadata, chunk_engine_version, uuid,
                last_modified, version, client_id, deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 0)
            """,
            (
                media_id,
                text,
                chunk_index,
                start_char,
                end_char,
                chunk_type,
                metadata,
                engine_version,
                str(uuid.uuid4()),
                _now(),
                db.client_id,
            ),
        )


def _seed_flat_chunks(
    db: MediaDatabase, media_id: int, texts: list[str], *, start_at: int = 0
) -> None:
    """Flat (NULL-family, unstamped) rows -- the pre-v6 library shape."""
    offset = 0
    for position, text in enumerate(texts):
        _insert_chunk(
            db,
            media_id,
            start_at + position,
            text,
            start_char=offset,
            end_char=offset + len(text),
        )
        offset += len(text)


def _media_version(db: MediaDatabase, media_id: int) -> int:
    row = (
        db.get_connection()
        .execute("SELECT version FROM Media WHERE id = ?", (media_id,))
        .fetchone()
    )
    return int(row["version"])


def _bump_media_version(db: MediaDatabase, media_id: int) -> None:
    """Sync-trigger-legal version bump (version+1 with fresh last_modified)."""
    with db.transaction() as conn:
        conn.execute(
            "UPDATE Media SET version = version + 1, last_modified = ? WHERE id = ?",
            (_now(), media_id),
        )


def _soft_delete_chunk(db: MediaDatabase, media_id: int, chunk_index: int) -> None:
    with db.transaction() as conn:
        conn.execute(
            """
            UPDATE UnvectorizedMediaChunks
            SET deleted = 1, version = version + 1, last_modified = ?
            WHERE media_id = ? AND chunk_index = ? AND deleted = 0
            """,
            (_now(), media_id, chunk_index),
        )


# ---------------------------------------------------------------------------
# Exact unit fetch
# ---------------------------------------------------------------------------


def test_exact_index_fetch_returns_requested_chunk_and_item_signals(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(
        media_db,
        media_id,
        ["alpha chunk zero", "beta chunk one", "gamma chunk two words here"],
    )

    result = service.get_library_media_chunks(media_id, chunk_index=2, budget=10_000)

    assert result is not None
    assert [chunk["chunk_index"] for chunk in result["chunks"]] == [2]
    chunk = result["chunks"][0]
    assert chunk["text"] == "gamma chunk two words here"
    assert chunk["chunk_type"] == "primary"
    assert chunk["start_char"] == len("alpha chunk zero") + len("beta chunk one")
    assert chunk["end_char"] == (
        len("alpha chunk zero") + len("beta chunk one") + len("gamma chunk two words here")
    )
    assert chunk["word_count"] == len(["gamma", "chunk", "two", "words", "here"])
    assert chunk["metadata"] == {}
    assert result["families"] == ["primary"]
    assert result["dropped_neighbors"] == 0
    assert result["media_version"] == _media_version(media_db, media_id)


def test_media_id_coerced_from_string_like(service: LocalMediaReadingService, media_db: MediaDatabase):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(media_db, media_id, ["only chunk"])

    result = service.get_library_media_chunks(str(media_id), chunk_index=0, budget=100)

    assert result is not None
    assert [chunk["chunk_index"] for chunk in result["chunks"]] == [0]


# ---------------------------------------------------------------------------
# Families (spec §8.10)
# ---------------------------------------------------------------------------


def test_null_and_typed_families_reported_and_filter_selects_family(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(media_db, media_id, ["flat zero", "flat one"])
    _insert_chunk(media_db, media_id, 0, "section zero", chunk_type="section")
    _insert_chunk(media_db, media_id, 1, "section one", chunk_type="section")

    default_result = service.get_library_media_chunks(
        media_id, chunk_index=0, budget=10_000
    )
    typed_result = service.get_library_media_chunks(
        media_id, chunk_index=0, chunk_type="section", budget=10_000
    )

    # families is the item-level disambiguation signal: reported the same
    # regardless of the filter (Task 3 errors on >1 family + omitted filter).
    assert default_result is not None and typed_result is not None
    assert default_result["families"] == ["primary", "section"]
    assert typed_result["families"] == ["primary", "section"]

    # Default filter = the primary (NULL) family; explicit filter selects it.
    assert [chunk["text"] for chunk in default_result["chunks"]] == ["flat zero"]
    assert default_result["chunks"][0]["chunk_type"] == "primary"
    assert [chunk["text"] for chunk in typed_result["chunks"]] == ["section zero"]
    assert typed_result["chunks"][0]["chunk_type"] == "section"


def test_primary_string_alias_selects_null_family(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(media_db, media_id, ["flat zero"])
    _insert_chunk(media_db, media_id, 0, "section zero", chunk_type="section")

    via_alias = service.get_library_media_chunks(
        media_id, chunk_index=0, chunk_type="primary", budget=10_000
    )
    via_default = service.get_library_media_chunks(media_id, chunk_index=0, budget=10_000)

    assert via_alias is not None and via_default is not None
    assert via_alias["chunks"] == via_default["chunks"]
    assert [chunk["text"] for chunk in via_alias["chunks"]] == ["flat zero"]


def test_sentinel_and_none_mean_primary_family(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(media_db, media_id, ["flat zero", "flat one"])

    via_sentinel = service.get_library_media_chunks(
        media_id, chunk_index=1, chunk_type=PRIMARY_CHUNK_FAMILY, budget=10_000
    )
    via_none = service.get_library_media_chunks(
        media_id, chunk_index=1, chunk_type=None, budget=10_000
    )

    assert via_sentinel is not None and via_none is not None
    assert [chunk["text"] for chunk in via_sentinel["chunks"]] == ["flat one"]
    assert via_none["chunks"] == via_sentinel["chunks"]


def test_filter_to_missing_family_reports_requested_chunk_absent(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(media_db, media_id, ["flat zero"])

    result = service.get_library_media_chunks(
        media_id, chunk_index=0, chunk_type="table", budget=10_000
    )

    # Contract is "absent", never a raise -- Task 3 maps it to the named error.
    assert result is not None
    assert result["chunks"] == []
    assert result["families"] == ["primary"]
    assert result["dropped_neighbors"] == 0


def test_out_of_range_index_reports_requested_chunk_absent(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(media_db, media_id, ["flat zero", "flat one"])

    result = service.get_library_media_chunks(media_id, chunk_index=99, budget=10_000)

    assert result is not None
    assert result["chunks"] == []
    assert result["families"] == ["primary"]
    assert result["media_version"] == _media_version(media_db, media_id)


# ---------------------------------------------------------------------------
# Context + byte budget (spec §8.12)
# ---------------------------------------------------------------------------


def test_context_neighbors_centered_and_ordered_under_budget(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(
        media_db,
        media_id,
        ["n0", "n1", "n2", "requested n3", "n4", "n5", "n6"],
    )

    result = service.get_library_media_chunks(
        media_id, chunk_index=3, context=2, budget=10_000
    )

    assert result is not None
    assert [chunk["chunk_index"] for chunk in result["chunks"]] == [1, 2, 3, 4, 5]
    # Centered: the requested chunk sits at the middle position.
    assert result["chunks"][2]["text"] == "requested n3"
    assert result["dropped_neighbors"] == 0


def test_context_window_clamped_at_item_edges_without_dropped_count(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(media_db, media_id, ["head n0", "n1", "n2", "n3"])

    result = service.get_library_media_chunks(
        media_id, chunk_index=0, context=2, budget=10_000
    )

    assert result is not None
    assert [chunk["chunk_index"] for chunk in result["chunks"]] == [0, 1, 2]
    assert result["dropped_neighbors"] == 0


def test_budget_overflow_drops_farther_neighbors_and_counts_them(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    # Uniform 10-byte neighbor texts: nearest-first inclusion is exact.
    _seed_flat_chunks(
        media_db,
        media_id,
        ["0123456789", "0123456789", "requested!", "0123456789", "0123456789"],
    )

    # context=3 -> 2 candidates on each side; budget fits exactly 2 neighbors.
    result = service.get_library_media_chunks(
        media_id, chunk_index=2, context=3, budget=20
    )

    assert result is not None
    assert [chunk["chunk_index"] for chunk in result["chunks"]] == [1, 2, 3]
    assert result["dropped_neighbors"] == 2


def test_budget_bounds_neighbors_only_requested_chunk_always_included(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(
        media_db,
        media_id,
        ["0123456789", "a very long requested chunk body", "0123456789"],
    )

    result = service.get_library_media_chunks(
        media_id, chunk_index=1, context=1, budget=0
    )

    assert result is not None
    assert [chunk["chunk_index"] for chunk in result["chunks"]] == [1]
    assert result["dropped_neighbors"] == 2


def test_budget_counts_bytes_not_characters(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    # "ααα" is 3 characters but 6 UTF-8 bytes; budget 5 fits the character
    # count but not the byte count -- §8.12's budget is bytes of text.
    _seed_flat_chunks(media_db, media_id, ["ααα", "requested", "ααα"])

    result = service.get_library_media_chunks(
        media_id, chunk_index=1, context=1, budget=5
    )

    assert result is not None
    assert [chunk["chunk_index"] for chunk in result["chunks"]] == [1]
    assert result["dropped_neighbors"] == 2


def test_neighbors_filter_to_requested_family(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(media_db, media_id, ["flat n0", "flat n1", "flat n2"])
    _insert_chunk(media_db, media_id, 0, "sec n0", chunk_type="section")
    _insert_chunk(media_db, media_id, 1, "sec n1", chunk_type="section")
    _insert_chunk(media_db, media_id, 2, "sec n2", chunk_type="section")

    result = service.get_library_media_chunks(
        media_id, chunk_index=1, chunk_type="section", context=1, budget=10_000
    )

    assert result is not None
    assert [chunk["text"] for chunk in result["chunks"]] == ["sec n0", "sec n1", "sec n2"]


# ---------------------------------------------------------------------------
# Degradation + staleness signals (spec §8.13 / §4.2 revision)
# ---------------------------------------------------------------------------


def test_item_without_stored_rows_returns_none(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)

    assert service.get_library_media_chunks(media_id, chunk_index=0, budget=100) is None
    assert (
        service.get_library_media_chunks(media_id + 4_242, chunk_index=0, budget=100)
        is None
    )


def test_pre_v6_unstamped_rows_readable_and_reported_legacy(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    # add_media_with_keywords persists unstamped rows: the real pre-v6 shape.
    legacy_id, _, _ = media_db.add_media_with_keywords(
        title="legacy ingest",
        media_type="plaintext",
        content="legacy body text",
        url=f"https://example.test/legacy/{uuid.uuid4()}",
        chunks=[
            {"text": "legacy zero", "start_char": 0, "end_char": 11},
            {"text": "legacy one", "start_char": 11, "end_char": 22},
        ],
    )
    assert legacy_id is not None

    result = service.get_library_media_chunks(
        int(legacy_id), chunk_index=1, budget=100
    )

    assert result is not None
    assert [chunk["chunk_index"] for chunk in result["chunks"]] == [1]
    assert result["chunks"][0]["text"] == "legacy one"
    assert result["engine_versions"] == ["legacy"]


def test_stamped_and_mixed_engine_versions_reported_verbatim(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _insert_chunk(
        media_db,
        media_id,
        0,
        "stamped",
        engine_version="9.9.9",
        metadata=json.dumps({"chunk_method": "sentences"}),
    )
    _insert_chunk(media_db, media_id, 1, "unstamped")

    result = service.get_library_media_chunks(media_id, chunk_index=0, budget=100)

    assert result is not None
    assert result["engine_versions"] == ["9.9.9", "legacy"]


def test_media_version_tracks_media_row_version(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(media_db, media_id, ["row"])
    before = service.get_library_media_chunks(media_id, chunk_index=0, budget=100)
    assert before is not None
    version_before = before["media_version"]

    _bump_media_version(media_db, media_id)

    after = service.get_library_media_chunks(media_id, chunk_index=0, budget=100)
    assert after is not None
    assert after["media_version"] == version_before + 1


def test_metadata_json_parsed_and_unparseable_falls_back_to_empty_dict(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _insert_chunk(
        media_db,
        media_id,
        0,
        "with metadata",
        metadata=json.dumps({"chunk_method": "sentences", "total_chunks": 2}),
    )
    _insert_chunk(media_db, media_id, 1, "broken metadata", metadata="not-json{")
    _insert_chunk(media_db, media_id, 2, "null metadata", metadata=None)

    result = service.get_library_media_chunks(
        media_id, chunk_index=0, context=2, budget=10_000
    )

    assert result is not None
    by_index = {chunk["chunk_index"]: chunk for chunk in result["chunks"]}
    assert by_index[0]["metadata"] == {"chunk_method": "sentences", "total_chunks": 2}
    assert by_index[1]["metadata"] == {}
    assert by_index[2]["metadata"] == {}


def test_deleted_chunk_rows_excluded(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(media_db, media_id, ["keep zero", "drop me", "keep two"])
    _soft_delete_chunk(media_db, media_id, 1)

    result = service.get_library_media_chunks(
        media_id, chunk_index=0, context=2, budget=10_000
    )

    assert result is not None
    assert [chunk["text"] for chunk in result["chunks"]] == ["keep zero", "keep two"]


def test_deleted_only_typed_family_dropped_from_families(
    service: LocalMediaReadingService, media_db: MediaDatabase
):
    media_id = _seed_media(media_db)
    _seed_flat_chunks(media_db, media_id, ["flat zero"])
    _insert_chunk(media_db, media_id, 0, "section zero", chunk_type="section")
    _soft_delete_chunk_by_type(media_db, media_id, "section")

    result = service.get_library_media_chunks(media_id, chunk_index=0, budget=100)

    assert result is not None
    assert result["families"] == ["primary"]


def _soft_delete_chunk_by_type(
    db: MediaDatabase, media_id: int, chunk_type: str
) -> None:
    with db.transaction() as conn:
        conn.execute(
            """
            UPDATE UnvectorizedMediaChunks
            SET deleted = 1, version = version + 1, last_modified = ?
            WHERE media_id = ? AND chunk_type = ? AND deleted = 0
            """,
            (_now(), media_id, chunk_type),
        )
