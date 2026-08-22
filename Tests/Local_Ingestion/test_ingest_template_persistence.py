"""Task 11 (PR D, AC 38): template persistence at the ingest write path.

Spec §9.2 tail: persisted chunks fill the ``chunking_template`` /
``chunking_params`` columns that migration v1->v2 added and nothing has ever
written, alongside the ``chunk_engine_version`` stamp; ``Media.chunking_config``
is written by the SAME persist seam in a shape BOTH existing readers
understand:

* ``ChunkingTemplateLibrary.get_documents_using_template`` matches
  ``chunking_config LIKE '%"template": "<name>"%'`` (default ``json.dumps``
  separators -- ``": "`` -- are load-bearing; a compact separator would break
  the LIKE while satisfying ``json_extract``);
* ``ChunkingTemplateLibrary.get_template_statistics`` groups by
  ``json_extract(chunking_config, '$.template')``.

The tests prove the round-trip against the real readers (not just raw SQL)
plus the raw LIKE/json_extract spelling, and pin that the no-template path
writes NOTHING (byte-identical to today).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    parse_local_file_for_ingest,
    persist_parsed_media,
)

TEMPLATE_TINY: Dict[str, Any] = {
    "name": "tiny-words",
    "chunking": {"method": "words", "config": {"max_size": 3, "overlap": 0}},
}

_WORDS = [f"w{i:02d}" for i in range(1, 25)]
_FIXTURE_TEXT = " ".join(_WORDS)


@pytest.fixture()
def media_db(tmp_path: Path) -> MediaDatabase:
    db = MediaDatabase(tmp_path / "media.db", client_id="test-template-parity")
    yield db
    db.close_connection()


@pytest.fixture(autouse=True)
def _no_config_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Determinism: no test picks up the developer machine's config."""
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda section, key=None, default=None: (
            default if default is not None else None
        ),
    )


def _ingest(
    db: MediaDatabase, source: Path, chunk_options: Dict[str, Any] | None
) -> int:
    payload = parse_local_file_for_ingest(
        source, {"chunk_options": chunk_options, "perform_analysis": False}
    )
    media_id, _, _ = persist_parsed_media(
        payload, db, overwrite_existing=True, generate_embeddings=False
    )
    assert media_id is not None
    return media_id


def _chunk_column_rows(
    db: MediaDatabase, media_id: int, column: str
) -> list[Any]:
    cursor = db.execute_query(
        f"SELECT {column} FROM UnvectorizedMediaChunks "
        "WHERE media_id = ? AND deleted = 0 ORDER BY chunk_index",
        (media_id,),
    )
    return [row[column] for row in cursor.fetchall()]


def _media_chunking_config(db: MediaDatabase, media_id: int) -> str | None:
    cursor = db.execute_query(
        "SELECT chunking_config FROM Media WHERE id = ?", (media_id,)
    )
    rows = cursor.fetchall()
    assert rows, "media row vanished"
    return rows[0]["chunking_config"]


# ---------------------------------------------------------------------------
# AC 38: the columns + both-reader-compatible chunking_config
# ---------------------------------------------------------------------------


def test_template_ingest_fills_chunk_columns_and_config(
    media_db: MediaDatabase, tmp_path: Path
) -> None:
    """A template ingest stamps every chunk row and the Media record."""
    source = tmp_path / "fixture.txt"
    source.write_text(_FIXTURE_TEXT, encoding="utf-8")

    media_id = _ingest(media_db, source, {"template": dict(TEMPLATE_TINY)})

    template_cells = _chunk_column_rows(media_db, media_id, "chunking_template")
    assert template_cells, "no chunk rows were persisted"
    assert set(template_cells) == {"tiny-words"}

    params_cells = _chunk_column_rows(media_db, media_id, "chunking_params")
    assert set(params_cells) == {
        json.dumps({"method": "words", "size": 3, "overlap": 0})
    }

    config_json = _media_chunking_config(media_db, media_id)
    assert config_json is not None
    parsed = json.loads(config_json)
    assert parsed["template"] == "tiny-words"


def test_chunking_config_satisfies_like_reader(
    media_db: MediaDatabase, tmp_path: Path
) -> None:
    """Reader 1: ``get_documents_using_template``'s LIKE pattern matches."""
    from tldw_chatbook.Chunking.chunking_interop_library import (
        get_chunking_service,
    )

    source = tmp_path / "fixture.txt"
    source.write_text(_FIXTURE_TEXT, encoding="utf-8")
    media_id = _ingest(media_db, source, {"template": dict(TEMPLATE_TINY)})

    # The reader's own SQL spelling, executed raw: proves the WRITER's JSON
    # separators (": ") match what the LIKE expects.
    cursor = media_db.execute_query(
        "SELECT id FROM Media WHERE chunking_config LIKE ? AND deleted = 0",
        (f'%"template": "{TEMPLATE_TINY["name"]}"%',),
    )
    assert [row["id"] for row in cursor.fetchall()] == [media_id]

    service = get_chunking_service(media_db)
    documents = service.get_documents_using_template("tiny-words")
    assert [doc["id"] for doc in documents] == [media_id]
    assert documents[0]["config"]["template"] == "tiny-words"


def test_chunking_config_satisfies_json_extract_reader(
    media_db: MediaDatabase, tmp_path: Path
) -> None:
    """Reader 2: ``get_template_statistics``' json_extract grouping sees it."""
    from tldw_chatbook.Chunking.chunking_interop_library import (
        get_chunking_service,
    )

    source = tmp_path / "fixture.txt"
    source.write_text(_FIXTURE_TEXT, encoding="utf-8")
    _ingest(media_db, source, {"template": dict(TEMPLATE_TINY)})

    cursor = media_db.execute_query(
        "SELECT json_extract(chunking_config, '$.template') AS t FROM Media "
        "WHERE chunking_config IS NOT NULL AND deleted = 0"
    )
    assert [row["t"] for row in cursor.fetchall()] == ["tiny-words"]

    stats = get_chunking_service(media_db).get_template_statistics()
    assert {"template": "tiny-words", "count": 1} in stats["most_used_templates"]
    assert stats["configured_documents"] == 1


def test_rechunk_resolution_reads_back_stored_choice(
    media_db: MediaDatabase, tmp_path: Path
) -> None:
    """The stored choice is what §9.1's re-chunk order resolves first."""
    from tldw_chatbook.Chunking.chunking_interop_library import (
        get_chunking_service,
    )
    from tldw_chatbook.Chunking.template_runtime import resolve_ingest_template

    get_chunking_service(media_db).create_template(
        name="tiny-words",
        description="persistence round-trip template",
        template_json={
            k: v for k, v in TEMPLATE_TINY.items() if k != "name"
        },
    )
    source = tmp_path / "fixture.txt"
    source.write_text(_FIXTURE_TEXT, encoding="utf-8")
    media_id = _ingest(media_db, source, {"template": dict(TEMPLATE_TINY)})

    stored = _media_chunking_config(media_db, media_id)
    assert stored is not None
    resolved = resolve_ingest_template(media_db, per_media=json.loads(stored)["template"])
    assert resolved is not None
    assert resolved["name"] == "tiny-words"


def test_plain_options_ingest_writes_no_template_persistence(
    media_db: MediaDatabase, tmp_path: Path
) -> None:
    """No template -> columns and chunking_config stay NULL (today's shape)."""
    source = tmp_path / "fixture.txt"
    source.write_text(_FIXTURE_TEXT, encoding="utf-8")

    media_id = _ingest(media_db, source, {"size": 500, "overlap": 100})

    assert set(_chunk_column_rows(media_db, media_id, "chunking_template")) == {None}
    assert set(_chunk_column_rows(media_db, media_id, "chunking_params")) == {None}
    assert _media_chunking_config(media_db, media_id) is None


def test_two_templates_persist_their_own_names(
    media_db: MediaDatabase, tmp_path: Path
) -> None:
    """Two different templates over the same source stamp their own names."""
    source = tmp_path / "fixture.txt"
    source.write_text(_FIXTURE_TEXT, encoding="utf-8")

    first = _ingest(media_db, source, {"template": dict(TEMPLATE_TINY)})
    second_template = dict(TEMPLATE_TINY, name="big-words")
    second_template["chunking"] = {
        "method": "words",
        "config": {"max_size": 12, "overlap": 0},
    }
    second = _ingest(media_db, source, {"template": second_template})

    assert first == second  # overwrite in place: same media row re-stamped
    assert set(_chunk_column_rows(media_db, second, "chunking_template")) == {
        "big-words"
    }
    assert json.loads(_media_chunking_config(media_db, second))["template"] == (
        "big-words"
    )
