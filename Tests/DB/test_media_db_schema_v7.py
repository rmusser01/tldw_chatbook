"""Schema v7 (spec §5, ACs 16-23 + 29): the ChunkingTemplates rebuild.

The v6→v7 migration is a table rebuild: new column set (``uuid``, ``tags``,
``is_builtin``, ``version``, ``deleted``), per-row pipeline→flat conversion,
quarantine of unconvertible rows, and seeding of the six server built-ins —
all inside ONE ADR-030 transaction so a seeded mid-rebuild failure leaves
the DB at v6 untouched.

The historical v6 fixtures come from ``historical_bootstrap_v6`` (patched
``_CURRENT_SCHEMA_VERSION`` + the production chain), never from dropping
tables and stamping a version back (AC 19).
"""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import (
    DatabaseError,
    MediaDatabase,
    SchemaError,
)
from tldw_chatbook.Chunking import template_runtime as tr
from tldw_chatbook.Chunking._template_conversion import (
    DEFAULT_METHOD,
    QUARANTINE_SUFFIX,
    convert_template_row,
)

from Tests.DB.historical_bootstrap_v6 import media_db_at_v6

SIX_BUILTINS = {
    "academic_paper",
    "code_documentation",
    "chat_conversation",
    "book_chapters",
    "transcript_dialogue",
    "legal_document",
}

OLD_SEEDS = {
    "general",
    "academic_paper",
    "code_documentation",
    "conversational",
    "contextual",
}

V6_COLUMNS = {
    "id",
    "name",
    "description",
    "template_json",
    "is_system",
    "created_at",
    "updated_at",
}

V7_COLUMNS = {
    "id",
    "uuid",
    "name",
    "description",
    "template_json",
    "tags",
    "is_builtin",
    "version",
    "deleted",
    "created_at",
    "updated_at",
}

SEED_TEXT = (
    "# Introduction\n\n"
    + " ".join(f"Sentence {i} carries enough words to be chunked." for i in range(40))
)


def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _row_by_name(conn: sqlite3.Connection, name: str) -> dict:
    row = conn.execute(
        "SELECT * FROM ChunkingTemplates WHERE name = ?", (name,)
    ).fetchone()
    assert row is not None, f"expected a row named {name!r}"
    return dict(row)


def _body(row: dict) -> dict:
    return json.loads(row["template_json"])


def _v6_path_with_rows(tmp_path: Path, extra_rows: list[tuple]) -> Path:
    """Genuine v6 DB (AC 19 bootstrap) plus extra seed rows, then closed.

    ``extra_rows`` are ``(name, description, template_json, is_system)``
    tuples inserted through the historical connection before close.
    """
    path = tmp_path / "v6.db"
    with media_db_at_v6(path) as db:
        conn = db.get_connection()
        if extra_rows:
            conn.executemany(
                "INSERT INTO ChunkingTemplates "
                "(name, description, template_json, is_system) "
                "VALUES (?, ?, ?, ?)",
                extra_rows,
            )
            conn.commit()
    return path


# A custom v6 pipeline row exercising every conversion rule at once:
# {type,params} ops, structural method repair, a mapped op
# (section_detection→extract_sections), dropped ops, and top-level tags.
CUSTOM_PIPELINE = {
    "name": "custom_pipeline",
    "base_method": "words",
    "pipeline": [
        {
            "stage": "preprocess",
            "operations": [
                {"type": "normalize_whitespace", "params": {"max_line_breaks": 2}},
                {
                    "type": "section_detection",
                    "params": {"headers": ["Abstract", "Methods"]},
                },
                {"type": "code_block_detection", "params": {}},
            ],
        },
        {
            "stage": "chunk",
            "method": "structural",
            "options": {"max_size": 500, "overlap": 50},
        },
        {
            "stage": "postprocess",
            "operations": [
                {"type": "add_context", "params": {"context_size": 2}},
            ],
        },
    ],
    "tags": ["custom", "pipeline"],
    "metadata": {"origin": "user"},
}

UNPARSEABLE_JSON = '{"name": "garbage_json", "pipeline": '  # trailing comma-json
UNCONVERTIBLE_BODY = json.dumps({"name": "garbage_shape", "description": "no method"})

V6_EXTRA_ROWS = [
    ("custom_pipeline", "user-made", json.dumps(CUSTOM_PIPELINE), 0),
    ("garbage_json", "broken json", UNPARSEABLE_JSON, 0),
    ("garbage_shape", "no method", UNCONVERTIBLE_BODY, 0),
]


# ---------------------------------------------------------------------------
# AC 29 (+ 16): fresh install
# ---------------------------------------------------------------------------


def test_fresh_install_lands_with_six_builtins_and_no_old_seed_bodies(tmp_path):
    db = MediaDatabase(str(tmp_path / "fresh.db"), client_id="test")
    conn = db.get_connection()
    version = conn.execute("SELECT version FROM schema_version").fetchone()["version"]
    assert version == MediaDatabase._CURRENT_SCHEMA_VERSION

    builtins = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM ChunkingTemplates WHERE is_builtin = 1"
        )
    }
    assert builtins == SIX_BUILTINS

    # None of the five old seeds survives in its seed role: no row anywhere
    # still carries the v6 pipeline shape, and the two old seed names that
    # the six re-cover now hold the six's flat bodies.
    for row in conn.execute("SELECT name, template_json FROM ChunkingTemplates"):
        body = json.loads(row["template_json"])
        assert "pipeline" not in body, row["name"]
        assert "base_method" not in body, row["name"]
    academic = _row_by_name(conn, "academic_paper")
    assert academic["is_builtin"] == 1
    assert _body(academic)["chunking"]["method"] == "sentences"

    # The three old seeds the six do not cover survive as converted,
    # non-builtin rows (nothing a user could have selected disappears).
    for name in ("general", "conversational", "contextual"):
        row = _row_by_name(conn, name)
        assert row["is_builtin"] == 0, name
    db.close_connection()


# ---------------------------------------------------------------------------
# AC 16: target DDL
# ---------------------------------------------------------------------------


def test_v7_column_shape_and_constraints(tmp_path):
    db = MediaDatabase(str(tmp_path / "shape.db"), client_id="test")
    conn = db.get_connection()

    info = {
        row["name"]: row
        for row in conn.execute("PRAGMA table_info(ChunkingTemplates)")
    }
    assert set(info) == V7_COLUMNS
    assert V6_COLUMNS - {"id", "name", "description", "template_json",
                         "created_at", "updated_at"} == {"is_system"}
    assert "is_system" not in info
    assert info["uuid"]["notnull"] == 1
    assert info["name"]["notnull"] == 1
    assert info["template_json"]["notnull"] == 1
    assert info["is_builtin"]["notnull"] == 1 and info["is_builtin"]["dflt_value"] == "0"
    assert info["version"]["notnull"] == 1 and info["version"]["dflt_value"] == "1"
    assert info["deleted"]["notnull"] == 1 and info["deleted"]["dflt_value"] == "0"

    # uuid column-level UNIQUE (autoindex from the table constraint)
    indexes = conn.execute("PRAGMA index_list(ChunkingTemplates)").fetchall()
    assert any(
        row["unique"] and row["origin"] == "u"
        for row in indexes
        if row["name"].startswith("sqlite_autoindex")
    )
    # the supporting indexes from §5.2
    index_names = {row["name"] for row in indexes}
    assert {"idx_chunking_templates_is_builtin", "idx_chunking_templates_deleted"} <= index_names

    # the partial unique live-name index, by behavior AND by shape
    live = next(
        (row for row in indexes if row["name"] == "idx_chunking_templates_name_live"),
        None,
    )
    assert live is not None
    assert live["unique"] == 1
    assert live["partial"] == 1
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'idx_chunking_templates_name_live'"
    ).fetchone()["sql"]
    assert "WHERE deleted = 0" in sql

    conn.execute(
        "INSERT INTO ChunkingTemplates (uuid, name, template_json) "
        "VALUES ('11111111-1111-1111-1111-111111111111', 'dup-name', '{}')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ChunkingTemplates (uuid, name, template_json) "
            "VALUES ('22222222-2222-2222-2222-222222222222', 'dup-name', '{}')"
        )
    # a soft-deleted row frees the name for a re-add
    conn.execute("UPDATE ChunkingTemplates SET deleted = 1 WHERE name = 'dup-name'")
    conn.execute(
        "INSERT INTO ChunkingTemplates (uuid, name, template_json) "
        "VALUES ('33333333-3333-3333-3333-333333333333', 'dup-name', '{}')"
    )
    db.close_connection()


def test_no_foreign_keys_reference_chunking_templates(tmp_path):
    # §5.3: foreign keys are ON and no table references ChunkingTemplates —
    # asserted, not trusted (the DROP inside the rebuild depends on it).
    db = MediaDatabase(str(tmp_path / "fk.db"), client_id="test")
    conn = db.get_connection()
    tables = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    ]
    offenders = []
    for table in tables:
        assert '"' not in table
        for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            if fk["table"] == "ChunkingTemplates":
                offenders.append(table)
    assert offenders == []
    db.close_connection()


def test_fk_guard_validates_sqlite_master_names(tmp_path):
    # Qodo on PR #1938: the guard builds ``PRAGMA foreign_key_list({name})``
    # from sqlite_master rows. A name the central ``sql_validation`` module
    # rejects must fail LOUD — this guard protects a DROP, so a table it
    # silently skipped would be a table it never actually checked.
    db = MediaDatabase(str(tmp_path / "fk-guard.db"), client_id="test")
    conn = db.get_connection()
    # sanity: every real table name passes validation and the guard is green
    MediaDatabase._assert_no_foreign_keys_reference(conn, "ChunkingTemplates")

    # a hostile sqlite_master name (here: quote-escaped identifier) is
    # rejected by sql_validation instead of being skipped
    conn.execute('CREATE TABLE "bad""name" (id INTEGER PRIMARY KEY)')
    with pytest.raises(SchemaError):
        MediaDatabase._assert_no_foreign_keys_reference(conn, "ChunkingTemplates")
    db.close_connection()


# ---------------------------------------------------------------------------
# AC 19: the historical fixture is genuine
# ---------------------------------------------------------------------------


def test_historical_v6_fixture_is_genuine(tmp_path):
    path = tmp_path / "v6.db"
    with media_db_at_v6(path) as db:
        conn = db.get_connection()
        assert conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()["version"] == 6
        assert _table_columns(conn, "ChunkingTemplates") == V6_COLUMNS
        seeds = conn.execute(
            "SELECT name, is_system FROM ChunkingTemplates"
        ).fetchall()
        assert {row["name"] for row in seeds} == OLD_SEEDS
        assert all(row["is_system"] == 1 for row in seeds)
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'update_chunking_templates_timestamp'"
        ).fetchone() is not None


# ---------------------------------------------------------------------------
# AC 17/18: genuine v6 upgrade — trigger survives; atomicity
# ---------------------------------------------------------------------------


def _upgraded_v6(tmp_path: Path) -> MediaDatabase:
    path = _v6_path_with_rows(tmp_path, V6_EXTRA_ROWS)
    return MediaDatabase(str(path), client_id="upgrade")


def test_genuine_v6_upgrades_to_v7(tmp_path):
    db = _upgraded_v6(tmp_path)
    conn = db.get_connection()
    assert conn.execute(
        "SELECT version FROM schema_version"
    ).fetchone()["version"] == MediaDatabase._CURRENT_SCHEMA_VERSION
    assert _table_columns(conn, "ChunkingTemplates") == V7_COLUMNS
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'ChunkingTemplates_v7'"
    ).fetchone() is None
    names = {
        row["name"]
        for row in conn.execute("SELECT name FROM ChunkingTemplates")
    }
    assert SIX_BUILTINS <= names
    assert {"general", "conversational", "contextual", "custom_pipeline"} <= names
    db.close_connection()


def test_update_timestamp_trigger_survives_rebuild(tmp_path):
    db = _upgraded_v6(tmp_path)
    conn = db.get_connection()
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
        "AND tbl_name = 'ChunkingTemplates' "
        "AND name = 'update_chunking_templates_timestamp'"
    ).fetchone() is not None

    before = _row_by_name(conn, "general")["updated_at"]
    time.sleep(1.1)  # CURRENT_TIMESTAMP has 1-second resolution
    conn.execute(
        "UPDATE ChunkingTemplates SET description = 'touched' WHERE name = 'general'"
    )
    after = _row_by_name(conn, "general")["updated_at"]
    assert after > before, "updated_at froze — the rebuild dropped the trigger"
    db.close_connection()


def test_seeded_mid_rebuild_failure_leaves_v6_intact(tmp_path, monkeypatch):
    # AC 18: DDL + conversion + seeding + version bump in ONE transaction.
    # A conversion that raises after rows have already been inserted into
    # the rebuild table must leave the DB at v6 with the original rows.
    import tldw_chatbook.Chunking._template_conversion as conv

    original = conv.convert_template_row
    calls = {"count": 0}

    def flaky(row):
        calls["count"] += 1
        if calls["count"] >= 3:  # general, conversational already copied
            raise RuntimeError("seeded mid-rebuild failure")
        return original(row)

    monkeypatch.setattr(conv, "convert_template_row", flaky)

    path = _v6_path_with_rows(tmp_path, V6_EXTRA_ROWS)
    with pytest.raises(DatabaseError):
        MediaDatabase(str(path), client_id="boom")

    check = sqlite3.connect(str(path))
    check.row_factory = sqlite3.Row
    try:
        assert check.execute(
            "SELECT version FROM schema_version"
        ).fetchone()[0] == 6
        # the original v6 table and every original row survive
        assert _table_columns(check, "ChunkingTemplates") == V6_COLUMNS
        names = {
            row["name"]
            for row in check.execute("SELECT name FROM ChunkingTemplates")
        }
        assert OLD_SEEDS <= names
        assert {"custom_pipeline", "garbage_json", "garbage_shape"} <= names
        # nothing from the rebuild survived — not even the temp table
        assert check.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'ChunkingTemplates_v7'"
        ).fetchone() is None
        # and the v6 trigger is still in place on the untouched table
        assert check.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'update_chunking_templates_timestamp'"
        ).fetchone() is not None
    finally:
        check.close()


# ---------------------------------------------------------------------------
# AC 20: conversion precedence
# ---------------------------------------------------------------------------


def test_conversion_precedence(tmp_path):
    db = _upgraded_v6(tmp_path)
    conn = db.get_connection()

    # is_system=1 rows covered by the six: dropped and re-seeded with the
    # server's flat bodies (the old academic_paper seed was structural).
    academic = _row_by_name(conn, "academic_paper")
    assert academic["is_builtin"] == 1
    assert _body(academic)["chunking"] == {
        "method": "sentences",
        "config": {"max_size": 5, "overlap": 1},
    }
    code_doc = _row_by_name(conn, "code_documentation")
    assert code_doc["is_builtin"] == 1
    assert _body(code_doc)["chunking"]["method"] == "structure_aware"

    # every is_builtin row is one of the six, nothing more
    assert {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM ChunkingTemplates WHERE is_builtin = 1"
        )
    } == SIX_BUILTINS

    # the three old seeds the six do not cover: converted + kept non-builtin
    general = _row_by_name(conn, "general")
    assert general["is_builtin"] == 0
    assert general["deleted"] == 0
    assert _body(general)["chunking"] == {
        "method": "words",
        "config": {"max_size": 400, "overlap": 100},
    }

    conversational = _row_by_name(conn, "conversational")
    assert conversational["is_builtin"] == 0
    assert _body(conversational)["chunking"]["method"] == "sentences"

    # contextual: method repaired to sentences, add_context recorded as dropped
    contextual = _row_by_name(conn, "contextual")
    assert contextual["is_builtin"] == 0
    contextual_body = _body(contextual)
    assert contextual_body["chunking"]["method"] == "sentences"
    assert contextual_body["metadata"]["_dropped_operations"] == ["add_context"]

    # a custom row: converted + kept non-builtin
    custom = _row_by_name(conn, "custom_pipeline")
    assert custom["is_builtin"] == 0
    assert custom["deleted"] == 0
    db.close_connection()


def test_custom_row_shadowing_a_builtin_name_is_left_alone(tmp_path):
    # Server idempotent seeding semantics (§5.3): a built-in name that
    # already exists as a custom row is left alone + logged, never
    # overwritten — so only five of the six builtins land.
    shadow = {
        "name": "book_chapters",
        "base_method": "words",
        "pipeline": [
            {"stage": "chunk", "method": "words", "options": {"max_size": 250}},
        ],
    }
    path = _v6_path_with_rows(
        tmp_path, [("book_chapters", "user override", json.dumps(shadow), 0)]
    )
    db = MediaDatabase(str(path), client_id="upgrade")
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT name, is_builtin, template_json FROM ChunkingTemplates "
        "WHERE name = 'book_chapters'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["is_builtin"] == 0
    assert json.loads(rows[0]["template_json"])["chunking"] == {
        "method": "words",
        "config": {"max_size": 250},
    }
    builtins = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM ChunkingTemplates WHERE is_builtin = 1"
        )
    }
    assert builtins == SIX_BUILTINS - {"book_chapters"}
    db.close_connection()


# ---------------------------------------------------------------------------
# AC 21/22: quarantine + dropped operations (integration through the rebuild)
# ---------------------------------------------------------------------------


def test_unconvertible_rows_are_quarantined(tmp_path):
    db = _upgraded_v6(tmp_path)
    conn = db.get_connection()

    for original_name, original_body in (
        ("garbage_json", UNPARSEABLE_JSON),
        ("garbage_shape", json.loads(UNCONVERTIBLE_BODY)),
    ):
        row = _row_by_name(conn, original_name + QUARANTINE_SUFFIX)
        assert row["deleted"] == 1, original_name
        body = _body(row)
        assert body["metadata"]["_unconverted"] == original_body
        # repairable: a chunking block exists so the row can be edited back
        assert body["chunking"]["method"] == DEFAULT_METHOD
        # the original live name is gone from the live set
        assert conn.execute(
            "SELECT 1 FROM ChunkingTemplates WHERE name = ? AND deleted = 0",
            (original_name,),
        ).fetchone() is None

    quarantined = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM ChunkingTemplates WHERE deleted = 1"
        )
    }
    assert quarantined == {
        "garbage_json" + QUARANTINE_SUFFIX,
        "garbage_shape" + QUARANTINE_SUFFIX,
    }
    db.close_connection()


def test_dropped_operations_recorded_and_section_detection_mapped(tmp_path):
    db = _upgraded_v6(tmp_path)
    conn = db.get_connection()
    row = _row_by_name(conn, "custom_pipeline")
    body = _body(row)

    # section_detection maps where the intent matches; params become config
    assert {
        op["operation"] for op in body.get("preprocessing", [])
    } == {"normalize_whitespace", "extract_sections"}
    mapped = next(
        op
        for op in body["preprocessing"]
        if op["operation"] == "extract_sections"
    )
    assert mapped["config"] == {"headers": ["Abstract", "Methods"]}

    # the other three registered-nowhere ops are dropped and recorded
    assert body["metadata"]["_dropped_operations"] == [
        "add_context",
        "code_block_detection",
    ]
    assert "postprocessing" not in body  # only op was dropped

    # structural → structure_aware (method repair)
    assert body["chunking"]["method"] == "structure_aware"

    # tags moved to the column and out of the JSON body
    assert json.loads(row["tags"]) == ["custom", "pipeline"]
    assert "tags" not in body
    assert "tags" not in body["metadata"]
    # user metadata survives
    assert body["metadata"]["origin"] == "user"
    db.close_connection()


# ---------------------------------------------------------------------------
# AC 23: the six built-ins execute against the live engine
# ---------------------------------------------------------------------------


def test_six_builtins_execute_against_live_engine(tmp_path):
    db = MediaDatabase(str(tmp_path / "fresh.db"), client_id="test")
    conn = db.get_connection()
    seeded = {
        row["name"]: row["template_json"]
        for row in conn.execute(
            "SELECT name, template_json FROM ChunkingTemplates WHERE is_builtin = 1"
        )
    }
    assert set(seeded) == SIX_BUILTINS

    for name, template_json in seeded.items():
        # resolve_template is the name→template seam; it must be green
        # against v7 rows (stable columns only).
        record = {"name": name, "template_json": template_json}
        resolved = tr.resolve_template(db, name)
        assert resolved is not None, name
        chunks = tr.apply_template(record, SEED_TEXT)
        assert chunks, name
        for index, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == index
            assert chunk["total_chunks"] == len(chunks)
            assert chunk["word_count"] == len(chunk["text"].split())
            assert chunk["metadata"]["offset_basis"]
    db.close_connection()


def test_method_name_repair_end_to_end(tmp_path):
    # §5.4 through a genuine upgrade: hierarchical → structure_aware with
    # config.hierarchical = true (via a custom row; contextual → sentences is
    # covered by the contextual seed in test_conversion_precedence).
    hierarchical = {
        "name": "hier_custom",
        "base_method": "hierarchical",
        "pipeline": [
            {
                "stage": "chunk",
                "method": "hierarchical",
                "options": {"max_size": 600, "overlap": 150},
            }
        ],
    }
    path = _v6_path_with_rows(
        tmp_path, [("hier_custom", None, json.dumps(hierarchical), 0)]
    )
    db = MediaDatabase(str(path), client_id="upgrade")
    body = _body(_row_by_name(db.get_connection(), "hier_custom"))
    assert body["chunking"]["method"] == "structure_aware"
    assert body["chunking"]["config"]["hierarchical"] is True
    assert body["chunking"]["config"]["max_size"] == 600
    db.close_connection()


# ---------------------------------------------------------------------------
# Pure conversion unit tests (Chunking/_template_conversion.py)
# ---------------------------------------------------------------------------


def _v6_row(body, *, name="row", is_system=0, tags_in_body=None):
    if tags_in_body is not None:
        body = dict(body)
        body["tags"] = tags_in_body
    return {
        "id": 1,
        "name": name,
        "description": "desc",
        "template_json": json.dumps(body) if not isinstance(body, str) else body,
        "is_system": is_system,
        "created_at": "2020-01-01 00:00:00",
        "updated_at": "2020-01-01 00:00:00",
    }


class TestConvertTemplateRow:
    def test_pipeline_to_flat_with_type_params_rewrite(self):
        converted = convert_template_row(
            _v6_row(
                {
                    "base_method": "words",
                    "pipeline": [
                        {
                            "stage": "preprocess",
                            "operations": [
                                {
                                    "type": "normalize_whitespace",
                                    "params": {"max_line_breaks": 2},
                                }
                            ],
                        },
                        {
                            "stage": "chunk",
                            "method": "words",
                            "options": {"max_size": 100, "overlap": 10},
                        },
                    ],
                }
            )
        )
        assert converted["deleted"] is False
        assert converted["is_builtin"] is False
        assert converted["version"] == 1
        body = json.loads(converted["template_json"])
        assert body["preprocessing"] == [
            {"operation": "normalize_whitespace", "config": {"max_line_breaks": 2}}
        ]
        assert body["chunking"] == {
            "method": "words",
            "config": {"max_size": 100, "overlap": 10},
        }
        assert "pipeline" not in body and "base_method" not in body

    def test_chunk_stage_method_wins_over_base_method(self):
        converted = convert_template_row(
            _v6_row(
                {
                    "base_method": "words",
                    "pipeline": [
                        {"stage": "chunk", "method": "sentences", "options": {}}
                    ],
                }
            )
        )
        assert json.loads(converted["template_json"])["chunking"]["method"] == "sentences"

    def test_base_method_used_when_no_chunk_stage(self):
        converted = convert_template_row(
            _v6_row(
                {
                    "base_method": "paragraphs",
                    "pipeline": [
                        {
                            "stage": "postprocess",
                            "operations": [{"type": "filter_empty", "params": {}}],
                        }
                    ],
                }
            )
        )
        body = json.loads(converted["template_json"])
        assert body["chunking"] == {"method": "paragraphs", "config": {}}
        assert body["postprocessing"] == [{"operation": "filter_empty", "config": {}}]

    def test_method_repair_structural_and_hierarchical_and_contextual(self):
        for old_method, expected, extra in (
            ("structural", "structure_aware", None),
            ("hierarchical", "structure_aware", True),
            ("contextual", "sentences", None),
        ):
            converted = convert_template_row(
                _v6_row(
                    {
                        "base_method": old_method,
                        "pipeline": [
                            {"stage": "chunk", "method": old_method, "options": {}}
                        ],
                    }
                )
            )
            chunking = json.loads(converted["template_json"])["chunking"]
            assert chunking["method"] == expected, old_method
            if extra is not None:
                assert chunking["config"]["hierarchical"] is True

    def test_uuid_generated_per_row(self):
        import uuid as uuid_module

        row = _v6_row({"base_method": "words", "pipeline": []})
        first = convert_template_row(row)
        second = convert_template_row(dict(row))
        uuid_module.UUID(first["uuid"])  # raises on non-UUID
        uuid_module.UUID(second["uuid"])
        assert first["uuid"] != second["uuid"]

    def test_unparseable_json_is_quarantined_not_dropped(self):
        converted = convert_template_row(
            _v6_row('{"name": "x", "oops": ', name="badjson")
        )
        assert converted["deleted"] is True
        assert converted["name"] == "badjson" + QUARANTINE_SUFFIX
        body = json.loads(converted["template_json"])
        assert body["metadata"]["_unconverted"] == '{"name": "x", "oops": '
        assert body["chunking"]["method"] == DEFAULT_METHOD

    def test_no_chunk_stage_and_no_base_method_is_quarantined(self):
        converted = convert_template_row(
            _v6_row({"name": "empty", "description": "nothing"}, name="empty")
        )
        assert converted["deleted"] is True
        assert converted["name"] == "empty" + QUARANTINE_SUFFIX
        body = json.loads(converted["template_json"])
        assert body["metadata"]["_unconverted"] == {
            "name": "empty",
            "description": "nothing",
        }

    def test_tags_from_top_level_then_metadata(self):
        converted = convert_template_row(
            _v6_row(
                {
                    "base_method": "words",
                    "pipeline": [],
                    "metadata": {"tags": ["from-metadata"], "keep": 1},
                }
            )
        )
        assert json.loads(converted["tags"]) == ["from-metadata"]
        body = json.loads(converted["template_json"])
        assert "tags" not in body
        assert "tags" not in body["metadata"]
        assert body["metadata"]["keep"] == 1

    def test_no_tags_leaves_column_null(self):
        converted = convert_template_row(
            _v6_row({"base_method": "words", "pipeline": []})
        )
        assert converted["tags"] is None

    def test_is_system_maps_to_is_builtin(self):
        converted = convert_template_row(
            _v6_row({"base_method": "words", "pipeline": []}, is_system=1)
        )
        assert converted["is_builtin"] is True

    def test_timestamps_preserved(self):
        converted = convert_template_row(
            _v6_row({"base_method": "words", "pipeline": []})
        )
        assert converted["created_at"] == "2020-01-01 00:00:00"
        assert converted["updated_at"] == "2020-01-01 00:00:00"

    def test_operation_config_spelling_passes_through(self):
        converted = convert_template_row(
            _v6_row(
                {
                    "base_method": "words",
                    "pipeline": [
                        {
                            "stage": "preprocess",
                            "operations": [
                                {
                                    "operation": "normalize_whitespace",
                                    "config": {"max_line_breaks": 1},
                                }
                            ],
                        },
                        {"stage": "chunk", "method": "words", "options": {}},
                    ],
                }
            )
        )
        body = json.loads(converted["template_json"])
        assert body["preprocessing"] == [
            {"operation": "normalize_whitespace", "config": {"max_line_breaks": 1}}
        ]
