"""ChunkingInteropService against Media DB schema v7 (task-8, ACs 24-27).

The CRUD rewrite that ships with the v6→v7 migration (spec §5.2.1): storage
and its only reader are atomic. Pinned here:

- no ``is_system`` anywhere in the production tree (AC 26 grep pin) — the
  only justified survivors are the v1→v2 migration DDL/seeds, the v6→v7 row
  converter's input mapping (both historical chain steps, task-7), and the
  unrelated OpenAI ``system_fingerprint`` helper;
- every read filters ``deleted = 0`` (AC 25/26);
- every write supplies a fresh ``uuid4`` and writes ``tags`` as the JSON
  column, through ``MediaDatabase.transaction()`` (AC 26);
- create/update refuse invalid bodies via the Task-6 server-parity
  validator with a NAMED error, while stored-invalid rows stay editable
  (update validates the NEW body only — AC 24);
- soft delete end to end: the row leaves listings and lookups, the name is
  reusable (partial unique index), and ``version`` increments on update
  (AC 25);
- duplicate/statistics surfaces read v7 columns.
"""

import json
import re
import uuid as uuid_module
from contextlib import contextmanager
from pathlib import Path

import pytest

from tldw_chatbook.Chunking.chunking_interop_library import (
    ChunkingInteropService,
    InvalidTemplateError,
    TemplateNotFoundError,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

PKG_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"

# The vendored engine tree mirrors upstream (which has its own template
# surfaces); `_shims` is ported-test scaffolding.
_VENDORED_PREFIXES = (("Chunking", "engine"), ("Chunking", "_shims"))

# `\bis_system\b` survivors, each with its reason. The v7 chunking column
# is `is_builtin`; `is_system` is only legitimate where HISTORY requires
# it (pinned by Tests/DB/test_media_db_schema_v7.py) or where the token
# merely collides with an unrelated concept.
_JUSTIFIED_IS_SYSTEM_FILES = {
    # v1→v2 migration DDL + the five original seeds, replayed verbatim by
    # the historical bootstrap (AC 19): the chain must keep creating the
    # v6 shape so v6→v7 stays testable end to end.
    "DB/Client_Media_DB_v2.py",
    # v6→v7 row conversion: the mechanical ``is_builtin ← is_system``
    # input mapping (spec §5.4).
    "Chunking/_template_conversion.py",
    # OpenAI response ``system_fingerprint`` helper — unrelated to
    # chunking templates; the token merely collides.
    "Prompt_Management/prompt_variables.py",
}

# `include_system` is chunking-template vocabulary (the pre-rewrite
# ``get_all_templates(include_system=...)`` switch) only inside these
# packages. Elsewhere it is the Prompts domain's unrelated pre-existing
# "include system prompts in this listing" flag (DB/Prompts_DB.py,
# tldw_api/client.py, the console prompts UI, voice-assistant interop).
_INCLUDE_SYSTEM_SCOPES = ("Chunking", "RAG_Admin")

VALID_BODY = {
    "name": "whatever",
    "chunking": {"method": "words", "config": {"max_size": 50, "overlap": 10}},
}
# Invalid per the Task-6 validator: unknown chunking method (the live
# registry rejects it), so a refused create proves the validator ran.
INVALID_BODY = {"chunking": {"method": "no_such_method", "config": {}}}


@pytest.fixture
def db(tmp_path):
    database = MediaDatabase(str(tmp_path / "crud.db"), client_id="test")
    yield database
    database.close_connection()


@pytest.fixture
def svc(db):
    return ChunkingInteropService(db)


def _raw_row(db, template_id):
    return db.get_connection().execute(
        "SELECT * FROM ChunkingTemplates WHERE id = ?", (template_id,)
    ).fetchone()


def _insert_stored_invalid(db, name="broken"):
    """Simulate a row that predates validate-on-write (or was minted by a
    conversion the validator would refuse): invalid body, live row."""
    with db.transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO ChunkingTemplates "
            "(uuid, name, description, template_json, tags, is_builtin, "
            "version, deleted) VALUES (?, ?, ?, ?, NULL, 0, 1, 0)",
            (
                str(uuid_module.uuid4()),
                name,
                "stored-invalid fixture",
                json.dumps(INVALID_BODY),
            ),
        )
        return cursor.lastrowid


@contextmanager
def _transaction_spy(db):
    """Record every ``db.transaction()`` entry and whether the connection
    was actually inside a transaction while the body ran."""
    original = db.transaction
    seen = []

    @contextmanager
    def spy():
        with original() as conn:
            seen.append(conn.in_transaction)
            yield conn

    db.transaction = spy
    try:
        yield seen
    finally:
        del db.transaction  # restore the bound method


# ---------------------------------------------------------------------------
# AC 26 — the grep pin: no is_system anywhere it is not justified
# ---------------------------------------------------------------------------


class TestNoIsSystemAnywhere:
    def test_no_is_system_in_production_tree(self):
        offenders = []
        for path in sorted(PKG_ROOT.rglob("*.py")):
            rel = path.relative_to(PKG_ROOT)
            if rel.parts[:2] in _VENDORED_PREFIXES:
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            posix = rel.as_posix()
            if re.search(r"\bis_system\b", source) and posix not in _JUSTIFIED_IS_SYSTEM_FILES:
                offenders.append(posix)
            elif (
                re.search(r"\binclude_system\b", source)
                and rel.parts[0] in _INCLUDE_SYSTEM_SCOPES
            ):
                offenders.append(posix)
        assert offenders == [], (
            f"chunking-template `is_system`/`include_system` survived outside "
            f"the justified historical files: {offenders}. The v7 column is "
            f"`is_builtin` (AC 26)."
        )


# ---------------------------------------------------------------------------
# AC 25/26 — reads filter deleted = 0 and speak v7 columns
# ---------------------------------------------------------------------------


class TestV7Reads:
    def test_get_all_templates_returns_v7_columns(self, svc):
        record = svc.get_all_templates()[0]
        for key in (
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
        ):
            assert key in record, f"missing v7 column {key}"
        assert "is_system" not in record
        assert record["deleted"] is False

    def test_get_all_templates_filters_deleted(self, db, svc):
        template_id = svc.create_template(
            name="doomed", description="d", template_json=VALID_BODY
        )
        db.get_connection().execute(
            "UPDATE ChunkingTemplates SET deleted = 1 WHERE id = ?", (template_id,)
        )
        db.get_connection().commit()

        assert all(t["name"] != "doomed" for t in svc.get_all_templates())

    def test_lookups_by_name_and_id_filter_deleted(self, db, svc):
        template_id = svc.create_template(
            name="vanished", description="d", template_json=VALID_BODY
        )
        db.get_connection().execute(
            "UPDATE ChunkingTemplates SET deleted = 1 WHERE id = ?", (template_id,)
        )
        db.get_connection().commit()

        assert svc.get_template_by_name("vanished") is None
        with pytest.raises(TemplateNotFoundError):
            svc.get_template_by_id(template_id)

    def test_builtin_flag_orders_and_filters(self, svc):
        svc.create_template(name="zz_custom", description="d", template_json=VALID_BODY)
        names = [t["name"] for t in svc.get_all_templates()]
        # Builtins (the six seeds) sort ahead of custom rows.
        assert names.index("academic_paper") < names.index("zz_custom")
        custom_only = [t["name"] for t in svc.get_all_templates(include_builtin=False)]
        assert "academic_paper" not in custom_only
        assert "zz_custom" in custom_only


# ---------------------------------------------------------------------------
# AC 24/26 — create: uuid supplied, tags column, validate-on-write
# ---------------------------------------------------------------------------


class TestCreate:
    def test_create_supplies_uuid_and_column_defaults(self, db, svc):
        template_id = svc.create_template(
            name="sourced", description="d", template_json=VALID_BODY
        )
        row = _raw_row(db, template_id)

        # uuid supplied by the write (NOT NULL UNIQUE would refuse None).
        uuid_module.UUID(row["uuid"])  # parses
        assert row["is_builtin"] == 0
        assert row["version"] == 1
        assert row["deleted"] == 0
        record = svc.get_template_by_id(template_id)
        assert record["uuid"] == row["uuid"]
        assert record["tags"] == []

    def test_uuids_are_unique_per_row(self, svc):
        first = svc.create_template("a", "d", VALID_BODY)
        second = svc.create_template("b", "d", VALID_BODY)
        assert svc.get_template_by_id(first)["uuid"] != svc.get_template_by_id(second)[
            "uuid"
        ]

    def test_create_writes_tags_as_json_column(self, db, svc):
        template_id = svc.create_template(
            name="tagged", description="d", template_json=VALID_BODY, tags=["x", "y"]
        )
        row = _raw_row(db, template_id)
        assert json.loads(row["tags"]) == ["x", "y"]
        assert svc.get_template_by_id(template_id)["tags"] == ["x", "y"]

    def test_create_moves_body_tags_into_the_column(self, db, svc):
        body = {**VALID_BODY, "tags": ["from-body"]}
        template_id = svc.create_template(name="bodytags", description="d", template_json=body)
        row = _raw_row(db, template_id)
        assert json.loads(row["tags"]) == ["from-body"]
        assert "tags" not in json.loads(row["template_json"])

    def test_create_refuses_invalid_template_with_named_error(self, db, svc):
        before = db.get_connection().execute(
            "SELECT COUNT(*) AS n FROM ChunkingTemplates"
        ).fetchone()["n"]

        with pytest.raises(InvalidTemplateError, match="valid"):
            svc.create_template(name="bad", description="d", template_json=INVALID_BODY)

        after = db.get_connection().execute(
            "SELECT COUNT(*) AS n FROM ChunkingTemplates"
        ).fetchone()["n"]
        assert after == before  # refused, not written

    def test_create_refuses_non_json_body(self, svc):
        with pytest.raises(Exception):
            svc.create_template(name="bad", description="d", template_json="{not json")

    def test_create_rejects_duplicate_live_name(self, svc):
        svc.create_template(name="dup", description="d", template_json=VALID_BODY)
        with pytest.raises(Exception, match="already exists"):
            svc.create_template(name="dup", description="d", template_json=VALID_BODY)

    def test_create_writes_via_transaction(self, db, svc):
        with _transaction_spy(db) as seen:
            svc.create_template(name="txn", description="d", template_json=VALID_BODY)
        assert seen and all(inside for inside in seen), (
            f"create must write inside media_db.transaction() (observed: {seen})"
        )


# ---------------------------------------------------------------------------
# AC 24/25 — update: validate the NEW body only; version increments
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_increments_version(self, svc):
        template_id = svc.create_template("v", "d", VALID_BODY)
        assert svc.get_template_by_id(template_id)["version"] == 1

        svc.update_template(template_id, description="d2")
        record = svc.get_template_by_id(template_id)
        assert record["version"] == 2
        assert record["description"] == "d2"

        svc.update_template(template_id, template_json=VALID_BODY)
        assert svc.get_template_by_id(template_id)["version"] == 3

    def test_stored_invalid_row_stays_editable(self, db, svc):
        template_id = _insert_stored_invalid(db)

        # A VALID new body repairs the row (the old body is never consulted).
        svc.update_template(
            template_id,
            template_json={
                "chunking": {"method": "sentences", "config": {"max_size": 8}}
            },
        )
        assert svc.get_template_by_id(template_id)["template_json"]

        # An INVALID new body is refused — validation gates the new body only
        # but still gates it.
        with pytest.raises(InvalidTemplateError):
            svc.update_template(template_id, template_json=INVALID_BODY)

    def test_update_refuses_invalid_new_body(self, svc):
        template_id = svc.create_template("u", "d", VALID_BODY)
        with pytest.raises(InvalidTemplateError):
            svc.update_template(template_id, template_json=INVALID_BODY)

    def test_update_tags_column(self, db, svc):
        template_id = svc.create_template("t", "d", VALID_BODY, tags=["old"])
        svc.update_template(template_id, tags=["new"])
        row = _raw_row(db, template_id)
        assert json.loads(row["tags"]) == ["new"]

    def test_update_rejects_name_collision(self, svc):
        first = svc.create_template("first", "d", VALID_BODY)
        second = svc.create_template("second", "d", VALID_BODY)
        with pytest.raises(Exception, match="already exists"):
            svc.update_template(second, name="first")
        # Keeping one's own name is not a collision.
        svc.update_template(second, name="second")
        assert svc.get_template_by_id(first)["name"] == "first"

    def test_update_writes_via_transaction(self, db, svc):
        template_id = svc.create_template("ut", "d", VALID_BODY)
        with _transaction_spy(db) as seen:
            svc.update_template(template_id, description="d2")
        assert seen and all(inside for inside in seen)


# ---------------------------------------------------------------------------
# AC 25 — soft delete end to end
# ---------------------------------------------------------------------------


class TestSoftDelete:
    def test_soft_delete_end_to_end(self, db, svc):
        template_id = svc.create_template("gone", "d", VALID_BODY)

        svc.delete_template(template_id)

        # Leaves every listing and lookup...
        assert all(t["name"] != "gone" for t in svc.get_all_templates())
        assert svc.get_template_by_name("gone") is None
        with pytest.raises(TemplateNotFoundError):
            svc.get_template_by_id(template_id)

        # ...but the row survives, soft-deleted.
        row = _raw_row(db, template_id)
        assert row is not None and row["deleted"] == 1

        # The partial unique index frees the name for a re-add.
        reborn = svc.create_template("gone", "reborn", VALID_BODY)
        assert reborn != template_id
        assert svc.get_template_by_name("gone")["id"] == reborn

    def test_delete_writes_via_transaction(self, db, svc):
        template_id = svc.create_template("dt", "d", VALID_BODY)
        with _transaction_spy(db) as seen:
            svc.delete_template(template_id)
        assert seen and all(inside for inside in seen)


# ---------------------------------------------------------------------------
# v7 columns through the remaining surfaces
# ---------------------------------------------------------------------------


class TestDuplicateAndStatistics:
    def test_duplicate_mints_new_uuid(self, db, svc):
        template_id = svc.create_template("orig", "d", VALID_BODY, tags=["t"])
        duplicate_id = svc.duplicate_template(template_id, "orig-copy")

        original = svc.get_template_by_id(template_id)
        duplicate = svc.get_template_by_id(duplicate_id)
        assert duplicate["uuid"] != original["uuid"]
        assert duplicate["is_builtin"] is False
        assert duplicate["tags"] == ["t"]
        assert json.loads(duplicate["template_json"])["chunking"] == json.loads(
            original["template_json"]
        )["chunking"]

    def test_statistics_speak_v7_columns_and_exclude_deleted(self, db, svc):
        # A fresh v7 DB carries the six built-ins plus the three old seeds
        # the six do not re-cover, converted to non-builtin rows (task-7).
        stats = svc.get_template_statistics()
        assert stats["total_templates"] == 9
        assert stats["builtin_templates"] == 6
        assert stats["custom_templates"] == 3
        assert "system_templates" not in stats

        template_id = svc.create_template("stat", "d", VALID_BODY)
        stats = svc.get_template_statistics()
        assert stats["custom_templates"] == 4

        svc.delete_template(template_id)
        stats = svc.get_template_statistics()
        assert stats["total_templates"] == 9
        assert stats["custom_templates"] == 3
