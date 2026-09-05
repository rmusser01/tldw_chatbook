"""Conflict-safe Chunking Lab saves against the canonical Media DB catalog."""

from __future__ import annotations

import copy
import json
import threading
import uuid as uuid_module
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from tldw_chatbook.Chunking.chunking_interop_library import (
    BuiltinTemplateError,
    ChunkingInteropService,
    InvalidTemplateError,
)
from tldw_chatbook.Chunking.lab_preflight import (
    METHOD_DEFAULTS,
    PRE_DEFAULTS,
    PreviewUnsupportedError,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import InputError, MediaDatabase
from tldw_chatbook.RAG_Admin.chunking_lab_service import (
    ExpectedTemplate,
    TemplateSaveConflict,
    save_lab_template,
)


@pytest.fixture
def media_db(tmp_path):
    database = MediaDatabase(str(tmp_path / "chunking-lab-save.db"), client_id="test")
    yield database
    database.close_connection()


def _expected(record: dict) -> ExpectedTemplate:
    return ExpectedTemplate(
        id=record["id"], uuid=record["uuid"], version=record["version"]
    )


def _body(**extra: object) -> dict:
    return {
        "preprocessing": [
            {"operation": "normalize_whitespace", "config": {"max_line_breaks": 2}}
        ],
        "chunking": {
            "method": "words",
            "config": {"max_size": 4, "overlap": 1, "language": "en"},
        },
        "classifier": {"media_types": ["document"]},
        "metadata": {"author": "Lab", "nested": {"kept": True}, **extra},
    }


def test_stale_lab_update_cannot_overwrite_newer_record(media_db):
    service = ChunkingInteropService(media_db)
    body = _body(tags=["embedded"])
    original = copy.deepcopy(body)
    record = save_lab_template(
        service,
        body=body,
        name="Test",
        description="Recipe",
        tags=["lab"],
    )
    expected = _expected(record)
    service.update_template(record["id"], description="Changed elsewhere")

    with pytest.raises(TemplateSaveConflict):
        save_lab_template(
            service,
            body=body,
            name="Test",
            description="Stale",
            tags=["new"],
            expected=expected,
        )

    assert (
        service.get_template_by_id(record["id"])["description"] == "Changed elsewhere"
    )
    assert body == original


def test_create_round_trips_advanced_body_and_canonical_tags_without_mutation(media_db):
    service = ChunkingInteropService(media_db)
    body = _body(tags=["body-tag"])
    original = copy.deepcopy(body)
    original_method_defaults = copy.deepcopy(METHOD_DEFAULTS)
    original_pre_defaults = copy.deepcopy(PRE_DEFAULTS)

    record = save_lab_template(
        service,
        body=body,
        name="Advanced",
        description="Lossless recipe",
        tags=["selected", "local"],
    )

    stored_body = json.loads(record["template_json"])
    assert record["uuid"]
    assert record["version"] == 1
    assert record["tags"] == ["selected", "local"]
    assert stored_body == {
        key: value for key, value in original.items() if key != "metadata"
    } | {
        "metadata": {"author": "Lab", "nested": {"kept": True}},
    }
    assert body == original
    assert METHOD_DEFAULTS == original_method_defaults
    assert PRE_DEFAULTS == original_pre_defaults


def test_lab_update_returns_refreshed_identity_and_version(media_db):
    service = ChunkingInteropService(media_db)
    created = save_lab_template(
        service,
        body=_body(),
        name="Versioned",
        description="First",
        tags=[],
    )

    updated = save_lab_template(
        service,
        body=_body(note="current"),
        name="Versioned renamed",
        description="Second",
        tags=["updated"],
        expected=_expected(created),
    )

    assert updated["id"] == created["id"]
    assert updated["uuid"] == created["uuid"]
    assert updated["version"] == created["version"] + 1
    assert updated["name"] == "Versioned renamed"
    assert updated["description"] == "Second"


@pytest.mark.parametrize("creating", [False, True])
def test_save_acknowledges_own_commit_before_peer_then_next_save_conflicts(
    media_db, monkeypatch, creating
):
    service = ChunkingInteropService(media_db)
    created = (
        None
        if creating
        else save_lab_template(
            service, body=_body(), name="Race", description="Original", tags=[]
        )
    )
    transaction = media_db.transaction
    interleaved = False

    @contextmanager
    def peer_after_commit(*args, **kwargs):
        nonlocal interleaved
        with transaction(*args, **kwargs) as connection:
            yield connection
        if not interleaved:
            interleaved = True
            record = service.get_template_by_name("Race")
            service.update_template(record["id"], description="Peer content")

    monkeypatch.setattr(media_db, "transaction", peer_after_commit)
    acknowledged = save_lab_template(
        service,
        body=_body(),
        name="Race",
        description="Local content",
        tags=[],
        expected=None if creating else _expected(created),
    )
    assert interleaved
    assert acknowledged["description"] == "Local content"
    assert acknowledged["version"] == (1 if creating else 2)
    with pytest.raises(TemplateSaveConflict):
        save_lab_template(
            service,
            body=_body(),
            name="Race",
            description="Local content",
            tags=[],
            expected=_expected(acknowledged),
        )
    assert (
        service.get_template_by_id(acknowledged["id"])["description"] == "Peer content"
    )


def test_expected_uuid_is_part_of_the_atomic_update_identity(media_db):
    service = ChunkingInteropService(media_db)
    body = _body()
    created = save_lab_template(
        service,
        body=body,
        name="Identity",
        description="Original",
        tags=[],
    )
    wrong_identity = ExpectedTemplate(
        id=created["id"], uuid=str(uuid_module.uuid4()), version=created["version"]
    )

    with pytest.raises(TemplateSaveConflict):
        save_lab_template(
            service,
            body=body,
            name="Identity",
            description="Wrong identity",
            tags=[],
            expected=wrong_identity,
        )

    assert service.get_template_by_id(created["id"])["description"] == "Original"


def test_deleted_expected_record_is_reported_as_a_save_conflict(media_db):
    service = ChunkingInteropService(media_db)
    body = _body()
    created = save_lab_template(
        service,
        body=body,
        name="Deleted",
        description="Original",
        tags=[],
    )
    service.delete_template(created["id"])

    with pytest.raises(TemplateSaveConflict, match="deleted"):
        save_lab_template(
            service,
            body=body,
            name="Deleted",
            description="Retained draft",
            tags=[],
            expected=_expected(created),
        )

    raw = (
        media_db.get_connection()
        .execute(
            "SELECT deleted, description FROM ChunkingTemplates WHERE id = ?",
            (created["id"],),
        )
        .fetchone()
    )
    assert raw["deleted"] == 1
    assert raw["description"] == "Original"


def test_unsupported_body_is_refused_before_catalog_write_and_is_unchanged(media_db):
    service = ChunkingInteropService(media_db)
    body = {
        "chunking": {"method": "sentences", "config": {"max_size": 4}},
        "metadata": {"large": "x" * 64},
    }
    original = copy.deepcopy(body)
    before = len(service.get_all_templates())

    with pytest.raises(PreviewUnsupportedError, match="chunking.method"):
        save_lab_template(
            service,
            body=body,
            name="Unsupported",
            description="Must not save",
            tags=[],
        )

    assert len(service.get_all_templates()) == before
    assert body == original


def test_recipe_resource_ceiling_runs_before_catalog_write(media_db):
    service = ChunkingInteropService(media_db)
    body = _body(payload="x" * 2_097_152)
    original = copy.deepcopy(body)
    before = len(service.get_all_templates())

    with pytest.raises(PreviewUnsupportedError, match="Recipe exceeds 2 MiB"):
        save_lab_template(
            service,
            body=body,
            name="Oversized",
            description="Must not save",
            tags=[],
        )

    assert len(service.get_all_templates()) == before
    assert body == original


@pytest.mark.parametrize("name", ["auto", "Auto", " AUTO "])
def test_reserved_auto_spelling_variants_are_refused(media_db, name):
    service = ChunkingInteropService(media_db)
    with pytest.raises(InvalidTemplateError, match="reserved"):
        save_lab_template(
            service,
            body=_body(),
            name=name,
            description="Reserved",
            tags=[],
        )


def test_builtin_is_copied_as_new_but_cannot_be_updated(media_db):
    service = ChunkingInteropService(media_db)
    builtin = next(
        record for record in service.get_all_templates() if record["is_builtin"]
    )
    body = _body()

    copied = save_lab_template(
        service,
        body=body,
        name="Builtin copy",
        description="Detached copy",
        tags=[],
    )
    assert copied["is_builtin"] is False

    with pytest.raises(BuiltinTemplateError):
        save_lab_template(
            service,
            body=body,
            name=builtin["name"],
            description="Cannot overwrite",
            tags=[],
            expected=_expected(builtin),
        )


def test_stored_invalid_row_can_be_repaired_with_expected_identity(media_db):
    service = ChunkingInteropService(media_db)
    with media_db.transaction() as connection:
        cursor = connection.execute(
            "INSERT INTO ChunkingTemplates "
            "(uuid, name, description, template_json, tags, is_builtin, version, deleted) "
            "VALUES (?, ?, ?, ?, NULL, 0, 1, 0)",
            (
                str(uuid_module.uuid4()),
                "Repair me",
                "Stored invalid",
                json.dumps({"chunking": {"method": "not-real"}}),
            ),
        )
    stored = service.get_template_by_id(cursor.lastrowid)

    repaired = save_lab_template(
        service,
        body=_body(),
        name="Repair me",
        description="Repaired",
        tags=["fixed"],
        expected=_expected(stored),
    )

    assert repaired["version"] == 2
    assert repaired["description"] == "Repaired"
    assert repaired["tags"] == ["fixed"]


def test_concurrent_same_name_creates_have_one_winner(media_db):
    service = ChunkingInteropService(media_db)
    start = threading.Barrier(2)

    def create() -> dict:
        start.wait()
        return save_lab_template(
            service,
            body=_body(),
            name="One live name",
            description="Concurrent",
            tags=[],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [pool.submit(create) for _ in range(2)]
        successes = []
        failures = []
        for outcome in outcomes:
            try:
                successes.append(outcome.result())
            except Exception as exc:  # noqa: BLE001 - classify the public result.
                failures.append(exc)

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], InputError)
    assert [record["name"] for record in service.get_all_templates()].count(
        "One live name"
    ) == 1


def test_expected_uuid_and_version_must_be_supplied_together(media_db):
    service = ChunkingInteropService(media_db)
    template_id = service.create_template("Pair", "Original", _body())
    record = service.get_template_by_id(template_id)

    with pytest.raises(InputError, match="together"):
        service.update_template(
            template_id,
            description="No update",
            expected_uuid=record["uuid"],
        )

    assert service.get_template_by_id(template_id)["description"] == "Original"
