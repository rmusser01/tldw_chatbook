"""Atomic Prompt/Recipe batch delete and restore integration coverage."""

from __future__ import annotations

import inspect
import sqlite3

import pytest
from loguru import logger

from tldw_chatbook.DB import Prompts_DB as prompts_db_module
from tldw_chatbook.DB.Prompts_DB import (
    DatabaseError,
    ExpectedVersionConflictError,
    PromptsDatabase,
)
from tldw_chatbook.Prompt_Management.prompt_batch_models import PromptBatchTarget
from tldw_chatbook.Prompt_Management.prompt_scope_service import LocalPromptService


@pytest.fixture()
def db(tmp_path) -> PromptsDatabase:
    """Use a real file-backed SQLite database for every mutation test."""
    database = PromptsDatabase(tmp_path / "batch-prompts.db", client_id="batch-test")
    try:
        yield database
    finally:
        database.close_connection()


def _add_artifact(
    db: PromptsDatabase,
    *,
    name: str,
    artifact_type: str = "prompt",
    keywords: list[str] | None = None,
    body: str = "unique body",
) -> tuple[int, str]:
    local_id, artifact_uuid, _message = db.add_prompt(
        name=name,
        author="Batch Author",
        details=body,
        system_prompt=f"system {body}",
        user_prompt=f"user {body}",
        keywords=keywords,
        artifact_type=artifact_type,
    )
    assert local_id is not None
    assert artifact_uuid is not None
    return local_id, artifact_uuid


def _row_state(db: PromptsDatabase, local_id: int) -> tuple[int, int]:
    row = (
        db.get_connection()
        .execute("SELECT deleted, version FROM Prompts WHERE id = ?", (local_id,))
        .fetchone()
    )
    assert row is not None
    return int(row["deleted"]), int(row["version"])


def _set_next_prompt_id(db: PromptsDatabase, next_id: int) -> None:
    db.get_connection().execute(
        "INSERT INTO sqlite_sequence(name, seq) VALUES ('Prompts', ?)",
        (next_id - 1,),
    )
    db.get_connection().commit()


def _assert_no_active_batch_rows(
    db: PromptsDatabase, local_ids: tuple[int, ...]
) -> None:
    assert all(_row_state(db, local_id)[0] == 1 for local_id in local_ids)


def _assert_all_active_batch_rows(
    db: PromptsDatabase, local_ids: tuple[int, ...]
) -> None:
    assert all(_row_state(db, local_id)[0] == 0 for local_id in local_ids)


def test_batch_delete_and_restore_preserve_exact_database_state(db):
    prompt_id, _ = _add_artifact(db, name="Batch Prompt", keywords=["Alpha", "Shared"])
    recipe_id, _ = _add_artifact(
        db,
        name="Batch Recipe",
        artifact_type="recipe",
        keywords=["Beta", "Shared"],
    )
    collection_service = LocalPromptService(db)
    collection_id = collection_service.create_prompt_collection(
        {"name": "Batch Collection", "prompt_ids": [prompt_id, recipe_id]}
    )["collection_id"]
    baseline_change_id = (
        db.get_connection().execute("SELECT MAX(change_id) FROM sync_log").fetchone()[0]
    )

    result = db.soft_delete_prompts(
        (PromptBatchTarget(prompt_id, 1), PromptBatchTarget(recipe_id, 1))
    )

    assert [entry.local_id for entry in result.entries] == sorted(
        (prompt_id, recipe_id)
    )
    assert [entry.artifact_type for entry in result.entries] == ["prompt", "recipe"]
    assert db.search_prompts("unique body")[0] == []
    assert _row_state(db, prompt_id) == (1, 2)
    assert _row_state(db, recipe_id) == (1, 2)
    assert db.fetch_keywords_for_prompt(prompt_id) == []
    assert db.fetch_keywords_for_prompt(recipe_id) == []
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM LocalPromptCollectionItems WHERE collection_id = ?",
            (collection_id,),
        )
        .fetchone()[0]
        == 2
    )
    assert db.browse_prompts(collection_id=collection_id)[3] == 0

    delete_events = db.get_sync_log_entries(since_change_id=baseline_change_id)
    prompt_deletes = [
        event
        for event in delete_events
        if event["entity"] == "Prompts" and event["operation"] == "delete"
    ]
    assert [event["version"] for event in prompt_deletes] == [2, 2]
    assert [event["payload"]["keywords"] for event in prompt_deletes] == [
        ["alpha", "shared"],
        ["beta", "shared"],
    ]
    assert (
        sum(
            event["entity"] == "PromptKeywordLinks" and event["operation"] == "unlink"
            for event in delete_events
        )
        == 4
    )

    restored = db.restore_deleted_prompts(result.targets)

    assert tuple(entry.local_id for entry in restored.entries) == tuple(
        entry.local_id for entry in result.entries
    )
    assert [entry.restored_version for entry in restored.entries] == [3, 3]
    assert {row["id"] for row in db.search_prompts("unique body")[0]} == {
        prompt_id,
        recipe_id,
    }
    assert db.fetch_keywords_for_prompt(prompt_id) == ["alpha", "shared"]
    assert db.fetch_keywords_for_prompt(recipe_id) == ["beta", "shared"]
    assert db.browse_prompts(collection_id=collection_id)[3] == 2
    assert (
        db.get_connection()
        .execute(
            "SELECT COUNT(*) FROM LocalPromptCollectionItems WHERE collection_id = ?",
            (collection_id,),
        )
        .fetchone()[0]
        == 2
    )

    restore_events = db.get_sync_log_entries(
        since_change_id=delete_events[-1]["change_id"]
    )
    prompt_restores = [
        event
        for event in restore_events
        if event["entity"] == "Prompts" and event["operation"] == "update"
    ]
    assert [event["version"] for event in prompt_restores] == [3, 3]
    assert [event["payload"]["keywords"] for event in prompt_restores] == [
        ["alpha", "shared"],
        ["beta", "shared"],
    ]
    assert (
        sum(
            event["entity"] == "PromptKeywordLinks" and event["operation"] == "link"
            for event in restore_events
        )
        == 4
    )


def test_batch_results_are_canonical_for_reverse_ordered_targets(db):
    first_id, _ = _add_artifact(db, name="First canonical")
    second_id, _ = _add_artifact(db, name="Second canonical", artifact_type="recipe")

    deleted = db.soft_delete_prompts(
        (PromptBatchTarget(second_id, 1), PromptBatchTarget(first_id, 1))
    )
    restored = db.restore_deleted_prompts(tuple(reversed(deleted.targets)))

    assert [entry.local_id for entry in deleted.entries] == [first_id, second_id]
    assert [entry.local_id for entry in restored.entries] == [first_id, second_id]


def test_batch_restore_recovers_keyword_soft_deleted_after_tombstone(db):
    prompt_id, _ = _add_artifact(
        db, name="Recover deleted keyword", keywords=["Recover Later"]
    )
    receipt = db.soft_delete_prompts((PromptBatchTarget(prompt_id, 1),))
    assert db.soft_delete_keyword("recover later") is True

    restored = db.restore_deleted_prompts(receipt.targets)

    assert restored.entries[0].restored_version == 3
    assert db.fetch_keywords_for_prompt(prompt_id) == ["recover later"]
    keyword = db.get_active_keyword_by_text("recover later")
    assert keyword is not None
    assert keyword["version"] == 3
    assert [
        row["id"] for row in db.search_prompts("recover later", ["keywords"])[0]
    ] == [prompt_id]


@pytest.mark.parametrize(
    "method_name", ["soft_delete_prompts", "restore_deleted_prompts"]
)
@pytest.mark.parametrize(
    "container_factory", [list, lambda values: TupleSubclass(values)]
)
def test_batch_methods_require_an_exact_tuple(db, method_name, container_factory):
    local_id, _ = _add_artifact(db, name=f"Exact tuple {method_name}")
    target = PromptBatchTarget(local_id, 1)

    with pytest.raises(TypeError, match="tuple"):
        getattr(db, method_name)(container_factory((target,)))

    assert _row_state(db, local_id) == (0, 1)


class TupleSubclass(tuple):
    """Prove that the DB boundary requires the exact immutable container type."""


@pytest.mark.parametrize(
    "method_name", ["soft_delete_prompts", "restore_deleted_prompts"]
)
def test_batch_methods_reject_empty_tuple_before_sql(db, method_name):
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)
    try:
        with pytest.raises(ValueError, match="non-empty"):
            getattr(db, method_name)(())
    finally:
        db.get_connection().set_trace_callback(None)

    assert statements == []


@pytest.mark.parametrize(
    "method_name", ["soft_delete_prompts", "restore_deleted_prompts"]
)
def test_batch_methods_reject_duplicate_local_ids_before_sql(db, method_name):
    local_id, _ = _add_artifact(db, name=f"Duplicate target {method_name}")
    targets = (PromptBatchTarget(local_id, 1), PromptBatchTarget(local_id, 2))
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)
    try:
        with pytest.raises(ValueError, match="unique"):
            getattr(db, method_name)(targets)
    finally:
        db.get_connection().set_trace_callback(None)

    assert statements == []
    assert _row_state(db, local_id) == (0, 1)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("local_id", True),
        ("local_id", 0),
        ("local_id", -1),
        ("local_id", 2**63),
        ("expected_version", True),
        ("expected_version", 0),
        ("expected_version", -1),
        ("expected_version", 2**63),
    ],
)
def test_batch_target_rejects_noncanonical_integer_values(field_name, value):
    values = {"local_id": 1, "expected_version": 1}
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        PromptBatchTarget(**values)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("local_id", True),
        ("local_id", 0),
        ("local_id", -1),
        ("local_id", 2**63),
        ("expected_version", True),
        ("expected_version", 0),
        ("expected_version", -1),
        ("expected_version", 2**63),
    ],
)
def test_batch_boundary_revalidates_tampered_targets_before_sql(db, field_name, value):
    target = object.__new__(PromptBatchTarget)
    object.__setattr__(target, "local_id", 1)
    object.__setattr__(target, "expected_version", 1)
    object.__setattr__(target, field_name, value)
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)
    try:
        with pytest.raises(ValueError, match=field_name):
            db.soft_delete_prompts((target,))
    finally:
        db.get_connection().set_trace_callback(None)

    assert statements == []


@pytest.mark.parametrize("failure", ["missing", "stale"])
def test_batch_delete_validates_every_target_before_first_write(
    db, monkeypatch, failure
):
    first_id, _ = _add_artifact(db, name=f"First delete {failure}")
    second_id, _ = _add_artifact(db, name=f"Second delete {failure}")
    second_target = (
        PromptBatchTarget(second_id + 999_999, 1)
        if failure == "missing"
        else PromptBatchTarget(second_id, 2)
    )
    helper_calls: list[int] = []
    original = db._delete_prompt_in_transaction

    def record_helper(*args, **kwargs):
        helper_calls.append(int(kwargs["row"]["id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "_delete_prompt_in_transaction", record_helper)

    with pytest.raises(ExpectedVersionConflictError):
        db.soft_delete_prompts((PromptBatchTarget(first_id, 1), second_target))

    assert helper_calls == []
    _assert_all_active_batch_rows(db, (first_id, second_id))


@pytest.mark.parametrize("failure", ["missing", "stale"])
def test_batch_restore_rejects_missing_or_stale_targets_atomically(db, failure):
    first_id, _ = _add_artifact(db, name=f"First restore {failure}")
    second_id, _ = _add_artifact(db, name=f"Second restore {failure}")
    receipt = db.soft_delete_prompts(
        (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
    )
    second_target = (
        PromptBatchTarget(second_id + 999_999, 2)
        if failure == "missing"
        else PromptBatchTarget(second_id, 1)
    )

    with pytest.raises(ExpectedVersionConflictError):
        db.restore_deleted_prompts((receipt.targets[0], second_target))

    _assert_no_active_batch_rows(db, (first_id, second_id))


def test_batch_restore_validates_every_recovery_payload_before_first_write(
    db, monkeypatch
):
    first_id, _ = _add_artifact(db, name="First restore payload", keywords=["one"])
    second_id, second_uuid = _add_artifact(
        db, name="Second restore payload", keywords=["two"]
    )
    receipt = db.soft_delete_prompts(
        (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
    )
    db.get_connection().execute(
        """
        UPDATE sync_log SET payload = ?
        WHERE entity = 'Prompts' AND entity_uuid = ?
          AND operation = 'delete' AND version = 2
        """,
        ('{"keywords": ["private two"]}', second_uuid),
    )
    db.get_connection().commit()
    helper_calls: list[int] = []
    original = db._restore_prompt_in_transaction

    def record_helper(*args, **kwargs):
        helper_calls.append(int(kwargs["row"]["id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(db, "_restore_prompt_in_transaction", record_helper)

    with pytest.raises(DatabaseError, match="batch restore"):
        db.restore_deleted_prompts(receipt.targets)

    assert helper_calls == []
    _assert_no_active_batch_rows(db, (first_id, second_id))


@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_batch_forced_second_mutation_failure_rolls_back_every_row(
    db, monkeypatch, operation
):
    first_id, _ = _add_artifact(db, name=f"First forced {operation}")
    second_id, _ = _add_artifact(db, name=f"Second forced {operation}")
    active_targets = (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
    if operation == "delete":
        targets = active_targets
        helper_name = "_delete_prompt_in_transaction"
    else:
        targets = db.soft_delete_prompts(active_targets).targets
        helper_name = "_restore_prompt_in_transaction"
    original = getattr(db, helper_name)
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise sqlite3.OperationalError("private second mutation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(db, helper_name, fail_second)

    with pytest.raises(DatabaseError) as caught:
        getattr(
            db,
            f"{'soft_delete' if operation == 'delete' else 'restore_deleted'}_prompts",
        )(targets)

    assert caught.value.__cause__ is None
    if operation == "delete":
        _assert_all_active_batch_rows(db, (first_id, second_id))
    else:
        _assert_no_active_batch_rows(db, (first_id, second_id))


@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_batch_result_construction_failure_rolls_back_before_commit(
    db, monkeypatch, operation
):
    first_id, _ = _add_artifact(db, name=f"First result {operation}")
    second_id, _ = _add_artifact(db, name=f"Second result {operation}")
    active_targets = (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
    if operation == "delete":
        targets = active_targets
        result_name = "PromptBatchDeleteResult"
        method_name = "soft_delete_prompts"
    else:
        targets = db.soft_delete_prompts(active_targets).targets
        result_name = "PromptBatchRestoreResult"
        method_name = "restore_deleted_prompts"

    class ExplodingResult:
        def __init__(self, **_kwargs):
            raise ValueError("private result-constructor failure")

    monkeypatch.setattr(prompts_db_module, result_name, ExplodingResult, raising=False)

    with pytest.raises(DatabaseError) as caught:
        getattr(db, method_name)(targets)

    assert caught.value.__cause__ is None
    if operation == "delete":
        _assert_all_active_batch_rows(db, (first_id, second_id))
    else:
        _assert_no_active_batch_rows(db, (first_id, second_id))


@pytest.mark.parametrize("operation", ["delete", "restore"])
def test_batch_uses_exactly_one_begin_immediate(db, operation):
    first_id, _ = _add_artifact(db, name=f"First begin {operation}")
    second_id, _ = _add_artifact(db, name=f"Second begin {operation}")
    active_targets = (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
    if operation == "delete":
        targets = active_targets
        method_name = "soft_delete_prompts"
    else:
        targets = db.soft_delete_prompts(active_targets).targets
        method_name = "restore_deleted_prompts"
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)
    try:
        getattr(db, method_name)(targets)
    finally:
        db.get_connection().set_trace_callback(None)

    begins = [
        statement.strip().upper()
        for statement in statements
        if statement.lstrip().upper().startswith("BEGIN")
    ]
    assert begins == ["BEGIN IMMEDIATE"]


@pytest.mark.parametrize("lookup", ["integer", "name", "uuid"])
def test_batch_refactor_preserves_legacy_single_lookup_and_mapping(db, lookup):
    local_id, artifact_uuid = _add_artifact(
        db,
        name=f"Legacy {lookup}",
        artifact_type="recipe",
        keywords=["Legacy Keyword"],
    )
    identifier: int | str = {
        "integer": local_id,
        "name": f"Legacy {lookup}",
        "uuid": artifact_uuid,
    }[lookup]

    assert db.soft_delete_prompt(identifier, expected_version=1) is True
    assert _row_state(db, local_id) == (1, 2)

    restored = db.restore_deleted_prompt(identifier, expected_version=2)

    assert isinstance(restored, dict)
    assert restored["id"] == local_id
    assert restored["version"] == 3
    assert restored["deleted"] == 0
    assert restored["artifact_type"] == "recipe"
    assert restored["keywords"] == ["legacy keyword"]


def test_batch_refactor_preserves_legacy_single_edge_contracts(db):
    local_id, _ = _add_artifact(db, name="Legacy edge")
    assert db.soft_delete_prompt("missing legacy prompt") is False

    with pytest.raises(ExpectedVersionConflictError):
        db.soft_delete_prompt(local_id, expected_version=2)

    assert _row_state(db, local_id) == (0, 1)
    assert db.soft_delete_prompt(local_id) is True
    assert _row_state(db, local_id) == (1, 2)
    assert list(inspect.signature(PromptsDatabase.soft_delete_prompt).parameters) == [
        "self",
        "prompt_id_or_name_or_uuid",
        "expected_version",
    ]
    assert list(
        inspect.signature(PromptsDatabase.restore_deleted_prompt).parameters
    ) == ["self", "prompt_id_or_name_or_uuid", "expected_version"]


def test_batch_refactor_routes_legacy_single_wrappers_through_shared_helpers(
    db, monkeypatch
):
    local_id, _ = _add_artifact(db, name="Legacy shared helper")
    delete_original = db._delete_prompt_in_transaction
    restore_original = db._restore_prompt_in_transaction
    calls: list[str] = []

    def delete_spy(*args, **kwargs):
        calls.append("delete")
        return delete_original(*args, **kwargs)

    def restore_spy(*args, **kwargs):
        calls.append("restore")
        return restore_original(*args, **kwargs)

    monkeypatch.setattr(db, "_delete_prompt_in_transaction", delete_spy)
    monkeypatch.setattr(db, "_restore_prompt_in_transaction", restore_spy)

    assert db.soft_delete_prompt(local_id) is True
    restored = db.restore_deleted_prompt(local_id, expected_version=2)

    assert calls == ["delete", "restore"]
    assert restored["version"] == 3


def test_batch_success_diagnostics_are_aggregate_and_privacy_safe(db):
    _set_next_prompt_id(db, 71_234_567)
    first_id, first_uuid = _add_artifact(
        db,
        name="PRIVATE NAME ALPHA",
        keywords=["PRIVATE KEYWORD ALPHA"],
        body="PRIVATE BODY ALPHA",
    )
    second_id, second_uuid = _add_artifact(
        db,
        name="PRIVATE NAME BETA",
        keywords=["PRIVATE KEYWORD BETA"],
        body="PRIVATE BODY BETA",
    )
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        deleted = db.soft_delete_prompts(
            (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
        )
        db.restore_deleted_prompts(deleted.targets)
    finally:
        logger.remove(sink)

    rendered = "\n".join(messages)
    assert "operation=delete count=2" in rendered
    assert "operation=restore count=2" in rendered
    for forbidden in (
        "PRIVATE NAME",
        "PRIVATE KEYWORD",
        "PRIVATE BODY",
        str(first_id),
        str(second_id),
        first_uuid,
        second_uuid,
    ):
        assert forbidden not in rendered


def test_batch_failure_is_bounded_and_privacy_safe(db, monkeypatch):
    _set_next_prompt_id(db, 81_234_567)
    first_id, first_uuid = _add_artifact(
        db, name="PRIVATE FAILURE NAME", keywords=["PRIVATE FAILURE KEYWORD"]
    )
    second_id, second_uuid = _add_artifact(db, name="Second failure row")
    private_exception = (
        f"PRIVATE EXCEPTION {first_id} {second_id} {first_uuid} {second_uuid}"
    )

    def fail_without_mutating(*_args, **_kwargs):
        raise sqlite3.OperationalError(private_exception)

    monkeypatch.setattr(db, "_delete_prompt_in_transaction", fail_without_mutating)
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        with pytest.raises(DatabaseError) as caught:
            db.soft_delete_prompts(
                (PromptBatchTarget(first_id, 1), PromptBatchTarget(second_id, 1))
            )
    finally:
        logger.remove(sink)

    rendered = "\n".join(messages)
    assert str(caught.value) == "Prompt batch delete failed."
    assert caught.value.__cause__ is None
    assert "operation=delete count=2 category=OperationalError" in rendered
    assert "Traceback" not in rendered
    for forbidden in (
        "PRIVATE FAILURE NAME",
        "PRIVATE FAILURE KEYWORD",
        private_exception,
        str(first_id),
        str(second_id),
        first_uuid,
        second_uuid,
    ):
        assert forbidden not in rendered
