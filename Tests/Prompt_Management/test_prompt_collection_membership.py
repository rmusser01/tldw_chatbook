"""File-backed contracts for local Prompt collection membership writes."""

from __future__ import annotations

from functools import partial

import pytest

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import LocalPromptService


@pytest.fixture
def local_prompts(tmp_path):
    database = PromptsDatabase(tmp_path / "prompt-memberships.db", client_id="test")
    prompt_ids = [
        database.add_prompt(
            name=name,
            author="Writer",
            details="Details",
            system_prompt="System",
            user_prompt="User",
            keywords=[],
            overwrite=False,
        )[0]
        for name in ("First Prompt", "Second Prompt", "Third Prompt")
    ]
    service = LocalPromptService(database)
    try:
        yield database, service, prompt_ids
    finally:
        database.close_connection()


def _create_collection(
    service: LocalPromptService, name: str, prompt_ids: list[int] | None = None
) -> int:
    return service.create_prompt_collection(
        {"name": name, "prompt_ids": prompt_ids or []}
    )["collection_id"]


def _members(database: PromptsDatabase, collection_id: int) -> list[tuple[int, int]]:
    rows = database.get_connection().execute(
        """
        SELECT prompt_id, position
        FROM LocalPromptCollectionItems
        WHERE collection_id = ?
        ORDER BY position, prompt_id
        """,
        (collection_id,),
    )
    return [(int(row["prompt_id"]), int(row["position"])) for row in rows]


def _collection_state(database: PromptsDatabase, collection_id: int) -> tuple:
    row = (
        database.get_connection()
        .execute(
            """
        SELECT name, description, version, deleted
        FROM LocalPromptCollections
        WHERE collection_id = ?
        """,
            (collection_id,),
        )
        .fetchone()
    )
    return tuple(row)


@pytest.mark.parametrize("operation", ["create", "update", "replace"])
def test_membership_writes_serialize_identity_validation_in_one_transaction(
    local_prompts, operation
):
    database, service, (first_prompt, second_prompt, _third_prompt) = local_prompts
    service.list_prompt_collections(limit=1)
    validation_markers = ["SELECT 1 FROM PROMPTS WHERE ID"]

    if operation == "create":
        action = partial(
            service.create_prompt_collection,
            {"name": "Traced Create", "prompt_ids": [first_prompt]},
        )
    else:
        collection_id = _create_collection(service, f"Traced {operation.title()}")
        if operation == "update":
            validation_markers.append("SELECT NAME FROM LOCALPROMPTCOLLECTIONS")
            action = partial(
                service.update_prompt_collection,
                collection_id,
                {"prompt_ids": [second_prompt]},
            )
        else:
            validation_markers.append("SELECT 1 FROM LOCALPROMPTCOLLECTIONS")
            action = partial(
                service.replace_prompt_collection_memberships,
                first_prompt,
                [collection_id],
            )

    statements: list[str] = []
    connection = database.get_connection()
    connection.set_trace_callback(statements.append)
    try:
        action()
    finally:
        connection.set_trace_callback(None)

    normalized = [" ".join(statement.upper().split()) for statement in statements]
    assert normalized.count("BEGIN IMMEDIATE") == 1
    begin = normalized.index("BEGIN IMMEDIATE")
    commit = normalized.index("COMMIT", begin)
    validations = []
    for marker in validation_markers:
        validation = next(
            index for index, statement in enumerate(normalized) if marker in statement
        )
        assert begin < validation < commit
        validations.append(validation)
    membership_write = next(
        index
        for index, statement in enumerate(normalized)
        if (
            statement.startswith("INSERT INTO LOCALPROMPTCOLLECTIONITEMS")
            or statement.startswith("DELETE FROM LOCALPROMPTCOLLECTIONITEMS")
        )
    )
    assert begin < max(validations) < membership_write < commit


def test_collection_update_replaces_only_that_collections_members(local_prompts):
    database, service, (first_prompt, second_prompt, _third_prompt) = local_prompts
    first_collection = _create_collection(
        service, "First Collection", [first_prompt, second_prompt]
    )
    second_collection = _create_collection(service, "Second Collection", [first_prompt])

    service.update_prompt_collection(first_collection, {"prompt_ids": [second_prompt]})

    assert _members(database, first_collection) == [(second_prompt, 0)]
    assert _members(database, second_collection) == [(first_prompt, 0)]


@pytest.mark.parametrize("member_kind", ["missing", "deleted"])
def test_collection_create_rolls_back_when_any_prompt_is_not_active(
    local_prompts, member_kind
):
    database, service, (active_prompt, deleted_prompt, _third_prompt) = local_prompts
    invalid_prompt = (2**63) - 1
    if member_kind == "deleted":
        assert database.soft_delete_prompt(deleted_prompt) is True
        invalid_prompt = deleted_prompt

    with pytest.raises(ValueError, match="active Prompt"):
        service.create_prompt_collection(
            {
                "name": f"Rejected {member_kind}",
                "description": "must roll back",
                "prompt_ids": [active_prompt, invalid_prompt],
            }
        )

    assert (
        database.get_connection()
        .execute(
            "SELECT 1 FROM LocalPromptCollections WHERE name = ?",
            (f"Rejected {member_kind}",),
        )
        .fetchone()
        is None
    )


def test_collection_update_rolls_back_fields_version_and_membership_on_invalid_prompt(
    local_prompts,
):
    database, service, (first_prompt, second_prompt, deleted_prompt) = local_prompts
    collection_id = _create_collection(service, "Original", [first_prompt])
    assert database.soft_delete_prompt(deleted_prompt) is True
    before_collection = _collection_state(database, collection_id)
    before_members = _members(database, collection_id)

    with pytest.raises(ValueError, match="active Prompt"):
        service.update_prompt_collection(
            collection_id,
            {
                "name": "Changed",
                "description": "Changed description",
                "prompt_ids": [second_prompt, deleted_prompt],
            },
        )

    assert _collection_state(database, collection_id) == before_collection
    assert _members(database, collection_id) == before_members


@pytest.mark.parametrize(
    "prompt_ids",
    [[True], [1.5], ["1"], [0], [-1], [2**63], [1, 1]],
)
def test_collection_create_rejects_malformed_prompt_ids_without_persisting(
    local_prompts, prompt_ids
):
    database, service, _prompts = local_prompts

    with pytest.raises(ValueError, match="prompt_ids"):
        service.create_prompt_collection(
            {"name": "Malformed Members", "prompt_ids": prompt_ids}
        )

    assert (
        database.get_connection()
        .execute(
            "SELECT 1 FROM LocalPromptCollections WHERE name = 'Malformed Members'"
        )
        .fetchone()
        is None
    )


def test_collection_update_rejects_inactive_or_missing_collection_before_membership(
    local_prompts,
):
    database, service, (first_prompt, second_prompt, _third_prompt) = local_prompts
    active_collection = _create_collection(service, "Active", [first_prompt])
    inactive_collection = _create_collection(service, "Inactive", [first_prompt])
    database.get_connection().execute(
        "UPDATE LocalPromptCollections SET deleted = 1 WHERE collection_id = ?",
        (inactive_collection,),
    )
    database.get_connection().commit()

    for collection_id in (inactive_collection, (2**63) - 1):
        with pytest.raises(ValueError, match="collection not found"):
            service.update_prompt_collection(
                collection_id, {"prompt_ids": [second_prompt]}
            )

    assert _members(database, inactive_collection) == [(first_prompt, 0)]
    assert _members(database, active_collection) == [(first_prompt, 0)]


def test_prompt_membership_replace_supports_multiple_collections_and_preserves_others(
    local_prompts,
):
    database, service, (first_prompt, second_prompt, _third_prompt) = local_prompts
    first_collection = _create_collection(service, "Zulu", [second_prompt])
    second_collection = _create_collection(service, "Alpha", [second_prompt])
    untouched_collection = _create_collection(service, "Untouched", [second_prompt])
    before_states = {
        collection_id: _collection_state(database, collection_id)
        for collection_id in (
            first_collection,
            second_collection,
            untouched_collection,
        )
    }

    outcome = service.replace_prompt_collection_memberships(
        first_prompt, [second_collection, first_collection]
    )
    listed = service.list_prompt_collection_memberships(first_prompt)

    assert outcome == {
        "prompt_id": first_prompt,
        "collection_ids": (first_collection, second_collection),
        "changed": True,
    }
    assert listed == {
        "prompt_id": first_prompt,
        "collection_ids": (first_collection, second_collection),
        "changed": False,
    }
    assert _members(database, first_collection) == [
        (second_prompt, 0),
        (first_prompt, 1),
    ]
    assert _members(database, second_collection) == [
        (second_prompt, 0),
        (first_prompt, 1),
    ]
    assert _collection_state(database, first_collection)[2] == (
        before_states[first_collection][2] + 1
    )
    assert _collection_state(database, second_collection)[2] == (
        before_states[second_collection][2] + 1
    )
    assert (
        _collection_state(database, untouched_collection)
        == before_states[untouched_collection]
    )


def test_prompt_membership_list_excludes_deleted_collections_and_sorts_active_ids(
    local_prompts,
):
    database, service, (prompt_id, _second_prompt, _third_prompt) = local_prompts
    first_active = _create_collection(service, "First Active", [prompt_id])
    deleted_collection = _create_collection(service, "Deleted", [prompt_id])
    last_active = _create_collection(service, "Last Active", [prompt_id])
    database.get_connection().execute(
        "UPDATE LocalPromptCollections SET deleted = 1 WHERE collection_id = ?",
        (deleted_collection,),
    )
    database.get_connection().commit()

    outcome = service.list_prompt_collection_memberships(prompt_id)

    assert outcome == {
        "prompt_id": prompt_id,
        "collection_ids": (first_active, last_active),
        "changed": False,
    }


def test_prompt_membership_replace_is_idempotent_without_any_mutation(local_prompts):
    database, service, (first_prompt, _second_prompt, _third_prompt) = local_prompts
    collection_id = _create_collection(service, "Idempotent", [first_prompt])
    before_changes = database.get_connection().total_changes
    before_collection = _collection_state(database, collection_id)
    before_members = _members(database, collection_id)

    outcome = service.replace_prompt_collection_memberships(
        first_prompt, [collection_id]
    )

    assert outcome == {
        "prompt_id": first_prompt,
        "collection_ids": (collection_id,),
        "changed": False,
    }
    assert database.get_connection().total_changes == before_changes
    assert _collection_state(database, collection_id) == before_collection
    assert _members(database, collection_id) == before_members


def test_prompt_membership_replace_can_clear_only_the_target_prompt(local_prompts):
    database, service, (first_prompt, second_prompt, _third_prompt) = local_prompts
    collection_id = _create_collection(
        service, "Clearable", [first_prompt, second_prompt]
    )

    outcome = service.replace_prompt_collection_memberships(first_prompt, [])

    assert outcome == {
        "prompt_id": first_prompt,
        "collection_ids": (),
        "changed": True,
    }
    assert _members(database, collection_id) == [(second_prompt, 1)]


@pytest.mark.parametrize("prompt_kind", ["missing", "deleted"])
def test_prompt_membership_methods_reject_non_active_prompt_without_writes(
    local_prompts, prompt_kind
):
    database, service, (first_prompt, deleted_prompt, _third_prompt) = local_prompts
    collection_id = _create_collection(service, "Existing", [first_prompt])
    invalid_prompt = (2**63) - 1
    if prompt_kind == "deleted":
        assert database.soft_delete_prompt(deleted_prompt) is True
        invalid_prompt = deleted_prompt
    before_collection = _collection_state(database, collection_id)
    before_members = _members(database, collection_id)

    with pytest.raises(ValueError, match="active Prompt"):
        service.list_prompt_collection_memberships(invalid_prompt)
    with pytest.raises(ValueError, match="active Prompt"):
        service.replace_prompt_collection_memberships(invalid_prompt, [collection_id])

    assert _collection_state(database, collection_id) == before_collection
    assert _members(database, collection_id) == before_members


def test_prompt_membership_replace_validates_all_collections_before_writes(
    local_prompts,
):
    database, service, (first_prompt, _second_prompt, _third_prompt) = local_prompts
    original_collection = _create_collection(service, "Original", [first_prompt])
    active_collection = _create_collection(service, "Active")
    inactive_collection = _create_collection(service, "Inactive")
    database.get_connection().execute(
        "UPDATE LocalPromptCollections SET deleted = 1 WHERE collection_id = ?",
        (inactive_collection,),
    )
    database.get_connection().commit()
    before_versions = {
        collection_id: _collection_state(database, collection_id)
        for collection_id in (original_collection, active_collection)
    }

    for invalid_collection in (inactive_collection, (2**63) - 1):
        with pytest.raises(ValueError, match="active collection"):
            service.replace_prompt_collection_memberships(
                first_prompt, [active_collection, invalid_collection]
            )

    assert _members(database, original_collection) == [(first_prompt, 0)]
    assert _members(database, active_collection) == []
    assert {
        collection_id: _collection_state(database, collection_id)
        for collection_id in (original_collection, active_collection)
    } == before_versions


def test_prompt_membership_methods_reject_malformed_identifiers(local_prompts):
    _database, service, (prompt_id, _second_prompt, _third_prompt) = local_prompts
    collection_id = _create_collection(service, "Valid")

    for malformed_prompt_id in (True, 0, -1, 1.5, "1", 2**63):
        with pytest.raises(ValueError, match="prompt_id"):
            service.list_prompt_collection_memberships(malformed_prompt_id)

    for malformed_collection_ids in (
        None,
        "1",
        {collection_id},
        [True],
        [1.5],
        ["1"],
        [0],
        [-1],
        [2**63],
        [collection_id, collection_id],
    ):
        with pytest.raises(ValueError, match="collection_ids"):
            service.replace_prompt_collection_memberships(
                prompt_id, malformed_collection_ids
            )


def test_prompt_membership_replace_rolls_back_a_mid_write_database_failure(
    local_prompts,
):
    database, service, (first_prompt, second_prompt, _third_prompt) = local_prompts
    original_collection = _create_collection(service, "Original", [first_prompt])
    first_target = _create_collection(service, "First Target", [second_prompt])
    failing_target = _create_collection(service, "Failing Target", [second_prompt])
    database.get_connection().execute(
        f"""
        CREATE TRIGGER fail_one_membership_insert
        BEFORE INSERT ON LocalPromptCollectionItems
        WHEN NEW.collection_id = {failing_target}
          AND NEW.prompt_id = {first_prompt}
        BEGIN
            SELECT RAISE(ABORT, 'forced membership failure');
        END
        """
    )
    database.get_connection().commit()
    before_states = {
        collection_id: _collection_state(database, collection_id)
        for collection_id in (original_collection, first_target, failing_target)
    }

    with pytest.raises(ValueError, match="membership update failed"):
        service.replace_prompt_collection_memberships(
            first_prompt, [first_target, failing_target]
        )

    assert _members(database, original_collection) == [(first_prompt, 0)]
    assert _members(database, first_target) == [(second_prompt, 0)]
    assert _members(database, failing_target) == [(second_prompt, 0)]
    assert {
        collection_id: _collection_state(database, collection_id)
        for collection_id in (original_collection, first_target, failing_target)
    } == before_states
