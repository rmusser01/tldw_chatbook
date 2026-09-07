import uuid
from collections.abc import Sequence
from typing import cast

import pytest

from tldw_chatbook.Sync_Interop.notes_organization import (
    NOTES_ORGANIZATION_DOMAINS,
    NotesOrganizationValidationError,
    new_organization_sync_id,
    organization_link_id,
    parse_notes_organization_payload,
    validate_organization_object_id,
    validate_resource_sync_id,
)


CANONICAL_SYNC_ID = "123e4567-e89b-42d3-a456-426614174000"
OTHER_SYNC_ID = "123e4567-e89b-42d3-a456-426614174001"


def test_notes_organization_domains_use_the_server_order() -> None:
    assert NOTES_ORGANIZATION_DOMAINS == (
        "notes.keyword",
        "notes.keyword_link",
        "notes.keyword_collection",
        "notes.keyword_collection_link",
        "notes.folder",
        "notes.folder_link",
    )


def test_organization_link_ids_match_server_vectors() -> None:
    assert organization_link_id(
        "notes.keyword_link", ["note", "note-123", "kw-456"]
    ) == "notes.keyword_link:sha256:10f9eab3be80b6e439ce1bcf8fae952527bde7d7e026d0e227f0a87ada963be0"
    assert organization_link_id(
        "notes.keyword_collection_link", ["collection-123", "kw-456"]
    ) == "notes.keyword_collection_link:sha256:e9427c2d8bc4cfa8586130bc1fcc54cf432ca6dbb3df77bab3e65033b6148199"
    assert organization_link_id(
        "notes.folder_link", ["note-123", "folder-456"]
    ) == "notes.folder_link:sha256:9076b60d9d8476f852736928ef3661cb06d9ba55696dd4504657c753f414b670"


@pytest.mark.parametrize(
    ("domain", "members"),
    [
        ("notes.keyword_link", ["note", "note-123"]),
        ("notes.keyword_collection_link", ["collection-123"]),
        ("notes.folder_link", ["note-123", "folder-456", "extra"]),
        ("notes.keyword", ["not", "a", "relationship"]),
        ("notes.folder_link", ["note-123", cast(str, 1)]),
    ],
)
def test_organization_link_id_rejects_invalid_identity_shape(
    domain: str,
    members: list[str],
) -> None:
    with pytest.raises(NotesOrganizationValidationError) as exc_info:
        organization_link_id(domain, members)

    assert exc_info.value.error_code == "notes_organization_link_identity_invalid"


@pytest.mark.parametrize("members", ["ab", b"ab"])
def test_organization_link_id_rejects_scalar_sequence_containers(
    members: str | bytes,
) -> None:
    with pytest.raises(NotesOrganizationValidationError) as exc_info:
        organization_link_id("notes.folder_link", cast(Sequence[str], members))

    assert exc_info.value.error_code == "notes_organization_link_identity_invalid"


def test_resource_sync_ids_are_canonical_lowercase_uuid4() -> None:
    sync_id = new_organization_sync_id()

    assert sync_id == str(uuid.UUID(sync_id))
    assert uuid.UUID(sync_id).version == 4
    assert validate_resource_sync_id(sync_id) == sync_id

    for invalid in (
        sync_id.upper(),
        "550e8400-e29b-11d4-a716-446655440000",
        "not-a-uuid",
        cast(str, 1),
    ):
        with pytest.raises(NotesOrganizationValidationError) as exc_info:
            validate_resource_sync_id(invalid)
        assert exc_info.value.error_code == "notes_organization_resource_sync_id_invalid"


@pytest.mark.parametrize(
    ("domain", "payload"),
    [
        ("notes.keyword", {"keyword": "research"}),
        (
            "notes.keyword_link",
            {
                "subject_type": "conversation",
                "subject_id": "conversation-1",
                "keyword_sync_id": CANONICAL_SYNC_ID,
            },
        ),
        (
            "notes.keyword_collection",
            {"name": "Research", "parent_sync_id": None},
        ),
        (
            "notes.keyword_collection_link",
            {
                "collection_sync_id": CANONICAL_SYNC_ID,
                "keyword_sync_id": OTHER_SYNC_ID,
            },
        ),
        ("notes.folder", {"name": "Research", "parent_sync_id": None}),
        (
            "notes.folder_link",
            {"note_id": CANONICAL_SYNC_ID, "folder_sync_id": OTHER_SYNC_ID},
        ),
    ],
)
def test_payloads_reject_unknown_fields(domain: str, payload: dict[str, object]) -> None:
    with pytest.raises(NotesOrganizationValidationError) as exc_info:
        parse_notes_organization_payload(domain, "upsert", {**payload, "extra": True})

    assert exc_info.value.error_code == "notes_organization_payload_invalid"


@pytest.mark.parametrize(
    ("domain", "field", "maximum"),
    [
        ("notes.keyword", "keyword", 100),
        ("notes.keyword_collection", "name", 255),
        ("notes.folder", "name", 500),
    ],
)
def test_resource_names_are_trimmed_and_enforce_server_bounds(
    domain: str,
    field: str,
    maximum: int,
) -> None:
    assert parse_notes_organization_payload(domain, "upsert", {field: f"  {'x' * maximum}  "})[
        field
    ] == "x" * maximum

    for invalid in ("   ", "x" * (maximum + 1)):
        with pytest.raises(NotesOrganizationValidationError) as exc_info:
            parse_notes_organization_payload(domain, "upsert", {field: invalid})
        assert exc_info.value.error_code == "notes_organization_payload_invalid"


@pytest.mark.parametrize(
    ("domain", "payload"),
    [
        ("notes.keyword", {}),
        ("notes.keyword_collection", {}),
        ("notes.folder", {}),
    ],
)
def test_resource_tombstones_require_empty_payloads(
    domain: str,
    payload: dict[str, object],
) -> None:
    assert parse_notes_organization_payload(domain, "tombstone", payload) == {}

    with pytest.raises(NotesOrganizationValidationError):
        parse_notes_organization_payload(domain, "tombstone", {"unexpected": True})


@pytest.mark.parametrize(
    ("domain", "payload"),
    [
        (
            "notes.keyword_link",
            {
                "subject_type": "note",
                "subject_id": CANONICAL_SYNC_ID,
                "keyword_sync_id": OTHER_SYNC_ID,
            },
        ),
        (
            "notes.keyword_collection_link",
            {
                "collection_sync_id": CANONICAL_SYNC_ID,
                "keyword_sync_id": OTHER_SYNC_ID,
            },
        ),
        (
            "notes.folder_link",
            {"note_id": CANONICAL_SYNC_ID, "folder_sync_id": OTHER_SYNC_ID},
        ),
    ],
)
def test_link_tombstones_require_the_full_identity_payload(
    domain: str,
    payload: dict[str, object],
) -> None:
    assert parse_notes_organization_payload(domain, "tombstone", payload) == payload

    with pytest.raises(NotesOrganizationValidationError):
        parse_notes_organization_payload(domain, "tombstone", {})


@pytest.mark.parametrize(
    ("domain", "payload"),
    [
        (
            "notes.keyword_link",
            {
                "subject_type": "note",
                "subject_id": "1",
                "keyword_sync_id": CANONICAL_SYNC_ID,
            },
        ),
        (
            "notes.keyword_link",
            {
                "subject_type": "conversation",
                "subject_id": "conversation-1",
                "keyword_sync_id": "1",
            },
        ),
        ("notes.keyword_collection", {"name": "Research", "parent_sync_id": "1"}),
        (
            "notes.keyword_collection_link",
            {"collection_sync_id": "1", "keyword_sync_id": CANONICAL_SYNC_ID},
        ),
        ("notes.folder", {"name": "Research", "parent_sync_id": "1"}),
        ("notes.folder_link", {"note_id": CANONICAL_SYNC_ID, "folder_sync_id": "1"}),
    ],
)
def test_payloads_reject_noncanonical_resource_references(
    domain: str,
    payload: dict[str, object],
) -> None:
    with pytest.raises(NotesOrganizationValidationError) as exc_info:
        parse_notes_organization_payload(domain, "upsert", payload)

    assert exc_info.value.error_code == "notes_organization_resource_sync_id_invalid"


def test_conversation_keyword_link_accepts_non_uuid_subject_identity() -> None:
    payload = {
        "subject_type": "conversation",
        "subject_id": " conversation-1 ",
        "keyword_sync_id": CANONICAL_SYNC_ID,
    }

    assert parse_notes_organization_payload("notes.keyword_link", "upsert", payload) == {
        **payload,
        "subject_id": "conversation-1",
    }


def test_object_ids_validate_resource_or_recomputed_link_identity() -> None:
    validate_organization_object_id(
        "notes.keyword", CANONICAL_SYNC_ID, {"keyword": "Research"}
    )

    payload = {
        "subject_type": "note",
        "subject_id": CANONICAL_SYNC_ID,
        "keyword_sync_id": OTHER_SYNC_ID,
    }
    object_id = organization_link_id("notes.keyword_link", list(payload.values()))
    validate_organization_object_id("notes.keyword_link", object_id, payload)

    with pytest.raises(NotesOrganizationValidationError) as exc_info:
        validate_organization_object_id(
            "notes.keyword_link",
            object_id,
            {**payload, "keyword_sync_id": CANONICAL_SYNC_ID},
        )
    assert exc_info.value.error_code == "notes_organization_link_identity_invalid"


@pytest.mark.parametrize(
    ("domain", "operation", "payload", "error_code"),
    [
        ("notes.unknown", "upsert", {}, "notes_organization_domain_invalid"),
        ("notes.keyword", "replace", {}, "notes_organization_operation_invalid"),
        (
            "notes.keyword",
            "upsert",
            cast(dict[str, object], []),
            "notes_organization_payload_invalid",
        ),
    ],
)
def test_validation_errors_expose_stable_codes(
    domain: str,
    operation: str,
    payload: dict[str, object],
    error_code: str,
) -> None:
    with pytest.raises(NotesOrganizationValidationError) as exc_info:
        parse_notes_organization_payload(domain, operation, payload)

    assert exc_info.value.error_code == error_code
