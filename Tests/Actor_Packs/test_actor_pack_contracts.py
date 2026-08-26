"""Pure trust-boundary tests for ``tldw.actor-pack/v1``."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping

import pytest

from tldw_chatbook.Actor_Packs.contracts import (
    ActorPackValidationError,
    actor_pack_content_digest,
    canonical_json_bytes,
    validate_actor_pack_document,
)
from tldw_chatbook.Actor_Packs import contracts as actor_pack_contracts

from .conftest import (
    PORTABLE_UUID,
    canonical_json,
    file_descriptor,
    with_content_digest,
)


def _rebuild(
    manifest: Mapping[str, object], files: Mapping[str, bytes]
) -> dict[str, object]:
    rebuilt = copy.deepcopy(dict(manifest))
    rebuilt["files"] = [
        file_descriptor(path, data) for path, data in sorted(files.items())
    ]
    return with_content_digest(rebuilt)


def _assert_category(
    category: str, manifest: Mapping[str, object], files: Mapping[str, bytes]
) -> None:
    with pytest.raises(ActorPackValidationError) as caught:
        validate_actor_pack_document(manifest, files)
    assert caught.value.category == category
    assert str(caught.value) == category


def test_minimal_character_document_is_immutable_and_path_free(
    minimal_character_manifest: Mapping[str, object],
    minimal_character_files: Mapping[str, bytes],
) -> None:
    document = validate_actor_pack_document(
        minimal_character_manifest, minimal_character_files
    )

    assert document.schema == "tldw.actor-pack/v1"
    assert document.actor_kind == "character"
    assert document.portable_uuid == PORTABLE_UUID
    assert document.payload_path == "actor/actor.json"
    assert document.portrait_path == "actor/portrait.png"
    assert document.sections == ()
    assert tuple(item.path for item in document.files) == (
        "actor/actor.json",
        "actor/portrait.png",
    )
    assert "/Users/" not in repr(document)
    assert "Guide" not in repr(document)
    with pytest.raises((AttributeError, TypeError)):
        document.actor_kind = "persona"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "category"),
    [
        ({"schema": "tldw.actor-pack/v2"}, "actor_pack_schema_unsupported"),
        ({"actor": None}, "actor_pack_manifest_invalid"),
        ({"files": []}, "actor_pack_inventory_invalid"),
        (
            {"required_features": ["future-capability"]},
            "actor_pack_feature_unsupported",
        ),
        ({"content_digest": "0" * 64}, "actor_pack_digest_mismatch"),
    ],
)
def test_manifest_contract_fails_closed_with_fixed_categories(
    mutation: Mapping[str, object],
    category: str,
    minimal_character_manifest: Mapping[str, object],
    minimal_character_files: Mapping[str, bytes],
) -> None:
    manifest = copy.deepcopy(dict(minimal_character_manifest))
    manifest.update(mutation)
    if "content_digest" not in mutation:
        manifest = with_content_digest(manifest)
    _assert_category(category, manifest, minimal_character_files)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/actor/actor.json",
        "../actor.json",
        "actor/../actor.json",
        "actor\\actor.json",
        "c:/actor.json",
        "Actor/actor.json",
        "actor/é.json",
        "actor//actor.json",
        "actor/./actor.json",
        "actor/con.txt",
        "actor/con.json",
        "actor/trailing.",
        "actor/trailing ",
    ],
)
def test_member_paths_are_canonical_lowercase_ascii_posix(
    unsafe_path: str,
    minimal_character_manifest: Mapping[str, object],
    minimal_character_files: Mapping[str, bytes],
) -> None:
    files = dict(minimal_character_files)
    actor_bytes = files.pop("actor/actor.json")
    files[unsafe_path] = actor_bytes
    manifest = copy.deepcopy(dict(minimal_character_manifest))
    actor = dict(manifest["actor"])  # type: ignore[arg-type]
    actor["payload"] = unsafe_path
    manifest["actor"] = actor
    manifest = _rebuild(manifest, files)

    _assert_category("actor_pack_path_invalid", manifest, files)


def test_declared_files_must_equal_supplied_files(
    minimal_character_manifest: Mapping[str, object],
    minimal_character_files: Mapping[str, bytes],
) -> None:
    files = dict(minimal_character_files)
    files["licenses/extra.txt"] = b"not declared"

    _assert_category("actor_pack_inventory_invalid", minimal_character_manifest, files)


def test_inventory_sha_and_size_are_rechecked(
    minimal_character_manifest: Mapping[str, object],
    minimal_character_files: Mapping[str, bytes],
) -> None:
    files = dict(minimal_character_files)
    files["actor/portrait.png"] += b"tampered"

    _assert_category("actor_pack_inventory_mismatch", minimal_character_manifest, files)


@pytest.mark.parametrize(
    "forbidden", ["id", "local_id", "record_id", "chat_id", "api_key", "path"]
)
def test_actor_payload_rejects_local_private_and_external_fields(
    forbidden: str,
    minimal_character_manifest: Mapping[str, object],
    minimal_character_files: Mapping[str, bytes],
) -> None:
    files = dict(minimal_character_files)
    actor = {
        "schema": "tldw.actor/v1",
        "actor_kind": "character",
        "portable_uuid": PORTABLE_UUID,
        "data": {"name": "Guide", forbidden: "private"},
    }
    files["actor/actor.json"] = canonical_json(actor)
    manifest = _rebuild(minimal_character_manifest, files)

    _assert_category("actor_pack_actor_invalid", manifest, files)


@pytest.mark.parametrize(
    ("actor_kind", "actor_data"),
    [
        ("persona", {"name": "Guide", "mode": "invalid"}),
        ("persona", {"name": "Guide", "is_active": {"yes": True}}),
        ("persona", {"name": "Guide", "voice_defaults": "speaker"}),
        ("character", {"name": "Guide", "tags": {"not": "a list"}}),
        ("character", {"name": "Guide", "character_book": {"entries": []}}),
        (
            "character",
            {
                "name": "Guide",
                "extensions": {"actor_pack_persona_portrait_owner": "spoofed"},
            },
        ),
    ],
)
def test_actor_payload_rejects_values_outside_local_mutation_contracts(
    actor_kind: str, actor_data: dict[str, object]
) -> None:
    payload = canonical_json(
        {
            "schema": "tldw.actor/v1",
            "actor_kind": actor_kind,
            "portable_uuid": PORTABLE_UUID,
            "data": actor_data,
        }
    )

    with pytest.raises(ActorPackValidationError) as caught:
        actor_pack_contracts.validate_actor_payload(
            payload,
            actor_kind=actor_kind,
            portable_uuid=PORTABLE_UUID,
        )

    assert caught.value.category == "actor_pack_actor_invalid"


def test_character_cannot_declare_persona_runtime_section(
    minimal_character_manifest: Mapping[str, object],
    minimal_character_files: Mapping[str, bytes],
) -> None:
    files = dict(minimal_character_files)
    files["persona-runtime/manifest.json"] = canonical_json(
        {"schema": "tldw.persona-visual/v1"}
    )
    manifest = copy.deepcopy(dict(minimal_character_manifest))
    manifest["sections"] = [
        {
            "kind": "persona-runtime",
            "manifest": "persona-runtime/manifest.json",
        }
    ]
    manifest = _rebuild(manifest, files)

    _assert_category("actor_pack_section_invalid", manifest, files)


def test_actor_pack_manifest_cannot_inventory_itself(
    minimal_character_manifest: Mapping[str, object],
    minimal_character_files: Mapping[str, bytes],
) -> None:
    manifest = copy.deepcopy(dict(minimal_character_manifest))
    inventory = list(manifest["files"])  # type: ignore[arg-type]
    inventory.append(
        {
            "path": "actor-pack.json",
            "bytes": 2,
            "sha256": hashlib.sha256(b"{}").hexdigest(),
        }
    )
    manifest["files"] = inventory
    manifest = with_content_digest(manifest)

    _assert_category("actor_pack_inventory_invalid", manifest, minimal_character_files)


def test_canonical_json_bytes_have_one_exact_utf8_representation() -> None:
    assert canonical_json_bytes({"z": 1, "é": "雪", "a": [True, None]}) == (
        b'{"a":[true,null],"z":1,"\xc3\xa9":"\xe9\x9b\xaa"}'
    )


def test_content_digest_has_an_independent_literal_oracle(
    minimal_character_manifest: Mapping[str, object],
) -> None:
    assert actor_pack_content_digest(minimal_character_manifest) == (
        "e39d6c3983e7f55c7361327ffaeabb8122df4ea5bfab0f1d55e057276887a252"
    )


def test_deterministic_zip_metadata_contract_is_frozen_without_a_writer() -> None:
    assert actor_pack_contracts.ZIP_COMPRESSION == 0
    assert actor_pack_contracts.ZIP_TIMESTAMP == (1980, 1, 1, 0, 0, 0)
    assert actor_pack_contracts.ZIP_CREATE_SYSTEM == 3
    assert actor_pack_contracts.ZIP_GENERAL_PURPOSE_FLAGS == 0
    assert actor_pack_contracts.ZIP_EXTERNAL_ATTR == 0o100644 << 16
    assert actor_pack_contracts.canonical_member_order(
        ["licenses/z.txt", "actor/portrait.png", "actor-pack.json", "actor/actor.json"]
    ) == (
        "actor-pack.json",
        "actor/actor.json",
        "actor/portrait.png",
        "licenses/z.txt",
    )


def test_character_projection_strips_only_known_local_authority_fields() -> None:
    payload = actor_pack_contracts.canonicalize_actor_payload(
        "character",
        PORTABLE_UUID,
        {
            "id": 42,
            "record_id": "local:42",
            "client_id": "private-client",
            "version": 7,
            "deleted": False,
            "created_at": "private",
            "last_modified": "private",
            "image": b"portrait-is-separate",
            "name": "Guide",
            "description": "A portable guide",
        },
    )

    assert payload == canonical_json(
        {
            "schema": "tldw.actor/v1",
            "actor_kind": "character",
            "portable_uuid": PORTABLE_UUID,
            "data": {"description": "A portable guide", "name": "Guide"},
        }
    )
    assert b"private" not in payload


def test_persona_projection_strips_linked_local_portrait_id() -> None:
    payload = actor_pack_contracts.canonicalize_actor_payload(
        "persona",
        PORTABLE_UUID,
        {
            "id": "local-persona-private",
            "record_id": "local:persona_profile:private",
            "character_card_id": 81,
            "version": 3,
            "deleted": False,
            "backend": "local",
            "created_at": "private",
            "last_modified": "private",
            "name": "Guide",
            "mode": "session_scoped",
            "is_active": True,
        },
    )

    decoded = canonical_json(
        {
            "schema": "tldw.actor/v1",
            "actor_kind": "persona",
            "portable_uuid": PORTABLE_UUID,
            "data": {
                "is_active": True,
                "mode": "session_scoped",
                "name": "Guide",
            },
        }
    )
    assert payload == decoded
    assert b"character_card_id" not in payload


@pytest.mark.parametrize(
    "private_key",
    ["api_key", "credentials", "chats", "provider_settings", "avatar_url"],
)
def test_projection_rejects_unknown_private_or_external_fields(
    private_key: str,
) -> None:
    with pytest.raises(ActorPackValidationError) as caught:
        actor_pack_contracts.canonicalize_actor_payload(
            "character", PORTABLE_UUID, {"name": "Guide", private_key: "private"}
        )
    assert caught.value.category == "actor_pack_actor_invalid"


def test_build_file_inventory_is_sorted_and_rechecks_exact_bytes(
    minimal_character_files: Mapping[str, bytes],
) -> None:
    inventory = actor_pack_contracts.build_file_inventory(
        dict(reversed(tuple(minimal_character_files.items())))
    )
    assert tuple(item.path for item in inventory) == (
        "actor/actor.json",
        "actor/portrait.png",
    )
    assert (
        inventory[0].sha256
        == hashlib.sha256(minimal_character_files["actor/actor.json"]).hexdigest()
    )


@pytest.mark.parametrize(
    "hostile",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": "x" * 4097},
    ],
)
def test_canonical_json_rejects_nonfinite_and_oversized_scalars(
    hostile: object,
) -> None:
    with pytest.raises(ActorPackValidationError) as caught:
        canonical_json_bytes(hostile)
    assert caught.value.category == "actor_pack_json_invalid"


def test_canonical_json_rejects_excessive_depth_and_nodes() -> None:
    deep: object = "leaf"
    for _ in range(65):
        deep = [deep]
    with pytest.raises(ActorPackValidationError, match="actor_pack_json_invalid"):
        canonical_json_bytes(deep)

    with pytest.raises(ActorPackValidationError, match="actor_pack_json_invalid"):
        canonical_json_bytes([None] * 20_001)


def test_portrait_must_decode_and_remain_within_pixel_limits(
    minimal_character_manifest: Mapping[str, object],
    minimal_character_files: Mapping[str, bytes],
) -> None:
    files = dict(minimal_character_files)
    files["actor/portrait.png"] = b"\x89PNG\r\n\x1a\nnot-a-decoded-image"
    manifest = _rebuild(minimal_character_manifest, files)

    _assert_category("actor_pack_portrait_invalid", manifest, files)


def test_persona_can_declare_both_typed_visual_sections(
    minimal_character_manifest: Mapping[str, object],
    minimal_character_files: Mapping[str, bytes],
) -> None:
    files = dict(minimal_character_files)
    persona_payload = actor_pack_contracts.canonicalize_actor_payload(
        "persona", PORTABLE_UUID, {"name": "Guide", "mode": "session_scoped"}
    )
    files["actor/actor.json"] = persona_payload
    files["shared-visual-identity/manifest.json"] = canonical_json(
        {"schema": "visual-identity/v1"}
    )
    files["persona-runtime/manifest.json"] = canonical_json(
        {"schema": "persona-visual/v1"}
    )
    manifest = copy.deepcopy(dict(minimal_character_manifest))
    manifest["actor"] = {
        **dict(manifest["actor"]),  # type: ignore[arg-type]
        "kind": "persona",
    }
    manifest["sections"] = [
        {
            "kind": "shared-visual-identity",
            "manifest": "shared-visual-identity/manifest.json",
        },
        {
            "kind": "persona-runtime",
            "manifest": "persona-runtime/manifest.json",
        },
    ]
    manifest = _rebuild(manifest, files)

    document = validate_actor_pack_document(manifest, files)
    assert document.actor_kind == "persona"
    assert tuple(section.kind for section in document.sections) == (
        "shared-visual-identity",
        "persona-runtime",
    )
