from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tldw_chatbook.Chat import citation_source_locators as locator_models
from tldw_chatbook.Chat.citation_source_locators import (
    AUTHORITY_IDS_PER_READ_AUTHORIZATION_MAX,
    EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX,
    LOCATOR_ENVELOPE_JSON_BYTES_MAX,
    RUNTIME_SOURCE_KIND_TO_CANONICAL_V1,
    SOURCE_INVENTORY_BY_SCOPE_V1,
    SOURCE_INVENTORY_V1,
    AuthorityScope,
    CanonicalSourceKind,
    CharacterCardLocatorPayloadV1,
    ChatHistoryLocatorPayloadV1,
    CitationReadAuthorization,
    ClaimLocatorPayloadV1,
    CurrentAuthorityLocatorLookup,
    DictionaryLocatorPayloadV1,
    InertLocatorCandidate,
    KanbanLocatorPayloadV1,
    LocatorBindingState,
    MediaLocatorPayloadV1,
    NoteLocatorPayloadV1,
    PromptLocatorPayloadV1,
    RebindAction,
    RebindDecision,
    SQLLocatorPayloadV1,
    SourceCapability,
    SourceCapabilityPolicy,
    SourceInventoryEntry,
    SourceLocatorEnvelope,
    SourceLocatorPayloadV1,
    WebContentLocatorPayloadV1,
    WorldBookLocatorPayloadV1,
    canonical_locator_json,
    parse_inert_locator_candidate,
    rebind_inert_locator,
    validate_native_locator,
)
from tldw_chatbook.Chat.citation_trace_models import EvidenceStorageMode


NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)
FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "rag_citation_provenance"
    / "source_inventory_v1.json"
)

PAYLOAD_MODEL_BY_KIND = {
    CanonicalSourceKind.MEDIA_DB: MediaLocatorPayloadV1,
    CanonicalSourceKind.NOTES: NoteLocatorPayloadV1,
    CanonicalSourceKind.CHAT_HISTORY: ChatHistoryLocatorPayloadV1,
    CanonicalSourceKind.CHARACTER_CARDS: CharacterCardLocatorPayloadV1,
    CanonicalSourceKind.WEB_CONTENT: WebContentLocatorPayloadV1,
    CanonicalSourceKind.PROMPTS: PromptLocatorPayloadV1,
    CanonicalSourceKind.WORLD_BOOKS: WorldBookLocatorPayloadV1,
    CanonicalSourceKind.DICTIONARIES: DictionaryLocatorPayloadV1,
    CanonicalSourceKind.KANBAN: KanbanLocatorPayloadV1,
    CanonicalSourceKind.SQL: SQLLocatorPayloadV1,
    CanonicalSourceKind.CLAIMS: ClaimLocatorPayloadV1,
}


def _payload(
    source_kind: CanonicalSourceKind,
    **values: object,
) -> SourceLocatorPayloadV1:
    return PAYLOAD_MODEL_BY_KIND[source_kind](**values)


def _local_locator(
    *,
    source_kind: CanonicalSourceKind = CanonicalSourceKind.MEDIA_DB,
    payload: SourceLocatorPayloadV1 | None = None,
) -> SourceLocatorEnvelope:
    return SourceLocatorEnvelope(
        binding_state=LocatorBindingState.NATIVE,
        source_kind=source_kind,
        authority_scope=AuthorityScope.LOCAL_PROFILE,
        authority_id="local-authority",
        governance_scope_id="profile-a",
        profile_id="profile-a",
        resolver_payload=payload or _payload(source_kind, item_id="item-1"),
    )


def _server_locator(
    source_kind: CanonicalSourceKind,
    *,
    payload: SourceLocatorPayloadV1 | None = None,
) -> SourceLocatorEnvelope:
    return SourceLocatorEnvelope(
        binding_state=LocatorBindingState.NATIVE,
        source_kind=source_kind,
        authority_scope=AuthorityScope.AUTHENTICATED_TENANT,
        authority_id="server-authority",
        governance_scope_id="tenant-a",
        authenticated_tenant_id="tenant-a",
        resolver_payload=payload or _payload(source_kind, item_id="item-1"),
    )


def _authorization(
    *,
    local: bool = True,
    authority_id: str | None = None,
    **capabilities: bool,
) -> CitationReadAuthorization:
    values = {
        capability.value: capabilities.get(capability.value, True)
        for capability in SourceCapability
    }
    if local:
        return CitationReadAuthorization(
            authority_scope=AuthorityScope.LOCAL_PROFILE,
            profile_id="profile-a",
            governance_scope_id="profile-a",
            allowlisted_authority_ids=(authority_id or "local-authority",),
            **values,
        )
    return CitationReadAuthorization(
        authority_scope=AuthorityScope.AUTHENTICATED_TENANT,
        authenticated_tenant_id="tenant-a",
        governance_scope_id="tenant-a",
        allowlisted_authority_ids=(authority_id or "server-authority",),
        **values,
    )


def _policy(
    *,
    storage_mode: EvidenceStorageMode = EvidenceStorageMode.EMBEDDED,
    **capabilities: bool,
) -> SourceCapabilityPolicy:
    values = {
        capability.value: capabilities.get(capability.value, True)
        for capability in SourceCapability
    }
    return SourceCapabilityPolicy(storage_mode=storage_mode, **values)


def test_locator_contracts_are_strict_frozen_versioned_and_deterministic() -> None:
    locator = _local_locator()
    encoded = canonical_locator_json(locator)

    assert locator.schema_version == 1
    assert locator.resolver_payload_version == 1
    assert locator.resolver_payload.schema_version == 1
    assert SourceLocatorEnvelope.model_validate_json(encoded) == locator
    assert (
        canonical_locator_json(SourceLocatorEnvelope.model_validate_json(encoded))
        == encoded
    )
    assert encoded == json.dumps(
        locator.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SourceLocatorEnvelope(**{**locator.model_dump(), "handler": "unsafe"})
    with pytest.raises(ValidationError, match="Input should be 1"):
        SourceLocatorEnvelope(**{**locator.model_dump(), "schema_version": 2})
    with pytest.raises(ValidationError, match="Input should be 1"):
        SourceLocatorEnvelope(**{**locator.model_dump(), "resolver_payload_version": 2})
    with pytest.raises(ValidationError, match="Input should be 1"):
        MediaLocatorPayloadV1(item_id="item", schema_version=2)
    with pytest.raises(ValidationError):
        SourceLocatorEnvelope(
            **{
                **locator.model_dump(),
                "binding_state": LocatorBindingState.INERT_IMPORTED,
            }
        )
    with pytest.raises(ValidationError, match="frozen"):
        locator.authority_id = "changed"  # type: ignore[misc]


def test_inventory_and_policy_contracts_reject_unknown_versions_and_selectors() -> None:
    entry = SOURCE_INVENTORY_V1[0]
    encoded = entry.model_dump_json()
    assert SourceInventoryEntry.model_validate_json(encoded) == entry
    assert (
        SourceInventoryEntry.model_validate_json(encoded).model_dump_json() == encoded
    )

    with pytest.raises(ValidationError, match="Input should be 1"):
        SourceInventoryEntry(**{**entry.model_dump(), "schema_version": 2})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SourceInventoryEntry(**{**entry.model_dump(), "resolver_class": "pkg.R"})
    with pytest.raises(ValidationError, match="Input should be 1"):
        SourceCapabilityPolicy(
            **{**entry.default_policy.model_dump(), "schema_version": 2}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("class", "pkg.Resolver"),
        ("module", "pkg.resolvers"),
        ("command", "open /tmp/x"),
        ("path", "/tmp/private"),
        ("absolute_path", "/etc/passwd"),
        ("handler", "subprocess"),
        ("url", "https://attacker.invalid"),
        ("fetch_url", "http://127.0.0.1/private"),
    ),
)
def test_locator_payload_rejects_data_selected_code_paths_and_urls(
    field: str, value: str
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MediaLocatorPayloadV1(item_id="item", **{field: value})


def test_unknown_source_kind_and_hostile_constructed_instance_fail_closed() -> None:
    locator = _local_locator()
    with pytest.raises(ValidationError):
        SourceLocatorEnvelope(**{**locator.model_dump(), "source_kind": "plugin"})

    hostile = SourceLocatorEnvelope.model_construct(
        **{**locator.model_dump(), "source_kind": "plugin"}
    )
    with pytest.raises(ValidationError):
        validate_native_locator(
            hostile,
            _authorization(),
            _policy(open_external=False),
        )
    with pytest.raises(ValidationError, match="payload source kind"):
        SourceLocatorEnvelope(
            **{
                **locator.model_dump(),
                "resolver_payload": SQLLocatorPayloadV1(item_id="result-1"),
            }
        )
    hostile_wrong_payload = SourceLocatorEnvelope.model_construct(
        **{
            **locator.model_dump(),
            "resolver_payload": SQLLocatorPayloadV1(item_id="result-1"),
        }
    )
    with pytest.raises(ValidationError):
        validate_native_locator(
            hostile_wrong_payload,
            _authorization(),
            _policy(open_external=False),
        )
    hostile_payload = {
        **locator.model_dump(mode="json"),
        "resolver_payload": {
            "schema_version": 1,
            "source_kind": "plugin",
            "item_id": "item-1",
        },
    }
    with pytest.raises(ValidationError):
        SourceLocatorEnvelope.model_validate_json(json.dumps(hostile_payload))


@pytest.mark.parametrize(
    ("source_kind", "field", "value"),
    (
        (CanonicalSourceKind.SQL, "message_id", "message-1"),
        (CanonicalSourceKind.PROMPTS, "page_number", 1),
        (CanonicalSourceKind.CHAT_HISTORY, "entry_id", "entry-1"),
        (CanonicalSourceKind.NOTES, "start_seconds", 1.0),
        (CanonicalSourceKind.MEDIA_DB, "message_id", "message-1"),
    ),
)
def test_source_kinds_reject_unrelated_typed_location_hints(
    source_kind: CanonicalSourceKind,
    field: str,
    value: str | int | float,
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _payload(source_kind, item_id="item", **{field: value})


@pytest.mark.parametrize(
    "relative_path",
    (
        "/absolute/note.md",
        "../note.md",
        "notes/../note.md",
        r"C:\notes\note.md",
        r"\\server\share\note.md",
        "~/note.md",
        "notes\x00/note.md",
        "notes/\nsecret.md",
        "notes//note.md",
        "notes/./note.md",
        "file://notes/note.md",
        "notes/note.md:stream",
    ),
)
def test_file_backed_notes_reject_unsafe_relative_path_semantics(
    relative_path: str,
) -> None:
    with pytest.raises(ValidationError):
        NoteLocatorPayloadV1(
            item_id="note-1",
            source_root_id="notes-root",
            relative_path=relative_path,
        )


@pytest.mark.parametrize(
    "source_root_id",
    ("/tmp", r"C:\notes", r"\\server\share", "../notes", "notes/root", "notes\x00"),
)
def test_note_source_root_is_a_bounded_opaque_id_not_a_path(
    source_root_id: str,
) -> None:
    with pytest.raises(ValidationError):
        NoteLocatorPayloadV1(
            item_id="note-1",
            source_root_id=source_root_id,
            relative_path="folder/note.md",
        )


def test_file_backed_note_requires_root_and_path_as_a_pair() -> None:
    for update in (
        {"source_root_id": "notes-root"},
        {"relative_path": "folder/note.md"},
    ):
        with pytest.raises(ValidationError, match="together"):
            NoteLocatorPayloadV1(item_id="note-1", **update)

    locator = _local_locator(
        source_kind=CanonicalSourceKind.NOTES,
        payload=NoteLocatorPayloadV1(
            item_id="note-1",
            source_root_id="notes-root",
            relative_path="folder/note.md",
        ),
    )
    assert locator.resolver_payload.relative_path == "folder/note.md"
    with pytest.raises(ValidationError, match="server note"):
        _server_locator(
            CanonicalSourceKind.NOTES,
            payload=NoteLocatorPayloadV1(
                item_id="note-1",
                source_root_id="notes-root",
                relative_path="folder/note.md",
            ),
        )


def test_locator_envelope_accepts_exact_16_kib_and_rejects_one_byte_over() -> None:
    base_payload = {
        "schema_version": 1,
        "source_kind": "notes",
        "item_id": "note-1",
        "source_root_id": "root",
        "relative_path": "",
        "chunk_id": None,
        "section_ordinal": None,
    }
    base_envelope = {
        "schema_version": 1,
        "binding_state": "native",
        "source_kind": "notes",
        "authority_scope": "local_profile",
        "authority_id": "local-authority",
        "governance_scope_id": "profile-a",
        "profile_id": "profile-a",
        "authenticated_tenant_id": None,
        "resolver_payload_version": 1,
        "resolver_payload": base_payload,
    }
    base_size = len(
        json.dumps(
            base_envelope,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    path_length = LOCATOR_ENVELOPE_JSON_BYTES_MAX - base_size
    assert path_length > 0

    exact = SourceLocatorEnvelope.model_validate_json(
        json.dumps(
            {
                **base_envelope,
                "resolver_payload": {
                    **base_payload,
                    "relative_path": "a" * path_length,
                },
            }
        )
    )
    assert len(canonical_locator_json(exact).encode("utf-8")) == (
        LOCATOR_ENVELOPE_JSON_BYTES_MAX
    )

    with pytest.raises(ValidationError, match="locator envelope"):
        SourceLocatorEnvelope.model_validate_json(
            json.dumps(
                {
                    **base_envelope,
                    "resolver_payload": {
                        **base_payload,
                        "relative_path": "a" * (path_length + 1),
                    },
                }
            )
        )


def test_authorization_is_strict_frozen_scope_bound_and_count_bounded() -> None:
    authorization = _authorization()
    exact_authorities = tuple(
        f"a-{index}" for index in range(AUTHORITY_IDS_PER_READ_AUTHORIZATION_MAX)
    )
    assert (
        CitationReadAuthorization(
            **{
                **authorization.model_dump(),
                "allowlisted_authority_ids": exact_authorities,
            }
        ).allowlisted_authority_ids
        == exact_authorities
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CitationReadAuthorization(**{**authorization.model_dump(), "authorized": True})
    with pytest.raises(ValidationError, match="frozen"):
        authorization.view_snapshot = False  # type: ignore[misc]
    with pytest.raises(ValidationError, match="exactly one"):
        CitationReadAuthorization(
            **{
                **authorization.model_dump(),
                "authenticated_tenant_id": "tenant-a",
            }
        )
    with pytest.raises(ValidationError, match="governance_scope_id"):
        CitationReadAuthorization(
            **{
                **authorization.model_dump(),
                "governance_scope_id": "profile-b",
            }
        )
    with pytest.raises(ValidationError):
        CitationReadAuthorization(
            **{
                **authorization.model_dump(),
                "allowlisted_authority_ids": tuple(
                    f"a-{index}"
                    for index in range(AUTHORITY_IDS_PER_READ_AUTHORIZATION_MAX + 1)
                ),
            }
        )
    hostile = CitationReadAuthorization.model_construct(
        **{
            **authorization.model_dump(),
            "allowlisted_authority_ids": tuple(
                f"a-{index}"
                for index in range(AUTHORITY_IDS_PER_READ_AUTHORIZATION_MAX + 1)
            ),
        }
    )
    with pytest.raises(ValidationError):
        validate_native_locator(
            _local_locator(),
            hostile,
            _policy(open_external=False),
        )

    exact_id = "é" * (EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX // 2)
    assert _authorization(authority_id=exact_id).allowlisted_authority_ids == (
        exact_id,
    )
    with pytest.raises(ValidationError, match="UTF-8 bytes"):
        _authorization(authority_id=f"{exact_id}a")


@pytest.mark.parametrize(
    "changed",
    ("authority", "profile", "governance", "tenant"),
)
def test_native_validation_rejects_authority_and_scope_mismatches(
    changed: str,
) -> None:
    locator = _local_locator()
    authorization = _authorization()
    if changed == "authority":
        authorization = _authorization(authority_id="another-authority")
    elif changed == "profile":
        locator = SourceLocatorEnvelope(
            **{
                **locator.model_dump(),
                "profile_id": "profile-b",
                "governance_scope_id": "profile-b",
            }
        )
    elif changed == "governance":
        hostile = locator.model_construct(
            **{**locator.model_dump(), "governance_scope_id": "profile-b"}
        )
        locator = hostile
    else:
        locator = _server_locator(CanonicalSourceKind.PROMPTS)
        authorization = CitationReadAuthorization(
            **{
                **_authorization(local=False).model_dump(),
                "authenticated_tenant_id": "tenant-b",
                "governance_scope_id": "tenant-b",
            }
        )

    with pytest.raises((ValidationError, ValueError)):
        validate_native_locator(locator, authorization, _policy(open_external=False))


def test_policy_storage_and_every_capability_are_independent() -> None:
    base = {capability.value: False for capability in SourceCapability}
    for storage_mode in EvidenceStorageMode:
        policy = SourceCapabilityPolicy(storage_mode=storage_mode, **base)
        assert policy.storage_mode is storage_mode

    for enabled in SourceCapability:
        policy = SourceCapabilityPolicy(
            storage_mode=EvidenceStorageMode.EMBEDDED,
            **{
                capability.value: capability is enabled
                for capability in SourceCapability
            },
        )
        assert policy.permits(enabled)
        assert all(
            not policy.permits(other)
            for other in SourceCapability
            if other is not enabled
        )


@pytest.mark.parametrize("capability", tuple(SourceCapability))
def test_native_validation_requires_each_policy_and_authorization_capability(
    capability: SourceCapability,
) -> None:
    locator = _server_locator(CanonicalSourceKind.WEB_CONTENT)
    policy_values = {item.value: item is not capability for item in SourceCapability}
    authorization_values = {
        item.value: item is not capability for item in SourceCapability
    }

    with pytest.raises(ValueError, match=capability.value):
        validate_native_locator(
            locator,
            _authorization(local=False),
            _policy(
                storage_mode=EvidenceStorageMode.SERVER_REFERENCE,
                **policy_values,
            ),
            required_capability=capability,
        )
    with pytest.raises(ValueError, match=capability.value):
        validate_native_locator(
            locator,
            _authorization(local=False, **authorization_values),
            _policy(storage_mode=EvidenceStorageMode.SERVER_REFERENCE),
            required_capability=capability,
        )


def test_missing_view_snapshot_capability_blocks_governed_hydration() -> None:
    with pytest.raises(ValueError, match="view_snapshot"):
        validate_native_locator(
            _local_locator(),
            _authorization(view_snapshot=False),
            _policy(open_external=False),
        )


def test_sql_is_always_snapshot_only_and_cannot_replay_or_open_paths() -> None:
    sql = _server_locator(CanonicalSourceKind.SQL)
    snapshot_policy = SourceCapabilityPolicy(
        storage_mode=EvidenceStorageMode.SERVER_REFERENCE,
        view_snapshot=True,
        view_source_identity=True,
        export=True,
    )
    assert (
        validate_native_locator(
            sql,
            _authorization(local=False),
            snapshot_policy,
        )
        == sql
    )

    for capability in (
        SourceCapability.RESOLVE_CURRENT,
        SourceCapability.OPEN_NATIVE,
        SourceCapability.OPEN_EXTERNAL,
        SourceCapability.COMPARE,
        SourceCapability.REFRESH_OBSERVATION,
    ):
        with pytest.raises(ValueError, match=capability.value):
            validate_native_locator(
                sql,
                _authorization(local=False),
                SourceCapabilityPolicy(
                    **{
                        **snapshot_policy.model_dump(),
                        capability.value: True,
                    }
                ),
                required_capability=capability,
            )

    with pytest.raises(ValidationError):
        SQLLocatorPayloadV1(item_id="sql-1", path="/var/lib/database")
    with pytest.raises(ValidationError):
        SQLLocatorPayloadV1(item_id="sql-1", command="SELECT * FROM secrets")


def test_claims_open_only_through_authorized_parent_lineage() -> None:
    claim_policy = SourceCapabilityPolicy(
        storage_mode=EvidenceStorageMode.SERVER_REFERENCE,
        view_snapshot=True,
        view_source_identity=True,
        resolve_current=True,
        open_native=True,
    )
    parent_policy = _policy(
        storage_mode=EvidenceStorageMode.SERVER_REFERENCE,
        open_external=False,
    )
    snapshot_only = _server_locator(CanonicalSourceKind.CLAIMS)
    with pytest.raises(ValidationError, match="appear together"):
        ClaimLocatorPayloadV1(item_id="claim-1", parent_media_id="media-1")
    with pytest.raises(ValueError, match="authorized parent"):
        validate_native_locator(
            snapshot_only,
            _authorization(local=False),
            claim_policy,
            required_capability=SourceCapability.OPEN_NATIVE,
        )

    claim = _server_locator(
        CanonicalSourceKind.CLAIMS,
        payload=ClaimLocatorPayloadV1(
            item_id="claim-1",
            parent_media_id="media-1",
            parent_chunk_id="chunk-1",
        ),
    )
    parent = _server_locator(
        CanonicalSourceKind.MEDIA_DB,
        payload=MediaLocatorPayloadV1(
            item_id="media-1",
            chunk_id="chunk-1",
        ),
    )
    with pytest.raises(ValueError, match="separately validated parent"):
        validate_native_locator(
            claim,
            _authorization(local=False),
            claim_policy,
            required_capability=SourceCapability.OPEN_NATIVE,
        )

    assert (
        validate_native_locator(
            claim,
            _authorization(local=False),
            claim_policy,
            required_capability=SourceCapability.OPEN_NATIVE,
            parent_locator=parent,
            parent_policy=parent_policy,
        )
        == claim
    )

    wrong_identity = _server_locator(
        CanonicalSourceKind.MEDIA_DB,
        payload=MediaLocatorPayloadV1(item_id="media-2", chunk_id="chunk-1"),
    )
    with pytest.raises(ValueError, match="parent lineage"):
        validate_native_locator(
            claim,
            _authorization(local=False),
            claim_policy,
            required_capability=SourceCapability.OPEN_NATIVE,
            parent_locator=wrong_identity,
            parent_policy=parent_policy,
        )
    assert (
        validate_native_locator(
            claim,
            _authorization(local=False),
            claim_policy,
            required_capability=SourceCapability.RESOLVE_CURRENT,
            parent_locator=parent,
            parent_policy=parent_policy,
        )
        == claim
    )


@pytest.mark.parametrize(
    "mismatch",
    ("authority", "tenant", "profile", "governance", "media", "chunk"),
)
def test_claims_reject_mismatched_parent_authority_scope_and_identity(
    mismatch: str,
) -> None:
    claim = _server_locator(
        CanonicalSourceKind.CLAIMS,
        payload=ClaimLocatorPayloadV1(
            item_id="claim-1",
            parent_media_id="media-1",
            parent_chunk_id="chunk-1",
        ),
    )
    parent = _server_locator(
        CanonicalSourceKind.MEDIA_DB,
        payload=MediaLocatorPayloadV1(item_id="media-1", chunk_id="chunk-1"),
    )
    authorization = _authorization(local=False)
    if mismatch == "authority":
        parent = SourceLocatorEnvelope(
            **{**parent.model_dump(), "authority_id": "other-authority"}
        )
        authorization = CitationReadAuthorization(
            **{
                **authorization.model_dump(),
                "allowlisted_authority_ids": (
                    "server-authority",
                    "other-authority",
                ),
            }
        )
    elif mismatch == "tenant":
        parent = SourceLocatorEnvelope(
            **{
                **parent.model_dump(),
                "authenticated_tenant_id": "tenant-b",
                "governance_scope_id": "tenant-b",
            }
        )
    elif mismatch == "profile":
        parent = _local_locator(
            source_kind=CanonicalSourceKind.MEDIA_DB,
            payload=MediaLocatorPayloadV1(item_id="media-1", chunk_id="chunk-1"),
        )
    elif mismatch == "governance":
        parent = SourceLocatorEnvelope.model_construct(
            **{**parent.model_dump(), "governance_scope_id": "tenant-b"}
        )
    elif mismatch == "media":
        parent = _server_locator(
            CanonicalSourceKind.MEDIA_DB,
            payload=MediaLocatorPayloadV1(item_id="media-2", chunk_id="chunk-1"),
        )
    else:
        parent = _server_locator(
            CanonicalSourceKind.MEDIA_DB,
            payload=MediaLocatorPayloadV1(item_id="media-1", chunk_id="chunk-2"),
        )

    with pytest.raises((ValidationError, ValueError)):
        validate_native_locator(
            claim,
            authorization,
            SourceCapabilityPolicy(
                storage_mode=EvidenceStorageMode.SERVER_REFERENCE,
                view_snapshot=True,
                resolve_current=True,
                open_native=True,
            ),
            required_capability=SourceCapability.OPEN_NATIVE,
            parent_locator=parent,
            parent_policy=_policy(
                storage_mode=EvidenceStorageMode.SERVER_REFERENCE,
                open_external=False,
            ),
        )


def test_runtime_mapping_and_static_inventory_equal_committed_contract_fixture() -> (
    None
):
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    runtime = {
        "inventory_version": 1,
        "runtime_mapping": dict(RUNTIME_SOURCE_KIND_TO_CANONICAL_V1),
        "entries": [entry.model_dump(mode="json") for entry in SOURCE_INVENTORY_V1],
    }

    assert runtime == fixture
    assert runtime["runtime_mapping"] == {
        "media": "media_db",
        "note": "notes",
        "conversation": "chat_history",
    }
    assert {entry.source_kind for entry in SOURCE_INVENTORY_V1} == set(
        CanonicalSourceKind
    )
    inventory_keys = {
        (entry.source_kind, entry.authority_scope) for entry in SOURCE_INVENTORY_V1
    }
    assert len(inventory_keys) == len(SOURCE_INVENTORY_V1)
    assert len(SOURCE_INVENTORY_V1) == len(CanonicalSourceKind) + 3
    assert set(SOURCE_INVENTORY_BY_SCOPE_V1) == inventory_keys
    with pytest.raises(TypeError):
        RUNTIME_SOURCE_KIND_TO_CANONICAL_V1["plugin"] = "sql"  # type: ignore[index]
    with pytest.raises(ValidationError, match="frozen"):
        SOURCE_INVENTORY_V1[0].locator_version = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        SOURCE_INVENTORY_BY_SCOPE_V1[
            (CanonicalSourceKind.SQL, AuthorityScope.LOCAL_PROFILE)
        ] = SOURCE_INVENTORY_V1[0]  # type: ignore[index]


@pytest.mark.parametrize(
    "source_kind",
    (
        CanonicalSourceKind.MEDIA_DB,
        CanonicalSourceKind.NOTES,
        CanonicalSourceKind.CHAT_HISTORY,
    ),
)
def test_shared_source_kinds_have_local_and_server_authority_variants(
    source_kind: CanonicalSourceKind,
) -> None:
    local = _local_locator(source_kind=source_kind)
    server = _server_locator(source_kind)
    entries = [
        entry for entry in SOURCE_INVENTORY_V1 if entry.source_kind is source_kind
    ]

    assert {entry.authority_scope for entry in entries} == {
        AuthorityScope.LOCAL_PROFILE,
        AuthorityScope.AUTHENTICATED_TENANT,
    }
    assert (
        validate_native_locator(
            local,
            _authorization(),
            _policy(open_external=False),
        )
        == local
    )
    assert (
        validate_native_locator(
            server,
            _authorization(local=False),
            _policy(
                storage_mode=EvidenceStorageMode.SERVER_REFERENCE,
                open_external=False,
            ),
        )
        == server
    )


def test_every_inventory_variant_has_a_matching_strict_payload_and_scope() -> None:
    for entry in SOURCE_INVENTORY_V1:
        local = entry.authority_scope is AuthorityScope.LOCAL_PROFILE
        locator = (
            _local_locator(source_kind=entry.source_kind)
            if local
            else _server_locator(entry.source_kind)
        )
        assert (
            validate_native_locator(
                locator,
                _authorization(local=local),
                entry.default_policy,
            )
            == locator
        )


def test_imported_and_legacy_candidates_stay_inert_after_parsing() -> None:
    hostile_raw = {
        "schema_version": 1,
        "class": "pkg.DynamicResolver",
        "command": "open /etc/passwd",
        "url": "http://127.0.0.1/private",
    }
    imported = parse_inert_locator_candidate(
        hostile_raw,
        candidate_id="candidate-imported",
        binding_state=LocatorBindingState.INERT_IMPORTED,
    )
    legacy = parse_inert_locator_candidate(
        hostile_raw,
        candidate_id="candidate-legacy",
        binding_state=LocatorBindingState.INERT_LEGACY,
    )

    assert imported.binding_state is LocatorBindingState.INERT_IMPORTED
    assert legacy.binding_state is LocatorBindingState.INERT_LEGACY
    assert json.loads(imported.candidate_json) == hostile_raw
    assert imported.candidate_json == legacy.candidate_json
    assert not isinstance(imported, SourceLocatorEnvelope)
    with pytest.raises(ValueError, match="inert"):
        parse_inert_locator_candidate(
            hostile_raw,
            candidate_id="candidate-native",
            binding_state=LocatorBindingState.NATIVE,
        )


def test_inert_candidate_is_bounded_strict_and_cannot_be_constructed_native() -> None:
    candidate = parse_inert_locator_candidate(
        {"legacy_path": "notes/old.md"},
        candidate_id="candidate-1",
        binding_state=LocatorBindingState.INERT_LEGACY,
    )
    with pytest.raises(ValidationError):
        InertLocatorCandidate(
            **{**candidate.model_dump(), "binding_state": LocatorBindingState.NATIVE}
        )
    with pytest.raises(ValueError, match="locator candidate"):
        parse_inert_locator_candidate(
            {"value": "a" * LOCATOR_ENVELOPE_JSON_BYTES_MAX},
            candidate_id="oversize",
            binding_state=LocatorBindingState.INERT_IMPORTED,
        )


def test_inert_candidate_accepts_exact_16_kib_and_rejects_one_byte_over() -> None:
    overhead = len(b'{"value":""}')
    exact = {
        "value": "a" * (LOCATOR_ENVELOPE_JSON_BYTES_MAX - overhead),
    }
    candidate = parse_inert_locator_candidate(
        exact,
        candidate_id="candidate-exact",
        binding_state=LocatorBindingState.INERT_IMPORTED,
    )

    assert len(candidate.candidate_json.encode("utf-8")) == (
        LOCATOR_ENVELOPE_JSON_BYTES_MAX
    )
    with pytest.raises(ValueError, match="locator candidate"):
        parse_inert_locator_candidate(
            {"value": f"{exact['value']}a"},
            candidate_id="candidate-over",
            binding_state=LocatorBindingState.INERT_IMPORTED,
        )


def test_inert_candidate_preflight_rejects_gross_string_before_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GrossJSONString(str):
        def __len__(self) -> int:
            return 100 * 1024 * 1024

    canonical_calls = 0

    def unexpected_canonical_json(value: object) -> str:
        nonlocal canonical_calls
        canonical_calls += 1
        raise AssertionError("canonical serialization must not run")

    monkeypatch.setattr(
        locator_models,
        "_canonical_json",
        unexpected_canonical_json,
    )
    with pytest.raises(ValueError, match="preflight"):
        parse_inert_locator_candidate(
            {"value": GrossJSONString("synthetic-gross-string")},
            candidate_id="candidate-gross",
            binding_state=LocatorBindingState.INERT_IMPORTED,
        )

    assert canonical_calls == 0


def test_inert_candidate_preflight_rejects_hostile_json_trees_with_bounded_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeceptiveJSONString(str):
        def __len__(self) -> int:
            return 0

        def encode(
            self,
            encoding: str = "utf-8",
            errors: str = "strict",
        ) -> bytes:
            return b""

    deep: object = "leaf"
    for _ in range(locator_models.INERT_LOCATOR_JSON_DEPTH_MAX + 1):
        deep = [deep]
    cycle: list[object] = []
    cycle.append(cycle)
    hostile_values = (
        deep,
        [0] * (locator_models.INERT_LOCATOR_JSON_ITEMS_MAX + 1),
        [[] for _ in range(locator_models.INERT_LOCATOR_JSON_CONTAINERS_MAX + 1)],
        {"k" * (locator_models.INERT_LOCATOR_JSON_KEY_UTF8_BYTES_MAX + 1): "v"},
        {1: "non-string-key"},
        {"value": object()},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": DeceptiveJSONString("x" * (LOCATOR_ENVELOPE_JSON_BYTES_MAX + 1))},
        cycle,
    )

    def unexpected_canonical_json(value: object) -> str:
        raise AssertionError("hostile tree reached canonical serialization")

    monkeypatch.setattr(
        locator_models,
        "_canonical_json",
        unexpected_canonical_json,
    )
    for index, hostile in enumerate(hostile_values):
        with pytest.raises(ValueError, match="preflight"):
            parse_inert_locator_candidate(
                hostile,
                candidate_id=f"candidate-hostile-{index}",
                binding_state=LocatorBindingState.INERT_LEGACY,
            )


def test_rebinding_requires_fresh_lookup_explicit_decision_and_matching_scope() -> None:
    candidate = parse_inert_locator_candidate(
        {"legacy_source_id": "note-1"},
        candidate_id="candidate-1",
        binding_state=LocatorBindingState.INERT_LEGACY,
    )
    locator = _local_locator(source_kind=CanonicalSourceKind.NOTES)
    lookup = CurrentAuthorityLocatorLookup(
        lookup_id="lookup-1",
        candidate_id=candidate.candidate_id,
        authority_id=locator.authority_id,
        governance_scope_id=locator.governance_scope_id,
        profile_id=locator.profile_id,
        native_locator=locator,
        observed_at=NOW,
        valid_until=NOW + timedelta(minutes=5),
    )
    decision = RebindDecision(
        candidate_id=candidate.candidate_id,
        lookup_id=lookup.lookup_id,
        action=RebindAction.APPROVE,
        decided_at=NOW,
    )
    before = candidate.model_dump_json()
    with pytest.raises(ValueError, match="resolve_current"):
        rebind_inert_locator(
            candidate,
            lookup,
            decision,
            _authorization(resolve_current=False, view_snapshot=True),
            now=NOW + timedelta(minutes=1),
        )

    rebound = rebind_inert_locator(
        candidate,
        lookup,
        decision,
        _authorization(),
        now=NOW + timedelta(minutes=1),
    )

    assert rebound == locator
    assert rebound is not locator
    assert candidate.model_dump_json() == before
    assert candidate.binding_state is LocatorBindingState.INERT_LEGACY

    cases = (
        (
            lookup,
            decision.model_copy(update={"action": RebindAction.REJECT}),
            NOW,
        ),
        (lookup, decision, lookup.valid_until + timedelta(microseconds=1)),
        (
            lookup.model_copy(update={"candidate_id": "another-candidate"}),
            decision,
            NOW,
        ),
        (
            lookup,
            decision.model_copy(update={"lookup_id": "another-lookup"}),
            NOW,
        ),
    )
    for hostile_lookup, hostile_decision, now in cases:
        with pytest.raises(ValueError):
            rebind_inert_locator(
                candidate,
                hostile_lookup,
                hostile_decision,
                _authorization(),
                now=now,
            )

    with pytest.raises(ValueError):
        rebind_inert_locator(
            candidate,
            lookup,
            decision,
            _authorization(authority_id="different-authority"),
            now=NOW,
        )


def test_current_authority_lookup_rejects_an_unbounded_freshness_window() -> None:
    locator = _local_locator(source_kind=CanonicalSourceKind.NOTES)
    with pytest.raises(ValidationError, match="freshness window"):
        CurrentAuthorityLocatorLookup(
            lookup_id="lookup-1",
            candidate_id="candidate-1",
            authority_id=locator.authority_id,
            governance_scope_id=locator.governance_scope_id,
            profile_id=locator.profile_id,
            native_locator=locator,
            observed_at=NOW,
            valid_until=NOW + timedelta(minutes=5, microseconds=1),
        )


def test_no_locator_derived_authorization_constructor_or_plugin_registry() -> None:
    assert not hasattr(CitationReadAuthorization, "from_locator")
    assert not hasattr(CitationReadAuthorization, "authorized")
    module_names = set(
        vars(
            __import__(
                "tldw_chatbook.Chat.citation_source_locators",
                fromlist=["*"],
            )
        )
    )
    assert (
        not {
            "register_resolver",
            "load_plugin",
            "resolver_class",
            "resolver_handler",
        }
        & module_names
    )
