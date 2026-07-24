from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tldw_chatbook.Chat.citation_source_locators import (
    AUTHORITY_IDS_PER_READ_AUTHORIZATION_MAX,
    EXTERNAL_OPAQUE_ID_UTF8_BYTES_MAX,
    LOCATOR_ENVELOPE_JSON_BYTES_MAX,
    RUNTIME_SOURCE_KIND_TO_CANONICAL_V1,
    SOURCE_INVENTORY_V1,
    AuthorityScope,
    CanonicalSourceKind,
    CitationReadAuthorization,
    CurrentAuthorityLocatorLookup,
    InertLocatorCandidate,
    LocatorBindingState,
    RebindAction,
    RebindDecision,
    SourceCapability,
    SourceCapabilityPolicy,
    SourceInventoryEntry,
    SourceLocatorEnvelope,
    SourceLocatorPayloadV1,
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
        resolver_payload=payload or SourceLocatorPayloadV1(item_id="item-1"),
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
        resolver_payload=payload or SourceLocatorPayloadV1(item_id="item-1"),
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
        SourceLocatorPayloadV1(item_id="item", schema_version=2)
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
        SourceLocatorPayloadV1(item_id="item", **{field: value})


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
    payload = SourceLocatorPayloadV1(item_id="item", **{field: value})
    locator_factory = (
        _local_locator
        if source_kind
        in {
            CanonicalSourceKind.MEDIA_DB,
            CanonicalSourceKind.NOTES,
            CanonicalSourceKind.CHAT_HISTORY,
        }
        else _server_locator
    )
    with pytest.raises(ValidationError, match="location hint"):
        locator_factory(source_kind=source_kind, payload=payload)


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
        SourceLocatorPayloadV1(
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
        SourceLocatorPayloadV1(
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
            SourceLocatorPayloadV1(item_id="note-1", **update)

    locator = _local_locator(
        source_kind=CanonicalSourceKind.NOTES,
        payload=SourceLocatorPayloadV1(
            item_id="note-1",
            source_root_id="notes-root",
            relative_path="folder/note.md",
        ),
    )
    assert locator.resolver_payload.relative_path == "folder/note.md"


def test_locator_envelope_accepts_exact_16_kib_and_rejects_one_byte_over() -> None:
    base_payload = {
        "schema_version": 1,
        "item_id": "note-1",
        "source_root_id": "root",
        "relative_path": "",
        "chunk_id": None,
        "message_id": None,
        "entry_id": None,
        "parent_source_kind": None,
        "parent_item_id": None,
        "page_number": None,
        "section_ordinal": None,
        "start_seconds": None,
        "end_seconds": None,
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
        SourceLocatorPayloadV1(item_id="sql-1", path="/var/lib/database")
    with pytest.raises(ValidationError):
        SourceLocatorPayloadV1(item_id="sql-1", command="SELECT * FROM secrets")


def test_claims_open_only_through_authorized_parent_lineage() -> None:
    snapshot_only = _server_locator(CanonicalSourceKind.CLAIMS)
    with pytest.raises(ValueError, match="authorized parent"):
        validate_native_locator(
            snapshot_only,
            _authorization(local=False),
            SourceCapabilityPolicy(
                storage_mode=EvidenceStorageMode.SERVER_REFERENCE,
                view_snapshot=True,
                view_source_identity=True,
                resolve_current=True,
                open_native=True,
            ),
            required_capability=SourceCapability.OPEN_NATIVE,
        )

    with_parent = _server_locator(
        CanonicalSourceKind.CLAIMS,
        payload=SourceLocatorPayloadV1(
            item_id="claim-1",
            parent_source_kind=CanonicalSourceKind.MEDIA_DB,
            parent_item_id="media-1",
        ),
    )
    assert (
        validate_native_locator(
            with_parent,
            _authorization(local=False),
            SourceCapabilityPolicy(
                storage_mode=EvidenceStorageMode.SERVER_REFERENCE,
                view_snapshot=True,
                view_source_identity=True,
                resolve_current=True,
                open_native=True,
            ),
            required_capability=SourceCapability.OPEN_NATIVE,
        )
        == with_parent
    )

    wrong_parent = _server_locator(
        CanonicalSourceKind.CLAIMS,
        payload=SourceLocatorPayloadV1(
            item_id="claim-1",
            parent_source_kind=CanonicalSourceKind.PROMPTS,
            parent_item_id="prompt-1",
        ),
    )
    with pytest.raises(ValueError, match="authorized parent"):
        validate_native_locator(
            wrong_parent,
            _authorization(local=False),
            SourceCapabilityPolicy(
                storage_mode=EvidenceStorageMode.SERVER_REFERENCE,
                view_snapshot=True,
                open_native=True,
            ),
            required_capability=SourceCapability.OPEN_NATIVE,
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
    assert len(SOURCE_INVENTORY_V1) == len(CanonicalSourceKind)
    with pytest.raises(TypeError):
        RUNTIME_SOURCE_KIND_TO_CANONICAL_V1["plugin"] = "sql"  # type: ignore[index]
    with pytest.raises(ValidationError, match="frozen"):
        SOURCE_INVENTORY_V1[0].locator_version = 2  # type: ignore[misc]


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
