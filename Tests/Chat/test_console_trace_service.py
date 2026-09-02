"""Bounded model-surface and normalized request-header persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import inspect
import json
import sqlite3
import threading
from typing import cast

import pytest

from tldw_chatbook.Chat.console_trace_final_values import (
    ProviderRequestShadowBundle,
    SurfaceDeltaAdmission,
    VerifiedSurfaceDelta,
    VerifiedSurfaceDeltaItem,
    VerifiedSurfaceReplacementRange,
    build_verified_surface_delta,
    verify_provider_request_shadow,
)
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    TraceOmission,
    new_opaque_id,
)
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    DerivedTraceProvenance,
    OmittedTraceProvenance,
    ProviderArtifactTraceProvenance,
    ProviderRequestProvenance,
    SavedRevisionTraceProvenance,
    TraceOmissionReason,
    TraceProvenanceSource,
    TraceTransformKind,
    request_route_provenance,
)
from tldw_chatbook.Chat.console_trace_repository import (
    ConsoleTraceRepository,
    RequestHeaderRecord,
    SurfaceReplacementRecord,
    TraceCallRecord,
)
from tldw_chatbook.Chat.console_trace_service import (
    ConsoleTraceService,
)
from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    TransactionContextManager,
)
from tldw_chatbook.DB.transaction_observer import (
    active_managed_transaction_count,
    begin_managed_transaction,
    current_managed_transaction,
    register_transaction_completion,
)


@pytest.fixture
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(
        str(tmp_path / "trace-service.sqlite"),
        "console-trace-service-test",
    )
    yield database
    database.close_connection()


@pytest.fixture
def repository() -> ConsoleTraceRepository:
    return ConsoleTraceRepository()


def _policy() -> FrozenTracePolicy:
    return FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)


def _continuation_value(label: str) -> dict[str, object]:
    checkpoint = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 2,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": "kimi-k2",
            "api_base_url": "https://moonshot.invalid/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": label,
                    "reasoning_blocks": [],
                    "calls": [
                        {
                            "call_id": "call_1",
                            "name": "lookup",
                            "arguments": "{}",
                            "state": "completed",
                            "result": label,
                        }
                    ],
                }
            ],
        }
    )
    encoded = dump_provider_continuation_json(checkpoint)
    assert encoded is not None
    value = json.loads(encoded)
    assert isinstance(value, dict)
    return value


def _owned_segment(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> tuple[str, str]:
    conversation_id = db.add_conversation({"title": "trace service"})
    assert conversation_id is not None
    with db.transaction() as cursor:
        segment = repository.create_segment(cursor)
        owner = repository.attach_owner(
            cursor,
            conversation_id=conversation_id,
            root_segment_id=segment.segment_id,
        )
    return owner.owner_id, segment.segment_id


def _revision(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    conversation_id: str,
    *,
    content: str,
) -> str:
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": content,
        }
    )
    assert message_id is not None
    with db.transaction() as cursor:
        row = cursor.execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
                 WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1""",
            (message_id,),
        ).fetchone()
        assert row is not None
        return str(row[0])


def _available_bundle(
    provenance: ProviderRequestProvenance,
    *,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None = None,
    continuations: list[object] | None = None,
    system: str | None = None,
    system_components: tuple[object, ...] = (),
    temperature: int | float = 0,
    endpoint: str = "HTTPS://API.Example.Invalid:443/v1/",
    preparation_identity: str | None = None,
    surface_boundary: object | None = None,
) -> ProviderRequestShadowBundle:
    values: dict[str, object] = {
        "api_endpoint": "openai",
        "messages_payload": messages,
        "model": "gpt-test",
        "temp": temperature,
        "response_format": {"type": "json_object"},
        "reasoning_effort": "medium",
    }
    if system is not None:
        values["system_message"] = system
    if tools is not None:
        values["tools"] = tools
    if continuations is not None:
        values["provider_continuations"] = continuations
    return verify_provider_request_shadow(
        actual_kwargs=values,
        expected_kwargs=dict(values),
        provenance=provenance,
        project_handler_kwargs=lambda kwargs: kwargs,
        endpoint_identity=endpoint,
        preparation_identity=preparation_identity,
        system_component_values=system_components,
        surface_boundary=surface_boundary,
    )


def _provenance(
    descriptors: tuple[object, ...],
    *,
    system: object | None = None,
    tools: tuple[object, ...] = (),
) -> ProviderRequestProvenance:
    return ProviderRequestProvenance(
        system_message=system,
        messages=descriptors,
        messages_payload=descriptors,
        tools=tools,
        metadata=(request_route_provenance(ConsoleRequestRoute.FRESH),),
    )


@dataclass(frozen=True)
class SurfaceReplacementPlan:
    predecessor_head_id: str
    start_node_id: str
    end_node_id: str
    start_sequence: int
    end_sequence: int
    replacement_index: int


def _persist(
    service: ConsoleTraceService,
    cursor: object,
    *,
    owner_id: str,
    segment_id: str,
    provenance: ProviderRequestProvenance,
    bundle: ProviderRequestShadowBundle,
    append_from_index: int = 0,
    previous_surface_head_id: str | None = None,
    replacement: SurfaceReplacementPlan | None = None,
    **kwargs: object,
):
    kwargs.pop("previous_header_id", None)
    kwargs.pop("unavailable_provider_name", None)
    kwargs.pop("unavailable_model_name", None)
    kwargs.pop("unavailable_endpoint_identity", None)
    if bundle.preparation_identity is None:
        bundle = replace(bundle, preparation_identity=new_opaque_id())
    message_descriptors = tuple(provenance.messages_payload)
    continuation_descriptors = tuple(provenance.continuations)
    all_descriptors = message_descriptors + continuation_descriptors
    descriptors = all_descriptors
    if not bundle.available:
        descriptors = descriptors or (
            OmittedTraceProvenance(
                TraceProvenanceSource.PROVIDER_OVERLAY,
                bundle.omission_reason or TraceOmissionReason.SOURCE_UNAVAILABLE,
            ),
        )
    predecessor = previous_surface_head_id
    replacement_range = None
    if replacement is not None:
        predecessor = replacement.predecessor_head_id
        descriptors = (descriptors[replacement.replacement_index],)
        replacement_range = VerifiedSurfaceReplacementRange(
            predecessor_head_id=replacement.predecessor_head_id,
            start_node_id=replacement.start_node_id,
            end_node_id=replacement.end_node_id,
            start_sequence=replacement.start_sequence,
            end_sequence=replacement.end_sequence,
            current_ordinal=replacement.replacement_index,
        )
    else:
        descriptors = descriptors[append_from_index:]
    checkpoint = (
        service.current_surface_checkpoint(segment_id, expected_head_id=predecessor)
        if predecessor is not None and bundle.available
        else None
    )
    admission = SurfaceDeltaAdmission(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=predecessor,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=bundle.preparation_identity or "missing",
        descriptors=descriptors,
        projection_checkpoint=checkpoint,
        replacement_range=replacement_range,
    )
    if bundle.available:
        bindings = {item.name: item for item in bundle.components}
        message_binding = bindings["messages_payload"]
        assert isinstance(message_binding.value, tuple)
        continuation_binding = bindings.get("provider_continuations")
        continuation_values = (
            () if continuation_binding is None else continuation_binding.value
        )
        assert isinstance(continuation_values, tuple)
        all_values = tuple(message_binding.value) + tuple(continuation_values)
        bootstrap = checkpoint is None and predecessor is not None
        admitted_values = (
            all_values
            if bootstrap
            else (
                (all_values[replacement.replacement_index],)
                if replacement is not None
                else all_values[append_from_index:]
            )
        )
        if replacement is not None:
            if replacement.replacement_index < len(message_descriptors):
                message_delta, continuation_delta = descriptors, ()
            else:
                message_delta, continuation_delta = (), descriptors
        else:
            message_delta = message_descriptors[append_from_index:]
            continuation_delta = continuation_descriptors[
                max(0, append_from_index - len(message_descriptors)) :
            ]
        delta_provenance = (
            provenance
            if bootstrap
            else replace(
                provenance,
                messages=message_delta,
                messages_payload=message_delta,
                continuations=continuation_delta,
            )
        )
        boundary = service.prepare_surface_provenance(
            cursor,
            checkpoint,
            provenance=delta_provenance,
            admission=admission,
            values=admitted_values,
        )
        provenance = boundary.provenance
        values = bundle.boundary_kwargs
        issued_values = boundary._provider_request_surface_values()
        values["messages_payload"] = issued_values["messages_payload"]
        values["provider_continuations"] = issued_values["provider_continuations"]
        bundle = verify_provider_request_shadow(
            actual_kwargs=values,
            expected_kwargs=dict(values),
            provenance=provenance,
            project_handler_kwargs=lambda kwargs: kwargs,
            literal_payload=bundle.literal_payload_value,
            endpoint_identity=bundle.endpoint_identity,
            preparation_identity=bundle.preparation_identity,
            system_component_values=bundle.system_leaf_components,
            surface_boundary=boundary,
        )
        assert bundle.available
    delta = build_verified_surface_delta(
        provenance,
        bundle,
        admission=admission,
    )
    return service.persist_request(
        cursor,
        owner_id=owner_id,
        segment_id=segment_id,
        provenance=provenance,
        bundle=bundle,
        surface_delta=delta,
        **kwargs,
    )


def _initial_bound_request(
    service: ConsoleTraceService,
    cursor: object,
    *,
    owner_id: str,
    segment_id: str,
    provenance: ProviderRequestProvenance,
    messages: list[dict[str, object]],
    continuations: list[object] | None = None,
) -> tuple[ProviderRequestShadowBundle, SurfaceDeltaAdmission, object]:
    preparation_identity = new_opaque_id()
    admission = SurfaceDeltaAdmission(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=None,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=preparation_identity,
        descriptors=tuple(provenance.messages_payload)
        + tuple(provenance.continuations),
    )
    boundary = service.prepare_surface_provenance(
        cursor,
        None,
        provenance=provenance,
        admission=admission,
        values=tuple(messages) + tuple(continuations or ()),
    )
    actual = {
        "api_endpoint": "openai",
        "model": "gpt-test",
        **boundary._provider_request_surface_values(),
    }
    bundle = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=dict(actual),
        provenance=boundary.provenance,
        project_handler_kwargs=lambda kwargs: kwargs,
        endpoint_identity="https://api.example.test/v1",
        preparation_identity=preparation_identity,
        surface_boundary=boundary,
    )
    return bundle, admission, boundary.provenance


def test_initial_available_bundle_is_exact_object_bound_before_any_write(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        bundle, admission, projected = _initial_bound_request(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            messages=[{"role": "user", "content": "legitimate"}],
        )
        delta = build_verified_surface_delta(projected, bundle, admission=admission)
        forged_components = tuple(
            replace(
                item,
                value=({"role": "user", "content": "RAW-FORGED-BUNDLE-CANARY"},),
            )
            if item.name == "messages_payload"
            else item
            for item in bundle.components
        )
        forged = replace(bundle, components=forged_components)
        with pytest.raises(ValueError, match="surface_child_binding"):
            service.persist_request(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=projected,
                bundle=forged,
                surface_delta=delta,
            )
        assert repository.get_surface_tail(cursor, segment_id) is None


def test_prepared_boundary_issued_values_are_recursively_immutable(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    preparation_identity = new_opaque_id()
    service = ConsoleTraceService(repository)
    admission = SurfaceDeltaAdmission(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=None,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=preparation_identity,
        descriptors=(descriptor,),
    )
    boundary = service.prepare_surface_provenance(
        db.get_connection().cursor(),
        None,
        provenance=provenance,
        admission=admission,
        values=({"role": "user", "content": "legitimate"},),
    )

    issued = boundary._provider_request_surface_values()
    with pytest.raises(TypeError):
        issued["messages_payload"] = ()  # type: ignore[index]
    message = cast(Mapping[str, object], issued["messages_payload"][0])
    with pytest.raises(TypeError):
        message["content"] = "RAW-MUTATION-CANARY"  # type: ignore[index]


def test_initial_boundary_rejects_publicly_rebound_provenance(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    rag = ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, _policy())
    project = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.PROJECT_INSTRUCTION, rag.policy
    )
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        bundle, admission, projected = _initial_bound_request(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=_provenance((rag,)),
            messages=[{"role": "user", "content": "provider-only"}],
        )
        rebound = replace(
            projected,
            messages=(project,),
            messages_payload=(project,),
        )
        rebound_admission = replace(admission, descriptors=(project,))
        with pytest.raises(ValueError, match="surface_verified_bundle"):
            build_verified_surface_delta(
                rebound,
                bundle,
                admission=rebound_admission,
            )
        assert repository.get_surface_tail(cursor, segment_id) is None


def test_initial_provider_artifact_bundle_cannot_replay_across_owner_segment(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    other_owner_id, other_segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        bundle, admission, projected = _initial_bound_request(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            messages=[{"role": "user", "content": "provider-only"}],
        )
        delta = build_verified_surface_delta(projected, bundle, admission=admission)
        replay = replace(
            delta,
            owner_id=other_owner_id,
            segment_id=other_segment_id,
        )
        with pytest.raises(ValueError, match="surface_child_binding"):
            service.persist_request(
                cursor,
                owner_id=other_owner_id,
                segment_id=other_segment_id,
                provenance=projected,
                bundle=bundle,
                surface_delta=replay,
            )
        assert repository.get_surface_tail(cursor, other_segment_id) is None


def test_replacement_component_ordinal_is_derived_from_active_projection(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    descriptors = tuple(
        ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, policy)
        for _ in range(2)
    )
    service = ConsoleTraceService(repository)
    provenance = _provenance(descriptors)
    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[
                    {"role": "user", "content": "A"},
                    {"role": "user", "content": "B"},
                ],
            ),
        )
        nodes = repository.read_surface_nodes(cursor, segment_id)
        checkpoint = service.current_surface_checkpoint(
            segment_id, expected_head_id=initial.surface_head_id
        )
        assert checkpoint is not None
        admission = SurfaceDeltaAdmission(
            owner_id=owner_id,
            segment_id=segment_id,
            predecessor_surface_head_id=initial.surface_head_id,
            route_identity=ConsoleRequestRoute.FRESH.value,
            preparation_identity=new_opaque_id(),
            descriptors=(descriptors[1],),
            projection_checkpoint=checkpoint,
            replacement_range=VerifiedSurfaceReplacementRange(
                predecessor_head_id=initial.surface_head_id,
                start_node_id=nodes[1].node_id,
                end_node_id=nodes[1].node_id,
                start_sequence=1,
                end_sequence=1,
                current_ordinal=1,
                component_name="messages_payload",
                component_ordinal=0,
            ),
        )
        with pytest.raises(ValueError, match="replacement_component_ordinal"):
            service.prepare_surface_provenance(
                cursor,
                checkpoint,
                provenance=_provenance((descriptors[1],)),
                admission=admission,
                values=({"role": "user", "content": "FORGED-A"},),
            )
        assert repository.get_surface_tail(cursor, segment_id) is not None
        assert (
            repository.get_surface_tail(cursor, segment_id).node_id
            == initial.surface_head_id
        )


def test_adjacent_sequential_replacements_anchor_to_active_target_position(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    descriptors = tuple(
        ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, policy)
        for _ in range(3)
    )
    provenance = _provenance(descriptors)
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[
                    {"role": "user", "content": value} for value in ("A", "B", "C")
                ],
            ),
        )
        original = repository.read_surface_nodes(cursor, segment_id)
    with db.transaction() as cursor:
        first = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[
                    {"role": "user", "content": value} for value in ("A1", "B", "C")
                ],
            ),
            replacement=SurfaceReplacementPlan(
                predecessor_head_id=initial.surface_head_id,
                start_node_id=original[0].node_id,
                end_node_id=original[0].node_id,
                start_sequence=original[0].sequence,
                end_sequence=original[0].sequence,
                replacement_index=0,
            ),
        )
    with db.transaction() as cursor:
        second = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[
                    {"role": "user", "content": value} for value in ("A1", "B1", "C")
                ],
            ),
            replacement=SurfaceReplacementPlan(
                predecessor_head_id=first.surface_head_id,
                start_node_id=original[1].node_id,
                end_node_id=original[1].node_id,
                start_sequence=original[1].sequence,
                end_sequence=original[1].sequence,
                replacement_index=1,
            ),
        )

    assert second.replacement is not None
    assert second.replacement.replacement.start_sequence == original[1].sequence


@pytest.mark.parametrize("components", [("FORGED-SYSTEM",), ()])
def test_saved_system_leaf_must_match_canonical_revision_before_write(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    components: tuple[str, ...],
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    revision_id = _revision(
        db, repository, owner.conversation_id, content="CANONICAL-SYSTEM"
    )
    message = OmittedTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT,
        TraceOmissionReason.SOURCE_UNAVAILABLE,
    )
    provenance = _provenance(
        (message,), system=SavedRevisionTraceProvenance(revision_id)
    )
    bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": "omitted"}],
        system="FORGED-SYSTEM",
        system_components=components,
    )
    with pytest.raises(ValueError, match="system_revision_value_mismatch"):
        with db.transaction() as cursor:
            _persist(
                ConsoleTraceService(repository),
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
            )
    cursor = db.get_connection().cursor()
    assert repository.get_surface_tail(cursor, segment_id) is None
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == 0
    )


def test_saved_messages_are_revision_only_and_header_reuses_exact_value(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    revision_id = _revision(
        db, repository, owner.conversation_id, content="ORDINARY-BODY-CANARY"
    )
    descriptor = SavedRevisionTraceProvenance(revision_id)
    provenance = _provenance((descriptor,))
    bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": "ORDINARY-BODY-CANARY"}],
    )
    service = ConsoleTraceService(repository)

    with db.transaction() as cursor:
        first = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        )
        statements: list[str] = []
        db.get_connection().set_trace_callback(statements.append)
        exact = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
            append_from_index=1,
            previous_surface_head_id=first.surface_head_id,
            previous_header_id=first.header.header_id,
        )
        db.get_connection().set_trace_callback(None)
        node = repository.get_surface_node(cursor, first.surface_head_id)
        artifact_bodies = [
            bytes(row[0])
            for row in cursor.execute(
                "SELECT sanitized_bytes FROM console_trace_artifacts"
            )
        ]

    assert node is not None and node.reference_kind == "revision"
    assert node.semantic_revision_id == revision_id
    assert node.artifact_id is None
    assert exact.header.header_id == first.header.header_id
    assert all(b"ORDINARY-BODY-CANARY" not in body for body in artifact_bodies)
    assert first.checkpoint is not None
    assert "ORDINARY-BODY-CANARY" not in repr(first.checkpoint)
    assert any("console_trace_request_headers" in statement for statement in statements)


def test_complete_header_reconstructs_components_and_changes_by_one_field(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    message = ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, policy)
    system = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RENDERED_SYSTEM, policy
    )
    tool = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.TOOL_DEFINITION, policy
    )
    provenance = _provenance((message,), system=system, tools=(tool,))
    bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": "retrieved context"}],
        system="rendered system",
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )
    changed_bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": "retrieved context"}],
        system="rendered system",
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        temperature=1,
    )
    service = ConsoleTraceService(repository)

    with db.transaction() as cursor:
        first = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        )
        changed = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=changed_bundle,
            append_from_index=1,
            previous_surface_head_id=first.surface_head_id,
        )
        reconstructed = service.reconstruct_header(cursor, first.header.header_id)

    assert changed.header.header_id != first.header.header_id
    assert reconstructed.header_id == first.header.header_id
    assert reconstructed.provider_name == "openai"
    assert reconstructed.model_name == "gpt-test"
    assert reconstructed.route_identity == "fresh"
    assert reconstructed.endpoint_identity == "https://api.example.invalid/v1"
    assert reconstructed.generation_parameters["temp"] == 0
    assert reconstructed.response_format == {"type": "json_object"}
    assert reconstructed.reasoning_controls == {"reasoning_effort": "medium"}
    assert {item.component_kind for item in reconstructed.components} == {
        "rendered_system_part",
        "tool_schema",
    }
    values_by_kind = {
        item.component_kind: item.value for item in reconstructed.components
    }
    assert values_by_kind == {
        "rendered_system_part": "rendered system",
        "tool_schema": {"function": {"name": "lookup"}, "type": "function"},
    }
    assert reconstructed.adapter_defaults["parameter_sources"]["temp"] == "explicit"
    assert reconstructed.adapter_defaults["parameter_sources"]["max_tokens"] == (
        "adapter_default"
    )
    assert reconstructed.adapter_defaults["surface_sources"] == {"rag_context": 1}


def test_provider_artifact_reuses_across_policies_but_policy_metadata_differs(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    descriptors = tuple(
        ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, _policy())
        for _ in range(2)
    )
    service = ConsoleTraceService(repository)
    artifact_ids: list[str] = []
    header_ids: list[str] = []
    for index, descriptor in enumerate(descriptors):
        if index:
            owner_id, segment_id = _owned_segment(db, repository)
        provenance = _provenance((descriptor,))
        bundle = _available_bundle(
            provenance,
            messages=[{"role": "user", "content": "same sanitized value"}],
        )
        with db.transaction() as cursor:
            persisted = _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
                append_from_index=0,
            )
            node = repository.get_surface_node(cursor, persisted.surface_head_id)
            assert node is not None and node.artifact_id is not None
            artifact_ids.append(node.artifact_id)
            header_ids.append(persisted.header.header_id)

    assert artifact_ids[0] == artifact_ids[1]
    assert header_ids[0] != header_ids[1]


def test_omission_is_content_free_and_never_creates_an_artifact(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    omission = OmittedTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT,
        TraceOmissionReason.SANITIZER_FAILED,
    )
    provenance = _provenance((omission,))
    bundle = ProviderRequestShadowBundle(
        available=False,
        omission_reason=TraceOmissionReason.SANITIZER_FAILED,
    )
    service = ConsoleTraceService(repository)

    with db.transaction() as cursor:
        persisted = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
            unavailable_provider_name="openai",
            unavailable_model_name="gpt-test",
            unavailable_endpoint_identity="unavailable",
        )
        node = repository.get_surface_node(cursor, persisted.surface_head_id)
        artifact_count = cursor.execute(
            "SELECT COUNT(*) FROM console_trace_artifacts"
        ).fetchone()[0]

    assert node is not None and node.reference_kind == "omission"
    assert node.omission_reason_code == "sanitizer_failed"
    assert artifact_count == 0
    assert persisted.header.provider_name == "unavailable"
    assert persisted.header.model_name == "unavailable"
    assert persisted.header.endpoint_identity == "unavailable"
    assert persisted.header.adapter_defaults["header_omissions"] == {
        "provider_request": "sanitizer_failed"
    }


def test_incremental_append_uses_tail_queries_and_stores_no_history_list(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    descriptors = tuple(
        SavedRevisionTraceProvenance(
            _revision(db, repository, owner.conversation_id, content=f"message-{index}")
        )
        for index in range(4)
    )
    service = ConsoleTraceService(repository)
    statements: list[str] = []
    connection = db.get_connection()
    connection.set_trace_callback(statements.append)
    try:
        with db.transaction() as cursor:
            first = _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=_provenance(descriptors[:3]),
                bundle=_available_bundle(
                    _provenance(descriptors[:3]),
                    messages=[
                        {"role": "user", "content": f"message-{index}"}
                        for index in range(3)
                    ],
                ),
            )
            second = _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=_provenance(descriptors),
                bundle=_available_bundle(
                    _provenance(descriptors),
                    messages=[
                        {"role": "user", "content": f"message-{index}"}
                        for index in range(4)
                    ],
                ),
                append_from_index=3,
                previous_surface_head_id=first.surface_head_id,
            )
            rows = repository.read_surface_nodes(cursor, segment_id)
    finally:
        connection.set_trace_callback(None)

    assert len(first.appended_nodes) == 3
    assert len(second.appended_nodes) == 1
    assert len(rows) == 4
    assert rows[-1].predecessor_node_id == rows[-2].node_id
    assert any("ORDER BY sequence DESC" in statement for statement in statements)
    assert not any("history" in field.name for field in fields(type(rows[-1])))


def test_repeated_history_transform_does_not_churn_header(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    saved = tuple(
        SavedRevisionTraceProvenance(
            _revision(db, repository, owner.conversation_id, content=f"item-{index}")
        )
        for index in range(2)
    )
    derived = tuple(
        DerivedTraceProvenance(TraceTransformKind.MESSAGE_REWRITE, (item,))
        for item in saved
    )
    first_provenance = _provenance(derived[:1])
    second_provenance = _provenance(derived)
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        first = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=first_provenance,
            bundle=_available_bundle(
                first_provenance,
                messages=[{"role": "user", "content": "item-0"}],
            ),
        )
        second = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=second_provenance,
            bundle=_available_bundle(
                second_provenance,
                messages=[
                    {"role": "user", "content": "item-0"},
                    {"role": "user", "content": "item-1"},
                ],
            ),
            append_from_index=1,
            previous_surface_head_id=first.surface_head_id,
            previous_header_id=first.header.header_id,
        )

    assert first.header.adapter_defaults["surface_transforms"] == {"message_rewrite": 1}
    assert second.header.header_id == first.header.header_id


def test_incremental_append_rejects_unverified_predecessor_head(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    bundle = _available_bundle(
        provenance, messages=[{"role": "user", "content": "seed"}]
    )
    service = ConsoleTraceService(repository)
    with pytest.raises(ValueError, match="surface_delta_admission"):
        with db.transaction() as cursor:
            _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
                append_from_index=1,
            )
    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        )
        before = len(repository.read_surface_nodes(cursor, segment_id))
        with pytest.raises(ValueError, match="surface_predecessor_mismatch"):
            _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
                append_from_index=1,
                previous_surface_head_id=new_opaque_id(),
            )
        assert len(repository.read_surface_nodes(cursor, segment_id)) == before
        assert initial.surface_head_id != new_opaque_id()


def test_delta_contract_has_no_prefix_skip_and_is_bound_to_owner_domain(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    assert (
        "append_from_index"
        not in inspect.signature(ConsoleTraceService.persist_request).parameters
    )
    owner_id, segment_id = _owned_segment(db, repository)
    other_owner_id, other_segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    bundle = _available_bundle(
        provenance, messages=[{"role": "user", "content": "delta"}]
    )
    item = VerifiedSurfaceDeltaItem("messages_payload", 0, descriptor)
    delta = VerifiedSurfaceDelta(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=None,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=bundle.preparation_identity or "missing",
        items=(item,),
    )
    with pytest.raises(ValueError, match="surface_delta_identity"):
        with db.transaction() as cursor:
            ConsoleTraceService(repository).persist_request(
                cursor,
                owner_id=other_owner_id,
                segment_id=other_segment_id,
                provenance=provenance,
                bundle=bundle,
                surface_delta=delta,
            )
    assert (
        repository.get_surface_tail(db.get_connection().cursor(), other_segment_id)
        is None
    )


def test_slice_b_delta_builder_drops_ordinary_body_and_has_content_safe_repr(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        bundle, admission, projected = _initial_bound_request(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            messages=[{"role": "user", "content": "DELTA-BODY-CANARY"}],
        )
        delta = build_verified_surface_delta(projected, bundle, admission=admission)

    assert not hasattr(delta.items[0], "provider_value")
    assert "DELTA-BODY-CANARY" not in repr(delta)
    assert "DELTA-BODY-CANARY" not in repr(delta.items[0])
    with pytest.raises(TypeError):
        VerifiedSurfaceDeltaItem(  # type: ignore[call-arg]
            "messages_payload", 0, descriptor, "DELTA-BODY-CANARY"
        )


def test_verified_delta_items_cannot_carry_provider_content() -> None:
    assert {field.name for field in fields(VerifiedSurfaceDeltaItem)} == {
        "component_name",
        "ordinal",
        "provenance",
    }


def test_forged_delta_ordinal_is_rejected_before_first_trace_insert(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": "RAW-PROVIDER-CANARY"}],
    )
    forged = VerifiedSurfaceDelta(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=None,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=bundle.preparation_identity or "missing",
        items=(VerifiedSurfaceDeltaItem("messages_payload", 1, descriptor),),
    )
    statements: list[str] = []
    with db.transaction() as cursor:
        db.get_connection().set_trace_callback(statements.append)
        with pytest.raises(ValueError, match="surface_child_binding"):
            ConsoleTraceService(repository).persist_request(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
                surface_delta=forged,
            )
        db.get_connection().set_trace_callback(None)

    assert "RAW-PROVIDER-CANARY" not in repr(forged)
    assert not any(
        statement.lstrip().upper().startswith("INSERT INTO CONSOLE_TRACE_")
        for statement in statements
    )


def test_service_has_no_caller_controlled_header_inheritance() -> None:
    assert (
        "previous_header_id"
        not in inspect.signature(ConsoleTraceService.persist_request).parameters
    )


def test_cross_owner_header_cannot_inherit_foreign_policy(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    first_owner, first_segment = _owned_segment(db, repository)
    second_owner, second_segment = _owned_segment(db, repository)
    policies = (_policy(), _policy())
    headers = []
    for owner_id, segment_id, policy in (
        (first_owner, first_segment, policies[0]),
        (second_owner, second_segment, policies[1]),
    ):
        descriptor = ProviderArtifactTraceProvenance(
            TraceProvenanceSource.RAG_CONTEXT, policy
        )
        provenance = _provenance((descriptor,))
        bundle = _available_bundle(
            provenance,
            messages=[{"role": "user", "content": "shared provider value"}],
        )
        with db.transaction() as cursor:
            headers.append(
                _persist(
                    ConsoleTraceService(repository),
                    cursor,
                    owner_id=owner_id,
                    segment_id=segment_id,
                    provenance=provenance,
                    bundle=bundle,
                ).header
            )

    assert headers[0].header_id != headers[1].header_id
    assert headers[0].adapter_defaults["artifact_policy_id"] == policies[0].policy_id
    assert headers[1].adapter_defaults["artifact_policy_id"] == policies[1].policy_id


def test_unavailable_header_uses_only_fixed_content_free_metadata() -> None:
    parameters = inspect.signature(ConsoleTraceService.persist_request).parameters
    assert "unavailable_provider_name" not in parameters
    assert "unavailable_model_name" not in parameters
    assert "unavailable_endpoint_identity" not in parameters


def test_nonempty_surface_requires_predecessor_even_for_all_new_suffix(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    bundle = _available_bundle(
        provenance, messages=[{"role": "user", "content": "seed"}]
    )
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        )
        before = len(repository.read_surface_nodes(cursor, segment_id))
        with pytest.raises(ValueError, match="surface_predecessor_mismatch"):
            _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
                append_from_index=0,
            )
        assert len(repository.read_surface_nodes(cursor, segment_id)) == before


def test_cached_full_projection_rejects_stale_prefix_before_append(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    revisions = tuple(
        _revision(db, repository, owner.conversation_id, content=value)
        for value in ("A", "B", "C")
    )
    service = ConsoleTraceService(repository)
    first_provenance = _provenance((SavedRevisionTraceProvenance(revisions[0]),))
    first_bundle = _available_bundle(
        first_provenance, messages=[{"role": "user", "content": "A"}]
    )
    with db.transaction() as cursor:
        first = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=first_provenance,
            bundle=first_bundle,
        )

    stale_descriptors = (
        SavedRevisionTraceProvenance(revisions[1]),
        SavedRevisionTraceProvenance(revisions[2]),
    )
    stale_provenance = _provenance(stale_descriptors)
    stale_bundle = _available_bundle(
        stale_provenance,
        messages=[
            {"role": "user", "content": "B"},
            {"role": "user", "content": "C"},
        ],
    )
    with pytest.raises(ValueError, match="surface_verified_bundle"):
        build_verified_surface_delta(
            stale_provenance,
            stale_bundle,
            admission=SurfaceDeltaAdmission(
                owner_id=owner_id,
                segment_id=segment_id,
                predecessor_surface_head_id=first.surface_head_id,
                route_identity=ConsoleRequestRoute.FRESH.value,
                preparation_identity=stale_bundle.preparation_identity or "missing",
                descriptors=(stale_descriptors[1],),
            ),
        )
    assert (
        len(repository.read_surface_nodes(db.get_connection().cursor(), segment_id))
        == 1
    )


def test_reopen_rejects_unconsumed_domain_values_missing_from_empty_delta(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    message_a = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST, policy
    )
    message_b = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST, policy
    )
    continuation = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.CONTINUATION, policy
    )

    def provenance_for(messages: tuple[object, ...]) -> ProviderRequestProvenance:
        return ProviderRequestProvenance(
            messages=messages,
            messages_payload=messages,
            continuations=(continuation,),
            metadata=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        )

    initial_provenance = provenance_for((message_a,))
    initial_bundle = _available_bundle(
        initial_provenance,
        messages=[{"role": "user", "content": "A"}],
        continuations=[_continuation_value("C")],
    )
    with db.transaction() as cursor:
        initial = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=initial_provenance,
            bundle=initial_bundle,
        )

    reopened_provenance = provenance_for((message_a, message_b))
    reopened_bundle = _available_bundle(
        reopened_provenance,
        messages=[
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ],
        continuations=[_continuation_value("C")],
    )
    admission = SurfaceDeltaAdmission(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=initial.surface_head_id,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=reopened_bundle.preparation_identity or "missing",
        descriptors=(),
    )
    before = repository.read_surface_nodes(db.get_connection().cursor(), segment_id)
    with pytest.raises(ValueError, match="surface_delta_alignment"):
        with db.transaction() as cursor:
            ConsoleTraceService(repository).prepare_surface_provenance(
                cursor,
                None,
                provenance=reopened_provenance,
                admission=admission,
                values=(
                    {"role": "user", "content": "A"},
                    {"role": "user", "content": "B"},
                    _continuation_value("C"),
                ),
            )
    assert (
        repository.read_surface_nodes(db.get_connection().cursor(), segment_id)
        == before
    )


def test_official_surface_capability_rejects_stale_full_projection(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    revisions = tuple(
        _revision(db, repository, owner.conversation_id, content=value)
        for value in ("A", "B", "C")
    )
    service = ConsoleTraceService(repository)
    first_provenance = _provenance((SavedRevisionTraceProvenance(revisions[0]),))
    with db.transaction() as cursor:
        first = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=first_provenance,
            bundle=_available_bundle(
                first_provenance,
                messages=[{"role": "user", "content": "A"}],
            ),
        )

    stale_descriptors = (
        SavedRevisionTraceProvenance(revisions[1]),
        SavedRevisionTraceProvenance(revisions[2]),
    )
    checkpoint = service.current_surface_checkpoint(
        segment_id, expected_head_id=first.surface_head_id
    )
    assert checkpoint is not None
    preparation_identity = new_opaque_id()
    admission = SurfaceDeltaAdmission(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=first.surface_head_id,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=preparation_identity,
        descriptors=(stale_descriptors[1],),
        projection_checkpoint=checkpoint,
    )
    boundary = service.prepare_surface_provenance(
        db.get_connection().cursor(),
        checkpoint,
        provenance=_provenance((stale_descriptors[1],)),
        admission=admission,
        values=({"role": "user", "content": "C"},),
    )
    projected = boundary.provenance
    stale_values = {
        "api_endpoint": "openai",
        "model": "gpt-test",
        "messages_payload": [
            {"role": "user", "content": "B"},
            {"role": "user", "content": "C"},
        ],
    }
    stale_bundle = verify_provider_request_shadow(
        actual_kwargs=stale_values,
        expected_kwargs=dict(stale_values),
        provenance=projected,
        project_handler_kwargs=lambda kwargs: kwargs,
        endpoint_identity="https://api.example.test/v1",
        preparation_identity=preparation_identity,
        surface_boundary=boundary,
    )
    assert not stale_bundle.available
    before = repository.read_surface_nodes(db.get_connection().cursor(), segment_id)
    with pytest.raises(ValueError, match="surface_prefix_mismatch"):
        build_verified_surface_delta(
            projected,
            stale_bundle,
            admission=admission,
        )
    assert (
        repository.read_surface_nodes(db.get_connection().cursor(), segment_id)
        == before
    )


def test_surface_capability_rejects_forgery_cross_service_and_allows_rollback_retry(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    revisions = tuple(
        _revision(db, repository, owner.conversation_id, content=value)
        for value in ("A", "C", "D")
    )
    service = ConsoleTraceService(repository)
    initial_provenance = _provenance((SavedRevisionTraceProvenance(revisions[0]),))
    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=initial_provenance,
            bundle=_available_bundle(
                initial_provenance,
                messages=[{"role": "user", "content": "A"}],
            ),
        )
    descriptors = tuple(SavedRevisionTraceProvenance(item) for item in revisions[:2])
    provenance = _provenance(descriptors)
    bundle = _available_bundle(
        provenance,
        messages=[
            {"role": "user", "content": "A"},
            {"role": "user", "content": "C"},
        ],
    )
    checkpoint = service.current_surface_checkpoint(
        segment_id, expected_head_id=initial.surface_head_id
    )
    admission = SurfaceDeltaAdmission(
        owner_id,
        segment_id,
        initial.surface_head_id,
        ConsoleRequestRoute.FRESH.value,
        bundle.preparation_identity or "missing",
        (descriptors[1],),
        checkpoint,
    )
    with pytest.raises(ValueError, match="surface_checkpoint_identity"):
        service.prepare_surface_provenance(
            db.get_connection().cursor(),
            checkpoint,
            provenance=_provenance((descriptors[1],)),
            admission=replace(
                admission,
                route_identity=ConsoleRequestRoute.RETRY.value,
            ),
            values=({"role": "user", "content": "C"},),
        )
    boundary = service.prepare_surface_provenance(
        db.get_connection().cursor(),
        checkpoint,
        provenance=_provenance((descriptors[1],)),
        admission=admission,
        values=({"role": "user", "content": "C"},),
    )
    provenance = boundary.provenance
    assert not boundary._verify_surface_values(
        provenance,
        {},
        {},
        object(),
    )
    with pytest.raises(ValueError, match="surface_verified_bundle"):
        boundary._bind_verified_bundle(provenance, bundle, object())
    values = bundle.boundary_kwargs
    values.update(boundary._provider_request_surface_values())
    bundle = verify_provider_request_shadow(
        actual_kwargs=values,
        expected_kwargs=dict(values),
        provenance=provenance,
        project_handler_kwargs=lambda kwargs: kwargs,
        endpoint_identity=bundle.endpoint_identity,
        preparation_identity=admission.preparation_identity,
        surface_boundary=boundary,
    )
    with pytest.raises(ValueError, match="surface_verified_bundle"):
        boundary._bind_verified_bundle(provenance, bundle, object())

    extend = getattr(checkpoint, "_extend_surface_projection")
    message_index = next(
        index
        for index, component in enumerate(bundle.components)
        if component.name == "messages_payload"
    )
    forged_components = list(bundle.components)
    forged_components[message_index] = replace(
        forged_components[message_index],
        value=({"role": "user", "content": "forged"},),
    )
    with pytest.raises(ValueError, match="surface_verified_bundle"):
        build_verified_surface_delta(
            provenance,
            replace(bundle, components=tuple(forged_components)),
            admission=admission,
        )

    with pytest.raises(ValueError, match="surface_prefix_mismatch"):
        extend(
            admission=admission,
            replacement=None,
            preparation_identity=bundle.preparation_identity,
            items=(
                VerifiedSurfaceDeltaItem(
                    "messages_payload",
                    0,
                    descriptors[0],
                ),
            ),
            bundle=bundle,
            surface_boundary_identity=id(bundle.surface_boundary),
            provenance=provenance,
        )

    with pytest.raises(ValueError, match="surface_verified_bundle"):
        build_verified_surface_delta(replace(provenance), bundle, admission=admission)
    with pytest.raises(ValueError, match="surface_verified_bundle"):
        build_verified_surface_delta(provenance, bundle, admission=replace(admission))
    delta = build_verified_surface_delta(provenance, bundle, admission=admission)
    with pytest.raises(ValueError, match="surface_verified_bundle"):
        build_verified_surface_delta(provenance, bundle, admission=admission)

    for rejected in (replace(delta, child_binding=object()), delta):
        target = service if rejected is not delta else ConsoleTraceService(repository)
        with pytest.raises(ValueError, match="surface_child_binding"):
            with db.transaction() as cursor:
                target.persist_request(
                    cursor,
                    owner_id=owner_id,
                    segment_id=segment_id,
                    provenance=provenance,
                    bundle=bundle,
                    surface_delta=rejected,
                )

    with pytest.raises(RuntimeError, match="rollback"):
        with db.transaction() as cursor:
            first_attempt = service.persist_request(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
                surface_delta=delta,
            )
            extended_descriptors = descriptors + (
                SavedRevisionTraceProvenance(revisions[2]),
            )
            extended_provenance = _provenance(extended_descriptors)
            _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=extended_provenance,
                bundle=_available_bundle(
                    extended_provenance,
                    messages=[
                        {"role": "user", "content": "A"},
                        {"role": "user", "content": "C"},
                        {"role": "user", "content": "D"},
                    ],
                ),
                append_from_index=2,
                previous_surface_head_id=first_attempt.surface_head_id,
            )
            raise RuntimeError("rollback")
    with db.transaction() as cursor:
        retried = service.persist_request(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
            surface_delta=delta,
        )
    assert retried.surface_head_id != initial.surface_head_id
    after = repository.read_surface_nodes(db.get_connection().cursor(), segment_id)
    with pytest.raises(ValueError, match="surface_predecessor_mismatch"):
        with db.transaction() as cursor:
            service.persist_request(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
                surface_delta=delta,
            )
    assert (
        repository.read_surface_nodes(db.get_connection().cursor(), segment_id) == after
    )


def test_noop_surface_child_is_one_shot_after_commit(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    with db.transaction() as cursor:
        initial = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[{"role": "user", "content": "A"}],
            ),
        )

    service = ConsoleTraceService(repository)
    unbound = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": "A"}],
    )
    admission = SurfaceDeltaAdmission(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=initial.surface_head_id,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=unbound.preparation_identity or "missing",
        descriptors=(),
    )
    boundary = service.prepare_surface_provenance(
        db.get_connection().cursor(),
        None,
        provenance=provenance,
        admission=admission,
        values=({"role": "user", "content": "A"},),
    )
    actual = unbound.boundary_kwargs
    actual.update(boundary._provider_request_surface_values())
    bundle = verify_provider_request_shadow(
        actual_kwargs=actual,
        expected_kwargs=dict(actual),
        provenance=boundary.provenance,
        project_handler_kwargs=lambda kwargs: kwargs,
        endpoint_identity=unbound.endpoint_identity,
        preparation_identity=unbound.preparation_identity,
        surface_boundary=boundary,
    )
    delta = build_verified_surface_delta(
        boundary.provenance,
        bundle,
        admission=admission,
    )
    with db.transaction() as cursor:
        first = service.persist_request(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=boundary.provenance,
            bundle=bundle,
            surface_delta=delta,
        )
    assert first.surface_head_id == initial.surface_head_id
    assert first.appended_nodes == ()

    with db.transaction() as cursor:
        other_service_noop = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[{"role": "user", "content": "A"}],
            ),
            append_from_index=1,
            previous_surface_head_id=initial.surface_head_id,
        )
    assert other_service_noop.surface_head_id == initial.surface_head_id

    with pytest.raises(ValueError, match="surface_child_binding"):
        with db.transaction() as cursor:
            service.persist_request(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=boundary.provenance,
                bundle=bundle,
                surface_delta=delta,
            )


def test_two_noop_children_remain_retryable_after_shared_outer_rollback(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    with db.transaction() as cursor:
        initial = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[{"role": "user", "content": "A"}],
            ),
        )

    service = ConsoleTraceService(repository)

    def prepare_noop() -> tuple[
        ProviderRequestProvenance,
        ProviderRequestShadowBundle,
        VerifiedSurfaceDelta,
    ]:
        unbound = _available_bundle(
            provenance,
            messages=[{"role": "user", "content": "A"}],
        )
        admission = SurfaceDeltaAdmission(
            owner_id=owner_id,
            segment_id=segment_id,
            predecessor_surface_head_id=initial.surface_head_id,
            route_identity=ConsoleRequestRoute.FRESH.value,
            preparation_identity=unbound.preparation_identity or "missing",
            descriptors=(),
        )
        boundary = service.prepare_surface_provenance(
            db.get_connection().cursor(),
            None,
            provenance=provenance,
            admission=admission,
            values=({"role": "user", "content": "A"},),
        )
        actual = unbound.boundary_kwargs
        actual.update(boundary._provider_request_surface_values())
        bundle = verify_provider_request_shadow(
            actual_kwargs=actual,
            expected_kwargs=dict(actual),
            provenance=boundary.provenance,
            project_handler_kwargs=lambda kwargs: kwargs,
            endpoint_identity=unbound.endpoint_identity,
            preparation_identity=unbound.preparation_identity,
            surface_boundary=boundary,
        )
        return (
            boundary.provenance,
            bundle,
            build_verified_surface_delta(
                boundary.provenance,
                bundle,
                admission=admission,
            ),
        )

    first = prepare_noop()
    second = prepare_noop()
    with pytest.raises(RuntimeError, match="rollback"):
        with db.transaction() as cursor:
            for projected, bundle, delta in (first, second):
                service.persist_request(
                    cursor,
                    owner_id=owner_id,
                    segment_id=segment_id,
                    provenance=projected,
                    bundle=bundle,
                    surface_delta=delta,
                )
            raise RuntimeError("rollback")

    for projected, bundle, delta in (first, second):
        with db.transaction() as cursor:
            retried = service.persist_request(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=projected,
                bundle=bundle,
                surface_delta=delta,
            )
        assert retried.surface_head_id == initial.surface_head_id


def test_consumed_capability_tracking_stays_bounded_on_long_lived_connection(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    descriptors = tuple(
        ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, policy)
        for _ in range(12)
    )
    values = tuple({"role": "user", "content": f"value-{index}"} for index in range(12))
    service = ConsoleTraceService(repository)
    head_id: str | None = None
    for length in range(1, 13):
        provenance = _provenance(descriptors[:length])
        with db.transaction() as cursor:
            persisted = _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=_available_bundle(
                    provenance,
                    messages=list(values[:length]),
                ),
                append_from_index=length - 1,
                previous_surface_head_id=head_id,
            )
        head_id = persisted.surface_head_id

    assert active_managed_transaction_count() == 0
    assert service._pending_child_uses == {}


def test_repeated_same_head_requests_keep_one_parent_capability(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[{"role": "user", "content": "A"}],
            ),
        )

    for _ in range(12):
        with db.transaction() as cursor:
            persisted = _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=_available_bundle(
                    provenance,
                    messages=[{"role": "user", "content": "A"}],
                ),
                append_from_index=1,
                previous_surface_head_id=initial.surface_head_id,
            )
        assert persisted.surface_head_id == initial.surface_head_id

    assert len(service._parent_capabilities) == 1


@pytest.mark.parametrize("commit_fails", [False, True])
def test_failed_rollback_releases_observers_and_reports_ambiguous_outcome(
    commit_fails: bool,
) -> None:
    class FailingConnection:
        def commit(self) -> None:
            if commit_fails:
                raise sqlite3.OperationalError("commit failed")

        def rollback(self) -> None:
            raise sqlite3.OperationalError("rollback failed")

    class FakeDB:
        def __init__(self) -> None:
            self._local = threading.local()
            self._local.transaction_depth = 1

    connection = cast(sqlite3.Connection, FailingConnection())
    manager = TransactionContextManager(cast(CharactersRAGDB, FakeDB()))
    manager.conn = connection
    manager.is_outermost_transaction = True
    token = begin_managed_transaction(connection)
    manager.transaction_observer_token = token
    outcomes: list[bool | None] = []
    register_transaction_completion(connection, token, outcomes.append)

    if commit_fails:
        with pytest.raises(CharactersRAGDBError, match="Commit failed"):
            manager.__exit__(None, None, None)
    else:
        failure = RuntimeError("block failed")
        with pytest.raises(CharactersRAGDBError, match="Rollback failed"):
            manager.__exit__(RuntimeError, failure, failure.__traceback__)

    assert outcomes == [None]
    assert active_managed_transaction_count() == 0


def test_transaction_completion_runs_every_callback_without_breaking_commit(
    db: CharactersRAGDB,
    caplog: pytest.LogCaptureFixture,
) -> None:
    outcomes: list[tuple[str, bool | None]] = []

    def fail_first(committed: bool | None) -> None:
        outcomes.append(("first", committed))
        raise RuntimeError("CALLBACK-SECRET-CANARY")

    with db.transaction() as cursor:
        token = current_managed_transaction(db.get_connection())
        assert token is not None
        register_transaction_completion(db.get_connection(), token, fail_first)
        register_transaction_completion(
            db.get_connection(),
            token,
            lambda committed: outcomes.append(("second", committed)),
        )
        cursor.execute("CREATE TEMP TABLE observer_commit_probe (value INTEGER)")
        cursor.execute("INSERT INTO observer_commit_probe VALUES (1)")

    assert outcomes == [("first", True), ("second", True)]
    assert (
        db.get_connection()
        .execute("SELECT COUNT(*) FROM observer_commit_probe")
        .fetchone()[0]
        == 1
    )
    assert active_managed_transaction_count() == 0
    assert "CALLBACK-SECRET-CANARY" not in caplog.text


def test_replacement_preparation_does_not_materialize_projection_ancestry(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    descriptors = tuple(
        ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, policy)
        for _ in range(3)
    )
    provenance = _provenance(descriptors)
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[
                    {"role": "user", "content": value} for value in ("A", "B", "C")
                ],
            ),
        )
        nodes = repository.read_surface_nodes(cursor, segment_id)

    checkpoint = service.current_surface_checkpoint(
        segment_id,
        expected_head_id=initial.surface_head_id,
    )
    assert checkpoint is not None
    parent = service._parent_capabilities[id(checkpoint)]

    def fail_materialize(_self: object) -> object:
        raise AssertionError("live preparation materialized full projection")

    monkeypatch.setattr(type(parent.root), "materialize", fail_materialize)
    monkeypatch.setattr(type(parent.descriptors), "materialize", fail_materialize)
    monkeypatch.setattr(
        type(parent.descriptors), "materialize_domains", fail_materialize
    )
    preparation_identity = new_opaque_id()
    replacement = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT,
        policy,
    )
    admission = SurfaceDeltaAdmission(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=initial.surface_head_id,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=preparation_identity,
        descriptors=(replacement,),
        projection_checkpoint=checkpoint,
        replacement_range=VerifiedSurfaceReplacementRange(
            predecessor_head_id=initial.surface_head_id,
            start_node_id=nodes[1].node_id,
            end_node_id=nodes[1].node_id,
            start_sequence=nodes[1].sequence,
            end_sequence=nodes[1].sequence,
            current_ordinal=1,
            component_name="messages_payload",
        ),
    )

    boundary = service.prepare_surface_provenance(
        db.get_connection().cursor(),
        checkpoint,
        provenance=_provenance((replacement,)),
        admission=admission,
        values=({"role": "user", "content": "replacement"},),
    )

    assert len(boundary.provenance.messages_payload) == 3


def test_live_projection_roots_path_copy_without_retaining_replacement_ancestry(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT,
        _policy(),
    )
    provenance = _provenance((descriptor,))
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        persisted = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[{"role": "user", "content": "A"}],
            ),
        )
    checkpoint = service.current_surface_checkpoint(
        segment_id,
        expected_head_id=persisted.surface_head_id,
    )
    assert checkpoint is not None
    parent = service._parent_capabilities[id(checkpoint)]
    root_type = type(parent.root)
    descriptor_root_type = type(parent.descriptors)
    reference = ("rag_context", "omission", "windowed")
    root = root_type(None, base=tuple((index, reference) for index in range(4_000)))
    descriptor_root = descriptor_root_type(
        None,
        base=tuple((index, descriptor) for index in range(4_000)),
        base_domains=tuple((index, "messages_payload") for index in range(4_000)),
    )

    for index in range(400):
        sequence = 4_000 + index
        root = root_type(
            root,
            appended=((sequence, reference),),
            replacement=(index, index),
        )
        descriptor_root = descriptor_root_type(
            descriptor_root,
            appended=((sequence, descriptor),),
            replacement=(index, index),
            appended_domains=((sequence, "messages_payload"),),
            removed_domain_counts=(("messages_payload", 1),),
        )
        assert root.parent is None
        assert descriptor_root.parent is None

    assert len(root.materialize()) == 4_000
    assert len(descriptor_root.materialize()) == 4_000


def test_verified_boundary_persistence_never_refolds_full_descriptors(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    revisions = tuple(
        _revision(db, repository, owner.conversation_id, content=value)
        for value in ("A", "B")
    )
    service = ConsoleTraceService(repository)
    first_descriptor = SavedRevisionTraceProvenance(revisions[0])
    first_provenance = _provenance((first_descriptor,))
    with db.transaction() as cursor:
        first = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=first_provenance,
            bundle=_available_bundle(
                first_provenance,
                messages=[{"role": "user", "content": "A"}],
            ),
        )
    next_descriptor = SavedRevisionTraceProvenance(revisions[1])
    checkpoint = service.current_surface_checkpoint(
        segment_id,
        expected_head_id=first.surface_head_id,
    )
    preparation_identity = new_opaque_id()
    admission = SurfaceDeltaAdmission(
        owner_id,
        segment_id,
        first.surface_head_id,
        ConsoleRequestRoute.FRESH.value,
        preparation_identity,
        (next_descriptor,),
        checkpoint,
    )
    boundary = service.prepare_surface_provenance(
        db.get_connection().cursor(),
        checkpoint,
        provenance=_provenance((next_descriptor,)),
        admission=admission,
        values=({"role": "user", "content": "B"},),
    )
    values = {
        "api_endpoint": "openai",
        "model": "gpt-test",
        **boundary._provider_request_surface_values(),
    }
    projection_type = type(boundary.provenance.messages_payload)
    descriptor_root_type = type(
        service._prepared_capabilities[id(boundary.provenance)].descriptors
    )

    def fail_refold(_self: object) -> object:
        raise AssertionError("full descriptor projection refolded during verification")

    monkeypatch.setattr(projection_type, "__iter__", fail_refold)
    monkeypatch.setattr(descriptor_root_type, "materialize", fail_refold)
    bundle = verify_provider_request_shadow(
        actual_kwargs=values,
        expected_kwargs=dict(values),
        provenance=boundary.provenance,
        project_handler_kwargs=lambda kwargs: kwargs,
        endpoint_identity="https://api.example.test/v1",
        preparation_identity=preparation_identity,
        surface_boundary=boundary,
    )
    assert bundle.available
    monkeypatch.setattr(
        ProviderRequestShadowBundle,
        "boundary_kwargs",
        property(lambda _self: (_ for _ in ()).throw(AssertionError("full thaw"))),
    )
    delta = build_verified_surface_delta(
        boundary.provenance,
        bundle,
        admission=admission,
    )
    with db.transaction() as cursor:
        persisted = service.persist_request(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=boundary.provenance,
            bundle=bundle,
            surface_delta=delta,
        )
    assert len(persisted.appended_nodes) == 1


def test_prepare_rejects_foreign_saved_revision_before_returning_boundary(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    first_revision = _revision(db, repository, owner.conversation_id, content="owned")
    service = ConsoleTraceService(repository)
    first_provenance = _provenance((SavedRevisionTraceProvenance(first_revision),))
    with db.transaction() as cursor:
        first = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=first_provenance,
            bundle=_available_bundle(
                first_provenance,
                messages=[{"role": "user", "content": "owned"}],
            ),
        )
    foreign_conversation = db.add_conversation({"title": "foreign"})
    assert foreign_conversation is not None
    foreign_revision = _revision(
        db, repository, foreign_conversation, content="private foreign body"
    )
    descriptor = SavedRevisionTraceProvenance(foreign_revision)
    checkpoint = service.current_surface_checkpoint(
        segment_id, expected_head_id=first.surface_head_id
    )
    preparation_identity = new_opaque_id()
    admission = SurfaceDeltaAdmission(
        owner_id,
        segment_id,
        first.surface_head_id,
        ConsoleRequestRoute.FRESH.value,
        preparation_identity,
        (descriptor,),
        checkpoint,
    )
    with pytest.raises(ValueError, match="revision_owner_mismatch"):
        service.prepare_surface_provenance(
            db.get_connection().cursor(),
            checkpoint,
            provenance=_provenance((descriptor,)),
            admission=admission,
            values=({"role": "user", "content": "private foreign body"},),
        )


def test_saved_revision_boundary_uses_canonical_multimodal_projection(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    first_revision = _revision(db, repository, owner.conversation_id, content="A")
    image_message = db.add_message(
        {
            "conversation_id": owner.conversation_id,
            "sender": "user",
            "content": "B",
            "image_data": b"image-bytes",
            "image_mime_type": "image/png",
        }
    )
    assert image_message is not None
    image_revision_row = (
        db.get_connection()
        .execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
             WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1""",
            (image_message,),
        )
        .fetchone()
    )
    assert image_revision_row is not None
    image_descriptor = SavedRevisionTraceProvenance(str(image_revision_row[0]))
    service = ConsoleTraceService(repository)
    first_provenance = _provenance((SavedRevisionTraceProvenance(first_revision),))
    with db.transaction() as cursor:
        first = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=first_provenance,
            bundle=_available_bundle(
                first_provenance,
                messages=[{"role": "user", "content": "A"}],
            ),
        )
    checkpoint = service.current_surface_checkpoint(
        segment_id, expected_head_id=first.surface_head_id
    )
    preparation_identity = new_opaque_id()
    admission = SurfaceDeltaAdmission(
        owner_id,
        segment_id,
        first.surface_head_id,
        ConsoleRequestRoute.FRESH.value,
        preparation_identity,
        (image_descriptor,),
        checkpoint,
    )
    boundary = service.prepare_surface_provenance(
        db.get_connection().cursor(),
        checkpoint,
        provenance=_provenance((image_descriptor,)),
        admission=admission,
        values=({"role": "user", "content": "caller value is ignored"},),
    )
    expected_image = {
        "role": "user",
        "content": [
            {"type": "text", "text": "B"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aW1hZ2UtYnl0ZXM="},
            },
        ],
    }
    assert list(boundary.messages_payload) == [
        {"role": "user", "content": "A"},
        expected_image,
    ]


def test_reopen_folds_refs_once_then_uses_current_head_cache(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    revision_ids = tuple(
        _revision(db, repository, owner.conversation_id, content=str(index))
        for index in range(3)
    )
    first_provenance = _provenance((SavedRevisionTraceProvenance(revision_ids[0]),))
    with db.transaction() as cursor:
        first = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=first_provenance,
            bundle=_available_bundle(
                first_provenance,
                messages=[{"role": "user", "content": "0"}],
            ),
        )

    reopened_service = ConsoleTraceService(repository)
    reference_key_calls = 0
    original_reference_key = ConsoleTraceService._reference_key

    def count_reference_keys(*args: object, **kwargs: object):
        nonlocal reference_key_calls
        reference_key_calls += 1
        return original_reference_key(*args, **kwargs)

    monkeypatch.setattr(ConsoleTraceService, "_reference_key", count_reference_keys)
    previous_head = first.surface_head_id
    statements: list[str] = []
    for length in (2, 3):
        descriptors = tuple(
            SavedRevisionTraceProvenance(revision_id)
            for revision_id in revision_ids[:length]
        )
        provenance = _provenance(descriptors)
        bundle = _available_bundle(
            provenance,
            messages=[
                {"role": "user", "content": str(index)} for index in range(length)
            ],
        )
        with db.transaction() as cursor:
            db.get_connection().set_trace_callback(statements.append)
            persisted = _persist(
                reopened_service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
                append_from_index=length - 1,
                previous_surface_head_id=previous_head,
            )
            db.get_connection().set_trace_callback(None)
        previous_head = persisted.surface_head_id

    full_fold_queries = [
        statement
        for statement in statements
        if "FROM console_trace_surface_nodes" in statement
        and "ORDER BY sequence, node_id" in statement
    ]
    assert len(full_fold_queries) == 1
    assert reference_key_calls == 1


def test_verified_reopen_projection_batches_saved_revision_reads(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    revision_ids = tuple(
        _revision(db, repository, owner.conversation_id, content=str(index))
        for index in range(22)
    )
    service = ConsoleTraceService(repository)
    first_descriptors = tuple(
        SavedRevisionTraceProvenance(revision_id) for revision_id in revision_ids[:20]
    )
    first_provenance = _provenance(first_descriptors)
    with db.transaction() as cursor:
        first = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=first_provenance,
            bundle=_available_bundle(
                first_provenance,
                messages=[
                    {"role": "user", "content": str(index)} for index in range(20)
                ],
            ),
        )
        descriptors = first_descriptors + (
            SavedRevisionTraceProvenance(revision_ids[20]),
        )
        provenance = _provenance(descriptors)
        reopened = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[
                    {"role": "user", "content": str(index)} for index in range(21)
                ],
            ),
            append_from_index=20,
            previous_surface_head_id=first.surface_head_id,
        )
        final_descriptors = descriptors + (
            SavedRevisionTraceProvenance(revision_ids[21]),
        )
        final_provenance = _provenance(final_descriptors)
        statements: list[str] = []
        db.get_connection().set_trace_callback(statements.append)
        _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=final_provenance,
            bundle=_available_bundle(
                final_provenance,
                messages=[
                    {"role": "user", "content": str(index)} for index in range(22)
                ],
            ),
            append_from_index=21,
            previous_surface_head_id=reopened.surface_head_id,
        )
        db.get_connection().set_trace_callback(None)

    message_reads = [
        statement
        for statement in statements
        if "SELECT" in statement and "FROM messages" in statement
    ]
    attachment_reads = [
        statement
        for statement in statements
        if "SELECT" in statement and "FROM message_attachments" in statement
    ]
    assert len(message_reads) <= 3
    assert len(attachment_reads) <= 3


def test_verified_reopen_projection_batches_artifact_identity_reads(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    descriptors = tuple(
        ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, policy)
        for _ in range(21)
    )
    first_provenance = _provenance(descriptors[:20])
    with db.transaction() as cursor:
        first = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=first_provenance,
            bundle=_available_bundle(
                first_provenance,
                messages=[
                    {"role": "user", "content": f"artifact-{index}"}
                    for index in range(20)
                ],
            ),
        )

        provenance = _provenance(descriptors)
        statements: list[str] = []
        db.get_connection().set_trace_callback(statements.append)
        _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[
                    {"role": "user", "content": f"artifact-{index}"}
                    for index in range(21)
                ],
            ),
            append_from_index=20,
            previous_surface_head_id=first.surface_head_id,
        )
        db.get_connection().set_trace_callback(None)

    identity_reads = [
        statement
        for statement in statements
        if "FROM console_trace_artifacts" in statement
        and "identity_digest" in statement
    ]
    batched_reads = [
        statement
        for statement in statements
        if "FROM console_trace_artifacts" in statement and "artifact_id IN" in statement
    ]
    assert len(identity_reads) <= 2
    assert len(batched_reads) <= 2


def test_continuation_domain_survives_append_and_reopen(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    values = tuple(_continuation_value(str(index)) for index in range(3))
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    message_id = db.add_message(
        {
            "conversation_id": owner.conversation_id,
            "sender": "assistant",
            "content": "visible",
            "provider_continuation_json": json.dumps(values[0]),
        }
    )
    assert message_id is not None
    row = (
        db.get_connection()
        .execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
             WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1""",
            (message_id,),
        )
        .fetchone()
    )
    assert row is not None
    descriptors = (
        DerivedTraceProvenance(
            TraceTransformKind.CONTINUATION_ATTACHMENT,
            (SavedRevisionTraceProvenance(str(row[0])),),
        ),
        *tuple(
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.CONTINUATION,
                policy,
            )
            for _ in range(2)
        ),
    )

    def provenance_for(items: tuple[object, ...]) -> ProviderRequestProvenance:
        return ProviderRequestProvenance(
            continuations=items,
            metadata=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        )

    first_provenance = provenance_for(descriptors[:1])
    first_bundle = _available_bundle(
        first_provenance,
        messages=[],
        continuations=list(values[:1]),
    )
    with db.transaction() as cursor:
        first = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=first_provenance,
            bundle=first_bundle,
        )

    service = ConsoleTraceService(repository)
    full_two = provenance_for(descriptors[:2])
    full_two_bundle = _available_bundle(
        full_two,
        messages=[],
        continuations=list(values[:2]),
    )
    with db.transaction() as cursor:
        second = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=full_two,
            bundle=full_two_bundle,
            append_from_index=1,
            previous_surface_head_id=first.surface_head_id,
        )
        checkpoint = service.current_surface_checkpoint(
            segment_id, expected_head_id=second.surface_head_id
        )
        assert checkpoint is not None
        full_three = provenance_for(descriptors)
        unbound = _available_bundle(
            full_three,
            messages=[],
            continuations=list(values),
        )
        admission = SurfaceDeltaAdmission(
            owner_id=owner_id,
            segment_id=segment_id,
            predecessor_surface_head_id=second.surface_head_id,
            route_identity=ConsoleRequestRoute.FRESH.value,
            preparation_identity=unbound.preparation_identity or "missing",
            descriptors=descriptors[2:],
            projection_checkpoint=checkpoint,
        )
        boundary = service.prepare_surface_provenance(
            cursor,
            checkpoint,
            provenance=provenance_for(descriptors[2:]),
            admission=admission,
            values=values[2:],
        )
        actual = unbound.boundary_kwargs
        actual.update(boundary._provider_request_surface_values())
        bundle = verify_provider_request_shadow(
            actual_kwargs=actual,
            expected_kwargs=dict(actual),
            provenance=boundary.provenance,
            project_handler_kwargs=lambda kwargs: kwargs,
            endpoint_identity=unbound.endpoint_identity,
            preparation_identity=unbound.preparation_identity,
            surface_boundary=boundary,
        )
        assert bundle.available
        delta = build_verified_surface_delta(
            boundary.provenance,
            bundle,
            admission=admission,
        )
        assert delta.items[0].component_name == "provider_continuations"
        third = service.persist_request(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=boundary.provenance,
            bundle=bundle,
            surface_delta=delta,
        )
        nodes = repository.read_surface_nodes(cursor, segment_id)

    assert third.surface_head_id == nodes[-1].node_id
    assert [node.component_kind for node in nodes] == [
        "continuation",
        "continuation",
        "continuation",
    ]


def test_saved_continuation_value_mismatch_rejects_before_any_trace_write(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    canonical = _continuation_value("canonical")
    message_id = db.add_message(
        {
            "conversation_id": owner.conversation_id,
            "sender": "assistant",
            "content": "visible",
            "provider_continuation_json": json.dumps(canonical),
        }
    )
    row = (
        db.get_connection()
        .execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
             WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1""",
            (message_id,),
        )
        .fetchone()
    )
    assert row is not None
    descriptor = DerivedTraceProvenance(
        TraceTransformKind.CONTINUATION_ATTACHMENT,
        (SavedRevisionTraceProvenance(str(row[0])),),
    )
    provenance = ProviderRequestProvenance(
        continuations=(descriptor,),
        metadata=(request_route_provenance(ConsoleRequestRoute.FRESH),),
    )
    forged = _continuation_value("forged")
    service = ConsoleTraceService(repository)
    preparation_identity = new_opaque_id()
    admission = SurfaceDeltaAdmission(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=None,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=preparation_identity,
        descriptors=(descriptor,),
    )

    with db.transaction() as cursor:
        boundary = service.prepare_surface_provenance(
            cursor,
            None,
            provenance=provenance,
            admission=admission,
            values=(forged,),
        )
        actual = {
            "api_endpoint": "openai",
            "model": "gpt-test",
            **boundary._provider_request_surface_values(),
        }
        bundle = verify_provider_request_shadow(
            actual_kwargs=actual,
            expected_kwargs=dict(actual),
            provenance=boundary.provenance,
            project_handler_kwargs=lambda kwargs: kwargs,
            endpoint_identity="https://api.example.test/v1",
            preparation_identity=preparation_identity,
            surface_boundary=boundary,
        )
        assert bundle.available is False
        assert bundle.omission_reason is TraceOmissionReason.ALIGNMENT_MISMATCH

    cursor = db.get_connection().cursor()
    assert repository.read_surface_nodes(cursor, segment_id) == ()
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_artifacts").fetchone()[0]
        == 0
    )
    assert (
        cursor.execute("SELECT COUNT(*) FROM console_trace_request_headers").fetchone()[
            0
        ]
        == 0
    )


def test_current_surface_delta_preserves_distinct_logical_message_projection(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    logical_message = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.MANDATORY_CONTEXT,
        policy,
    )
    first_wire = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST,
        policy,
    )
    second_wire = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST,
        policy,
    )
    initial_provenance = ProviderRequestProvenance(
        messages=(logical_message,),
        messages_payload=(first_wire,),
        metadata=(request_route_provenance(ConsoleRequestRoute.FRESH),),
    )
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=initial_provenance,
            bundle=_available_bundle(
                initial_provenance,
                messages=[{"role": "user", "content": "first"}],
            ),
        )
        extended_provenance = replace(
            initial_provenance,
            messages_payload=(first_wire, second_wire),
        )
        _admission, boundary = service.prepare_current_surface_delta(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            route_identity=ConsoleRequestRoute.FRESH.value,
            preparation_identity=new_opaque_id(),
            provenance=extended_provenance,
            values=(
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ),
        )

    assert boundary.provenance.messages == (logical_message,)


def test_current_surface_delta_appends_message_before_unchanged_continuation(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    first_message = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST,
        policy,
    )
    second_message = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST,
        policy,
    )
    third_message = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST,
        policy,
    )
    continuation = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.CONTINUATION,
        policy,
    )
    first_value = {"role": "user", "content": "first"}
    second_value = {"role": "user", "content": "second"}
    third_value = {"role": "user", "content": "third"}
    continuation_value = _continuation_value("unchanged")
    initial_provenance = ProviderRequestProvenance(
        messages=(first_message,),
        messages_payload=(first_message,),
        continuations=(continuation,),
        metadata=(request_route_provenance(ConsoleRequestRoute.FRESH),),
    )
    service = ConsoleTraceService(repository)

    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=initial_provenance,
            bundle=_available_bundle(
                initial_provenance,
                messages=[first_value],
                continuations=[continuation_value],
            ),
        )
        extended_provenance = replace(
            initial_provenance,
            messages=(first_message, second_message),
            messages_payload=(first_message, second_message),
        )
        bundle = _available_bundle(
            extended_provenance,
            messages=[first_value, second_value],
            continuations=[continuation_value],
        )
        admission, boundary = service.prepare_current_surface_delta(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            route_identity=ConsoleRequestRoute.FRESH.value,
            preparation_identity=bundle.preparation_identity or "missing",
            provenance=extended_provenance,
            values=(first_value, second_value, continuation_value),
        )
        actual = dict(bundle.boundary_kwargs)
        actual.update(boundary._provider_request_surface_values())
        bound_bundle = verify_provider_request_shadow(
            actual_kwargs=actual,
            expected_kwargs=dict(actual),
            provenance=boundary.provenance,
            project_handler_kwargs=lambda kwargs: kwargs,
            endpoint_identity=bundle.endpoint_identity,
            preparation_identity=bundle.preparation_identity,
            surface_boundary=boundary,
        )
        second_persisted = service.persist_request(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=boundary.provenance,
            bundle=bound_bundle,
            surface_delta=build_verified_surface_delta(
                boundary.provenance,
                bound_bundle,
                admission=admission,
            ),
        )
        third_provenance = replace(
            initial_provenance,
            messages=(first_message, second_message, third_message),
            messages_payload=(first_message, second_message, third_message),
        )
        third_bundle = _available_bundle(
            third_provenance,
            messages=[first_value, second_value, third_value],
            continuations=[continuation_value],
        )
        third_admission, third_boundary = service.prepare_current_surface_delta(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            route_identity=ConsoleRequestRoute.FRESH.value,
            preparation_identity=third_bundle.preparation_identity or "missing",
            provenance=third_provenance,
            values=(first_value, second_value, third_value, continuation_value),
        )
        actual = dict(third_bundle.boundary_kwargs)
        actual.update(third_boundary._provider_request_surface_values())
        bound_third_bundle = verify_provider_request_shadow(
            actual_kwargs=actual,
            expected_kwargs=dict(actual),
            provenance=third_boundary.provenance,
            project_handler_kwargs=lambda kwargs: kwargs,
            endpoint_identity=third_bundle.endpoint_identity,
            preparation_identity=third_bundle.preparation_identity,
            surface_boundary=third_boundary,
        )
        third_persisted = service.persist_request(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=third_boundary.provenance,
            bundle=bound_third_bundle,
            surface_delta=build_verified_surface_delta(
                third_boundary.provenance,
                bound_third_bundle,
                admission=third_admission,
            ),
        )

    assert admission.predecessor_surface_head_id == initial.surface_head_id
    assert admission.descriptors == (second_message,)
    assert admission.replacement_range is None
    assert tuple(boundary.provenance.messages_payload) == (
        first_message,
        second_message,
    )
    assert tuple(boundary.provenance.continuations) == (continuation,)
    assert [node.component_kind for node in second_persisted.appended_nodes] == [
        "active_request"
    ]
    assert third_admission.descriptors == (third_message,)
    assert third_admission.replacement_range is None
    assert tuple(third_boundary.provenance.messages_payload) == (
        first_message,
        second_message,
        third_message,
    )
    assert tuple(third_boundary.provenance.continuations) == (continuation,)
    assert [node.component_kind for node in third_persisted.appended_nodes] == [
        "active_request"
    ]


def test_interleaved_domains_reopen_and_replace_continuation_by_local_ordinal(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    message_a = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST, policy
    )
    message_b = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST, policy
    )
    continuation = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.CONTINUATION, policy
    )
    replacement_continuation = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.CONTINUATION, policy
    )
    message_values = (
        {"role": "user", "content": "A"},
        {"role": "user", "content": "B"},
    )
    continuation_values = (
        _continuation_value("old"),
        _continuation_value("replacement"),
    )

    def provenance_for(
        messages: tuple[object, ...], continuations: tuple[object, ...]
    ) -> ProviderRequestProvenance:
        return ProviderRequestProvenance(
            messages=messages,
            messages_payload=messages,
            continuations=continuations,
            metadata=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        )

    initial_provenance = provenance_for((message_a,), ())
    initial_bundle = _available_bundle(
        initial_provenance,
        messages=[message_values[0]],
    )
    with db.transaction() as cursor:
        initial = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=initial_provenance,
            bundle=initial_bundle,
        )

    continuation_provenance = provenance_for((message_a,), (continuation,))
    continuation_bundle = _available_bundle(
        continuation_provenance,
        messages=[message_values[0]],
        continuations=[continuation_values[0]],
    )
    with db.transaction() as cursor:
        continued = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=continuation_provenance,
            bundle=continuation_bundle,
            append_from_index=1,
            previous_surface_head_id=initial.surface_head_id,
        )

    interleaved_provenance = provenance_for((message_a, message_b), (continuation,))
    interleaved_bundle = _available_bundle(
        interleaved_provenance,
        messages=list(message_values),
        continuations=[continuation_values[0]],
    )
    interleaved_admission = SurfaceDeltaAdmission(
        owner_id=owner_id,
        segment_id=segment_id,
        predecessor_surface_head_id=continued.surface_head_id,
        route_identity=ConsoleRequestRoute.FRESH.value,
        preparation_identity=interleaved_bundle.preparation_identity or "missing",
        descriptors=(message_b,),
    )
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        boundary = service.prepare_surface_provenance(
            cursor,
            None,
            provenance=interleaved_provenance,
            admission=interleaved_admission,
            values=(*message_values, continuation_values[0]),
        )
        actual = interleaved_bundle.boundary_kwargs
        actual.update(boundary._provider_request_surface_values())
        bound_bundle = verify_provider_request_shadow(
            actual_kwargs=actual,
            expected_kwargs=dict(actual),
            provenance=boundary.provenance,
            project_handler_kwargs=lambda kwargs: kwargs,
            endpoint_identity=interleaved_bundle.endpoint_identity,
            preparation_identity=interleaved_bundle.preparation_identity,
            surface_boundary=boundary,
        )
        interleaved = service.persist_request(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=boundary.provenance,
            bundle=bound_bundle,
            surface_delta=build_verified_surface_delta(
                boundary.provenance,
                bound_bundle,
                admission=interleaved_admission,
            ),
        )
        nodes = repository.read_surface_nodes(cursor, segment_id)
        assert [node.component_kind for node in nodes] == [
            "active_request",
            "continuation",
            "active_request",
        ]
        replacement_provenance = provenance_for(
            (message_a, message_b), (replacement_continuation,)
        )
        unbound = _available_bundle(
            replacement_provenance,
            messages=list(message_values),
            continuations=[continuation_values[1]],
        )
        admission = SurfaceDeltaAdmission(
            owner_id=owner_id,
            segment_id=segment_id,
            predecessor_surface_head_id=interleaved.surface_head_id,
            route_identity=ConsoleRequestRoute.FRESH.value,
            preparation_identity=unbound.preparation_identity or "missing",
            descriptors=(replacement_continuation,),
            replacement_range=VerifiedSurfaceReplacementRange(
                predecessor_head_id=interleaved.surface_head_id,
                start_node_id=nodes[1].node_id,
                end_node_id=nodes[1].node_id,
                start_sequence=nodes[1].sequence,
                end_sequence=nodes[1].sequence,
                current_ordinal=1,
                component_name="provider_continuations",
                component_ordinal=None,
            ),
        )
        checkpoint = service.current_surface_checkpoint(
            segment_id,
            expected_head_id=interleaved.surface_head_id,
        )
        assert checkpoint is not None
        admission = replace(admission, projection_checkpoint=checkpoint)
        boundary = service.prepare_surface_provenance(
            cursor,
            checkpoint,
            provenance=provenance_for((), (replacement_continuation,)),
            admission=admission,
            values=(continuation_values[1],),
        )
        actual = unbound.boundary_kwargs
        actual.update(boundary._provider_request_surface_values())
        bound_bundle = verify_provider_request_shadow(
            actual_kwargs=actual,
            expected_kwargs=dict(actual),
            provenance=boundary.provenance,
            project_handler_kwargs=lambda kwargs: kwargs,
            endpoint_identity=unbound.endpoint_identity,
            preparation_identity=unbound.preparation_identity,
            surface_boundary=boundary,
        )
        delta = build_verified_surface_delta(
            boundary.provenance,
            bound_bundle,
            admission=admission,
        )
        assert delta.replacement is not None
        assert delta.replacement.current_ordinal == 1
        assert delta.replacement.item.component_name == "provider_continuations"
        assert delta.replacement.item.ordinal == 0
        replaced = service.persist_request(
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=boundary.provenance,
            bundle=bound_bundle,
            surface_delta=delta,
        )

    assert replaced.appended_nodes[-1].component_kind == "continuation"


def test_replacement_rebases_tool_loop_overlay_at_replaced_message(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    descriptors = tuple(
        ProviderArtifactTraceProvenance(TraceProvenanceSource.TOOL_CALL, policy)
        for _ in range(2)
    )
    initial_provenance = replace(_provenance(descriptors), tool_loop=(0,))
    initial_bundle = _available_bundle(
        initial_provenance,
        messages=[
            {"role": "tool", "content": "old"},
            {"role": "user", "content": "next"},
        ],
    )
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=initial_provenance,
            bundle=initial_bundle,
        )
        nodes = repository.read_surface_nodes(cursor, segment_id)
        replacement_descriptor = ProviderArtifactTraceProvenance(
            TraceProvenanceSource.ACTIVE_REQUEST, policy
        )
        replacement_provenance = _provenance((replacement_descriptor,))
        replacement_bundle = _available_bundle(
            replacement_provenance,
            messages=[{"role": "user", "content": "replacement"}],
        )
        checkpoint = service.current_surface_checkpoint(
            segment_id, expected_head_id=initial.surface_head_id
        )
        assert checkpoint is not None
        admission = SurfaceDeltaAdmission(
            owner_id=owner_id,
            segment_id=segment_id,
            predecessor_surface_head_id=initial.surface_head_id,
            route_identity=ConsoleRequestRoute.FRESH.value,
            preparation_identity=replacement_bundle.preparation_identity or "missing",
            descriptors=(replacement_descriptor,),
            projection_checkpoint=checkpoint,
            replacement_range=VerifiedSurfaceReplacementRange(
                predecessor_head_id=initial.surface_head_id,
                start_node_id=nodes[0].node_id,
                end_node_id=nodes[0].node_id,
                start_sequence=0,
                end_sequence=0,
                current_ordinal=0,
            ),
        )
        boundary = service.prepare_surface_provenance(
            cursor,
            checkpoint,
            provenance=replacement_provenance,
            admission=admission,
            values=({"role": "user", "content": "replacement"},),
        )

    assert boundary.provenance.tool_loop == ()


def test_replacement_validates_range_reopens_and_is_bounded(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    descriptors = tuple(
        ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, policy)
        for _ in range(200)
    )
    provenance = _provenance(descriptors)
    bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": f"rag-{index}"} for index in range(200)],
    )
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        )
        nodes = repository.read_surface_nodes(cursor, segment_id, page_size=200)
        replacement_descriptor = ProviderArtifactTraceProvenance(
            TraceProvenanceSource.CONTEXT_SUMMARY, policy
        )
        replacement_descriptors = (replacement_descriptor, *descriptors[150:])
        replacement_provenance = _provenance(replacement_descriptors)
        replacement_bundle = _available_bundle(
            replacement_provenance,
            messages=[
                {"role": "user", "content": "summary"},
                *[
                    {"role": "user", "content": f"rag-{index}"}
                    for index in range(150, 200)
                ],
            ],
        )
        replaced = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=replacement_provenance,
            bundle=replacement_bundle,
            replacement=SurfaceReplacementPlan(
                predecessor_head_id=initial.surface_head_id,
                start_node_id=nodes[0].node_id,
                end_node_id=nodes[149].node_id,
                start_sequence=0,
                end_sequence=149,
                replacement_index=0,
            ),
        )

    db.close_connection()
    reopened = db.get_connection().cursor()
    stored = repository.read_surface_replacements(reopened, segment_id)
    assert replaced.replacement is not None
    assert stored == (replaced.replacement,)
    assert stored[0].replacement.start_sequence == 0
    assert stored[0].replacement.end_sequence == 149
    assert not any(
        "history" in field.name for field in fields(SurfaceReplacementRecord)
    )


def test_reopen_replays_overlapping_replacements_in_causal_order(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    descriptors = tuple(
        ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, policy)
        for _ in range(200)
    )
    provenance = _provenance(descriptors)
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[{"role": "user", "content": str(i)} for i in range(200)],
            ),
        )
        nodes = repository.read_surface_nodes(cursor, segment_id, page_size=200)
        summary = ProviderArtifactTraceProvenance(
            TraceProvenanceSource.CONTEXT_SUMMARY, policy
        )
        first_descriptors = (*descriptors[:100], summary, *descriptors[151:])
        summary_provenance = _provenance(first_descriptors)
        summary_bundle = _available_bundle(
            summary_provenance,
            messages=[
                *[{"role": "user", "content": str(i)} for i in range(100)],
                {"role": "user", "content": "first"},
                *[{"role": "user", "content": str(i)} for i in range(151, 200)],
            ],
        )
        first = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=summary_provenance,
            bundle=summary_bundle,
            replacement=SurfaceReplacementPlan(
                initial.surface_head_id,
                nodes[100].node_id,
                nodes[150].node_id,
                100,
                150,
                100,
            ),
        )
        service = ConsoleTraceService(repository)
        reopened_head = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=summary_provenance,
            bundle=summary_bundle,
            append_from_index=len(first_descriptors),
            previous_surface_head_id=first.surface_head_id,
        )
        assert reopened_head.surface_head_id == first.surface_head_id
        second_descriptors = (*descriptors[:50], summary)
        second_provenance = _provenance(second_descriptors)
        second = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=second_provenance,
            bundle=_available_bundle(
                second_provenance,
                messages=[
                    *[{"role": "user", "content": str(i)} for i in range(50)],
                    {"role": "user", "content": "second"},
                ],
            ),
            replacement=SurfaceReplacementPlan(
                first.surface_head_id,
                nodes[50].node_id,
                first.surface_head_id,
                50,
                200,
                50,
            ),
        )
        cached = service._surface_projection(
            cursor,
            segment_id,
            repository.get_surface_node(cursor, second.surface_head_id),
        )
        reopened = ConsoleTraceService(repository)._surface_projection(
            cursor,
            segment_id,
            repository.get_surface_node(cursor, second.surface_head_id),
        )

    assert reopened.entries == cached.entries
    assert len(reopened.entries) == 51


def test_replacement_over_limit_records_typed_omission_without_range_row(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    descriptors = tuple(
        ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, policy)
        for _ in range(257)
    )
    service = ConsoleTraceService(repository)
    provenance = _provenance(descriptors)
    bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": f"seed-{index}"} for index in range(257)],
    )
    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        )
        replacement_provenance = _provenance((descriptors[0],))
        result = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=replacement_provenance,
            bundle=_available_bundle(
                replacement_provenance,
                messages=[{"role": "user", "content": "seed-0"}],
            ),
            replacement=SurfaceReplacementPlan(
                predecessor_head_id=initial.surface_head_id,
                start_node_id=initial.appended_nodes[0].node_id,
                end_node_id=initial.appended_nodes[256].node_id,
                start_sequence=0,
                end_sequence=256,
                replacement_index=0,
            ),
        )
        node = repository.get_surface_node(cursor, result.surface_head_id)

    assert result.replacement is None
    assert node is not None and node.reference_kind == "omission"
    assert node.omission_reason_code == "unsupported_replacement_span"


def test_invalid_over_limit_replacement_fails_before_writing_omission(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    _other_owner_id, other_segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT, _policy()
    )
    provenance = _provenance((descriptor,))
    bundle = _available_bundle(
        provenance, messages=[{"role": "user", "content": "seed"}]
    )
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        initial = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        )
        other = repository.append_surface_node(
            cursor,
            segment_id=other_segment_id,
            sequence=0,
            predecessor_node_id=None,
            component_kind="rag_context",
            reference=TraceOmission("rag_context", "source_unavailable"),
        )
        before = len(repository.read_surface_nodes(cursor, segment_id))
        with pytest.raises(ValueError, match="replacement_range_mismatch"):
            _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
                replacement=SurfaceReplacementPlan(
                    predecessor_head_id=initial.surface_head_id,
                    start_node_id=other.node_id,
                    end_node_id=other.node_id,
                    start_sequence=0,
                    end_sequence=256,
                    replacement_index=0,
                ),
            )
        assert len(repository.read_surface_nodes(cursor, segment_id)) == before


def test_schema_and_models_have_no_prior_history_payload_fields(
    db: CharactersRAGDB,
) -> None:
    forbidden = {"history", "messages", "body", "digest", "source_list"}
    for model in (TraceCallRecord, RequestHeaderRecord, SurfaceReplacementRecord):
        assert forbidden.isdisjoint(field.name for field in fields(model))
    cursor = db.get_connection().cursor()
    for table in (
        "console_trace_calls",
        "console_trace_request_headers",
        "console_trace_surface_replacements",
    ):
        columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
        assert forbidden.isdisjoint(columns)


def test_wrong_owner_revision_and_noncontiguous_replacement_fail_closed(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    other_owner_id, _other_segment_id = _owned_segment(db, repository)
    other = repository.get_owner(db.get_connection().cursor(), other_owner_id)
    assert other is not None and other.conversation_id is not None
    wrong_revision = _revision(
        db, repository, other.conversation_id, content="wrong-domain"
    )
    provenance = _provenance((SavedRevisionTraceProvenance(wrong_revision),))
    service = ConsoleTraceService(repository)

    with pytest.raises(ValueError, match="revision_owner_mismatch"):
        with db.transaction() as cursor:
            _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=_available_bundle(
                    provenance,
                    messages=[{"role": "user", "content": "wrong-domain"}],
                ),
            )

    connection = db.get_connection()
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM console_trace_surface_nodes WHERE segment_id = ?",
            (segment_id,),
        ).fetchone()[0]
        == 0
    )


def test_service_requires_caller_transaction_before_any_write(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    omission = OmittedTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT,
        TraceOmissionReason.SOURCE_UNAVAILABLE,
    )
    provenance = _provenance((omission,))
    bundle = ProviderRequestShadowBundle(
        available=False,
        omission_reason=TraceOmissionReason.SOURCE_UNAVAILABLE,
    )
    cursor = db.get_connection().cursor()
    with pytest.raises(RuntimeError, match="caller_transaction_required"):
        _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
            unavailable_provider_name="openai",
            unavailable_model_name="gpt-test",
            unavailable_endpoint_identity="unavailable",
        )

    assert (
        cursor.execute(
            "SELECT COUNT(*) FROM console_trace_surface_nodes WHERE segment_id = ?",
            (segment_id,),
        ).fetchone()[0]
        == 0
    )
    assert (
        cursor.execute(
            "SELECT COUNT(*) FROM console_trace_events WHERE segment_id = ?",
            (segment_id,),
        ).fetchone()[0]
        == 0
    )


def test_unavailable_bundle_does_not_persist_canary_anywhere(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    canary = "SUPER-SECRET-FIRST-WRITE-CANARY"
    provenance = _provenance(
        (
            OmittedTraceProvenance(
                TraceProvenanceSource.RAG_CONTEXT,
                TraceOmissionReason.ALIGNMENT_MISMATCH,
            ),
        )
    )
    service = ConsoleTraceService(repository)
    bundle = ProviderRequestShadowBundle(
        available=False,
        literal_payload={"model": "leak", "canary": canary},
        omission_reason=TraceOmissionReason.ALIGNMENT_MISMATCH,
    )
    with pytest.raises(ValueError, match="unavailable_bundle_content"):
        with db.transaction() as cursor:
            _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
                unavailable_provider_name="openai",
                unavailable_model_name="gpt-test",
                unavailable_endpoint_identity="unavailable",
            )
    with db.transaction() as cursor:
        dump = b"\n".join(
            bytes(row[0]) if isinstance(row[0], bytes) else str(row[0]).encode()
            for table in (
                "console_trace_artifacts",
                "console_trace_request_headers",
                "console_trace_surface_nodes",
                "console_trace_events",
            )
            for row in cursor.execute(f"SELECT * FROM {table}")
        )
        assert not dump
    assert canary.encode() not in dump


def test_repository_tail_helpers_use_desc_limit_one(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    _ = owner_id
    statements: list[str] = []
    connection = db.get_connection()
    connection.set_trace_callback(statements.append)
    try:
        cursor = connection.cursor()
        assert repository.get_surface_tail(cursor, segment_id) is None
        assert repository.get_event_tail(cursor, segment_id) is None
    finally:
        connection.set_trace_callback(None)
    normalized = [" ".join(statement.split()).upper() for statement in statements]
    assert any(
        "ORDER BY SEQUENCE DESC" in item and "LIMIT 1" in item for item in normalized
    )
    assert sum("LIMIT 1" in item for item in normalized) >= 2


def test_system_policy_change_changes_header_without_duplicating_artifact(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    revision_id = _revision(db, repository, owner.conversation_id, content="hello")
    saved = SavedRevisionTraceProvenance(revision_id)
    service = ConsoleTraceService(repository)
    headers = []
    component_artifacts = []
    previous_head: str | None = None
    for index in range(2):
        system = ProviderArtifactTraceProvenance(
            TraceProvenanceSource.RENDERED_SYSTEM,
            _policy(),
        )
        provenance = _provenance((saved,), system=system)
        bundle = _available_bundle(
            provenance,
            messages=[{"role": "user", "content": "hello"}],
            system="same rendered system",
        )
        with db.transaction() as cursor:
            persisted = _persist(
                service,
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
                append_from_index=index,
                previous_surface_head_id=previous_head,
            )
            headers.append(persisted.header.header_id)
            component_artifacts.append(persisted.header.components[0].artifact_id)
            previous_head = persisted.surface_head_id

    assert headers[0] != headers[1]
    assert component_artifacts[0] == component_artifacts[1]


def test_literal_payload_persists_only_non_history_envelope(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    revision_id = _revision(
        db, repository, owner.conversation_id, content="LITERAL-HISTORY-CANARY"
    )
    provenance = _provenance((SavedRevisionTraceProvenance(revision_id),))
    values = {
        "api_endpoint": "llama_cpp",
        "messages_payload": [{"role": "user", "content": "LITERAL-HISTORY-CANARY"}],
        "model": "local-model",
        "streaming": False,
    }
    bundle = verify_provider_request_shadow(
        actual_kwargs=values,
        expected_kwargs=dict(values),
        provenance=provenance,
        project_handler_kwargs=lambda kwargs: kwargs,
        literal_payload={
            "model": "local-model",
            "messages": [{"role": "user", "content": "LITERAL-HISTORY-CANARY"}],
            "stream": False,
        },
        endpoint_identity="http://127.0.0.1:8080/v1/chat/completions",
    )
    with db.transaction() as cursor:
        persisted = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        )
        literal_component = next(
            item
            for item in persisted.header.components
            if item.component_kind == "provider_literal_envelope"
        )
        artifact = repository.get_artifact(cursor, literal_component.artifact_id)
        assert artifact is not None
        literal_envelope = json.loads(artifact.sanitized_bytes)
        all_artifacts = b"\n".join(
            bytes(row[0])
            for row in cursor.execute(
                "SELECT sanitized_bytes FROM console_trace_artifacts"
            )
        )

    assert literal_envelope == {"model": "local-model", "stream": False}
    assert persisted.header.adapter_defaults["literal_surface_field"] == "messages"
    assert b"LITERAL-HISTORY-CANARY" not in all_artifacts


def test_saved_system_framing_uses_revision_refs_not_header_artifact(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    system_revision = _revision(
        db, repository, owner.conversation_id, content="SAVED-SYSTEM-CANARY"
    )
    second_system_revision = _revision(
        db, repository, owner.conversation_id, content="SECOND-SYSTEM-CANARY"
    )
    message_revision = _revision(db, repository, owner.conversation_id, content="hi")
    system = DerivedTraceProvenance(
        TraceTransformKind.SINGLE_PREAMBLE,
        (
            SavedRevisionTraceProvenance(system_revision),
            SavedRevisionTraceProvenance(second_system_revision),
            SavedRevisionTraceProvenance(system_revision),
        ),
    )
    message = SavedRevisionTraceProvenance(message_revision)
    provenance = _provenance((message,), system=system)
    bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": "hi"}],
        system=("SAVED-SYSTEM-CANARY\n\nSECOND-SYSTEM-CANARY\n\nSAVED-SYSTEM-CANARY"),
        system_components=(
            "SAVED-SYSTEM-CANARY",
            "SECOND-SYSTEM-CANARY",
            "SAVED-SYSTEM-CANARY",
        ),
    )
    with db.transaction() as cursor:
        service = ConsoleTraceService(repository)
        header = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        ).header
        reconstructed = service.reconstruct_header(
            cursor,
            header.header_id,
        )
        artifact_bytes = b"\n".join(
            bytes(row[0])
            for row in cursor.execute(
                "SELECT sanitized_bytes FROM console_trace_artifacts"
            )
        )
        surface_nodes = repository.read_surface_nodes(cursor, segment_id)

    assert "system_surface_head_id" not in header.adapter_defaults
    assert [node.semantic_revision_id for node in surface_nodes] == [message_revision]
    assert reconstructed.system_revision_ids == (
        system_revision,
        second_system_revision,
        system_revision,
    )
    assert reconstructed.system_composition == (
        {"kind": "transform_start", "transform": "single_preamble"},
        {"kind": "revision", "revision_id": system_revision},
        {"kind": "revision", "revision_id": second_system_revision},
        {"kind": "revision", "revision_id": system_revision},
        {"kind": "transform_end", "transform": "single_preamble"},
    )
    assert not any(
        component.component_kind == "rendered_system" for component in header.components
    )
    assert b"SAVED-SYSTEM-CANARY" not in artifact_bytes


def test_mixed_single_preamble_reconstructs_ordered_system_composition(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    owner = repository.get_owner(db.get_connection().cursor(), owner_id)
    assert owner is not None and owner.conversation_id is not None
    revision_id = _revision(
        db, repository, owner.conversation_id, content="SAVED-MIXED-CANARY"
    )
    policy = _policy()
    system = DerivedTraceProvenance(
        TraceTransformKind.SINGLE_PREAMBLE,
        (
            SavedRevisionTraceProvenance(revision_id),
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RENDERED_SYSTEM, policy
            ),
            OmittedTraceProvenance(
                TraceProvenanceSource.PROJECT_INSTRUCTION,
                TraceOmissionReason.WINDOWED,
            ),
        ),
    )
    message = OmittedTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT,
        TraceOmissionReason.SOURCE_UNAVAILABLE,
    )
    provenance = _provenance((message,), system=system)
    bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": "omitted"}],
        system="SAVED-MIXED-CANARY\n\nPROVIDER-SYSTEM-CANARY",
        system_components=("SAVED-MIXED-CANARY", "PROVIDER-SYSTEM-CANARY"),
    )
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        persisted = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        )
        reconstructed = service.reconstruct_header(cursor, persisted.header.header_id)
        artifact_bytes = b"\n".join(
            bytes(row[0])
            for row in cursor.execute(
                "SELECT sanitized_bytes FROM console_trace_artifacts"
            )
        )

    assert reconstructed.system_composition == (
        {"kind": "transform_start", "transform": "single_preamble"},
        {"kind": "revision", "revision_id": revision_id},
        {"kind": "artifact", "component_ordinal": 0},
        {
            "kind": "omission",
            "source": "project_instruction",
            "reason": "windowed",
        },
        {"kind": "transform_end", "transform": "single_preamble"},
    )
    assert b"PROVIDER-SYSTEM-CANARY" in artifact_bytes
    assert b"SAVED-MIXED-CANARY" not in artifact_bytes


def test_mixed_system_parts_must_compose_verified_provider_value() -> None:
    policy = _policy()
    system = DerivedTraceProvenance(
        TraceTransformKind.SINGLE_PREAMBLE,
        (
            SavedRevisionTraceProvenance(new_opaque_id()),
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RENDERED_SYSTEM, policy
            ),
        ),
    )
    provenance = _provenance((), system=system)

    bundle = _available_bundle(
        provenance,
        messages=[],
        system="actual complete system",
        system_components=("saved", "unrelated fake part"),
    )

    assert bundle.available is False
    assert bundle.omission_reason is TraceOmissionReason.ALIGNMENT_MISMATCH
    assert "unrelated fake part" not in repr(bundle)


def test_nested_saved_system_over_bound_fails_before_trace_write(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    nested = DerivedTraceProvenance(
        TraceTransformKind.SINGLE_PREAMBLE,
        tuple(SavedRevisionTraceProvenance(new_opaque_id()) for _ in range(256)),
    )
    system = DerivedTraceProvenance(
        TraceTransformKind.SINGLE_PREAMBLE,
        (nested, SavedRevisionTraceProvenance(new_opaque_id())),
    )
    message = OmittedTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT,
        TraceOmissionReason.SOURCE_UNAVAILABLE,
    )
    provenance = _provenance((message,), system=system)
    bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": "omitted"}],
        system="unavailable",
    )
    with pytest.raises(ValueError, match="system_revision_span"):
        with db.transaction() as cursor:
            _persist(
                ConsoleTraceService(repository),
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
            )

    cursor = db.get_connection().cursor()
    for table in (
        "console_trace_policies",
        "console_trace_artifacts",
        "console_trace_surface_nodes",
        "console_trace_events",
        "console_trace_request_headers",
    ):
        assert cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_oversize_mixed_system_is_rejected_before_first_trace_insert(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    nested = DerivedTraceProvenance(
        TraceTransformKind.SINGLE_PREAMBLE,
        tuple(
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RENDERED_SYSTEM, policy
            )
            for _ in range(256)
        ),
    )
    system = DerivedTraceProvenance(
        TraceTransformKind.SINGLE_PREAMBLE,
        (
            nested,
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RENDERED_SYSTEM, policy
            ),
        ),
    )
    message = OmittedTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT,
        TraceOmissionReason.SOURCE_UNAVAILABLE,
    )
    provenance = _provenance((message,), system=system)
    bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": "omitted"}],
        system="\n\n".join(f"part-{index}" for index in range(257)),
        system_components=tuple(f"part-{index}" for index in range(257)),
    )
    statements: list[str] = []
    with db.transaction() as cursor:
        db.get_connection().set_trace_callback(statements.append)
        with pytest.raises(ValueError, match="system_composition_span"):
            _persist(
                ConsoleTraceService(repository),
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                provenance=provenance,
                bundle=bundle,
            )
        db.get_connection().set_trace_callback(None)

    assert not any(
        statement.lstrip().upper().startswith("INSERT INTO CONSOLE_TRACE_")
        for statement in statements
    )


def test_header_structural_provenance_includes_derived_non_surface_inputs(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    system = DerivedTraceProvenance(
        TraceTransformKind.SYSTEM_FRAMING,
        (
            ProviderArtifactTraceProvenance(
                TraceProvenanceSource.RENDERED_SYSTEM, policy
            ),
            OmittedTraceProvenance(
                TraceProvenanceSource.PROJECT_INSTRUCTION,
                TraceOmissionReason.WINDOWED,
            ),
        ),
    )
    message = ProviderArtifactTraceProvenance(TraceProvenanceSource.RAG_CONTEXT, policy)
    provenance = ProviderRequestProvenance(
        system_message=system,
        messages=(message,),
        messages_payload=(message,),
        thinking=(
            OmittedTraceProvenance(
                TraceProvenanceSource.THINKING,
                TraceOmissionReason.SOURCE_UNAVAILABLE,
            ),
        ),
        metadata=(
            request_route_provenance(ConsoleRequestRoute.FRESH),
            DerivedTraceProvenance(
                TraceTransformKind.PROVIDER_OVERLAY,
                (message,),
                artifact=ProviderArtifactTraceProvenance(
                    TraceProvenanceSource.PROVIDER_OVERLAY, policy
                ),
            ),
        ),
    )
    bundle = _available_bundle(
        provenance,
        messages=[{"role": "user", "content": "rag"}],
        system="rendered",
    )
    with db.transaction() as cursor:
        header = _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        ).header

    assert header.adapter_defaults["header_transforms"] == {
        "provider_overlay": 1,
        "system_framing": 1,
    }
    assert header.adapter_defaults["header_provenance_omissions"] == {
        "project_instruction:windowed": 1,
        "thinking:source_unavailable": 1,
    }
    assert header.adapter_defaults["header_sources"] == {
        "project_instruction": 1,
        "provider_overlay": 1,
        "rag_context": 1,
        "rendered_system": 1,
        "thinking": 1,
    }


def test_verified_secret_canaries_are_absent_from_every_trace_owned_cell(
    db: CharactersRAGDB,
    repository: ConsoleTraceRepository,
) -> None:
    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    secret = "resolved-provider-secret-canary"
    message = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RAG_CONTEXT,
        policy,
    )
    system = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.RENDERED_SYSTEM,
        policy,
    )
    tool = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.TOOL_DEFINITION,
        policy,
    )
    provenance = _provenance((message,), system=system, tools=(tool,))
    values = {
        "api_endpoint": "openai",
        "messages_payload": [{"role": "user", "content": f"rag {secret}"}],
        "system_message": f"system {secret}",
        "tools": [{"description": f"tool {secret}"}],
        "model": "gpt-test",
        "api_key": secret,
        "temp": 0,
    }
    bundle = verify_provider_request_shadow(
        actual_kwargs=values,
        expected_kwargs=dict(values),
        provenance=provenance,
        project_handler_kwargs=lambda kwargs: kwargs,
        known_credentials=(secret,),
        endpoint_identity=f"https://api.example.invalid/v1?token={secret}",
    )
    assert bundle.available and bundle.redacted

    with db.transaction() as cursor:
        _persist(
            ConsoleTraceService(repository),
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=bundle,
        )
        cells = []
        for table in (
            "console_trace_artifacts",
            "console_trace_request_headers",
            "console_trace_header_components",
            "console_trace_surface_nodes",
            "console_trace_surface_replacements",
            "console_trace_events",
            "console_trace_response_links",
        ):
            cells.extend(
                cell for row in cursor.execute(f"SELECT * FROM {table}") for cell in row
            )
    serialized = b"\n".join(
        bytes(cell) if isinstance(cell, bytes) else str(cell).encode("utf-8")
        for cell in cells
    )
    assert secret.encode("utf-8") not in serialized
    assert b"[credential omitted]" in serialized
