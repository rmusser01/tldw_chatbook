"""Project-context renewal keeps the completed-turn proof closed and bounded."""

from contextlib import closing
from dataclasses import replace

import pytest

from Tests.Chat.test_console_trace_service import (
    _completed_run_compound_preparation,
    _compound_storage_snapshot,
    _policy,
    _provenance,
)
from tldw_chatbook.Chat.console_trace_final_values import CompletedToolTurnWitness
from tldw_chatbook.Chat.console_trace_provenance import (
    ProviderArtifactTraceProvenance,
    TraceProvenanceSource,
)
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    with closing(
        CharactersRAGDB(tmp_path / "project-turn.sqlite", "project-turn")
    ) as database:
        yield database


@pytest.fixture
def repository():
    return ConsoleTraceRepository()


@pytest.mark.parametrize("count", [-1, 257, True, 1.0, "1"])
def test_project_turn_witness_rejects_invalid_context_counts(count):
    from tldw_chatbook.Chat.console_trace_models import new_opaque_id

    with pytest.raises(ValueError, match="completed_project_context_count"):
        CompletedToolTurnWitness(
            *(new_opaque_id() for _ in range(4)), project_context_count=count
        )


@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "disabled",
        "missing_tag",
        "wrong_role",
        "wrong_policy",
        "wrong_source",
        "wrong_count",
        "wrong_anchor",
    ],
)
def test_project_turn_preparation_validates_current_context_and_prior_anchor(
    db, repository, scenario
):
    """The new context is admitted only after the exact saved response/user pair."""
    service, admission, _, values = _completed_run_compound_preparation(db, repository)
    witness = replace(
        admission.completed_tool_turn,
        project_context_count=0 if scenario == "disabled" else 1,
    )
    with db.transaction() as cursor:
        origin = repository.get_call(cursor, witness.origin_call_id)
        policy = repository.get_policy(cursor, origin.policy_id)
    if scenario == "wrong_policy":
        policy = replace(_policy(), credential_filter_version="credentials-v2")
    context = {
        "role": "assistant" if scenario == "wrong_role" else "user",
        "content": "current project guidance",
        "_chatbook_ephemeral_origin": "project_instructions",
    }
    if scenario == "missing_tag":
        context.pop("_chatbook_ephemeral_origin")
    source = (
        TraceProvenanceSource.ACTIVE_REQUEST
        if scenario == "wrong_source"
        else TraceProvenanceSource.PROJECT_INSTRUCTION
    )
    descriptors = admission.descriptors
    if scenario != "disabled":
        descriptors += (ProviderArtifactTraceProvenance(source, policy),)
        values += (context,)
    if scenario == "wrong_count":
        witness = replace(witness, project_context_count=2)
    if scenario in {"wrong_source", "wrong_count"}:
        with pytest.raises(ValueError, match="surface_delta_shape"):
            replace(admission, descriptors=descriptors, completed_tool_turn=witness)
        return
    if scenario == "wrong_anchor":
        with db.transaction() as cursor:
            node = repository.read_lineage_surface_nodes(
                cursor,
                segment_id=admission.segment_id,
                start_sequence=2,
                end_sequence=2,
            )[0]
        admission = replace(
            admission,
            replacement_range=replace(
                admission.replacement_range,
                start_sequence=2,
                start_node_id=node.node_id,
                current_ordinal=2,
                component_ordinal=2,
            ),
        )
    admission = replace(admission, descriptors=descriptors, completed_tool_turn=witness)
    with db.transaction() as cursor:
        before = _compound_storage_snapshot(cursor)
        if scenario in {"valid", "disabled"}:
            boundary = service.prepare_surface_provenance(
                cursor,
                admission.projection_checkpoint,
                provenance=_provenance(descriptors),
                admission=admission,
                values=values,
            )
            actual = boundary._provider_request_surface_values()["messages_payload"]
            assert tuple(dict(value) for value in actual[-len(values) :]) == values
        else:
            expected = (
                "completed_tool_turn_range"
                if scenario == "wrong_anchor"
                else "completed_project_context_value"
            )
            with pytest.raises(ValueError, match=expected):
                service.prepare_surface_provenance(
                    cursor,
                    admission.projection_checkpoint,
                    provenance=_provenance(descriptors),
                    admission=admission,
                    values=values,
                )
        assert _compound_storage_snapshot(cursor) == before


@pytest.mark.parametrize(
    "fixture_args, expected",
    [
        ({"duplicate_append": True}, "completed_tool_turn_lineage"),
        ({"tool_count": 257}, "completed_tool_turn_range"),
        ({"artifact_source": "active_request"}, "completed_tool_turn_range"),
    ],
)
def test_project_transition_keeps_old_suffix_lineage_and_span_guards(
    db, repository, fixture_args, expected
):
    """Opting into context renewal cannot admit malformed prior-run evidence."""
    service, admission, provenance, values = _completed_run_compound_preparation(
        db, repository, **fixture_args
    )
    admission = replace(
        admission,
        completed_tool_turn=replace(
            admission.completed_tool_turn, project_context_count=0
        ),
    )
    with db.transaction() as cursor:
        before = _compound_storage_snapshot(cursor)
        with pytest.raises(ValueError, match=expected):
            service.prepare_surface_provenance(
                cursor,
                admission.projection_checkpoint,
                provenance=provenance,
                admission=admission,
                values=values,
            )
        assert _compound_storage_snapshot(cursor) == before


@pytest.mark.parametrize(
    "scenario",
    ["valid", "stream_complete", "wrong_surface", "wrong_turn", "wrong_policy"],
)
def test_project_fallback_requires_the_exact_preceding_failed_stream(
    db, repository, scenario
):
    """A fallback response cannot replace artifacts owned by a different stream."""
    from tldw_chatbook.Chat.console_trace_models import (
        SemanticRevisionRef,
        TraceCallState,
        new_opaque_id,
    )

    service, admission, provenance, values = _completed_run_compound_preparation(
        db,
        repository,
        terminal_outcome=(
            TraceCallState.COMPLETE
            if scenario == "stream_complete"
            else TraceCallState.ERROR
        ),
    )
    witness = admission.completed_tool_turn
    with db.transaction() as cursor:
        failed = repository.get_call(cursor, witness.terminal_call_id)
        origin = repository.get_call(cursor, witness.origin_call_id)
        policy_id = failed.policy_id
        if scenario == "wrong_policy":
            policy = _policy()
            repository.ensure_policy(cursor, policy)
            policy_id = policy.policy_id
        fallback = repository.reserve_call(
            cursor,
            owner_id=failed.owner_id,
            segment_id=failed.segment_id,
            turn_id=(new_opaque_id() if scenario == "wrong_turn" else failed.turn_id),
            run_id=new_opaque_id(),
            call_sequence=0,
            idempotency_key=new_opaque_id(),
            policy_id=policy_id,
        )
        event_tail = repository.get_event_tail(cursor, failed.segment_id)
        repository.append_event(
            cursor,
            segment_id=failed.segment_id,
            sequence=event_tail.sequence + 1,
            event_type="call_boundary",
            call_id=fallback.call_id,
        )
        header = repository.create_or_reuse_request_header(
            cursor,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="llama_fallback",
            endpoint_identity="https://api.example.invalid/v1",
            generation_parameters={},
            adapter_defaults={},
            response_format={},
            reasoning_controls={},
            components=(),
        )
        repository.bind_call(
            cursor,
            call_id=fallback.call_id,
            surface_node_id=(
                origin.surface_node_id
                if scenario == "wrong_surface"
                else failed.surface_node_id
            ),
            request_header_id=header.header_id,
            provider_name="openai",
            model_name="gpt-test",
            route_identity="llama_fallback",
        )
        for target in (
            TraceCallState.DISPATCH_STARTED,
            TraceCallState.RESPONSE_STARTED,
            TraceCallState.COMPLETE,
        ):
            repository.advance_call_state(
                cursor,
                call_id=fallback.call_id,
                target=target,
                occurred_at="2026-09-07T00:00:00Z",
            )
        repository.store_response_link(
            cursor,
            call_id=fallback.call_id,
            response=SemanticRevisionRef(witness.assistant_revision_id),
        )
        admission = replace(
            admission,
            completed_tool_turn=replace(
                witness,
                origin_call_id=fallback.call_id,
                terminal_call_id=fallback.call_id,
                project_context_count=0,
            ),
        )
        before = _compound_storage_snapshot(cursor)
        if scenario == "valid":
            service.prepare_surface_provenance(
                cursor,
                admission.projection_checkpoint,
                provenance=provenance,
                admission=admission,
                values=values,
            )
        else:
            with pytest.raises(ValueError, match="completed_project_fallback_lineage"):
                service.prepare_surface_provenance(
                    cursor,
                    admission.projection_checkpoint,
                    provenance=provenance,
                    admission=admission,
                    values=values,
                )
        assert _compound_storage_snapshot(cursor) == before
