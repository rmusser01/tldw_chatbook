from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass, replace
from inspect import signature
from typing import cast
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from tldw_chatbook.Chat import console_trace_models as trace_models
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    InvalidTraceTransition,
    MAX_SURFACE_REPLACEMENT_SPAN,
    SemanticRevisionRef,
    SurfaceBoundary,
    SurfaceReplacement,
    TraceCallState,
    TraceContentRef,
    TraceOmission,
    TraceOutcome,
    is_terminal_call_state,
    new_opaque_id,
    validate_call_transition,
)


def test_call_state_vocabulary_covers_reservation_dispatch_and_terminal_outcomes() -> None:
    expected_members = {
        "RESERVED": "reserved",
        "NOT_DISPATCHED": "not_dispatched",
        "DISPATCH_STARTED": "dispatch_started",
        "DISPATCH_UNKNOWN": "dispatch_unknown",
        "RESPONSE_STARTED": "response_started",
        "COMPLETE": "complete",
        "STOPPED": "stopped",
        "ERROR": "error",
        "INTERRUPTED": "interrupted",
        "ABANDONED": "abandoned",
    }
    assert {
        name: member.value for name, member in TraceCallState.__members__.items()
    } == expected_members
    assert len(TraceCallState.__members__) == len(tuple(TraceCallState))


def test_opaque_ids_are_random_uuid_values_without_content_input() -> None:
    first = new_opaque_id()
    second = new_opaque_id()

    assert UUID(first).version == 4
    assert UUID(second).version == 4
    assert first != second


@pytest.mark.parametrize(
    "invalid_id",
    (
        "",
        "not-a-uuid",
        "550e8400e29b41d4a716446655440000",
        "550E8400-E29B-41D4-A716-446655440000",
        "550e8400-e29b-41d4-a716-446655440000 ",
        str(uuid5(NAMESPACE_URL, "message body")),
    ),
)
def test_trace_references_reject_noncanonical_or_content_derived_ids(
    invalid_id: str,
) -> None:
    with pytest.raises(ValueError, match="content_id"):
        TraceContentRef(
            content_id=invalid_id,
            content_kind="sanitized_artifact",
        )


def test_every_logical_identity_field_uses_the_same_uuidv4_contract() -> None:
    replace_record = cast(Callable[..., object], replace)
    records = (
        TraceContentRef(new_opaque_id(), "sanitized_artifact"),
        TraceOmission("provider_overlay", "sanitizer_unavailable"),
        SemanticRevisionRef(new_opaque_id()),
        SurfaceBoundary(new_opaque_id(), 0, new_opaque_id()),
        SurfaceReplacement(
            predecessor_head_id=new_opaque_id(),
            start_node_id=new_opaque_id(),
            end_node_id=new_opaque_id(),
            start_sequence=0,
            end_sequence=0,
            replacement_node_id=new_opaque_id(),
        ),
        FrozenTracePolicy(
            policy_id=new_opaque_id(),
            credential_filter_version="credentials-v1",
            pii_redaction_enabled=True,
            pii_ruleset_revision_id=new_opaque_id(),
        ),
    )

    for record in records:
        for model_field in fields(record):
            value = getattr(record, model_field.name)
            if not model_field.name.endswith("_id") or value is None:
                continue
            with pytest.raises(ValueError, match=model_field.name):
                replace_record(record, **{model_field.name: "not-a-uuid"})


def test_content_references_are_immutable_structural_records() -> None:
    reference = TraceContentRef(
        content_id=new_opaque_id(),
        content_kind="sanitized_artifact",
    )

    with pytest.raises(FrozenInstanceError):
        reference.content_id = new_opaque_id()  # type: ignore[misc]


def test_outcomes_are_only_terminal_provider_call_results() -> None:
    expected_members = {
        "COMPLETE": "complete",
        "STOPPED": "stopped",
        "ERROR": "error",
        "INTERRUPTED": "interrupted",
        "ABANDONED": "abandoned",
    }
    assert {
        name: member.value for name, member in TraceOutcome.__members__.items()
    } == expected_members
    assert len(TraceOutcome.__members__) == len(tuple(TraceOutcome))


_ORDINARY_TRANSITIONS = frozenset(
    {
        (TraceCallState.RESERVED, TraceCallState.NOT_DISPATCHED),
        (TraceCallState.RESERVED, TraceCallState.DISPATCH_STARTED),
        (TraceCallState.DISPATCH_STARTED, TraceCallState.DISPATCH_UNKNOWN),
        (TraceCallState.DISPATCH_STARTED, TraceCallState.RESPONSE_STARTED),
        (TraceCallState.DISPATCH_STARTED, TraceCallState.ERROR),
        (TraceCallState.RESPONSE_STARTED, TraceCallState.COMPLETE),
        (TraceCallState.RESPONSE_STARTED, TraceCallState.STOPPED),
        (TraceCallState.RESPONSE_STARTED, TraceCallState.ERROR),
        (TraceCallState.RESPONSE_STARTED, TraceCallState.INTERRUPTED),
    }
)
_TERMINAL_STATES = frozenset(
    {
        TraceCallState.NOT_DISPATCHED,
        TraceCallState.DISPATCH_UNKNOWN,
        TraceCallState.COMPLETE,
        TraceCallState.STOPPED,
        TraceCallState.ERROR,
        TraceCallState.INTERRUPTED,
        TraceCallState.ABANDONED,
    }
)


@pytest.mark.parametrize("provider_operation_inactive", (False, True))
@pytest.mark.parametrize("target", tuple(TraceCallState))
@pytest.mark.parametrize("current", tuple(TraceCallState))
def test_call_transition_matrix_is_exhaustive(
    current: TraceCallState,
    target: TraceCallState,
    provider_operation_inactive: bool,
) -> None:
    permitted = (current, target) in _ORDINARY_TRANSITIONS or (
        provider_operation_inactive
        and current is TraceCallState.DISPATCH_STARTED
        and target is TraceCallState.ABANDONED
    )
    if permitted:
        assert (
            validate_call_transition(
                current,
                target,
                provider_operation_inactive=provider_operation_inactive,
            )
            is target
        )
        return

    with pytest.raises(InvalidTraceTransition):
        validate_call_transition(
            current,
            target,
            provider_operation_inactive=provider_operation_inactive,
        )


@pytest.mark.parametrize("state", tuple(TraceCallState))
def test_terminal_classification_covers_every_call_state(state: TraceCallState) -> None:
    assert is_terminal_call_state(state) is (state in _TERMINAL_STATES)


def test_omission_records_are_content_free() -> None:
    omission = TraceOmission(
        component_kind="provider_overlay",
        reason_code="sanitizer_unavailable",
    )

    payload = asdict(omission)
    assert set(payload) == {"omission_id", "component_kind", "reason_code"}
    assert UUID(payload["omission_id"]).version == 4
    assert not {
        "body",
        "content",
        "digest",
        "hash",
        "matched_value",
        "original_length",
    }.intersection(payload)


@pytest.mark.parametrize(
    "invalid_token",
    (
        "",
        "raw body text",
        "secret=sk-proj-example",
        "line\nbreak",
        "nul\x00byte",
        "résumé",
        "UPPER_CASE",
        "x" * 65,
    ),
)
def test_omission_identifier_tokens_reject_free_text_and_secret_like_values(
    invalid_token: str,
) -> None:
    for field_name in ("component_kind", "reason_code"):
        values = {
            "component_kind": "provider_overlay",
            "reason_code": "sanitizer_unavailable",
        }
        values[field_name] = invalid_token
        with pytest.raises(ValueError, match=field_name):
            TraceOmission(**values)


def test_semantic_revision_references_expose_only_opaque_identity() -> None:
    reference = SemanticRevisionRef(revision_id=new_opaque_id())

    assert set(asdict(reference)) == {"revision_id"}
    with pytest.raises(FrozenInstanceError):
        reference.revision_id = new_opaque_id()  # type: ignore[misc]


def test_surface_boundaries_reject_negative_segment_sequences() -> None:
    with pytest.raises(ValueError, match="sequence"):
        SurfaceBoundary(
            segment_id=new_opaque_id(),
            sequence=-1,
            surface_head_id=new_opaque_id(),
        )

    boundary = SurfaceBoundary(
        segment_id=new_opaque_id(),
        sequence=0,
        surface_head_id=new_opaque_id(),
    )
    with pytest.raises(FrozenInstanceError):
        boundary.sequence = 1  # type: ignore[misc]


@pytest.mark.parametrize("invalid_sequence", (True, False, 1.0, "1", -1))
def test_surface_boundary_sequence_requires_plain_nonnegative_integer(
    invalid_sequence: object,
) -> None:
    with pytest.raises(ValueError, match="sequence"):
        SurfaceBoundary(
            segment_id=new_opaque_id(),
            sequence=cast(int, invalid_sequence),
            surface_head_id=new_opaque_id(),
        )


def test_surface_replacement_is_one_bounded_contiguous_range() -> None:
    replacement = SurfaceReplacement(
        predecessor_head_id=new_opaque_id(),
        start_node_id=new_opaque_id(),
        end_node_id=new_opaque_id(),
        start_sequence=10,
        end_sequence=10 + MAX_SURFACE_REPLACEMENT_SPAN - 1,
        replacement_node_id=new_opaque_id(),
    )

    assert set(asdict(replacement)) == {
        "predecessor_head_id",
        "start_node_id",
        "end_node_id",
        "start_sequence",
        "end_sequence",
        "replacement_node_id",
    }
    with pytest.raises(ValueError, match="ordered"):
        SurfaceReplacement(
            predecessor_head_id=new_opaque_id(),
            start_node_id=new_opaque_id(),
            end_node_id=new_opaque_id(),
            start_sequence=3,
            end_sequence=2,
            replacement_node_id=new_opaque_id(),
        )
    with pytest.raises(ValueError, match="at most"):
        SurfaceReplacement(
            predecessor_head_id=new_opaque_id(),
            start_node_id=new_opaque_id(),
            end_node_id=new_opaque_id(),
            start_sequence=0,
            end_sequence=MAX_SURFACE_REPLACEMENT_SPAN,
            replacement_node_id=new_opaque_id(),
        )


@pytest.mark.parametrize("field_name", ("start_sequence", "end_sequence"))
@pytest.mark.parametrize("invalid_sequence", (True, False, 1.0, "1", -1))
def test_surface_replacement_sequences_require_plain_nonnegative_integers(
    field_name: str,
    invalid_sequence: object,
) -> None:
    sequences = {"start_sequence": 0, "end_sequence": 1}
    sequences[field_name] = cast(int, invalid_sequence)

    with pytest.raises(ValueError, match=field_name):
        SurfaceReplacement(
            predecessor_head_id=new_opaque_id(),
            start_node_id=new_opaque_id(),
            end_node_id=new_opaque_id(),
            replacement_node_id=new_opaque_id(),
            **sequences,
        )


def test_frozen_policy_requires_pii_ruleset_provenance_when_masking_is_enabled() -> None:
    with pytest.raises(ValueError, match="ruleset"):
        FrozenTracePolicy(
            policy_id=new_opaque_id(),
            credential_filter_version="credentials-v1",
            pii_redaction_enabled=True,
            pii_ruleset_revision_id=None,
        )

    policy = FrozenTracePolicy(
        policy_id=new_opaque_id(),
        credential_filter_version="credentials-v1",
        pii_redaction_enabled=True,
        pii_ruleset_revision_id=new_opaque_id(),
    )
    with pytest.raises(FrozenInstanceError):
        policy.pii_redaction_enabled = False  # type: ignore[misc]


def test_logical_records_cannot_grow_content_digest_or_history_fields() -> None:
    record_types = (
        TraceContentRef,
        TraceOmission,
        SemanticRevisionRef,
        SurfaceBoundary,
        SurfaceReplacement,
        FrozenTracePolicy,
    )
    forbidden_fragments = {
        "body",
        "content_digest",
        "content_hash",
        "history",
        "shadowed",
        "source_list",
    }

    for record_type in record_types:
        assert is_dataclass(record_type)
        assert getattr(record_type, "__dataclass_params__").frozen is True
        field_names = {field.name for field in fields(record_type)}
        assert not forbidden_fragments.intersection(field_names)

    assert not {
        name
        for name in vars(trace_models)
        if "canonical" in name.lower()
        and ("hash" in name.lower() or "digest" in name.lower())
    }
    assert not signature(new_opaque_id).parameters
