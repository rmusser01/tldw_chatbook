"""Surface refusal fences expose structure without retaining request content."""

import pytest

from Tests.Chat.test_console_send_diagnostics import assert_export
from Tests.Chat.test_console_send_diagnostics import (
    sinks as sinks,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Chat.test_console_trace_service import (
    _available_bundle,
    _owned_segment,
    _persist,
    _policy,
    _provenance,
)
from Tests.Chat.test_console_trace_service import (
    db as db,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Chat.test_console_trace_service import (
    repository as repository,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.Chat.console_send_diagnostics import send_diagnostic_scope
from tldw_chatbook.Chat.console_trace_models import new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ProviderArtifactTraceProvenance,
    TraceProvenanceSource,
)
from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService


@pytest.mark.asyncio
@pytest.mark.parametrize("span_limit", [False, True])
async def test_surface_refusal_count_and_span_reach_support_exports(
    db, repository, sinks, monkeypatch, span_limit
):
    owner_id, segment_id = _owned_segment(db, repository)
    descriptor = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST, _policy()
    )
    provenance = _provenance((descriptor, descriptor))
    initial = [
        {"role": "user", "content": "PRIVATE-DRAFT-31977"},
        {"role": "assistant", "content": "private answer"},
    ]
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(provenance, messages=initial),
        )
    if span_limit:
        monkeypatch.setattr(
            "tldw_chatbook.Chat.console_trace_service.MAX_SURFACE_REPLACEMENT_SPAN", 1
        )
    incoming = ({"role": "user", "content": "different private content"},) * (
        1 if span_limit else 2
    )
    replacement = _provenance((descriptor,) * len(incoming))
    async with send_diagnostic_scope("controller_submit"):
        with db.transaction() as cursor:
            with pytest.raises(ValueError, match="^unsupported_surface_change$"):
                service.prepare_current_surface_delta(
                    cursor,
                    owner_id=owner_id,
                    segment_id=segment_id,
                    route_identity=ConsoleRequestRoute.FRESH.value,
                    preparation_identity=new_opaque_id(),
                    provenance=replacement,
                    values=incoming,
                )
    text = assert_export(
        sinks,
        "surface_refusal_kind=count_or_span",
        "surface_prefix=0",
        "surface_suffix=0",
        f"surface_incoming_changed={len(incoming)}",
        "surface_active_changed=2",
        "surface_replacement_span=2",
        "surface_domains=messages_payload",
    )
    for private in (
        "private answer",
        "different private content",
        owner_id,
        segment_id,
    ):
        assert private not in text


@pytest.mark.asyncio
async def test_surface_refusal_sequence_gap_reaches_support_exports(
    db, repository, sinks
):
    from dataclasses import replace

    from Tests.Chat.test_console_trace_service import _continuation_value

    owner_id, segment_id = _owned_segment(db, repository)
    policy = _policy()
    message = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.ACTIVE_REQUEST, policy
    )
    continuation = ProviderArtifactTraceProvenance(
        TraceProvenanceSource.CONTINUATION, policy
    )
    provenance = replace(_provenance((message,)), continuations=(continuation,))
    value = _continuation_value("PRIVATE-DRAFT-31977")
    service = ConsoleTraceService(repository)
    with db.transaction() as cursor:
        first = _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=provenance,
            bundle=_available_bundle(
                provenance,
                messages=[{"role": "user", "content": "first private"}],
                continuations=[value],
            ),
        )
        appended = _provenance((message,))
        _persist(
            service,
            cursor,
            owner_id=owner_id,
            segment_id=segment_id,
            provenance=appended,
            bundle=_available_bundle(
                appended, messages=[{"role": "user", "content": "second private"}]
            ),
            previous_surface_head_id=first.surface_head_id,
        )
    async with send_diagnostic_scope("controller_submit"):
        with db.transaction() as cursor:
            with pytest.raises(ValueError, match="^unsupported_surface_change$"):
                service.prepare_current_surface_delta(
                    cursor,
                    owner_id=owner_id,
                    segment_id=segment_id,
                    route_identity=ConsoleRequestRoute.FRESH.value,
                    preparation_identity=new_opaque_id(),
                    provenance=provenance,
                    values=({"role": "user", "content": "new private"}, value),
                )
    text = assert_export(
        sinks,
        "surface_refusal_kind=sequence_gap",
        "surface_prefix=0",
        "surface_suffix=1",
        "surface_incoming_changed=1",
        "surface_active_changed=2",
        "surface_replacement_span=3",
        "surface_start_sequence=0",
        "surface_end_sequence=2",
    )
    assert "new private" not in text
