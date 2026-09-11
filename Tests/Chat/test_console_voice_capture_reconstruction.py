"""Capture durability survives existing software-only compaction projections."""

from dataclasses import replace
import pytest

from Tests.Chat.test_console_context_compaction import (
    _semantic,
    _durable_units,
    _prepare,
    _resolved,
    _range_semantic,
    _range_effective_memory,
)
from tldw_chatbook.Chat.console_context_compaction import (
    plan_compaction,
    CompactionPromptSnapshot,
)
from tldw_chatbook.Chat.console_context_policy import ContextCarryForwardMode
from tldw_chatbook.Chat.console_prepared_request import PreparedConsoleRequest
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestProvenance,
    ConsoleUnitProvenance,
    ProviderArtifactTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenanceSource,
)


@pytest.mark.parametrize("range_memory", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_compaction_reconstruction_keeps_exact_capture_authority(range_memory, capture):
    units = _durable_units(5 if range_memory else 3)
    effective = None
    if range_memory:
        effective = _range_effective_memory(
            tuple(row for unit in units for row in unit.messages),
            start_message_id="u1",
            end_message_id="a1",
        )
        semantic = _range_semantic(units, effective)
    else:
        semantic = _semantic()
    if capture:
        policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)

        def artifact(source):
            return ProviderArtifactTraceProvenance(source, policy)

        provenance = ConsoleRequestProvenance(
            system=tuple(
                artifact(TraceProvenanceSource.RENDERED_SYSTEM) for _ in semantic.system
            ),
            memory=tuple(
                artifact(TraceProvenanceSource.CONTEXT_SUMMARY) for _ in semantic.memory
            ),
            mandatory=(),
            compactable=tuple(
                ConsoleUnitProvenance(
                    tuple(
                        SavedRevisionTraceProvenance(new_opaque_id())
                        for _ in unit.messages
                    )
                )
                for unit in semantic.compactable
            ),
            active_request=(artifact(TraceProvenanceSource.ACTIVE_REQUEST),),
            active_thinking=(),
            active_continuations=(),
            tools=(),
            capture_policy=policy,
        )
        semantic = replace(
            semantic, provenance=provenance, capture_durability="durable"
        )
    seen = []

    def prepare_main(candidate):
        seen.append(candidate)
        assert candidate.capture_durability == ("durable" if capture else None)
        assert (candidate.provenance is not None) is capture
        if capture:
            assert (
                candidate.provenance.capture_policy
                is semantic.provenance.capture_policy
            )
            assert (
                candidate.provenance.compactable
                == semantic.provenance.compactable[-len(candidate.compactable) :]
                if candidate.compactable
                else candidate.provenance.compactable == ()
            )
        return _prepare(candidate)

    result = plan_compaction(
        semantic=semantic,
        prepared_before=_prepare(semantic),
        durable_units=units,
        resolved_policy=_resolved(
            budget=2000, carry=ContextCarryForwardMode.MEMORY_WITH_LATEST_EXCHANGE
        ),
        prompt=CompactionPromptSnapshot("Preserve decisions."),
        effective_memory=effective,
        prepare_main=prepare_main,
        prepare_auxiliary=lambda messages, cap: _prepare(
            PreparedConsoleRequest(active_request=messages), response_tokens=cap
        ),
    )
    assert result.plan is not None
    assert len(seen) >= 2
