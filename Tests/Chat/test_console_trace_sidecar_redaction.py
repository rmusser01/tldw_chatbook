"""Provider replay uses exact saved values while trace storage stays redacted."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_trace_sidecar_replay import (
    replay_harness as replay_harness,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_provenance import (
    DerivedTraceProvenance,
    ProviderArtifactTraceProvenance,
    TraceProvenanceSource,
    TraceTransformKind,
)
from tldw_chatbook.Chat.console_trace_redaction import CredentialSanitizer
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.Chat.provider_continuation import (
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
)


def _save_reasoning(
    harness: SimpleNamespace, text: str
) -> ProviderContinuationCheckpoint:
    checkpoint = replace(
        harness.checkpoint,
        rounds=(replace(harness.checkpoint.rounds[0], reasoning_blocks=(text,)),),
    )
    message_id = harness.prior.persisted_message_id
    saved = harness.database.get_message_by_id(message_id)
    assert harness.database.update_provider_continuation(
        message_id=message_id,
        expected_message_version=saved["version"],
        provider_continuation_json=dump_provider_continuation_json(checkpoint),
        content="prior answer",
    )
    harness.store._message_or_raise(harness.prior.id).provider_continuation = checkpoint
    harness.checkpoint = checkpoint
    return checkpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [True], indirect=True)
async def test_credential_filter_equivalence_cannot_admit_changed_saved_checkpoint(
    replay_harness,
):
    harness = replay_harness
    canonical = _save_reasoning(harness, "Bearer sk-proj-" + "x" * 48)
    forged = replace(
        canonical,
        rounds=(
            replace(
                canonical.rounds[0],
                reasoning_blocks=("Bearer sk-proj-" + "y" * 48,),
            ),
        ),
    )
    sanitizer = CredentialSanitizer()
    canonical_mask = sanitizer.sanitize(
        json.loads(dump_provider_continuation_json(canonical))
    )
    forged_mask = sanitizer.sanitize(
        json.loads(dump_provider_continuation_json(forged))
    )
    assert canonical != forged
    assert canonical_mask.available and forged_mask.available
    assert canonical_mask.redacted and canonical_mask.value == forged_mask.value
    harness.store._message_or_raise(harness.prior.id).provider_continuation = forged

    result = await harness.controller.submit_draft(
        "ordinary request", session_id="session-1"
    )

    assert harness.prepared[-1].continuation_groups[0].checkpoint == forged
    assert result.accepted and not result.provider_started
    assert harness.entries == []
    assert (
        harness.database.get_connection()
        .execute(
            "SELECT COUNT(*) FROM console_trace_calls "
            "WHERE dispatch_started_at IS NOT NULL"
        )
        .fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [True], indirect=True)
async def test_unsaved_continuation_owner_cannot_gain_masked_prefix_authority(
    replay_harness,
):
    harness = replay_harness
    canonical = _save_reasoning(harness, "Bearer sk-proj-" + "x" * 48)
    first = await harness.controller.submit_draft(
        "ordinary request", session_id="session-1"
    )
    assert first.provider_started and len(harness.entries) == 1
    forged = replace(
        canonical,
        rounds=(
            replace(
                canonical.rounds[0],
                reasoning_blocks=("Bearer sk-proj-" + "y" * 48,),
            ),
        ),
    )
    with harness.database.transaction() as cursor:
        policy_id, owner_id = cursor.execute(
            "SELECT policy_id, owner_id FROM console_trace_calls"
        ).fetchone()
        policy = harness.factory.repository.get_policy(cursor, policy_id)
        durable_key = tuple(
            cursor.execute(
                "SELECT component_kind, reference_kind, artifact_id "
                "FROM console_trace_surface_nodes WHERE component_kind = 'continuation'"
            ).fetchone()
        )
        durable_values = harness.factory.service._resolve_reference_values(
            cursor, (durable_key,), owner_id=owner_id
        )
        descriptor = DerivedTraceProvenance(
            TraceTransformKind.CONTINUATION_ATTACHMENT,
            (
                ProviderArtifactTraceProvenance(
                    TraceProvenanceSource.ACTIVE_REQUEST, policy
                ),
            ),
            artifact=ProviderArtifactTraceProvenance(
                TraceProvenanceSource.CONTINUATION, policy
            ),
        )
        assert not harness.factory.service._durable_reference_matches(
            cursor,
            descriptor,
            json.loads(dump_provider_continuation_json(forged)),
            durable_key,
            durable_values,
            owner_id=owner_id,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [True], indirect=True)
@pytest.mark.parametrize(
    ("redaction", "cold"),
    [("credential", False), ("credential", True), ("pii", True)],
)
async def test_redacted_continuation_replays_raw_saved_value_across_successors(
    replay_harness, redaction, cold
):
    harness = replay_harness
    canary = (
        "Bearer sk-proj-" + "x" * 48
        if redaction == "credential"
        else "replay-owner@example.test"
    )
    expected_mask = (
        "[credential omitted]" if redaction == "credential" else "[PII omitted]"
    )
    canonical = _save_reasoning(harness, canary)
    if redaction == "pii":
        before = harness.controller.capture_policy_snapshot("session-1")
        await harness.controller.replace_conversation_trace_privacy(
            "session-1",
            capture_enabled=True,
            pii_redaction_enabled=True,
            expected_policy_revision=before.policy_revision,
        )
        assert harness.controller.capture_policy_snapshot(
            "session-1"
        ).pii_redaction_enabled

    previous_traces = []
    previous_heads = {}
    for text in ("ordinary request", "ordinary successor"):
        if cold:
            harness.factory = ConsoleTraceBoundaryFactory(harness.database)
        result = await harness.controller.submit_draft(text, session_id="session-1")
        assert result.accepted and result.provider_started, (
            result,
            harness.failures,
            harness.build_errors,
        )
        issued = harness.entries[-1]["provider_continuations"]
        assert isinstance(issued, tuple)
        assert len(issued) == 1 and type(issued[0]) is ProviderContinuationCheckpoint
        assert issued[0] == canonical
        assert issued[0].rounds[0].reasoning_blocks == (canary,)

        saved = harness.store.get_message(result.user_message_id)
        reader = ConsoleTraceNativeReader(harness.database)
        trace = reader.read_calls(saved.persisted_message_id)
        assert len(trace) == 1
        captured = json.dumps(trace[0].capture.request)
        assert canary not in captured and expected_mask in captured
        previous_traces.append((saved.persisted_message_id, trace))
        for message_id, original in previous_traces:
            assert reader.read_calls(message_id) == original

        heads = dict(
            harness.database.get_connection().execute(
                "SELECT call_id, surface_node_id FROM console_trace_calls"
            )
        )
        assert all(heads[call_id] == head for call_id, head in previous_heads.items())
        previous_heads = heads
        artifacts = (
            harness.database.get_connection()
            .execute("SELECT sanitized_bytes FROM console_trace_artifacts")
            .fetchall()
        )
        assert artifacts
        assert all(canary.encode() not in bytes(row[0]) for row in artifacts)

    assert len(harness.entries) == 2
    assert (
        harness.store.get_message(harness.prior.id).provider_continuation == canonical
    )
