"""Original Applied values cross the wire without local authority or retirement."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat import console_control_dispatch as controls
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_configuration_preparation import (
    ConsoleTurnCaptureSelection,
)
from tldw_chatbook.Chat.console_context_policy import (
    CompactionFailureBehavior,
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCarryForwardMode,
    ContextCompactionMode,
    ContextCompactionRepresentation,
)
from tldw_chatbook.Chat.console_execution_protocol import (
    ConsoleProtocolError,
    WireIdentity,
    encode_message,
)
from tldw_chatbook.Chat.console_execution_result_values import encode_control_applied
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleEndpointDraft,
    ConsoleModelDraft,
    ConsoleSettingsAction,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
    ConsoleSettingsSurface,
)


IDENTITY = controls.ControlIdentity(
    "profile", "service", 2, "request", "session", "incarnation", 3
)
REPLY = WireIdentity("profile", "service", 9, "request", "session", "incarnation", 3)


def _payload(command, result):
    message = encode_control_applied(REPLY, command, result)
    assert message.kind == "applied"
    assert message.identity is REPLY
    return encode_message(message)["payload"]


def test_scalar_outcomes_preserve_false_and_distinct_reply_sequence():
    assert _payload(controls.UpdateInputs(IDENTITY, "draft", (1, 0)), 4) == {
        "draft_revision": 4
    }
    assert _payload(controls.Drain(IDENTITY), False) == {"applied": False}
    assert _payload(controls.ApplySourceRevision(IDENTITY, object()), 5) == {
        "source_revision": 5
    }
    assert _payload(controls.OpenSession(IDENTITY), ("new", "new-incarnation")) == {
        "session_id": "new",
        "incarnation_id": "new-incarnation",
    }
    selection = ConsoleTurnCaptureSelection(
        ConsoleProviderSelection("openai"), None, None, None, None, True, True
    )
    received = controls.ReceiveTurn(IDENTITY, "turn", 4, selection)
    assert _payload(received, "turn") == {"turn_id": "turn"}
    with pytest.raises(ConsoleProtocolError):
        _payload(received, "other-turn")


@pytest.mark.parametrize("result", [True, -1, 1.5])
def test_draft_revision_rejects_bool_and_invalid_numbers(result):
    with pytest.raises(ConsoleProtocolError):
        _payload(controls.UpdateInputs(IDENTITY, "draft", (1, 0)), result)


@pytest.mark.parametrize("result", [1, None])
def test_boolean_outcome_rejects_integer_and_absent_result(result):
    with pytest.raises(ConsoleProtocolError):
        _payload(controls.Drain(IDENTITY), result)


@pytest.mark.parametrize(
    "changes",
    [
        {"profile_epoch": "other"},
        {"request_id": "other"},
        {"session_id": "other", "incarnation_id": "other-incarnation"},
        {"source_revision": 4},
    ],
)
def test_reply_must_correlate_with_original_command(changes):
    with pytest.raises(ConsoleProtocolError):
        encode_control_applied(
            replace(REPLY, **changes), controls.Drain(IDENTITY), True
        )


def test_exact_command_and_result_classes_and_unsupported_operation_refuse():
    class DerivedDrain(controls.Drain):
        pass

    for command in (
        DerivedDrain(IDENTITY),
        SimpleNamespace(identity=IDENTITY),
        controls.ReadProjection(IDENTITY),
    ):
        with pytest.raises(ConsoleProtocolError):
            _payload(command, True)
    with pytest.raises(ConsoleProtocolError):
        _payload(controls.OpenSession(IDENTITY), ["session", "incarnation"])
    with pytest.raises(ConsoleProtocolError):
        _payload(
            controls.FenceSource(IDENTITY, "configuration"),
            SimpleNamespace(source="configuration", revision=4),
        )


def test_fence_exports_only_values_and_preserves_source_correlation():
    fence = controls.SourceFence("configuration", 4, object(), object())
    assert _payload(controls.FenceSource(IDENTITY, "configuration"), fence) == {
        "source": "configuration",
        "revision": 4,
    }
    with pytest.raises(ConsoleProtocolError):
        _payload(controls.FenceSource(IDENTITY, "workspace"), fence)


def _settings_outcome():
    original = ConsoleSessionSettings("openai", model="before", temperature=0.7)
    normalized = replace(original, model="accepted", temperature=None, top_p=None)
    context = ConsoleContextPolicyOverrides(
        budget_mode=ContextBudgetMode.CUSTOM,
        custom_budget_tokens=4096,
        compaction_mode=ContextCompactionMode.ASK,
        compaction_representation=ContextCompactionRepresentation.TEXT_SUMMARY,
        trigger_ratio=0.8,
        target_ratio=0.55,
        summary_max_tokens=1024,
        failure_behavior=CompactionFailureBehavior.STOP_AND_ASK,
        carry_forward_mode=ContextCarryForwardMode.MEMORY_WITH_LATEST_EXCHANGE,
    )
    field = ConsoleSettingsFieldDraft(
        "temperature", None, None, ConsoleSettingsFieldProvenance.CARRIED, True
    )
    endpoint = ConsoleEndpointDraft("", "openai", False, True)
    draft = ConsoleSettingsDraftState(
        original,
        context,
        (field,),
        (ConsoleModelDraft("openai", "before", original, (field,), endpoint),),
        endpoint,
    )
    submission = ConsoleSettingsSubmission(
        "submission",
        ConsoleSettingsAction.APPLY_TO_CHAT,
        ConsoleSettingsSurface.FULL_SETTINGS,
        ConsoleSettingsOrigin("session", None, 1),
        draft,
        None,
        frozenset({"streaming", "temperature"}),
    )
    accepted = replace(submission, draft=replace(draft, settings=normalized))
    result = ConsoleSettingsLiveCommit(
        "submission",
        "session",
        "persisted",
        1,
        6,
        2,
        normalized,
        context,
        accepted,
        object(),
    )
    return controls.ApplySettings(IDENTITY, submission), result


def test_settings_uses_normalized_submission_and_detaches_without_lease():
    command, result = _settings_outcome()
    message = encode_control_applied(REPLY, command, result)
    payload = encode_message(message)["payload"]
    assert payload["settings"]["model"] == "accepted"
    assert payload["settings"]["temperature"] is None
    accepted = payload["accepted_submission"]
    assert accepted["draft"]["settings"]["model"] == "accepted"
    assert command.submission.draft.settings.model == "before"
    assert accepted["action"] == "apply_to_chat"
    assert accepted["surface"] == "full_settings"
    assert accepted["draft"]["field_drafts"][0]["provenance"] == "carried"
    assert accepted["default_field_mask"] == ["streaming", "temperature"]
    assert payload["context_policy_overrides"]["budget_mode"] == "custom"
    assert payload["context_policy_overrides"]["carry_forward_mode"] == (
        "memory_with_latest_exchange"
    )
    assert "durability_admission" not in payload
    assert "retired" not in payload
    from tldw_chatbook.Chat.console_execution_worker import _submission

    # Exercise the real receiving adapter, including all five nonempty enums.
    assert _submission(accepted) == result.accepted_submission
    accepted["draft"]["model_drafts"].clear()
    assert message.payload["accepted_submission"]["draft"]["model_drafts"]
    with pytest.raises(TypeError):
        message.payload["settings"]["model"] = "changed"


def test_settings_rejects_wrong_commit_identity_and_opaque_nested_values():
    command, result = _settings_outcome()
    for invalid in (
        replace(result, submission_id="other"),
        replace(result, session_id="other"),
        replace(result, settings=object()),
        replace(result, accepted_submission=object()),
        replace(
            result,
            accepted_submission=replace(
                result.accepted_submission, submission_id="other"
            ),
        ),
    ):
        with pytest.raises(ConsoleProtocolError):
            _payload(command, invalid)
    field = result.accepted_submission.draft.field_drafts[0]
    invalid_draft = replace(
        result.accepted_submission.draft,
        field_drafts=(replace(field, effective_value=object()),),
    )
    with pytest.raises(ConsoleProtocolError):
        _payload(
            command,
            replace(
                result,
                accepted_submission=replace(
                    result.accepted_submission, draft=invalid_draft
                ),
            ),
        )


def _commit_plan_notice():
    from tldw_chatbook.Chat.console_chat_store import (
        ConsoleDurableAcceptanceFingerprint,
    )
    from tldw_chatbook.Chat.console_durable_commit_plan import (
        ConsoleDurableRecoveryProjection,
    )
    from tldw_chatbook.Chat.message_metadata import AgentChatStartMetadata

    fingerprint = ConsoleDurableAcceptanceFingerprint(
        "preparation",
        "session",
        "saved",
        "a" * 64,
        "attempt",
        "agent_chat_start",
        None,
        "user",
        "assistant",
        "b" * 64,
    )
    recovery = ConsoleDurableRecoveryProjection(
        fingerprint=fingerprint,
        user_content_sha256="c" * 64,
        parent_message_id=None,
        attachments=((0, "e" * 64, "image/png", ""),),
        frozen_authority_json=(
            '{"policy":{"auto_retrieve":"never","assistant_access":"blocked",'
            '"policy_revision":null,"source":"new_session","error_code":null},'
            '"direct_library_tools":false,'
            '"source_types":["notes","media","conversations"],'
            '"scope_snapshot":{"note_ids":[],"media_ids":[],'
            '"conversations_allowed":false},'
            '"provider_intent":{"provider":"openai","model":null,"endpoint":null},'
            '"attempt_id":"attempt"}'
        ),
        resolved_destination_json=(
            '{"provider":"openai","model":null,"endpoint_identity":"default",'
            '"egress_class":"unknown"}'
        ),
        reconstructability_json=(
            '{"attachments_reconstructable":true,"evidence_reconstructable":true,'
            '"prefill_reconstructable":true,"opaque_reference":null}'
        ),
        project_context_sha256="d" * 64,
        first_persist=True,
        generation_revision=0,
        context_policy_revision=0,
        roleplay_identity_revision=0,
        user_root_fork=False,
        agent_chat_start=AgentChatStartMetadata("attempt", "source-run", "source-chat"),
        context_policy_initial_revision=None,
    )
    return controls.CommitPlanned("request", "turn", 1, fingerprint, recovery)


def test_commit_planned_keeps_original_body_free_facts_detached():
    from tldw_chatbook.Chat.console_execution_protocol import decode_message
    from tldw_chatbook.Chat.console_execution_result_values import (
        encode_commit_planned,
    )

    original = _commit_plan_notice()
    message = encode_commit_planned(REPLY, original)
    encoded = encode_message(message)
    assert message.kind == "commit_planned" and message.operation == "ReceiveTurn"
    assert message.identity is REPLY
    assert encode_message(decode_message(encoded)) == encoded
    payload = encoded["payload"]
    assert payload["received_request_id"] == "request"
    assert payload["fingerprint"] == payload["recovery"]["fingerprint"]
    assert payload["fingerprint"]["digest"] == "b" * 64
    assert payload["recovery"]["attachments"] == [[0, "e" * 64, "image/png", ""]]
    assert payload["recovery"]["agent_chat_start"] == {
        "attempt_id": "attempt",
        "source_run_id": "source-run",
        "source_conversation_id": "source-chat",
    }
    assert payload["recovery"]["checkpoint"] is None
    assert "user_content" not in payload["recovery"]
    assert "retired" not in payload
    payload["recovery"]["attachments"][0][3] = "changed"
    assert message.payload["recovery"]["attachments"][0][3] == ""


def test_commit_planned_refuses_wrong_request_session_fingerprint_and_checkpoint():
    from tldw_chatbook.Chat.console_execution_result_values import (
        encode_commit_planned,
    )

    original = _commit_plan_notice()
    invalid_recovery = replace(
        original.recovery,
        fingerprint=replace(original.fingerprint, digest="f" * 64),
    )
    for identity, projection in (
        (SimpleNamespace(request_id="request"), original),
        (replace(REPLY, request_id="different"), original),
        (replace(REPLY, session_id="different"), original),
        (REPLY, replace(original, recovery=invalid_recovery)),
        (REPLY, replace(original, recovery=object())),
        (
            REPLY,
            replace(original, recovery=replace(original.recovery, checkpoint=object())),
        ),
        (REPLY, SimpleNamespace(received_request_id="request")),
    ):
        with pytest.raises(ConsoleProtocolError):
            encode_commit_planned(identity, projection)


def test_commit_planned_rejects_opaque_values_and_retains_existing_codec_checks():
    from tldw_chatbook.Chat.console_execution_result_values import (
        encode_commit_planned,
    )

    original = _commit_plan_notice()
    for recovery in (
        replace(original.recovery, attachments=[(0, "e" * 64, "image/png", "")]),
        replace(original.recovery, attachments=((0, "e" * 64, "image/png", object()),)),
        replace(original.recovery, agent_chat_start=object()),
        replace(original.recovery, frozen_authority_json='{"arbitrary":"body"}'),
    ):
        with pytest.raises(ConsoleProtocolError):
            encode_commit_planned(REPLY, replace(original, recovery=recovery))
    with pytest.raises(ConsoleProtocolError):
        encode_commit_planned(REPLY, replace(original, ordinal=True))
