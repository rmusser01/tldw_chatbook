"""Detach original control results without exporting authority or retirement."""

from __future__ import annotations

from typing import NoReturn

from . import console_control_dispatch as controls
from .console_context_policy import (
    CompactionFailureBehavior,
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCarryForwardMode,
    ContextCompactionMode,
    ContextCompactionRepresentation,
)
from .console_execution_projection_values import _settings as _session_settings
from .console_execution_protocol import (
    ConsoleProtocolError,
    ConsoleWireMessage,
    WireIdentity,
)
from .console_settings_apply import (
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


def _fail(reason: str) -> NoReturn:
    raise ConsoleProtocolError(f"Console control result: {reason}") from None


def _exact(value: object, expected: type) -> None:
    if type(value) is not expected:
        _fail(f"expected {expected.__name__}")


def _enum(value: object, expected: type) -> str:
    _exact(value, expected)
    return value.value


def _optional_enum(value: object, expected: type) -> str | None:
    return None if value is None else _enum(value, expected)


def _context_policy(value: object) -> dict:
    _exact(value, ConsoleContextPolicyOverrides)
    return {
        "budget_mode": _optional_enum(value.budget_mode, ContextBudgetMode),
        "custom_budget_tokens": value.custom_budget_tokens,
        "compaction_mode": _optional_enum(value.compaction_mode, ContextCompactionMode),
        "compaction_representation": _optional_enum(
            value.compaction_representation, ContextCompactionRepresentation
        ),
        "trigger_ratio": value.trigger_ratio,
        "target_ratio": value.target_ratio,
        "summary_max_tokens": value.summary_max_tokens,
        "failure_behavior": _optional_enum(
            value.failure_behavior, CompactionFailureBehavior
        ),
        "carry_forward_mode": _optional_enum(
            value.carry_forward_mode, ContextCarryForwardMode
        ),
    }


def _field_drafts(values: object) -> list:
    _exact(values, tuple)
    result = []
    for value in values:
        _exact(value, ConsoleSettingsFieldDraft)
        result.append(
            {
                "name": value.name,
                "effective_value": value.effective_value,
                "profile_override": value.profile_override,
                "provenance": _enum(value.provenance, ConsoleSettingsFieldProvenance),
                "dirty": value.dirty,
            }
        )
    return result


def _endpoint(value: object) -> dict | None:
    if value is None:
        return None
    _exact(value, ConsoleEndpointDraft)
    return {
        "value": value.value,
        "bound_provider_config_key": value.bound_provider_config_key,
        "dirty": value.dirty,
        "checked": value.checked,
    }


def _draft(value: object) -> dict:
    _exact(value, ConsoleSettingsDraftState)
    _exact(value.model_drafts, tuple)
    models = []
    for model in value.model_drafts:
        _exact(model, ConsoleModelDraft)
        models.append(
            {
                "provider": model.provider,
                "model": model.model,
                "settings": _session_settings(model.settings, False),
                "field_drafts": _field_drafts(model.field_drafts),
                "endpoint_draft": _endpoint(model.endpoint_draft),
            }
        )
    return {
        "settings": _session_settings(value.settings, False),
        "context_policy_overrides": _context_policy(value.context_policy_overrides),
        "field_drafts": _field_drafts(value.field_drafts),
        "model_drafts": models,
        "endpoint_draft": _endpoint(value.endpoint_draft),
    }


def _submission(value: object) -> dict:
    _exact(value, ConsoleSettingsSubmission)
    _exact(value.origin, ConsoleSettingsOrigin)
    _exact(value.default_field_mask, frozenset)
    for name in value.default_field_mask:
        _exact(name, str)
    return {
        "submission_id": value.submission_id,
        "action": _enum(value.action, ConsoleSettingsAction),
        "surface": _enum(value.surface, ConsoleSettingsSurface),
        "origin": {
            "session_id": value.origin.session_id,
            "persisted_conversation_id": value.origin.persisted_conversation_id,
            "conversation_binding_revision": value.origin.conversation_binding_revision,
        },
        "draft": _draft(value.draft),
        "user_display_name_override": value.user_display_name_override,
        "default_field_mask": sorted(value.default_field_mask),
    }


def _live_commit(command: controls.ApplySettings, value: object) -> dict:
    _exact(value, ConsoleSettingsLiveCommit)
    _exact(command.submission, ConsoleSettingsSubmission)
    if (
        value.submission_id != command.submission.submission_id
        or value.session_id != command.identity.session_id
    ):
        _fail("settings result does not match original command")
    accepted = value.accepted_submission
    if accepted is not None:
        _exact(accepted, ConsoleSettingsSubmission)
        if accepted.submission_id != value.submission_id:
            _fail("accepted submission does not match live commit")
    return {
        "submission_id": value.submission_id,
        "session_id": value.session_id,
        "persisted_conversation_id": value.persisted_conversation_id,
        "conversation_binding_revision": value.conversation_binding_revision,
        "generation_revision": value.generation_revision,
        "context_policy_revision": value.context_policy_revision,
        "settings": _session_settings(value.settings, False),
        "context_policy_overrides": _context_policy(value.context_policy_overrides),
        "accepted_submission": None if accepted is None else _submission(accepted),
    }


_OPERATIONS = {
    controls.UpdateInputs: "UpdateInputs",
    controls.ReceiveTurn: "ReceiveTurn",
    controls.OpenSession: "OpenSession",
    controls.CloseSession: "CloseSession",
    controls.ResolveDecision: "ResolveDecision",
    controls.CancelTurn: "CancelTurn",
    controls.CommitPlanAck: "CommitPlanAck",
    controls.CloseIntake: "CloseIntake",
    controls.Drain: "Drain",
    controls.Shutdown: "Shutdown",
    controls.ApplySourceRevision: "ApplySourceRevision",
    controls.FenceSource: "FenceSource",
    controls.ApplySettings: "ApplySettings",
}


def encode_control_applied(
    identity: WireIdentity, command: object, result: object
) -> ConsoleWireMessage:
    """Encode the fixed Applied value of an original dispatched command.

    Args:
        identity: Reply identity, with its own outbound sequence.
        command: Exact original local command retained by the backend.
        result: That command's original Applied result.

    Returns:
        A validated, detached Applied message without local capabilities.

    Raises:
        ConsoleProtocolError: The command, correlation or result is unsupported.
    """
    _exact(identity, WireIdentity)
    operation = _OPERATIONS.get(type(command))
    if operation is None:
        _fail("unsupported command")
    _exact(command.identity, controls.ControlIdentity)
    if any(
        getattr(identity, name) != getattr(command.identity, name)
        for name in (
            "profile_epoch",
            "service_epoch",
            "request_id",
            "session_id",
            "incarnation_id",
            "source_revision",
        )
    ):
        _fail("reply identity does not match original command")

    if type(command) is controls.UpdateInputs:
        _exact(result, int)
        payload = {"draft_revision": result}
    elif type(command) is controls.ReceiveTurn:
        _exact(result, str)
        if result != command.turn_id:
            _fail("turn result does not match original command")
        payload = {"turn_id": result}
    elif type(command) is controls.OpenSession:
        _exact(result, tuple)
        if len(result) != 2:
            _fail("expected original session/incarnation pair")
        payload = {"session_id": result[0], "incarnation_id": result[1]}
    elif type(command) is controls.ApplySourceRevision:
        _exact(result, int)
        payload = {"source_revision": result}
    elif type(command) is controls.FenceSource:
        _exact(result, controls.SourceFence)
        if result.source != command.source:
            _fail("fence result does not match original command")
        payload = {"source": result.source, "revision": result.revision}
    elif type(command) is controls.ApplySettings:
        payload = _live_commit(command, result)
    else:
        _exact(result, bool)
        payload = {"applied": result}
    return ConsoleWireMessage(identity, "applied", operation, payload)


def encode_commit_planned(
    identity: WireIdentity, projection: controls.CommitPlanned
) -> ConsoleWireMessage:
    """Detach an original plan notice without reconstructing its authority.

    Args:
        identity: Reply identity already correlated by the backend endpoint.
        projection: Exact original body-free notice from the retained pending plan.

    Returns:
        A closed CommitPlanned message, with no checkpoint or native owner.

    Raises:
        ConsoleProtocolError: Types, request/session pairing or values changed.
    """
    from .console_chat_store import ConsoleDurableAcceptanceFingerprint
    from .console_durable_commit_plan import ConsoleDurableRecoveryProjection
    from .message_metadata import AgentChatStartMetadata

    _exact(identity, WireIdentity)
    _exact(projection, controls.CommitPlanned)
    if projection.received_request_id != identity.request_id:
        _fail("commit plan request does not match reply")
    fingerprint, recovery = projection.fingerprint, projection.recovery
    _exact(fingerprint, ConsoleDurableAcceptanceFingerprint)
    _exact(recovery, ConsoleDurableRecoveryProjection)
    _exact(recovery.fingerprint, ConsoleDurableAcceptanceFingerprint)
    if (
        recovery.fingerprint != fingerprint
        or fingerprint.session_id != identity.session_id
        or recovery.checkpoint is not None
    ):
        _fail("commit plan recovery or session changed")
    fact = {
        name: getattr(fingerprint, name)
        for name in (
            "preparation_id",
            "session_id",
            "conversation_id",
            "title_hash",
            "attempt_id",
            "origin",
            "queue_entry_id",
            "user_message_id",
            "assistant_message_id",
            "digest",
        )
    }
    body = {
        name: getattr(recovery, name)
        for name in (
            "user_content_sha256",
            "parent_message_id",
            "frozen_authority_json",
            "resolved_destination_json",
            "reconstructability_json",
            "project_context_sha256",
            "first_persist",
            "generation_revision",
            "context_policy_revision",
            "roleplay_identity_revision",
            "user_root_fork",
            "context_policy_initial_revision",
            "initial_user_version",
            "initial_assistant_version",
            "initial_checkpoint_revision",
            "checkpoint",
        )
    }
    _exact(recovery.attachments, tuple)
    for row in recovery.attachments:
        _exact(row, tuple)
    body["attachments"] = [list(row) for row in recovery.attachments]
    body["fingerprint"] = fact
    provenance = recovery.agent_chat_start
    if provenance is not None:
        _exact(provenance, AgentChatStartMetadata)
    body["agent_chat_start"] = (
        None
        if provenance is None
        else {
            name: getattr(provenance, name)
            for name in ("attempt_id", "source_run_id", "source_conversation_id")
        }
    )
    return ConsoleWireMessage(
        identity,
        "commit_planned",
        "ReceiveTurn",
        {
            "received_request_id": projection.received_request_id,
            "turn_id": projection.turn_id,
            "ordinal": projection.ordinal,
            "fingerprint": fact,
            "recovery": body,
        },
    )
