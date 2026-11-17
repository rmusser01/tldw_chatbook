"""Validate and detach fixed Console protocol values without loading owners."""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import NoReturn

__all__ = [
    "PROTOCOL_VERSION",
    "ConsoleProtocolError",
    "ConsoleWireMessage",
    "WireIdentity",
    "decode_message",
    "encode_message",
]

PROTOCOL_VERSION = 1
# Same container nesting bound as MCP.permission_process.read_message.
_MAX_DEPTH = 128
_MAX_STAGED_ATTACHMENTS = 5  # console_chat_store.MAX_PENDING_ATTACHMENTS

type JSONValue = (
    str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
)
type _Check = Callable[[object, str], None]


class ConsoleProtocolError(ValueError):
    """A message is outside the fixed detached Console wire contract."""


def _reject(path: str, reason: str) -> NoReturn:
    # Diagnostic paths come from the schema, not from untrusted keys/values.
    raise ConsoleProtocolError(f"{path}: {reason}")


def _check_string(value: object, path: str) -> None:
    if type(value) is not str:
        _reject(path, "expected text")
    if not value.isascii():
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ConsoleProtocolError(f"{path}: invalid Unicode text") from None


def _detach(value: object, path: str, depth: int, active: set[int], *, thaw=False):
    """Copy between exact JSON containers and the codec's frozen containers."""
    kind = type(value)
    if kind is str:
        _check_string(value, path)
        return value
    if value is None or kind in (bool, int):
        return value
    if kind is float:
        if not math.isfinite(value):
            _reject(path, "nonfinite number")
        return value
    mapping_type, array_type = (MappingProxyType, tuple) if thaw else (dict, list)
    if kind not in (mapping_type, array_type):
        _reject(path, "expected JSON data")
    if depth > _MAX_DEPTH:
        _reject(path, "excessive nesting")
    identity = id(value)
    if identity in active:
        _reject(path, "cyclic value")
    active.add(identity)
    try:
        if kind is array_type:
            items = (
                _detach(item, path, depth + 1, active, thaw=thaw) for item in value
            )
            return list(items) if thaw else tuple(items)
        result = {}
        for key, item in value.items():
            _check_string(key, path)
            result[key] = _detach(item, path, depth + 1, active, thaw=thaw)
        return result if thaw else MappingProxyType(result)
    finally:
        active.remove(identity)


def _text(*, minimum: int = 0, maximum: int | None = None) -> _Check:
    def check(value, path):
        _check_string(value, path)
        if len(value) < minimum or maximum is not None and len(value) > maximum:
            _reject(path, "text length out of bounds")

    return check


def _integer(minimum: int | None = None) -> _Check:
    def check(value, path):
        if type(value) is not int or minimum is not None and value < minimum:
            _reject(path, "invalid integer")

    return check


def _number(value: object, path: str) -> None:
    if type(value) not in (int, float):
        _reject(path, "expected number")
    if type(value) is float and not math.isfinite(value):
        _reject(path, "nonfinite number")


def _boolean(value: object, path: str) -> None:
    if type(value) is not bool:
        _reject(path, "expected boolean")


def _null(value: object, path: str) -> None:
    if value is not None:
        _reject(path, "expected null")


def _scalar(value: object, path: str) -> None:
    if value is not None and type(value) not in (str, bool, int, float):
        _reject(path, "expected scalar value")


def _optional(check: _Check) -> _Check:
    def validate(value, path):
        if value is not None:
            check(value, path)

    return validate


def _enum(*choices: str) -> _Check:
    def check(value, path):
        if type(value) is not str or value not in choices:
            _reject(path, "unknown enum value")

    return check


def _pattern(pattern: str) -> _Check:
    expression = re.compile(pattern, re.ASCII)

    def check(value, path):
        if type(value) is not str or expression.fullmatch(value) is None:
            _reject(path, "invalid identifier")

    return check


def _object(value: object, path: str) -> None:
    if type(value) is not MappingProxyType:
        _reject(path, "expected object")


def _map(check: _Check, *, keys: _Check | None = None) -> _Check:
    def validate(value, path):
        _object(value, path)
        for key, item in value.items():
            if keys is not None:
                keys(key, path)
            check(item, path)

    return validate


def _record(**fields: _Check) -> _Check:
    def check(value, path):
        _object(value, path)
        if value.keys() != fields.keys():
            _reject(path, "unexpected or missing fields")
        for name, validate in fields.items():
            validate(value[name], f"{path}.{name}")

    return check


def _array(
    check: _Check, *, maximum: int | None = None, unique: bool = False
) -> _Check:
    def validate(value, path):
        if type(value) is not tuple:
            _reject(path, "expected array")
        if maximum is not None and len(value) > maximum:
            _reject(path, "too many items")
        for index, item in enumerate(value):
            check(item, f"{path}[{index}]")
        if unique and len(set(value)) != len(value):
            _reject(path, "repeated item")

    return validate


def _tuple(*items: _Check) -> _Check:
    def validate(value, path):
        if type(value) is not tuple or len(value) != len(items):
            _reject(path, "unexpected tuple length")
        for index, check in enumerate(items):
            check(value[index], f"{path}[{index}]")

    return validate


def _tagged(variants: dict[str, dict[str, _Check]]) -> _Check:
    checks = {
        tag: _record(kind=_enum(tag), **fields) for tag, fields in variants.items()
    }

    def validate(value, path):
        _object(value, path)
        tag = value.get("kind")
        if type(tag) is not str or tag not in checks:
            _reject(path, "unknown variant")
        checks[tag](value, path)

    return validate


def _base64(value: object, path: str) -> None:
    _check_string(value, path)
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError:
        raise ConsoleProtocolError(f"{path}: malformed base64") from None
    if base64.b64encode(decoded).decode("ascii") != value:
        _reject(path, "noncanonical base64")


_TEXT = _text()
_NONEMPTY = _text(minimum=1)
_REFERENCE = _text(minimum=1, maximum=256)
_INT = _integer()
_REVISION = _integer(0)
_SEQUENCE = _integer(1)
_TURN_ID = _pattern(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}")
_ERROR_ID = _pattern(r"[a-z][a-z0-9_]{0,63}")
_DIGEST = _pattern(r"[0-9a-f]{64}")
_IDS = _array(_NONEMPTY, unique=True)
_AUTHORED_TOKEN = _tuple(_REVISION, _REVISION)
_MAYBE_TEXT = _optional(_TEXT)
_MAYBE_REVISION = _optional(_REVISION)
_EMPTY = _record()


@dataclass(frozen=True, slots=True)
class WireIdentity:
    """Inert identity data. Sequence and epoch custody belong to the endpoint."""

    profile_epoch: str
    service_epoch: str
    sequence: int
    request_id: str
    session_id: str = ""
    incarnation_id: str = ""
    source_revision: int = 0

    def __post_init__(self) -> None:
        _check_identity(self)


_IDENTITY_FIELDS = {
    "profile_epoch": _REFERENCE,
    "service_epoch": _REFERENCE,
    "sequence": _SEQUENCE,
    "request_id": _REFERENCE,
    "session_id": _text(maximum=256),
    "incarnation_id": _text(maximum=256),
    "source_revision": _REVISION,
}


def _check_identity(identity: WireIdentity) -> None:
    if type(identity) is not WireIdentity:
        _reject("identity", "expected WireIdentity")
    for name, check in _IDENTITY_FIELDS.items():
        check(getattr(identity, name), f"identity.{name}")
    if bool(identity.session_id) != bool(identity.incarnation_id):
        _reject("identity", "session and incarnation must be paired")


# console_session_settings.ConsoleSessionSettings and
# console_chat_models.ConsoleProviderSelection share these input field types.
_OPTIONAL_GENERATION = {
    "min_p": _optional(_number),
    "top_k": _optional(_INT),
    "max_tokens": _optional(_INT),
    "seed": _optional(_INT),
    "presence_penalty": _optional(_number),
    "frequency_penalty": _optional(_number),
    "reasoning_effort": _MAYBE_TEXT,
    "reasoning_summary": _MAYBE_TEXT,
    "verbosity": _MAYBE_TEXT,
    "thinking_effort": _MAYBE_TEXT,
    "thinking_budget_tokens": _optional(_INT),
    "streaming": _boolean,
}
_SESSION_SETTINGS = _record(
    provider=_NONEMPTY,
    model=_MAYBE_TEXT,
    base_url=_MAYBE_TEXT,
    # Original provider rebasing can commit blank unsupported samplers.
    temperature=_optional(_number),
    top_p=_optional(_number),
    **_OPTIONAL_GENERATION,
    character_label=_TEXT,
    system_prompt=_MAYBE_TEXT,
    source=_enum("derived", "user"),
    pinned_prefill=_MAYBE_TEXT,
    persona_memory_mode=_optional(_enum("read_only", "read_write")),
)
_CONTEXT_POLICY = _record(
    budget_mode=_optional(_enum("automatic", "custom")),
    custom_budget_tokens=_optional(_INT),
    compaction_mode=_optional(_enum("ask", "automatic", "off")),
    compaction_representation=_optional(
        _enum("text_summary", "visual_transcript", "hybrid")
    ),
    trigger_ratio=_optional(_number),
    target_ratio=_optional(_number),
    summary_max_tokens=_optional(_INT),
    failure_behavior=_optional(_enum("stop_and_ask", "omit_older_context")),
    carry_forward_mode=_optional(
        _enum("memory_with_recent_turns", "memory_with_latest_exchange")
    ),
)
_FIELD_DRAFTS = _array(
    _record(
        name=_NONEMPTY,
        effective_value=_scalar,
        profile_override=_scalar,
        provenance=_enum("inherited", "explicit", "carried"),
        dirty=_boolean,
    )
)
_ENDPOINT_DRAFT = _optional(
    _record(
        value=_TEXT,
        bound_provider_config_key=_TEXT,
        dirty=_boolean,
        checked=_boolean,
    )
)
_SETTINGS_DRAFT = _record(
    settings=_SESSION_SETTINGS,
    context_policy_overrides=_CONTEXT_POLICY,
    field_drafts=_FIELD_DRAFTS,
    model_drafts=_array(
        _record(
            provider=_NONEMPTY,
            model=_MAYBE_TEXT,
            settings=_SESSION_SETTINGS,
            field_drafts=_FIELD_DRAFTS,
            endpoint_draft=_ENDPOINT_DRAFT,
        )
    ),
    endpoint_draft=_ENDPOINT_DRAFT,
)
_SUBMISSION = _record(
    submission_id=_NONEMPTY,
    action=_enum("apply_to_chat", "save_model_default", "make_new_chat_default"),
    surface=_enum("quick_popover", "full_settings"),
    origin=_record(
        session_id=_REFERENCE,
        persisted_conversation_id=_MAYBE_TEXT,
        conversation_binding_revision=_REVISION,
    ),
    draft=_SETTINGS_DRAFT,
    user_display_name_override=_MAYBE_TEXT,
    default_field_mask=_IDS,
)
_LIVE_COMMIT = _record(
    submission_id=_NONEMPTY,
    session_id=_REFERENCE,
    persisted_conversation_id=_MAYBE_TEXT,
    conversation_binding_revision=_REVISION,
    generation_revision=_REVISION,
    context_policy_revision=_REVISION,
    settings=_SESSION_SETTINGS,
    context_policy_overrides=_CONTEXT_POLICY,
    accepted_submission=_optional(_SUBMISSION),
)


_PROVIDER_SELECTION = _record(
    provider=_NONEMPTY,
    base_url=_MAYBE_TEXT,
    base_url_is_pinned=_boolean,
    execution_provider=_TEXT,
    configured_endpoint_fallback_allowed=_boolean,
    endpoint_provenance=_enum("durable_configuration", "ephemeral_session"),
    explicit_model=_MAYBE_TEXT,
    configured_model=_MAYBE_TEXT,
    temperature=_optional(_number),
    top_p=_optional(_number),
    **_OPTIONAL_GENERATION,
    system_prompt=_MAYBE_TEXT,
    workspace_context=_record(
        active_workspace_id=_TEXT,
        staged_sources=_array(
            _record(
                source_id=_NONEMPTY,
                label=_TEXT,
                source_type=_NONEMPTY,
                workspace_id=_MAYBE_TEXT,
            )
        ),
        active_run_id=_MAYBE_TEXT,
        handoff_id=_MAYBE_TEXT,
    ),
)
_SELECTION = _record(
    provider_selection=_PROVIDER_SELECTION,
    presentation_context=_optional(
        _record(
            user_name=_TEXT,
            assistant_kind=_MAYBE_TEXT,
            character_name=_MAYBE_TEXT,
            assistant_name=_MAYBE_TEXT,
            revision=_REVISION,
            transcript_style=_enum("neutral", "role_accents", "immersive_rp"),
        )
    ),
    rag_defaults=_optional(_object),
    tool_configuration=_optional(_object),
    skill_workspace_id=_MAYBE_TEXT,
    project_bindings_eligible=_boolean,
    agent_runtime_enabled=_boolean,
)
_ATTACHMENT = _record(
    attachment_id=_NONEMPTY,
    file_path=_TEXT,
    display_name=_TEXT,
    file_type=_NONEMPTY,
    insert_mode=_enum("attachment"),
    data=_optional(_base64),
    mime_type=_MAYBE_TEXT,
    text_content=_MAYBE_TEXT,
    original_size=_REVISION,
    processed_size=_REVISION,
)
_INPUT_FIELDS = dict(
    draft=_TEXT,
    authored_token=_AUTHORED_TOKEN,
    settings=_optional(_SESSION_SETTINGS),
    settings_revision=_REVISION,
    attachment_revision=_REVISION,
    attachment_ids=_array(_NONEMPTY, maximum=_MAX_STAGED_ATTACHMENTS, unique=True),
    attachments=_array(_ATTACHMENT, maximum=_MAX_STAGED_ATTACHMENTS),
    one_shot_prefill=_MAYBE_TEXT,
    prefill_revision=_REVISION,
    session_id=_REFERENCE,
    incarnation_id=_REFERENCE,
)
_PRESSED_INPUTS = _record(**_INPUT_FIELDS)
# A pristine backend session has no authored token until a real input update.
_PROJECTED_INPUTS = _record(
    **{**_INPUT_FIELDS, "authored_token": _optional(_AUTHORED_TOKEN)}
)


# The local decision/permission unions require an explicit variant tag even
# when the corresponding operation has no value fields.
_APPROVAL_VALUE = _enum(
    "approve_once", "approve_session", "allow_matching", "always_allow", "deny"
)
_DECISION = _tagged(
    {
        "ApprovalRoundDecision": {
            "round_id": _REFERENCE,
            "call_ids": _IDS,
            "decisions": _map(_APPROVAL_VALUE, keys=_NONEMPTY),
        },
        "ApprovalPresentationDecision": {
            "round_id": _REFERENCE, "call_ids": _IDS,
            "attachment_id": _REFERENCE, "visit": _SEQUENCE,
        },
        "ProjectBindingDecision": {
            "decision_id": _REFERENCE,
            "action": _enum("select", "disable", "cancel"),
            "binding_id": _MAYBE_TEXT,
        },
        "ProjectDispatchDecision": {
            "decision_id": _REFERENCE,
            "action": _enum("proceed", "cancel", "disable"),
        },
        "InitialHookDecision": {
            "review_id": _REFERENCE,
            "generation": _REVISION,
            "result": _record(kind=_enum("ready", "cancel", "settings")),
            "presentation_request_id": _REFERENCE,
        },
        "ChatCreateDecision": {
            "request_id": _REFERENCE,
            "allow": _boolean,
            "remember": _boolean,
        },
        "QuestionDecision": {
            "request_id": _REFERENCE,
            "answers": _array(
                _record(
                    question=_text(maximum=500),
                    selected=_array(_text(maximum=100), maximum=4),
                    other_text=_optional(_text(maximum=500)),
                    unanswered=_boolean,
                ),
                maximum=4,
            ),
        },
        "SkillInstallDecision": {"request_id": _REFERENCE, "allow": _boolean},
        "SkillScriptDecision": {
            "request_id": _REFERENCE,
            "allow": _boolean,
            "remember": _boolean,
        },
        "WorktreeMergeDecision": {"request_id": _REFERENCE, "allow": _boolean},
    }
)

def _decision_with_reasons(value, path):
    """Retain optional original denial provenance on the fixed approval variant."""
    _object(value, path)
    if value.get("kind") != "ApprovalRoundDecision" or "denial_reasons" not in value:
        _DECISION(value, path)
        return
    _DECISION(MappingProxyType({
        key: item for key, item in value.items() if key != "denial_reasons"
    }), path)
    reasons = value["denial_reasons"]
    _map(_TEXT, keys=_NONEMPTY)(reasons, f"{path}.denial_reasons")
    if any(value["decisions"].get(key) != "deny" for key in reasons):
        _reject(path, "denial reason does not address an explicit denial")


_HUB_TOOL = _record(
    server_key=_NONEMPTY,
    server_label=_TEXT,
    source=_enum("local", "builtin", "server"),
    name=_NONEMPTY,
    description=_TEXT,
    input_schema=_optional(_object),
    tags=_array(_TEXT),
    stale=_boolean,
    executable=_boolean,
)
_TOOL = {"server_key": _NONEMPTY, "tool_name": _NONEMPTY}
_PROFILE = {"profile_id": _NONEMPTY}
_EXPECT_POLICY = {
    "expected_profile_digest": _optional(_DIGEST),
    "expected_revision": _MAYBE_REVISION,
}
_STATE = _enum("allow", "ask", "deny")
_READ_TOOLS = {"tools": _array(_HUB_TOOL), **_PROFILE}
_REVIEW = {"review_request_id": _REFERENCE}
_PERMISSION = _tagged(
    {
        "ReadPermissionSnapshot": {"strict": _boolean},
        "ReadPermissionInventory": {},
        "ReadKillSwitch": {},
        "ReadEffectiveStates": _READ_TOOLS,
        "CapturePermissionRender": _READ_TOOLS,
        "ListSessionApprovals": _PROFILE,
        "RevokeSessionApproval": {**_TOOL, **_PROFILE},
        "ClearSessionApprovals": {"profile_id": _optional(_NONEMPTY)},
        "SetToolState": {
            **_TOOL,
            **_PROFILE,
            **_EXPECT_POLICY,
            "ui_state": _optional(_STATE),
            "tool": _optional(_HUB_TOOL),
            "expected_policy_chain_digest": _optional(_DIGEST),
        },
        "SetServerDefault": {
            "server_key": _NONEMPTY,
            "state": _optional(_STATE),
            **_PROFILE,
            **_EXPECT_POLICY,
        },
        "SetGlobalDefault": {"state": _STATE, **_PROFILE, **_EXPECT_POLICY},
        "SetKillSwitch": {"value": _boolean},
        "AddArgRule": {
            **_TOOL,
            **_PROFILE,
            "args": _object,
            "tool": _optional(_HUB_TOOL),
        },
        "RemoveArgRule": {**_TOOL, **_PROFILE, "rule_id": _NONEMPTY},
        "ReadArgRules": {**_TOOL, **_PROFILE},
        "DismissSuggestion": _TOOL,
        "RestoreSuggestion": _TOOL,
        "ClearSuggestionDismissals": {},
        "InspectImport": {
            "archive_path": _NONEMPTY,
            "destination_id": _NONEMPTY,
            "mappings": _array(
                _record(source_server_key=_NONEMPTY, destination_server_key=_NONEMPTY)
            ),
        },
        "ImportUnboundProfile": _REVIEW,
        "RemoveImportedProfile": {
            "profile_id": _NONEMPTY,
            "expected_revision": _REVISION,
        },
        "ListToolProfiles": {},
        "ReviewFirstBind": {
            "workspace_id": _NONEMPTY,
            "intended_defaults": _record(
                assistant_kind=_enum("persona"),
                assistant_id=_NONEMPTY,
                persona_memory_mode=_enum("read_only", "read_write"),
                voice=_null,
                style=_null,
                tool_policy_profile_id=_MAYBE_TEXT,
            ),
            "action": _enum("create", "set", "replace"),
        },
        "ConfirmFirstBind": _REVIEW,
    }
)


_FINGERPRINT = _record(
    preparation_id=_TURN_ID,
    session_id=_REFERENCE,
    conversation_id=_TURN_ID,
    title_hash=_DIGEST,
    attempt_id=_TURN_ID,
    origin=_enum("manual", "queued", "agent_chat_start"),
    queue_entry_id=_optional(_TURN_ID),
    user_message_id=_TURN_ID,
    assistant_message_id=_TURN_ID,
    digest=_DIGEST,
)
_PLAN = {
    "received_request_id": _REFERENCE,
    "turn_id": _TURN_ID,
    "ordinal": _SEQUENCE,
    "fingerprint": _FINGERPRINT,
}
_SOURCE = _enum("configuration", "workspace")


def _open_session_request(value, path):
    fields = dict(title=_TEXT, workspace_id=_MAYBE_TEXT, conversation_id=_MAYBE_TEXT)
    _object(value, path)
    if "new_chat" in value:
        fields.update(new_chat=_boolean, ephemeral=_boolean, new_chat_default_generation=_REVISION)
        if value["new_chat"] is not True or value.get("conversation_id") is not None:
            _reject(path, "invalid new-chat intent")
    _record(**fields)(value, path)


def _projection_request(value, path):
    _object(value, path)
    if not value:
        return  # Original callers request a complete first snapshot.
    fields = {"known_display_revision": _MAYBE_REVISION}
    if "known_marker_revision" in value:
        fields["known_marker_revision"] = _MAYBE_REVISION
    if "run_history" in value:
        fields["run_history"] = _record(
            conversation_id=_NONEMPTY, include_superseded=_boolean,
            limit=_optional(_INT), agent_kind=_MAYBE_TEXT,
        )
    if "view_checkpoint" in value:
        fields["view_checkpoint"] = _record(
            attachment_id=_NONEMPTY, visit=_SEQUENCE,
            action=_enum("mount", "visible", "hidden"),
        )
    _record(**fields)(value, path)


def _projection_result(value, path):
    from .console_execution_projection_result import validate_projection_payload

    validate_projection_payload(_detach(value, path, 2, set(), thaw=True))


_COMMANDS = {
    "UpdateInputs": _record(
        draft=_TEXT,
        authored_token=_AUTHORED_TOKEN,
        expected_settings_revision=_MAYBE_REVISION,
        expected_settings=_optional(_SESSION_SETTINGS),
        expected_attachment_ids=_optional(_IDS),
        remove_attachment_ids=_IDS,
        prefill=_optional(_record(expected_revision=_REVISION, value=_MAYBE_TEXT)),
    ),
    "ReceiveTurn": _record(
        turn_id=_TURN_ID,
        draft_revision=_REVISION,
        selection=_SELECTION,
        expected_settings_revision=_MAYBE_REVISION,
        expected_attachment_revision=_MAYBE_REVISION,
        expected_prefill_revision=_MAYBE_REVISION,
        pressed_inputs=_PRESSED_INPUTS,
    ),
    "ApplySettings": _record(submission=_SUBMISSION),
    "OpenSession": _open_session_request,
    "CloseSession": _record(expected_revision=_REVISION),
    "ResolveDecision": _record(decision=_decision_with_reasons),
    "ProjectDecisionPresentation": _record(
        attachment_id=_REFERENCE, visit=_SEQUENCE, decision_id=_REFERENCE,
        action=_enum("mounted", "hidden"),
    ),
    "ReadProjection": _projection_request,
    "CancelTurn": _record(turn_id=_TURN_ID),
    "PermissionAction": _record(operation=_PERMISSION),
    "FenceSource": _record(source=_SOURCE),
    "CloseIntake": _EMPTY,
    "Drain": _EMPTY,
    "Shutdown": _EMPTY,
    "CommitPlanAck": _record(**_PLAN),
    "ReplaceConfiguration": _record(
        fence_request_id=_REFERENCE,
        serialized=_TEXT,
        expected_snapshot=_optional(_record(path=_NONEMPTY, serialized=_MAYBE_TEXT)),
        create_backup=_boolean,
    ),
    "RefreshConfiguration": _record(fence_request_id=_REFERENCE),
    "SetWorkspaceBindingAccess": _record(
        fence_request_id=_REFERENCE, binding_id=_NONEMPTY, allow_write=_boolean
    ),
    "ApplySourceRevision": _record(publication_request_id=_REFERENCE),
}


def _unique_pairs(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        _reject("recovery", "duplicate embedded field")
    return value


def _embedded_json(check: _Check, max_bytes: int) -> _Check:
    def validate(value, path):
        _check_string(value, path)
        if len(value.encode("utf-8")) > max_bytes:
            _reject(path, "checkpoint value too large")
        try:
            decoded = json.loads(value, object_pairs_hook=_unique_pairs)
        except (ValueError, RecursionError):
            raise ConsoleProtocolError(f"{path}: invalid embedded JSON") from None
        check(_detach(decoded, path, 1, set()), path)

    return validate


# Exact body-free metadata issued by console_dispatch_checkpoint's dumpers.
_AUTHORITY_JSON = _embedded_json(
    _record(
        policy=_record(
            auto_retrieve=_enum("never", "automatic"),
            assistant_access=_enum("blocked", "allowed"),
            policy_revision=_optional(_SEQUENCE),
            source=_enum(
                "new_session", "durable", "missing", "temporary", "unavailable"
            ),
            error_code=_optional(_ERROR_ID),
        ),
        direct_library_tools=_boolean,
        source_types=_tuple(_enum("notes"), _enum("media"), _enum("conversations")),
        scope_snapshot=_record(
            note_ids=_array(_TURN_ID, unique=True),
            media_ids=_array(_TURN_ID, unique=True),
            conversations_allowed=_boolean,
        ),
        provider_intent=_record(
            provider=_NONEMPTY, model=_MAYBE_TEXT, endpoint=_MAYBE_TEXT
        ),
        attempt_id=_TURN_ID,
    ),
    4096,
)
_DESTINATION_FIELDS = {
    "provider": _NONEMPTY,
    "model": _MAYBE_TEXT,
    "endpoint_identity": _NONEMPTY,
    "egress_class": _enum("on_device", "private_network", "public_network", "unknown"),
}
_DESTINATION_V1 = _record(**_DESTINATION_FIELDS)
_DESTINATION_V2 = _record(
    **_DESTINATION_FIELDS,
    schema_version=_INT,
    continuation_provider=_NONEMPTY,
    continuation_protocol=_NONEMPTY,
    continuation_base_url=_MAYBE_TEXT,
)


def _destination(value, path):
    _object(value, path)
    if "schema_version" not in value:
        _DESTINATION_V1(value, path)
        return
    _DESTINATION_V2(value, path)
    if value["schema_version"] != 2:
        _reject(path, "unknown destination schema version")


_RECOVERY = _record(
    fingerprint=_FINGERPRINT,
    user_content_sha256=_DIGEST,
    parent_message_id=_optional(_TURN_ID),
    attachments=_array(
        _tuple(_REVISION, _DIGEST, _NONEMPTY, _TEXT), maximum=_MAX_STAGED_ATTACHMENTS
    ),
    frozen_authority_json=_AUTHORITY_JSON,
    resolved_destination_json=_embedded_json(_destination, 2048),
    reconstructability_json=_embedded_json(
        _record(
            attachments_reconstructable=_boolean,
            evidence_reconstructable=_boolean,
            prefill_reconstructable=_boolean,
            opaque_reference=_optional(
                _pattern(r"opaque:[A-Za-z0-9][A-Za-z0-9._:-]{0,199}")
            ),
        ),
        2048,
    ),
    project_context_sha256=_DIGEST,
    first_persist=_boolean,
    generation_revision=_REVISION,
    context_policy_revision=_REVISION,
    roleplay_identity_revision=_REVISION,
    user_root_fork=_boolean,
    agent_chat_start=_optional(
        _record(
            attempt_id=_TURN_ID, source_run_id=_TURN_ID, source_conversation_id=_TURN_ID
        )
    ),
    context_policy_initial_revision=_optional(_SEQUENCE),
    initial_user_version=_SEQUENCE,
    initial_assistant_version=_SEQUENCE,
    initial_checkpoint_revision=_SEQUENCE,
    checkpoint=_null,  # A planned, uncommitted turn has no checkpoint yet.
)
_COMMIT_PLANNED = _record(**_PLAN, recovery=_RECOVERY)
_APPLIED_BOOL = _record(applied=_boolean)


def _stored_json(_value: object, _path: str) -> None:
    # _detach has checked every node. The original reader owns stored schema policy.
    return None


def _iso_datetime(value: object, path: str) -> None:
    _check_string(value, path)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        _reject(path, "invalid timestamp")
    if parsed.tzinfo is None or parsed.isoformat() != value:
        _reject(path, "expected timezone-aware ISO timestamp")


_PERMISSION_SNAPSHOT_FIELDS = {
    "payload": _object,
    "generation": _TEXT,
    "file_exists": _boolean,
}
_PERMISSION_SNAPSHOT = _record(**_PERMISSION_SNAPSHOT_FIELDS)
_EFFECTIVE_STATE_ROWS = _array(
    _record(
        server_key=_TEXT,
        tool_name=_TEXT,
        state=_STATE,
        origin=_TEXT,
        config_changed=_boolean,
        risk_floored=_boolean,
    )
)
_TOOL_PAIR = _tuple(_TEXT, _TEXT)


def _effective_states(value, path):
    _EFFECTIVE_STATE_ROWS(value, path)
    if len({(row["server_key"], row["tool_name"]) for row in value}) != len(value):
        _reject(path, "repeated tool key")


_ARG_RULE = _record(
    rule_id=_TEXT, args_json=_TEXT, created_at=_stored_json, profile_id=_TEXT
)


def _arg_rule(value, path):
    _ARG_RULE(value, path)
    if value["rule_id"] != value["args_json"]:
        _reject(f"{path}.rule_id", "does not match args_json")


_PORTABLE_AUTHORITY = _enum("mcp", "builtin")
_PORTABLE_RULE = _record(
    authority=_PORTABLE_AUTHORITY,
    server_key=_TEXT,
    tool_name=_TEXT,
    state=_STATE,
    contract_sha256=_optional(_DIGEST),
)
_PORTABLE_RULES = _array(_PORTABLE_RULE)
_PRODUCER_PAIR = _tuple(_TEXT, _TEXT)


def _producer(value, path):
    if value != ():
        _PRODUCER_PAIR(value, path)


_PROFILE_PRESENTATION = _record(
    profile_id=_TEXT,
    origin=_enum("local", "workspace-managed", "imported"),
    lifecycle_valid=_boolean,
    binding_state=_enum("bound", "unbound"),
    first_bind_confirmation_required=_boolean,
    reference_counts=_tuple(_REVISION, _REVISION),
    posture_counts=_tuple(_REVISION, _REVISION, _REVISION),
    receipt_health=_enum("not_applicable", "available", "unavailable"),
    removal_eligible=_boolean,
    removal_blocker=_optional(
        _enum(
            "lifecycle_invalid",
            "not_imported",
            "referenced",
            "in_use",
            "receipt_unavailable",
        )
    ),
    revision=_MAYBE_REVISION,
    policy_digest=_MAYBE_TEXT,
)
_TOOL_PAIRS = _array(_TOOL_PAIR)
_BINDING_SUMMARY = _record(
    global_fallback=_STATE,
    builtin_fallback=_STATE,
    allow_server_fallbacks=_array(_TEXT),
    stored_exact_allows=_TOOL_PAIRS,
    effective_allows=_TOOL_PAIRS,
    unavailable_allows=_TOOL_PAIRS,
    downgraded_allows=_TOOL_PAIRS,
    high_risk_allows=_TOOL_PAIRS,
    allow_count=_REVISION,
    ask_count=_REVISION,
    deny_count=_REVISION,
    inventory_digest=_TEXT,
    effective_asks=_TOOL_PAIRS,
    effective_denies=_TOOL_PAIRS,
)
_PROFILE_RECEIPT_FIELDS = {
    "profile_id": _TEXT,
    "policy_digest": _TEXT,
    "revision": _REVISION,
    "receipt_id": _TEXT,
}
_PERMISSION_RESULT = _tagged(
    {
        "ReadPermissionSnapshot": {
            "result": _tagged(
                {
                    "compatibility": {"payload": _object},
                    "snapshot": _PERMISSION_SNAPSHOT_FIELDS,
                }
            )
        },
        "ReadPermissionInventory": {"result": _PERMISSION_SNAPSHOT},
        "ReadKillSwitch": {"result": _boolean},
        "ReadEffectiveStates": {"result": _effective_states},
        "CapturePermissionRender": {
            "result": _record(
                snapshot=_PERMISSION_SNAPSHOT,
                effective_states=_effective_states,
                profile_identity=_optional(_tuple(_TEXT, _MAYBE_REVISION, _TEXT)),
            )
        },
        "ListSessionApprovals": {"result": _array(_TOOL_PAIR, unique=True)},
        "RevokeSessionApproval": {"result": _boolean},
        "ClearSessionApprovals": {"result": _null},
        "SetToolState": {"result": _null},
        "SetServerDefault": {"result": _null},
        "SetGlobalDefault": {"result": _null},
        "SetKillSwitch": {"result": _null},
        "AddArgRule": {"result": _null},
        "RemoveArgRule": {"result": _boolean},
        "ReadArgRules": {"result": _array(_arg_rule)},
        "DismissSuggestion": {"result": _null},
        "RestoreSuggestion": {"result": _null},
        "ClearSuggestionDismissals": {"result": _null},
        "InspectImport": {
            "result": _record(
                display_name=_TEXT,
                producer=_producer,
                destination_id=_TEXT,
                expires_at=_iso_datetime,
                mappings=_array(
                    _record(source_server_key=_TEXT, destination_server_key=_TEXT)
                ),
                matched=_array(
                    _record(
                        source_rule=_PORTABLE_RULE,
                        destination_identity=_tuple(_PORTABLE_AUTHORITY, _TEXT, _TEXT),
                        destination_contract_sha256=_DIGEST,
                        destination_connected=_boolean,
                    )
                ),
                changed=_PORTABLE_RULES,
                missing=_PORTABLE_RULES,
                pending_denies=_PORTABLE_RULES,
                omitted_allow_ask=_PORTABLE_RULES,
                fallbacks=_array(
                    _record(
                        authority=_PORTABLE_AUTHORITY,
                        server_key=_TEXT,
                        state=_enum("ask", "deny"),
                    )
                ),
            )
        },
        "ImportUnboundProfile": {
            "result": _record(
                installed=_record(**_PROFILE_RECEIPT_FIELDS),
                store_generation=_TEXT,
                reconciled=_boolean,
            )
        },
        "RemoveImportedProfile": {
            "result": _record(
                tombstone=_record(
                    **_PROFILE_RECEIPT_FIELDS,
                    profile_kind=_enum("tool_pack_tombstone"),
                    receipt_digest=_TEXT,
                ),
                store_generation=_TEXT,
                reconciled=_boolean,
            )
        },
        "ListToolProfiles": {
            "result": _record(
                profiles=_array(_PROFILE_PRESENTATION),
                unavailable_category=_optional(
                    _enum(
                        "authority_unavailable",
                        "references_unavailable",
                        "inventory_unavailable",
                    )
                ),
            )
        },
        "ReviewFirstBind": {
            "result": _record(
                workspace_id=_TEXT,
                action=_enum("create", "set", "replace"),
                intended_defaults_digest=_TEXT,
                profile_id=_TEXT,
                policy_digest=_TEXT,
                revision=_REVISION,
                expires_at=_iso_datetime,
                summary=_BINDING_SUMMARY,
            )
        },
        "ConfirmFirstBind": {"result": _record(confirmation_request_id=_REFERENCE)},
    }
)
_APPLIED = {
    "ReadProjection": _projection_result,
    "UpdateInputs": _record(draft_revision=_REVISION),
    "ReceiveTurn": _record(turn_id=_TURN_ID),
    "ApplySettings": _LIVE_COMMIT,
    "OpenSession": _record(session_id=_REFERENCE, incarnation_id=_REFERENCE),
    "CloseSession": _APPLIED_BOOL,
    "ResolveDecision": _APPLIED_BOOL,
    "ProjectDecisionPresentation": _APPLIED_BOOL,
    "PermissionAction": _PERMISSION_RESULT,
    "CancelTurn": _APPLIED_BOOL,
    "FenceSource": _record(source=_SOURCE, revision=_REVISION),
    "SetWorkspaceBindingAccess": _record(
        binding_id=_NONEMPTY,
        workspace_id=_NONEMPTY,
        access=_enum("ro", "rw"),
        source_revision=_REVISION,
    ),
    "ApplySourceRevision": _record(source_revision=_REVISION),
    "CommitPlanAck": _APPLIED_BOOL,
    "CloseIntake": _APPLIED_BOOL,
    "Drain": _APPLIED_BOOL,
    "Shutdown": _APPLIED_BOOL,
}
_RECEIVED = _record(sequence=_SEQUENCE)
_RETIRED = _record(operation_id=_REFERENCE, retired=_boolean)
_FAILED = _record(code=_ERROR_ID, stage=_ERROR_ID)
_READY = _record(pid=_SEQUENCE, source_revision=_REVISION)


def _validate_message(identity, kind, operation, payload):
    if type(kind) is not str or type(operation) is not str:
        _reject("message", "invalid kind or operation")
    if kind == "ready":
        if operation != "Initialize":
            _reject("message", "Ready requires Initialize")
        _READY(payload, "payload")
        return
    if operation not in _COMMANDS:
        _reject("message", "unknown operation")
    if kind == "command":
        _COMMANDS[operation](payload, "payload")
        if (
            operation == "ApplySettings"
            and payload["submission"]["origin"]["session_id"] != identity.session_id
        ):
            _reject("payload.submission.origin", "session does not match identity")
        if operation == "ReceiveTurn":
            pressed = payload["pressed_inputs"]
            if (
                pressed["session_id"] != identity.session_id
                or pressed["incarnation_id"] != identity.incarnation_id
                or tuple(item["attachment_id"] for item in pressed["attachments"])
                != pressed["attachment_ids"]
            ):
                _reject(
                    "payload.pressed_inputs",
                    "pressed identity or attachment order changed",
                )
        return
    if kind == "input_projected" and operation == "ReceiveTurn":
        _record(turn_id=_TURN_ID, draft_committed=_boolean)(payload, "payload")
        return
    if kind == "terminal" and operation == "ReceiveTurn":
        _record(turn_id=_TURN_ID, accepted=_boolean)(payload, "payload")
        return
    if kind == "commit_planned" and operation == "ReceiveTurn":
        _COMMIT_PLANNED(payload, "payload")
        recovery = payload["recovery"]
        if recovery["fingerprint"] != payload["fingerprint"]:
            _reject("payload.recovery", "fingerprint does not match plan")
        if (
            len(
                json.dumps(
                    _detach(recovery, "payload.recovery", 3, set(), thaw=True),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            > 16_384
        ):
            _reject("payload.recovery", "recovery projection too large")
        return
    if kind == "applied":
        check = _APPLIED.get(operation)
    else:
        check = {"received": _RECEIVED, "retired": _RETIRED, "failed": _FAILED}.get(
            kind
        )
    if check is None:
        _reject("message", "unsupported reply kind or operation")
    check(payload, "payload")
    if (
        kind == "applied"
        and operation == "ReadProjection"
        and (
            payload["source_revision"] != identity.source_revision
            or payload["session_id"] != identity.session_id
            or payload["incarnation_id"] != identity.incarnation_id
        )
    ):
        _reject("payload", "projection identity changed")
    if kind == "applied" and operation == "ApplySettings":
        accepted = payload["accepted_submission"]
        if (
            accepted is not None
            and accepted["submission_id"] != payload["submission_id"]
        ):
            _reject(
                "payload.accepted_submission", "submission does not match live commit"
            )
    if (
        kind == "applied"
        and operation == "PermissionAction"
        and payload["kind"] == "ConfirmFirstBind"
        and payload["result"]["confirmation_request_id"] != identity.request_id
    ):
        _reject(
            "payload.result.confirmation_request_id", "request does not match identity"
        )


@dataclass(frozen=True, slots=True)
class ConsoleWireMessage:
    """One detached message, with read-only objects and tuples in its payload."""

    identity: WireIdentity
    kind: str
    operation: str
    payload: Mapping[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        _check_identity(self.identity)
        payload = _detach(self.payload, "payload", 2, set())
        _validate_message(self.identity, self.kind, self.operation, payload)
        object.__setattr__(self, "payload", payload)


def encode_message(message: ConsoleWireMessage) -> dict[str, JSONValue]:
    """Validate a Console message and return fresh plain JSON containers."""
    if type(message) is not ConsoleWireMessage:
        _reject("message", "expected ConsoleWireMessage")
    _check_identity(message.identity)
    payload = _detach(message.payload, "payload", 2, set(), thaw=True)
    _validate_message(
        message.identity, message.kind, message.operation, message.payload
    )
    return {
        "version": PROTOCOL_VERSION,
        "identity": {
            name: getattr(message.identity, name) for name in _IDENTITY_FIELDS
        },
        "kind": message.kind,
        "operation": message.operation,
        "payload": payload,
    }


def decode_message(payload: object) -> ConsoleWireMessage:
    """Validate plain JSON values and detach the exact Console envelope."""
    if (
        type(payload) is not dict
        or any(type(key) is not str for key in payload)
        or payload.keys() != {"version", "identity", "kind", "operation", "payload"}
    ):
        _reject("message", "unexpected or missing envelope fields")
    if type(payload["version"]) is not int or payload["version"] != PROTOCOL_VERSION:
        _reject("version", "unsupported version")
    identity = payload["identity"]
    if (
        type(identity) is not dict
        or any(type(key) is not str for key in identity)
        or identity.keys() != _IDENTITY_FIELDS.keys()
    ):
        _reject("identity", "unexpected or missing identity fields")
    return ConsoleWireMessage(
        WireIdentity(**identity),
        payload["kind"],
        payload["operation"],
        payload["payload"],
    )
