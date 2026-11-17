"""The Console wire boundary accepts only the fixed detached values."""

import base64
from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Chat.console_execution_protocol import (
    PROTOCOL_VERSION,
    ConsoleProtocolError,
    ConsoleWireMessage,
    WireIdentity,
    decode_message,
    encode_message,
)


def envelope(operation, payload, *, kind="command"):
    return {
        "version": PROTOCOL_VERSION,
        "identity": {
            "profile_epoch": "profile",
            "service_epoch": "service",
            "sequence": 1,
            "request_id": "request",
            "session_id": "session",
            "incarnation_id": "incarnation",
            "source_revision": 0,
        },
        "kind": kind,
        "operation": operation,
        "payload": payload,
    }


def test_original_turn_checkpoints_remain_distinct_from_retirement():
    for kind, flag in (("input_projected", "draft_committed"), ("terminal", "accepted")):
        message = decode_message(envelope("ReceiveTurn", {"turn_id": "turn", flag: False}, kind=kind))
        assert encode_message(message)["payload"][flag] is False
        with pytest.raises(ConsoleProtocolError):
            decode_message(envelope("ReceiveTurn", {"turn_id": "turn", flag: True, "retired": True}, kind=kind))
        with pytest.raises(ConsoleProtocolError):
            decode_message(envelope("ReadProjection", {"turn_id": "turn", flag: True}, kind=kind))
        with pytest.raises(ConsoleProtocolError):
            decode_message(envelope("ReceiveTurn", {"turn_id": "turn", flag: 1}, kind=kind))


def test_open_session_keeps_stock_shape_and_closed_new_chat_intent():
    original = {"title": "New chat", "workspace_id": None, "conversation_id": None}
    assert encode_message(decode_message(envelope("OpenSession", original)))["payload"] == original
    new_chat = dict(original, new_chat=True, ephemeral=True, new_chat_default_generation=3)
    assert encode_message(decode_message(envelope("OpenSession", new_chat)))["payload"] == new_chat
    for invalid in (
        dict(new_chat, conversation_id="saved"),
        dict(new_chat, new_chat=False),
        dict(new_chat, ephemeral=1),
        dict(new_chat, new_chat_default_generation=-1),
        dict(original, ephemeral=True),
    ):
        with pytest.raises(ConsoleProtocolError):
            decode_message(envelope("OpenSession", invalid))


def attachment(identifier, content):
    return {
        "attachment_id": identifier,
        "file_path": "",
        "display_name": f"{identifier}.png",
        "file_type": "image",
        "insert_mode": "attachment",
        "data": base64.b64encode(content).decode("ascii"),
        "mime_type": "image/png",
        "text_content": None,
        "original_size": len(content),
        "processed_size": len(content),
    }


def receive_payload():
    return {
        "turn_id": "turn",
        "draft_revision": 3,
        "expected_settings_revision": 7,
        "expected_attachment_revision": 2,
        "expected_prefill_revision": 0,
        "selection": {
            "provider_selection": {
                "provider": "openai",
                "base_url": None,
                "base_url_is_pinned": False,
                "execution_provider": "",
                "configured_endpoint_fallback_allowed": True,
                "endpoint_provenance": "durable_configuration",
                "explicit_model": "vision-model",
                "configured_model": None,
                "temperature": None,
                "top_p": None,
                "min_p": None,
                "top_k": None,
                "max_tokens": None,
                "seed": None,
                "presence_penalty": None,
                "frequency_penalty": None,
                "reasoning_effort": None,
                "reasoning_summary": None,
                "verbosity": None,
                "thinking_effort": None,
                "thinking_budget_tokens": None,
                "streaming": True,
                "system_prompt": None,
                "workspace_context": {
                    "active_workspace_id": "global",
                    "staged_sources": [],
                    "active_run_id": None,
                    "handoff_id": None,
                },
            },
            "presentation_context": None,
            "rag_defaults": None,
            "tool_configuration": {},
            "skill_workspace_id": None,
            "project_bindings_eligible": True,
            "agent_runtime_enabled": False,
        },
        "pressed_inputs": {
            "draft": "Captured before the later edit",
            "authored_token": [12, 2],
            "session_id": "session",
            "incarnation_id": "incarnation",
            "settings": None,
            "settings_revision": 7,
            "attachment_revision": 2,
            "attachment_ids": ["second", "first"],
            "attachments": [attachment("second", b"z"), attachment("first", b"a")],
            "one_shot_prefill": "",
            "prefill_revision": 0,
        },
    }


def test_pressed_input_round_trip_detaches_both_directions():
    source = envelope("ReceiveTurn", receive_payload())
    message = decode_message(source)
    source["payload"]["pressed_inputs"]["draft"] = "Later draft"
    source["payload"]["pressed_inputs"]["attachments"].reverse()

    pressed = message.payload["pressed_inputs"]
    assert pressed["draft"] == "Captured before the later edit"
    assert pressed["attachment_ids"] == ("second", "first")
    assert pressed["one_shot_prefill"] == ""
    assert pressed["settings"] is None
    with pytest.raises(TypeError):
        pressed["attachments"][0]["display_name"] = "changed"
    with pytest.raises(FrozenInstanceError):
        message.operation = "CancelTurn"

    encoded = encode_message(message)
    assert encoded == envelope("ReceiveTurn", receive_payload())
    encoded["payload"]["pressed_inputs"]["attachments"][0]["data"] = ""
    assert encode_message(message) == envelope("ReceiveTurn", receive_payload())


@pytest.mark.parametrize("fault", ["noncanonical_base64", "wrong_order", "sixth_item"])
def test_rejects_invalid_pressed_attachment_values(fault):
    payload = receive_payload()
    pressed = payload["pressed_inputs"]
    if fault == "noncanonical_base64":
        # "eh==" and "eg==" decode to the same byte, with unequal pad bits.
        pressed["attachments"][0]["data"] = "eh=="
    elif fault == "wrong_order":
        pressed["attachment_ids"].reverse()
    else:
        for index in range(4):
            identifier = f"more-{index}"
            pressed["attachments"].append(attachment(identifier, b"x"))
            pressed["attachment_ids"].append(identifier)
    with pytest.raises(ConsoleProtocolError):
        decode_message(envelope("ReceiveTurn", payload))


def test_update_inputs_preserves_empty_and_null_and_has_no_executed_draft_cap():
    payload = {
        "draft": "x" * (4 * 1024 * 1024 + 1),
        "authored_token": [0, 0],
        "expected_settings_revision": None,
        "expected_settings": None,
        "expected_attachment_ids": None,
        "remove_attachment_ids": [],
        "prefill": {"expected_revision": 0, "value": None},
    }
    raw = envelope("UpdateInputs", payload)
    assert encode_message(decode_message(raw)) == raw


def test_apply_outcome_keeps_accepted_submission_and_excludes_the_local_lease():
    settings = {
        "provider": "openai",
        "model": "accepted-model",
        "base_url": None,
        "temperature": None,
        "top_p": 0.9,
        "min_p": None,
        "top_k": None,
        "max_tokens": None,
        "seed": None,
        "presence_penalty": None,
        "frequency_penalty": None,
        "reasoning_effort": None,
        "reasoning_summary": None,
        "verbosity": None,
        "thinking_effort": None,
        "thinking_budget_tokens": None,
        "streaming": False,
        "character_label": "",
        "system_prompt": "",
        "source": "user",
        "pinned_prefill": None,
        "persona_memory_mode": None,
    }
    policy = {
        "budget_mode": None,
        "custom_budget_tokens": None,
        "compaction_mode": "ask",
        "compaction_representation": None,
        "trigger_ratio": None,
        "target_ratio": None,
        "summary_max_tokens": None,
        "failure_behavior": None,
        "carry_forward_mode": None,
    }
    binding = {
        "session_id": "session",
        "persisted_conversation_id": "saved",
        "conversation_binding_revision": 0,
    }
    submission = {
        "submission_id": "submission",
        "action": "apply_to_chat",
        "surface": "full_settings",
        "origin": binding,
        "draft": {
            "settings": settings,
            "context_policy_overrides": policy,
            "field_drafts": [
                {
                    "name": "temperature",
                    "effective_value": None,
                    "profile_override": None,
                    "provenance": "inherited",
                    "dirty": False,
                }
            ],
            "model_drafts": [],
            "endpoint_draft": None,
        },
        "user_display_name_override": None,
        "default_field_mask": [],
    }
    command = envelope("ApplySettings", {"submission": submission})
    assert encode_message(decode_message(command)) == command
    outcome = {
        "submission_id": "submission",
        **binding,
        "generation_revision": 1,
        "context_policy_revision": 2,
        "settings": settings,
        "context_policy_overrides": policy,
        "accepted_submission": submission,
    }
    raw = envelope("ApplySettings", outcome, kind="applied")
    message = decode_message(raw)
    assert (
        message.payload["accepted_submission"]["draft"]["settings"]["model"]
        == "accepted-model"
    )
    assert encode_message(message) == raw
    for unknown in ("durability_admission", "durable"):
        outcome[unknown] = None
        with pytest.raises(ConsoleProtocolError):
            decode_message(raw)
        del outcome[unknown]


def test_commit_planned_and_ack_retain_the_complete_fingerprint():
    fingerprint = {
        "preparation_id": "preparation",
        "session_id": "session",
        "conversation_id": "saved",
        "title_hash": "a" * 64,
        "attempt_id": "attempt",
        "origin": "manual",
        "queue_entry_id": None,
        "user_message_id": "user",
        "assistant_message_id": "assistant",
        "digest": "b" * 64,
    }
    raw = envelope(
        "CommitPlanAck",
        {
            "received_request_id": "receive-request",
            "turn_id": "turn",
            "ordinal": 1,
            "fingerprint": fingerprint,
        },
    )
    assert encode_message(decode_message(raw)) == raw
    recovery = {
        "fingerprint": fingerprint.copy(),
        "user_content_sha256": "c" * 64,
        "parent_message_id": None,
        "attachments": [],
        "frozen_authority_json": (
            '{"policy":{"auto_retrieve":"never","assistant_access":"blocked",'
            '"policy_revision":null,"source":"new_session","error_code":null},'
            '"direct_library_tools":false,"source_types":["notes","media","conversations"],'
            '"scope_snapshot":{"note_ids":[],"media_ids":[],"conversations_allowed":false},'
            '"provider_intent":{"provider":"openai","model":null,"endpoint":null},'
            '"attempt_id":"attempt"}'
        ),
        "resolved_destination_json": (
            '{"provider":"openai","model":null,"endpoint_identity":"default",'
            '"egress_class":"unknown"}'
        ),
        "reconstructability_json": (
            '{"attachments_reconstructable":true,"evidence_reconstructable":true,'
            '"prefill_reconstructable":true,"opaque_reference":null}'
        ),
        "project_context_sha256": "d" * 64,
        "first_persist": True,
        "generation_revision": 0,
        "context_policy_revision": 0,
        "roleplay_identity_revision": 0,
        "user_root_fork": False,
        "agent_chat_start": None,
        "context_policy_initial_revision": None,
        "initial_user_version": 1,
        "initial_assistant_version": 1,
        "initial_checkpoint_revision": 1,
        "checkpoint": None,
    }
    planned = envelope(
        "ReceiveTurn", {**raw["payload"], "recovery": recovery}, kind="commit_planned"
    )
    assert encode_message(decode_message(planned)) == planned
    recovery["fingerprint"]["digest"] = "e" * 64
    with pytest.raises(ConsoleProtocolError, match="fingerprint does not match"):
        decode_message(planned)
    del fingerprint["title_hash"]
    with pytest.raises(ConsoleProtocolError):
        decode_message(raw)


@pytest.mark.parametrize(
    "update",
    [
        {"kind": "UnregisteredPermission"},
        {"args": {"value": float("nan")}},
        {"args": {"guard": lambda: None}},
        {"source_authority": {}},
    ],
)
def test_named_permission_arguments_reject_unknown_or_non_json_data(update):
    operation = {
        "kind": "AddArgRule",
        "server_key": "local:docs",
        "tool_name": "read",
        "args": {"path": ["docs", {"relative": True}]},
        "tool": None,
        "profile_id": "default",
    }
    raw = envelope("PermissionAction", {"operation": operation})
    assert encode_message(decode_message(raw)) == raw
    operation.update(update)
    with pytest.raises(ConsoleProtocolError):
        decode_message(raw)


def test_named_json_field_keeps_framing_depth_and_rejects_cycles():
    args = {"value": None}
    operation = {
        "kind": "AddArgRule",
        "server_key": "local:docs",
        "tool_name": "read",
        "args": args,
        "tool": None,
        "profile_id": "default",
    }
    raw = envelope("PermissionAction", {"operation": operation})
    # The args object has depth four, so 124 list levels reach depth 128.
    for _ in range(124):
        args["value"] = [args["value"]]
    assert encode_message(decode_message(raw)) == raw
    args["value"] = [args["value"]]
    with pytest.raises(ConsoleProtocolError, match="excessive nesting"):
        decode_message(raw)
    args["value"] = args
    with pytest.raises(ConsoleProtocolError, match="cyclic value"):
        decode_message(raw)


def test_process_local_publication_and_first_bind_tokens_are_not_payload_fields():
    values = {
        "identity": WireIdentity("profile", "service", 2, "request", "", "", 0),
        "kind": "command",
        "operation": "ApplySourceRevision",
    }
    message = ConsoleWireMessage(**values, payload={"publication_request_id": "prior"})
    assert encode_message(message)["payload"] == {"publication_request_id": "prior"}
    with pytest.raises(ConsoleProtocolError):
        ConsoleWireMessage(**values, payload={"publication": object()})
    with pytest.raises(ConsoleProtocolError):
        decode_message(
            envelope(
                "PermissionAction",
                {
                    "operation": {
                        "kind": "ConfirmFirstBind",
                        "review": {"token": "local"},
                    }
                },
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("sequence", True), ("source_revision", -1), ("incarnation_id", "")],
)
def test_identity_follows_original_exact_integer_and_pairing_rules(field, value):
    raw = envelope("ReadProjection", {})
    raw["identity"][field] = value
    with pytest.raises(ConsoleProtocolError):
        decode_message(raw)


def test_reply_shapes_and_kinds_are_closed():
    ready = envelope("Initialize", {"pid": 1, "source_revision": 0}, kind="ready")
    assert encode_message(decode_message(ready)) == ready
    retired = envelope(
        "ReceiveTurn", {"operation_id": "turn", "retired": False}, kind="retired"
    )
    assert encode_message(decode_message(retired)) == retired
    with pytest.raises(ConsoleProtocolError):
        decode_message(envelope("ReadProjection", {"messages": []}, kind="ready"))
    with pytest.raises(ConsoleProtocolError):
        decode_message(
            envelope("CancelTurn", {"sequence": 1, "extra": None}, kind="received")
        )
    with pytest.raises(ConsoleProtocolError):
        decode_message(envelope("ReadProjection", {"messages": []}, kind="applied"))


def permission_applied(kind, result):
    return envelope(
        "PermissionAction", {"kind": kind, "result": result}, kind="applied"
    )


def test_permission_snapshots_preserve_only_designated_legacy_json():
    payload = {"profiles": {"old": {"tool_pack_lifecycle": [None, {"odd": True}]}}}
    snapshot = {
        "payload": payload,
        "generation": "sha256:revision",
        "file_exists": True,
    }
    replies = [
        permission_applied(
            "ReadPermissionSnapshot", {"kind": "compatibility", "payload": payload}
        ),
        permission_applied("ReadPermissionSnapshot", {"kind": "snapshot", **snapshot}),
        permission_applied("ReadPermissionInventory", snapshot),
    ]
    for reply in replies:
        message = decode_message(reply)
        encoded = encode_message(message)
        assert encoded == reply
        assert encoded["payload"]["result"]["payload"] is not payload
        encoded["payload"]["result"]["payload"]["profiles"].clear()
        assert encode_message(message) == reply

    with pytest.raises(ConsoleProtocolError, match="unexpected or missing fields"):
        decode_message(
            permission_applied("ReadPermissionInventory", {**snapshot, "raw": {}})
        )
    for invalid in (("not a JSON array",), object(), float("inf")):
        payload["legacy"] = invalid
        with pytest.raises(ConsoleProtocolError):
            decode_message(replies[0])


def test_permission_render_and_sessions_reject_repeated_tool_keys():
    row = {
        "server_key": "local:docs",
        "tool_name": "read",
        "state": "ask",
        "origin": "tool_override",
        "config_changed": True,
        "risk_floored": False,
    }
    other = {**row, "tool_name": "list", "config_changed": False}
    render = {
        "snapshot": {
            "payload": {},
            "generation": "missing:revision",
            "file_exists": False,
        },
        "effective_states": [row, other],
        "profile_identity": ["policy", None, "chain"],
    }
    raw = permission_applied("CapturePermissionRender", render)
    assert encode_message(decode_message(raw)) == raw
    render["profile_identity"] = None
    assert encode_message(decode_message(raw)) == raw
    render["effective_states"].append({**row, "state": "deny"})
    with pytest.raises(ConsoleProtocolError, match="repeated tool key"):
        decode_message(raw)

    pairs = [["local:docs", "list"], ["local:docs", "read"]]
    raw = permission_applied("ListSessionApprovals", pairs)
    assert encode_message(decode_message(raw)) == raw
    pairs.append(pairs[0])
    with pytest.raises(ConsoleProtocolError, match="repeated item"):
        decode_message(raw)


def test_permission_void_boolean_and_legacy_rule_results():
    for kind, result, invalid in [
        ("SetToolState", None, True),
        ("RevokeSessionApproval", False, None),
        ("ReadKillSwitch", True, 1),
    ]:
        raw = permission_applied(kind, result)
        assert encode_message(decode_message(raw)) == raw
        with pytest.raises(ConsoleProtocolError):
            decode_message(permission_applied(kind, invalid))

    rule = {
        "rule_id": '{"path":"docs"}',
        "args_json": '{"path":"docs"}',
        "created_at": {"legacy": [None, 3]},
        "profile_id": "default",
    }
    raw = permission_applied("ReadArgRules", [rule])
    assert encode_message(decode_message(raw)) == raw
    rule["rule_id"] = "another rule"
    with pytest.raises(ConsoleProtocolError, match="rule_id"):
        decode_message(raw)


def test_permission_import_presentation_cannot_carry_review_authority():
    rule = {
        "authority": "mcp",
        "server_key": "external:docs",
        "tool_name": "read",
        "state": "ask",
        "contract_sha256": "a" * 64,
    }
    review = {
        "display_name": "Docs",
        "producer": ["tldw", "1.0"],
        "destination_id": "docs",
        "expires_at": "2026-11-16T12:00:00.123456+00:00",
        "mappings": [
            {
                "source_server_key": "external:docs",
                "destination_server_key": "external:reading",
            }
        ],
        "matched": [
            {
                "source_rule": rule,
                "destination_identity": ["mcp", "external:reading", "read"],
                "destination_contract_sha256": "a" * 64,
                "destination_connected": False,
            }
        ],
        "changed": [],
        "missing": [],
        "pending_denies": [],
        "omitted_allow_ask": [],
        "fallbacks": [
            {"authority": "mcp", "server_key": "external:reading", "state": "deny"}
        ],
    }
    raw = permission_applied("InspectImport", review)
    assert encode_message(decode_message(raw)) == raw
    review["review"] = object()
    with pytest.raises(ConsoleProtocolError, match="expected JSON data"):
        decode_message(raw)
    review["review"] = {"archive_path": "local"}
    with pytest.raises(ConsoleProtocolError, match="unexpected or missing fields"):
        decode_message(raw)
    del review["review"]
    rule["authority"] = "unregistered"
    with pytest.raises(ConsoleProtocolError, match="unknown enum value"):
        decode_message(raw)


def test_permission_confirmation_references_the_same_request_without_guard_bytes():
    raw = permission_applied("ConfirmFirstBind", {"confirmation_request_id": "request"})
    assert encode_message(decode_message(raw)) == raw
    for kind, result in [
        ("ConfirmFirstBind", {"confirmation_request_id": "prior"}),
        (
            "ConfirmFirstBind",
            {"confirmation_request_id": "request", "guard_token": "local"},
        ),
        ("UnknownPermission", None),
    ]:
        with pytest.raises(ConsoleProtocolError):
            decode_message(permission_applied(kind, result))
