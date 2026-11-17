"""Fixed pending-head values keep presentation separate from original owners."""

from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path

import pytest

from tldw_chatbook.Agents.hook_permissions import HookReviewSnapshot
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Agents.tool_catalog import ToolExecutionPolicy
from tldw_chatbook.Chat.approval_presentation import (
    _build_approval_payload,
    _check_argument_depth,
    profile_authority,
)
from tldw_chatbook.Chat.console_chat_controller import ConsolePendingDecisionProjection
from tldw_chatbook.Chat.console_execution_decision_values import (
    decode_pending_decision,
    encode_pending_decision,
)
from tldw_chatbook.Chat.console_execution_protocol import ConsoleProtocolError
from tldw_chatbook.config import HookConfigSnapshot
from tldw_chatbook.MCP.permission_process import read_message, write_message


def _approval(arguments=None):
    call = MCPPendingCall(
        llm_name="read_file",
        server_key="local:docs",
        tool_name="read_file",
        server_label="Local tools",
        arguments=(
            {"path": "resolved.txt", "api_key": "synthetic-original"}
            if arguments is None
            else arguments
        ),
        captured_arguments=(
            {"path": "requested.txt", "details": ["kept", "in order"]}
            if arguments is None
            else arguments
        ),
        reason="ask",
        options=("approve_once", "deny"),
        call_id="call",
        effects=("reads_local",),
        full_command="original command",
        warning="Original warning",
        scope_notice="Original scope",
        rationale="Original rationale",
        description="Original description",
        presentation_authority=profile_authority(
            "local", "default", "Bound folder", "call"
        ),
    )
    payload = _build_approval_payload(
        "round",
        "session",
        "run",
        [call],
        0.0,
        None,
        read_global_ToolExecutionPolicy=lambda: ToolExecutionPolicy,
    )
    payload.update(_decision_type="approval", _decision_id="round", _decision_order=1)
    payload["presentation_state"]["local-owner"] = object()
    return ConsolePendingDecisionProjection(
        "approval", "session", "round", None, payload
    )


def test_actual_approval_preserves_calls_details_and_finishing_omissions():
    original = _approval()
    encoded = encode_pending_decision(original)
    assert type(encoded["payload"]["calls"][0]["arguments"]) is str
    restored = decode_pending_decision(encoded)
    assert type(restored) is ConsolePendingDecisionProjection
    payload = restored.payload
    assert payload["calls"][0]["arguments"] == {
        "path": "resolved.txt",
        "api_key": "synthetic-original",
    }
    assert payload["view"].rows[0].argument_sets[0]["path"] == "requested.txt"
    assert payload["calls"][0]["rationale"] == "Original rationale"
    assert payload["calls"][0]["effects"] == ["reads_local"]
    assert payload["summary"] is None
    assert "phase" not in payload
    assert "deadline_monotonic" not in payload
    assert "presentation_state" not in payload
    assert not any(key.startswith("_decision_") for key in payload)
    encoded["payload"]["calls"][0]["arguments"] = "{}"
    assert payload["calls"][0]["arguments"]["path"] == "resolved.txt"

    finishing = dict(original.payload)
    finishing.pop("view")
    finishing.pop("presentation_revision")
    finishing["phase"] = "finishing"
    completed_head = decode_pending_decision(
        encode_pending_decision(replace(original, payload=finishing))
    )
    assert completed_head.payload["phase"] == "finishing"
    assert "view" not in completed_head.payload
    assert "presentation_revision" not in completed_head.payload


def test_skill_heads_preserve_argv_and_absent_run_identity():
    script = {
        "skill_name": "original-skill",
        "script_path": "scripts/example.py",
        "mechanism": "interpreter",
        "interpreter": "python",
        "is_binary": False,
        "args": ["a b", "a", "b", "line\nbreak", ""],
        "request_id": "request",
        "session_id": "session",
        "timeout_seconds": 0.0,
        "deadline_monotonic": None,
    }
    restored = decode_pending_decision(
        encode_pending_decision(
            ConsolePendingDecisionProjection(
                "skill_script", "session", "request", None, script
            )
        )
    )
    assert restored.payload["args"] == script["args"]
    assert "run_id" not in restored.payload
    script["args"].clear()
    assert restored.payload["args"] == ["a b", "a", "b", "line\nbreak", ""]
    install = {
        "url": "https://example.test/original-skill",
        "request_id": "request",
        "session_id": "session",
        "run_id": "",
        "timeout_seconds": 0.0,
    }
    installed_head = decode_pending_decision(
        encode_pending_decision(
            ConsolePendingDecisionProjection(
                "skill_install", "session", "request", 0.0, install
            )
        )
    )
    assert installed_head.payload == install


def test_absence_zero_and_unlimited_timing_remain_distinct_without_new_clock():
    assert encode_pending_decision(None) is None
    assert decode_pending_decision(None) is None
    original = _approval()
    for remaining in (None, 0.0, 2.5):
        payload = dict(original.payload, timeout_seconds=remaining or 0.0)
        restored = decode_pending_decision(
            encode_pending_decision(
                replace(original, remaining_active_seconds=remaining, payload=payload)
            )
        )
        assert restored.remaining_active_seconds == remaining
        assert restored.payload["timeout_seconds"] == (remaining or 0.0)
        assert "deadline_monotonic" not in restored.payload


def test_hook_head_excludes_actual_snapshot_and_only_carries_routing_facts():
    config = HookConfigSnapshot(
        Path("private-config.toml"), True, {"private": object()}, "local-stamp"
    )
    snapshot = HookReviewSnapshot(
        config, (), Path("private-permissions.json"), ("local", 1), "review"
    )
    original = ConsolePendingDecisionProjection(
        "hook_review",
        "session",
        "review",
        None,
        {
            "session_id": "session",
            "generation": 2,
            "snapshot": snapshot,
            "waiting_for_send": True,
            "busy": False,
            "timeout_seconds": 0.0,
            "_decision_type": "hook_review",
            "_decision_id": "review",
            "_decision_order": 2,
        },
    )
    restored = decode_pending_decision(encode_pending_decision(original))
    assert restored.payload == {
        "session_id": "session",
        "generation": 2,
        "waiting_for_send": True,
        "busy": False,
        "timeout_seconds": 0.0,
    }
    assert "snapshot" not in restored.payload
    with pytest.raises(ConsoleProtocolError):
        encode_pending_decision(
            replace(original, payload=dict(original.payload, snapshot=object()))
        )


def test_unknown_opaque_mismatched_or_duplicate_embedded_values_refuse():
    original = _approval()
    for invalid in (
        replace(original, decision_type="question"),
        replace(original, remaining_active_seconds=True),
        replace(original, payload=dict(original.payload, arbitrary=object())),
        replace(original, payload=dict(original.payload, round_id="other")),
    ):
        with pytest.raises(ConsoleProtocolError):
            encode_pending_decision(invalid)
    for arguments in ('{"key":1,"key":2}', '{"value":NaN}', "{} trailing"):
        encoded = encode_pending_decision(original)
        encoded["payload"]["calls"][0]["arguments"] = arguments
        with pytest.raises(ConsoleProtocolError):
            decode_pending_decision(encoded)
    encoded = encode_pending_decision(original)
    encoded["payload"]["presentation_state"] = {}
    with pytest.raises(ConsoleProtocolError):
        decode_pending_decision(encoded)
    malformed = deepcopy(encode_pending_decision(original))
    del malformed["payload"]["calls"][0]["path_precheck_failed"]
    with pytest.raises(ConsoleProtocolError):
        decode_pending_decision(malformed)


def test_original_argument_depth_survives_frame_wrappers_and_remains_bounded():
    nested = 0
    for _ in range(126):
        nested = {"nested": nested}
    arguments = {"path": "file.txt", "nested": nested}
    _check_argument_depth(arguments)
    encoded = encode_pending_decision(_approval(arguments))
    stream = io.BytesIO()
    write_message(stream, {"projection": {"decision": encoded}})
    stream.seek(0)
    restored = decode_pending_decision(read_message(stream)["projection"]["decision"])
    assert restored.payload["calls"][0]["arguments"] == arguments
    _check_argument_depth(restored.payload["view"].rows[0].argument_sets[0])
    encoded["payload"]["calls"][0]["arguments"] = json.dumps({"extra": arguments})
    with pytest.raises(ConsoleProtocolError):
        decode_pending_decision(encoded)
