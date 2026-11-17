"""Approval display values preserve captured details without carrying grants."""

from copy import deepcopy
from dataclasses import replace
import io
import json
from types import MappingProxyType

import pytest

from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat.approval_presentation import (
    ApprovalArgumentsTooDeep,
    ApprovalBatchView,
    _check_argument_depth,
    capture_approval_view,
    profile_authority,
    scope_copy,
)
from tldw_chatbook.Chat.console_execution_approval_values import (
    decode_approval_view,
    encode_approval_view,
)
from tldw_chatbook.Chat.console_execution_protocol import ConsoleProtocolError
from tldw_chatbook.MCP.permission_process import read_message, write_message
from tldw_chatbook.UI.Console_Modules.approval_details import iter_redacted_details


def _view():
    owner = profile_authority("mcp", "default", "Captured binding", "shared_group")
    first = MCPPendingCall(
        llm_name="read_file",
        server_key="external:docs",
        tool_name="read_file",
        server_label="Documentation",
        arguments={"path": "resolved-first.txt"},
        captured_arguments={
            "path": "requested-first.txt",
            "argv": ("--mode", "plain"),
            "nested": {"values": [None, True, 2, 1.5, "é\nretained"]},
            "api_key": "synthetic-captured-secret",
        },
        reason="ask",
        call_id="shared",
        presentation_authority=owner,
    )
    second = replace(
        first,
        arguments={"path": "resolved-second.txt"},
        captured_arguments={"path": "requested-second.txt"},
    )
    unknown = replace(
        first,
        call_id="unknown",
        arguments={"path": "unknown.txt"},
        captured_arguments=None,
        presentation_authority=None,
    )
    return capture_approval_view(
        (first, second, unknown),
        round_id="round",
        session_id="session",
        run_id="run",
        revision=3,
    )


def _with_arguments(view, arguments):
    row = replace(
        view.rows[0],
        call_count=1,
        targets=(view.rows[0].targets[0],),
        argument_sets=(MappingProxyType(arguments),),
    )
    return replace(view, rows=(row,), call_count=1)


def test_original_captured_view_round_trips_order_scope_and_complete_details():
    original = _view()
    encoded = encode_approval_view(original)
    assert type(encoded["rows"][0]["argument_sets"][0]) is str
    restored = decode_approval_view(encoded)
    assert type(restored) is ApprovalBatchView
    assert restored == original
    assert encode_approval_view(restored) == encoded
    assert [row.verdict_key for row in restored.rows] == ["shared", "unknown"]
    row = restored.rows[0]
    assert row.targets == ("resolved-first.txt", "resolved-second.txt")
    assert [item["path"] for item in row.argument_sets] == [
        "requested-first.txt",
        "requested-second.txt",
    ]
    assert list(row.argument_sets[0]) == ["path", "argv", "nested", "api_key"]
    assert row.argument_sets[0]["argv"] == ["--mode", "plain"]
    assert row.argument_sets[0]["api_key"] == "synthetic-captured-secret"
    assert restored.rows[1].authority.profile_id is None
    assert restored.rows[1].legal_decisions == ("approve_once", "deny")
    for decision in row.legal_decisions:
        assert scope_copy(row, decision) == scope_copy(original.rows[0], decision)
    old_details = "".join(
        page.text for page in iter_redacted_details(original.rows[0].argument_sets)
    )
    new_details = "".join(
        page.text for page in iter_redacted_details(row.argument_sets)
    )
    assert new_details == old_details
    assert "synthetic-captured-secret" not in new_details


def test_both_directions_detach_nested_values_and_restore_original_mapping_shape():
    original = _view()
    encoded = encode_approval_view(original)
    original.rows[0].argument_sets[0]["nested"]["values"].append("later")
    first_arguments = json.loads(encoded["rows"][0]["argument_sets"][0])
    assert first_arguments["nested"]["values"][-1] == ("é\nretained")
    restored = decode_approval_view(encoded)
    separate = decode_approval_view(encoded)
    encoded["rows"][0]["argument_sets"][0] = "{}"
    assert restored.rows[0].argument_sets[0]["nested"]["values"]
    restored.rows[0].argument_sets[0]["nested"]["values"].clear()
    assert separate.rows[0].argument_sets[0]["nested"]["values"]
    assert type(restored.rows[0].argument_sets[0]) is MappingProxyType
    with pytest.raises(TypeError):
        restored.rows[0].argument_sets[0]["path"] = "changed"


def test_unknown_missing_or_opaque_record_fields_refuse():
    encoded = encode_approval_view(_view())
    unknown = deepcopy(encoded)
    unknown["rows"][0]["authority"]["permission_store"] = {}
    missing = deepcopy(encoded)
    del missing["rows"][0]["argument_sets"]
    for value in (unknown, missing, object()):
        with pytest.raises(ConsoleProtocolError):
            decode_approval_view(value)

    class DerivedView(ApprovalBatchView):
        pass

    original = _view()
    derived = DerivedView(
        original.round_id,
        original.session_id,
        original.run_id,
        original.revision,
        original.rows,
        original.call_count,
        original.bulk_once,
        original.bulk_deny,
    )
    with pytest.raises(ConsoleProtocolError):
        encode_approval_view(derived)


def test_argument_json_refuses_cycles_opaque_values_invalid_keys_and_nonfinite():
    cycle = []
    cycle.append(cycle)
    for invalid in (
        {"value": cycle},
        {"value": object()},
        {"value": float("nan")},
        {"value": float("inf")},
        {1: "invalid key"},
    ):
        with pytest.raises(ConsoleProtocolError):
            encode_approval_view(_with_arguments(_view(), invalid))
    for invalid in (
        '{"value":NaN}',
        '{"value":1e999}',
        '{"value":Infinity}',
        '{"same":1,"same":2}',
        '{"value":',
        "[]",
        {"value": "not embedded JSON"},
    ):
        encoded = encode_approval_view(_view())
        encoded["rows"][0]["argument_sets"] = [invalid]
        with pytest.raises(ConsoleProtocolError):
            decode_approval_view(encoded)


def test_counts_booleans_and_choice_vocabulary_remain_exact():
    original = _view()
    for invalid in (
        replace(original, revision=True),
        replace(original, bulk_once=1),
        replace(
            original,
            rows=(replace(original.rows[0], legal_decisions=("invented",)),),
        ),
    ):
        with pytest.raises(ConsoleProtocolError):
            encode_approval_view(invalid)


def test_argument_depth_matches_original_leaf_counting_boundary():
    arguments = 0
    for _ in range(127):
        arguments = {"nested": arguments}
    _check_argument_depth(arguments)
    encoded = encode_approval_view(_with_arguments(_view(), arguments))
    stream = io.BytesIO()
    write_message(stream, {"approval_view": encoded})
    stream.seek(0)
    restored = decode_approval_view(read_message(stream)["approval_view"])
    _check_argument_depth(restored.rows[0].argument_sets[0])
    too_deep = {"nested": arguments}
    with pytest.raises(ApprovalArgumentsTooDeep):
        _check_argument_depth(too_deep)
    with pytest.raises(ConsoleProtocolError):
        encode_approval_view(_with_arguments(_view(), too_deep))
    encoded["rows"][0]["argument_sets"] = [json.dumps(too_deep)]
    with pytest.raises(ConsoleProtocolError):
        decode_approval_view(encoded)
