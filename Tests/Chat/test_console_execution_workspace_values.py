"""Workspace facts stay finite and preserve original display domains."""

import json

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker
from tldw_chatbook.Chat.console_execution_protocol import ConsoleProtocolError
from tldw_chatbook.Chat.console_execution_workspace_values import (
    ConsoleWorkspaceSessionFacts,
    ConsoleWorkspaceSurface,
    ConsoleWorkspaceCount,
    ConsoleWorkspaceReceipt,
    decode_workspace_surface,
    encode_workspace_surface,
)
from tldw_chatbook.Workspaces.conversation_attention import ConversationAttentionFact


def test_workspace_surface_closed_roundtrip_preserves_empty_zero_and_none():
    original = ConsoleWorkspaceSurface(
        None,
        False,
        (
            ConsoleWorkspaceSessionFacts(
                "one", "", 0, "", None, 0, ConsoleRunMarker.NONE
            ),
            ConsoleWorkspaceSessionFacts(
                "two",
                "2026-11-17T01:02:03Z",
                None,
                None,
                "",
                2,
                ConsoleRunMarker.NEEDS_APPROVAL,
                (ConversationAttentionFact("approval", "Approval required"),),
            ),
        ),
        progress_counts=(ConsoleWorkspaceCount("two", 2),),
        subagent_counts=(ConsoleWorkspaceCount("saved", 3),),
        subagent_count_ids=("saved",),
        unseen_receipts=(ConsoleWorkspaceReceipt("saved", "two", "failed"),),
    )
    wire = encode_workspace_surface(original)
    decoded = decode_workspace_surface(json.loads(json.dumps(wire, allow_nan=False)))
    assert decoded == original
    assert type(decoded.sessions[0]) is ConsoleWorkspaceSessionFacts
    assert decoded.sessions[1].run_marker is ConsoleRunMarker.NEEDS_APPROVAL
    assert decoded.sessions[1].attention[0].kind == "approval"
    assert decoded.subagent_counts[0].count == 3
    assert decoded.unseen_receipts[0].status == "failed"
    wire["sessions"][0]["character_name"] = "changed wire"
    assert decoded.sessions[0].character_name == ""
    with pytest.raises(ConsoleProtocolError):
        decode_workspace_surface({**wire, "store": object()})
    wire["sessions"][0]["character_id"] = True
    with pytest.raises(ConsoleProtocolError):
        decode_workspace_surface(wire)
    wire["sessions"][0]["character_id"] = 0
    wire["sessions"][1]["run_marker"] = "unknown"
    with pytest.raises(ConsoleProtocolError):
        decode_workspace_surface(wire)
