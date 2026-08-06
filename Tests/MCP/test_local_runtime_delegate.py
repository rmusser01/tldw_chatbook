"""Tests for `LocalMCPRuntimeDelegate`'s own `tools/call` refusal (Fix Round A,
PR-T3 review, Item 2).

Task 6 (PR-T3) refused a raw `tools/call` in
`UnifiedMCPControlPlaneService.run_action()` (`_refuse_raw_tool_call`), one
layer above this delegate. But `LocalMCPRuntimeDelegate.request()`'s
`tools/call` branch calls `self.execute_tool()` directly -- no gate, no log
-- and any caller that reaches `request()`/`batch()` WITHOUT going through
`run_action()` (a direct call, or `run_runtime_batch()`'s own per-item loop,
which calls `runtime_delegate.request()` and is reachable off
`app.local_mcp_control_service`, a public attribute) bypassed Task 6's
refusal entirely.

This is the durable backstop: the SAME refusal, enforced one layer lower, so
every caller passes it regardless of what dispatched to `request()`/
`batch()`. It does not replace the control-plane pre-dispatch scan in
`run_action()` -- that scan is what gives `runtime.batch` its all-or-nothing
property (checked before ANY item dispatches, since the batch runs serially
and a per-item refusal at the delegate would only stop the offending item
itself, not prevent items before it from having already run). Both layers
share `RAW_TOOL_CALL_REFUSED_MESSAGE` so the copy cannot drift between them.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.MCP.local_runtime_delegate import (
    RAW_TOOL_CALL_REFUSED_MESSAGE,
    LocalMCPRuntimeDelegate,
)


def _delegate() -> LocalMCPRuntimeDelegate:
    return LocalMCPRuntimeDelegate(manifest_provider=lambda: {})


def _stub_execute_tool(delegate: LocalMCPRuntimeDelegate) -> list[tuple[str, dict]]:
    """Replaces `execute_tool` with a recording stub so a test can assert it
    was never reached, without exercising any real tool's business logic."""
    executed: list[tuple[str, dict]] = []

    async def _fake_execute_tool(tool_name, arguments=None):
        executed.append((tool_name, dict(arguments or {})))
        return {"ok": True}

    delegate.execute_tool = _fake_execute_tool  # type: ignore[method-assign]
    return executed


@pytest.mark.asyncio
async def test_request_tools_call_is_refused_without_executing():
    """A direct caller of `request()` -- bypassing
    `UnifiedMCPControlPlaneService.run_action()`'s pre-dispatch scan
    entirely -- must still be refused."""
    delegate = _delegate()
    executed = _stub_execute_tool(delegate)

    with pytest.raises(PermissionError, match="Execute Local Tool"):
        await delegate.request(
            "tools/call", {"name": "calculator", "arguments": {"x": 1}}
        )

    assert executed == []


@pytest.mark.asyncio
async def test_request_tools_call_refusal_message_matches_shared_constant():
    delegate = _delegate()
    _stub_execute_tool(delegate)

    with pytest.raises(PermissionError) as exc_info:
        await delegate.request("tools/call", {"name": "calculator"})

    assert str(exc_info.value) == RAW_TOOL_CALL_REFUSED_MESSAGE


@pytest.mark.asyncio
async def test_batch_containing_tools_call_does_not_execute_it():
    """`batch()` loops `request()` -- it is a third mouth on the same seam
    (no in-tree caller today, but exactly what a future refactor could wire
    up). A `tools/call` item inside a batch must not execute, even though
    `batch()` catches per-item errors rather than propagating them -- the
    refusal shows up as a failed item, not a silent skip."""
    delegate = _delegate()
    executed = _stub_execute_tool(delegate)

    results = await delegate.batch(
        [
            {"method": "tools/list"},
            {"method": "tools/call", "params": {"name": "calculator"}},
        ]
    )

    assert executed == []
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False
    assert RAW_TOOL_CALL_REFUSED_MESSAGE in results[1]["error"]


@pytest.mark.asyncio
async def test_other_request_methods_are_still_dispatched():
    """The refusal is scoped to `tools/call` -- every other protocol method
    on the delegate is untouched."""
    delegate = _delegate()
    _stub_execute_tool(delegate)

    result = await delegate.request("tools/list", {})

    assert result == {"tools": []}
