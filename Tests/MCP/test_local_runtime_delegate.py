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
    RawToolCallRefusedError,
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
async def test_request_tools_call_raises_the_typed_error():
    """Item 2 (PR-T3 fix round D): drift-proofing at the RAISE SITE, same
    precedent as `test_protocol_diagnostics_reports_tools_call_as_
    unsupported` just below. `UI/MCP_Modules/mcp_inspector.py`'s Advanced
    runner narrows its own classifier to this TYPE -- if this raise site
    ever reverted to a bare `PermissionError`, that narrowed handler would
    silently stop recognizing this refusal, and nothing else in this suite
    would fail to say so."""
    delegate = _delegate()
    _stub_execute_tool(delegate)

    with pytest.raises(RawToolCallRefusedError):
        await delegate.request("tools/call", {"name": "calculator"})


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


def test_protocol_diagnostics_reports_tools_call_as_unsupported():
    """Fix Round C (PR-T3 review), Item 1: the capability surface must not
    contradict the enforcement above. `request()`'s `tools/call` branch
    refuses unconditionally, so `get_protocol_diagnostics()` -- the surface
    an agent reads via `run_action("runtime.protocol.inspect")` before
    planning a `runtime.request` call -- must report it as unsupported
    rather than as an ordinary, freely-callable method. This asserts
    against the REAL delegate, not a hand-written double, so a future
    change re-flipping the flag (or removing the entry) is caught here even
    if a fake elsewhere drifts."""
    delegate = _delegate()

    diagnostics = delegate.get_protocol_diagnostics()

    methods_by_name = {entry["name"]: entry["supported"] for entry in diagnostics["methods"]}
    assert methods_by_name["tools/call"] is False
    # Every other advertised method stays truthfully supported -- the
    # refusal is scoped to `tools/call` alone, not a blanket downgrade.
    for name, supported in methods_by_name.items():
        if name == "tools/call":
            continue
        assert supported is True, name


def test_protocol_capabilities_flags_tools_call_as_unavailable():
    """Item 4 (PR-T3 fix round D), closing the scope Fix Round C left open.
    `get_protocol_capabilities()`'s `request_methods` still lists
    `tools/call` (Fix Round C's own reasoning: `request()` genuinely
    recognizes the method by name, so removing it would itself be
    inaccurate) -- but an agent reading ONLY this method, never
    cross-referencing `get_protocol_diagnostics()`, used to have no way to
    learn it would be refused. `unavailable_request_methods` closes that:
    asserted against the REAL delegate, not a hand-written double, so a
    future change re-flipping the flag (or removing the entry) is caught
    here even if a fake elsewhere drifts."""
    delegate = _delegate()

    capabilities = delegate.get_protocol_capabilities()

    assert "tools/call" in capabilities["request_methods"]
    assert capabilities["unavailable_request_methods"] == ["tools/call"]
    # Every other recognized method is absent from the unavailable list --
    # the refusal is scoped to `tools/call` alone.
    for method in capabilities["request_methods"]:
        if method == "tools/call":
            continue
        assert method not in capabilities["unavailable_request_methods"]
