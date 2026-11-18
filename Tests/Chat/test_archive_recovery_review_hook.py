"""Archive approvals still use the extracted current-dev interrupt owner."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Agents.test_archive_recovery_tools import admitted
from Tests.Chat.test_archive_recovery_operations import (
    recovery_stores as recovery_stores,  # noqa: PLC0414 - shared SQLite fixture
)
from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor
from tldw_chatbook.Chat.archive_recovery.runtime import ArchiveRecoveryRuntime
from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision,expected",
    [
        ("approve_once", "proceed"),
        ("deny", "Recovery search was not approved."),
        ("approve_session", "Recovery search was not approved."),
    ],
)
async def test_extracted_review_keeps_archive_exact_once_approval(
    recovery_stores, decision, expected
):
    app, _, _ = recovery_stores
    runtime = ArchiveRecoveryRuntime(app)
    runtime.is_answerable = lambda _: True
    provider = admitted(runtime)
    assert await asyncio.to_thread(provider.activate, "primary-run")
    pending = []

    def approve(rows):
        pending.extend(rows)
        return {row.call_id: decision for row in rows}

    hook = build_tool_review_hook(
        SimpleNamespace(begin_turn=lambda _: None),
        SimpleNamespace(tool_for=lambda _: None),
        None,
        approve,
        recovery_provider_reader=lambda: provider,
        recovery_persona_rules=(
            {
                "rule_kind": "mcp_tool",
                "rule_name": "search_archived_conversations",
                "allowed": True,
                "require_confirmation": True,
            },
        ),
    )
    try:
        with use_run_actor(CurrentRunActor("primary", "primary-run", None)):
            verdicts = await asyncio.to_thread(
                hook,
                [ToolCall("search_archived_conversations", {}, "search-call")],
                "primary-run",
            )
        assert len(pending) == 1
        assert pending[0].server_key == "agent:archive_recovery"
        assert pending[0].options == ("approve_once", "deny")
        assert verdicts["search-call"] == expected
    finally:
        await runtime.shutdown()
