"""Console transcript coverage for the consecutive-denial terminal outcome."""

import pytest

from Tests.Agents.test_denial_circuit_breaker import run_batch
from Tests.console_provider_doubles import provider_resolution
from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


class _QueuedGateway:
    async def resolve_for_send(self, _selection):
        return provider_resolution(ready=True, provider="llama_cpp", visible_copy="")

    async def stream_chat(self, _resolution, _messages, **_kwargs):
        yield "retry accepted"


def _denial_outcome():
    denied = ToolResult.blocked(
        "PRIVATE denial payload must not reach terminal copy",
        approval_decision="denied",
    )
    outcome, invoked, records, model_calls = run_batch([denied] * 3)
    assert invoked == ["c0", "c1", "c2"]
    assert len(model_calls) == 1
    assert len([record for record in records if record[0] == "error"]) == 1
    return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_placeholder", [False, True])
async def test_denial_terminal_preserves_partial_and_allows_next_submit(
    missing_placeholder, tmp_path
):
    store = ConsoleChatStore()
    gateway = _QueuedGateway()
    db = AgentRunsDB(tmp_path / "runs.db", client_id="denial-terminal")
    try:
        bridge = ConsoleAgentBridge(
            agent_runs_db=db,
            store=store,
            provider_gateway=gateway,
        )
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            provider="llama_cpp",
            model="test-model",
            agent_bridge=bridge,
            agent_runtime_enabled=True,
        )
        session = store.create_session(title="denial", ephemeral=True)
        partial = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
        )
        store.append_stream_chunk(partial.id, "partial plan")
        placeholder_id = partial.id
        if missing_placeholder:
            placeholder_id = "missing-placeholder"

        outcome = _denial_outcome()
        result = controller._finalize_agent_failure(
            placeholder_id,
            session.id,
            outcome,
            variant_mode=False,
        )

        assert result.accepted is True and result.should_clear_draft is True
        assert controller.run_state_for(session.id).status is ConsoleRunStatus.FAILED
        rows = store.messages_for_session(session.id)
        system_rows = [r for r in rows if r.role is ConsoleMessageRole.SYSTEM]
        assert (
            len(
                [
                    r
                    for r in system_rows
                    if "3 consecutive tool calls were denied" in r.content
                ]
            )
            == 1
        )
        assistants = [r for r in rows if r.role is ConsoleMessageRole.ASSISTANT]
        assert any(
            r.content == "partial plan" and r.status == "failed" for r in assistants
        )
        assert controller._agent_failure_visible_copy(outcome).startswith(
            "Agent stopped:"
        )
        assert outcome.denial_count == 3
        assert "PRIVATE denial payload" not in "\n".join(r.content for r in rows)

        retry = await controller.submit_draft("try again", session_id=session.id)
        assert retry.accepted is True and retry.should_clear_draft is True
        assert retry.visible_copy == "retry accepted"
        assert retry.terminal_status is ConsoleRunStatus.COMPLETED
    finally:
        db.close()
