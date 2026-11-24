import asyncio
import ast
from pathlib import Path

import pytest

from Tests.Research_Workspace.test_chat_models import turn
from tldw_chatbook.Research_Workspace.chat_service import (
    ResearchChatService,
    collect_answer,
)
from tldw_chatbook.Research_Workspace.contracts import QualifiedWorkspaceRef


def test_research_chat_imports_no_agent_or_tool_runtime():
    research_package = (
        Path(__file__).resolve().parents[2] / "tldw_chatbook" / "Research_Workspace"
    )
    for name in (
        "chat_service.py",
        "local_chat.py",
        "chat_models.py",
        "egress_consent.py",
    ):
        tree = ast.parse((research_package / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not set((node.module or "").split(".")).intersection(
                    {
                        "Agents",
                        "Tools",
                        "MCP",
                        "ACP",
                        "console_runtime",
                        "console_approval",
                        "tool_catalog",
                    }
                ), (name, node.module)


@pytest.mark.asyncio
async def test_stop_targets_the_qualified_turn_and_closes_the_stream():
    started, closed = asyncio.Event(), asyncio.Event()
    request = turn()
    chunks = []

    async def stream():
        try:
            yield "Partial answer"
            started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def send(request, on_chunk, cancel):
        return await collect_answer(stream(), on_chunk, cancel)

    service = ResearchChatService(send)
    task = asyncio.create_task(service.send(request, chunks.append, asyncio.Event()))
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert not service.stop(
            QualifiedWorkspaceRef("local", "other"), request.turn_id
        )
        assert service.stop(request.workspace, request.turn_id)
        result = await asyncio.wait_for(task, 2)
        assert result.status == "cancelled"
        assert result.text == "Partial answer"
        assert chunks == ["Partial answer"]
        assert closed.is_set()
        assert not service.stop(request.workspace, request.turn_id)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
