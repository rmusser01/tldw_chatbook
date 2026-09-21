"""Server tool errors must survive the real stdio/client/control-plane path."""

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from Tests.private_profile import private_profile_test
from tldw_chatbook.MCP.client import MCPClient
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        {"text": "private-malformed-sentinel"},
        [None, "private-malformed-sentinel"],
        [{"type": "text", "text": {"secret": "private-malformed-sentinel"}}],
    ],
)
@private_profile_test
async def test_malformed_error_content_stays_failed_without_logging_body(
    request, content
):
    """An invalid server body stays a recoverable failure at the client boundary."""
    client = MCPClient()
    session = SimpleNamespace(
        call_tool=AsyncMock(return_value=SimpleNamespace(isError=True, content=content))
    )
    client.sessions["result-review"] = session
    logs = []
    sink = logger.add(lambda message: logs.append(str(message)))
    try:
        assert await client.call_tool("result-review", "review_echo", {}) == {
            "error": "MCP tool reported an error."
        }
        assert client.sessions["result-review"] is session
        assert "private-malformed-sentinel" not in "".join(logs)
    finally:
        logger.remove(sink)


@asynccontextmanager
async def connected_tool(
    tmp_path: Path,
) -> AsyncIterator[tuple[MCPClient, UnifiedMCPControlPlaneService, Path]]:
    """Connect the real client to a tool-result fixture in isolated storage.

    Args:
        tmp_path: Caller-owned directory for fixture state, trace, and store.

    Yields:
        Connected client, real control plane, and the tool-result state path.

    Raises:
        RuntimeError: The local fixture cannot be connected.
    """
    state = tmp_path / "result.json"
    state.write_text(json.dumps({"content": []}))
    store = LocalMCPStore(tmp_path / "store.json")
    store.save_profile(
        LocalExternalMCPProfile(
            profile_id="result-review",
            command=sys.executable,
            args=(
                str(Path(__file__).parent / "fixtures/stdio_tool_result_server.py"),
                str(state),
                str(tmp_path / "trace.jsonl"),
                str(tmp_path.resolve()),
            ),
        )
    )
    client = MCPClient()
    local = LocalMCPControlService(store=store, client=client, manifest_provider=dict)
    service = UnifiedMCPControlPlaneService(
        target_store=None,
        context_store=None,
        local_service=local,
        server_service=None,
    )
    try:
        await local.connect_profile("result-review")
        yield client, service, state
    finally:
        await client.disconnect_from_server("result-review")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ([{"type": "text", "text": "Fixture tool failed."}], "Fixture tool failed."),
        (
            [
                {"type": "text", "text": "First error."},
                {"type": "image", "data": "not-exposed", "mimeType": "image/png"},
                {"type": "text", "text": "Second error."},
            ],
            "First error.\nSecond error.",
        ),
        ([], "MCP tool reported an error."),
        (
            [{"type": "image", "data": "not-exposed", "mimeType": "image/png"}],
            "MCP tool reported an error.",
        ),
        ([{"type": "text", "text": "   "}], "MCP tool reported an error."),
    ],
)
@private_profile_test
async def test_stdio_tool_error_uses_text_only_failure_and_keeps_connection(
    request, tmp_path, content, expected
):
    async with connected_tool(tmp_path) as (client, _service, state):
        session = client.sessions["result-review"]
        state.write_text(json.dumps({"isError": True, "content": content}))
        assert await client.call_tool(
            "result-review", "review_echo", {"message": "fail"}
        ) == {"error": expected}
        assert client.sessions["result-review"] is session
        assert session.process.returncode is None
        state.write_text(
            json.dumps({"content": [{"type": "text", "text": "Recovered"}]})
        )
        assert await client.call_tool(
            "result-review", "review_echo", {"message": "retry"}
        ) == {"result": [{"type": "text", "text": "Recovered"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [None, False])
@private_profile_test
async def test_success_with_absent_or_false_error_flag_keeps_existing_shape(
    request, tmp_path, flag
):
    async with connected_tool(tmp_path) as (client, _service, state):
        content = [{"type": "text", "text": "Success"}]
        payload = {"content": content}
        if flag is not None:
            payload["isError"] = flag
        state.write_text(json.dumps(payload))
        assert await client.call_tool(
            "result-review", "review_echo", {"message": "ok"}
        ) == {"result": content}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flag", [1, 0, "true", "false", None, [], {}, {"secret": "private-flag-sentinel"}]
)
@private_profile_test
async def test_non_boolean_error_flag_is_bounded_failure_and_can_retry(
    request, tmp_path, flag
):
    """Malformed protocol flags cannot become success or expose validator inputs."""
    async with connected_tool(tmp_path) as (client, service, state):
        session = client.sessions["result-review"]
        state.write_text(
            json.dumps(
                {
                    "isError": flag,
                    "content": [{"type": "text", "text": "private-body-sentinel"}],
                }
            )
        )
        logs = []
        sink = logger.add(lambda message: logs.append(str(message)))
        try:
            with pytest.raises(RuntimeError, match="^Invalid MCP tool result$"):
                await service.execute_hub_tool(
                    "local:result-review",
                    "review_echo",
                    {"message": "fail"},
                    registered_argument_names={"message"},
                )
        finally:
            logger.remove(sink)
        record = service.execution_log.read_recent(1)[0]
        assert record["ok"] is False and record["status"] == "error"
        assert record["error_category"] == "execution_failed"
        for sentinel in ("private-body-sentinel", "private-flag-sentinel"):
            assert sentinel not in "".join(logs)
            assert sentinel not in service.execution_log.path.read_text()
        assert client.sessions["result-review"] is session
        assert session.process.returncode is None
        state.write_text(
            json.dumps({"content": [{"type": "text", "text": "Recovered"}]})
        )
        assert await service.execute_hub_tool(
            "local:result-review",
            "review_echo",
            {"message": "retry"},
            registered_argument_names={"message"},
        ) == {"result": [{"type": "text", "text": "Recovered"}]}
        assert service.execution_log.read_recent(1)[0]["ok"] is True


@pytest.mark.asyncio
@private_profile_test
async def test_control_plane_audits_tool_failure_without_error_body_and_recovers(
    request, tmp_path
):
    async with connected_tool(tmp_path) as (client, service, state):
        error_body = "Fixture failure api_key=private-test-sentinel"
        state.write_text(
            json.dumps(
                {"isError": True, "content": [{"type": "text", "text": error_body}]}
            )
        )
        logs = []
        sink = logger.add(lambda message: logs.append(str(message)))
        try:
            with pytest.raises(RuntimeError, match="Fixture failure"):
                await service.execute_hub_tool(
                    "local:result-review",
                    "review_echo",
                    {"message": "fail"},
                    registered_argument_names={"message"},
                )
        finally:
            logger.remove(sink)
        record = service.execution_log.read_recent(1)[0]
        assert record["ok"] is False and record["status"] == "error"
        assert record["error_category"] == "execution_failed"
        assert "private-test-sentinel" not in service.execution_log.path.read_text()
        assert "private-test-sentinel" not in "".join(logs)
        assert "result-review" in client.sessions
        state.write_text(
            json.dumps(
                {"isError": False, "content": [{"type": "text", "text": "Recovered"}]}
            )
        )
        assert await service.execute_hub_tool(
            "local:result-review",
            "review_echo",
            {"message": "retry"},
            registered_argument_names={"message"},
        ) == {"result": [{"type": "text", "text": "Recovered"}]}
        record = service.execution_log.read_recent(1)[0]
        assert record["ok"] is True and record["status"] == "success"
