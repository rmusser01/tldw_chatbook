"""MCP progress is optional, per active token, and never changes final results."""

import asyncio

import pytest

from Tests.MCP.test_client_catalog_pagination import _bare_connection
from tldw_chatbook.Agents.tool_output import tool_output_scope


@pytest.mark.asyncio
async def test_progress_is_scoped_to_active_call_and_final_only_requests_stay_identical():
    from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import ProducerLifetime

    connection = _bare_connection()
    connection._producer_lifetime = ProducerLifetime()
    seen = []
    requests = []
    release = asyncio.Event()

    async def request(method, params):
        requests.append(params)
        if "_meta" in params:
            token = params["_meta"]["progressToken"]

            async def notify(**values):
                await connection._handle_incoming_payload(
                    {
                        "method": "notifications/progress",
                        "params": {"progressToken": token, **values},
                    }
                )

            await notify(progress=1, total=2, message="first")
            await notify(progress=0, message="backwards")
            await notify(progress=True, message="boolean")
            await notify(progress=float("nan"), message="nonfinite")
            await connection._handle_incoming_payload(
                {
                    "method": "notifications/progress",
                    "params": {
                        "progressToken": [],
                        "progress": 2,
                        "message": "malformed",
                    },
                }
            )
            await release.wait()
        return {"content": [{"type": "text", "text": "final"}], "isError": False}

    connection.request = request
    with tool_output_scope(seen.append):
        task = asyncio.create_task(connection.call_tool("echo", {}))
        await asyncio.sleep(0.15)
        assert seen and "first" in seen[-1]
        assert not task.done()
        release.set()
        result = await task
    assert result.content[0]["text"] == "final"
    assert "backwards" not in str(seen) and "nonfinite" not in str(seen)
    count = len(seen)
    token = requests[0]["_meta"]["progressToken"]
    await connection._handle_incoming_payload(
        {
            "method": "notifications/progress",
            "params": {"progressToken": token, "progress": 3, "message": "late"},
        }
    )
    assert len(seen) == count
    await connection.call_tool("echo", {})
    assert requests[-1] == {"name": "echo", "arguments": {}}


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_real_stdio_concurrent_calls_receive_only_their_own_progress(tmp_path):
    import sys
    from pathlib import Path

    from tldw_chatbook.MCP.client import _StdioJSONRPCConnection

    fixture = Path(__file__).parent / "fixtures/stdio_progress_server.py"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        str(fixture),
        str(tmp_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    connection = _StdioJSONRPCConnection(
        process, client_name="private-progress", request_timeout_seconds=5
    )
    first, second = [], []
    ready = [asyncio.Event(), asyncio.Event()]
    loop = asyncio.get_running_loop()

    async def run(name, output, event):
        def observe(text):
            output.append(text)
            loop.call_soon_threadsafe(event.set)

        with tool_output_scope(observe):
            return await connection.call_tool(name, {})

    calls = [
        asyncio.create_task(run("first", first, ready[0])),
        asyncio.create_task(run("second", second, ready[1])),
    ]
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), 3)
        assert all(not call.done() for call in calls)
        assert "second" not in str(first) and "first" not in str(second)
        (tmp_path / "release").touch()
        results = await asyncio.gather(*calls)
        assert [r.content[0]["text"] for r in results] == [
            "first final",
            "second final",
        ]
        await asyncio.sleep(0.15)
        assert "late" not in str(first + second)
    finally:
        (tmp_path / "release").touch()
        await connection.close()
        await asyncio.gather(*calls, return_exceptions=True)
