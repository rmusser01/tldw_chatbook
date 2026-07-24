from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.Tools.tool_executor import Tool, ToolExecutor


PRIVATE_SENTINEL = "TOOL-PRIVATE-SENTINEL-sk-not-a-real-key"


class _EchoTool(Tool):
    calls = 0

    @property
    def name(self) -> str:
        return "private_echo"

    @property
    def description(self) -> str:
        return "Return a private value for contract testing."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "delay": {"type": "number"},
                "fail": {"type": "boolean"},
            },
        }

    async def execute(self, **kwargs):
        type(self).calls += 1
        if kwargs.get("delay"):
            await asyncio.sleep(kwargs["delay"])
        if kwargs.get("fail"):
            raise RuntimeError(PRIVATE_SENTINEL)
        return {"private_result": PRIVATE_SENTINEL}


def _call(arguments, call_id: str = "private-call-id") -> dict:
    return {
        "id": call_id,
        "function": {
            "name": "private_echo",
            "arguments": arguments,
        },
    }


@pytest.mark.asyncio
async def test_history_is_metadata_only_while_immediate_result_is_unchanged() -> None:
    executor = ToolExecutor()
    executor.register_tool(_EchoTool())
    try:
        result = await executor.execute_tool_call(
            _call({"query": PRIVATE_SENTINEL, PRIVATE_SENTINEL: "unknown-value"})
        )
        history = executor.get_execution_history()
    finally:
        executor.executor.shutdown(wait=False)

    assert result["result"] == {"private_result": PRIVATE_SENTINEL}
    assert len(history) == 1
    record = history[0]
    assert record["tool_name"] == "private_echo"
    assert record["status"] == "success"
    assert record["started_at"].endswith("+00:00")
    assert record["argument_names"] == ["query"]
    assert record["unknown_argument_count"] == 1
    assert record["result_type"] == "dict"
    assert record["result_size"] > 0
    assert PRIVATE_SENTINEL not in repr(history)
    assert "result" not in record
    assert "error" not in record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "timeout_seconds", "expected_status"),
    [
        ("{not-json " + PRIVATE_SENTINEL, 30, "parse_error"),
        ({"query": PRIVATE_SENTINEL, "delay": 0.05}, 0.001, "timeout"),
        ({"query": PRIVATE_SENTINEL, "fail": True}, 30, "error"),
    ],
)
async def test_failure_history_never_retains_payload_or_exception_text(
    arguments,
    timeout_seconds: float,
    expected_status: str,
) -> None:
    executor = ToolExecutor(timeout_seconds=timeout_seconds)
    executor.register_tool(_EchoTool())
    try:
        result = await executor.execute_tool_call(_call(arguments))
        history = executor.get_execution_history()
    finally:
        executor.executor.shutdown(wait=False)

    assert "error" in result
    if expected_status == "error":
        assert PRIVATE_SENTINEL in repr(result)
    assert history[-1]["status"] == expected_status
    assert PRIVATE_SENTINEL not in repr(history)
    assert "error" not in history[-1]
    if expected_status == "error":
        assert history[-1]["exception_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_history_is_hard_bounded_at_100_records() -> None:
    executor = ToolExecutor()
    executor.register_tool(_EchoTool())
    try:
        for index in range(105):
            await executor.execute_tool_call(_call({"query": PRIVATE_SENTINEL}, str(index)))
        history = executor.get_execution_history(limit=1000)
    finally:
        executor.executor.shutdown(wait=False)

    assert len(history) == 100
    assert PRIVATE_SENTINEL not in repr(history)


@pytest.mark.asyncio
async def test_cache_hit_keeps_result_contract_and_payload_free_history() -> None:
    _EchoTool.calls = 0
    executor = ToolExecutor(enable_cache=True)
    executor.register_tool(_EchoTool())
    call = _call({"query": PRIVATE_SENTINEL})
    try:
        first = await executor.execute_tool_call(call)
        second = await executor.execute_tool_call(call)
        history = executor.get_execution_history()
    finally:
        executor.executor.shutdown(wait=False)

    assert first["result"] == {"private_result": PRIVATE_SENTINEL}
    assert second == {
        "tool_call_id": "private-call-id",
        "result": {"private_result": PRIVATE_SENTINEL},
        "cached": True,
    }
    assert _EchoTool.calls == 1
    assert history[-1]["status"] == "cached"
    assert history[-1]["cache_hit"] is True
    assert PRIVATE_SENTINEL not in repr(history)


@pytest.mark.asyncio
async def test_public_history_snapshot_cannot_mutate_retained_records() -> None:
    executor = ToolExecutor()
    executor.register_tool(_EchoTool())
    try:
        await executor.execute_tool_call(_call({"query": PRIVATE_SENTINEL}))
        snapshot = executor.get_execution_history()
        snapshot[0]["payload"] = PRIVATE_SENTINEL
        retained = executor.get_execution_history()
    finally:
        executor.executor.shutdown(wait=False)

    assert "payload" not in retained[0]
    assert PRIVATE_SENTINEL not in repr(retained)
