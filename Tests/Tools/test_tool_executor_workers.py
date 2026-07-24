"""Worker, cancellation, and batch contracts for ToolExecutor."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import tldw_chatbook.Tools.tool_executor as tool_executor_module
from tldw_chatbook.Tools.tool_executor import Tool, ToolExecutor


class _UnexpectedControlFlow(BaseException):
    pass


def _call(call_id: str, **arguments: Any) -> dict:
    return {
        "id": call_id,
        "function": {
            "name": "controlled",
            "arguments": arguments,
        },
    }


class _ControlledTool(Tool):
    def __init__(self) -> None:
        self.active = 0
        self.peak_active = 0
        self.releases: dict[str, asyncio.Event] = {}
        self.started: dict[str, asyncio.Event] = {}
        self.execution_delays: dict[str, float] = {}
        self.failures: set[str] = set()
        self.cancel_keys: set[str] = set()
        self.control_flow_keys: set[str] = set()
        self.cancelled: set[str] = set()

    @property
    def name(self) -> str:
        return "controlled"

    @property
    def description(self) -> str:
        return "Controlled asynchronous tool for worker-contract tests."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        }

    async def execute(self, **kwargs) -> dict:
        key = kwargs["key"]
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.started.setdefault(key, asyncio.Event()).set()
        try:
            release = self.releases.get(key)
            if release is not None:
                await release.wait()
            delay = self.execution_delays.get(key, 0)
            if delay:
                await asyncio.sleep(delay)
            if key in self.cancel_keys:
                raise asyncio.CancelledError
            if key in self.control_flow_keys:
                raise _UnexpectedControlFlow(key)
            if key in self.failures:
                raise RuntimeError(f"failure-{key}")
            return {"key": key}
        except asyncio.CancelledError:
            self.cancelled.add(key)
            raise
        finally:
            self.active -= 1


@pytest.mark.parametrize("max_workers", [True, 0, -1, 1.5])
def test_invalid_max_workers_fail_during_construction(max_workers: object) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        ToolExecutor(max_workers=max_workers)


@pytest.mark.parametrize(
    "timeout_seconds",
    [True, 0, -1, float("nan"), float("inf")],
)
def test_invalid_timeout_fails_during_construction(timeout_seconds: object) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        ToolExecutor(timeout_seconds=timeout_seconds)


@pytest.mark.asyncio
async def test_max_workers_bounds_actual_tool_execution() -> None:
    tool = _ControlledTool()
    executor = ToolExecutor(max_workers=2, timeout_seconds=1)
    executor.register_tool(tool)
    calls = [_call(str(index), key=str(index)) for index in range(5)]
    for index in range(5):
        tool.releases[str(index)] = asyncio.Event()
        tool.started[str(index)] = asyncio.Event()

    batch = asyncio.create_task(executor.execute_tool_calls(calls))
    await asyncio.wait_for(
        asyncio.gather(
            tool.started["0"].wait(),
            tool.started["1"].wait(),
        ),
        timeout=1,
    )
    await asyncio.sleep(0)
    assert tool.peak_active == 2
    assert sum(event.is_set() for event in tool.started.values()) == 2

    for index in range(5):
        tool.releases[str(index)].set()
    results = await asyncio.wait_for(batch, timeout=1)

    assert [result["tool_call_id"] for result in results] == [
        str(index) for index in range(5)
    ]
    assert tool.peak_active == 2


@pytest.mark.asyncio
async def test_queue_wait_is_outside_execution_timeout() -> None:
    tool = _ControlledTool()
    executor = ToolExecutor(max_workers=1, timeout_seconds=0.2)
    executor.register_tool(tool)
    tool.releases["first"] = asyncio.Event()
    tool.started["first"] = asyncio.Event()
    tool.started["second"] = asyncio.Event()
    tool.execution_delays["second"] = 0.15

    first = asyncio.create_task(
        executor.execute_tool_call(_call("first", key="first"))
    )
    await asyncio.wait_for(tool.started["first"].wait(), timeout=1)
    second = asyncio.create_task(
        executor.execute_tool_call(_call("second", key="second"))
    )

    await asyncio.sleep(0.08)
    assert not tool.started["second"].is_set()
    tool.releases["first"].set()

    first_result, second_result = await asyncio.gather(first, second)
    assert "error" not in first_result
    assert "error" not in second_result


def _assert_one_cancelled_record(executor: ToolExecutor) -> None:
    history = executor.get_execution_history()
    assert len(history) == 1
    assert history[0]["status"] == "cancelled"
    assert history[0]["argument_names"] == ["key"]
    assert history[0]["duration_ms"] >= 0
    assert "secret-value" not in repr(history)
    assert "result" not in history[0]
    assert "error" not in history[0]


@pytest.mark.asyncio
async def test_cancellation_during_cache_lookup_is_terminal_and_propagates() -> None:
    executor = ToolExecutor(enable_cache=True)
    executor.register_tool(_ControlledTool())
    entered = asyncio.Event()

    async def blocking_get(*_args):
        entered.set()
        await asyncio.Event().wait()

    executor.cache.get = blocking_get
    task = asyncio.create_task(
        executor.execute_tool_call(_call("cache-read", key="secret-value"))
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_one_cancelled_record(executor)


@pytest.mark.asyncio
async def test_cancellation_while_queued_is_terminal_and_propagates() -> None:
    tool = _ControlledTool()
    executor = ToolExecutor(max_workers=1)
    executor.register_tool(tool)
    tool.releases["occupant"] = asyncio.Event()
    tool.started["occupant"] = asyncio.Event()

    occupant = asyncio.create_task(
        executor.execute_tool_call(_call("occupant", key="occupant"))
    )
    await asyncio.wait_for(tool.started["occupant"].wait(), timeout=1)
    queued = asyncio.create_task(
        executor.execute_tool_call(_call("queued", key="secret-value"))
    )
    while not executor._execution_semaphore._waiters:
        await asyncio.sleep(0)
    queued.cancel()

    with pytest.raises(asyncio.CancelledError):
        await queued
    _assert_one_cancelled_record(executor)

    tool.releases["occupant"].set()
    assert "error" not in await occupant


@pytest.mark.asyncio
async def test_cancellation_during_tool_execution_is_terminal_and_propagates() -> None:
    tool = _ControlledTool()
    executor = ToolExecutor()
    executor.register_tool(tool)
    tool.releases["secret-value"] = asyncio.Event()
    tool.started["secret-value"] = asyncio.Event()

    task = asyncio.create_task(
        executor.execute_tool_call(_call("executing", key="secret-value"))
    )
    await asyncio.wait_for(tool.started["secret-value"].wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_one_cancelled_record(executor)
    assert tool.active == 0


@pytest.mark.asyncio
async def test_cancellation_during_cache_write_is_terminal_and_propagates() -> None:
    executor = ToolExecutor(enable_cache=True)
    executor.register_tool(_ControlledTool())
    entered = asyncio.Event()
    executor.cache.get = lambda *_: asyncio.sleep(0, result=None)

    async def blocking_set(*_args):
        entered.set()
        await asyncio.Event().wait()

    executor.cache.set = blocking_set
    task = asyncio.create_task(
        executor.execute_tool_call(_call("cache-write", key="secret-value"))
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_one_cancelled_record(executor)


@pytest.mark.asyncio
async def test_timeout_releases_capacity_and_records_one_terminal_item() -> None:
    tool = _ControlledTool()
    executor = ToolExecutor(max_workers=1, timeout_seconds=0.01)
    executor.register_tool(tool)
    tool.releases["blocked"] = asyncio.Event()
    tool.started["blocked"] = asyncio.Event()

    timed_out = await executor.execute_tool_call(_call("blocked", key="blocked"))
    next_result = await executor.execute_tool_call(_call("next", key="next"))

    assert "timed out" in timed_out["error"]
    assert "error" not in next_result
    assert [item["status"] for item in executor.get_execution_history()] == [
        "timeout",
        "success",
    ]


@pytest.mark.asyncio
async def test_ordinary_error_releases_capacity_and_records_terminal_item() -> None:
    tool = _ControlledTool()
    tool.failures.add("fails")
    executor = ToolExecutor(max_workers=1)
    executor.register_tool(tool)

    failed = await executor.execute_tool_call(_call("fails", key="fails"))
    next_result = await executor.execute_tool_call(_call("next", key="next"))

    assert "failure-fails" in failed["error"]
    assert "error" not in next_result
    assert [item["status"] for item in executor.get_execution_history()] == [
        "error",
        "success",
    ]


@pytest.mark.asyncio
async def test_parse_error_records_exactly_one_terminal_item() -> None:
    executor = ToolExecutor()
    executor.register_tool(_ControlledTool())
    call = _call("parse", key="unused")
    call["function"]["arguments"] = "{not-json"

    result = await executor.execute_tool_call(call)

    assert "Invalid JSON" in result["error"]
    assert [item["status"] for item in executor.get_execution_history()] == [
        "parse_error"
    ]


@pytest.mark.asyncio
async def test_batch_returns_request_order_after_out_of_order_completion() -> None:
    tool = _ControlledTool()
    tool.execution_delays.update({"first": 0.03, "second": 0.01, "third": 0.02})
    executor = ToolExecutor(max_workers=3)
    executor.register_tool(tool)

    results = await executor.execute_tool_calls(
        [
            _call("first", key="first"),
            _call("second", key="second"),
            _call("third", key="third"),
        ]
    )

    assert [result["tool_call_id"] for result in results] == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
async def test_batch_keeps_ordinary_leaf_failure_as_ordered_result() -> None:
    tool = _ControlledTool()
    tool.failures.add("second")
    executor = ToolExecutor(max_workers=3)
    executor.register_tool(tool)

    results = await executor.execute_tool_calls(
        [
            _call("first", key="first"),
            _call("second", key="second"),
            _call("third", key="third"),
        ]
    )

    assert [result["tool_call_id"] for result in results] == [
        "first",
        "second",
        "third",
    ]
    assert "error" not in results[0]
    assert "failure-second" in results[1]["error"]
    assert "error" not in results[2]


async def _captured_batch(
    executor: ToolExecutor, calls: list[dict]
) -> tuple[asyncio.Task, list[asyncio.Task]]:
    children: list[asyncio.Task] = []
    original = executor.execute_tool_call

    async def capture(call: dict) -> dict:
        children.append(asyncio.current_task())
        return await original(call)

    executor.execute_tool_call = capture
    batch = asyncio.create_task(executor.execute_tool_calls(calls))
    while len(children) < len(calls):
        await asyncio.sleep(0)
    return batch, children


@pytest.mark.asyncio
async def test_parent_batch_cancellation_cancels_and_drains_every_child() -> None:
    tool = _ControlledTool()
    executor = ToolExecutor(max_workers=1)
    executor.register_tool(tool)
    tool.releases["first"] = asyncio.Event()
    tool.started["first"] = asyncio.Event()
    calls = [
        _call("first", key="first"),
        _call("second", key="second"),
        _call("third", key="third"),
    ]

    batch, children = await _captured_batch(executor, calls)
    await asyncio.wait_for(tool.started["first"].wait(), timeout=1)
    batch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await batch

    assert all(child.done() for child in children)
    assert tool.cancelled == {"first"}
    assert tool.active == 0
    assert [item["status"] for item in executor.get_execution_history()] == [
        "cancelled",
        "cancelled",
        "cancelled",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("control_flow", ["cancelled", "base_exception"])
async def test_child_control_flow_cancels_and_drains_unfinished_siblings(
    control_flow: str,
) -> None:
    tool = _ControlledTool()
    executor = ToolExecutor(max_workers=2)
    executor.register_tool(tool)
    tool.releases["control"] = asyncio.Event()
    tool.releases["blocked"] = asyncio.Event()
    tool.started["control"] = asyncio.Event()
    tool.started["blocked"] = asyncio.Event()
    if control_flow == "cancelled":
        tool.cancel_keys.add("control")
        expected_exception = asyncio.CancelledError
    else:
        tool.control_flow_keys.add("control")
        expected_exception = _UnexpectedControlFlow

    calls = [
        _call("control", key="control"),
        _call("blocked", key="blocked"),
        _call("queued", key="queued"),
    ]
    batch, children = await _captured_batch(executor, calls)
    await asyncio.wait_for(
        asyncio.gather(
            tool.started["control"].wait(),
            tool.started["blocked"].wait(),
        ),
        timeout=1,
    )
    tool.releases["control"].set()

    with pytest.raises(expected_exception):
        await asyncio.wait_for(batch, timeout=1)

    assert all(child.done() for child in children)
    assert "blocked" in tool.cancelled
    assert tool.active == 0


def test_reload_replaces_global_without_retired_pool_state(monkeypatch) -> None:
    retired = ToolExecutor(max_workers=1)
    replacement = ToolExecutor(max_workers=3)
    monkeypatch.setattr(tool_executor_module, "_global_executor", retired)

    def install_replacement():
        tool_executor_module._global_executor = replacement
        return replacement

    monkeypatch.setattr(
        tool_executor_module,
        "get_tool_executor",
        install_replacement,
    )

    assert tool_executor_module.reload_tool_executor() is replacement
    assert tool_executor_module._global_executor is replacement
    assert not hasattr(retired, "executor")
