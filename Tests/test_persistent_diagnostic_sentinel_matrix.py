from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Logging_Config import (
    PrivateRotatingFileHandler,
    _forward_loguru_to_standard,
)
from tldw_chatbook.MCP.execution_log import MCPExecutionLog, build_record
from tldw_chatbook.Tools.tool_executor import Tool, ToolExecutor
from tldw_chatbook.Utils.persistent_diagnostics import (
    PersistentDiagnosticFilter,
    log_persistent_metadata,
)


PRIVATE_SENTINEL = "MATRIX-PRIVATE-SENTINEL-sk-not-a-real-key"


class _MatrixTool(Tool):
    @property
    def name(self) -> str:
        return "matrix_tool"

    @property
    def description(self) -> str:
        return "Exercise persistent diagnostic paths."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "delay": {"type": "number"},
            },
        }

    async def execute(self, **kwargs):
        if kwargs.get("delay"):
            await asyncio.sleep(kwargs["delay"])
        return {"private": PRIVATE_SENTINEL}


def _handler(path: Path) -> PrivateRotatingFileHandler:
    handler = PrivateRotatingFileHandler(
        path,
        maxBytes=350,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    handler.addFilter(PersistentDiagnosticFilter())
    return handler


def _generations(path: Path) -> str:
    return "\n".join(
        candidate.read_text(encoding="utf-8")
        for candidate in sorted(path.parent.glob(path.name + "*"))
        if candidate.is_file()
    )


def _emit_owned_loguru_payload(module_name: str, message: str) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / Path(*module_name.split("."))
    ).with_suffix(".py")
    code = compile(
        "loguru_logger.debug(message)",
        str(source),
        "exec",
    )
    exec(
        code,
        {
            "__name__": module_name,
            "loguru_logger": loguru_logger,
            "message": message,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    ["success", "parsing_error", "timeout", "cache", "streaming", "http_error"],
)
async def test_application_sink_sentinel_matrix(scenario: str, tmp_path: Path) -> None:
    path = tmp_path / "application.log"
    handler = _handler(path)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    sink_id = loguru_logger.add(_forward_loguru_to_standard, level="DEBUG")
    executor: ToolExecutor | None = None
    try:
        if scenario in {"success", "parsing_error", "timeout", "cache"}:
            executor = ToolExecutor(
                timeout_seconds=0.001 if scenario == "timeout" else 30,
                enable_cache=scenario == "cache",
            )
            executor.register_tool(_MatrixTool())
            arguments = (
                "{not-json " + PRIVATE_SENTINEL
                if scenario == "parsing_error"
                else {
                    "query": PRIVATE_SENTINEL,
                    **({"delay": 0.05} if scenario == "timeout" else {}),
                }
            )
            call = {
                "id": "matrix-call",
                "function": {"name": "matrix_tool", "arguments": arguments},
            }
            await executor.execute_tool_call(call)
            if scenario == "cache":
                await executor.execute_tool_call(call)
        else:
            module_name = "tldw_chatbook.LLM_Calls.LLM_API_Calls"
            _emit_owned_loguru_payload(
                module_name,
                f"{scenario} body={PRIVATE_SENTINEL}",
            )
            log_persistent_metadata(
                logging.getLogger(module_name),
                logging.INFO,
                "provider_request",
                provider="openai",
                status=scenario,
                payload_length=len(PRIVATE_SENTINEL),
                streaming=scenario == "streaming",
                status_code=503 if scenario == "http_error" else 200,
            )
    finally:
        loguru_logger.remove(sink_id)
        root.removeHandler(handler)
        root.setLevel(old_level)
        handler.close()

    persisted = _generations(path)
    assert PRIVATE_SENTINEL not in persisted
    assert "sk-not-a-real-key" not in persisted
    if scenario == "cache":
        assert "status=cached" in persisted
        assert "cache_hit=true" in persisted
    elif scenario == "parsing_error":
        assert "status=parse_error" in persisted
    elif scenario == "timeout":
        assert "status=timeout" in persisted
    elif scenario == "success":
        assert "status=success" in persisted
    else:
        assert f"status={scenario}" in persisted


@pytest.mark.parametrize(
    ("scenario", "ok", "error_category", "exception_type", "status_code"),
    [
        ("success", True, None, None, None),
        ("http_error", False, "http_error", "HTTPStatusError", 503),
        ("parsing_error", False, "parse_error", "JSONDecodeError", None),
        ("timeout", False, "timeout", "TimeoutError", None),
        ("streaming", True, None, None, 200),
        ("cached", True, None, None, None),
    ],
)
def test_mcp_jsonl_sentinel_matrix(
    scenario: str,
    ok: bool,
    error_category: str | None,
    exception_type: str | None,
    status_code: int | None,
    tmp_path: Path,
) -> None:
    execution_log = MCPExecutionLog(tmp_path / "mcp_execution_log.jsonl")
    record = build_record(
        server_key="local:docs",
        tool_name="search",
        initiator="agent",
        decision="approved",
        ok=ok,
        status=scenario,
        duration_ms=12,
        error_category=error_category,
        exception_type=exception_type,
        status_code=status_code,
        arguments={"query": PRIVATE_SENTINEL, PRIVATE_SENTINEL: "unknown"},
        registered_argument_names={"query"},
        result={"private": PRIVATE_SENTINEL} if ok else None,
    )

    execution_log.append(record)

    path = tmp_path / "mcp_execution_log.jsonl"
    raw = path.read_text(encoding="utf-8")
    rows = execution_log.read_recent()
    assert PRIVATE_SENTINEL not in raw
    assert "sk-not-a-real-key" not in raw
    assert rows[0]["status"] == scenario
    assert rows[0]["argument_names"] == ["query"]
    assert rows[0]["unknown_argument_count"] == 1
    assert "arguments" not in rows[0]
    assert "result_excerpt" not in rows[0]
    assert "error" not in rows[0]
