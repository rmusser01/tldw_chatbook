from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.MCP.execution_log import ExecutionRecord, MCPExecutionLog, build_record


@pytest.fixture()
def log(tmp_path: Path) -> MCPExecutionLog:
    return MCPExecutionLog(tmp_path / "mcp_execution_log.jsonl", max_records_per_file=3)


def _record(tool: str = "search_docs", **kw) -> ExecutionRecord:
    defaults = dict(server_key="local:docs", tool_name=tool, initiator="test",
                    ok=True, duration_ms=12, arguments={"query": "x"})
    defaults.update(kw)
    return build_record(**defaults)


def test_append_and_read_recent_roundtrip(log, tmp_path):
    log.append(_record("a"))
    log.append(_record("b", ok=False, error="boom"))
    rows = log.read_recent()
    assert [r["tool_name"] for r in rows] == ["b", "a"]  # newest first
    assert rows[0]["error"] == "boom" and rows[0]["ok"] is False
    raw = (tmp_path / "mcp_execution_log.jsonl").read_text().strip().splitlines()
    assert all(json.loads(line) for line in raw)  # valid JSONL


def test_rotation_keeps_two_generations(log, tmp_path):
    for i in range(7):  # cap 3 per file -> rotations
        log.append(_record(f"t{i}"))
    active = tmp_path / "mcp_execution_log.jsonl"
    rotated = tmp_path / "mcp_execution_log.jsonl.1"
    assert active.exists() and rotated.exists()
    names = [r["tool_name"] for r in log.read_recent(limit=100)]
    assert names[0] == "t6"          # newest first
    assert len(names) <= 6           # bounded: at most two generations
    assert "t0" not in names         # oldest generation dropped


def test_arguments_redacted_and_capture_off_drops_them():
    kept = build_record(server_key="local:docs", tool_name="t", initiator="test",
                        ok=True, duration_ms=1,
                        arguments={"api_key": "sk-123", "query": "ok"})
    assert kept.arguments["api_key"] == "***"
    assert kept.arguments["query"] == "ok"
    dropped = build_record(server_key="local:docs", tool_name="t", initiator="test",
                           ok=True, duration_ms=1,
                           arguments={"api_key": "sk-123"}, capture_args=False)
    assert dropped.arguments is None


def test_read_recent_survives_corrupt_line(log, tmp_path):
    log.append(_record("good"))
    with (tmp_path / "mcp_execution_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    log.append(_record("after"))
    names = [r["tool_name"] for r in log.read_recent()]
    assert "good" in names and "after" in names  # corrupt line skipped, no crash
