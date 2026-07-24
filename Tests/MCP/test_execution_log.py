from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook.MCP.execution_log import (
    ExecutionRecord,
    MCPExecutionLog,
    build_record,
)
from tldw_chatbook.Utils.private_paths import PrivatePathError


@pytest.fixture()
def log(tmp_path: Path) -> MCPExecutionLog:
    return MCPExecutionLog(tmp_path / "mcp_execution_log.jsonl", max_records_per_file=3)


def _record(tool: str = "search_docs", **kw) -> ExecutionRecord:
    defaults = dict(
        server_key="local:docs",
        tool_name=tool,
        initiator="test",
        ok=True,
        duration_ms=12,
        arguments={"query": "x"},
    )
    defaults.update(kw)
    return build_record(**defaults)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


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
    assert names[0] == "t6"  # newest first
    assert len(names) <= 6  # bounded: at most two generations
    assert "t0" not in names  # oldest generation dropped


def test_arguments_redacted_and_capture_off_drops_them():
    kept = build_record(
        server_key="local:docs",
        tool_name="t",
        initiator="test",
        ok=True,
        duration_ms=1,
        arguments={"api_key": "sk-123", "query": "ok"},
    )
    assert kept.arguments["api_key"] == "***"
    assert kept.arguments["query"] == "ok"
    dropped = build_record(
        server_key="local:docs",
        tool_name="t",
        initiator="test",
        ok=True,
        duration_ms=1,
        arguments={"api_key": "sk-123"},
        capture_args=False,
    )
    assert dropped.arguments is None


def test_read_recent_survives_corrupt_line(log, tmp_path):
    log.append(_record("good"))
    with (tmp_path / "mcp_execution_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    log.append(_record("after"))
    names = [r["tool_name"] for r in log.read_recent()]
    assert "good" in names and "after" in names  # corrupt line skipped, no crash


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_append_and_rotation_keep_parent_and_generations_private(log, tmp_path):
    tmp_path.chmod(0o755)

    for index in range(4):
        log.append(_record(f"private-{index}"))

    active = tmp_path / "mcp_execution_log.jsonl"
    rotated = tmp_path / "mcp_execution_log.jsonl.1"
    assert _mode(tmp_path) == 0o700
    assert _mode(active) == 0o600
    assert _mode(rotated) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_existing_generations_are_hardened_before_read_and_append(tmp_path):
    active = tmp_path / "mcp_execution_log.jsonl"
    rotated = tmp_path / "mcp_execution_log.jsonl.1"
    active.write_text(json.dumps({"tool_name": "active"}) + "\n", encoding="utf-8")
    rotated.write_text(json.dumps({"tool_name": "rotated"}) + "\n", encoding="utf-8")
    active.chmod(0o644)
    rotated.chmod(0o644)
    execution_log = MCPExecutionLog(active, max_records_per_file=3)

    execution_log.read_recent()
    execution_log.append(_record("after"))

    assert _mode(active) == 0o600
    assert _mode(rotated) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
@pytest.mark.parametrize("operation", ["read", "append"])
def test_active_generation_symlink_is_rejected_without_touching_sentinel(
    tmp_path,
    operation,
):
    outside = tmp_path.parent / f"mcp-outside-{operation}-SENTINEL"
    outside.write_text("preserve", encoding="utf-8")
    active = tmp_path / "mcp_execution_log.jsonl"
    active.symlink_to(outside)
    execution_log = MCPExecutionLog(active, max_records_per_file=3)

    if operation == "read":
        assert execution_log.read_recent() == []
    else:
        with pytest.raises(PrivatePathError):
            execution_log.append(_record("blocked"))

    assert outside.read_text(encoding="utf-8") == "preserve"


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
def test_rotated_generation_symlink_blocks_append_without_touching_sentinel(
    tmp_path,
):
    outside = tmp_path.parent / "mcp-rotated-outside-SENTINEL"
    outside.write_text("preserve", encoding="utf-8")
    active = tmp_path / "mcp_execution_log.jsonl"
    rotated = tmp_path / "mcp_execution_log.jsonl.1"
    rotated.symlink_to(outside)
    execution_log = MCPExecutionLog(active, max_records_per_file=3)

    with pytest.raises(PrivatePathError):
        execution_log.append(_record("blocked"))

    assert not active.exists()
    assert outside.read_text(encoding="utf-8") == "preserve"


@pytest.mark.skipif(os.name != "posix", reason="POSIX parent contract")
def test_shared_writable_ancestor_is_rejected_without_creating_log(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    shared.chmod(0o777)
    active = shared / "owned" / "mcp_execution_log.jsonl"
    execution_log = MCPExecutionLog(active)

    with pytest.raises(PrivatePathError):
        execution_log.append(_record("blocked"))

    assert not active.parent.exists()
