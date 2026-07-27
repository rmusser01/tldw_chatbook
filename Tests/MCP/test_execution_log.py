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
        arguments={"query": "private input"},
        registered_argument_names={"query"},
        result={"private": "result"},
    )
    defaults.update(kw)
    return build_record(**defaults)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_append_and_read_recent_roundtrip(log, tmp_path):
    log.append(_record("a"))
    log.append(
        _record(
            "b",
            ok=False,
            status="error",
            error_category="execution_failed",
            exception_type="RuntimeError",
            result=None,
        )
    )
    rows = log.read_recent()
    assert [r["tool_name"] for r in rows] == ["b", "a"]  # newest first
    assert rows[0]["error_category"] == "execution_failed"
    assert rows[0]["exception_type"] == "RuntimeError"
    assert rows[0]["ok"] is False
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


def test_payload_values_are_replaced_by_registered_argument_and_result_metadata():
    private = "MCP-PRIVATE-SENTINEL-sk-not-a-real-key"
    record = build_record(
        server_key="local:docs",
        tool_name="t",
        initiator="test",
        ok=True,
        duration_ms=1,
        arguments={
            "query": private,
            "limit": 10,
            private: "unknown argument value",
        },
        registered_argument_names={"query", "limit"},
        result={"secret": private, "data": [1, 2]},
    )

    assert record.argument_names == ("limit", "query")
    assert record.unknown_argument_count == 1
    assert record.result_type == "dict"
    assert record.result_size == 2
    assert private not in repr(record)
    assert not hasattr(record, "arguments")
    assert not hasattr(record, "result_excerpt")
    assert not hasattr(record, "error")


def test_error_record_accepts_categories_not_raw_exception_text():
    private = "MCP-ERROR-SENTINEL-sk-not-a-real-key"
    record = build_record(
        server_key="local:docs",
        tool_name="t",
        initiator="test",
        ok=False,
        duration_ms=1,
        status="http_error",
        error_category="http_error",
        exception_type="HTTPStatusError",
        status_code=503,
    )

    assert record.error_category == "http_error"
    assert record.exception_type == "HTTPStatusError"
    assert record.status_code == 503
    assert private not in repr(record)


def test_append_defensively_sanitizes_identity_fields(tmp_path):
    private = "MCP-IDENTITY-SENTINEL-sk-not-a-real-key"
    record = ExecutionRecord(
        ts="invalid",
        server_key=private,
        tool_name=private,
        initiator=private,
        decision=private,
        ok=False,
        status="error",
        duration_ms=1,
        error_category=private,
        exception_type=private,
        status_code=None,
        argument_names=(private,),
        unknown_argument_count=1,
        result_type=private,
        result_size=0,
    )
    execution_log = MCPExecutionLog(tmp_path / "mcp_execution_log.jsonl")

    execution_log.append(record)

    raw = (tmp_path / "mcp_execution_log.jsonl").read_text(encoding="utf-8")
    assert private not in raw
    assert raw.count("invalid") >= 5


def test_read_recent_survives_corrupt_line(log, tmp_path):
    log.append(_record("good"))
    with (tmp_path / "mcp_execution_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    log.append(_record("after"))
    names = [r["tool_name"] for r in log.read_recent()]
    assert "good" in names and "after" in names  # corrupt line skipped, no crash


@pytest.mark.parametrize("generation", ["active", "rotated"])
def test_legacy_payload_records_are_migrated_off_disk_on_read(
    tmp_path: Path,
    generation: str,
) -> None:
    private = "MCP-LEGACY-SENTINEL-sk-not-a-real-key"
    active = tmp_path / "mcp_execution_log.jsonl"
    selected = active if generation == "active" else active.with_name(active.name + ".1")
    selected.write_text(
        json.dumps(
            {
                "ts": "2026-07-24T00:00:00+00:00",
                "server_key": "local:docs",
                "tool_name": "search",
                "initiator": "test",
                "decision": "allowed",
                "ok": False,
                "duration_ms": 12,
                "error": private,
                "arguments": {"query": private},
                "result_excerpt": private,
            }
        )
        + "\n"
        + "{torn "
        + private,
        encoding="utf-8",
    )
    execution_log = MCPExecutionLog(active)

    rows = execution_log.read_recent()

    raw = selected.read_text(encoding="utf-8")
    assert private not in raw
    assert "arguments" not in raw
    assert "result_excerpt" not in raw
    assert "error" not in json.loads(raw)
    assert len(rows) == 1
    assert rows[0]["error_category"] == "legacy_error"
    assert rows[0]["unknown_argument_count"] == 1
    assert rows[0]["argument_names"] == []


def test_legacy_payload_records_are_migrated_before_append(tmp_path: Path) -> None:
    private = "MCP-APPEND-LEGACY-SENTINEL-sk-not-a-real-key"
    active = tmp_path / "mcp_execution_log.jsonl"
    active.write_text(
        json.dumps(
            {
                "tool_name": "old",
                "arguments": {"query": private},
                "result_excerpt": private,
                "error": private,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    execution_log = MCPExecutionLog(active)

    execution_log.append(_record("new"))

    raw = active.read_text(encoding="utf-8")
    assert private not in raw
    assert "arguments" not in raw
    assert "result_excerpt" not in raw
    assert all("error" not in json.loads(line) for line in raw.splitlines())


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
