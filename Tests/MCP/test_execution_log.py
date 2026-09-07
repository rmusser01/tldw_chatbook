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


# ---------------------------------------------------------------------------
# TASK-21134: the legacy scrub must not re-run on an unchanged generation
# ---------------------------------------------------------------------------


def _count_parses(monkeypatch) -> dict[str, int]:
    """Count full-line JSON decodes inside the execution-log module."""
    from tldw_chatbook.MCP import execution_log as module

    counter = {"loads": 0}
    real_loads = json.loads

    def counting_loads(payload, *args, **kwargs):
        counter["loads"] += 1
        return real_loads(payload, *args, **kwargs)

    monkeypatch.setattr(module.json, "loads", counting_loads)
    return counter


def test_steady_state_appends_do_not_reparse_the_whole_log(tmp_path, monkeypatch):
    """A scrub already applied to bytes we wrote must not be applied again.

    Before TASK-21134 each append re-read and re-parsed every line of both
    generations: 499 json.loads and 4.6 ms of wall time per tool invocation at
    the 500-record cap, purely to re-derive bytes identical to the ones the
    previous append had just written.
    """
    execution_log = MCPExecutionLog(
        tmp_path / "mcp_execution_log.jsonl", max_records_per_file=200
    )
    for index in range(60):
        execution_log.append(_record(f"warm-{index}"))

    counter = _count_parses(monkeypatch)
    for index in range(5):
        execution_log.append(_record(f"hot-{index}"))

    assert counter["loads"] == 0, (
        f"appends re-parsed the log {counter['loads']} times"
    )


def test_appends_after_a_rotation_do_not_reparse_either_generation(
    tmp_path, monkeypatch
):
    """Both generations are cached, so a rotated file is scrubbed once."""
    execution_log = MCPExecutionLog(
        tmp_path / "mcp_execution_log.jsonl", max_records_per_file=20
    )
    for index in range(45):  # forces at least two rotations
        execution_log.append(_record(f"warm-{index}"))
    assert (tmp_path / "mcp_execution_log.jsonl.1").exists()

    counter = _count_parses(monkeypatch)
    for index in range(5):
        execution_log.append(_record(f"hot-{index}"))

    assert counter["loads"] == 0


def test_a_generation_changed_behind_our_back_is_still_scrubbed(tmp_path):
    """The cache is a staleness check, never a licence to trust stale bytes."""
    private = "MCP-CACHE-LEGACY-SENTINEL-sk-not-a-real-key"
    active = tmp_path / "mcp_execution_log.jsonl"
    execution_log = MCPExecutionLog(active, max_records_per_file=200)
    execution_log.append(_record("warm"))  # primes the cache

    # Another writer (or an older build) appends a legacy payload row.
    with active.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "tool_name": "old",
                    "arguments": {"query": private},
                    "result_excerpt": private,
                }
            )
            + "\n"
        )

    execution_log.append(_record("after"))

    raw = active.read_text(encoding="utf-8")
    assert private not in raw
    assert "result_excerpt" not in raw
    assert [row["tool_name"] for row in execution_log.read_recent()] == [
        "after",
        "old",
        "warm",
    ]


def test_a_byte_for_byte_same_size_replacement_misses_the_cache(tmp_path):
    """Neither size nor mtime alone can decide staleness.

    The replacement below is padded to the EXACT byte length of the file it
    displaces AND has its mtime restored to the displaced file's (what an
    mtime-preserving restore or ``rsync -t`` does), so a fingerprint built on
    either field alone would report a hit and hand back the stale, unscrubbed
    bytes. Only the inode tells these two files apart.
    """
    private = "MCP-REPLACED-SENTINEL-sk-not-a-real-key"
    active = tmp_path / "mcp_execution_log.jsonl"
    execution_log = MCPExecutionLog(active, max_records_per_file=200)
    for index in range(4):
        execution_log.append(_record(f"warm-{index}"))

    original = active.stat()
    target = original.st_size
    legacy = {"tool_name": "old", "result_excerpt": private, "pad": ""}
    padding = target - len((json.dumps(legacy) + "\n").encode("utf-8"))
    assert padding >= 0, "widen the warm-up above"
    legacy["pad"] = "x" * padding
    line = json.dumps(legacy) + "\n"
    assert len(line.encode("utf-8")) == target

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(line, encoding="utf-8")
    replacement.replace(active)
    os.utime(active, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert active.stat().st_size == original.st_size
    assert active.stat().st_mtime_ns == original.st_mtime_ns
    assert active.stat().st_ino != original.st_ino

    execution_log.append(_record("after"))

    raw = active.read_text(encoding="utf-8")
    assert private not in raw
    assert [row["tool_name"] for row in execution_log.read_recent()] == [
        "after",
        "old",
    ]


def test_a_concurrent_append_is_not_pinned_by_the_cache(tmp_path, monkeypatch):
    """Another writer landing inside our own append window must not be cached.

    A second process can append between the bytes we compute and the stat we
    fingerprint them with. Caching then would pin content already short of the
    file, under a fingerprint claiming it is current.
    """
    from tldw_chatbook.MCP import execution_log as module

    private = "MCP-CONCURRENT-SENTINEL-sk-not-a-real-key"
    active = tmp_path / "mcp_execution_log.jsonl"
    execution_log = MCPExecutionLog(active, max_records_per_file=200)
    execution_log.append(_record("warm"))

    real_identity = module.MCPExecutionLog._identity
    seen = {"active_calls": 0}

    def intruding_identity(path):
        # One append makes two _identity calls on the active generation: the
        # staleness check, then the one that fingerprints what we just wrote.
        # The window this guards is between them, so intrude before the second.
        if Path(path) == active:
            seen["active_calls"] += 1
            if seen["active_calls"] == 2:
                with active.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"tool_name": "other", "result_excerpt": private})
                        + "\n"
                    )
        return real_identity(path)

    monkeypatch.setattr(
        module.MCPExecutionLog, "_identity", staticmethod(intruding_identity)
    )
    execution_log.append(_record("ours"))
    monkeypatch.undo()

    # The intruder's row must be visible and scrubbed, not overwritten by a
    # stale cached copy of the log.
    names = [row["tool_name"] for row in execution_log.read_recent()]
    assert names == ["other", "ours", "warm"], names
    assert private not in active.read_text(encoding="utf-8")
