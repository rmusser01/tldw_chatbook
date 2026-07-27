"""Writer: binding, segmentation, durability, degradation."""

from pathlib import Path

import pytest

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Agents.run_log_format import iter_records


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Pin the writer's resolved root to a temp dir."""
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    return tmp_path


def make(root_dir, **kw):
    writer = RunLogWriter(**kw)
    writer.bind("run-abc")
    return writer


def test_unbound_writer_writes_nothing(root):
    writer = RunLogWriter()
    assert writer.append(run_id="r", kind="primary", type="model", content="x") is None
    assert not (root / "agent-runs").exists()


def test_bind_creates_the_run_directory_and_gitignore(root):
    make(root)
    assert (root / "agent-runs" / "run-abc").is_dir()
    assert (root / "agent-runs" / ".gitignore").read_text() == "*\n"


def test_existing_gitignore_is_never_overwritten(root):
    (root / "agent-runs").mkdir()
    (root / "agent-runs" / ".gitignore").write_text("keep me\n")
    make(root)
    assert (root / "agent-runs" / ".gitignore").read_text() == "keep me\n"


def test_records_are_numbered_monotonically_from_one(root):
    writer = make(root)
    numbers = [
        writer.append(run_id="r", kind="primary", type="model", content=str(i))
        for i in range(3)
    ]
    assert numbers == [1, 2, 3]


def test_a_child_run_shares_the_parent_counter(root):
    writer = make(root)
    writer.append(run_id="parent", kind="primary", type="model", content="a")
    writer.append(run_id="child", kind="subagent", type="model", content="b")
    data = (root / "agent-runs" / "run-abc" / "logs.0001.txt").read_bytes()
    parsed = list(iter_records(data))
    assert [(p.number, p.run_id) for p in parsed] == [(1, "parent"), (2, "child")]


def test_second_bind_is_ignored(root):
    writer = make(root)
    writer.bind("run-other")
    assert writer.log_dir.name == "run-abc"


def test_segment_rolls_and_no_record_spans_a_boundary(root):
    writer = make(root, segment_bytes=400)
    for _ in range(6):
        writer.append(run_id="r", kind="primary", type="model", content="x" * 100)
    run_dir = root / "agent-runs" / "run-abc"
    segments = sorted(run_dir.glob("logs.*.txt"))
    assert len(segments) > 1
    # Every segment parses standalone: no record straddles a boundary.
    total = 0
    for segment in segments:
        parsed = list(iter_records(segment.read_bytes()))
        assert parsed, f"{segment.name} parsed to nothing"
        total += len(parsed)
    assert total == 6


def test_oversized_record_is_capped_and_marked(root):
    writer = make(root, max_record_bytes=50)
    writer.append(run_id="r", kind="primary", type="tool_result", content="y" * 500)
    data = (root / "agent-runs" / "run-abc" / "logs.0001.txt").read_bytes()
    (parsed,) = list(iter_records(data))
    assert len(parsed.content.encode()) <= 50
    assert parsed.truncated_from == 500


def test_unresolvable_root_deactivates_the_writer(monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is False
    assert writer.append(run_id="r", kind="primary", type="model", content="x") is None


def test_write_failure_deactivates_rather_than_raising(root, monkeypatch):
    writer = make(root)
    assert writer.append(run_id="r", kind="primary", type="model", content="a") == 1

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_bytes", boom)
    assert writer.append(run_id="r", kind="primary", type="model", content="b") is None
    assert writer.is_active is False


def test_config_can_disable_logging_entirely(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module, "_setting", lambda key, default: False if key == "run_log_enabled" else default
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is False
    assert not (root / "agent-runs").exists()


def test_config_overrides_the_directory_name(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: "my-logs" if key == "run_log_dir_name" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert (root / "my-logs" / "run-abc").is_dir()


def test_write_manifest_emits_readable_json(root):
    writer = make(root)
    writer.write_manifest({"run_id": "run-abc", "status": "done"})
    import json

    manifest = json.loads((root / "agent-runs" / "run-abc" / "MANIFEST").read_text())
    assert manifest["status"] == "done"
    assert manifest["segments"] == []  # nothing appended yet


def test_manifest_records_segments_after_appends(root):
    writer = make(root, segment_bytes=400)
    for _ in range(6):
        writer.append(run_id="r", kind="primary", type="model", content="x" * 100)
    writer.write_manifest({"status": "done"})
    import json

    manifest = json.loads((root / "agent-runs" / "run-abc" / "MANIFEST").read_text())
    assert len(manifest["segments"]) > 1
    assert manifest["record_count"] == 6


def test_manifest_failure_never_raises(root, monkeypatch):
    writer = make(root)
    monkeypatch.setattr(writer, "_write_bytes", lambda *a, **k: (_ for _ in ()).throw(OSError))
    writer.write_manifest({"status": "done"})  # must not raise


def test_close_is_safe_to_call_twice_and_on_an_inactive_writer(root):
    writer = make(root)
    writer.close()
    writer.close()
    RunLogWriter().close()


def test_concurrent_appends_produce_unique_numbers_and_no_corruption(root):
    import threading

    writer = make(root)

    def worker(index):
        for _ in range(20):
            writer.append(
                run_id=f"r{index}", kind="primary", type="model", content=f"payload-{index}"
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = []
    for segment in sorted((root / "agent-runs" / "run-abc").glob("logs.*.txt")):
        records.extend(iter_records(segment.read_bytes()))
    numbers = sorted(r.number for r in records)
    assert numbers == list(range(1, 81))


def test_non_numeric_segment_bytes_uses_default(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: "not-a-number" if key == "run_log_segment_bytes" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is True
    assert writer._segment_bytes == 4_000_000  # default


def test_negative_max_record_bytes_uses_default(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: -999 if key == "run_log_max_record_bytes" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is True
    assert writer._max_record_bytes == 1_000_000  # default


def test_bind_idempotent_after_failed_first_bind(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module, "_setting", lambda key, default: False if key == "run_log_enabled" else default
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is False
    assert writer.log_dir is None

    # Second bind with different run_id must not activate or create directory
    monkeypatch.setattr(
        run_log_module, "_setting", lambda key, default: True if key == "run_log_enabled" else default
    )
    writer.bind("run-other")
    assert writer.is_active is False
    assert writer.log_dir is None
    assert not (root / "agent-runs" / "run-other").exists()


def test_path_traversal_with_dotdot_is_rejected(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: "../escape" if key == "run_log_dir_name" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is False
    assert writer.log_dir is None
    assert not (root / "escape").exists()
    assert not (root / "agent-runs").exists()


def test_real_resolve_log_root_prefers_workspace_over_sandbox(monkeypatch):
    """Test real resolve_log_root() prefers workspace folder over sandbox root."""
    from pathlib import Path
    from unittest.mock import MagicMock

    tmp_sandbox = Path("/tmp/sandbox")
    tmp_workspace = Path("/tmp/workspace")

    def mock_tool_sandbox_root():
        return tmp_sandbox

    def mock_allowed_file_roots(write=False, sandbox_root=None):
        # Return (sandbox, workspace) tuple; resolve_log_root should prefer workspace
        return [tmp_sandbox, tmp_workspace]

    # Stub the imports at the point resolve_log_root uses them
    import tldw_chatbook.Tools.file_operation_tools as file_tools

    monkeypatch.setattr(file_tools, "_tool_sandbox_root", mock_tool_sandbox_root)

    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    monkeypatch.setattr(ws_roots, "allowed_file_roots", mock_allowed_file_roots)

    result = run_log_module.resolve_log_root()
    assert result == tmp_workspace  # Prefers workspace over sandbox


def test_resolve_log_root_returns_none_on_exception(monkeypatch):
    """Test that resolve_log_root returns None (logging off) when resolution raises."""
    monkeypatch.setattr(
        run_log_module, "resolve_log_root", lambda: None
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is False
    assert writer.log_dir is None
