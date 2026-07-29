# Tests/Agents/test_run_log_resolve_existing.py
"""TASK-870: `resolve_existing_log_dir` -- the read-only counterpart to
`RunLogWriter.bind()` that the Console's "View full log" affordance uses to
check whether a PAST (possibly long-finished) run has a log at all, without
needing that run's own writer instance.
"""

from pathlib import Path

import pytest

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.run_log import RunLogWriter, resolve_existing_log_dir


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Pin log-root resolution to a temp dir, mirroring test_run_log_writer.py."""
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    return tmp_path


def test_returns_none_when_nothing_was_ever_written(root):
    assert resolve_existing_log_dir("run-never-existed") is None


def test_finds_a_real_writer_bound_run(root):
    writer = RunLogWriter()
    writer.bind("run-abc")
    writer.append(run_id="run-abc", kind="primary", type="model", content="hello")

    found = resolve_existing_log_dir("run-abc")

    assert found == root / "agent-runs" / "run-abc"
    assert found.is_dir()


def test_finds_the_dotted_sandbox_fallback_directory_name(root):
    # TASK-870: the writer dots the directory under the sandbox fallback
    # (bind()'s own "Final-review CRITICAL 2"). resolve_existing_log_dir
    # must find a log written under EITHER name without being told which
    # one applies -- simulate the dotted case directly, since this test
    # only cares about directory-name resolution, not root-fallback logic.
    run_dir = root / ".agent-runs" / "run-dotted"
    run_dir.mkdir(parents=True)
    (run_dir / "logs.0001.txt").write_bytes(b"#@# 000001 run=run-dotted kind=primary type=model ts=- bytes=1\nx")

    found = resolve_existing_log_dir("run-dotted")

    assert found == run_dir


def test_returns_none_for_a_directory_with_no_segment_files(root):
    # A run directory can exist (e.g. mkdir raced ahead of the first
    # append) without holding any actual log content yet.
    (root / "agent-runs" / "run-empty").mkdir(parents=True)

    assert resolve_existing_log_dir("run-empty") is None


def test_returns_none_when_no_root_is_resolvable(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)

    assert resolve_existing_log_dir("run-anything") is None


def test_a_different_runs_directory_is_never_matched(root):
    writer = RunLogWriter()
    writer.bind("run-abc")
    writer.append(run_id="run-abc", kind="primary", type="model", content="hello")

    assert resolve_existing_log_dir("run-xyz") is None
