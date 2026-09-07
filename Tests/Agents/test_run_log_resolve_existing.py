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
    # TASK-1270 (merged to origin/dev ahead of this branch's rebase): bind()
    # now dots the log directory UNCONDITIONALLY, not only under the
    # sandbox-fallback case this test originally covered -- so a freshly
    # bound run lands under ".agent-runs", not "agent-runs". See
    # DEFAULT_DIR_NAME's module-level comment for the full history.
    writer = RunLogWriter()
    writer.bind("run-abc")
    writer.append(run_id="run-abc", kind="primary", type="model", content="hello")

    found = resolve_existing_log_dir("run-abc")

    assert found == root / ".agent-runs" / "run-abc"
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


# -- Review finding F: `run_id` is caller-supplied and must be validated as
# a single, safe path component before being joined onto the resolved log
# root -- `RunLogWriter.bind()` already gets this for free via its own
# `is_within(run_dir, root)` containment check; this read-only counterpart
# had no such check at all before this fix. --


def test_rejects_a_run_id_containing_a_path_separator(root):
    assert resolve_existing_log_dir("sub/dir") is None
    assert resolve_existing_log_dir("sub\\dir") is None


def test_rejects_a_traversal_run_id(root):
    assert resolve_existing_log_dir("..") is None
    assert resolve_existing_log_dir(".") is None


def test_rejects_an_empty_or_whitespace_run_id(root):
    assert resolve_existing_log_dir("") is None
    assert resolve_existing_log_dir("   ") is None


def test_rejects_an_absolute_run_id_and_never_escapes_the_root(root):
    # Plant a real, matching "log directory" OUTSIDE root -- pathlib's `/`
    # operator replaces the whole left-hand path when the right-hand side
    # is absolute, so `root / dir_name / run_id` with an absolute `run_id`
    # would (pre-fix) resolve directly to this directory, escaping `root`
    # entirely despite `resolve_existing_log_dir` never being told about
    # it via `resolve_log_root`.
    evil_dir = root.parent / "evil-outside-root"
    evil_dir.mkdir()
    (evil_dir / "logs.0001.txt").write_bytes(
        b"#@# 000001 run=x kind=primary type=model ts=- bytes=1\nx"
    )

    assert resolve_existing_log_dir(str(evil_dir)) is None
