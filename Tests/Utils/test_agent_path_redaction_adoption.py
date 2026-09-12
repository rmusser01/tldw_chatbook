"""TASK-19558: model-supplied paths are not echoed into the log.

`Utils/path_validation.validate_path` prints the full user path AND its
resolved form at WARNING/ERROR unless `redact_paths=True` is passed, and at
this task's branch base only 12 of its 30 call sites passed it. Two of the
misses were on the paths that are not the user's at all -- they are the
model's:

* `Tools/local_tool_impls._resolve_in_workspace`, the choke point for the
  ADR-032 local tool family (`fs_read`/`fs_list`/`fs_glob`/`fs_grep`/
  `fs_write`/...).
* `Utils/path_validation.validate_path_multi`, the choke point for the
  in-process builtin family (`ReadFileTool`/`WriteFileTool`/
  `ListDirectoryTool`/`EditFileTool`, via `Tools/file_operation_tools.py`) --
  and worse, it calls `validate_path` once PER ROOT, so one probe wrote the
  path several times.

Both are prompt-injection-reachable: text in a fetched page or an ingested
document can make the model attempt `../../.ssh/id_rsa`, and the log line
then carries the attacker's string plus the real filesystem layout into the
diagnostics bundle a user may later share.

What redaction does NOT change is asserted here too: the refusal the MODEL
receives is built by the tool layer from its own argument and still names
the path, because that message is the model's recovery route.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.Tools.local_tool_impls import LocalToolError, read_file
from tldw_chatbook.Utils.path_validation import validate_path, validate_path_multi

# Priming import, not a dependency of these assertions. On the SUCCESS path
# `local_tool_impls` -> `sensitive_paths.resolve_sensitive_context` lazily
# imports `RAG_Search.config_profiles`, and that package's `__init__` has a
# circular edge (`config_profiles` -> `simplified/__init__` ->
# `enhanced_rag_service_v2` -> back into `config_profiles`) that only raises
# when this test module is the first thing in the process to walk it.
# Pre-existing and unrelated to TASK-19558; importing it up front here keeps
# that unrelated fragility from masquerading as a redaction failure.
# Measured 2026-08-23: dev has since fixed the cycle itself (TASK-21160,
# `ae018308b`), and these seven tests pass without this line on a tree that
# has it -- so DROP this import when the branch lands on a dev containing
# that fix. It is still required at this branch's own base.
# The cycle only resolves when `simplified` is entered first, so this line
# must come before any import of `config_profiles`.
import tldw_chatbook.RAG_Search.simplified  # noqa: E402,F401  isort:skip

SECRET_LEAF = "t19558-secret-leaf-name"


@pytest.fixture()
def captured_logs():
    records: list[str] = []
    sink_id = logger.add(records.append, level="DEBUG", format="{message}")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def test_validate_path_without_redaction_still_echoes_the_path(
    tmp_path: Path, captured_logs: list[str]
) -> None:
    """The mechanism, pinned: this is what the un-swept call sites do."""
    with pytest.raises(ValueError):
        validate_path(f"../{SECRET_LEAF}", tmp_path)
    assert any(SECRET_LEAF in line for line in captured_logs)


def test_local_tool_path_refusal_does_not_log_the_model_supplied_path(
    tmp_path: Path, captured_logs: list[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(LocalToolError):
        read_file(f"../{SECRET_LEAF}", workspace_root=workspace)
    assert not any(SECRET_LEAF in line for line in captured_logs), captured_logs


def test_local_tool_refusal_still_tells_the_model_which_path_it_named(
    tmp_path: Path,
) -> None:
    """Redaction bounds the LOG, not the model-facing recovery message."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(LocalToolError) as excinfo:
        read_file(f"../{SECRET_LEAF}", workspace_root=workspace)
    assert SECRET_LEAF in str(excinfo.value)


def test_multi_root_validation_does_not_log_the_model_supplied_path(
    tmp_path: Path, captured_logs: list[str]
) -> None:
    """The builtin file-tool choke point, which logged once per root."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    with pytest.raises(ValueError):
        validate_path_multi(f"../{SECRET_LEAF}", [root_a, root_b])
    assert not any(SECRET_LEAF in line for line in captured_logs), captured_logs


def test_multi_root_refusal_still_names_the_path_and_the_consulted_roots(
    tmp_path: Path,
) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    with pytest.raises(ValueError) as excinfo:
        validate_path_multi(f"../{SECRET_LEAF}", [root_a, root_b])
    message = str(excinfo.value)
    assert SECRET_LEAF in message
    assert str(root_a.resolve()) in message and str(root_b.resolve()) in message


def test_multi_root_validation_still_accepts_a_path_under_a_later_root(
    tmp_path: Path,
) -> None:
    """First-match-wins behaviour is unchanged by the redaction flag."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    target = root_b / "file.txt"
    target.write_text("x")
    assert validate_path_multi(target, [root_a, root_b]) == target.resolve()


def test_local_tool_read_still_works_inside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("hello\n")
    assert "hello" in read_file("note.txt", workspace_root=workspace)
