import asyncio
from pathlib import Path

import pytest

from tldw_chatbook.Tools import file_operation_tools as fot


def test_sandbox_root_is_real_dir_not_literal(monkeypatch, tmp_path):
    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(tmp_path / "tool_sandbox"))
    root = fot._tool_sandbox_root()
    assert root == (tmp_path / "tool_sandbox").resolve()
    assert root.is_dir()  # created
    assert root.name != "file" and root.name != "directory"


def test_read_file_rejects_traversal_outside_sandbox(monkeypatch, tmp_path):
    sandbox = tmp_path / "tool_sandbox"
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())
    sandbox.mkdir(parents=True, exist_ok=True)
    # write a secret OUTSIDE the sandbox
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    result = asyncio.run(fot.ReadFileTool().execute(file_path="../secret.txt"))
    assert result.get("success") is False or "error" in result  # rejected, not leaked
    assert "top secret" not in str(result)


def test_read_file_reads_inside_sandbox(monkeypatch, tmp_path):
    sandbox = tmp_path / "tool_sandbox"
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "hello.txt").write_text("inside content")
    result = asyncio.run(fot.ReadFileTool().execute(file_path="hello.txt"))
    assert "inside content" in str(result)
