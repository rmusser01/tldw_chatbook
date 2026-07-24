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


def test_read_write_work_under_dotted_ancestor_root(monkeypatch, tmp_path):
    """Regression test: the real default sandbox root lives under a dotted
    ancestor (``get_user_data_dir()/"tool_sandbox"`` resolves to something
    like ``~/.local/share/tldw_cli/.../tool_sandbox``). validate_path must
    not reject in-sandbox paths just because the sandbox *root* itself sits
    under a dotted directory component -- only a dotted component in the
    user-supplied (relative) portion of the path should be rejected.
    """
    # Simulate the real default: sandbox under a `.local`-style dotted ancestor.
    sandbox = tmp_path / ".local" / "share" / "tldw" / "tool_sandbox"
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())
    sandbox.mkdir(parents=True, exist_ok=True)

    # A normal in-sandbox write must succeed even though the sandbox root
    # itself is nested under dotted directories.
    write_result = asyncio.run(
        fot.WriteFileTool().execute(file_path="note.txt", content="hi there")
    )
    assert "error" not in write_result
    assert (sandbox / "note.txt").read_text() == "hi there"

    # And the corresponding read must succeed too.
    read_result = asyncio.run(fot.ReadFileTool().execute(file_path="note.txt"))
    assert "error" not in read_result
    assert read_result.get("content") == "hi there"

    # A user-supplied path containing a dotted component is STILL rejected --
    # the security property is preserved, only the sandbox's own location is
    # exempted from the hidden-file check.
    bad_result = asyncio.run(fot.ReadFileTool().execute(file_path=".secret"))
    assert "error" in bad_result
