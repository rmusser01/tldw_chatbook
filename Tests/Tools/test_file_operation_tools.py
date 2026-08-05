"""Characterization tests for the legacy file-operation tools.

Pins the caller-visible contract of ReadFileTool / ListDirectoryTool /
WriteFileTool (return dict shapes, error semantics, parameter names) as
they delegate path confinement to Tools/local_tool_impls.resolve_workspace_path.

NOTE: Tests/conftest.py's autouse isolate_test_environment fixture creates
tmp_path/"test_data", so each test uses a dedicated workspace subdir.
"""

import pytest

from tldw_chatbook.Tools.file_operation_tools import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


# --- ReadFileTool ---------------------------------------------------------


@pytest.mark.asyncio
async def test_read_file_success_dict_shape(ws):
    target = ws / "note.txt"
    target.write_text("line one\nline two\n", encoding="utf-8")
    result = await ReadFileTool(workspace_root=ws).execute(file_path="note.txt")
    assert "error" not in result
    assert result["content"] == "line one\nline two\n"
    assert result["encoding"] == "utf-8"
    assert result["lines"] == 2
    assert result["size_bytes"] == target.stat().st_size
    assert result["file_path"] == str(target.resolve())


@pytest.mark.asyncio
async def test_read_file_no_path():
    result = await ReadFileTool().execute()
    assert result == {"error": "No file path provided"}


@pytest.mark.asyncio
async def test_read_file_not_found(ws):
    result = await ReadFileTool(workspace_root=ws).execute(file_path="nope.txt")
    assert result["error"] == "File not found: nope.txt"
    assert result["absolute_path"] == str((ws / "nope.txt").resolve())


@pytest.mark.asyncio
async def test_read_file_rejects_directory(ws):
    (ws / "subdir").mkdir()
    result = await ReadFileTool(workspace_root=ws).execute(file_path="subdir")
    assert result["error"] == "Path is not a file: subdir"
    assert result["path_type"] == "directory"


@pytest.mark.asyncio
async def test_read_file_decode_error_suggests_encoding(ws):
    (ws / "latin.txt").write_bytes("café".encode("latin-1"))
    result = await ReadFileTool(workspace_root=ws).execute(file_path="latin.txt")
    assert "Unable to decode file with utf-8 encoding" in result["error"]
    assert "suggestion" in result
    # ...and the caller-supplied encoding path still works
    ok = await ReadFileTool(workspace_root=ws).execute(
        file_path="latin.txt", encoding="latin-1"
    )
    assert ok["content"] == "café"


@pytest.mark.asyncio
async def test_read_file_outside_workspace_root_is_an_error_dict(ws):
    outside = ws.parent / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    result = await ReadFileTool(workspace_root=ws).execute(file_path="../secret.txt")
    assert "error" in result
    assert "content" not in result


@pytest.mark.asyncio
async def test_read_file_default_root_is_cwd(ws, monkeypatch):
    (ws / "here.txt").write_text("cwd content", encoding="utf-8")
    monkeypatch.chdir(ws)
    result = await ReadFileTool().execute(file_path="here.txt")
    assert result["content"] == "cwd content"


# --- WriteFileTool --------------------------------------------------------


@pytest.mark.asyncio
async def test_write_file_create_then_overwrite_then_append(ws):
    tool = WriteFileTool(workspace_root=ws)
    created = await tool.execute(file_path="out.txt", content="a\nb\n")
    assert created["action"] == "created"
    assert created["lines_written"] == 2
    assert created["size_bytes"] == 4
    assert created["encoding"] == "utf-8"
    assert created["file_path"] == str((ws / "out.txt").resolve())

    overwritten = await tool.execute(file_path="out.txt", content="z")
    assert overwritten["action"] == "overwritten"
    assert (ws / "out.txt").read_text() == "z"

    appended = await tool.execute(file_path="out.txt", content="zz", mode="append")
    assert appended["action"] == "appended to"
    assert (ws / "out.txt").read_text() == "zzz"


@pytest.mark.asyncio
async def test_write_file_missing_args():
    tool = WriteFileTool()
    assert await tool.execute() == {"error": "No file path provided"}
    assert await tool.execute(file_path="x.txt") == {"error": "No content provided"}


@pytest.mark.asyncio
async def test_write_file_missing_parent_suggests_create_directories(ws):
    result = await WriteFileTool(workspace_root=ws).execute(
        file_path="sub/out.txt", content="x"
    )
    assert "Parent directory does not exist" in result["error"]
    assert result["suggestion"] == "Set create_directories=true to create it"


@pytest.mark.asyncio
async def test_write_file_create_directories(ws):
    result = await WriteFileTool(workspace_root=ws).execute(
        file_path="sub/deep/out.txt", content="x", create_directories=True
    )
    assert result["action"] == "created"
    assert (ws / "sub" / "deep" / "out.txt").read_text() == "x"


@pytest.mark.asyncio
async def test_write_file_outside_workspace_root_writes_nothing(ws):
    result = await WriteFileTool(workspace_root=ws).execute(
        file_path="../escape.txt", content="x"
    )
    assert "error" in result
    assert not (ws.parent / "escape.txt").exists()


# --- ListDirectoryTool ----------------------------------------------------


@pytest.fixture
def listing_ws(ws):
    (ws / "b.txt").write_text("bb", encoding="utf-8")
    (ws / "a_dir").mkdir()
    (ws / "a_dir" / "inner.txt").write_text("i", encoding="utf-8")
    (ws / "a_dir" / "deep").mkdir()
    (ws / "a_dir" / "deep" / "bottom.txt").write_text("d", encoding="utf-8")
    (ws / ".hidden").write_text("h", encoding="utf-8")
    return ws


@pytest.mark.asyncio
async def test_list_directory_flat_shape(listing_ws):
    result = await ListDirectoryTool(workspace_root=listing_ws).execute(
        directory_path="."
    )
    assert "error" not in result
    assert result["total_entries"] == 2
    assert result["file_count"] == 1
    assert result["directory_count"] == 1
    by_name = {e["name"]: e for e in result["entries"]}
    assert by_name["b.txt"] == {
        "name": "b.txt",
        "path": "b.txt",
        "type": "file",
        "size_bytes": 2,
        "depth": 0,
    }
    assert by_name["a_dir"]["type"] == "directory"
    assert by_name["a_dir"]["size_bytes"] is None


@pytest.mark.asyncio
async def test_list_directory_recursive_with_depth(listing_ws):
    result = await ListDirectoryTool(workspace_root=listing_ws).execute(
        directory_path=".", recursive=True, max_depth=2
    )
    paths = {(e["path"], e["depth"]) for e in result["entries"]}
    assert ("a_dir", 0) in paths
    assert ("a_dir/inner.txt", 1) in paths
    assert ("a_dir/deep", 1) in paths
    assert ("a_dir/deep/bottom.txt", 2) in paths


@pytest.mark.asyncio
async def test_list_directory_hidden_opt_in(listing_ws):
    result = await ListDirectoryTool(workspace_root=listing_ws).execute(
        directory_path=".", include_hidden=True
    )
    names = {e["name"] for e in result["entries"]}
    assert ".hidden" in names


@pytest.mark.asyncio
async def test_list_directory_not_found(ws):
    result = await ListDirectoryTool(workspace_root=ws).execute(
        directory_path="nope"
    )
    assert result["error"] == "Directory not found: nope"
    assert result["absolute_path"] == str((ws / "nope").resolve())


@pytest.mark.asyncio
async def test_list_directory_rejects_file(ws):
    (ws / "f.txt").write_text("x", encoding="utf-8")
    result = await ListDirectoryTool(workspace_root=ws).execute(
        directory_path="f.txt"
    )
    assert result["error"] == "Path is not a directory: f.txt"
    assert result["path_type"] == "file"


@pytest.mark.asyncio
async def test_list_directory_outside_workspace_root_is_an_error_dict(ws):
    result = await ListDirectoryTool(workspace_root=ws).execute(
        directory_path=".."
    )
    assert "error" in result
    assert "entries" not in result
