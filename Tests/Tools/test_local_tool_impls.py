import pytest

from tldw_chatbook.Tools.local_tool_impls import (
    LocalToolError,
    list_directory,
    read_file,
    resolve_workspace_path,
)


def test_resolve_workspace_path_confines(tmp_path):
    assert resolve_workspace_path("a/b", tmp_path) == (tmp_path / "a/b").resolve()
    with pytest.raises(LocalToolError, match="outside the workspace root"):
        resolve_workspace_path("../x", tmp_path)


def test_list_directory_shows_dirs_first_then_files(tmp_path):
    # NOTE: Tests/conftest.py's autouse isolate_test_environment fixture
    # creates tmp_path/"test_data", so use a dedicated workspace subdir.
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "zeta.txt").write_text("z")
    (ws / "alpha").mkdir()
    (ws / "alpha" / "inner.txt").write_text("i")
    out = list_directory(".", workspace_root=ws)
    lines = out.splitlines()
    assert lines[0] == "alpha/"
    assert lines[1] == "zeta.txt"


def test_list_directory_caps_entries(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    for i in range(10):
        (ws / f"f{i}.txt").write_text("x")
    out = list_directory(".", workspace_root=ws, max_entries=3)
    assert out.count("\n") + 1 == 4  # 3 entries + truncation notice
    assert "7 more entries" in out


def test_list_directory_rejects_file_and_missing(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    with pytest.raises(LocalToolError, match="not a directory"):
        list_directory("f.txt", workspace_root=tmp_path)
    with pytest.raises(LocalToolError, match="not a directory"):
        list_directory("nope", workspace_root=tmp_path)


def test_fs_read_line_numbered(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.txt").write_text("one\ntwo\nthree\n")
    out = read_file("a.txt", workspace_root=ws)
    assert out.splitlines() == ["1\tone", "2\ttwo", "3\tthree"]


def test_fs_read_offset_limit(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    out = read_file("a.txt", workspace_root=ws, offset=3, limit=2)
    assert out.splitlines() == ["3\tline3", "4\tline4"]


def test_fs_read_offset_past_eof_returns_notice(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.txt").write_text("only\n")
    assert "past end of file" in read_file("a.txt", workspace_root=ws, offset=99)


def test_fs_read_refuses_binary(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    with pytest.raises(LocalToolError, match="binary"):
        read_file("img.png", workspace_root=ws)


def test_fs_read_missing_file(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    with pytest.raises(LocalToolError, match="not found"):
        read_file("nope.txt", workspace_root=ws)
