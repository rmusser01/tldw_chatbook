import pytest

from tldw_chatbook.Tools.local_tool_impls import (
    LocalToolError,
    list_directory,
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
