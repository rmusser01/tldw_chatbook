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


def test_list_directory_caps_the_scan(tmp_path, monkeypatch):
    """Beyond MAX_SCAN_ENTRIES the scan stops: only the scanned entries are
    sorted/listed (dirs-first still holds for them) and a "directory too
    large" notice is appended."""
    from tldw_chatbook.Tools import local_tool_impls

    monkeypatch.setattr(local_tool_impls, "MAX_SCAN_ENTRIES", 5)
    ws = tmp_path / "ws"
    ws.mkdir()
    for i in range(8):
        (ws / f"f{i}.txt").write_text("x")
    (ws / "adir").mkdir()
    out = list_directory(".", workspace_root=ws)
    lines = out.splitlines()
    assert lines[-1] == "… (directory too large; showing first 5 of many entries)"
    # 5 scanned entries, all within the display cap, no truncation notice.
    assert "more entries" not in out
    assert len(lines) == 6
    # Dirs-first contract preserved within the scanned set.
    scanned_names = lines[:-1]
    dir_positions = [i for i, n in enumerate(scanned_names) if n.endswith("/")]
    assert dir_positions == list(range(len(dir_positions)))
