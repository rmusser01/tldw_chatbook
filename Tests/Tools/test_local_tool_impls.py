import pytest

from tldw_chatbook.Tools.local_tool_impls import (
    LocalToolError,
    edit_file,
    glob_files,
    grep_files,
    list_directory,
    read_file,
    resolve_workspace_path,
    write_file,
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


def test_fs_write_creates_file(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    out = write_file("new.txt", "hello\n", workspace_root=ws)
    assert (ws / "new.txt").read_text() == "hello\n"
    assert "wrote" in out and "new.txt" in out


def test_fs_write_overwrites(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("old")
    write_file("f.txt", "new", workspace_root=ws)
    assert (ws / "f.txt").read_text() == "new"


def test_fs_write_requires_existing_parent(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    with pytest.raises(LocalToolError, match="parent directory"):
        write_file("no/such/dir/f.txt", "x", workspace_root=ws)


def test_fs_write_confined(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    with pytest.raises(LocalToolError, match="outside the workspace root"):
        write_file("../evil.txt", "x", workspace_root=ws)


def test_fs_edit_unique_match(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("alpha beta gamma")
    out = edit_file("f.txt", "beta", "BETA", workspace_root=ws)
    assert (ws / "f.txt").read_text() == "alpha BETA gamma"
    assert "1 replacement" in out


def test_fs_edit_requires_match(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("alpha")
    with pytest.raises(LocalToolError, match="not found"):
        edit_file("f.txt", "zzz", "q", workspace_root=ws)


def test_fs_edit_ambiguous_match_reports_count(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("dup dup dup")
    with pytest.raises(LocalToolError, match="3 times"):
        edit_file("f.txt", "dup", "x", workspace_root=ws)


def test_fs_edit_replace_all(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("dup dup dup")
    out = edit_file("f.txt", "dup", "x", workspace_root=ws, replace_all=True)
    assert (ws / "f.txt").read_text() == "x x x"
    assert "3 replacements" in out


def test_fs_glob_matches_and_sorts_by_mtime(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    import os, time
    old = ws / "old.py"; old.write_text("x")
    new = ws / "new.py"; new.write_text("x")
    (ws / "skip.txt").write_text("x")
    past = time.time() - 100
    os.utime(old, (past, past))
    out = glob_files("*.py", workspace_root=ws)
    assert out.splitlines() == ["new.py", "old.py"]  # newest first


def test_fs_glob_recursive_and_cap(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "sub").mkdir()
    (ws / "sub" / "deep.py").write_text("x")
    assert "sub/deep.py" in glob_files("**/*.py", workspace_root=ws)


def test_fs_grep_line_numbers(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.py").write_text("def foo():\n    return 1\n")
    out = grep_files("def foo", workspace_root=ws)
    assert "a.py:1:def foo():" in out


def test_fs_grep_files_with_matches_and_count(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.py").write_text("hit\nhit\n")
    (ws / "b.py").write_text("hit\n")
    assert set(grep_files("hit", workspace_root=ws, mode="files").splitlines()) == {"a.py", "b.py"}
    assert "a.py:2" in grep_files("hit", workspace_root=ws, mode="count")


def test_fs_grep_caps_output(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "big.py").write_text("hit\n" * 500)
    out = grep_files("hit", workspace_root=ws, max_results=10)
    assert "more, truncated" in out


def test_fs_glob_cannot_escape_workspace_via_dotdot(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (tmp_path / "outside.py").write_text("x")
    (ws / "inner.py").write_text("x")
    out = glob_files("../*.py", workspace_root=ws)
    assert "outside.py" not in out
