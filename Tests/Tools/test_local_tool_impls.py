import hashlib
import json
import threading
from pathlib import Path

import pytest

from tldw_chatbook.Tools import local_tool_impls
from tldw_chatbook.Tools.local_tool_impls import (
    MAX_READ_CHARS,
    LocalToolError,
    edit_file,
    glob_files,
    grep_files,
    list_directory,
    read_file,
    resolve_workspace_path,
    write_file,
)
from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion


@pytest.mark.parametrize(
    ("relative", "exclusion", "is_directory", "expected"),
    (
        (".", SensitiveExclusion("file", "."), True, True),
        (".", SensitiveExclusion("subtree", "."), True, True),
        ("child.txt", SensitiveExclusion("direct_children", "."), False, True),
        ("nested/child.txt", SensitiveExclusion("direct_children", "."), False, False),
        ("secret/child.txt", SensitiveExclusion("direct_children", "secret"), False, True),
        ("secret/nested.txt", SensitiveExclusion("file", "secret/nested.txt"), False, True),
        ("secret/nested/child.txt", SensitiveExclusion("file", "secret/nested.txt"), False, False),
        ("secret/nested/child.txt", SensitiveExclusion("subtree", "secret"), False, True),
        ("nested/credentials", SensitiveExclusion("name", "credentials"), False, True),
        ("credentials", SensitiveExclusion("name", "credentials"), True, False),
    ),
)
def test_relative_sensitive_exclusion_matcher_covers_root_and_each_kind(
    relative: str,
    exclusion: SensitiveExclusion,
    is_directory: bool,
    expected: bool,
) -> None:
    assert (
        local_tool_impls._is_relative_sensitive_path(
            Path(relative), (exclusion,), is_directory=is_directory
        )
        is expected
    )


def test_resolve_workspace_path_confines(tmp_path):
    assert resolve_workspace_path("a/b", tmp_path) == (tmp_path / "a/b").resolve()
    with pytest.raises(LocalToolError, match="outside the workspace root"):
        resolve_workspace_path("../x", tmp_path)


def test_stat_path_returns_only_allowlisted_workspace_metadata(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "note.txt"
    target.write_text("hello", encoding="utf-8")

    fields = dict(
        line.split(": ", 1)
        for line in local_tool_impls.stat_path(
            "note.txt", workspace_root=ws
        ).splitlines()
    )

    assert fields.keys() == {"path", "type", "size", "modified_ns", "mode"}
    assert fields["path"] == "note.txt"
    assert fields["type"] == "file"
    assert fields["size"] == "5"
    assert fields["modified_ns"].isdigit()
    assert len(fields["mode"]) == 4


def test_stat_path_uses_the_shared_confinement_and_sensitive_path_choke_point(
    tmp_path, monkeypatch
):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "safe.txt").write_text("safe", encoding="utf-8")

    with pytest.raises(LocalToolError, match="outside the workspace root"):
        local_tool_impls.stat_path("../outside.txt", workspace_root=ws)

    monkeypatch.setattr(
        "tldw_chatbook.Tools.local_tool_impls.is_sensitive_path",
        lambda path, **_kwargs: Path(path).name == "safe.txt",
    )
    with pytest.raises(LocalToolError, match="protected path"):
        local_tool_impls.stat_path("safe.txt", workspace_root=ws)


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


def test_fs_read_keeps_in_root_symlinks_and_refuses_escaping_symlinks(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    inside = ws / "inside.txt"
    inside.write_text("inside\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    import os

    os.symlink(inside, ws / "inside-link.txt")
    os.symlink(outside, ws / "outside-link.txt")

    assert "1\tinside" in read_file("inside-link.txt", workspace_root=ws)
    with pytest.raises(LocalToolError, match="outside the workspace root"):
        read_file("outside-link.txt", workspace_root=ws)


def test_fs_read_empty_file_returns_notice(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "empty.txt").write_text("")
    assert read_file("empty.txt", workspace_root=ws) == "(empty file)"
    # offset-past-EOF on an empty file reports the same notice
    assert read_file("empty.txt", workspace_root=ws, offset=5) == "(empty file)"


def test_fs_read_limit_zero_returns_no_lines(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.txt").write_text("one\ntwo\n")
    assert read_file("a.txt", workspace_root=ws, limit=0) == ""


def test_fs_read_offset_zero_and_negative_clamp_to_first_line(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.txt").write_text("one\ntwo\n")
    first = read_file("a.txt", workspace_root=ws, offset=1)
    assert read_file("a.txt", workspace_root=ws, offset=0) == first
    assert read_file("a.txt", workspace_root=ws, offset=-3) == first


def test_fs_read_truncates_at_max_chars(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "big.txt").write_text("".join(f"line{i} " + "x" * 40 + "\n" for i in range(1000)))
    out = read_file("big.txt", workspace_root=ws)
    assert out.endswith("… [truncated]")
    assert len(out) <= MAX_READ_CHARS + len("\n… [truncated]")


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


def test_fs_write_preserves_existing_file_mode(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "script.sh"
    target.write_text("old")
    target.chmod(0o751)

    write_file("script.sh", "new", workspace_root=ws)

    assert target.read_text() == "new"
    assert target.stat().st_mode & 0o7777 == 0o751


def test_fs_write_dry_run_returns_bounded_exact_state_without_mutation(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "AGENTS.md"
    target.write_text("old\n")

    result = json.loads(
        write_file(
            "AGENTS.md",
            "new\n",
            workspace_root=ws,
            dry_run=True,
        )
    )

    assert result["target_state"] == "present"
    assert result["current_sha256"] == hashlib.sha256(b"old\n").hexdigest()
    assert result["replacement_sha256"] == hashlib.sha256(b"new\n").hexdigest()
    assert result["replacement_bytes"] == 4
    assert "-old" in result["diff"] and "+new" in result["diff"]
    assert target.read_text() == "old\n"


def test_fs_write_expected_digest_refuses_stale_state_without_mutation(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "AGENTS.md"
    target.write_text("user edit")

    with pytest.raises(LocalToolError, match="precondition"):
        write_file(
            "AGENTS.md",
            "replacement",
            workspace_root=ws,
            expected_sha256="0" * 64,
        )

    assert target.read_text() == "user edit"


def test_fs_write_expected_absent_refuses_file_created_after_preview(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "AGENTS.md"
    target.write_text("intervening")

    with pytest.raises(LocalToolError, match="precondition"):
        write_file(
            "AGENTS.md",
            "replacement",
            workspace_root=ws,
            expected_absent=True,
        )

    assert target.read_text() == "intervening"


def test_fs_write_preconditions_are_mutually_exclusive(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(LocalToolError, match="mutually exclusive"):
        write_file(
            "AGENTS.md",
            "replacement",
            workspace_root=ws,
            expected_sha256="0" * 64,
            expected_absent=True,
        )


def test_two_same_expectation_writers_allow_exactly_one(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "AGENTS.md"
    target.write_text("before")
    expected = hashlib.sha256(b"before").hexdigest()
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def writer(content: str) -> None:
        barrier.wait()
        try:
            write_file(
                "AGENTS.md",
                content,
                workspace_root=ws,
                expected_sha256=expected,
            )
        except LocalToolError:
            outcomes.append("stale")
        else:
            outcomes.append("written")

    threads = [
        threading.Thread(target=writer, args=("first",)),
        threading.Thread(target=writer, args=("second",)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["stale", "written"]
    assert target.read_text() in {"first", "second"}


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


def test_fs_edit_rejects_identical_strings(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("alpha")
    with pytest.raises(LocalToolError, match="identical"):
        edit_file("f.txt", "alpha", "alpha", workspace_root=ws)
    assert (ws / "f.txt").read_text() == "alpha"  # untouched


def test_fs_edit_refuses_non_utf8_file(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_bytes(b"\xff\xfe not utf-8")
    with pytest.raises(LocalToolError, match="not valid UTF-8"):
        edit_file("f.txt", "not", "x", workspace_root=ws)


def test_fs_edit_unencodable_new_string_preserves_file(tmp_path):
    # A lone surrogate (reachable via tool-call JSON "ﺀ") must fail
    # BEFORE the file is truncated — the original content stays intact.
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("alpha beta")
    with pytest.raises(LocalToolError, match="UTF-8"):
        edit_file("f.txt", "beta", "\ud800", workspace_root=ws)
    assert (ws / "f.txt").read_text() == "alpha beta"


def test_fs_edit_preserves_crlf_line_endings(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_bytes(b"a\r\nb")
    edit_file("f.txt", "b", "c", workspace_root=ws)
    assert (ws / "f.txt").read_bytes() == b"a\r\nc"


def test_fs_write_unencodable_content_preserves_file(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "f.txt").write_text("keep me")
    with pytest.raises(LocalToolError, match="UTF-8"):
        write_file("f.txt", "lone surrogate: \ud800", workspace_root=ws)
    assert (ws / "f.txt").read_text() == "keep me"


def test_relative_mutation_bodies_use_the_supplied_io_root_and_preserve_bytes(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "f.txt").write_bytes(b"before\r\n")

    edit_result = local_tool_impls._edit_relative_file(
        Path("f.txt"),
        "before",
        "after",
        workspace=ws,
        display_path="f.txt",
    )
    write_result = local_tool_impls._write_relative_file(
        Path("new.txt"),
        "created\n",
        workspace=ws,
        display_path="new.txt",
    )

    assert edit_result == "made 1 replacement in f.txt"
    assert write_result == "wrote 8 characters to new.txt"
    assert (ws / "f.txt").read_bytes() == b"after\r\n"
    assert (ws / "new.txt").read_bytes() == b"created\n"


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


def test_fs_glob_dotdot_reentry_renders_workspace_relative(tmp_path):
    # "../ws/*.py" leaves and re-enters the root; the match is confined but
    # must still render workspace-relative, not as "../ws/a.py".
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "a.py").write_text("x")
    assert glob_files("../ws/*.py", workspace_root=ws) == "a.py"


def test_fs_grep_skips_symlinks_escaping_workspace(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    import os
    outside = tmp_path / "outside.txt"
    outside.write_text("secret hit\n")
    os.symlink(outside, ws / "link.txt")          # escapes the root -> skipped
    inside = ws / "inside.txt"
    inside.write_text("real hit\n")
    os.symlink(inside, ws / "inner_link.txt")     # stays inside root -> still read
    out = grep_files("hit", workspace_root=ws)
    assert "secret" not in out
    assert not any(line.startswith("link.txt:") for line in out.splitlines())
    assert "inner_link.txt:1:real hit" in out


def test_fs_glob_lists_symlinked_files_by_name(tmp_path):
    # Listing a symlink's name is not a confinement violation (no content
    # read); pin current behavior so only grep changes.
    ws = tmp_path / "ws"; ws.mkdir()
    import os
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    os.symlink(outside, ws / "link.txt")
    assert "link.txt" in glob_files("*.txt", workspace_root=ws)


def test_glob_and_grep_reject_nonpositive_max_results(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("hit\n")
    for fn, pattern in ((glob_files, "*.py"), (grep_files, "hit")):
        with pytest.raises(LocalToolError, match="max_results"):
            fn(pattern, workspace_root=ws, max_results=0)
        with pytest.raises(LocalToolError, match="max_results"):
            fn(pattern, workspace_root=ws, max_results=-5)


def test_grep_skips_racy_entries(tmp_path, monkeypatch):
    """An OSError during per-file inspection skips that entry, not the search."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "good.py").write_text("hit\n")
    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self.name == "gone.py":
            raise FileNotFoundError("raced away")
        return real_stat(self, *args, **kwargs)

    (ws / "gone.py").write_text("hit\n")
    monkeypatch.setattr(Path, "stat", flaky_stat)
    out = grep_files("hit", workspace_root=ws)
    assert "good.py:1:hit" in out
