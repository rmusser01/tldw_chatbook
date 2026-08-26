import shutil
import subprocess

import pytest

from tldw_chatbook.Tools.local_tool_impls import LocalToolError
from tldw_chatbook.Tools.patch_tool_impls import (
    FilesystemPatchError,
    PATCH_MAX_BYTES,
    PATCH_MAX_FILES,
    PATCH_MAX_HUNKS,
    parse_patch_targets,
    patch_files,
)

MODIFY_DIFF = """\
--- a/notes.txt
+++ b/notes.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+BETA
 gamma
"""

CREATE_DIFF = """\
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,2 @@
+one
+two
"""


def _ws(tmp_path):
    # Tests/conftest.py's autouse isolate_test_environment fixture creates
    # tmp_path/"test_data", so use a dedicated workspace subdir.
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def test_apply_modify(tmp_path):
    ws = _ws(tmp_path)
    (ws / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    result = patch_files(MODIFY_DIFF, workspace_root=ws)
    assert "patched notes.txt" in result
    assert (ws / "notes.txt").read_text() == "alpha\nBETA\ngamma\n"


def test_apply_create(tmp_path):
    ws = _ws(tmp_path)
    result = patch_files(CREATE_DIFF, workspace_root=ws)
    assert "patched new.txt" in result
    assert (ws / "new.txt").read_text() == "one\ntwo\n"


def test_dry_run_writes_nothing(tmp_path):
    ws = _ws(tmp_path)
    (ws / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    result = patch_files(MODIFY_DIFF, workspace_root=ws, dry_run=True)
    assert "would patch notes.txt" in result
    assert (ws / "notes.txt").read_text() == "alpha\nbeta\ngamma\n"

    result = patch_files(CREATE_DIFF, workspace_root=ws, dry_run=True)
    assert "would patch new.txt" in result
    assert not (ws / "new.txt").exists()


def test_parse_patch_targets_reuses_bounded_parser_and_normalizes_paths():
    plans = parse_patch_targets(MODIFY_DIFF + CREATE_DIFF)

    assert [(plan.action, plan.new_path) for plan in plans] == [
        ("modify", "notes.txt"),
        ("create", "new.txt"),
    ]


@pytest.mark.parametrize(
    ("diff_text", "reason"),
    [
        ("not a diff", "invalid_diff"),
        (
            "--- a/old.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n",
            "delete_not_supported",
        ),
        (
            "--- a/old.txt\n+++ b/new.txt\n@@ -1 +1 @@\n-old\n+new\n",
            "rename_not_supported",
        ),
    ],
)
def test_parse_patch_targets_preserves_parser_reason_codes(diff_text, reason):
    with pytest.raises(FilesystemPatchError) as caught:
        parse_patch_targets(diff_text)

    assert caught.value.reason_code == reason


@pytest.mark.parametrize("dry_run", [False, True])
def test_patch_execution_and_dry_run_use_shared_target_parser(
    monkeypatch, tmp_path, dry_run
):
    import tldw_chatbook.Tools.patch_tool_impls as patch_module

    ws = _ws(tmp_path)
    (ws / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    calls = []
    real_parser = patch_module.parse_patch_targets

    def recording_parser(diff_text):
        calls.append(diff_text)
        return real_parser(diff_text)

    monkeypatch.setattr(patch_module, "parse_patch_targets", recording_parser)

    patch_files(MODIFY_DIFF, workspace_root=ws, dry_run=dry_run)

    assert calls == [MODIFY_DIFF]


def test_context_mismatch(tmp_path):
    ws = _ws(tmp_path)
    (ws / "notes.txt").write_text("alpha\ndelta\ngamma\n")
    with pytest.raises(LocalToolError, match="patch_context_mismatch"):
        patch_files(MODIFY_DIFF, workspace_root=ws)
    assert (ws / "notes.txt").read_text() == "alpha\ndelta\ngamma\n"


def test_delete_and_rename_refused(tmp_path):
    ws = _ws(tmp_path)
    (ws / "old.txt").write_text("x\n")
    delete_diff = """\
--- a/old.txt
+++ /dev/null
@@ -1,1 +0,0 @@
-x
"""
    with pytest.raises(LocalToolError, match="delete_not_supported"):
        patch_files(delete_diff, workspace_root=ws)

    rename_diff = """\
--- a/old.txt
+++ b/renamed.txt
@@ -1,1 +1,1 @@
-x
+y
"""
    with pytest.raises(LocalToolError, match="rename_not_supported"):
        patch_files(rename_diff, workspace_root=ws)
    assert (ws / "old.txt").read_text() == "x\n"
    assert not (ws / "renamed.txt").exists()


def test_malformed_diff(tmp_path):
    ws = _ws(tmp_path)
    with pytest.raises(LocalToolError, match="invalid_diff"):
        patch_files("this is not a diff at all", workspace_root=ws)
    with pytest.raises(LocalToolError, match="invalid_diff"):
        patch_files("", workspace_root=ws)
    # file header with no hunks
    with pytest.raises(LocalToolError, match="invalid_diff"):
        patch_files("--- a/x.txt\n+++ b/x.txt\n", workspace_root=ws)


def test_limits(tmp_path):
    ws = _ws(tmp_path)

    big_line = "+" + ("x" * PATCH_MAX_BYTES)
    big_diff = f"--- /dev/null\n+++ b/big.txt\n@@ -0,0 +1,1 @@\n{big_line}\n"
    with pytest.raises(LocalToolError, match="diff_too_large"):
        patch_files(big_diff, workspace_root=ws)

    many_files = "".join(
        f"--- /dev/null\n+++ b/f{i}.txt\n@@ -0,0 +1,1 @@\n+x\n"
        for i in range(PATCH_MAX_FILES + 1)
    )
    with pytest.raises(LocalToolError, match="diff_file_limit_exceeded"):
        patch_files(many_files, workspace_root=ws)

    many_hunks = "--- a/h.txt\n+++ b/h.txt\n" + "".join(
        "@@ -1,1 +1,1 @@\n x\n" for _ in range(PATCH_MAX_HUNKS + 1)
    )
    with pytest.raises(LocalToolError, match="diff_hunk_limit_exceeded"):
        patch_files(many_hunks, workspace_root=ws)


def test_confinement(tmp_path):
    ws = _ws(tmp_path)
    escape_diff = """\
--- /dev/null
+++ b/../evil.txt
@@ -0,0 +1,1 @@
+pwned
"""
    with pytest.raises(LocalToolError, match="invalid_patch_path"):
        patch_files(escape_diff, workspace_root=ws)
    assert not (tmp_path / "evil.txt").exists()


def test_confinement_symlink_escape(tmp_path):
    ws = _ws(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("alpha\nbeta\ngamma\n")
    (ws / "link.txt").symlink_to(outside)
    link_diff = """\
--- a/link.txt
+++ b/link.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+BETA
 gamma
"""
    # The parser accepts "link.txt"; the wrapper's per-file
    # resolve_workspace_path must refuse the symlink escape.
    with pytest.raises(LocalToolError, match="outside the workspace root"):
        patch_files(link_diff, workspace_root=ws)
    assert outside.read_text() == "alpha\nbeta\ngamma\n"


def test_crlf_preserved(tmp_path):
    ws = _ws(tmp_path)
    (ws / "notes.txt").write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    patch_files(MODIFY_DIFF, workspace_root=ws)
    assert (ws / "notes.txt").read_bytes() == b"alpha\r\nBETA\r\ngamma\r\n"


def test_no_newline_marker(tmp_path):
    ws = _ws(tmp_path)
    (ws / "f.txt").write_bytes(b"alpha\nbeta")
    no_newline_diff = """\
--- a/f.txt
+++ b/f.txt
@@ -1,2 +1,2 @@
 alpha
-beta
\\ No newline at end of file
+BETA
\\ No newline at end of file
"""
    patch_files(no_newline_diff, workspace_root=ws)
    assert (ws / "f.txt").read_bytes() == b"alpha\nBETA"

    bad_marker_diff = """\
--- a/f.txt
+++ b/f.txt
@@ -1,1 +1,1 @@
\\ No newline at end of file
 alpha
"""
    with pytest.raises(LocalToolError, match="invalid_no_newline_marker"):
        patch_files(bad_marker_diff, workspace_root=ws)


def test_hunk_line_count_validation(tmp_path):
    ws = _ws(tmp_path)
    (ws / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    bad_count_diff = """\
--- a/notes.txt
+++ b/notes.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+BETA
"""
    with pytest.raises(LocalToolError, match="invalid_hunk_line_count"):
        patch_files(bad_count_diff, workspace_root=ws)


def test_header_metadata_and_paths_with_spaces(tmp_path):
    ws = _ws(tmp_path)
    (ws / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    (ws / "my file.txt").write_text("one\ntwo\n")

    tab_metadata_diff = (
        "--- a/notes.txt\t2026-01-01 10:00:00.000000000 +0000\n"
        "+++ b/notes.txt\t2026-01-01 11:00:00.000000000 +0000\n"
        "@@ -1,3 +1,3 @@\n alpha\n-beta\n+BETA\n gamma\n"
    )
    patch_files(tab_metadata_diff, workspace_root=ws)
    assert (ws / "notes.txt").read_text() == "alpha\nBETA\ngamma\n"

    space_metadata_diff = (
        "--- a/my file.txt 2026-01-01 10:00:00 +0000\n"
        "+++ b/my file.txt 2026-01-01 11:00:00 +0000\n"
        "@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n"
    )
    patch_files(space_metadata_diff, workspace_root=ws)
    assert (ws / "my file.txt").read_text() == "one\nTWO\n"


def test_multi_file_atomicity_note(tmp_path):
    # Documents behavior, not a goal: files are applied sequentially, so a
    # failing LATER file leaves earlier files patched. The error names the
    # failed file so the model can recover.
    ws = _ws(tmp_path)
    (ws / "a.txt").write_text("alpha\nbeta\ngamma\n")
    (ws / "b.txt").write_text("different\ncontent\n")
    two_file_diff = """\
--- a/a.txt
+++ b/a.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+BETA
 gamma
--- a/b.txt
+++ b/b.txt
@@ -1,2 +1,2 @@
-missing
+found
 context
"""
    with pytest.raises(LocalToolError, match=r"patch_context_mismatch\]: b\.txt"):
        patch_files(two_file_diff, workspace_root=ws)
    assert (ws / "a.txt").read_text() == "alpha\nBETA\ngamma\n"  # stays patched
    assert (ws / "b.txt").read_text() == "different\ncontent\n"


def test_create_requires_missing_target_and_existing_parent(tmp_path):
    ws = _ws(tmp_path)
    (ws / "new.txt").write_text("already here\n")
    with pytest.raises(LocalToolError, match="already exists"):
        patch_files(CREATE_DIFF, workspace_root=ws)
    assert (ws / "new.txt").read_text() == "already here\n"

    no_parent_diff = CREATE_DIFF.replace("new.txt", "no/such/dir/new.txt")
    with pytest.raises(LocalToolError, match="parent directory does not exist"):
        patch_files(no_parent_diff, workspace_root=ws)


def test_modify_requires_existing_utf8_target(tmp_path):
    ws = _ws(tmp_path)
    with pytest.raises(LocalToolError, match="file not found"):
        patch_files(MODIFY_DIFF, workspace_root=ws)

    (ws / "notes.txt").write_bytes(b"\xff\xfe binary-ish")
    with pytest.raises(LocalToolError, match="not valid UTF-8"):
        patch_files(MODIFY_DIFF, workspace_root=ws)
    assert (ws / "notes.txt").read_bytes() == b"\xff\xfe binary-ish"


# --- Critical 1: pure-insertion hunks (@@ -N,0) insert AFTER line N ---------


def test_pure_insertion_hunk_applies_after_line_n(tmp_path):
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text("a\nb\nc\n")
    insert_diff = """\
--- a/f.txt
+++ b/f.txt
@@ -2,0 +3,1 @@
+X
"""
    patch_files(insert_diff, workspace_root=ws)
    assert (ws / "f.txt").read_text() == "a\nb\nX\nc\n"


def test_pure_insertion_at_top(tmp_path):
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text("a\nb\n")
    top_diff = """\
--- a/f.txt
+++ b/f.txt
@@ -0,0 +1,1 @@
+X
"""
    patch_files(top_diff, workspace_root=ws)
    assert (ws / "f.txt").read_text() == "X\na\nb\n"


def test_gnu_u0_diff_applies_byte_exact(tmp_path):
    diff_bin = shutil.which("diff")
    if not diff_bin:
        pytest.skip("diff(1) not available")
    ws = _ws(tmp_path)
    (ws / "a").mkdir()
    (ws / "b").mkdir()
    old = "one\ntwo\nthree\nfour\nfive\n"
    new = "one\ntwo\ninserted\nthree\nCHANGED\nfive\n"
    (ws / "a/f.txt").write_text(old)
    (ws / "b/f.txt").write_text(new)
    proc = subprocess.run(
        [diff_bin, "-U0", "a/f.txt", "b/f.txt"],
        cwd=ws, capture_output=True, text=True,
    )
    assert proc.returncode == 1  # files differ; diff(1) exits 1
    assert "@@ -2,0 +3" in proc.stdout  # pure insertion hunk (`,1` optional)

    (ws / "f.txt").write_text(old)
    patch_files(proc.stdout, workspace_root=ws)
    ours = (ws / "f.txt").read_bytes()
    assert ours == new.encode()

    patch_bin = shutil.which("patch")
    if patch_bin:
        (ws / "ref").mkdir()
        (ws / "ref/f.txt").write_text(old)
        # BSD patch can't parse BSD diff's header timestamps, so pass the
        # target file explicitly; the hunks (what we cross-check) are the same.
        subprocess.run(
            [patch_bin, "-s", "f.txt"],
            cwd=ws / "ref", input=proc.stdout, text=True, check=True,
        )
        assert (ws / "ref/f.txt").read_bytes() == ours


# --- Important 2: real git diff output (inter-file preamble lines) ----------


def _git(repo, *args, **kwargs):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, **kwargs,
    )


def test_real_git_diff_multi_file_applies(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", check=True)
    (repo / "one.txt").write_text("alpha\nbeta\ngamma\n")
    (repo / "two.txt").write_text("red\ngreen\nblue\n")
    _git(repo, "add", "-A", check=True)
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t",
         "-c", "commit.gpgsign=false", "commit", "-qm", "init", check=True)
    (repo / "one.txt").write_text("alpha\nBETA\ngamma\n")
    (repo / "two.txt").write_text("red\ngreen\nblue\nviolet\n")
    (repo / "added.txt").write_text("brand\nnew\n")
    _git(repo, "add", "-A", check=True)
    proc = _git(repo, "diff", "--cached", check=True)
    assert "diff --git" in proc.stdout
    assert "index " in proc.stdout
    assert "new file mode" in proc.stdout

    ws = _ws(tmp_path)
    (ws / "one.txt").write_text("alpha\nbeta\ngamma\n")
    (ws / "two.txt").write_text("red\ngreen\nblue\n")
    result = patch_files(proc.stdout, workspace_root=ws)
    assert "patched one.txt" in result
    assert "patched two.txt" in result
    assert "patched added.txt" in result
    assert (ws / "one.txt").read_bytes() == (repo / "one.txt").read_bytes()
    assert (ws / "two.txt").read_bytes() == (repo / "two.txt").read_bytes()
    assert (ws / "added.txt").read_bytes() == (repo / "added.txt").read_bytes()


def test_garbage_section_still_invalid_diff(tmp_path):
    ws = _ws(tmp_path)
    bad = "--- a/x.txt\n+++ b/x.txt\nthis is not a hunk\n"
    with pytest.raises(LocalToolError, match="invalid_diff"):
        patch_files(bad, workspace_root=ws)


# --- Important 3: removing a line whose content starts with "-- " -----------


def test_removal_of_dash_dash_content(tmp_path):
    ws = _ws(tmp_path)
    (ws / "q.sql").write_text("select 1;\n-- foo\nselect 2;\n")
    sql_diff = """\
--- a/q.sql
+++ b/q.sql
@@ -1,3 +1,2 @@
 select 1;
--- foo
 select 2;
"""
    patch_files(sql_diff, workspace_root=ws)
    assert (ws / "q.sql").read_text() == "select 1;\nselect 2;\n"


# --- Minor 4: BOM-prefixed diff ----------------------------------------------


def test_bom_prefixed_diff(tmp_path):
    ws = _ws(tmp_path)
    (ws / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    patch_files("\ufeff" + MODIFY_DIFF, workspace_root=ws)
    assert (ws / "notes.txt").read_text() == "alpha\nBETA\ngamma\n"
