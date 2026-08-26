"""Working-tree status/diff/untracked-preview engine (TASK-16801 arc B, T2).

Every test drives REAL git in a temp repo -- the engine has no mockable
seam by design (spec `Docs/superpowers/specs/2026-08-20-console-review-git-modes-design.md`,
AC #5, no mocked git).
"""
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Workspaces.change_tracking import ChangedFile
from tldw_chatbook.Workspaces.git_workspace import (
    CurrentRootStatus,
    detect_git_workspace,
    untracked_preview,
    working_tree_diff,
    working_tree_status,
)


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("base\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    return root


def test_status_lists_modified_added_deleted(repo):
    (repo / "a.txt").write_text("edit\n")  # M
    (repo / "new.txt").write_text("x\n")  # ?? -> A + untracked
    (repo / "gone.txt").write_text("y\n")
    _git(repo, "add", "gone.txt")
    _git(repo, "commit", "-qm", "add gone")
    (repo / "gone.txt").unlink()  # D
    info = detect_git_workspace(repo)
    status = working_tree_status(repo, info)
    assert isinstance(status, CurrentRootStatus)
    assert status.info is info
    by_path = {f.path: f for f in status.files}
    assert isinstance(by_path["a.txt"], ChangedFile)
    assert by_path["a.txt"].status == "M" and by_path["a.txt"].adds == 1
    assert by_path["new.txt"].status == "A" and "new.txt" in status.untracked
    assert by_path["gone.txt"].status == "D"
    assert "a.txt" not in status.untracked
    assert "gone.txt" not in status.untracked


def test_staged_rename_new_then_old(repo):
    _git(repo, "mv", "a.txt", "b.txt")
    status = working_tree_status(repo, detect_git_workspace(repo))
    row = {f.path: f for f in status.files}["b.txt"]
    assert row.status == "R" and row.old_path == "a.txt"


def test_untracked_directory_lists_per_file(repo):
    (repo / "sub").mkdir()
    (repo / "sub" / "x.txt").write_text("x\n")
    status = working_tree_status(repo, detect_git_workspace(repo))
    assert "sub/x.txt" in {f.path for f in status.files}  # -uall pin
    assert "sub/x.txt" in status.untracked


def test_added_then_deleted_collapses_to_D(repo):
    (repo / "tmp.txt").write_text("x\n")
    _git(repo, "add", "tmp.txt")
    (repo / "tmp.txt").unlink()  # XY == "AD"
    status = working_tree_status(repo, detect_git_workspace(repo))
    assert {f.path: f for f in status.files}["tmp.txt"].status == "D"


def test_path_with_spaces_and_utf8(repo):
    name = "wei rd ü.txt"  # "wei rd ü.txt"
    (repo / name).write_text("hello\n")
    status = working_tree_status(repo, detect_git_workspace(repo))
    assert name in status.untracked
    by_path = {f.path: f for f in status.files}
    assert by_path[name].status == "A"
    # The preview path and diff path must round-trip the same way.
    preview = untracked_preview(repo, name, max_lines=10)
    assert preview.splitlines()[0] == f"new file: {name}"


def test_unborn_repo_all_files_untracked_no_numstat(tmp_path):
    # fresh init + one file; working_tree_status must not run `diff HEAD`
    # (spec §2 probe 4) -- assert adds/dels are 0 and path in untracked.
    root = tmp_path / "fresh"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "new.txt").write_text("x\n")
    info = detect_git_workspace(root)
    assert info.unborn
    status = working_tree_status(root, info)
    assert "new.txt" in status.untracked
    by_path = {f.path: f for f in status.files}
    assert by_path["new.txt"].status == "A"
    assert by_path["new.txt"].adds == 0
    assert by_path["new.txt"].dels == 0
    assert by_path["new.txt"].binary is False


def test_untracked_preview_text_bounded_and_binary(repo):
    (repo / "t.txt").write_text("\n".join(str(i) for i in range(100)) + "\n")
    text = untracked_preview(repo, "t.txt", max_lines=10)
    assert text.splitlines()[0].startswith("new file: t.txt")
    assert sum(1 for line in text.splitlines() if line.startswith("+")) == 10
    assert "truncated" in text
    (repo / "b.bin").write_bytes(b"\x00\x01\x02")
    assert "binary" in untracked_preview(repo, "b.bin", max_lines=10)


def test_untracked_preview_short_file_not_truncated(repo):
    (repo / "short.txt").write_text("one\ntwo\n")
    text = untracked_preview(repo, "short.txt", max_lines=10)
    assert "truncated" not in text
    assert "+one" in text.splitlines() and "+two" in text.splitlines()


def test_untracked_preview_missing_file_is_honest_not_raising(repo):
    text = untracked_preview(repo, "does-not-exist.txt", max_lines=10)
    assert "does-not-exist.txt" in text
    assert "\n" not in text.strip()  # one-line error text


def test_working_tree_diff_returns_unified(repo):
    (repo / "a.txt").write_text("edit\n")
    assert "-base" in working_tree_diff(repo, "a.txt")


# ---------------------------------------------------------------------------
# The review pane cannot be FAKED by repository configuration.
#
# This is a REVIEW surface: the user decides what to commit from what this
# pane shows. Two repository-supplied diff drivers make it lie, and both
# are reachable by anything that can write `.git/config` -- the same
# precondition the arc already accepted for the push-refspec vector.
# `Tools/git_tool_impls.py` already ports `--no-ext-diff`/`--no-textconv`/
# `--no-color` as one "machine-safe" set for exactly this reason.
# ---------------------------------------------------------------------------


def _real_edit(repo: Path) -> None:
    (repo / "a.txt").write_text("REAL EDIT the user must see\n")


def test_diff_external_cannot_fabricate_the_review_pane(repo, tmp_path):
    """`diff.external` printed `TOTALLY FABRICATED DIFF OUTPUT` pre-fix."""
    fake = tmp_path / "fake.sh"
    fake.write_text("#!/bin/bash\necho 'TOTALLY FABRICATED DIFF OUTPUT'\n")
    fake.chmod(0o755)
    _git(repo, "config", "diff.external", str(fake))
    _real_edit(repo)

    diff = working_tree_diff(repo, "a.txt")

    assert "TOTALLY FABRICATED" not in diff, (
        f"the review pane rendered a repository-supplied fabrication: {diff!r}"
    )
    assert "REAL EDIT the user must see" in diff, (
        f"...and must show the file's real change instead; got {diff!r}"
    )
    assert "-base" in diff, diff


def test_a_textconv_driver_cannot_blank_the_review_pane(repo, tmp_path):
    """A textconv driver rendered NOTHING for a genuinely changed file.

    The nastier half of the pair: the row is still listed as changed with
    real +1/-1 counts (`--numstat` ignores textconv), so the pane reads as
    "this file has no textual difference" for a file that does.
    """
    driver = tmp_path / "tc.sh"
    driver.write_text("#!/bin/bash\necho IDENTICAL\n")
    driver.chmod(0o755)
    _git(repo, "config", "diff.fake.textconv", str(driver))
    (repo / ".gitattributes").write_text("a.txt diff=fake\n")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-qm", "attrs")
    _real_edit(repo)

    diff = working_tree_diff(repo, "a.txt")

    assert diff.strip(), "the review pane was BLANK for a changed file"
    assert "REAL EDIT the user must see" in diff, diff
    assert "IDENTICAL" not in diff, diff

    # The row the pane contradicts: still listed, still counted.
    status = working_tree_status(repo, detect_git_workspace(repo))
    changed = {f.path: (f.adds, f.dels) for f in status.files}
    assert changed["a.txt"] == (1, 1), changed


def test_color_config_cannot_inject_ansi_into_the_review_pane(repo):
    """`color.ui = always` colorizes even a captured (non-tty) diff.

    Third member of the same family, and why the precedent's flag set is
    ported whole rather than two-thirds of it: the pane would render raw
    escape sequences instead of a diff.
    """
    _git(repo, "config", "color.ui", "always")
    _real_edit(repo)

    diff = working_tree_diff(repo, "a.txt")

    assert "\x1b[" not in diff, f"ANSI escapes reached the pane: {diff!r}"
    assert "REAL EDIT the user must see" in diff, diff


def test_machine_safe_flags_do_not_regress_normal_spaced_or_utf8_diffs(repo):
    (repo / "sub dir").mkdir()
    (repo / "sub dir" / "spaced file.txt").write_text("x\n")
    (repo / "ünïcode–π.txt").write_text("y\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "more")
    (repo / "a.txt").write_text("edited\n")
    (repo / "sub dir" / "spaced file.txt").write_text("x2\n")
    (repo / "ünïcode–π.txt").write_text("y2\n")

    assert "+edited" in working_tree_diff(repo, "a.txt")
    assert "+x2" in working_tree_diff(repo, "sub dir/spaced file.txt")
    assert "+y2" in working_tree_diff(repo, "ünïcode–π.txt")


def test_untracked_preview_is_unaffected_by_a_hostile_diff_driver(repo, tmp_path):
    """The untracked path renders from a plain read, never from git."""
    fake = tmp_path / "fake.sh"
    fake.write_text("#!/bin/bash\necho 'TOTALLY FABRICATED DIFF OUTPUT'\n")
    fake.chmod(0o755)
    _git(repo, "config", "diff.external", str(fake))
    (repo / "new.txt").write_text("brand new content\n")

    preview = untracked_preview(repo, "new.txt", 20)

    assert "+brand new content" in preview, preview
    assert "TOTALLY FABRICATED" not in preview, preview


def test_working_tree_diff_is_not_polluted_by_a_pathspec_magic_filename(repo):
    """`git diff HEAD -- ':!nothing'` renders OTHER files' diffs pre-fix.

    Same root cause as the commit-side index hijack: `--` stops option
    parsing, not pathspec MAGIC, and `:!<x>` is an exclude pathspec. The
    diff pane would show a file the user did not click. Fixed by
    `GIT_LITERAL_PATHSPECS=1` in `_user_git_env`.
    """
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    (repo / ":!nothing").write_text("hostile\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "third")
    (repo / "a.txt").write_text("secret edit\n")
    (repo / "b.txt").write_text("another secret edit\n")
    (repo / ":!nothing").write_text("hostile2\n")

    diff = working_tree_diff(repo, ":!nothing")

    headers = [line for line in diff.splitlines() if line.startswith("diff --git")]
    assert len(headers) == 1, (
        f"the diff for ONE file rendered {len(headers)} files: {headers!r}"
    )
    assert "secret edit" not in diff, "another file's content leaked into the pane"
    assert "hostile2" in diff, diff


def test_clean_tree_yields_no_files_no_untracked(repo):
    status = working_tree_status(repo, detect_git_workspace(repo))
    assert status.files == ()
    assert status.untracked == frozenset()


def test_status_root_matches_info_root_for_a_symlinked_caller_path(repo, tmp_path):
    # A caller passing a non-canonical spelling of the root (here, a
    # symlink) must get back a `CurrentRootStatus.root` that agrees with
    # `status.info.root` -- Task 5 keys detection results by root string
    # and Task 6 matches pseudo-rows to roots, so the two fields silently
    # disagreeing (raw caller path vs. GitWorkspaceInfo's `.resolve()`d
    # path) would break both.
    link = tmp_path / "link-to-repo"
    link.symlink_to(repo)
    info = detect_git_workspace(link)
    status = working_tree_status(link, info)
    assert status.root == status.info.root


def test_untracked_preview_refuses_a_symlink_escaping_the_root(repo, tmp_path):
    """Qodo #1: an untracked SYMLINK pointing outside the workspace must not
    have its target's content rendered into the review pane.

    Reachable because agent write tools can create files (and symlinks) in a
    workspace root, and the review pane's text flows back to the model
    through the V1.5 annotate/delivery loop -- so a rendered `~/.ssh/id_rsa`
    is an exfiltration path, not just a display bug.
    """
    secret = tmp_path / "outside_secret.txt"
    secret.write_text("SECRET-OUTSIDE-CONTENT\n")
    (repo / "escape_link").symlink_to(secret)

    # git really does list it as an ordinary untracked entry.
    status = working_tree_status(repo, detect_git_workspace(repo))
    assert "escape_link" in status.untracked

    out = untracked_preview(repo, "escape_link", 20)
    assert "SECRET-OUTSIDE-CONTENT" not in out
    assert "outside the workspace" in out


def test_untracked_preview_still_renders_a_normal_nested_file(repo):
    """The refusal must not swallow legitimate nested paths."""
    (repo / "sub").mkdir()
    (repo / "sub" / "ok.txt").write_text("hello\n")
    out = untracked_preview(repo, "sub/ok.txt", 20)
    assert "+hello" in out
