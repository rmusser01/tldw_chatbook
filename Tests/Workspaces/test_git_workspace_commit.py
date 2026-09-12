"""Commit engine for change-review git modes (TASK-16801 arc B, T3).

Every test drives REAL git in a temp repo -- the engine has no mockable
seam by design (spec
`Docs/superpowers/specs/2026-08-20-console-review-git-modes-design.md`,
AC #5, no mocked git).

`test_prestaged_unrelated_entry_survives` is THE regression pin for the
arc's canonical would-have-shipped bug: a bare `git commit -m` would sweep
in whatever the user had already staged in a terminal. The commit recipe
here is `git add -A -- <selected>` then `git commit -m <msg> -- <selected>`
(a PATHSPEC commit) -- verified empirically (spec §2 probe 1) to commit
EXACTLY the selected paths and leave an unrelated pre-staged index entry
staged and uncommitted.
"""
import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Workspaces.git_workspace import (
    CommitRefusedError,
    CommitResult,
    GitStepOutcome,
    GitWorkspaceError,
    commit_selected,
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


def test_prestaged_unrelated_entry_survives(repo):
    # spec §2 probe 1 -- THE regression pin for the index-hijack trap.
    (repo / "keep.txt").write_text("user-staged\n")
    _git(repo, "add", "keep.txt")
    (repo / "a.txt").write_text("agent\n")
    result = commit_selected(
        repo, ["a.txt"], "agent work", None, run_active=lambda: False
    )
    assert isinstance(result, CommitResult)
    assert result.short_sha
    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert committed == ["a.txt"]
    # The pre-staged, unrelated file must survive staged and UNCOMMITTED --
    # a bare `git commit -m` would have swept it in.
    assert _git(repo, "status", "--porcelain") == "A  keep.txt"


def test_run_active_refuses_before_touching(repo):
    (repo / "a.txt").write_text("x\n")
    with pytest.raises(CommitRefusedError):
        commit_selected(repo, ["a.txt"], "m", None, run_active=lambda: True)
    assert "a.txt" not in _git(repo, "diff", "--cached", "--name-only")
    # Nothing was staged and no commit happened -- the edit is still
    # unstaged (visible in the worktree diff, absent from the index).
    assert "a.txt" in _git(repo, "diff", "--name-only")
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"


def test_new_branch_created_then_committed(repo):
    (repo / "a.txt").write_text("x\n")
    result = commit_selected(
        repo, ["a.txt"], "m", "feat/xyz", run_active=lambda: False
    )
    assert result.short_sha
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "feat/xyz"
    # The commit landed on the NEW branch, not main.
    assert _git(repo, "log", "-1", "--format=%s") == "m"


def test_bad_branch_name_refused_preflight(repo):
    (repo / "a.txt").write_text("x\n")
    result = commit_selected(
        repo, ["a.txt"], "m", "-bad", run_active=lambda: False
    )
    assert result.short_sha is None
    assert result.outcomes[0].step == "validate-branch" and not result.outcomes[0].ok
    # nothing staged, no commit happened, still on main
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"
    assert "a.txt" in _git(repo, "diff", "--name-only")
    assert "a.txt" not in _git(repo, "diff", "--cached", "--name-only")
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "main"


def test_existing_branch_stops_before_commit(repo):
    # `checkout -b main` fails (branch already exists) -- no new commit,
    # nothing staged.
    (repo / "a.txt").write_text("x\n")
    result = commit_selected(
        repo, ["a.txt"], "m", "main", run_active=lambda: False
    )
    assert result.short_sha is None
    failing = [o for o in result.outcomes if not o.ok]
    assert len(failing) == 1
    assert failing[0].step == "create-branch"
    assert failing[0].detail  # honest stderr excerpt, never blank
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"
    assert "a.txt" in _git(repo, "diff", "--name-only")
    assert "a.txt" not in _git(repo, "diff", "--cached", "--name-only")


def test_merge_in_progress_refused(repo):
    # Build a REAL conflicted merge: two branches diverge from `base`,
    # editing the same line of a.txt, so the merge stops with conflict
    # markers and MERGE_HEAD present -- never hand-write MERGE_HEAD.
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "a.txt").write_text("feature change\n")
    _git(repo, "commit", "-qam", "feature change")
    _git(repo, "checkout", "-q", "main")
    (repo / "a.txt").write_text("main change\n")
    _git(repo, "commit", "-qam", "main change")
    merge_proc = subprocess.run(
        ["git", "merge", "feature"], cwd=repo, capture_output=True, text=True
    )
    assert merge_proc.returncode != 0  # genuine conflict, not a fast-forward
    assert (repo / ".git" / "MERGE_HEAD").exists()

    result = commit_selected(
        repo, ["a.txt"], "m", None, run_active=lambda: False
    )
    assert result.short_sha is None
    assert result.outcomes[0].step == "in-progress-check" and not result.outcomes[0].ok
    assert "merge" in result.outcomes[0].detail
    # Nothing new was staged/committed by the refused attempt.
    assert _git(repo, "rev-list", "--count", "HEAD") == "2"


def test_dash_leading_message_commits_literally(repo):
    # spec §2 probe 5 -- `-m`'s sticky-arg consumption makes a dash-leading
    # message safe as a plain argv element (never option-injection).
    (repo / "a.txt").write_text("x\n")
    result = commit_selected(
        repo, ["a.txt"], "--amend", None, run_active=lambda: False
    )
    assert result.short_sha
    assert _git(repo, "log", "-1", "--format=%s") == "--amend"
    assert _git(repo, "rev-list", "--count", "HEAD") == "2"  # a NEW commit, not an amend


def test_deletion_only_selection_commits(repo):
    (repo / "a.txt").unlink()
    result = commit_selected(
        repo, ["a.txt"], "remove a", None, run_active=lambda: False
    )
    assert result.short_sha
    tracked = _git(repo, "ls-tree", "-r", "HEAD", "--name-only").splitlines()
    assert "a.txt" not in tracked
    assert _git(repo, "status", "--porcelain") == ""


def test_unborn_first_commit_works(tmp_path):
    # spec §2 probe 4 -- the pathspec commit works as the first commit on
    # a fresh, unborn-HEAD repo.
    root = tmp_path / "fresh"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("x\n")
    result = commit_selected(
        root, ["a.txt"], "first commit", None, run_active=lambda: False
    )
    assert result.short_sha
    assert _git(root, "rev-list", "--count", "HEAD") == "1"
    assert _git(root, "show", "--name-only", "--format=", "HEAD").strip() == "a.txt"


def test_empty_files_refused(repo):
    with pytest.raises(GitWorkspaceError):
        commit_selected(repo, [], "m", None, run_active=lambda: False)
    assert _git(repo, "status", "--porcelain") == ""


def test_blank_message_refused(repo):
    (repo / "a.txt").write_text("x\n")
    with pytest.raises(GitWorkspaceError):
        commit_selected(repo, ["a.txt"], "   ", None, run_active=lambda: False)
    assert "a.txt" in _git(repo, "diff", "--name-only")
    assert "a.txt" not in _git(repo, "diff", "--cached", "--name-only")
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"


def test_pathspec_magic_in_a_filename_cannot_hijack_the_index(repo):
    """`--` blocks OPTIONS; it does not block PATHSPEC MAGIC.

    A file literally named `:!nothing` is a legal filename, `git status`
    lists it verbatim, and the UI checkbox carries it verbatim. Pre-fix,
    passing it after `--` made git read it as the exclude pathspec
    "everything except paths matching `nothing`" -- so a ONE-file
    selection swept a.txt, b.txt and c.txt into the commit. That is the
    canonical index-hijack bug reached through a FILENAME, and the fix is
    `GIT_LITERAL_PATHSPECS=1` in `_user_git_env`.
    """
    (repo / "b.txt").write_text("b\n")
    (repo / "c.txt").write_text("c\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "more files")
    (repo / ":!nothing").write_text("hostile\n")
    (repo / "a.txt").write_text("a2\n")
    (repo / "b.txt").write_text("b2\n")
    (repo / "c.txt").write_text("c2\n")

    result = commit_selected(
        repo, [":!nothing"], "just the one file", None, run_active=lambda: False
    )

    assert result.short_sha
    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert committed == [":!nothing"], (
        f"a 1-file selection committed {len(committed)} files: {committed!r}"
    )
    # The three unselected files are still dirty and uncommitted.
    dirty = sorted(
        line.split()[-1] for line in _git(repo, "status", "--porcelain").splitlines()
    )
    assert dirty == ["a.txt", "b.txt", "c.txt"], dirty


def test_a_normal_path_is_unaffected_by_a_pathspec_magic_filename(repo):
    """The other half: selecting a normal file must not sweep the magic one."""
    (repo / ":!nothing").write_text("hostile\n")
    (repo / "a.txt").write_text("a2\n")

    result = commit_selected(
        repo, ["a.txt"], "only a", None, run_active=lambda: False
    )

    assert result.short_sha
    assert _git(repo, "show", "--name-only", "--format=", "HEAD").splitlines() == [
        "a.txt"
    ]
    assert _git(repo, "status", "--porcelain") == "?? :!nothing"


def test_literal_pathspecs_do_not_regress_spaced_and_utf8_paths(repo):
    """`GIT_LITERAL_PATHSPECS=1` must not break ordinary selections."""
    (repo / "sub dir").mkdir()
    (repo / "sub dir" / "spaced file.txt").write_text("x\n")
    (repo / "ünïcode–π.txt").write_text("y\n")
    (repo / "plain.txt").write_text("z\n")
    selection = ["sub dir/spaced file.txt", "ünïcode–π.txt", "plain.txt"]

    result = commit_selected(
        repo, selection, "mixed names", None, run_active=lambda: False
    )

    assert result.short_sha
    committed = _git(
        repo, "-c", "core.quotePath=false", "show", "--name-only", "--format=", "HEAD"
    ).splitlines()
    assert sorted(committed) == sorted(selection), committed


# ---------------------------------------------------------------------------
# Paths absent from the WORKTREE: a queued `git mv` rename, a `git rm`
# staged deletion, and a plain unstaged deletion (P2(b)).
#
# The engine shares ONE pathspec between `git add -A --` and
# `git commit --`; `git add` refuses a path present in neither the
# worktree nor the index, so a staged rename or deletion used to dead-end
# at the stage step. The recipe -- verified across all five cases below --
# filters the ADD pathspec to worktree-present paths (`os.path.lexists`,
# never `Path.exists`, or a BROKEN SYMLINK is silently dropped from the
# add) while keeping the FULL pathspec on the commit, and skips the add
# entirely when the filtered list is empty (`git add -A --` with an empty
# pathspec stages the WHOLE TREE).
# ---------------------------------------------------------------------------


def test_queued_rename_commits_as_a_rename(repo):
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    _git(repo, "mv", "a.txt", "renamed.txt")
    (repo / "b.txt").write_text("unrelated edit\n")

    result = commit_selected(
        repo, ["a.txt", "renamed.txt"], "moved it", None, run_active=lambda: False
    )

    assert result.short_sha, [(o.step, o.detail) for o in result.outcomes]
    assert _git(repo, "show", "--name-status", "--format=", "HEAD") == (
        "R100\ta.txt\trenamed.txt"
    )
    # The unselected, unrelated edit stays out of the commit AND unstaged.
    # (`_git` strips, so the porcelain XY column's leading space is gone.)
    assert _git(repo, "status", "--porcelain") == "M b.txt"


def test_staged_deletion_commits(repo):
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    _git(repo, "rm", "-q", "a.txt")

    result = commit_selected(
        repo, ["a.txt"], "removed it", None, run_active=lambda: False
    )

    assert result.short_sha, [(o.step, o.detail) for o in result.outcomes]
    assert _git(repo, "show", "--name-status", "--format=", "HEAD") == "D\ta.txt"
    assert _git(repo, "status", "--porcelain") == ""


def test_staged_deletion_alongside_a_modified_file_commits_both(repo):
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    _git(repo, "rm", "-q", "a.txt")
    (repo / "b.txt").write_text("edited\n")

    result = commit_selected(
        repo, ["a.txt", "b.txt"], "both", None, run_active=lambda: False
    )

    assert result.short_sha, [(o.step, o.detail) for o in result.outcomes]
    assert _git(repo, "show", "--name-status", "--format=", "HEAD").splitlines() == [
        "D\ta.txt",
        "M\tb.txt",
    ]
    assert _git(repo, "status", "--porcelain") == ""


def test_a_selection_entirely_absent_from_the_worktree_skips_the_add_step(repo):
    """The trap: `git add -A --` with an EMPTY pathspec stages the WHOLE tree."""
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    _git(repo, "rm", "-q", "a.txt")
    (repo / "unrelated.txt").write_text("must not be swept in\n")

    result = commit_selected(
        repo, ["a.txt"], "only the deletion", None, run_active=lambda: False
    )

    assert result.short_sha
    assert _git(repo, "show", "--name-only", "--format=", "HEAD").splitlines() == [
        "a.txt"
    ]
    assert "stage" not in [o.step for o in result.outcomes], (
        "the add step must be SKIPPED, not run with an empty pathspec; got "
        f"{[o.step for o in result.outcomes]!r}"
    )
    assert _git(repo, "status", "--porcelain") == "?? unrelated.txt"


def test_a_broken_symlink_is_still_staged(repo):
    """`os.path.lexists`, not `Path.exists` -- a broken symlink IS present."""
    (repo / "brokenlink").symlink_to("/definitely/not/here")

    result = commit_selected(
        repo, ["brokenlink"], "add the link", None, run_active=lambda: False
    )

    assert result.short_sha, [(o.step, o.detail) for o in result.outcomes]
    assert _git(repo, "show", "--name-status", "--format=", "HEAD") == "A\tbrokenlink"
    assert _git(repo, "status", "--porcelain") == ""


def test_commit_result_and_step_outcome_are_frozen_dataclasses():
    outcome = GitStepOutcome(step="stage", ok=True)
    with pytest.raises(Exception):
        outcome.ok = False  # type: ignore[misc]
    result = CommitResult(outcomes=(outcome,), short_sha="abc1234")
    with pytest.raises(Exception):
        result.short_sha = None  # type: ignore[misc]
