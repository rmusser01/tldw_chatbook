"""TASK-28238 phase 2: agent worktree lifecycle."""

import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Agents.agent_worktree import (
    AgentWorktree,
    WorktreeRefusal,
    create_agent_worktree,
    discard_agent_worktree,
    prune_stale_agent_worktrees,
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "a.txt").write_text("base\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "base")
    return root


def test_create_yields_isolated_checkout_at_head(repo):
    wt = create_agent_worktree(repo, "run-abc12345")
    assert isinstance(wt, AgentWorktree), getattr(wt, "message", wt)
    assert wt.worktree_path.is_dir()
    assert (wt.worktree_path / "a.txt").read_text() == "base\n"
    assert wt.branch == "run-abc12345" or wt.branch.endswith("run-abc12345")
    # a write in the worktree is invisible in the shared tree
    (wt.worktree_path / "a.txt").write_text("child change\n")
    assert (repo / "a.txt").read_text() == "base\n"
    discard_agent_worktree(repo, wt)


def test_create_refuses_non_git_root(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    refusal = create_agent_worktree(plain, "run-x")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "not_a_git_repo"


def test_discard_removes_worktree_and_branch(repo):
    wt = create_agent_worktree(repo, "run-gone1234")
    assert isinstance(wt, AgentWorktree)
    assert discard_agent_worktree(repo, wt) is None
    assert not wt.worktree_path.exists()
    branches = _git(repo, "branch", "--list", wt.branch)
    assert wt.branch not in branches


def test_uncommitted_shared_changes_do_not_carry(repo):
    (repo / "a.txt").write_text("dirty uncommitted\n")
    wt = create_agent_worktree(repo, "run-clean555")
    assert isinstance(wt, AgentWorktree)
    # clean checkout of HEAD, not the dirty tree (spec decision)
    assert (wt.worktree_path / "a.txt").read_text() == "base\n"
    discard_agent_worktree(repo, wt)


# -- I3 (TASK-28238 P2 T7 final fix wave): force=False GC-safe discard ----
#
# `discard_agent_worktree`'s force=True default is today's explicit-
# discard-tool behavior (`worktree remove --force`, `branch -D`) --
# unchanged. `prune_stale_agent_worktrees` (implicit GC) passes force=False:
# a plain `git worktree remove` refuses a DIRTY worktree instead of
# destroying it, and a successful removal's branch delete uses lowercase
# `-d`, which refuses an unmerged branch -- unlike the explicit tool, GC
# must never destroy work the user never confirmed discarding.


def test_discard_force_false_removes_clean_worktree_and_branch(repo):
    wt = create_agent_worktree(repo, "run-forcefls")
    assert discard_agent_worktree(repo, wt, force=False) is None
    assert not wt.worktree_path.exists()
    branches = _git(repo, "branch", "--list", wt.branch)
    assert wt.branch not in branches


def test_discard_force_false_refuses_a_dirty_worktree(repo):
    wt = create_agent_worktree(repo, "run-dirtyone")
    (wt.worktree_path / "a.txt").write_text("uncommitted child change\n")
    refusal = discard_agent_worktree(repo, wt, force=False)
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "worktree_remove_failed"
    # Left on disk untouched, exactly as the discard tool's own schema
    # promises for the "never merged/discarded" case.
    assert wt.worktree_path.is_dir()
    assert (wt.worktree_path / "a.txt").read_text() == "uncommitted child change\n"
    discard_agent_worktree(repo, wt)  # force=True cleanup for the test itself


def test_discard_force_false_leaves_an_unmerged_branch_ref_behind(repo):
    """A CLEAN worktree (removable without --force) whose branch carries a
    commit not merged into the shared tree's checked-out branch: the
    worktree goes, but `branch -d` refuses an unmerged branch on purpose --
    the ref survives so the commit is never silently lost."""
    wt = create_agent_worktree(repo, "run-uniquecm")
    (wt.worktree_path / "b.txt").write_text("agent commit\n")
    _git(
        wt.worktree_path,
        "-c", "user.name=t", "-c", "user.email=t@t",
        "add", "-A",
    )
    _git(
        wt.worktree_path,
        "-c", "user.name=t", "-c", "user.email=t@t",
        "commit", "-m", "agent work",
    )
    refusal = discard_agent_worktree(repo, wt, force=False)
    assert refusal is None
    assert not wt.worktree_path.exists()
    branches = _git(repo, "branch", "--list", wt.branch)
    assert wt.branch in branches  # ref survives -- the commit is not lost


def test_prune_removes_only_dead_runs(repo):
    """(a) a clean worktree gets removed AND its ancestor branch deleted;
    (c) a run id in `live_run_ids` is skipped entirely."""
    live = create_agent_worktree(repo, "run-live0001")
    dead = create_agent_worktree(repo, "run-dead0001")
    assert isinstance(live, AgentWorktree) and isinstance(dead, AgentWorktree)
    removed = prune_stale_agent_worktrees(repo, live_run_ids={"run-live0001"})
    assert removed == 1
    assert live.worktree_path.exists()
    assert not dead.worktree_path.exists()
    assert dead.branch not in _git(repo, "branch", "--list", dead.branch)
    discard_agent_worktree(repo, live)


def test_prune_skips_a_dirty_worktree(repo):
    """(b) implicit GC must never destroy uncommitted work -- a dirty
    worktree is left on disk, exactly as the tool schemas promise."""
    dirty = create_agent_worktree(repo, "run-dirtygc1")
    (dirty.worktree_path / "a.txt").write_text("uncommitted change\n")
    removed = prune_stale_agent_worktrees(repo, live_run_ids=set())
    assert removed == 0
    assert dirty.worktree_path.is_dir()
    assert (dirty.worktree_path / "a.txt").read_text() == "uncommitted change\n"
    discard_agent_worktree(repo, dirty)  # force=True cleanup for the test itself


def test_prune_removes_a_clean_worktree_but_leaves_an_unmerged_branch_ref(repo):
    """(d) a CLEAN worktree (removable) whose branch carries a commit not
    merged into the shared tree: the worktree goes, but the commit is
    never silently lost -- the branch ref survives."""
    wt = create_agent_worktree(repo, "run-uniquegc")
    (wt.worktree_path / "b.txt").write_text("agent commit\n")
    _git(wt.worktree_path, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(
        wt.worktree_path,
        "-c", "user.name=t", "-c", "user.email=t@t",
        "commit", "-m", "agent work",
    )
    removed = prune_stale_agent_worktrees(repo, live_run_ids=set())
    assert removed == 1
    assert not wt.worktree_path.exists()
    assert wt.branch in _git(repo, "branch", "--list", wt.branch)


def test_create_refuses_when_worktree_base_unwritable(tmp_path, repo, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory\n")
    monkeypatch.setattr(
        "tldw_chatbook.Agents.agent_worktree._worktrees_base",
        lambda: blocker / "sub",
    )
    result = create_agent_worktree(repo, "run-blocked1")
    assert isinstance(result, WorktreeRefusal)
    assert result.reason_code == "worktree_create_failed"


from tldw_chatbook.Agents.agent_worktree import (  # noqa: E402
    MergeOutcome,
    merge_agent_worktree_changes,
    preview_agent_worktree_diffstat,
)


def test_apply_mode_lands_uncommitted_diff(repo):
    wt = create_agent_worktree(repo, "run-apply001")
    (wt.worktree_path / "a.txt").write_text("child version\n")
    (wt.worktree_path / "new.txt").write_text("brand new\n")
    outcome = merge_agent_worktree_changes(repo, wt, mode="apply")
    assert isinstance(outcome, MergeOutcome), getattr(outcome, "message", outcome)
    assert outcome.commit_sha is None
    assert (repo / "a.txt").read_text() == "child version\n"
    assert (repo / "new.txt").read_text() == "brand new\n"
    # uncommitted: the shared tree is dirty, no new commit on HEAD
    status = _git(repo, "status", "--porcelain")
    assert " a.txt" in status or "M a.txt" in status.replace("  ", " ")
    assert "a.txt" in outcome.diffstat
    discard_agent_worktree(repo, wt)


def test_merge_mode_creates_merge_commit(repo):
    wt = create_agent_worktree(repo, "run-merge001")
    (wt.worktree_path / "a.txt").write_text("merged version\n")
    before = _git(repo, "rev-parse", "HEAD").strip()
    outcome = merge_agent_worktree_changes(repo, wt, mode="merge")
    assert isinstance(outcome, MergeOutcome), getattr(outcome, "message", outcome)
    after = _git(repo, "rev-parse", "HEAD").strip()
    assert outcome.commit_sha == after != before
    assert (repo / "a.txt").read_text() == "merged version\n"
    parents = _git(repo, "log", "-1", "--format=%P").split()
    assert len(parents) == 2  # a real --no-ff merge commit
    discard_agent_worktree(repo, wt)


def test_apply_conflict_refuses_atomically_naming_file(repo):
    wt = create_agent_worktree(repo, "run-conflict1")
    (wt.worktree_path / "a.txt").write_text("child side\n")
    (repo / "a.txt").write_text("user side\n")  # conflicting shared-tree change
    refusal = merge_agent_worktree_changes(repo, wt, mode="apply")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "apply_conflict"
    assert "a.txt" in refusal.message
    assert (repo / "a.txt").read_text() == "user side\n"  # untouched
    discard_agent_worktree(repo, wt)


def test_merge_conflict_aborts_and_names_file(repo):
    wt = create_agent_worktree(repo, "run-conflict2")
    (wt.worktree_path / "a.txt").write_text("child side\n")
    (repo / "a.txt").write_text("user side committed\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "user change")
    before = _git(repo, "rev-parse", "HEAD").strip()
    refusal = merge_agent_worktree_changes(repo, wt, mode="merge")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "merge_conflict"
    assert "a.txt" in refusal.message
    assert _git(repo, "rev-parse", "HEAD").strip() == before  # aborted cleanly
    assert "user side committed" in (repo / "a.txt").read_text()
    discard_agent_worktree(repo, wt)


def test_clean_worktree_is_nothing_to_merge(repo):
    wt = create_agent_worktree(repo, "run-noop0001")
    refusal = merge_agent_worktree_changes(repo, wt, mode="apply")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "nothing_to_merge"
    discard_agent_worktree(repo, wt)


def test_apply_conflict_names_file_for_already_exists_shape(repo):
    """Real git stderr for a new-file collision is a DIFFERENT shape than a
    content conflict ("error: <file>: already exists in working directory"
    vs. "error: patch failed: <file>:<line>") -- both must name the file.
    """
    wt = create_agent_worktree(repo, "run-newfile1")
    (wt.worktree_path / "new.txt").write_text("child content\n")
    (repo / "new.txt").write_text("shared content\n")  # independent, untracked
    refusal = merge_agent_worktree_changes(repo, wt, mode="apply")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "apply_conflict"
    assert "new.txt" in refusal.message
    discard_agent_worktree(repo, wt)


def test_apply_patch_tempfile_write_failure_is_refusal_not_exception(repo, monkeypatch):
    wt = create_agent_worktree(repo, "run-tmpfail1")
    (wt.worktree_path / "a.txt").write_text("child change\n")

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("tempfile.NamedTemporaryFile", boom)
    refusal = merge_agent_worktree_changes(repo, wt, mode="apply")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "apply_conflict"
    assert "disk full" in refusal.message
    discard_agent_worktree(repo, wt)


def test_auto_commit_failure_is_refusal_not_silent_nothing_to_merge(repo, monkeypatch):
    from tldw_chatbook.Agents import agent_worktree as mod

    wt = create_agent_worktree(repo, "run-cmtfail1")
    (wt.worktree_path / "a.txt").write_text("child change\n")
    real_git = mod._git

    def fake_git(repo_root, *args):
        if "commit" in args:
            return 128, "", "fatal: unable to auto-detect email address"
        return real_git(repo_root, *args)

    monkeypatch.setattr(mod, "_git", fake_git)
    refusal = merge_agent_worktree_changes(repo, wt, mode="apply")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "worktree_commit_failed"
    discard_agent_worktree(repo, wt)


def test_status_read_failure_is_a_refusal_not_nothing_to_merge(repo, monkeypatch):
    """Finding 8 (Qodo round, Medium): the comment already says a status
    failure must not fall through to "nothing_to_merge" (that would
    silently strand the child's work -- lost on discard, no trail) --
    but the code only ever branched on `code == 0 and out.strip()`, so a
    nonzero status read fell through exactly as the comment warns
    against, since `out.strip()` on the ignored garbage output reads
    falsy too.
    """
    from tldw_chatbook.Agents import agent_worktree as mod

    wt = create_agent_worktree(repo, "run-statfail1")
    (wt.worktree_path / "a.txt").write_text("child change\n")
    real_git = mod._git

    def fake_git(repo_root, *args):
        if args[:2] == ("status", "--porcelain"):
            return 128, "", "fatal: not a git repository (simulated)"
        return real_git(repo_root, *args)

    monkeypatch.setattr(mod, "_git", fake_git)
    refusal = merge_agent_worktree_changes(repo, wt, mode="apply")
    assert isinstance(refusal, WorktreeRefusal)
    assert refusal.reason_code == "worktree_commit_failed"
    assert refusal.reason_code != "nothing_to_merge"
    discard_agent_worktree(repo, wt)


# -- TASK-28238 phase 2 T6: preview_agent_worktree_diffstat ----------------
#
# The confirm card must show a diffstat before the user consents to a
# merge -- but a child's work is typically still UNCOMMITTED in its own
# worktree (nothing commits it there; `merge_agent_worktree_changes` only
# does so itself, right before computing its own post-commit diffstat).
# A preview that diffed `base_sha..branch` in `repo_root` (mirroring that
# post-commit diff verbatim) would therefore read empty for the common
# case. This diffs `base_sha` against the WORKTREE's own working tree
# instead (`git diff --stat <base_sha>`, no second ref, run in
# `wt.worktree_path`) -- a single non-mutating command that sees
# uncommitted work too.


def test_preview_diffstat_sees_uncommitted_child_work(repo):
    wt = create_agent_worktree(repo, "run-preview1")
    (wt.worktree_path / "a.txt").write_text("child version\n")
    diffstat = preview_agent_worktree_diffstat(repo, wt)
    assert "a.txt" in diffstat
    # non-mutating: nothing was committed by computing the preview.
    assert _git(wt.worktree_path, "status", "--porcelain").strip() != ""
    discard_agent_worktree(repo, wt)


def test_preview_diffstat_sees_untracked_new_file(repo):
    """`git diff` alone never lists untracked files -- a brand-new file is
    as common a shape of "child's work" as an edited one."""
    wt = create_agent_worktree(repo, "run-preview4")
    (wt.worktree_path / "new.txt").write_text("brand new\n")
    diffstat = preview_agent_worktree_diffstat(repo, wt)
    assert "new.txt" in diffstat
    assert _git(wt.worktree_path, "status", "--porcelain").strip() == "?? new.txt"
    discard_agent_worktree(repo, wt)


def test_preview_diffstat_empty_for_untouched_worktree(repo):
    wt = create_agent_worktree(repo, "run-preview2")
    assert preview_agent_worktree_diffstat(repo, wt) == ""
    discard_agent_worktree(repo, wt)


def test_preview_diffstat_never_raises_on_missing_worktree(repo):
    wt = create_agent_worktree(repo, "run-preview3")
    discard_agent_worktree(repo, wt)  # worktree_path no longer exists
    assert preview_agent_worktree_diffstat(repo, wt) == ""
