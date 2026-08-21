"""Provider seam for change-review git modes (TASK-16801 arc B, T5).

Drives the REAL `AgentRunsChangeReviewProvider` (never a hand-rolled fake --
the fixture-invented-shapes trap has bitten this repo four separate times)
over a FILE-backed `AgentRunsDB` (never `:memory:` -- in-memory SQLite is
thread-affine and breaks under the screen's threading, per the established
`Tests/UI/test_change_review_screen.py` fixture idiom) and real temp git
repos (per `Tests/Workspaces/test_git_workspace_push.py`'s `_git`/`repo`/
`bare` fixture pattern).

These tests pin the provider's THIN-wrapper contract:

- `git_actions_enabled()` / `detect_git()` implement the `[change_review]
  git_actions` kill switch -- OFF makes `detect_git` return `{}`, which is
  the single check that hides the whole `current` mode from the screen.
- `detect_git()` keys its result by the RESOLVED root string, so two
  spellings of the same directory dedupe to one key and one detection call
  (the exact bug class TASK-16801's Task 2 engine layer was fixed against
  by construction -- `GitWorkspaceInfo.root`/`CurrentRootStatus.root` are
  always resolved paths).
- The wrapped engine functions have DELIBERATELY ASYMMETRIC error
  postures, preserved unchanged through the wrappers: `commit_selected`
  RAISES (`CommitRefusedError`/`GitWorkspaceError`); `push_current` RAISES
  for detached HEAD / no remote but RETURNS a `PushResult` (state can be
  `"failed"`) for an ordinary push outcome; `pr_url` NEVER raises, always
  returning `str | GitWorkspaceRefusal`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import tldw_chatbook.config as config_module
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.change_review_screen import (
    AgentRunsChangeReviewProvider,
)
from tldw_chatbook.Workspaces.change_tracking import ChangedFile, ShadowRepoService
from tldw_chatbook.Workspaces.git_workspace import (
    CommitRefusedError,
    CurrentRootStatus,
    GitWorkspaceError,
    GitWorkspaceInfo,
    GitWorkspaceRefusal,
    PushResult,
)


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _patch_git_actions(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    """Patch `get_cli_setting` so `[change_review] git_actions` resolves
    to `value` -- every other lookup returns its own default untouched,
    so this never disturbs anything but the kill switch under test."""

    def fake(section, key=None, default=None):
        if section == "change_review" and key == "git_actions":
            return value
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake)


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


@pytest.fixture()
def bare(tmp_path: Path) -> Path:
    bare_root = tmp_path / "bare.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(bare_root)],
        check=True,
        capture_output=True,
    )
    return bare_root


@pytest.fixture()
def provider(tmp_path: Path) -> AgentRunsChangeReviewProvider:
    # File-backed, never `:memory:` (in-memory SQLite is thread-affine and
    # breaks under the screen's threading -- see the module docstring).
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    return AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="conv-1"
    )


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


def test_git_actions_enabled_default_true(monkeypatch):
    def fake(section, key=None, default=None):
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake)
    assert AgentRunsChangeReviewProvider.git_actions_enabled() is True


def test_git_actions_enabled_reads_explicit_off(monkeypatch):
    _patch_git_actions(monkeypatch, False)
    assert AgentRunsChangeReviewProvider.git_actions_enabled() is False


def test_git_actions_enabled_survives_garbage_config(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("config exploded")

    monkeypatch.setattr(config_module, "get_cli_setting", boom)
    # Bad config never breaks review -- and never silently disables a
    # feature that shipped ON.
    assert AgentRunsChangeReviewProvider.git_actions_enabled() is True


# ---------------------------------------------------------------------------
# detect_git
# ---------------------------------------------------------------------------


def test_detect_git_empty_when_kill_switch_off(monkeypatch, provider, repo):
    _patch_git_actions(monkeypatch, False)
    assert provider.detect_git([str(repo)]) == {}


def test_detect_git_returns_real_info_when_enabled(monkeypatch, provider, repo):
    _patch_git_actions(monkeypatch, True)
    result = provider.detect_git([str(repo)])

    key = str(repo.resolve())
    assert key in result
    info = result[key]
    assert isinstance(info, GitWorkspaceInfo)
    assert info.root == repo.resolve()
    assert info.branch == "main"
    assert info.unborn is False


def test_detect_git_noncanonical_spelling_dedupes_and_resolves(
    monkeypatch, provider, repo
):
    _patch_git_actions(monkeypatch, True)
    # A non-canonical spelling of the SAME directory (a `..` traversal
    # through the parent) -- the resolved key must be the one shared with
    # the canonical spelling, so a caller keying off `GitWorkspaceInfo.root`
    # (always resolved) never misses this entry.
    noncanonical = str(repo / ".." / repo.name)
    assert noncanonical != str(repo)

    result = provider.detect_git([str(repo), noncanonical])

    assert len(result) == 1
    key = str(repo.resolve())
    assert key in result
    assert isinstance(result[key], GitWorkspaceInfo)


def test_detect_git_non_repo_root_is_none(monkeypatch, provider, tmp_path):
    _patch_git_actions(monkeypatch, True)
    not_repo = tmp_path / "not_a_repo"
    not_repo.mkdir()
    result = provider.detect_git([str(not_repo)])
    assert result[str(not_repo.resolve())] is None


def test_detect_git_survives_garbage_config_by_staying_enabled(
    monkeypatch, provider, repo
):
    def boom(*_args, **_kwargs):
        raise RuntimeError("config exploded")

    monkeypatch.setattr(config_module, "get_cli_setting", boom)
    result = provider.detect_git([str(repo)])
    # Garbage config degrades to "enabled" (same as the switch guard), so
    # detection still runs rather than the surface silently vanishing.
    assert isinstance(result[str(repo.resolve())], GitWorkspaceInfo)


# ---------------------------------------------------------------------------
# current_status / current_diff_text / untracked_preview
# ---------------------------------------------------------------------------


def test_current_status_reads_real_working_tree(provider, repo):
    (repo / "a.txt").write_text("changed\n")
    (repo / "new.txt").write_text("new content\n")

    status = provider.current_status(str(repo))

    assert isinstance(status, CurrentRootStatus)
    assert status.root == repo.resolve()
    assert "new.txt" in status.untracked
    paths = {f.path for f in status.files}
    assert {"a.txt", "new.txt"} <= paths


def test_current_status_raises_for_non_repo_root(provider, tmp_path):
    not_repo = tmp_path / "not_a_repo"
    not_repo.mkdir()
    with pytest.raises(GitWorkspaceError):
        provider.current_status(str(not_repo))


def test_current_diff_text_returns_tracked_diff(provider, repo):
    (repo / "a.txt").write_text("changed content\n")
    diff = provider.current_diff_text(str(repo), ChangedFile(path="a.txt", status="M"))
    assert "-base" in diff
    assert "+changed content" in diff


def test_untracked_preview_honors_diff_display_max_lines(tmp_path, repo):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="conv-1", diff_display_max_lines=100
    )

    big = repo / "big.txt"
    big.write_text("\n".join(f"line-{i}" for i in range(250)) + "\n")

    text = provider.untracked_preview(str(repo), "big.txt")

    assert text.count("+line-") == 100
    assert "truncated at 100 lines" in text


# ---------------------------------------------------------------------------
# commit_selected -- raises, run_active thread-through
# ---------------------------------------------------------------------------


def test_commit_selected_refuses_when_run_active(provider, repo):
    provider.run_active = lambda: True
    (repo / "new.txt").write_text("x\n")

    with pytest.raises(CommitRefusedError):
        provider.commit_selected(str(repo), ["new.txt"], "add new", None)

    # Proves the refusal happened BEFORE touching git -- nothing staged.
    status = _git(repo, "status", "--porcelain")
    assert "new.txt" in status
    assert not status.startswith("A ")


def test_commit_selected_commits_when_run_not_active(provider, repo):
    (repo / "new.txt").write_text("x\n")

    result = provider.commit_selected(str(repo), ["new.txt"], "add new", None)

    assert result.short_sha is not None
    log = _git(repo, "log", "-1", "--format=%s")
    assert log == "add new"


def test_commit_selected_raises_workspace_error_on_empty_files(provider, repo):
    with pytest.raises(GitWorkspaceError):
        provider.commit_selected(str(repo), [], "msg", None)


# ---------------------------------------------------------------------------
# push_current -- raises for preconditions, RETURNS a result otherwise
# ---------------------------------------------------------------------------


def test_push_current_pushes_then_reports_up_to_date(monkeypatch, provider, repo, bare):
    _patch_git_actions(monkeypatch, True)
    _git(repo, "remote", "add", "origin", str(bare))

    info = provider.detect_git([str(repo)])[str(repo.resolve())]
    first = provider.push_current(str(repo), info, None)
    assert isinstance(first, PushResult)
    assert first.state == "pushed"

    info_after = provider.detect_git([str(repo)])[str(repo.resolve())]
    second = provider.push_current(str(repo), info_after, None)
    assert isinstance(second, PushResult)
    assert second.state == "up_to_date"


def test_push_current_raises_when_no_remote_configured(monkeypatch, provider, repo):
    _patch_git_actions(monkeypatch, True)
    info = provider.detect_git([str(repo)])[str(repo.resolve())]
    with pytest.raises(GitWorkspaceError):
        provider.push_current(str(repo), info, None)


# ---------------------------------------------------------------------------
# pr_url -- never raises, isinstance-checked
# ---------------------------------------------------------------------------


def test_pr_url_refuses_without_upstream(monkeypatch, provider, repo):
    _patch_git_actions(monkeypatch, True)
    info = provider.detect_git([str(repo)])[str(repo.resolve())]
    result = provider.pr_url(str(repo), info)
    assert isinstance(result, GitWorkspaceRefusal)


def test_pr_url_builds_github_compare_url(monkeypatch, provider, repo):
    _patch_git_actions(monkeypatch, True)
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "add", "origin", "https://github.com/acme/proj.git")
    # Establish upstream tracking WITHOUT a real network fetch/push -- a
    # local remote-tracking ref plus the branch config is all `@{upstream}`
    # resolution needs (verified against real git; a config-only fake
    # without the ref present leaves `@{upstream}` unresolved).
    _git(repo, "update-ref", "refs/remotes/origin/main", sha)
    _git(repo, "config", "branch.main.remote", "origin")
    _git(repo, "config", "branch.main.merge", "refs/heads/main")

    info = provider.detect_git([str(repo)])[str(repo.resolve())]
    assert info.upstream == "origin/main"

    result = provider.pr_url(str(repo), info)
    assert result == "https://github.com/acme/proj/compare/main?expand=1"
