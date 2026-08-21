"""The Review screen's COMMIT UI — TASK-16801 T7 (spec §5).

This is the surface through which a user writes to their REAL git
repository, so nothing here is simulated: the shipped CSS bundle, the real
`AgentRunsChangeReviewProvider` (never a hand-rolled fake — the
fixture-invented-shapes trap has bitten this repo five separate times), a
FILE-backed `AgentRunsDB` (never `:memory:` — in-memory SQLite is
thread-affine and every git action here lands off a worker thread), and
real temporary git repositories that are really committed into.

Two guards get pinned here that nothing else can prove:

- the **fresh preflight** (spec §5 step 2): the modal's checklist comes
  from a NEW `git status` read taken at modal-open, never from the
  possibly-stale tree the screen last rendered. Driven by creating a file
  on disk AFTER the view has loaded and asserting it is in the modal while
  it is still absent from the tree behind it.
- the **post-commit memo invalidation** (`_diff_cache_generation`): the
  commit reload is the FIRST consumer of the bump Task 6 added, so a stale
  pre-commit diff must not be servable from the memo afterwards.

Porcelain note (repo lesson): `git status --porcelain` output run through
a `.strip()`ing helper EATS the leading space that distinguishes staged
from unstaged, so every "did this file get committed" assertion below goes
through `git show --name-only`, `git diff`/`git diff --cached`, or
`git ls-files --others` instead.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest
from rich.text import Text
from textual.app import App
from textual.widgets import Button, Checkbox, Input, Select, Static, Tree

import tldw_chatbook.config as config_module
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.change_review_screen import (
    COMMIT_CLEAN_TREE_REASON,
    COMMIT_RUN_ACTIVE_REFUSAL,
    CURRENT_MODE_COMMENT_REFUSAL,
    CURRENT_MODE_SENTINEL,
    GIT_ACTION_WORKER_GROUP,
    AgentRunsChangeReviewProvider,
    ChangeGitCommitModal,
    ChangeReviewScreen,
)
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _git_ok(cwd: Path, *args: str) -> int:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return proc.returncode


def _init_repo(root: Path, *, commit: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    if commit:
        (root / "a.txt").write_text("base\n")
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", "base")
    return root


def _commit_count(root: Path) -> int:
    proc = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return int(proc.stdout.strip()) if proc.returncode == 0 else 0


def _committed_paths(root: Path) -> list[str]:
    """Paths in HEAD's commit — never a `.strip()`ed porcelain compare."""
    raw = _git(root, "show", "--name-only", "--pretty=format:", "HEAD")
    return [line for line in raw.splitlines() if line]


def _untracked(root: Path) -> list[str]:
    raw = _git(root, "ls-files", "--others", "--exclude-standard")
    return [line for line in raw.splitlines() if line]


def _patch_git_actions(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    """Pin `[change_review] git_actions` to ``value`` for this test."""

    def fake(section, key=None, default=None):
        if section == "change_review" and key == "git_actions":
            return value
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake)


def _record_turn(db, tracker, root: Path, run_id: str, mutate) -> None:
    """One real tracked turn: baseline, mutate the tree, end, store rows."""
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    mutate()
    for rec in tracker.end_turn(handle):
        db.record_change_snapshot(
            run_id=run_id,
            root=rec.root,
            baseline_sha=rec.baseline_sha,
            end_sha=rec.end_sha,
            files_changed=rec.files_changed,
            adds=rec.adds,
            dels=rec.dels,
            tracking_error=rec.tracking_error,
            untracked_oversize=rec.untracked_oversize,
            nested_repos=rec.nested_repos,
        )


class _Harness(App[None]):
    CSS_PATH = str(BUNDLE)

    def __init__(self, provider, workspace_roots=None) -> None:
        super().__init__()
        self._provider = provider
        self._workspace_roots = workspace_roots

    def on_mount(self) -> None:
        self.push_screen(
            ChangeReviewScreen(
                self._provider, workspace_roots=self._workspace_roots
            )
        )


async def _wait_for(pilot, predicate, what: str, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {what}")


async def _open_screen(pilot, app) -> ChangeReviewScreen:
    return await _wait_for(
        pilot,
        lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
        "review screen",
    )


async def _wait_for_detection(pilot, screen) -> ChangeReviewScreen:
    await _wait_for(
        pilot,
        lambda: screen.git_detection_settled or None,
        "git detection landed",
    )
    return screen


async def _wait_idle(pilot, app, group: str) -> None:
    def settled():
        live = [
            w
            for w in app.workers
            if w.group == group and w.state.name in ("PENDING", "RUNNING")
        ]
        return not live

    await _wait_for(pilot, lambda: settled() or None, f"{group} worker idle")


def _tree_labels(tree: Tree) -> list[str]:
    labels: list[str] = []

    def walk(node):
        labels.append(str(node.label))
        for child in node.children:
            walk(child)

    walk(tree.root)
    return labels


def _static_text(screen, selector: str) -> str:
    renderable = screen.query_one(selector, Static).renderable
    if isinstance(renderable, Text):
        return renderable.plain
    return str(renderable)


async def _enter_current_mode(pilot, app) -> ChangeReviewScreen:
    """Open the screen and switch it to the REAL working tree."""
    screen = await _open_screen(pilot, app)
    await _wait_for_detection(pilot, screen)
    screen.query_one("#change-review-turn-select", Select).value = (
        CURRENT_MODE_SENTINEL
    )
    await _wait_for(
        pilot,
        lambda: screen._current_mode_active() or None,
        "the current-mode selection",
    )
    await _wait_idle(pilot, app, "change-review-current")
    await pilot.pause()
    return screen


async def _wait_for_modal(pilot, app, what: str) -> ChangeGitCommitModal:
    """Wait for the commit modal AND for its children to be mounted."""
    modal = await _wait_for(
        pilot,
        lambda: app.screen if isinstance(app.screen, ChangeGitCommitModal) else None,
        what,
    )
    await _wait_for(
        pilot,
        lambda: bool(modal.query("#change-git-commit-yes")) or None,
        f"{what} to finish composing",
    )
    return modal


async def _open_commit_modal(pilot, app, screen) -> ChangeGitCommitModal:
    screen.action_git_commit()
    return await _wait_for_modal(pilot, app, "the commit modal")


def _modal_paths(modal) -> list[str]:
    return [p for box in modal.query(Checkbox) for p in box.file_paths]


async def _submit_modal(
    pilot,
    modal,
    message: str,
    *,
    uncheck: tuple[str, ...] = (),
    branch: str | None = None,
) -> None:
    modal.query_one("#change-git-commit-message", Input).value = message
    if branch is not None:
        modal.query_one("#change-git-commit-branch", Input).value = branch
    for box in modal.query(Checkbox):
        if any(path in uncheck for path in box.file_paths):
            box.value = False
    modal.query_one("#change-git-commit-yes", Button).press()
    await pilot.pause()


def _make_provider(tmp_path, conversation_id, **kwargs):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    return AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id=conversation_id, **kwargs
    ), db, service


@pytest.fixture()
def commit_fixture(tmp_path):
    """A real repo with one tracked modification and one untracked add."""
    repo = _init_repo(tmp_path / "repo")
    (repo / "a.txt").write_text("changed\n")
    (repo / "brand_new.txt").write_text("hello from the working tree\n")
    provider, _db, _service = _make_provider(tmp_path, "conv-commit")
    return provider, repo


@pytest.fixture()
def turn_fixture(tmp_path):
    """A real repo that is ALSO one recorded turn's workspace root."""
    repo = _init_repo(tmp_path / "repo")
    provider, db, service = _make_provider(tmp_path, "conv-turn")
    tracker = ChangeTurnTracker(service=service)
    run1 = db.create_run(conversation_id="conv-turn", agent_kind="primary")
    _record_turn(
        db, tracker, repo, run1,
        lambda: (repo / "turn_one.txt").write_text("one\n"),
    )
    (repo / "a.txt").write_text("changed\n")
    return provider, repo, run1


# ---------------------------------------------------------------------------
# The button + the mode-aware affordances (spec §5, §8; T6 review finding a)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_button_is_offered_only_in_current_mode(
    monkeypatch, turn_fixture
):
    """Spec §5: `Commit…` is visible/enabled ONLY against the working tree."""
    _patch_git_actions(monkeypatch, True)
    provider, repo, run1 = turn_fixture
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)
        commit_btn = screen.query_one("#change-review-git-commit-btn", Button)

        assert screen._current_mode_active() is False, (
            "the screen must still OPEN on the recorded turn"
        )
        assert commit_btn.display is False, (
            "commit must not be offered against a recorded snapshot turn"
        )

        await _enter_current_mode(pilot, app)
        assert commit_btn.display is True
        assert commit_btn.disabled is False, (
            "a working tree with changes must offer a live commit button"
        )

        # ...and it goes away again on the way back to a recorded turn.
        screen.query_one("#change-review-turn-select", Select).value = run1
        await _wait_for(
            pilot,
            lambda: (screen._current_mode_active() is False) or None,
            "the turn view",
        )
        await pilot.pause()
        assert commit_btn.display is False


@pytest.mark.asyncio
async def test_commit_button_is_disabled_with_a_reason_on_a_clean_tree(
    monkeypatch, tmp_path
):
    """Spec §8: a disabled action carries its reason, never a dead control."""
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "clean_repo")
    provider, _db, _service = _make_provider(tmp_path, "conv-clean")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        await _wait_for(
            pilot,
            lambda: ("working tree clean" in screen.diff_pane_text()) or None,
            "the clean-tree copy",
        )
        commit_btn = screen.query_one("#change-review-git-commit-btn", Button)
        assert commit_btn.display is True, (
            "the mode is still entered on a clean tree (spec §4)"
        )
        assert commit_btn.disabled is True
        assert COMMIT_CLEAN_TREE_REASON in str(commit_btn.tooltip), (
            f"the disabled button must say why; got {commit_btn.tooltip!r}"
        )

        # The keyboard path refuses with the same copy rather than dying.
        notes: list[tuple] = []
        app.notify = lambda *a, **kw: notes.append((a, kw))
        screen.action_git_commit()
        await pilot.pause()
        assert notes, "the commit action must refuse audibly on a clean tree"
        assert COMMIT_CLEAN_TREE_REASON in str(notes[0][0][0])
        assert app.screen is screen, "no modal may be pushed with nothing to commit"


@pytest.mark.asyncio
async def test_snapshot_only_affordances_present_as_unavailable(
    monkeypatch, turn_fixture
):
    """T6 re-review finding (a): they must LOOK unavailable in current mode.

    The notify-on-press refusals stay as the backstop — this asserts BOTH
    halves, so neither the presentation nor the refusal can be dropped.
    """
    _patch_git_actions(monkeypatch, True)
    provider, repo, _run1 = turn_fixture
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)
        comment_btn = screen.query_one("#change-review-comment-file-btn", Button)

        assert comment_btn.disabled is False, (
            "against a recorded turn the comment button is genuinely live"
        )
        turn_footer = _static_text(screen, "#change-review-footer")
        assert "C comment file" in turn_footer

        await _enter_current_mode(pilot, app)
        assert comment_btn.disabled is True, (
            "the comment button must not look live in current mode"
        )
        assert CURRENT_MODE_COMMENT_REFUSAL in str(comment_btn.tooltip)
        current_footer = _static_text(screen, "#change-review-footer")
        assert current_footer != turn_footer, (
            "the footer must stop advertising the snapshot-only keys"
        )
        assert "recorded turn" in current_footer, (
            f"the footer must say why they are unavailable: {current_footer!r}"
        )

        # Backstop kept: `C` still refuses with copy (never a dead key).
        notes: list[tuple] = []
        app.notify = lambda *a, **kw: notes.append((a, kw))
        await screen.action_comment_file()
        await pilot.pause()
        assert notes, "`C` must still notify in current mode"
        assert CURRENT_MODE_COMMENT_REFUSAL in str(notes[0][0][0])


# ---------------------------------------------------------------------------
# Refusal first, then a FRESH preflight (spec §5 steps 1-2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_run_refuses_before_any_modal_or_status_read(
    monkeypatch, tmp_path
):
    """Spec §5 step 1: the run-active refusal is a notify — never a modal."""
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "repo")
    (repo / "a.txt").write_text("changed\n")
    provider, _db, _service = _make_provider(
        tmp_path, "conv-active", run_active=lambda: True
    )
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)

        # Count at the PROVIDER boundary: the refusal must land before the
        # fresh preflight read, not merely before the modal push.
        status_calls: list[str] = []
        real_status = provider.current_status

        def counting_status(root):
            status_calls.append(root)
            return real_status(root)

        provider.current_status = counting_status
        notes: list[tuple] = []
        app.notify = lambda *a, **kw: notes.append((a, kw))

        screen.action_git_commit()
        await pilot.pause()
        await pilot.pause()

        assert notes, "an active run must refuse audibly"
        assert COMMIT_RUN_ACTIVE_REFUSAL in str(notes[0][0][0])
        assert app.screen is screen, "no commit modal may be pushed"
        assert status_calls == [], (
            f"no preflight read may run for a refused commit; got {status_calls!r}"
        )


@pytest.mark.asyncio
async def test_modal_checklist_comes_from_a_fresh_status_read(
    monkeypatch, commit_fixture
):
    """Spec §5 step 2: the modal never trusts the view it was opened from.

    A file created AFTER the tree rendered is invisible to the view and
    MUST still be in the modal's checklist.
    """
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        await _wait_for(
            pilot,
            lambda: (
                lambda ls: ls if any("a.txt" in item for item in ls) else None
            )(_tree_labels(screen.query_one(Tree))),
            "the working tree's files",
        )

        (repo / "late.txt").write_text("created after the view loaded\n")
        stale = "\n".join(_tree_labels(screen.query_one(Tree)))
        assert "late.txt" not in stale, (
            "the rendered view must genuinely be stale for this test to mean "
            f"anything; got {stale!r}"
        )

        modal = await _open_commit_modal(pilot, app, screen)
        paths = _modal_paths(modal)
        assert "late.txt" in paths, (
            f"the modal must list what commit will actually see; got {paths!r}"
        )
        assert "a.txt" in paths and "brand_new.txt" in paths
        assert "late.txt" not in "\n".join(_tree_labels(screen.query_one(Tree))), (
            "the fresh read is the modal's, not a silent reload of the view"
        )


# ---------------------------------------------------------------------------
# The modal itself (spec §5 step 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_files_are_pre_checked_and_unchecking_excludes_one(
    monkeypatch, commit_fixture
):
    """All pre-checked; an unchecked file stays OUT of the commit."""
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)

        boxes = list(modal.query(Checkbox))
        assert boxes, "the checklist must list the working tree's files"
        assert all(box.value for box in boxes), "every file starts checked"

        await _submit_modal(
            pilot, modal, "commit from review", uncheck=("brand_new.txt",)
        )
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await pilot.pause()

        committed = _committed_paths(repo)
        assert "a.txt" in committed, f"the checked file must land; got {committed!r}"
        assert "brand_new.txt" not in committed, (
            f"an unchecked file must be excluded; got {committed!r}"
        )
        assert "brand_new.txt" in _untracked(repo), (
            "the excluded file must survive untouched in the working tree"
        )


@pytest.mark.asyncio
async def test_blank_message_blocks_the_submit(monkeypatch, commit_fixture):
    """A required, stripped-nonempty message (spec §5 step 3)."""
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    before = _commit_count(repo)
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)

        await _submit_modal(pilot, modal, "   ")
        await pilot.pause()

        assert app.screen is modal, "a blank message must not dismiss the modal"
        error = modal.query_one("#change-git-commit-error", Static)
        assert "message" in str(error.renderable).lower(), (
            f"the modal must say why it refused; got {error.renderable!r}"
        )
        assert _commit_count(repo) == before, "nothing may be committed"

        # And it recovers: a real message goes through.
        await _submit_modal(pilot, modal, "a real message")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await pilot.pause()
        assert _commit_count(repo) == before + 1


@pytest.mark.asyncio
async def test_main_branch_warning_is_rendered_and_never_blocks(
    monkeypatch, commit_fixture
):
    """Committing to `main` WARNS (spec §5 step 3) — it does not block."""
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    before = _commit_count(repo)
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)

        warning = modal.query_one("#change-git-commit-warnings", Static)
        text = str(warning.renderable)
        assert "committing directly to main" in text, (
            f"the main/master warning must be rendered; got {text!r}"
        )

        await _submit_modal(pilot, modal, "warned but allowed")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await pilot.pause()
        assert _commit_count(repo) == before + 1, (
            "the warning must never block the commit"
        )


@pytest.mark.asyncio
async def test_detached_head_warning_is_rendered(monkeypatch, tmp_path):
    """Detached HEAD warns that the commit lands on no branch."""
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "detached_repo")
    (repo / "b.txt").write_text("second\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    _git(repo, "checkout", "-q", "--detach")
    (repo / "a.txt").write_text("changed while detached\n")
    provider, _db, _service = _make_provider(tmp_path, "conv-detached")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)

        text = str(modal.query_one("#change-git-commit-warnings", Static).renderable)
        assert "not be on any branch" in text, (
            f"a detached HEAD must be warned about; got {text!r}"
        )
        assert "committing directly to" not in text, (
            "a detached HEAD has no branch to warn about committing to"
        )


@pytest.mark.asyncio
async def test_modal_names_the_repository_it_will_act_on(
    monkeypatch, commit_fixture
):
    """Spec §6: the confirm modal always NAMES its target root."""
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)

        target = str(modal.query_one("#change-git-commit-target", Static).renderable)
        assert str(repo.resolve()) in target, (
            f"the modal must name the repository it commits into; got {target!r}"
        )
        assert "main" in target, "and the branch the commit will land on"


@pytest.mark.asyncio
async def test_an_unstaged_rename_commits_as_its_two_real_rows(
    monkeypatch, tmp_path
):
    """A shell `mv` is a deletion + an untracked add — and commits whole."""
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "moved_repo")
    (repo / "renamed.txt").write_text((repo / "a.txt").read_text())
    (repo / "a.txt").unlink()
    provider, _db, _service = _make_provider(tmp_path, "conv-moved")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)

        assert sorted(_modal_paths(modal)) == ["a.txt", "renamed.txt"], (
            "git reports an unstaged rename as two independent rows"
        )
        await _submit_modal(pilot, modal, "moved it")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await pilot.pause()

        tracked = _git(repo, "ls-tree", "--name-only", "HEAD").splitlines()
        assert tracked == ["renamed.txt"], (
            f"HEAD must carry the whole move; got {tracked!r}"
        )
        assert _git(repo, "diff", "--cached", "--name-only") == ""


@pytest.mark.asyncio
async def test_an_index_recorded_rename_refuses_loudly_and_stages_nothing(
    monkeypatch, tmp_path
):
    """A `git mv` rename is ONE row carrying BOTH paths — and today it fails.

    Known limitation, pinned deliberately rather than papered over (see
    `_commit_entries`' docstring and the T7 report): the engine shares one
    pathspec between `git add -A --` and `git commit --`, and an
    index-recorded rename's OLD path exists in neither the worktree nor
    the index, so the add step exits fatal. What matters — and what this
    asserts — is that the refusal is LOUD, that nothing is committed, and
    that the user's staged rename survives intact. The alternative (send
    only the new path) would commit an ADD while leaving the old path in
    HEAD: a commit that silently contradicts the row the user checked.
    """
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "rename_repo")
    _git(repo, "mv", "a.txt", "renamed.txt")
    before = _commit_count(repo)
    provider, _db, _service = _make_provider(tmp_path, "conv-rename")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)

        boxes = list(modal.query(Checkbox))
        assert len(boxes) == 1, "a recorded rename is ONE row, not two"
        assert sorted(boxes[0].file_paths) == ["a.txt", "renamed.txt"], (
            f"both paths must ride the one checkbox; got {boxes[0].file_paths!r}"
        )
        assert "a.txt → renamed.txt" in str(boxes[0].label)

        notes: list[tuple] = []
        app.notify = lambda *a, **kw: notes.append((a, kw))
        await _submit_modal(pilot, modal, "rename it")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await pilot.pause()

        messages = [str(call[0][0]) for call in notes]
        assert any("Commit failed at stage" in m for m in messages), (
            f"the blocking step must be named with git's own error; got {messages!r}"
        )
        assert _commit_count(repo) == before, "nothing may be committed"
        staged = _git(repo, "diff", "--cached", "--name-status")
        assert "renamed.txt" in staged and "a.txt" in staged, (
            f"the user's staged rename must survive untouched; got {staged!r}"
        )


@pytest.mark.asyncio
async def test_escape_cancels_without_committing(monkeypatch, commit_fixture):
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    before = _commit_count(repo)
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)
        modal.query_one("#change-git-commit-message", Input).value = "not sent"

        await pilot.press("escape")
        await _wait_for(
            pilot,
            lambda: (app.screen is screen) or None,
            "the modal to close",
        )
        await pilot.pause()
        assert _commit_count(repo) == before, "escape must commit nothing"


# ---------------------------------------------------------------------------
# The commit itself (spec §5 steps 4-5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commit_lands_notifies_the_sha_and_reloads_the_view(
    monkeypatch, commit_fixture
):
    """The AC's e2e case, driven through the BUTTON (not just the action)."""
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    before = _commit_count(repo)
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        await _wait_for(
            pilot,
            lambda: (
                lambda ls: ls if any("brand_new.txt" in i for i in ls) else None
            )(_tree_labels(screen.query_one(Tree))),
            "the working tree's files",
        )
        notes: list[tuple] = []
        app.notify = lambda *a, **kw: notes.append((a, kw))

        screen.query_one("#change-review-git-commit-btn", Button).press()
        modal = await _wait_for_modal(pilot, app, "the commit modal (button path)")
        await _submit_modal(pilot, modal, "everything from the review screen")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await pilot.pause()

        assert _commit_count(repo) == before + 1
        sha = _git(repo, "rev-parse", "--short", "HEAD")
        messages = [str(call[0][0]) for call in notes]
        assert any(sha in message for message in messages), (
            f"the success notify must carry the short sha {sha!r}; got {messages!r}"
        )
        assert any("2 file" in message for message in messages), (
            f"...and the file count; got {messages!r}"
        )
        committed = _committed_paths(repo)
        assert sorted(committed) == ["a.txt", "brand_new.txt"], committed

        # The view reloads: the committed files leave the tree.
        await _wait_for(
            pilot,
            lambda: ("working tree clean" in screen.diff_pane_text()) or None,
            "the reloaded (now clean) working tree",
        )
        labels = "\n".join(_tree_labels(screen.query_one(Tree)))
        assert "a.txt" not in labels and "brand_new.txt" not in labels


@pytest.mark.asyncio
async def test_a_pre_staged_unrelated_file_survives_the_commit(
    monkeypatch, commit_fixture
):
    """Spec §2 probe 1, at the UI seam: the pathspec commit hijacks nothing."""
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    (repo / "staged_elsewhere.txt").write_text("staged in a terminal\n")
    _git(repo, "add", "--", "staged_elsewhere.txt")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)

        await _submit_modal(
            pilot, modal, "only mine", uncheck=("staged_elsewhere.txt",)
        )
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await pilot.pause()

        committed = _committed_paths(repo)
        assert "staged_elsewhere.txt" not in committed, committed
        cached = _git(repo, "diff", "--cached", "--name-only")
        assert "staged_elsewhere.txt" in cached.splitlines(), (
            f"the user's pre-staged work must stay staged; got {cached!r}"
        )


@pytest.mark.asyncio
async def test_create_branch_first_checks_out_the_new_branch(
    monkeypatch, commit_fixture
):
    """The optional "create branch first" field (spec §5 step 4)."""
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)

        await _submit_modal(
            pilot, modal, "on a fresh branch", branch="feat/from-review"
        )
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await pilot.pause()

        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "feat/from-review"
        assert "a.txt" in _committed_paths(repo)


@pytest.mark.asyncio
async def test_a_failing_step_is_named_with_its_git_error(
    monkeypatch, commit_fixture
):
    """Spec §5 step 5: failures name the STEP + the excerpt, never "git failed"."""
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    before = _commit_count(repo)
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)
        notes: list[tuple] = []
        app.notify = lambda *a, **kw: notes.append((a, kw))

        # `-bad` is refused by `check-ref-format` (spec §2 probe 3) — the
        # validate-branch step blocks before anything mutates.
        await _submit_modal(pilot, modal, "never lands", branch="-bad")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await pilot.pause()

        messages = [str(call[0][0]) for call in notes]
        assert any("validate-branch" in message for message in messages), (
            f"the blocking step must be named; got {messages!r}"
        )
        assert _commit_count(repo) == before, "nothing may be committed"
        assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


@pytest.mark.asyncio
async def test_merge_in_progress_refuses_with_copy_and_commits_nothing(
    monkeypatch, tmp_path
):
    """Spec §5 step 3: an in-progress merge refuses with a reason."""
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "merge_repo")
    _git(repo, "checkout", "-q", "-b", "other")
    (repo / "a.txt").write_text("from other\n")
    _git(repo, "commit", "-qam", "other side")
    _git(repo, "checkout", "-q", "main")
    (repo / "a.txt").write_text("from main\n")
    _git(repo, "commit", "-qam", "main side")
    assert _git_ok(repo, "merge", "other") != 0, "the merge must really conflict"
    assert _git_ok(repo, "rev-parse", "--verify", "-q", "MERGE_HEAD") == 0
    before = _commit_count(repo)

    provider, _db, _service = _make_provider(tmp_path, "conv-merge")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)
        notes: list[tuple] = []
        app.notify = lambda *a, **kw: notes.append((a, kw))

        await _submit_modal(pilot, modal, "should be refused")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await pilot.pause()

        messages = [str(call[0][0]) for call in notes]
        assert any(
            "merge/rebase/cherry-pick" in message for message in messages
        ), f"the refusal reason must be shown; got {messages!r}"
        assert _commit_count(repo) == before, "no commit may land mid-merge"
        assert _git_ok(repo, "rev-parse", "--verify", "-q", "MERGE_HEAD") == 0, (
            "the merge must still be in progress"
        )


@pytest.mark.asyncio
async def test_buttons_are_disabled_while_the_commit_worker_runs(
    monkeypatch, commit_fixture
):
    """No double-dispatch: the button is dead while the worker is live."""
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    release = threading.Event()
    entered = threading.Event()
    real_commit = provider.commit_selected

    def blocking_commit(root, files, message, new_branch):
        entered.set()
        release.wait(10)
        return real_commit(root, files, message, new_branch)

    provider.commit_selected = blocking_commit

    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        modal = await _open_commit_modal(pilot, app, screen)
        commit_btn = screen.query_one("#change-review-git-commit-btn", Button)

        # Commit only ONE file, so the tree still has changes afterwards and
        # "re-enabled" is a real assertion rather than the clean-tree state.
        await _submit_modal(
            pilot, modal, "one file only", uncheck=("brand_new.txt",)
        )
        await _wait_for(pilot, lambda: entered.is_set() or None, "the commit worker")
        await pilot.pause()

        assert screen._git_busy is True
        assert commit_btn.disabled is True, (
            "the commit button must be dead while a commit is in flight"
        )

        release.set()
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await _wait_for(
            pilot,
            lambda: (screen._git_busy is False) or None,
            "the commit worker to land",
        )
        await _wait_for(
            pilot,
            lambda: (commit_btn.disabled is False) or None,
            "the commit button to come back",
        )
        assert "a.txt" in _committed_paths(repo)


# ---------------------------------------------------------------------------
# The post-commit reload (T6 review finding (b))
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stale_diff_is_not_served_after_the_commit_reload(
    monkeypatch, commit_fixture
):
    """The commit reload is the FIRST consumer of Task 6's memo invalidation.

    `_diff_text_for` is a ONE-ENTRY memo keyed ``(generation, id(row),
    path)``. To make a stale hit possible at all, the memo must still hold
    the pre-commit key when the post-commit read happens — so this commits
    EVERY file: the reloaded tree is then empty, nothing re-renders, and
    the entry survives the reload untouched. The read afterwards uses the
    captured pre-commit row (exactly what an open comment input holds), so
    the key matches byte for byte and only the invalidation can stop the
    stale text being served.

    Two assertions, deliberately: the behavioral one above, and a direct
    pin on ``_diff_cache_generation``. They fail to DIFFERENT mutations —
    the generation bump alone is belt-and-braces for a one-entry memo (the
    same block also nulls ``_diff_cache_key``, which is what actually
    invalidates it), so only the second assertion is red when just the
    bump is removed. Both are needed for the block to be fully pinned.
    """
    _patch_git_actions(monkeypatch, True)
    provider, repo = commit_fixture
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        await _wait_for(
            pilot,
            lambda: (
                lambda ls: ls if any("a.txt" in i for i in ls) else None
            )(_tree_labels(screen.query_one(Tree))),
            "the working tree's files",
        )
        screen.select_file("a.txt")
        await pilot.pause()
        row, change = screen._leaves[screen._focused_leaf]
        assert change.path == "a.txt"
        before = screen._diff_text_for(row, change)
        assert "+changed" in before, (
            f"the pre-commit diff must really be memoized; got {before!r}"
        )
        assert screen._diff_cache_key == (
            screen._diff_cache_generation,
            id(row),
            "a.txt",
        ), "the memo must really be holding this leaf's entry"
        generation_before = screen._diff_cache_generation

        modal = await _open_commit_modal(pilot, app, screen)
        await _submit_modal(pilot, modal, "commit everything")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await _wait_for(
            pilot,
            lambda: ("working tree clean" in screen.diff_pane_text()) or None,
            "the reloaded (now clean) working tree",
        )
        await _wait_idle(pilot, app, "change-review-current")
        await pilot.pause()

        after = screen._diff_text_for(row, change)
        assert "+changed" not in after, (
            "the pre-commit diff must not survive the reload in the memo; "
            f"got {after!r}"
        )
        assert screen._diff_cache_generation > generation_before, (
            "the commit reload must bump the diff-cache generation "
            f"({generation_before} -> {screen._diff_cache_generation})"
        )
