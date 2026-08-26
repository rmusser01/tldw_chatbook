"""The Review screen's PUSH + Open-PR UI — TASK-16801 T8 (spec §6).

Push writes to a REMOTE, so nothing here is simulated: the shipped CSS
bundle, the real `AgentRunsChangeReviewProvider` (never a hand-rolled fake
— the fixture-invented-shapes trap has bitten this repo five separate
times), a FILE-backed `AgentRunsDB` (never `:memory:` — in-memory SQLite is
thread-affine and every git action here lands off a worker thread), real
temporary git repositories, and a REAL local bare remote (`git init
--bare`) that is really pushed into and really read back.

The only stub anywhere in this file is `app.open_url` for the PR case —
a URL-open stub, never a git mock: the compare URL itself is built by the
real engine from a real remote and a real upstream ref.

Two contrasts get pinned here that nothing else can prove:

- **push is NOT refused during an active run, while commit IS** (spec §6
  states this explicitly, in contrast to §5) — asserted side by side in
  ONE test, because the easy mistake is to copy the commit refusal.
- **never `--force`/`--force-with-lease`** — a non-fast-forward rejection
  surfaces git's own stderr excerpt, asserted against the argv every git
  invocation on the push path actually received.

Porcelain note (repo lesson): `git status --porcelain` run through a
`.strip()`ing helper EATS the leading space that distinguishes staged from
unstaged, so every "did this land" assertion below goes through
`git rev-parse` / `git log` against the BARE remote, or `git diff`, never a
stripped porcelain compare.
"""
from __future__ import annotations

import inspect
import re
import subprocess
import threading
import time
from pathlib import Path

import pytest
from rich.text import Text
from textual.app import App
from textual.widgets import Button, Select, Static

import tldw_chatbook.config as config_module
import tldw_chatbook.UI.Screens.change_review_screen as change_review_module
import tldw_chatbook.Workspaces.git_workspace as git_workspace_module
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.change_review_screen import (
    COMMIT_RUN_ACTIVE_REFUSAL,
    CURRENT_MODE_SENTINEL,
    GIT_ACTION_WORKER_GROUP,
    PR_NO_UPSTREAM_REASON,
    PR_TURN_MODE_REFUSAL,
    PUSH_DETACHED_REASON,
    PUSH_NO_REMOTE_REASON,
    PUSH_OPTION_BRANCH_REASON,
    PUSH_TURN_MODE_REFUSAL,
    AgentRunsChangeReviewProvider,
    ChangeGitPushModal,
    ChangeReviewScreen,
)
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
from tldw_chatbook.Workspaces.git_workspace import PushResult as _PushResult

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)

#: Spec §6 / AC #2: the PR refusal for an unsupported host must NAME them.
SUPPORTED_PR_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org")


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


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


def _init_bare(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return path


def _bare_sha(bare: Path, ref: str = "main") -> str | None:
    """The BARE remote's own ref — the only proof a push really landed."""
    proc = subprocess.run(
        ["git", "--git-dir", str(bare), "rev-parse", ref],
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _bare_log(bare: Path, ref: str = "main") -> list[str]:
    proc = subprocess.run(
        ["git", "--git-dir", str(bare), "log", "--format=%s", ref],
        capture_output=True,
        text=True,
    )
    return [line for line in proc.stdout.splitlines() if line] if proc.returncode == 0 else []


def _patch_git_actions(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    """Pin `[change_review] git_actions` to ``value`` for this test."""

    def fake(section, key=None, default=None):
        if section == "change_review" and key == "git_actions":
            return value
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake)


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


async def _wait_for(pilot, predicate, what: str, timeout: float = 20.0):
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


async def _wait_idle(pilot, app, group: str) -> None:
    def settled():
        live = [
            w
            for w in app.workers
            if w.group == group and w.state.name in ("PENDING", "RUNNING")
        ]
        return not live

    await _wait_for(pilot, lambda: settled() or None, f"{group} worker idle")


def _current_mode_landed(screen) -> bool:
    """Whether a current-mode LOAD has landed (not merely been selected).

    Same race as the commit suite's helper (T7 fix round 1, proven with an
    instrumented probe): setting the ``Select``'s value flips
    ``_current_mode_active()`` SYNCHRONOUSLY while the load is dispatched
    later on the message pump, so waiting on the selection or on an idle
    worker group can both return before anything has been read.
    """
    if screen._leaves:
        return all(row.get("kind") == "git_current" for row, _c in screen._leaves)
    text = screen.diff_pane_text()
    return "working tree clean" in text or "working tree unavailable" in text


async def _enter_current_mode(pilot, app) -> ChangeReviewScreen:
    """Open the screen and switch it to the REAL working tree."""
    screen = await _open_screen(pilot, app)
    await _wait_for(
        pilot, lambda: screen.git_detection_settled or None, "git detection landed"
    )
    turn_select = await _ready_select(pilot, screen, "#change-review-turn-select")
    turn_select.value = CURRENT_MODE_SENTINEL
    await _wait_for(
        pilot,
        lambda: screen._current_mode_active() or None,
        "the current-mode selection",
    )
    await _wait_for(
        pilot,
        lambda: _current_mode_landed(screen) or None,
        "the working tree to LAND (never merely to be selected)",
    )
    await _wait_idle(pilot, app, "change-review-current")
    await pilot.pause()
    return screen


async def _wait_for_push_modal(pilot, app, what: str = "the push modal"):
    modal = await _wait_for(
        pilot,
        lambda: app.screen if isinstance(app.screen, ChangeGitPushModal) else None,
        what,
    )
    await _wait_for(
        pilot,
        lambda: bool(modal.query("#change-git-push-yes")) or None,
        f"{what} to finish composing",
    )
    return modal


def _capture_notifies(app) -> list[str]:
    notes: list[str] = []
    app.notify = lambda *a, **kw: notes.append(str(a[0]) if a else "")
    return notes


async def _wait_for_note(pilot, notes: list[str], needle: str) -> str:
    return await _wait_for(
        pilot,
        lambda: next((n for n in notes if needle in n), None),
        f"a notification containing {needle!r} (got {notes!r})",
    )


def _select_values(select: Select) -> list[str]:
    """A `Select`'s real option values (`Select.NULL` padding excluded)."""
    return [value for _label, value in select._options if isinstance(value, str)]


async def _ready_select(pilot, node, selector: str) -> Select:
    """Wait for a ``Select``'s OWN internals before driving its value.

    Not a defensive pause: assigning ``Select.value`` runs Textual's
    ``_watch_value`` SYNCHRONOUSLY, and that watcher reaches into the
    widget's internal ``SelectCurrent`` → ``#label`` ``Static``. On a
    just-pushed modal that grandchild can still be mid-compose, and the
    assignment then dies with ``NoMatches`` inside Textual rather than
    anywhere in our code. Observed 1 run in 8 before this wait existed
    (traceback: ``_select.py:615 select_current.update(prompt)``).

    Args:
        pilot: The test pilot.
        node: The modal or screen owning the selector.
        selector: The ``Select``'s selector.

    Returns:
        The composed ``Select``.
    """
    select = node.query_one(selector, Select)
    await _wait_for(
        pilot,
        lambda: bool(select.query("#label")) or None,
        f"{selector} to finish composing its own children",
    )
    return select


def _executable_source(module) -> str:
    """``module``'s source with COMMENTS AND STRINGS removed.

    A plain substring search over the file cannot tell a call from the
    docstring that forbids it — this module's own docstrings say the words
    "webbrowser.open" precisely to record the ban. Tokenizing and dropping
    comment/string tokens leaves only what actually executes, so the
    assertion below fails on a real call and never on prose about one.
    """
    import io
    import tokenize

    kept: list[str] = []
    for token in tokenize.generate_tokens(
        io.StringIO(inspect.getsource(module)).readline
    ):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def _static_text(screen, selector: str) -> str:
    renderable = screen.query_one(selector, Static).renderable
    if isinstance(renderable, Text):
        return renderable.plain
    return str(renderable)


def _make_provider(tmp_path, conversation_id, **kwargs):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    return (
        AgentRunsChangeReviewProvider(
            db=db, service=service, conversation_id=conversation_id, **kwargs
        ),
        db,
        service,
    )


def _repo_with_remote(tmp_path, name: str = "repo") -> tuple[Path, Path]:
    """A real repo with one unpushed commit and a real bare `origin`."""
    repo = _init_repo(tmp_path / name)
    bare = _init_bare(tmp_path / f"{name}-remote.git")
    _git(repo, "remote", "add", "origin", str(bare))
    return repo, bare


def _spy_on_git_argv(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Record every git argv the engine builds — real git still runs.

    This is an OBSERVER, not a mock: the wrapped function is called and its
    real result returned, so the repository state the assertions read back
    is produced by actual git.
    """
    captured: list[tuple[str, ...]] = []
    real = git_workspace_module._run_user_git

    def spy(root, *args, **kwargs):
        captured.append(tuple(args))
        return real(root, *args, **kwargs)

    monkeypatch.setattr(git_workspace_module, "_run_user_git", spy)
    return captured


# ---------------------------------------------------------------------------
# The buttons and their disabled reasons (spec §6, §8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_and_pr_are_offered_only_in_current_mode(monkeypatch, tmp_path):
    """Spec §6: both act on the REAL repository, never on a recorded turn."""
    _patch_git_actions(monkeypatch, True)
    repo, _bare = _repo_with_remote(tmp_path)
    provider, _db, _service = _make_provider(tmp_path, "conv-offered")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for(
            pilot, lambda: screen.git_detection_settled or None, "detection"
        )
        push_btn = screen.query_one("#change-review-git-push-btn", Button)
        pr_btn = screen.query_one("#change-review-git-pr-btn", Button)
        assert push_btn.display is False, "push must not be offered on a turn view"
        assert pr_btn.display is False, "PR must not be offered on a turn view"

        # ...and the keyboard path refuses with copy rather than acting.
        notes = _capture_notifies(app)
        screen.action_git_push()
        screen.action_git_pr()
        await pilot.pause()
        assert any(PUSH_TURN_MODE_REFUSAL in note for note in notes), notes
        assert any(PR_TURN_MODE_REFUSAL in note for note in notes), notes
        assert app.screen is screen, "no modal may open outside current mode"

        await _enter_current_mode(pilot, app)
        assert push_btn.display is True
        assert push_btn.disabled is False, "a repo with a remote must offer push"


@pytest.mark.asyncio
async def test_push_is_disabled_with_a_reason_without_a_remote(
    monkeypatch, tmp_path
):
    """AC #2 / spec §6: no remote ⇒ disabled, and it says WHY."""
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "no_remote")
    provider, _db, _service = _make_provider(tmp_path, "conv-no-remote")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        push_btn = screen.query_one("#change-review-git-push-btn", Button)
        assert push_btn.display is True, "the affordance is present, not hidden"
        assert push_btn.disabled is True
        assert PUSH_NO_REMOTE_REASON in str(push_btn.tooltip), (
            f"the disabled button must say why; got {push_btn.tooltip!r}"
        )

        notes = _capture_notifies(app)
        screen.action_git_push()
        await pilot.pause()
        await pilot.pause()
        assert notes, "the keyboard path must refuse audibly, never silently"
        assert PUSH_NO_REMOTE_REASON in notes[0]
        assert app.screen is screen, "no modal may open with nowhere to push"


@pytest.mark.asyncio
async def test_push_is_disabled_with_a_reason_on_a_detached_head(
    monkeypatch, tmp_path
):
    """Spec §6: detached HEAD ⇒ disabled with "no branch checked out"."""
    _patch_git_actions(monkeypatch, True)
    repo, _bare = _repo_with_remote(tmp_path)
    _git(repo, "checkout", "-q", "--detach")
    provider, _db, _service = _make_provider(tmp_path, "conv-detached")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        push_btn = screen.query_one("#change-review-git-push-btn", Button)
        assert push_btn.disabled is True
        assert PUSH_DETACHED_REASON in str(push_btn.tooltip), (
            f"got {push_btn.tooltip!r}"
        )

        notes = _capture_notifies(app)
        screen.action_git_push()
        await pilot.pause()
        await pilot.pause()
        assert notes and PUSH_DETACHED_REASON in notes[0]


@pytest.mark.asyncio
async def test_push_is_not_refused_during_a_run_while_commit_is(
    monkeypatch, tmp_path
):
    """Spec §6's explicit contrast with §5, asserted SIDE BY SIDE.

    Push only ships already-committed state (the working tree is
    untouched), so an active run must NOT block it — while a commit, which
    writes the tree the run is still editing, must.
    """
    _patch_git_actions(monkeypatch, True)
    repo, _bare = _repo_with_remote(tmp_path)
    (repo / "a.txt").write_text("changed by the run\n")
    provider, _db, _service = _make_provider(
        tmp_path, "conv-run-active", run_active=lambda: True
    )
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        notes = _capture_notifies(app)

        # COMMIT: refused, audibly, with no modal.
        screen.action_git_commit()
        await pilot.pause()
        await pilot.pause()
        assert any(COMMIT_RUN_ACTIVE_REFUSAL in note for note in notes), notes
        assert app.screen is screen, "an active run must not open a commit modal"

        # PUSH: the very same active run, and it goes straight through.
        screen.action_git_push()
        modal = await _wait_for_push_modal(pilot, app)
        assert isinstance(modal, ChangeGitPushModal)
        assert not any(COMMIT_RUN_ACTIVE_REFUSAL in note for note in notes[1:]), (
            f"push must not borrow commit's run-active refusal; got {notes!r}"
        )


# ---------------------------------------------------------------------------
# e2e against a REAL bare remote (spec §6, §9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_push_sets_upstream_and_refreshes_the_ahead_count(
    monkeypatch, tmp_path
):
    """The AC's named case, end to end through the BUTTON.

    No upstream ⇒ `push -u <remote> <branch>`; the BARE remote's ref really
    moves; the header's ahead/upstream line re-reads to ↑0 against the new
    upstream; a second push reports "up to date" instead of lying.
    """
    _patch_git_actions(monkeypatch, True)
    repo, bare = _repo_with_remote(tmp_path)
    provider, _db, _service = _make_provider(tmp_path, "conv-e2e")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        assert _bare_sha(bare) is None, "the bare remote starts empty"
        banner_before = _static_text(screen, "#change-review-banner")
        assert "no upstream" in banner_before, banner_before

        notes = _capture_notifies(app)
        screen.query_one("#change-review-git-push-btn", Button).press()
        modal = await _wait_for_push_modal(pilot, app)
        assert str(repo.resolve()) in _static_text(modal, "#change-git-push-target"), (
            "the modal must NAME the repository it will push from"
        )
        modal.query_one("#change-git-push-yes", Button).press()

        await _wait_for_note(pilot, notes, "Pushed")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)

        assert _bare_sha(bare) == _git(repo, "rev-parse", "HEAD"), (
            "the BARE remote's ref must really carry our commit"
        )
        assert _bare_log(bare) == ["base"]
        assert _git(repo, "rev-parse", "--abbrev-ref", "@{upstream}") == "origin/main", (
            "a first push must have set the upstream (`-u`)"
        )

        # The header re-reads: the mode now knows about the upstream.
        await _wait_for(
            pilot,
            lambda: (
                "origin/main" in _static_text(screen, "#change-review-banner")
            )
            or None,
            "the header to re-read the new upstream",
        )
        banner_after = _static_text(screen, "#change-review-banner")
        assert "↑0" in banner_after, (
            f"the ahead count must refresh after the push; got {banner_after!r}"
        )

        # Second push: nothing to send, and it says so rather than "Pushed".
        notes.clear()
        await _wait_idle(pilot, app, "change-review-current")
        screen.query_one("#change-review-git-push-btn", Button).press()
        modal2 = await _wait_for_push_modal(pilot, app, "the second push modal")
        modal2.query_one("#change-git-push-yes", Button).press()
        note = await _wait_for_note(pilot, notes, "up to date")
        assert "Pushed" not in note, f"a no-op push must not claim a push; got {note!r}"


@pytest.mark.asyncio
async def test_a_non_fast_forward_rejection_is_surfaced_and_never_forced(
    monkeypatch, tmp_path
):
    """AC #3 / spec §6: git's own refusal, verbatim — and NEVER `--force`.

    The divergence is real: a SECOND clone commits and pushes first, so our
    push is genuinely rejected by git rather than by anything we arranged.
    """
    _patch_git_actions(monkeypatch, True)
    bare = _init_bare(tmp_path / "shared.git")
    repo = _init_repo(tmp_path / "ours")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-q", "-u", "origin", "main")

    other = tmp_path / "theirs"
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(other)],
        capture_output=True,
        text=True,
        check=True,
    )
    _git(other, "config", "user.email", "o@o")
    _git(other, "config", "user.name", "o")
    (other / "a.txt").write_text("theirs\n")
    _git(other, "commit", "-qam", "theirs first")
    _git(other, "push", "-q", "origin", "main")
    remote_head_before = _bare_sha(bare)

    # Our own divergent commit.
    (repo / "a.txt").write_text("ours\n")
    _git(repo, "commit", "-qam", "ours second")

    provider, _db, _service = _make_provider(tmp_path, "conv-nonff")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        captured = _spy_on_git_argv(monkeypatch)
        pushed: list[object] = []
        real_push = provider.push_current

        def spy_push(root, info, remote):
            result = real_push(root, info, remote)
            pushed.append(result)
            return result

        provider.push_current = spy_push
        notes = _capture_notifies(app)

        screen.query_one("#change-review-git-push-btn", Button).press()
        modal = await _wait_for_push_modal(pilot, app)
        modal.query_one("#change-git-push-yes", Button).press()
        note = await _wait_for_note(pilot, notes, "rejected")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)

        assert "Pushed" not in note, f"a rejected push must not read as one: {note!r}"
        assert _bare_sha(bare) == remote_head_before, (
            "a rejected push must leave the remote exactly where it was"
        )
        # The engine's own detail reaches the user UNCHANGED — the screen
        # surfaces it, never re-maps or re-summarizes it (which is also how
        # the engine's credential-helper hint gets through).
        assert pushed and pushed[0].state == "failed"
        assert pushed[0].detail, "the engine must carry git's excerpt"
        assert pushed[0].detail in note, (
            f"the screen must surface the engine's detail verbatim; "
            f"detail={pushed[0].detail!r} note={note!r}"
        )

        push_argvs = [argv for argv in captured if argv and argv[0] == "push"]
        assert push_argvs, f"no push argv was captured; got {captured!r}"
        for argv in captured:
            assert "--force" not in argv, argv
            assert "--force-with-lease" not in argv, argv
            assert not any(a.startswith("--force") for a in argv), argv


@pytest.mark.asyncio
async def test_the_remote_select_appears_and_targets_the_chosen_remote(
    monkeypatch, tmp_path
):
    """Spec §6: no upstream + >1 remote ⇒ a `Select`, and it is obeyed."""
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "two_remotes")
    origin = _init_bare(tmp_path / "origin.git")
    backup = _init_bare(tmp_path / "backup.git")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "remote", "add", "backup", str(backup))
    provider, _db, _service = _make_provider(tmp_path, "conv-remotes")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        notes = _capture_notifies(app)
        screen.query_one("#change-review-git-push-btn", Button).press()
        modal = await _wait_for_push_modal(pilot, app)

        remote_select = await _ready_select(pilot, modal, "#change-git-push-remote")
        assert remote_select.display is True, (
            "two remotes and no upstream must offer a choice, not guess"
        )
        options = _select_values(remote_select)
        assert set(options) == {"origin", "backup"}, options

        # Deliberately the option that is NOT the default. `git remote -v`
        # lists remotes alphabetically, so "backup" is `remotes[0]` — a
        # mutation that ignores the Select entirely and takes the first
        # remote would PASS a test that picked it (verified: it did).
        bares = {"origin": origin, "backup": backup}
        chosen, unchosen = options[-1], options[0]
        assert chosen != unchosen, options

        remote_select.value = chosen
        await pilot.pause()
        modal.query_one("#change-git-push-yes", Button).press()
        await _wait_for_note(pilot, notes, "Pushed")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)

        assert _bare_sha(bares[chosen]) == _git(repo, "rev-parse", "HEAD"), (
            f"the CHOSEN remote ({chosen}) must be the one that received the push"
        )
        assert _bare_sha(bares[unchosen]) is None, (
            f"the unchosen remote ({unchosen}) must be untouched"
        )
        assert (
            _git(repo, "rev-parse", "--abbrev-ref", "@{upstream}") == f"{chosen}/main"
        )


@pytest.mark.asyncio
async def test_a_single_remote_is_pushed_without_asking(monkeypatch, tmp_path):
    """The common case: one remote ⇒ no `Select`, no guessing to do."""
    _patch_git_actions(monkeypatch, True)
    repo, _bare = _repo_with_remote(tmp_path)
    provider, _db, _service = _make_provider(tmp_path, "conv-one-remote")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        screen.query_one("#change-review-git-push-btn", Button).press()
        modal = await _wait_for_push_modal(pilot, app)
        assert modal.query_one("#change-git-push-remote", Select).display is False
        assert "origin" in _static_text(modal, "#change-git-push-summary"), (
            "the modal must still NAME the remote it will push to"
        )


@pytest.mark.asyncio
async def test_the_root_select_targets_one_repository_of_several(
    monkeypatch, tmp_path
):
    """Spec §6's multi-root rule, on the CLEAN tree where it is reachable.

    Two detected repositories and no focused leaf (both clean) — the modal
    must carry a root `Select`, and only the chosen repository's remote may
    move.
    """
    _patch_git_actions(monkeypatch, True)
    repo_a, bare_a = _repo_with_remote(tmp_path, "alpha")
    repo_b, bare_b = _repo_with_remote(tmp_path, "beta")
    provider, _db, _service = _make_provider(tmp_path, "conv-multi")
    app = _Harness(provider, workspace_roots=[str(repo_a), str(repo_b)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        assert screen._leaves == [], "both trees are clean — no focused leaf"
        notes = _capture_notifies(app)
        screen.query_one("#change-review-git-push-btn", Button).press()
        modal = await _wait_for_push_modal(pilot, app)

        root_select = await _ready_select(pilot, modal, "#change-git-push-root")
        options = _select_values(root_select)
        assert set(options) == {str(repo_a.resolve()), str(repo_b.resolve())}, options

        root_select.value = str(repo_b.resolve())
        await pilot.pause()
        assert str(repo_b.resolve()) in _static_text(
            modal, "#change-git-push-target"
        ), "the modal must re-name the root it now acts on"

        modal.query_one("#change-git-push-yes", Button).press()
        await _wait_for_note(pilot, notes, "Pushed")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)

        assert _bare_sha(bare_b) == _git(repo_b, "rev-parse", "HEAD")
        assert _bare_sha(bare_a) is None, (
            "the repository the user did NOT choose must be untouched"
        )


@pytest.mark.asyncio
async def test_escape_cancels_the_push_modal_and_pushes_nothing(
    monkeypatch, tmp_path
):
    """The confirm gate is real: an abandoned dialog writes nothing."""
    _patch_git_actions(monkeypatch, True)
    repo, bare = _repo_with_remote(tmp_path)
    provider, _db, _service = _make_provider(tmp_path, "conv-cancel")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        screen.query_one("#change-review-git-push-btn", Button).press()
        await _wait_for_push_modal(pilot, app)
        await pilot.press("escape")
        await _wait_for(
            pilot,
            lambda: (app.screen is screen) or None,
            "the modal to close",
        )
        await pilot.pause()
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        assert _bare_sha(bare) is None, "a cancelled push must push nothing"


# ---------------------------------------------------------------------------
# Worker discipline (T7 carry-forwards)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_push_worker_never_rides_the_status_read_group(
    monkeypatch, tmp_path
):
    """A push and a status read must never be able to cancel each other.

    Both worker groups are exclusive, so sharing one would let a re-render
    cancel an in-flight push to a remote (and vice versa).
    """
    _patch_git_actions(monkeypatch, True)
    repo, _bare = _repo_with_remote(tmp_path)
    provider, _db, _service = _make_provider(tmp_path, "conv-group")
    app = _Harness(provider, workspace_roots=[str(repo)])
    release = threading.Event()
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        real_push = provider.push_current

        def blocking_push(root, info, remote):
            release.wait(15)
            return real_push(root, info, remote)

        provider.push_current = blocking_push
        notes = _capture_notifies(app)
        try:
            screen.query_one("#change-review-git-push-btn", Button).press()
            modal = await _wait_for_push_modal(pilot, app)
            modal.query_one("#change-git-push-yes", Button).press()
            live = await _wait_for(
                pilot,
                lambda: [
                    w
                    for w in app.workers
                    if w.state.name == "RUNNING"
                    and w.group == GIT_ACTION_WORKER_GROUP
                ]
                or None,
                "the blocked push worker",
            )
            assert GIT_ACTION_WORKER_GROUP != "change-review-current", (
                "the git-action group must be distinct from the status-read group"
            )
            assert not [
                w
                for w in app.workers
                if w.state.name in ("PENDING", "RUNNING")
                and w.group == "change-review-current"
            ], "the push must not be sitting in the status-read group"

            # The buttons are disabled for the duration (no double-dispatch).
            push_btn = screen.query_one("#change-review-git-push-btn", Button)
            commit_btn = screen.query_one("#change-review-git-commit-btn", Button)
            assert screen._git_busy is True
            assert push_btn.disabled is True
            assert commit_btn.disabled is True

            # A REAL status read dispatched over the in-flight push.
            screen._load_current_mode()
            await pilot.pause()
            assert live[0].state.name != "CANCELLED", (
                "a status read must never cancel an in-flight push"
            )
        finally:
            release.set()
        await _wait_for_note(pilot, notes, "Pushed")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        push_btn = screen.query_one("#change-review-git-push-btn", Button)
        assert push_btn.disabled is False, "the buttons must come back"
        assert screen._git_busy is False


# ---------------------------------------------------------------------------
# Open PR (spec §6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pr_is_disabled_until_the_branch_has_an_upstream(
    monkeypatch, tmp_path
):
    """Spec §6: no upstream ⇒ disabled with "push the branch first"."""
    _patch_git_actions(monkeypatch, True)
    repo, _bare = _repo_with_remote(tmp_path)
    provider, _db, _service = _make_provider(tmp_path, "conv-pr-none")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        pr_btn = screen.query_one("#change-review-git-pr-btn", Button)
        assert pr_btn.display is True
        assert pr_btn.disabled is True
        assert PR_NO_UPSTREAM_REASON in str(pr_btn.tooltip), (
            f"got {pr_btn.tooltip!r}"
        )

        opened: list[str] = []
        app.open_url = lambda url, **kw: opened.append(url)
        notes = _capture_notifies(app)
        screen.action_git_pr()
        await pilot.pause()
        await pilot.pause()
        assert notes and PR_NO_UPSTREAM_REASON in notes[0]
        assert opened == [], "nothing may be opened without an upstream"


@pytest.mark.asyncio
async def test_pr_opens_the_exact_compare_url_via_app_open_url(
    monkeypatch, tmp_path
):
    """Spec §6: the compare URL, opened through `app.open_url` only.

    The upstream is established with real git plumbing (`update-ref` +
    `--set-upstream-to`) against a real github remote URL, so the URL under
    test is built by the real engine from real refs.
    """
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "prrepo")
    _git(repo, "checkout", "-q", "-b", "feat/x")
    _git(repo, "remote", "add", "origin", "https://github.com/o/r.git")
    _git(repo, "update-ref", "refs/remotes/origin/feat/x", "HEAD")
    _git(repo, "branch", "--set-upstream-to=origin/feat/x")
    provider, _db, _service = _make_provider(tmp_path, "conv-pr-url")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        pr_btn = screen.query_one("#change-review-git-pr-btn", Button)
        assert pr_btn.disabled is False, (
            f"an upstream on a supported host must enable PR; {pr_btn.tooltip!r}"
        )

        opened: list[str] = []
        app.open_url = lambda url, **kw: opened.append(url)
        pr_btn.press()
        await _wait_for(pilot, lambda: opened or None, "the browser open")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        assert opened == ["https://github.com/o/r/compare/feat/x?expand=1"], opened


def test_the_pr_opener_never_reaches_for_webbrowser() -> None:
    """Spec §6: `webbrowser.open`'s stdout can corrupt the TUI."""
    assert "webbrowser" not in _executable_source(change_review_module), (
        "the review screen must open URLs through `app.open_url` only"
    )
    assert "self.app.open_url(" in inspect.getsource(change_review_module)


@pytest.mark.asyncio
async def test_pr_is_disabled_naming_the_supported_hosts(monkeypatch, tmp_path):
    """AC #2: an unsupported host is refused BY NAME, not silently."""
    _patch_git_actions(monkeypatch, True)
    repo, bare = _repo_with_remote(tmp_path)
    # A real push to a real local bare remote: a filesystem URL is a
    # perfectly good git remote and a perfectly impossible PR host.
    _git(repo, "push", "-q", "-u", "origin", "main")
    provider, _db, _service = _make_provider(tmp_path, "conv-pr-host")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        pr_btn = screen.query_one("#change-review-git-pr-btn", Button)
        await _wait_for(
            pilot,
            lambda: (pr_btn.disabled and pr_btn.tooltip) or None,
            "the PR button's unsupported-host reason",
        )
        tooltip = str(pr_btn.tooltip)
        for host in SUPPORTED_PR_HOSTS:
            assert host in tooltip, f"{host!r} missing from {tooltip!r}"

        opened: list[str] = []
        app.open_url = lambda url, **kw: opened.append(url)
        notes = _capture_notifies(app)
        screen.action_git_pr()
        await pilot.pause()
        await pilot.pause()
        assert notes, "the keyboard path must refuse audibly"
        assert all(host in notes[0] for host in SUPPORTED_PR_HOSTS), notes
        assert opened == []


@pytest.mark.asyncio
async def test_pr_re_reads_rather_than_trusting_the_cached_link(
    monkeypatch, tmp_path
):
    """The never-pruned cache must not be able to open a wrong page.

    The affordance's PR state comes from the LAST load, so it can be
    describing a repository that has since changed under the app. Here the
    remote is repointed at an unsupported host after the load — the press
    must re-read, find git's live answer, and refuse BY NAME instead of
    opening the URL it had cached a moment ago.
    """
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "stale")
    _git(repo, "checkout", "-q", "-b", "feat/x")
    _git(repo, "remote", "add", "origin", "https://github.com/o/r.git")
    _git(repo, "update-ref", "refs/remotes/origin/feat/x", "HEAD")
    _git(repo, "branch", "--set-upstream-to=origin/feat/x")
    provider, _db, _service = _make_provider(tmp_path, "conv-pr-stale")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        pr_btn = screen.query_one("#change-review-git-pr-btn", Button)
        assert pr_btn.disabled is False, "the load saw a linkable github remote"

        # The world moves under the cache.
        _git(repo, "remote", "set-url", "origin", "https://git.example.invalid/o/r.git")

        opened: list[str] = []
        app.open_url = lambda url, **kw: opened.append(url)
        notes = _capture_notifies(app)
        pr_btn.press()
        note = await _wait_for_note(pilot, notes, SUPPORTED_PR_HOSTS[0])
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        assert all(host in note for host in SUPPORTED_PR_HOSTS), note
        assert opened == [], (
            f"a stale cached link must never be opened; got {opened!r}"
        )


@pytest.mark.asyncio
async def test_pr_asks_which_repository_when_several_qualify(
    monkeypatch, tmp_path
):
    """Spec §6's multi-root rule applies to PR too — never a dead control."""
    _patch_git_actions(monkeypatch, True)
    roots = []
    for name, owner in (("alpha", "o1"), ("beta", "o2")):
        repo = _init_repo(tmp_path / name)
        _git(repo, "remote", "add", "origin", f"https://github.com/{owner}/{name}.git")
        _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
        _git(repo, "branch", "--set-upstream-to=origin/main")
        roots.append(repo)
    provider, _db, _service = _make_provider(tmp_path, "conv-pr-multi")
    app = _Harness(provider, workspace_roots=[str(r) for r in roots])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        pr_btn = screen.query_one("#change-review-git-pr-btn", Button)
        assert pr_btn.disabled is False, (
            "two PR-able repositories must not disable the control"
        )
        opened: list[str] = []
        app.open_url = lambda url, **kw: opened.append(url)
        pr_btn.press()
        modal = await _wait_for_push_modal(pilot, app, "the PR target modal")
        root_select = await _ready_select(pilot, modal, "#change-git-push-root")
        assert modal.query_one("#change-git-push-remote", Select).display is False, (
            "a PR needs no remote choice — the upstream already names one"
        )
        root_select.value = str(roots[1].resolve())
        await pilot.pause()
        modal.query_one("#change-git-push-yes", Button).press()
        await _wait_for(pilot, lambda: opened or None, "the browser open")
        assert opened == ["https://github.com/o2/beta/compare/main?expand=1"], opened


# ---------------------------------------------------------------------------
# Argument injection through the remote NAME (T8 review, FIX 1)
# ---------------------------------------------------------------------------


def _point_upstream_at(repo: Path, remote_name: str, branch: str = "main") -> None:
    """Set ``branch``'s upstream to ``remote_name`` by writing `.git/config`.

    `git config branch.<b>.remote -- --force` cannot express this: git eats
    the `--` as the VALUE. Writing the config file is not a workaround for
    the test's convenience — it is the exact shape a hostile repository
    ships (a `.git/config` in a tarball, or a template a tool wrote), and
    the resulting state is ordinary, fully-functional git state: after this,
    `rev-parse --abbrev-ref @{upstream}` and `%(upstream:remotename)` both
    resolve, which is verified inline below.
    """
    config = repo / ".git" / "config"
    text = config.read_text()
    text = re.sub(
        r'\[branch "' + re.escape(branch) + r'"\]\n(\s*\w+ = .*\n)*',
        f'[branch "{branch}"]\n\tremote = {remote_name}\n'
        f"\tmerge = refs/heads/{branch}\n",
        text,
    )
    config.write_text(text)
    # The state must be REAL, or the test proves nothing about real git.
    assert (
        _git(repo, "for-each-ref", "--format=%(upstream:remotename)",
             f"refs/heads/{branch}")
        == remote_name
    )


@pytest.mark.asyncio
async def test_an_option_shaped_remote_can_never_force_push(monkeypatch, tmp_path):
    """A remote NAMED `--force` must be refused, not handed to git.

    This is argument injection, not a flag: `git remote add -- --force <url>`
    succeeds, the name lands in argv position 1 of `git push <remote>`, and
    git then reads it as the `--force` OPTION. Verified against real git
    before the fix existed — the push reported success and the second
    clone's commit was GONE from the bare remote (`+ 4fd1108...c1e7731
    main -> main (forced update)`), which the UI reported as
    "Pushed main to --force".

    Every existing no-force assertion missed this because nothing in our
    code ever writes the string `--force`; the repository supplies it.
    """
    _patch_git_actions(monkeypatch, True)
    bare = _init_bare(tmp_path / "shared.git")
    repo = _init_repo(tmp_path / "ours")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-q", "-u", "origin", "main")

    # A second clone lands a commit the remote must not lose.
    other = tmp_path / "theirs"
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(other)],
        capture_output=True,
        text=True,
        check=True,
    )
    _git(other, "config", "user.email", "o@o")
    _git(other, "config", "user.name", "o")
    (other / "a.txt").write_text("theirs\n")
    _git(other, "commit", "-qam", "theirs must survive")
    _git(other, "push", "-q", "origin", "main")
    theirs_sha = _bare_sha(bare)

    # The hostile bit: a remote literally named `--force`, tracked by the
    # branch. Our own divergent commit makes the push a NON-fast-forward,
    # so an honest push MUST be rejected — only a forced one can succeed.
    _git(repo, "remote", "add", "--", "--force", str(bare))
    _git(repo, "update-ref", "refs/remotes/--force/main", "HEAD")
    _point_upstream_at(repo, "--force")
    (repo / "a.txt").write_text("ours\n")
    _git(repo, "commit", "-qam", "ours diverges")

    provider, _db, _service = _make_provider(tmp_path, "conv-injection")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        captured = _spy_on_git_argv(monkeypatch)
        notes = _capture_notifies(app)

        screen.query_one("#change-review-git-push-btn", Button).press()
        modal = await _wait_for_push_modal(pilot, app)
        modal.query_one("#change-git-push-yes", Button).press()
        await _wait_for(pilot, lambda: notes or None, "the push outcome")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)

        # THE assertion: the other clone's commit is still the remote's tip.
        assert _bare_sha(bare) == theirs_sha, (
            "a force-push destroyed another clone's commit — the remote is "
            f"now {_bare_sha(bare)!r}, was {theirs_sha!r}"
        )
        assert not any("Pushed" in note for note in notes), (
            f"an injected push must never report success; got {notes!r}"
        )
        # Nothing option-shaped may reach git in a remote/repository slot.
        for argv in captured:
            if argv and argv[0] == "push":
                assert not any(
                    arg.startswith("-") and arg not in ("-u",) for arg in argv[1:]
                ), f"an option reached git's push argv: {argv!r}"


def test_the_engine_refuses_an_option_shaped_remote_directly(tmp_path) -> None:
    """The same guard at the ENGINE seam, independent of any UI.

    `push_current` is public and takes `remote` from its caller, so the
    refusal has to live there too — a future caller that does not go
    through the screen must not be able to reintroduce this.
    """
    from tldw_chatbook.Workspaces.git_workspace import (
        GitWorkspaceError,
        detect_git_workspace,
        push_current,
    )

    repo = _init_repo(tmp_path / "engine")
    bare = _init_bare(tmp_path / "engine-remote.git")
    _git(repo, "remote", "add", "origin", str(bare))
    info = detect_git_workspace(repo)

    for hostile in ("--force", "--mirror", "-f", "--delete"):
        with pytest.raises(GitWorkspaceError) as excinfo:
            push_current(repo, info, hostile)
        assert "unsupported remote name" in str(excinfo.value), excinfo.value
    assert _bare_sha(bare) is None, "no refused call may have pushed anything"

    # The ordinary name still works, so the guard is a filter, not a wall.
    assert push_current(repo, info, "origin").state == "pushed"


@pytest.mark.asyncio
async def test_the_ui_seam_only_ever_forwards_a_detected_remote(
    monkeypatch, tmp_path
):
    """Defense in depth: the screen forwards only names detection found.

    The engine's refusal is the real fix (the hostile name IS a detected
    remote there); this second layer covers the other direction — a modal
    result that names a remote no detection ever reported.
    """
    _patch_git_actions(monkeypatch, True)
    repo, bare = _repo_with_remote(tmp_path)
    provider, _db, _service = _make_provider(tmp_path, "conv-seam")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        seen: list[object] = []
        real_push = provider.push_current

        def spy_push(root, info, remote):
            seen.append(remote)
            return real_push(root, info, remote)

        provider.push_current = spy_push
        notes = _capture_notifies(app)

        info = screen._current_infos[str(repo.resolve())]
        screen._dispatch_push(
            {
                "action": "push",
                "root": str(repo.resolve()),
                "remote": "--force",
                "info": info,
            }
        )
        await _wait_for(pilot, lambda: notes or None, "the refusal")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)

        assert seen == [], (
            f"an undetected remote must never reach the engine; got {seen!r}"
        )
        assert _bare_sha(bare) is None, "nothing may have been pushed"
        assert not any("Pushed" in note for note in notes), notes


@pytest.mark.asyncio
async def test_push_targets_the_upstreams_remote_not_the_first_one(
    monkeypatch, tmp_path
):
    """FIX 2: with an upstream, the UPSTREAM's remote is the target.

    Two remotes and an upstream on the NON-first one — the case no earlier
    test had. Dropping `if info.upstream is not None: return None` from the
    modal's `_resolve_remote` survived mutation until this existed, and
    silently pushed to whichever remote git happened to list first.
    """
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "upstream_pick")
    # `git remote -v` lists alphabetically, so `alpha` is remotes[0] and
    # `zulu` — the upstream's remote — is deliberately NOT first.
    alpha = _init_bare(tmp_path / "alpha.git")
    zulu = _init_bare(tmp_path / "zulu.git")
    _git(repo, "remote", "add", "alpha", str(alpha))
    _git(repo, "remote", "add", "zulu", str(zulu))
    _git(repo, "push", "-q", "-u", "zulu", "main")
    (repo / "a.txt").write_text("second\n")
    _git(repo, "commit", "-qam", "second")
    zulu_before = _bare_sha(zulu)

    provider, _db, _service = _make_provider(tmp_path, "conv-upstream-pick")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        info = screen._current_infos[str(repo.resolve())]
        assert info.upstream_remote == "zulu", info
        assert [name for name, _url in info.remotes][0] == "alpha", info.remotes

        notes = _capture_notifies(app)
        screen.query_one("#change-review-git-push-btn", Button).press()
        modal = await _wait_for_push_modal(pilot, app)
        assert modal.query_one("#change-git-push-remote", Select).display is False, (
            "an upstream already names its remote — nothing to choose"
        )
        modal.query_one("#change-git-push-yes", Button).press()
        await _wait_for_note(pilot, notes, "Pushed")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)

        assert _bare_sha(zulu) == _git(repo, "rev-parse", "HEAD"), (
            "the UPSTREAM's remote must have received the push"
        )
        assert _bare_sha(zulu) != zulu_before
        assert _bare_sha(alpha) is None, (
            "the first-listed remote must be untouched — it is not the upstream"
        )


@pytest.mark.asyncio
async def test_a_superseded_push_landing_is_discarded(monkeypatch, tmp_path):
    """FIX 3: a stale push landing must not re-enable the git affordances.

    `_set_git_busy(False)` from a superseded landing would unlock all three
    buttons while a NEWER git action is still running — the exact
    double-dispatch the busy flag exists to prevent.
    """
    _patch_git_actions(monkeypatch, True)
    repo, _bare = _repo_with_remote(tmp_path)
    provider, _db, _service = _make_provider(tmp_path, "conv-superseded")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        notes = _capture_notifies(app)
        info = screen._current_infos[str(repo.resolve())]

        # A landing from a dispatch that has since been superseded.
        stale_token = object()
        screen._git_action_token = stale_token
        screen._set_git_busy(True)
        screen._git_action_token = object()  # a newer action takes over

        screen._land_push_result(
            stale_token,
            _PushResult("pushed"),
            info,
            "origin",
        )
        await pilot.pause()
        assert screen._git_busy is True, (
            "a superseded push landing must not unlock the affordances"
        )
        assert notes == [], f"nor announce an outcome; got {notes!r}"

        # The PR landing carries the same guard.
        screen._land_pr_url(stale_token, "https://github.com/o/r/compare/main")
        await pilot.pause()
        assert screen._git_busy is True
        assert notes == []

        # The LIVE token still lands, so the guard is not a blanket mute.
        screen._land_push_result(
            screen._git_action_token, _PushResult("pushed"), info, "origin"
        )
        await pilot.pause()
        assert screen._git_busy is False
        assert any("Pushed" in note for note in notes), notes


@pytest.mark.asyncio
async def test_an_unexpected_push_error_is_not_dressed_as_a_refusal(
    monkeypatch, tmp_path
):
    """FIX 4: a BUG must not read like the engine's honest refusal copy."""
    _patch_git_actions(monkeypatch, True)
    repo, _bare = _repo_with_remote(tmp_path)
    provider, _db, _service = _make_provider(tmp_path, "conv-bug")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)

        def exploding_push(root, info, remote):
            raise TypeError("a real bug inside the push path")

        provider.push_current = exploding_push
        seen: list[tuple] = []
        app.notify = lambda *a, **kw: seen.append((str(a[0]) if a else "", kw))

        screen.query_one("#change-review-git-push-btn", Button).press()
        modal = await _wait_for_push_modal(pilot, app)
        modal.query_one("#change-git-push-yes", Button).press()
        await _wait_for(pilot, lambda: seen or None, "the failure report")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)

        message, kwargs = seen[0]
        assert "Push could not run" in message, (
            f"an unexpected error must be marked as one; got {message!r}"
        )
        assert "a real bug inside the push path" in message, (
            "and must still carry the original text"
        )
        assert kwargs.get("severity") == "error", (
            f"a bug is not a warning-level refusal; got {kwargs!r}"
        )
        assert screen._git_busy is False, "the buttons must still come back"


# ---------------------------------------------------------------------------
# Argument injection through the BRANCH name (T8 re-review, FIX round 2)
# ---------------------------------------------------------------------------


def _bare_refs(bare: Path) -> list[str]:
    """Every ref in the bare remote — the whole surface a mirror push eats."""
    raw = subprocess.run(
        ["git", "--git-dir", str(bare), "for-each-ref", "--format=%(refname)"],
        capture_output=True,
        text=True,
    ).stdout
    return sorted(line for line in raw.splitlines() if line)


def _point_head_at(repo: Path, branch: str) -> None:
    """Check out a branch whose name git's porcelain refuses to create.

    `git checkout -b -- --mirror` is refused ("not a valid branch name"),
    but `update-ref` + a one-line `.git/HEAD` is ordinary plumbing and the
    result is ordinary git state — which is the point: a repository can
    simply SHIP this, and `symbolic-ref --short -q HEAD` (what detection
    reads) then reports the option-shaped name. Verified inline.
    """
    _git(repo, "update-ref", f"refs/heads/{branch}", "HEAD")
    (repo / ".git" / "HEAD").write_text(f"ref: refs/heads/{branch}\n")
    assert _git(repo, "symbolic-ref", "--short", "-q", "HEAD") == branch


def _shared_remote_with_a_colleagues_work(tmp_path) -> tuple[Path, Path, str]:
    """A bare remote carrying someone else's branch, tag and latest commit.

    Returns:
        ``(our repo, the bare remote, the colleague's HEAD sha)``. Our repo
        DIVERGES from the remote's `main`, so a mirror push can only
        succeed by force-rewinding it.
    """
    bare = _init_bare(tmp_path / "shared.git")
    theirs = tmp_path / "theirs"
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(theirs)],
        capture_output=True, text=True, check=True,
    )
    _git(theirs, "config", "user.email", "o@o")
    _git(theirs, "config", "user.name", "o")
    (theirs / "f.txt").write_text("base\n")
    _git(theirs, "add", "-A")
    _git(theirs, "commit", "-qm", "base")
    _git(theirs, "push", "-q", "origin", "main")
    _git(theirs, "branch", "release")
    _git(theirs, "push", "-q", "origin", "release")
    _git(theirs, "tag", "v1")
    _git(theirs, "push", "-q", "origin", "v1")

    ours = tmp_path / "ours"
    subprocess.run(
        ["git", "clone", "-q", str(bare), str(ours)],
        capture_output=True, text=True, check=True,
    )
    _git(ours, "config", "user.email", "t@t")
    _git(ours, "config", "user.name", "t")

    # The colleague moves on AFTER we cloned, so our main truly diverges.
    (theirs / "f.txt").write_text("theirs\n")
    _git(theirs, "commit", "-qam", "theirs must survive")
    _git(theirs, "push", "-q", "origin", "main")
    theirs_sha = _git(theirs, "rev-parse", "HEAD")

    (ours / "f.txt").write_text("ours\n")
    _git(ours, "commit", "-qam", "ours diverges")
    return ours, bare, theirs_sha


@pytest.mark.parametrize("hostile_branch", ["--mirror", "--all"])
def test_the_engine_refuses_an_option_shaped_branch_directly(
    tmp_path, hostile_branch
) -> None:
    """A branch NAMED `--mirror` must never reach `git push`'s argv.

    The second half of the same injection class as the remote name, and it
    was left open by fix round 1: `push_current`'s no-upstream branch builds
    `("push", "-u", <remote>, info.branch)`, and `info.branch` comes
    straight from `symbolic-ref` with no validator. My round-1 audit cleared
    this slot on the grounds that `check-ref-format` covers branch names —
    that was WRONG twice over: that validator only guards
    `commit_selected`'s NEW-branch path, and
    `git check-ref-format refs/heads/--mirror` exits **0** anyway.

    Verified against real git before the fix existed, with this exact
    fixture: `refs/heads/release` was DELETED from the bare remote,
    `refs/heads/main` was force-rewound off the colleague's commit, and
    junk `refs/remotes/origin/*` refs were pushed in. `--all` instead
    published every local branch, leaking private WIP.

    The assertion below is on the DESTRUCTION, not on the exception: the
    remote's entire ref list and its `main` must be byte-identical after.
    """
    from tldw_chatbook.Workspaces.git_workspace import (
        GitWorkspaceError,
        detect_git_workspace,
        push_current,
    )

    repo, bare, theirs_sha = _shared_remote_with_a_colleagues_work(tmp_path)
    _git(repo, "branch", "secret-wip")  # `--all` would publish this
    _point_head_at(repo, hostile_branch)
    refs_before = _bare_refs(bare)
    assert "refs/heads/release" in refs_before and "refs/tags/v1" in refs_before

    info = detect_git_workspace(repo)
    # The fixture must really be the dangerous shape, or this proves nothing.
    assert info.branch == hostile_branch, info
    assert info.upstream is None, "the `-u` argv branch is the one under test"

    # NOT `pytest.raises`: that would fail on the missing exception and
    # never reach the assertions that matter. The subject of this test is
    # the REMOTE's contents, so the call is made defensively and the
    # destruction is checked FIRST — pre-fix, the failure below is the
    # rewritten ref list, exactly as the reviewer reproduced it.
    raised: Exception | None = None
    try:
        push_current(repo, info, None)
    except GitWorkspaceError as exc:
        raised = exc

    refs_after = _bare_refs(bare)
    assert refs_after == refs_before, (
        "the remote's refs were rewritten — "
        f"before={refs_before!r} after={refs_after!r}"
    )
    assert _bare_sha(bare) == theirs_sha, (
        "the colleague's commit was destroyed by a forced update"
    )
    assert "refs/heads/secret-wip" not in refs_after, (
        "a private local branch was published to the shared remote"
    )
    assert raised is not None, "the push must be refused, not merely harmless"
    assert "unsupported branch name" in str(raised), raised


@pytest.mark.asyncio
async def test_a_legitimately_dashed_branch_name_still_pushes(
    monkeypatch, tmp_path
):
    """The guard is a filter, not a wall: only a LEADING dash is refused."""
    _patch_git_actions(monkeypatch, True)
    repo, bare = _repo_with_remote(tmp_path)
    _git(repo, "checkout", "-q", "-b", "feat/my-branch")
    provider, _db, _service = _make_provider(tmp_path, "conv-dashed")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        push_btn = screen.query_one("#change-review-git-push-btn", Button)
        assert push_btn.disabled is False, (
            f"an ordinary dashed branch must be pushable; {push_btn.tooltip!r}"
        )
        notes = _capture_notifies(app)
        push_btn.press()
        modal = await _wait_for_push_modal(pilot, app)
        modal.query_one("#change-git-push-yes", Button).press()
        await _wait_for_note(pilot, notes, "Pushed")
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)

        assert _bare_sha(bare, "feat/my-branch") == _git(
            repo, "rev-parse", "HEAD"
        )
        assert (
            _git(repo, "rev-parse", "--abbrev-ref", "@{upstream}")
            == "origin/feat/my-branch"
        )


@pytest.mark.asyncio
async def test_push_is_disabled_with_a_reason_for_an_option_shaped_branch(
    monkeypatch, tmp_path
):
    """Spec §8 half of the same fix: unavailable AND it says why.

    The engine refusal is the security guard; this keeps the button from
    looking live and failing on press (the reviewer's repro announced
    "Pushed --mirror to origin" from an ENABLED button with no tooltip).
    """
    _patch_git_actions(monkeypatch, True)
    repo, bare, theirs_sha = _shared_remote_with_a_colleagues_work(tmp_path)
    _point_head_at(repo, "--mirror")
    refs_before = _bare_refs(bare)
    provider, _db, _service = _make_provider(tmp_path, "conv-branch-ui")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        push_btn = screen.query_one("#change-review-git-push-btn", Button)
        notes = _capture_notifies(app)

        # The FULL gesture, driven to completion. Asserting only that the
        # button is greyed out would leave the destructive path untested:
        # pre-fix this opens a modal, and confirming it really does rewrite
        # the remote. Driving it means this test fails on the DESTRUCTION.
        screen.action_git_push()
        await pilot.pause()
        await pilot.pause()
        if isinstance(app.screen, ChangeGitPushModal):
            app.screen.query_one("#change-git-push-yes", Button).press()
            await pilot.pause()
        await _wait_idle(pilot, app, GIT_ACTION_WORKER_GROUP)
        await pilot.pause()

        refs_after = _bare_refs(bare)
        assert refs_after == refs_before, (
            "the remote's refs were rewritten through the UI — "
            f"before={refs_before!r} after={refs_after!r}"
        )
        assert _bare_sha(bare) == theirs_sha, (
            "the colleague's commit was destroyed through the UI"
        )
        assert not any("Pushed" in note for note in notes), notes

        # ...and it presents as unavailable rather than failing on press.
        assert push_btn.disabled is True, (
            "an option-shaped branch must not offer a live push button"
        )
        assert PUSH_OPTION_BRANCH_REASON in str(push_btn.tooltip), (
            f"got {push_btn.tooltip!r}"
        )
        assert notes and PUSH_OPTION_BRANCH_REASON in notes[0], notes


@pytest.mark.asyncio
async def test_a_bug_inside_push_submit_reports_itself(monkeypatch, tmp_path):
    """TASK-19703 AC #2, push side.

    Same shape as the commit modal's: the broad `except` that keeps a
    Textual handler from raising also swallowed genuine bugs, leaving the
    confirm button inert. This modal has no inline error Static, so the
    report goes through `notify`.
    """
    _patch_git_actions(monkeypatch, True)
    repo, _bare = _repo_with_remote(tmp_path)
    (repo / "a.txt").write_text("changed\n")
    provider, _db, _service = _make_provider(tmp_path, "conv-submit-bug")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        screen.action_git_push()
        modal = await _wait_for_push_modal(pilot, app)

        def _explode(*_a, **_kw):
            raise RuntimeError("SYNTHETIC-19703 push submit bug")

        monkeypatch.setattr(type(modal), "dismiss", _explode, raising=True)
        notes = _capture_notifies(app)
        modal._submit()
        await pilot.pause()

    assert any("Could not submit" in note for note in notes), (
        f"a bug in push submit must tell the user; got {notes!r}"
    )


# ---------------------------------------------------------------------------
# TASK-19701: the confirm dialog names where the push will actually land.
#
# `remote.<name>.pushurl` and `url.<other>.pushInsteadOf` both send a push to
# a DIFFERENT host than the fetch URL. Verified against real git: both are
# reflected in `git remote -v`'s (push) line and in `git remote get-url
# --push`, and detection already parses that line — so the effective URL was
# in hand all along and only the dialog was withholding it, naming the
# remote alias but never its destination. A terminal told the user more than
# the confirm dialog did, on the one screen whose job is to state what a
# button will do before they press it.
#
# Decision (AC #1): SURFACE, do not refuse. Both settings are legitimate,
# widely used git configuration (fetch over https, push over ssh is a
# standard corporate setup); refusing would break normal workflows to
# protect against nothing this app is entitled to override.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_dialog_names_the_effective_url_under_pushurl(
    monkeypatch, tmp_path
):
    """`remote.origin.pushurl` redirects the push to another host."""
    _patch_git_actions(monkeypatch, True)
    repo, _bare = _repo_with_remote(tmp_path)
    (repo / "a.txt").write_text("changed\n")
    _git(repo, "config", "remote.origin.pushurl", "ssh://git@pushtarget.invalid/r.git")

    provider, _db, _service = _make_provider(tmp_path, "conv-pushurl")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        screen.action_git_push()
        modal = await _wait_for_push_modal(pilot, app)
        text = "\n".join(
            str(w.renderable) for w in modal.query(Static)
        )

    assert "pushtarget.invalid" in text, (
        f"the dialog must name where the push actually lands; got {text!r}"
    )


@pytest.mark.asyncio
async def test_push_dialog_names_the_effective_url_under_pushinsteadof(
    monkeypatch, tmp_path
):
    """`url.<other>.pushInsteadOf` rewrites the push URL by prefix."""
    _patch_git_actions(monkeypatch, True)
    repo, bare = _repo_with_remote(tmp_path)
    (repo / "a.txt").write_text("changed\n")
    _git(
        repo,
        "config",
        "url.ssh://git@insteadof.invalid/.pushInsteadOf",
        str(bare),
    )

    provider, _db, _service = _make_provider(tmp_path, "conv-insteadof")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        screen.action_git_push()
        modal = await _wait_for_push_modal(pilot, app)
        text = "\n".join(
            str(w.renderable) for w in modal.query(Static)
        )

    assert "insteadof.invalid" in text, (
        f"the dialog must name the rewritten destination; got {text!r}"
    )


@pytest.mark.asyncio
async def test_push_dialog_shows_the_plain_url_without_a_redirect(
    monkeypatch, tmp_path
):
    """Control: with no redirect configured the destination is still named,
    so the disclosure is normal copy rather than a scary special case."""
    _patch_git_actions(monkeypatch, True)
    repo, bare = _repo_with_remote(tmp_path)
    (repo / "a.txt").write_text("changed\n")

    provider, _db, _service = _make_provider(tmp_path, "conv-plain")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        screen.action_git_push()
        modal = await _wait_for_push_modal(pilot, app)
        text = "\n".join(
            str(w.renderable) for w in modal.query(Static)
        )

    assert str(bare) in text, (
        f"the ordinary destination must be named too; got {text!r}"
    )


@pytest.mark.asyncio
async def test_push_dialog_names_every_destination_of_a_multi_pushurl_remote(
    monkeypatch, tmp_path
):
    """Qodo #2 (PR #1959): a remote may push to SEVERAL destinations.

    Naming only the first would make the dialog state a smaller
    destination set than reality — worse than saying nothing, since the
    user would believe it.
    """
    _patch_git_actions(monkeypatch, True)
    repo, bare = _repo_with_remote(tmp_path)
    (repo / "a.txt").write_text("changed\n")
    second = tmp_path / "second.git"
    _git(repo, "init", "-q", "--bare", str(second)) if False else subprocess.run(
        ["git", "init", "-q", "--bare", str(second)], check=True
    )
    _git(repo, "config", "--add", "remote.origin.pushurl", str(bare))
    _git(repo, "config", "--add", "remote.origin.pushurl", str(second))

    provider, _db, _service = _make_provider(tmp_path, "conv-multi-pushurl")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app)
        screen.action_git_push()
        modal = await _wait_for_push_modal(pilot, app)
        text = "\n".join(str(w.renderable) for w in modal.query(Static))

    assert str(bare) in text and str(second) in text, (
        f"BOTH destinations must be named; got {text!r}"
    )
