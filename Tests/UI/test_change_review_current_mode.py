"""The Review screen's `current` (real working-tree) mode — TASK-16801 T6.

Everything here is REAL: the shipped CSS bundle, the real
`AgentRunsChangeReviewProvider` (never a hand-rolled fake — the
fixture-invented-shapes trap has bitten this repo four separate times), a
FILE-backed `AgentRunsDB` (never `:memory:` — in-memory SQLite is
thread-affine and this mode lands its results off a worker thread), real
temp git repositories, and turns recorded by the real `ChangeTurnTracker`.

Two guards get pinned here that nothing else can prove:

- the **kill switch** (`[change_review] git_actions`) is the single check
  that makes the whole mode vanish — the present/absent pair below differs
  ONLY in that flag, so a broken guard fails one of them;
- the **stale-land guard**: the mode's status read runs on a worker, and a
  landing that arrives after the user has switched back to a recorded turn
  must be discarded rather than dropping working-tree rows into a turn
  view. That is driven here by blocking the worker body on a real
  `threading.Event`, not by hoping for a timing window.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest
from rich.text import Text
from textual.app import App
from textual.widgets import Select, Static, Tree

import tldw_chatbook.config as config_module
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.change_review_screen import (
    CURRENT_MODE_SENTINEL,
    AgentRunsChangeReviewProvider,
    ChangeReviewDiffPane,
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


def _patch_git_actions(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    """Pin `[change_review] git_actions` to ``value`` for this test.

    Every other lookup returns its own default untouched, so this never
    disturbs anything but the kill switch under test (and it keeps the
    tests independent of whatever the developer's real config says).
    """

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


async def _wait_for(pilot, predicate, what: str, timeout: float = 10.0):
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
    """Wait until the (worker-side) repo detection has landed."""
    await _wait_for(
        pilot,
        lambda: screen.git_detection_settled or None,
        "git detection landed",
    )
    return screen


async def _wait_idle(pilot, app, group: str) -> None:
    """Wait until no worker in ``group`` is pending/running."""

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


def _select_values(screen) -> list:
    return [value for _label, value in screen.turn_select_options()]


@pytest.fixture()
def git_review_fixture(tmp_path):
    """A real repo that is ALSO the recorded turns' workspace root.

    Working-tree state after the fixture: `a.txt` modified (tracked),
    `brand_new.txt` untracked, plus the two turn files left behind on disk
    (also untracked) — so `current` mode lists strictly more than any
    single turn does, which is what the mode-switch assertions key on.
    """
    repo = _init_repo(tmp_path / "repo")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    conv = "conv-1"
    run1 = db.create_run(conversation_id=conv, agent_kind="primary")
    run2 = db.create_run(conversation_id=conv, agent_kind="primary")
    _record_turn(
        db, tracker, repo, run1,
        lambda: (repo / "turn_one.txt").write_text("one\n"),
    )
    _record_turn(
        db, tracker, repo, run2,
        lambda: (repo / "turn_two.txt").write_text("two\n"),
    )
    (repo / "a.txt").write_text("changed\n")
    (repo / "brand_new.txt").write_text("hello from the working tree\n")
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id=conv
    )
    return provider, repo, db, run1, run2


@pytest.fixture()
def plain_review_fixture(tmp_path):
    """The same shape, but the workspace root is NOT a git repository."""
    root = tmp_path / "plain"
    root.mkdir()
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    conv = "conv-1"
    run1 = db.create_run(conversation_id=conv, agent_kind="primary")
    _record_turn(
        db, tracker, root, run1,
        lambda: (root / "turn_one.txt").write_text("one\n"),
    )
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id=conv
    )
    return provider, root, db, run1


# ---------------------------------------------------------------------------
# Pseudo-entry presence / absence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pseudo_entry_first_and_screen_still_opens_on_latest_turn(
    monkeypatch, git_review_fixture
):
    """The `current` entry is OFFERED first, never selected by default.

    Byte-compatible open behavior (spec §4): the pseudo-entry is listed
    ahead of the turns, but the screen still opens on the LATEST turn.
    """
    _patch_git_actions(monkeypatch, True)
    provider, repo, _db, _run1, run2 = git_review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)

        options = screen.turn_select_options()
        assert options, "the select must carry options"
        label, value = options[0]
        assert value == CURRENT_MODE_SENTINEL, (
            "the working-tree entry must be FIRST in the list"
        )
        assert "Working tree (current)" in label
        assert "main" in label, f"the label must name the branch: {label!r}"

        select = screen.query_one("#change-review-turn-select", Select)
        assert select.value == run2, (
            "the screen must still OPEN on the latest turn, never on the "
            "pseudo-entry"
        )
        labels = "\n".join(_tree_labels(screen.query_one(Tree)))
        assert "turn_two.txt" in labels
        assert "brand_new.txt" not in labels, (
            "opening on a turn must not show working-tree-only files"
        )


@pytest.mark.asyncio
async def test_pseudo_entry_absent_when_kill_switch_off(
    monkeypatch, git_review_fixture
):
    """Identical fixture to the present-case above; ONLY the switch flips.

    Also pins spec §8's stronger claim: off means NO DETECTION AT ALL --
    not "detection runs and its result is discarded". The counting shim
    sits at the provider boundary, so a screen that dispatched the probe
    anyway fails here even though the provider would have returned ``{}``.
    """
    _patch_git_actions(monkeypatch, False)
    provider, repo, _db, _run1, run2 = git_review_fixture
    detect_calls: list[list[str]] = []
    real_detect = provider.detect_git

    def counting_detect(roots):
        detect_calls.append(list(roots))
        return real_detect(roots)

    provider.detect_git = counting_detect

    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)
        await _wait_for(
            pilot, lambda: screen._leaves or None, "the turn's leaves"
        )

        assert CURRENT_MODE_SENTINEL not in _select_values(screen)
        assert screen.query_one(
            "#change-review-turn-select", Select
        ).value == run2
        assert detect_calls == [], (
            "the kill switch must stop detection from running at all; "
            f"got {detect_calls!r}"
        )


@pytest.mark.asyncio
async def test_detection_runs_when_the_kill_switch_is_on(
    monkeypatch, git_review_fixture
):
    """The other half of the pair above: switch ON ⇒ the probe DOES run,
    over the candidate roots, and offers the mode."""
    _patch_git_actions(monkeypatch, True)
    provider, repo, _db, _run1, _run2 = git_review_fixture
    detect_calls: list[list[str]] = []
    real_detect = provider.detect_git

    def counting_detect(roots):
        detect_calls.append(list(roots))
        return real_detect(roots)

    provider.detect_git = counting_detect

    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)

        assert detect_calls, "detection must run when the switch is on"
        assert str(repo) in detect_calls[0], (
            f"the recorded rows' root must be a candidate: {detect_calls!r}"
        )
        assert CURRENT_MODE_SENTINEL in _select_values(screen)


@pytest.mark.asyncio
async def test_prepending_the_entry_does_not_reload_the_open_turn(
    monkeypatch, git_review_fixture
):
    """Adding the pseudo-entry rebuilds the Select's options, which resets
    its value and posts `Select.Changed` for the restore too. That must
    NOT re-run the open turn's git work (the screen already carries a
    "loaded every turn twice" scar) or throw away the focused leaf."""
    _patch_git_actions(monkeypatch, True)
    provider, repo, _db, _run1, run2 = git_review_fixture
    loads: list[object] = []
    real_changed_files = provider.changed_files

    def counting_changed_files(row):
        loads.append(row.get("id"))
        return real_changed_files(row)

    provider.changed_files = counting_changed_files

    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)
        await _wait_for(
            pilot,
            lambda: CURRENT_MODE_SENTINEL in _select_values(screen) or None,
            "the pseudo-entry",
        )
        # Let any queued Select.Changed from the options rebuild drain.
        for _ in range(5):
            await pilot.pause(0.05)

        assert len(loads) == 1, (
            f"the open turn must be loaded exactly once; got {loads!r}"
        )
        assert screen.query_one(
            "#change-review-turn-select", Select
        ).value == run2


@pytest.mark.asyncio
async def test_pseudo_entry_absent_for_non_repo_root(
    monkeypatch, plain_review_fixture
):
    """AC #4: not a repository ⇒ the mode is simply not offered."""
    _patch_git_actions(monkeypatch, True)
    provider, root, _db, run1 = plain_review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)

        assert CURRENT_MODE_SENTINEL not in _select_values(screen)
        assert screen.query_one(
            "#change-review-turn-select", Select
        ).value == run1


@pytest.mark.asyncio
async def test_pseudo_entry_absent_without_candidate_roots(
    monkeypatch, tmp_path
):
    """No turns and no workspace roots ⇒ nothing to detect, no entry."""
    _patch_git_actions(monkeypatch, True)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="empty-conv"
    )
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)

        assert _select_values(screen) == []
        assert "No file changes recorded" in screen.diff_pane_text()


@pytest.mark.asyncio
async def test_workspace_roots_kwarg_contributes_a_candidate_root(
    monkeypatch, tmp_path, plain_review_fixture
):
    """Candidates are row roots ∪ the opener's live workspace roots."""
    _patch_git_actions(monkeypatch, True)
    provider, plain_root, _db, _run1 = plain_review_fixture
    repo = _init_repo(tmp_path / "live_repo")
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)

        assert CURRENT_MODE_SENTINEL in _select_values(screen), (
            "a repo passed only via workspace_roots must still offer the mode"
        )


@pytest.mark.asyncio
async def test_root_inside_repo_refusal_hides_entry_and_banners_the_reason(
    monkeypatch, tmp_path, plain_review_fixture
):
    """Confinement refusal: no entry, and the banner says WHY (spec §3/§8)."""
    _patch_git_actions(monkeypatch, True)
    provider, plain_root, _db, _run1 = plain_review_fixture
    repo = _init_repo(tmp_path / "outer_repo")
    inside = repo / "nested_workspace"
    inside.mkdir()
    app = _Harness(provider, workspace_roots=[str(inside)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)

        assert CURRENT_MODE_SENTINEL not in _select_values(screen)
        banner = screen.query_one("#change-review-banner", Static)
        text = _static_text(screen, "#change-review-banner")
        assert "workspace is inside a repository" in text, (
            f"the refusal reason must reach the banner; got {text!r}"
        )
        assert banner.display is True


# ---------------------------------------------------------------------------
# Entering the mode: real status, real diffs, real previews
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_current_mode_lists_working_tree_and_renders_real_diffs(
    monkeypatch, git_review_fixture
):
    _patch_git_actions(monkeypatch, True)
    provider, repo, _db, _run1, _run2 = git_review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)

        screen.query_one(
            "#change-review-turn-select", Select
        ).value = CURRENT_MODE_SENTINEL
        labels = await _wait_for(
            pilot,
            lambda: (
                lambda ls: ls if any("brand_new.txt" in item for item in ls) else None
            )(_tree_labels(screen.query_one(Tree))),
            "the working tree's files",
        )
        joined = "\n".join(labels)
        assert "a.txt" in joined, "a tracked modification must be listed"
        assert "brand_new.txt" in joined, "an untracked add must be listed"
        assert screen._current_mode_active() is True

        # Every leaf carries the pinned pseudo-row shape.
        for row, _change in screen._leaves:
            assert row["kind"] == "git_current"
            assert row["id"] == -1
            assert row["root"] == str(repo.resolve())

        # A TRACKED file renders the real `git diff HEAD` output.
        screen.select_file("a.txt")
        tracked = await _wait_for(
            pilot,
            lambda: (lambda t: t if "-base" in t else None)(
                screen.diff_pane_text()
            ),
            "a.txt's working-tree diff",
        )
        assert "+changed" in tracked

        # An UNTRACKED file renders the synthesized preview.
        screen.select_file("brand_new.txt")
        preview = await _wait_for(
            pilot,
            lambda: (lambda t: t if "new file:" in t else None)(
                screen.diff_pane_text()
            ),
            "brand_new.txt's untracked preview",
        )
        assert "new file: brand_new.txt" in preview
        assert "+hello from the working tree" in preview

        # Header/banner: branch + upstream state; totals like the turns get.
        banner_text = _static_text(screen, "#change-review-banner")
        assert "main" in banner_text
        assert "no upstream" in banner_text
        totals = _static_text(screen, "#change-review-totals")
        assert totals.startswith(f"{len(screen._leaves)} files")
        assert "+1" in totals and "−1" in totals, (
            f"tracked line counts must reach the totals line: {totals!r}"
        )


@pytest.mark.asyncio
async def test_clean_working_tree_enters_the_mode_and_says_so(
    monkeypatch, tmp_path
):
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "clean_repo")
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="conv-clean"
    )
    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)

        screen.query_one(
            "#change-review-turn-select", Select
        ).value = CURRENT_MODE_SENTINEL
        text = await _wait_for(
            pilot,
            lambda: (lambda t: t if "working tree clean" in t else None)(
                screen.diff_pane_text()
            ),
            "the clean-tree copy",
        )
        assert "working tree clean" in text
        assert screen._leaves == []


@pytest.mark.asyncio
async def test_unborn_head_renders_every_file_through_the_preview_path(
    monkeypatch, tmp_path
):
    """Unborn HEAD (spec §2 probe 4): `git diff HEAD` is a FATAL error.

    The count is taken at the PROVIDER boundary (a counting shim around
    `current_diff_text`), never by mocking git — a git mock would prove
    only that the mock was not called.
    """
    _patch_git_actions(monkeypatch, True)
    repo = _init_repo(tmp_path / "unborn_repo", commit=False)
    (repo / "first.txt").write_text("brand new tree\n")
    (repo / "second.txt").write_text("also new\n")
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="conv-unborn"
    )
    diff_calls: list[str] = []
    real_diff = provider.current_diff_text

    def counting_diff(root, change):
        diff_calls.append(change.path)
        return real_diff(root, change)

    provider.current_diff_text = counting_diff

    app = _Harness(provider, workspace_roots=[str(repo)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)

        label = screen.turn_select_options()[0][0]
        assert "no commits yet" in label, (
            f"an unborn branch must say so in the label: {label!r}"
        )

        screen.query_one(
            "#change-review-turn-select", Select
        ).value = CURRENT_MODE_SENTINEL
        await _wait_for(
            pilot,
            lambda: (
                lambda ls: ls if any("second.txt" in item for item in ls) else None
            )(_tree_labels(screen.query_one(Tree))),
            "the unborn tree's files",
        )

        for path in ("first.txt", "second.txt"):
            screen.select_file(path)
            text = await _wait_for(
                pilot,
                lambda p=path: (lambda t: t if f"new file: {p}" in t else None)(
                    screen.diff_pane_text()
                ),
                f"{path}'s preview",
            )
            assert f"new file: {path}" in text

        assert diff_calls == [], (
            "no file on an unborn branch may be routed through "
            f"`git diff HEAD`; got {diff_calls!r}"
        )


# ---------------------------------------------------------------------------
# Mode gating (spec §4.1 row-consumers table)
# ---------------------------------------------------------------------------


async def _enter_current_mode(pilot, app, provider) -> ChangeReviewScreen:
    screen = await _open_screen(pilot, app)
    await _wait_for_detection(pilot, screen)
    screen.query_one(
        "#change-review-turn-select", Select
    ).value = CURRENT_MODE_SENTINEL
    # Wait for the WORKING TREE's leaves specifically -- a bare
    # "_leaves is truthy" wait is satisfied instantly by the turn view the
    # screen opened on, and every assertion after it would be measuring
    # the wrong view.
    await _wait_for(
        pilot,
        lambda: (
            screen._leaves
            and all(
                row.get("kind") == "git_current" for row, _c in screen._leaves
            )
        )
        or None,
        "the working tree's leaves",
    )
    await _wait_for(
        pilot, lambda: screen._focused_leaf >= 0 or None, "a focused leaf"
    )
    return screen


@pytest.mark.asyncio
async def test_revert_actions_no_op_in_current_mode(
    monkeypatch, git_review_fixture
):
    _patch_git_actions(monkeypatch, True)
    provider, repo, _db, _run1, _run2 = git_review_fixture
    before = (repo / "a.txt").read_text()
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app, provider)
        notify_calls: list[tuple[tuple, dict]] = []
        app.notify = lambda *a, **kw: notify_calls.append((a, kw))

        screen.action_revert_file()
        await pilot.pause()
        assert notify_calls, "revert in current mode must notify, not no-op silently"
        assert "recorded turns" in str(notify_calls[0][0][0])
        assert app.screen is screen, "no confirm modal may be pushed"

        notify_calls.clear()
        screen.action_undo_all()
        await pilot.pause()
        assert notify_calls, "undo-all in current mode must notify"
        assert "recorded turns" in str(notify_calls[0][0][0])
        assert app.screen is screen

        assert (repo / "a.txt").read_text() == before, (
            "nothing on disk may be touched by a gated revert"
        )


@pytest.mark.asyncio
async def test_comment_paths_no_op_in_current_mode(
    monkeypatch, git_review_fixture
):
    """`C`, the button, and the diff pane's `c` all refuse — and the
    pseudo-row's `id=-1` never reaches the notes DB."""
    _patch_git_actions(monkeypatch, True)
    provider, repo, db, run1, run2 = git_review_fixture
    # A REAL pre-existing note, so "unchanged" is measured against a
    # non-empty table -- an assertion that only ever reads 0 rows would
    # pass even if the query were broken.
    provider.add_change_note(
        run_id=run2,
        root=str(repo),
        path="turn_two.txt",
        hunk_index=-1,
        hunk_header="",
        hunk_excerpt="",
        note="a note recorded against the turn",
    )
    before = len(provider.notes_for_run(run2))
    assert before == 1

    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app, provider)
        notify_calls: list[tuple[tuple, dict]] = []
        app.notify = lambda *a, **kw: notify_calls.append((a, kw))

        await screen.action_comment_file()
        await pilot.pause()
        assert notify_calls, "C in current mode must notify"
        assert "comments attach to recorded turns" in str(notify_calls[0][0][0])

        notify_calls.clear()
        screen.action_focus_diff()
        await pilot.pause()
        assert isinstance(
            app.focused, ChangeReviewDiffPane
        ), "the diff pane must hold focus for the `c` path"
        await pilot.press("c")
        await pilot.pause()
        assert notify_calls, "`c` in current mode must notify"
        assert "comments attach to recorded turns" in str(notify_calls[0][0][0])

        assert not screen.query(".change-review-comment-input"), (
            "no comment input may be mounted in current mode"
        )
        assert provider.notes_for_run(run1) == []
        assert len(provider.notes_for_run(run2)) == before, (
            "the pseudo row's id=-1 must never reach the notes DB"
        )
        assert screen._marked_diff_lines == set(), (
            "the notes-marker computation must short-circuit in current mode"
        )
        strip = screen.query_one("#change-review-notes-strip")
        assert strip.display is False


@pytest.mark.asyncio
async def test_notes_are_never_queried_in_current_mode(
    monkeypatch, git_review_fixture
):
    """The pseudo row has no snapshot id — the notes read must not run.

    Spec §4.1 names ``_current_mode_active()`` as the guard here, and this
    test exercises THAT guard specifically: ``_active_turn`` is forced back
    to a real turn while the Select still sits on the sentinel, so the
    ``_active_turn is None`` early return (which would otherwise mask the
    named guard entirely) cannot be what stops the read. Without that
    setup the assertion passes even with the mode guard deleted — verified
    — and the guard would silently rot the first time anything set
    ``_active_turn`` during current mode.
    """
    _patch_git_actions(monkeypatch, True)
    provider, repo, _db, _run1, run2 = git_review_fixture
    reads: list[str] = []
    real_notes = provider.notes_for_run

    def counting_notes(run_id):
        reads.append(run_id)
        return real_notes(run_id)

    provider.notes_for_run = counting_notes
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app, provider)
        turn = next(t for t in screen._turns if t.run_id == run2)
        screen._active_turn = turn
        assert screen._current_mode_active() is True, (
            "the Select must still be on the sentinel for this to test the "
            "mode guard rather than the active-turn guard"
        )
        reads.clear()
        screen.action_next_file()
        screen.action_previous_file()
        screen._refresh_notes_ui_for_focused_leaf()
        await pilot.pause()
        assert reads == [], (
            f"current mode must never query notes; got {reads!r}"
        )
        assert screen._marked_diff_lines == set()


# ---------------------------------------------------------------------------
# Stale-land guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_landing_is_discarded_after_switching_back_to_a_turn(
    monkeypatch, git_review_fixture
):
    """The bug this guard exists for: the worker lands working-tree rows
    into a turn view because the user switched back while it was in flight.

    Driven deterministically — the worker body blocks on a real event
    until after the Select has moved back to the turn.
    """
    _patch_git_actions(monkeypatch, True)
    provider, repo, _db, _run1, run2 = git_review_fixture
    gate = threading.Event()
    real_status = provider.current_status

    def blocked_status(root):
        gate.wait(timeout=15.0)
        return real_status(root)

    provider.current_status = blocked_status

    app = _Harness(provider)
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            screen = await _open_screen(pilot, app)
            await _wait_for_detection(pilot, screen)
            select = screen.query_one("#change-review-turn-select", Select)

            select.value = CURRENT_MODE_SENTINEL
            await pilot.pause()
            assert "Loading working tree" in screen.diff_pane_text(), (
                "entering the mode must show a loading state while the "
                "worker runs"
            )

            # Switch back to the recorded turn while the worker is blocked.
            select.value = run2
            await _wait_for(
                pilot,
                lambda: (
                    screen._active_turn is not None
                    and screen._active_turn.run_id == run2
                )
                or None,
                "the turn view reloaded",
            )

            gate.set()
            await _wait_idle(pilot, app, "change-review-current")
            await pilot.pause()

            labels = "\n".join(_tree_labels(screen.query_one(Tree)))
            assert "turn_two.txt" in labels
            assert "brand_new.txt" not in labels, (
                "a stale current-mode landing must NOT drop working-tree "
                "rows into a turn view"
            )
            assert all(
                row.get("kind") != "git_current" for row, _c in screen._leaves
            )
            assert screen._current_mode_active() is False
    finally:
        gate.set()


@pytest.mark.asyncio
async def test_landing_with_a_superseded_token_is_discarded(
    monkeypatch, git_review_fixture
):
    """A landing whose dispatch token has been superseded is dropped even
    while the mode is still selected (the second half of the guard)."""
    _patch_git_actions(monkeypatch, True)
    provider, repo, _db, _run1, _run2 = git_review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _enter_current_mode(pilot, app, provider)
        before = list(screen._leaves)
        assert before, "the mode must have landed real rows first"

        screen._land_current_mode(object(), [])
        await pilot.pause()

        assert screen._leaves == before, (
            "a superseded token's landing must not wipe the live view"
        )
        assert "working tree clean" not in screen.diff_pane_text()


@pytest.mark.asyncio
async def test_one_root_failing_between_detect_and_status_degrades_alone(
    monkeypatch, tmp_path
):
    """Per-root failure isolation: a root that vanishes (or is refused)
    between detection and status must not abort the other roots."""
    _patch_git_actions(monkeypatch, True)
    good = _init_repo(tmp_path / "good_repo")
    (good / "a.txt").write_text("changed\n")
    doomed = _init_repo(tmp_path / "doomed_repo")
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="conv-multi"
    )
    real_status = provider.current_status

    def failing_status(root):
        if Path(root).name == "doomed_repo":
            from tldw_chatbook.Workspaces.git_workspace import GitWorkspaceError

            raise GitWorkspaceError("repository vanished mid-load")
        return real_status(root)

    provider.current_status = failing_status

    app = _Harness(provider, workspace_roots=[str(doomed), str(good)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)
        screen.query_one(
            "#change-review-turn-select", Select
        ).value = CURRENT_MODE_SENTINEL

        labels = await _wait_for(
            pilot,
            lambda: (
                lambda ls: ls if any("a.txt" in item for item in ls) else None
            )(_tree_labels(screen.query_one(Tree))),
            "the surviving root's files",
        )
        assert "a.txt" in "\n".join(labels), (
            "the healthy root must still render"
        )
        banner = _static_text(screen, "#change-review-banner")
        assert "doomed_repo" in banner, (
            f"the failed root must be named honestly: {banner!r}"
        )
        assert "repository vanished mid-load" in banner


@pytest.mark.asyncio
async def test_every_root_failing_never_claims_the_tree_is_clean(
    monkeypatch, tmp_path
):
    """When EVERY root's status read fails there are no statuses to land --
    which must not be reported as an empty-but-healthy working tree.

    "working tree clean" is a positive claim ABOUT THE USER'S REPOSITORY;
    printing it next to a banner saying the tree could not be read is the
    pane asserting something false (spec §8's honesty requirement). The
    two surfaces have to agree.
    """
    _patch_git_actions(monkeypatch, True)
    doomed = _init_repo(tmp_path / "doomed_repo")
    (doomed / "unseen.txt").write_text("never read\n")
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="conv-doomed"
    )

    def failing_status(root):
        from tldw_chatbook.Workspaces.git_workspace import GitWorkspaceError

        raise GitWorkspaceError("repository vanished mid-load")

    provider.current_status = failing_status

    app = _Harness(provider, workspace_roots=[str(doomed)])
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _open_screen(pilot, app)
        await _wait_for_detection(pilot, screen)
        screen.query_one(
            "#change-review-turn-select", Select
        ).value = CURRENT_MODE_SENTINEL

        banner = await _wait_for(
            pilot,
            lambda: (
                lambda b: b if "repository vanished mid-load" in b else None
            )(_static_text(screen, "#change-review-banner")),
            "the failed root's banner line",
        )
        assert "doomed_repo" in banner

        pane = screen.diff_pane_text()
        assert "working tree clean" not in pane, (
            "the pane must not claim a clean tree when nothing could be "
            f"read; got {pane!r}"
        )
        assert "unavailable" in pane, (
            f"the pane must point at the failure instead; got {pane!r}"
        )
        assert screen._leaves == []
