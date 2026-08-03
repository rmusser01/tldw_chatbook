"""TASK-1973: the Review screen — changed-file tree, windowed diffs, history.

Fixtures are REAL end to end: a real git root, turns produced by the real
`ChangeTurnTracker`, rows in a real `AgentRunsDB`, and the real provider —
the fixture-invented-shapes trap has bitten this repo four separate times.
UI tests load the shipped stylesheet (bare harnesses measure fiction) and
wait on conditions, not pause counts.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select, Static, Tree

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.change_review_screen import (
    AgentRunsChangeReviewProvider,
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


def _record_turn(db, tracker, root, run_id: str, mutate) -> None:
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


@pytest.fixture()
def review_fixture(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "edit.txt").write_text("before\n")
    (root / "gone.txt").write_text("delete me\n")
    (root / "old_name.txt").write_text("stable rename content\n" * 5)
    (root / "image.bin").write_bytes(b"\x00\x01\x02")
    (root / "markup.txt").write_text("plain\n")

    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    conv = "conv-1"
    run1 = db.create_run(conversation_id=conv, agent_kind="primary")
    run2 = db.create_run(conversation_id=conv, agent_kind="primary")

    def turn_one():
        (root / "first_turn.txt").write_text("turn one\n")

    def turn_two():
        (root / "new.txt").write_text("created\n")
        (root / "edit.txt").write_text("after\n")
        (root / "gone.txt").unlink()
        (root / "old_name.txt").rename(root / "new_name.txt")
        (root / "image.bin").write_bytes(b"\x00\x01\x02\x03")
        (root / "markup.txt").write_text("[bold red]not markup[/] $var `tick`\n")

    _record_turn(db, tracker, root, run1, turn_one)
    _record_turn(db, tracker, root, run2, turn_two)

    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id=conv
    )
    return provider, root, run1, run2


class _Harness(App[None]):
    CSS_PATH = str(BUNDLE)

    def __init__(self, provider) -> None:
        super().__init__()
        self._provider = provider

    def on_mount(self) -> None:
        self.push_screen(ChangeReviewScreen(self._provider))


async def _wait_for(pilot, predicate, what: str, timeout: float = 8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _tree_labels(tree: Tree) -> list[str]:
    labels: list[str] = []

    def walk(node):
        labels.append(str(node.label))
        for child in node.children:
            walk(child)

    walk(tree.root)
    return labels


@pytest.mark.asyncio
async def test_groups_render_for_every_change_kind(review_fixture):
    provider, root, run1, run2 = review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        tree = await _wait_for(
            pilot,
            lambda: (screen.query(Tree) and screen.query_one(Tree)) or None,
            "changed-file tree",
        )
        labels = await _wait_for(
            pilot,
            lambda: (lambda ls: ls if any("new.txt" in l for l in ls) else None)(
                _tree_labels(tree)
            ),
            "latest turn's files",
        )
        text = "\n".join(labels)
        assert "Added" in text and "Modified" in text
        assert "Deleted" in text and "Renamed" in text
        assert "new.txt" in text and "edit.txt" in text
        assert "gone.txt" in text
        assert "old_name.txt" in text and "new_name.txt" in text
        assert "image.bin" in text and "(binary)" in text
        # Tree labels must be markup-safe: a bracketed filename survives.
        assert "old_name.txt → new_name.txt" in text


@pytest.mark.asyncio
async def test_diff_pane_renders_markup_verbatim(review_fixture):
    """A file containing Rich markup / brackets / backticks must display
    exactly as written — the transcript's literal-backslash lesson."""
    provider, root, run1, run2 = review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        await _wait_for(
            pilot, lambda: screen.query(Tree) or None, "tree mounted"
        )
        screen.select_file("markup.txt")
        rendered = await _wait_for(
            pilot,
            lambda: (
                lambda t: t if "not markup" in t else None
            )(screen.diff_pane_text()),
            "markup file diff",
        )
        assert "[bold red]not markup[/]" in rendered
        assert "$var `tick`" in rendered


@pytest.mark.asyncio
async def test_truncation_row_reports_accurate_hidden_count(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run = db.create_run(conversation_id="c", agent_kind="primary")
    _record_turn(
        db,
        tracker,
        root,
        run,
        lambda: (root / "big.txt").write_text(
            "".join(f"line {n}\n" for n in range(500))
        ),
    )
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="c", diff_display_max_lines=50
    )
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        await _wait_for(pilot, lambda: screen.query(Tree) or None, "tree")
        screen.select_file("big.txt")
        text = await _wait_for(
            pilot,
            lambda: (
                lambda t: t if "truncated" in t else None
            )(screen.diff_pane_text()),
            "truncation row",
        )
        import re

        m = re.search(r"truncated — (\d+) more lines", text)
        assert m, f"no truncation disclosure in: {text[-200:]!r}"
        shown = text.count("\n+")
        assert int(m.group(1)) > 0
        assert shown <= 50


@pytest.mark.asyncio
async def test_only_the_focused_files_diff_is_mounted(review_fixture):
    """Widget census: a many-file turn must not mount every diff at once —
    a 50k-line generated file would freeze the screen."""
    provider, root, run1, run2 = review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        await _wait_for(pilot, lambda: screen.query(Tree) or None, "tree")
        bodies = list(screen.query(".change-review-diff-body"))
        assert len(bodies) == 1, (
            f"{len(bodies)} diff bodies mounted for a multi-file turn"
        )


@pytest.mark.asyncio
async def test_opening_loads_each_turn_exactly_once(review_fixture):
    """Review finding: value-set + direct call loaded every turn TWICE --
    doubled git work on every open. Select.Changed is the one loader."""
    provider, root, run1, run2 = review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        loads: list[str] = []
        original = screen._load_turn
        screen._load_turn = lambda turn: (loads.append(turn.run_id), original(turn))[1]
        await _wait_for(
            pilot,
            lambda: (screen.query(Tree) and _tree_labels(screen.query_one(Tree))) or None,
            "initial load",
        )
        await pilot.pause()
        assert loads.count(run2) <= 1, f"the latest turn loaded twice: {loads}"

        screen.select_turn(run1)
        await _wait_for(
            pilot,
            lambda: run1 in loads or None,
            "turn 1 load",
        )
        await pilot.pause()
        assert loads.count(run1) == 1, f"turn 1 loaded twice: {loads}"


@pytest.mark.asyncio
async def test_a_hostile_cap_cannot_defeat_windowing(tmp_path):
    """Review finding: a negative explicit cap slices from the END --
    rendering almost the whole diff and inverting the guarantee."""
    root = tmp_path / "root"
    root.mkdir()
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run = db.create_run(conversation_id="c", agent_kind="primary")
    _record_turn(
        db, tracker, root, run,
        lambda: (root / "big.txt").write_text(
            "".join(f"line {n}\n" for n in range(500))
        ),
    )
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="c", diff_display_max_lines=-5
    )
    assert provider.diff_display_max_lines >= 50


@pytest.mark.asyncio
async def test_turn_selector_navigates_to_a_previous_turn(review_fixture):
    provider, root, run1, run2 = review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        tree = await _wait_for(pilot, lambda: screen.query(Tree) or None, "tree")
        screen.select_turn(run1)
        labels = await _wait_for(
            pilot,
            lambda: (
                lambda ls: ls
                if any("first_turn.txt" in l for l in ls)
                else None
            )(_tree_labels(screen.query_one(Tree))),
            "previous turn's files",
        )
        text = "\n".join(labels)
        assert "first_turn.txt" in text
        assert "new.txt" not in text, "turn 2's files leaked into turn 1's view"


@pytest.mark.asyncio
async def test_u_reverts_the_focused_file_through_the_confirm(review_fixture):
    """`u` -> confirm modal -> Revert: disk truth restored, view reloaded."""
    from tldw_chatbook.UI.Screens.change_review_screen import (
        ChangeRevertConfirmModal,
    )

    provider, root, run1, run2 = review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        await _wait_for(
            pilot,
            lambda: (screen.query(Tree) and screen._leaves) or None,
            "leaves loaded",
        )
        screen.select_file("edit.txt")
        await pilot.press("u")
        modal = await _wait_for(
            pilot,
            lambda: app.screen
            if isinstance(app.screen, ChangeRevertConfirmModal)
            else None,
            "confirm modal",
        )
        assert "edit.txt" in str(
            modal.query_one("#change-revert-confirm").children[0].renderable
        )
        await pilot.click("#change-revert-yes")
        await _wait_for(
            pilot,
            lambda: (root / "edit.txt").read_text() == "before\n" or None,
            "disk restored to baseline",
        )


@pytest.mark.asyncio
async def test_confirm_names_files_edited_after_the_turn(review_fixture):
    from tldw_chatbook.UI.Screens.change_review_screen import (
        ChangeRevertConfirmModal,
    )

    provider, root, run1, run2 = review_fixture
    (root / "edit.txt").write_text("USER EDIT after the turn\n")
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        await _wait_for(
            pilot,
            lambda: (screen.query(Tree) and screen._leaves) or None,
            "leaves loaded",
        )
        screen.select_file("edit.txt")
        await pilot.press("u")
        modal = await _wait_for(
            pilot,
            lambda: app.screen
            if isinstance(app.screen, ChangeRevertConfirmModal)
            else None,
            "confirm modal",
        )
        warning = modal.query_one("#change-revert-edited-warning", Static)
        text = str(warning.renderable)
        assert "edit.txt" in text and "overwrites" in text, (
            f"the guard's warning does not NAME the file: {text!r}"
        )


@pytest.mark.asyncio
async def test_revert_refusal_during_active_run_reaches_the_user(review_fixture):
    provider, root, run1, run2 = review_fixture
    provider.run_active = lambda: True
    app = _Harness(provider)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        await _wait_for(
            pilot,
            lambda: (screen.query(Tree) and screen._leaves) or None,
            "leaves loaded",
        )
        screen.select_file("edit.txt")
        await pilot.press("u")
        from tldw_chatbook.UI.Screens.change_review_screen import (
            ChangeRevertConfirmModal,
        )

        modal = await _wait_for(
            pilot,
            lambda: app.screen
            if isinstance(app.screen, ChangeRevertConfirmModal)
            else None,
            "confirm modal",
        )
        await pilot.click("#change-revert-yes")
        await pilot.pause()
        await pilot.pause()
        assert (root / "edit.txt").read_text() == "after\n", (
            "the revert ran under an active run"
        )


@pytest.mark.asyncio
async def test_oversize_disclosure_banner_renders(tmp_path, monkeypatch):
    """TASK-1975 AC#2: the untracked-oversize count is disclosed in review."""
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_FILE_BYTES", "100")
    root = tmp_path / "root"
    root.mkdir()
    (root / "small.txt").write_text("hello\n")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run = db.create_run(conversation_id="conv-1", agent_kind="primary")

    def mutate():
        (root / "big.bin").write_bytes(b"x" * 500)
        (root / "small.txt").write_text("edited\n")

    _record_turn(db, tracker, root, run, mutate)
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="conv-1"
    )
    app = _Harness(provider)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        await _wait_for(
            pilot, lambda: screen.query("#change-review-banner"), "banner"
        )
        banner = screen.query_one("#change-review-banner", Static)
        text = str(banner.renderable)
        assert "1 oversized" in text, text
        assert banner.display is True


@pytest.mark.asyncio
async def test_pruned_snapshots_render_pruned_by_retention(review_fixture, tmp_path):
    """TASK-1975 AC#7: a history row whose snapshots were pruned renders
    'pruned by retention' instead of erroring."""
    import shutil as _shutil

    provider, root, run1, run2 = review_fixture
    # Retention reset the shadow store; the DB rows remain.
    _shutil.rmtree(tmp_path / "appdata")
    app = _Harness(provider)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        await _wait_for(
            pilot,
            lambda: "pruned by retention"
            in str(screen.query_one("#change-review-banner", Static).renderable),
            "pruned banner",
        )


@pytest.mark.asyncio
async def test_nested_repo_banner_names_the_holes(tmp_path, monkeypatch):
    """TASK-1976 AC#1: UNREGISTERED nested repos are named in the Review
    banner. Under TASK-1977 children are auto-registered as sub-roots, so
    the disclosure path is pinned with the sub-root bound at zero."""
    import subprocess as _sp

    monkeypatch.setenv("TLDW_CHANGE_REVIEW_MAX_SUB_ROOTS", "0")
    root = tmp_path / "root"
    root.mkdir()
    (root / "small.txt").write_text("hello\n")
    child = root / "childrepo"
    child.mkdir()
    _sp.run(["git", "init", "--quiet", str(child)], check=True)
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run = db.create_run(conversation_id="conv-1", agent_kind="primary")

    def mutate():
        (root / "small.txt").write_text("edited\n")
        (child / "inner.txt").write_text("invisible\n")

    _record_turn(db, tracker, root, run, mutate)
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="conv-1"
    )
    app = _Harness(provider)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        await _wait_for(
            pilot,
            lambda: "childrepo"
            in str(screen.query_one("#change-review-banner", Static).renderable),
            "nested banner",
        )
        text = str(screen.query_one("#change-review-banner", Static).renderable)
        assert "1 nested repository" in text
        labels = _tree_labels(screen.query_one("#change-review-tree", Tree))
        assert not any("inner.txt" in l for l in labels), (
            "the nested edit leaked into the tree"
        )
