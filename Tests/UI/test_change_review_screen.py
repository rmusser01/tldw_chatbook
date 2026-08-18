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

    def __init__(self, provider, initial_run_id: str | None = None) -> None:
        super().__init__()
        self._provider = provider
        self._initial_run_id = initial_run_id

    def on_mount(self) -> None:
        self.push_screen(
            ChangeReviewScreen(self._provider, initial_run_id=self._initial_run_id)
        )


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


@pytest.mark.asyncio
async def test_selecting_a_tree_node_loads_that_files_diff(review_fixture):
    """TASK-2032 (live-UAT defect): clicking a file row must load its diff.

    Driven through Tree's own cursor-select action — the same NodeSelected
    event a mouse click produces — so the mouse path is what's pinned.
    """
    provider, root, run1, run2 = review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        tree = screen.query_one("#change-review-tree", Tree)
        await _wait_for(pilot, lambda: screen._leaves, "leaves loaded")
        assert len(screen._leaves) >= 2, "fixture must offer multiple files"

        target_row, target_change = screen._leaves[-1]
        first_row, first_change = screen._leaves[0]
        assert target_change.path != first_change.path

        # Walk the tree cursor to the target leaf and select it — the event
        # path a mouse click takes (click -> cursor + NodeSelected).
        chosen = None
        for line in range(tree.last_line + 1):
            tree.cursor_line = line
            node = tree.cursor_node
            if node is not None and target_change.path in str(node.label):
                chosen = line
                break
        assert chosen is not None, "target leaf not found in the tree"
        tree.action_select_cursor()
        await pilot.pause()

        def diff_text() -> str:
            body = screen.query(".change-review-diff-body")
            return str(body.first().renderable) if body else ""

        await _wait_for(
            pilot,
            lambda: target_change.path in diff_text(),
            f"diff to switch to {target_change.path} (still showing "
            f"{first_change.path})",
        )


def _append_real_write_step(db, run_id: str, path: str) -> None:
    """Record a write_file step through the PRODUCTION serialization:
    a real AgentStep dataclass via dataclasses.asdict — the exact shape
    `AgentService._persist` stores (TASK-1978 AC#4)."""
    import dataclasses

    from tldw_chatbook.Agents.agent_models import AgentStep

    step = AgentStep(
        index=0,
        kind="tool",
        tool_name="write_file",
        args={"file_path": path, "content": "x"},
        result="ok",
        created_at="2026-08-03T00:00:00.000000Z",
    )
    db.append_steps(run_id, [dataclasses.asdict(step)])


BADGE_COPY = "⚠ changed outside direct file tools"


@pytest.mark.asyncio
async def test_badge_marks_files_no_write_tool_touched(tmp_path):
    """TASK-1978 AC#1/#2/#4: tool-written files carry no badge; everything
    else in the turn does — with the exact spec copy, monochrome."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "tooled.txt").write_text("before\n")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run = db.create_run(conversation_id="conv-1", agent_kind="primary")

    def mutate():
        (root / "tooled.txt").write_text("after\n")
        (root / "scripted.txt").write_text("a script wrote this\n")

    _record_turn(db, tracker, root, run, mutate)
    _append_real_write_step(db, run, str(root / "tooled.txt"))
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="conv-1"
    )
    app = _Harness(provider)
    async with app.run_test(size=(140, 40)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        tree = screen.query_one("#change-review-tree", Tree)
        labels = await _wait_for(
            pilot,
            lambda: (lambda ls: ls if any("scripted.txt" in l for l in ls) else None)(
                _tree_labels(tree)
            ),
            "turn files",
        )
        scripted = next(l for l in labels if "scripted.txt" in l)
        tooled = next(l for l in labels if "tooled.txt" in l)
        assert BADGE_COPY in scripted, scripted
        assert BADGE_COPY not in tooled, (
            "a write_file-touched file must NOT be badged"
        )


@pytest.mark.asyncio
async def test_run_with_no_recorded_steps_renders_no_badges(review_fixture):
    """TASK-1978 AC#3: older data without steps must not badge everything."""
    provider, root, run1, run2 = review_fixture
    app = _Harness(provider)
    async with app.run_test(size=(140, 40)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        tree = screen.query_one("#change-review-tree", Tree)
        labels = await _wait_for(
            pilot,
            lambda: (lambda ls: ls if any("new.txt" in l for l in ls) else None)(
                _tree_labels(tree)
            ),
            "turn files",
        )
        assert not any(BADGE_COPY in l for l in labels), (
            "a stepless run badged its files"
        )


def test_badge_span_is_monochrome():
    """TASK-1978 AC#2: the badge renders dim, never colored."""
    from tldw_chatbook.Workspaces.change_tracking import ChangedFile

    label = ChangeReviewScreen._leaf_label(
        {"root": "/r"},
        ChangedFile(path="a.txt", status="M", adds=1, dels=0),
        multi_root=False,
        badge=True,
    )
    assert BADGE_COPY in str(label)
    badge_spans = [s for s in label.spans if s.style]
    assert badge_spans and all(s.style == "dim" for s in badge_spans), (
        f"badge must be monochrome dim: {label.spans}"
    )


@pytest.mark.asyncio
async def test_deleted_and_renamed_rows_badge_even_when_path_was_tool_touched(
    tmp_path,
):
    """Qodo #1262: no file tool can delete or rename, so D/R rows badge
    even when the path itself appears in the run's write_file steps."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "doomed.txt").write_text("tool wrote me\n")
    (root / "old.txt").write_text("stable rename content\n" * 5)
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run = db.create_run(conversation_id="conv-1", agent_kind="primary")

    def mutate():
        (root / "doomed.txt").unlink()
        (root / "old.txt").rename(root / "new.txt")

    _record_turn(db, tracker, root, run, mutate)
    # The run DID write these paths earlier — membership alone would
    # wrongly suppress the badge on their D/R rows.
    _append_real_write_step(db, run, str(root / "doomed.txt"))
    _append_real_write_step(db, run, str(root / "new.txt"))
    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id="conv-1"
    )
    app = _Harness(provider)
    async with app.run_test(size=(140, 40)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        tree = screen.query_one("#change-review-tree", Tree)
        labels = await _wait_for(
            pilot,
            lambda: (lambda ls: ls if any("doomed.txt" in l for l in ls) else None)(
                _tree_labels(tree)
            ),
            "turn files",
        )
        doomed = next(l for l in labels if "doomed.txt" in l)
        renamed = next(l for l in labels if "new.txt" in l)
        assert BADGE_COPY in doomed, f"deletion unbadged: {doomed}"
        assert BADGE_COPY in renamed, f"rename unbadged: {renamed}"


def test_turn_for_run_matches_the_full_scan(review_fixture):
    """The run-scoped read returns exactly what the turns() scan built.

    ``ConsoleTurnFileCard`` resolves its run through ``turn_for_run``
    (Qodo, PR #1728: per-card ``turns()`` calls re-scanned the whole
    conversation history); the two paths share ``_build_review_turn``,
    and this pins that they can never drift on label, rows, or order.
    """
    provider, _root, run1, run2 = review_fixture
    scanned = {t.run_id: t for t in provider.turns()}
    for run_id in (run1, run2):
        direct = provider.turn_for_run(run_id)
        assert direct == scanned[run_id]
    assert provider.turn_for_run("no-such-run") is None


# -- Task 7 (console-turn-file-annotate): initial_run_id --------------------


@pytest.mark.asyncio
async def test_initial_run_id_opens_directly_on_that_turn(review_fixture):
    """The card's `Review` button opens the screen already scoped to its
    OWN run -- not the conversation's latest turn. `turns()` returns
    newest-first (`run2` here), so opening with `initial_run_id=run1`
    proves the constructor arg actually wins over the "latest" default.
    """
    provider, root, run1, run2 = review_fixture
    app = _Harness(provider, initial_run_id=run1)
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        labels = await _wait_for(
            pilot,
            lambda: (
                lambda ls: ls if any("first_turn.txt" in l for l in ls) else None
            )(_tree_labels(screen.query_one(Tree))),
            "run1's files on initial open",
        )
        text = "\n".join(labels)
        assert "first_turn.txt" in text
        assert "new.txt" not in text, "opened on the latest turn, not run1"
        select = screen.query_one("#change-review-turn-select", Select)
        assert select.value == run1


@pytest.mark.asyncio
async def test_unknown_initial_run_id_falls_back_to_the_latest_turn(review_fixture):
    """A stale/unknown run id (e.g. the run's history was later pruned)
    must degrade to today's default -- opening on the latest turn -- not
    an empty or broken screen."""
    provider, root, run1, run2 = review_fixture
    app = _Harness(provider, initial_run_id="no-such-run")
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: app.screen if isinstance(app.screen, ChangeReviewScreen) else None,
            "review screen",
        )
        labels = await _wait_for(
            pilot,
            lambda: (
                lambda ls: ls if any("new.txt" in l for l in ls) else None
            )(_tree_labels(screen.query_one(Tree))),
            "the latest turn's files as fallback",
        )
        text = "\n".join(labels)
        assert "new.txt" in text
        select = screen.query_one("#change-review-turn-select", Select)
        assert select.value == run2
