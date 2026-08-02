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
