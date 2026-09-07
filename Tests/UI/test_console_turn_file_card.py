"""Turn file card: header, async rows, expandable capped diffs.

Runs on the REAL app CSS stack (screen css + bundle): geometry measured
without the bundle is not measured (task-15110's lesson).
"""
import time
from threading import Event
from pathlib import Path

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button

from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.css import build_css
from tldw_chatbook.Widgets.Console.console_turn_file_card import (
    MAX_MOUNTED_HUNKS,
    ConsoleTurnFileCard,
)

_CSS_DIR = Path(build_css.__file__).parent
_SELF, _SCOPED = build_css.screen_css_paths(_CSS_DIR)

MARKER = "✎ Edited 2 files  +8 −3 — review with `v`"


class _FakeProvider:
    diff_display_max_lines = 2000

    def __init__(self):
        from tldw_chatbook.UI.Screens.change_review_screen import ReviewTurn

        self._row = {"root": "/ws", "kind": "turn", "tracking_error": None,
                     "files_changed": 2, "adds": 8, "dels": 3,
                     "baseline_sha": "b", "end_sha": "e", "run_id": "run-1"}
        self._turn = ReviewTurn(run_id="run-1", label="t", rows=(self._row,))

    def turns(self):
        return [self._turn]

    def changed_files(self, row):
        from tldw_chatbook.Workspaces.change_tracking import ChangedFile

        assert row is self._row
        return [ChangedFile(path="a.py", status="M", adds=5, dels=3),
                ChangedFile(path="b.md", status="A", adds=3, dels=0)]

    def diff_text(self, row, path):
        assert path in ("a.py", "b.md")
        return "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old line\n+new line\n"


def _multi_hunk_diff_text(n_hunks: int, body_lines_per_hunk: int) -> str:
    """Build a synthetic unified diff with ``n_hunks`` hunks, each carrying
    a unique marker in its header and body so segmentation can be asserted
    independent of any single hunk's content.
    """
    lines = ["--- a/big.py", "+++ b/big.py"]
    for hunk_idx in range(n_hunks):
        start = hunk_idx * 10 + 1
        lines.append(
            f"@@ -{start},{body_lines_per_hunk} +{start},{body_lines_per_hunk} "
            f"@@ hunk_{hunk_idx}_marker"
        )
        for line_idx in range(body_lines_per_hunk):
            lines.append(f"+hunk{hunk_idx}_body_line{line_idx}")
    return "\n".join(lines) + "\n"


class _MultiHunkProvider(_FakeProvider):
    """Fake provider whose ``diff_text`` returns a multi-hunk synthetic
    diff, with a configurable display cap and a call counter -- used to
    assert per-hunk segmentation/elision and that collapse/re-expand reuses
    the cache instead of re-fetching.
    """

    def __init__(
        self,
        *,
        n_hunks: int = 3,
        body_lines_per_hunk: int = 5,
        diff_display_max_lines: int = 2000,
    ):
        super().__init__()
        self.diff_display_max_lines = diff_display_max_lines
        self._n_hunks = n_hunks
        self._body_lines_per_hunk = body_lines_per_hunk
        self.diff_text_calls = 0

    def diff_text(self, row, path):
        self.diff_text_calls += 1
        return _multi_hunk_diff_text(self._n_hunks, self._body_lines_per_hunk)


class _NotesCapableMultiHunkProvider(_MultiHunkProvider):
    """A ``_MultiHunkProvider`` that also offers the notes trio, with one
    pre-existing note pinned to a hunk PAST the mount ceiling -- used to
    assert the elision tail cheaply flags when an elided hunk carries a
    note (TASK-17611, AC#1).
    """

    def __init__(self, *, elided_note_hunk_index: int, **kwargs):
        super().__init__(**kwargs)
        self._elided_note_hunk_index = elided_note_hunk_index

    def notes_for_run(self, run_id):
        idx = self._elided_note_hunk_index
        start = idx * 10 + 1
        header = (
            f"@@ -{start},{self._body_lines_per_hunk} "
            f"+{start},{self._body_lines_per_hunk} @@ hunk_{idx}_marker"
        )
        return [
            {
                "id": 1,
                "note": "watch this one",
                "delivered_at": None,
                "root": "/ws",
                "path": "a.py",
                "hunk_index": idx,
                "hunk_header": header,
                "snapshot_id": None,
            }
        ]

    def add_change_note(self, **kwargs):
        raise NotImplementedError("not exercised by this test")

    def delete_change_note(self, note_id):
        raise NotImplementedError("not exercised by this test")


class _Host(ConsolidatedCSSApp):
    CSS_PATH = [str(_SELF), str(_CSS_DIR / "tldw_cli_modular.tcss"), str(_SCOPED)]

    def compose(self) -> ComposeResult:
        yield ConsoleTurnFileCard(
            MARKER, "run-1", lambda: _FakeProvider(),
            id="card-under-test",
        )


class _UndoCaptureHost(ConsolidatedCSSApp):
    """Card host that records direct Undo All requests without handling them."""

    CSS_PATH = [str(_SELF), str(_CSS_DIR / "tldw_cli_modular.tcss"), str(_SCOPED)]

    def __init__(self, provider) -> None:
        super().__init__()
        self._provider = provider
        self.undo_requests: list[ConsoleTurnFileCard.UndoAllRequested] = []

    def compose(self) -> ComposeResult:
        yield ConsoleTurnFileCard(
            MARKER,
            "run-1",
            lambda: self._provider,
            id="card-under-test",
        )

    @on(ConsoleTurnFileCard.UndoAllRequested)
    def capture_undo(self, event: ConsoleTurnFileCard.UndoAllRequested) -> None:
        event.stop()
        self.undo_requests.append(event)


async def _settled_card(pilot):
    card = pilot.app.query_one("#card-under-test", ConsoleTurnFileCard)
    for _ in range(60):
        if card.query(".console-turn-file-row"):
            break
        await pilot.pause(0.02)
    return card


async def _undo_ready_card(pilot):
    """Wait until row loading has made the destructive action available."""
    card = await _settled_card(pilot)
    undo = card.query_one(".console-turn-file-undo-all-btn", Button)
    for _ in range(60):
        if not undo.disabled:
            break
        await pilot.pause(0.02)
    assert undo.disabled is False
    return card


@pytest.mark.asyncio
async def test_undo_all_is_visible_disabled_while_loading_then_enabled():
    """The destructive action cannot fire before exact snapshot rows resolve."""

    release = Event()

    class _BlockingProvider(_FakeProvider):
        def turn_for_run(self, run_id):
            assert run_id == "run-1"
            release.wait(timeout=3)
            return self._turn

    provider = _BlockingProvider()
    app = _UndoCaptureHost(provider)
    async with app.run_test(size=(120, 40)) as pilot:
        card = pilot.app.query_one("#card-under-test", ConsoleTurnFileCard)
        undo = card.query_one(".console-turn-file-undo-all-btn", Button)
        review = card.query_one(".console-turn-file-review-btn", Button)
        assert str(undo.label) == "Undo All"
        assert undo.disabled is True
        assert review.disabled is False

        release.set()
        card = await _undo_ready_card(pilot)
        for _ in range(60):
            if not undo.disabled:
                break
            await pilot.pause(0.02)
        assert undo.disabled is False


@pytest.mark.asyncio
async def test_undo_all_press_posts_once_and_enters_busy_state():
    """A double press cannot dispatch two destructive operations."""

    app = _UndoCaptureHost(_FakeProvider())
    async with app.run_test(size=(120, 40)) as pilot:
        card = await _undo_ready_card(pilot)
        undo = card.query_one(".console-turn-file-undo-all-btn", Button)
        undo.focus()
        await pilot.press("enter")
        await pilot.press("enter")
        await pilot.pause()

        assert len(app.undo_requests) == 1
        assert app.undo_requests[0].run_id == "run-1"
        assert app.undo_requests[0].card is card
        assert undo.disabled is True
        assert str(undo.label) == "Undoing…"


@pytest.mark.asyncio
async def test_undo_completion_keeps_history_and_review_reachable():
    """Undo marks state; it never erases the historical changed-file card."""

    app = _UndoCaptureHost(_FakeProvider())
    async with app.run_test(size=(120, 40)) as pilot:
        card = await _undo_ready_card(pilot)
        undo = card.query_one(".console-turn-file-undo-all-btn", Button)
        await card.begin_undo_all()
        card.finish_undo_all(success=True)
        await pilot.pause()

        assert str(undo.label) == "Undone"
        assert undo.disabled is True
        assert len(card.query(".console-turn-file-row")) == 2
        assert card.query_one(".console-turn-file-review-btn", Button).disabled is False


@pytest.mark.asyncio
async def test_header_shows_marker_and_rows_load_async():
    async with _Host().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        header = card.query_one(".console-turn-file-header")
        assert MARKER.split(" — ")[0] in str(header.render())
        rows = list(card.query(".console-turn-file-row"))
        assert len(rows) == 2
        assert "a.py" in str(rows[0].render())
        assert "+5" in str(rows[0].render()) and "−3" in str(rows[0].render())


@pytest.mark.asyncio
async def test_expand_shows_capped_scrolling_diff():
    async with _Host().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        row = card.query(".console-turn-file-row").first()
        row.focus()
        await pilot.press("enter")
        body = None
        for _ in range(60):
            bodies = card.query(".console-turn-file-diff")
            if bodies and bodies.first().display:
                body = bodies.first()
                break
            await pilot.pause(0.02)
        assert body is not None, "diff body never displayed"
        # `body` is the VerticalScroll container -- its own render() is a
        # Blank placeholder (containers paint children, not self-content).
        # The diff text lives on the mounted per-hunk Static child(ren).
        diff_text_widget = body.query_one(".console-turn-file-hunk")
        assert "+new line" in str(diff_text_widget.render())
        assert str(body.styles.overflow_y) == "auto"
        assert body.styles.max_height is not None
        # collapse again: display-managed, never unmounted
        row.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not body.display and body.is_mounted


async def _expand_first_row(pilot, card):
    """Press the first row open and return its diff body once displayed."""
    row = card.query(".console-turn-file-row").first()
    row.focus()
    await pilot.press("enter")
    for _ in range(60):
        bodies = card.query(".console-turn-file-diff")
        if bodies and bodies.first().display:
            return bodies.first()
        await pilot.pause(0.02)
    raise AssertionError("diff body never displayed")


@pytest.mark.asyncio
async def test_expand_multi_hunk_diff_mounts_one_block_per_hunk():
    """Expanding a row whose diff has 3 hunks mounts exactly 3
    ``.console-turn-file-hunk`` statics and 3 ``.console-turn-file-hunk-
    actions`` rows inside that row's diff body -- one pair per hunk.
    """
    provider = _MultiHunkProvider(n_hunks=3, body_lines_per_hunk=5)

    class _MultiHunkHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: provider, id="card-under-test"
            )

    async with _MultiHunkHost().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)
        hunks = list(body.query(".console-turn-file-hunk"))
        actions = list(body.query(".console-turn-file-hunk-actions"))
        assert len(hunks) == 3
        assert len(actions) == 3
        for hunk_idx in range(3):
            assert f"hunk_{hunk_idx}_marker" in str(hunks[hunk_idx].render())


@pytest.mark.asyncio
async def test_expand_hunk_past_old_cap_still_present():
    """A diff longer than ``diff_display_max_lines`` still yields ONE BLOCK
    PER HUNK, with per-hunk elision -- hunks past where the OLD flat-Static
    global cap would have cut off the whole diff are still present (and,
    later, annotatable).

    Pre-fix (flat ``.console-turn-file-diff-text`` Static) this is RED: the
    old code capped the WHOLE joined diff text at ``diff_display_max_lines``
    lines, so with a small cap the THIRD hunk's header never appeared in the
    rendered output at all -- it fell past the global cutoff before its
    line was ever reached. The per-hunk display cap (``max(1,
    diff_display_max_lines // len(hunks))``) guarantees every hunk gets its
    own block, each with its own honest "... N more lines" elision,
    regardless of how many hunks precede it.
    """
    # 3 hunks x 5 body lines + prelude(2) + 3 headers = 20 lines total.
    # diff_display_max_lines=4: the OLD global cap only ever showed the
    # prelude + hunk 0's header + one body line -- hunks 1 and 2 (including
    # their headers) never rendered at all.
    provider = _MultiHunkProvider(
        n_hunks=3, body_lines_per_hunk=5, diff_display_max_lines=4
    )

    class _CappedHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: provider, id="card-under-test"
            )

    async with _CappedHost().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)
        hunks = list(body.query(".console-turn-file-hunk"))
        assert len(hunks) == 3, (
            "every hunk must get its own block, even past the old global cap"
        )
        combined = "\n".join(str(hunk.render()) for hunk in hunks)
        assert "hunk_2_marker" in combined, (
            "third hunk's header must survive segmentation past the old "
            "global line cap"
        )
        # Per-hunk elision: cap=4 // 3 == 1 body line kept per hunk, so
        # each hunk's block carries an honest "more lines" tail rather than
        # silently vanishing.
        assert "more lines" in str(hunks[0].render())


@pytest.mark.asyncio
async def test_expand_past_hunk_ceiling_mounts_capped_blocks_and_honest_tail():
    """TASK-17611 (AC#1): a file with more hunks than ``MAX_MOUNTED_HUNKS``
    stops mounting one block per hunk past the cap and shows an honest
    "... N more hunks" tail instead of an unbounded number of widget
    groups.
    """
    n_hunks = MAX_MOUNTED_HUNKS + 10
    provider = _MultiHunkProvider(n_hunks=n_hunks, body_lines_per_hunk=1)

    class _CeilingHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: provider, id="card-under-test"
            )

    async with _CeilingHost().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)
        hunks = list(body.query(".console-turn-file-hunk"))
        assert len(hunks) == MAX_MOUNTED_HUNKS, (
            "must stop mounting per-hunk blocks at the ceiling, not mount "
            "one per hunk unbounded"
        )
        tails = list(body.query(".console-turn-file-hunk-tail"))
        assert len(tails) == 1, "exactly one honest elision tail, not zero/many"
        tail_text = str(tails[0].render())
        assert "10 more hunks" in tail_text
        assert "Review" in tail_text


@pytest.mark.asyncio
async def test_hunk_ceiling_tail_flags_when_an_elided_hunk_carries_a_note():
    """TASK-17611 (AC#1): the ceiling's "N more hunks" tail must not hide
    the note affordance dishonestly -- when at least one elided hunk
    carries an existing note, the tail says so.
    """
    n_hunks = MAX_MOUNTED_HUNKS + 10
    provider = _NotesCapableMultiHunkProvider(
        elided_note_hunk_index=MAX_MOUNTED_HUNKS,  # first hunk past the cap
        n_hunks=n_hunks,
        body_lines_per_hunk=1,
    )

    class _CeilingNotesHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: provider, id="card-under-test"
            )

    async with _CeilingNotesHost().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)
        tail = body.query_one(".console-turn-file-hunk-tail")
        tail_text = str(tail.render())
        assert "10 more hunks" in tail_text
        assert "1 carrying note" in tail_text


@pytest.mark.asyncio
async def test_hunk_ceiling_budgets_per_hunk_lines_by_mounted_count_not_total():
    """Qodo round (TASK-17611): ``per_hunk_cap`` must divide
    ``diff_display_max_lines`` by how many hunks are actually MOUNTED
    (bounded by ``MAX_MOUNTED_HUNKS``), not the file's total hunk count --
    a file with many more hunks than the ceiling must not starve the 50
    hunks that DO get mounted of their fair per-hunk line budget.
    """
    n_hunks = 200
    body_lines_per_hunk = 20
    diff_display_max_lines = 200
    provider = _MultiHunkProvider(
        n_hunks=n_hunks,
        body_lines_per_hunk=body_lines_per_hunk,
        diff_display_max_lines=diff_display_max_lines,
    )

    class _BudgetHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: provider, id="card-under-test"
            )

    async with _BudgetHost().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)
        hunks = list(body.query(".console-turn-file-hunk"))
        assert len(hunks) == MAX_MOUNTED_HUNKS

        # cap // MOUNTED (200 // 50 == 4 lines/hunk) is the correct
        # budget -- the pre-fix bug divided by the file's TOTAL hunk count
        # instead (200 // 200 == 1 line/hunk), starving every mounted
        # block down to its first body line alone. Hunk index 1 (not 0,
        # to sidestep the first hunk's extra prelude text) must show its
        # 4th body line (index 3) and an honest "16 more lines" tail --
        # neither is reachable under the old total-count division, which
        # would cap it to 1 line and a "19 more lines" tail instead.
        second_hunk_text = str(hunks[1].render())
        assert "hunk1_body_line3" in second_hunk_text, (
            "per-hunk line budget must be computed from the MOUNTED hunk "
            "count, not the file's total hunk count"
        )
        assert "hunk1_body_line4" not in second_hunk_text
        assert "16 more lines" in second_hunk_text


@pytest.mark.asyncio
async def test_expand_collapse_reexpand_reuses_cache_single_diff_text_call():
    """Collapsing and re-expanding a row reuses the cached hunks -- the
    provider's ``diff_text`` is called exactly once, not once per expand.
    """
    provider = _MultiHunkProvider(n_hunks=1, body_lines_per_hunk=2)

    class _CountingHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: provider, id="card-under-test"
            )

    async with _CountingHost().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        await _expand_first_row(pilot, card)  # expand (cache miss)
        row = card.query(".console-turn-file-row").first()
        row.focus()
        await pilot.press("enter")  # collapse
        await pilot.pause()
        await pilot.press("enter")  # re-expand (cache hit)
        await pilot.pause(0.3)
        assert provider.diff_text_calls == 1


@pytest.mark.asyncio
async def test_expand_provider_construction_failure_never_crashes_app():
    """Pins the Critical fix: the factory succeeds on its FIRST call (used
    by `_load_rows`, which was already guarded) but raises on its SECOND
    call (used by `on_button_pressed` on first expand, which was NOT). An
    exception escaping a Textual `on_*` handler propagates to
    `app._handle_exception()`, which unconditionally exits the app -- so
    this must degrade the row instead, leaving the app fully responsive.
    """
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeProvider()
        raise RuntimeError("shadow repo transiently unavailable")

    class _FlakyHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", factory, id="card-under-test"
            )

    async with _FlakyHost().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        rows = list(card.query(".console-turn-file-row"))
        assert len(rows) == 2

        rows[0].focus()
        await pilot.press("enter")
        await pilot.pause(0.3)

        bodies = list(card.query(".console-turn-file-diff"))
        assert bodies[0].display is False, (
            "diff body must stay hidden when provider construction fails"
        )
        assert card.is_mounted
        assert pilot.app.is_running, (
            "a provider-construction failure on expand must not crash the app"
        )

        # The app must still be responsive after the failure -- a press on
        # the OTHER row (a fresh factory call) is handled the same way,
        # not just tolerated once by accident.
        rows[1].focus()
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert bodies[1].display is False
        assert card.is_mounted
        assert pilot.app.is_running


@pytest.mark.asyncio
async def test_real_provider_two_windows_on_same_root_no_duplicates_own_diffs(tmp_path):
    """PR3a-1 Task 6c regression: a run's ``change_snapshots`` can hold
    rows from TWO windows -- the turn's own window and its surviving
    sub-agents' post-turn window (``console_agent_bridge.py``'s
    ``_close_post_turn_change_window``) -- and BOTH windows can cover the
    SAME root, with BOTH markers carrying the SAME ``change_review_run_id``.

    Driven over the REAL stack (real ``ChangeTurnTracker``/shadow repo, a
    real ``AgentRunsDB``, the real ``AgentRunsChangeReviewProvider``) --
    copying the fixture pattern from ``test_change_review_screen.py``'s
    ``review_fixture``/``_record_turn``; that module's docstring explains
    why fake provider shapes are banned here. Uses a ``tmp_path`` FILE
    (not ``:memory:``) deliberately: ``ConsoleTurnFileCard`` reads the
    provider off ``asyncio.to_thread`` (a different OS thread than the one
    that wrote the rows), and ``AgentRunsDB`` holds one connection PER
    THREAD (``_held_connection``) -- a ``:memory:`` database is private to
    the connection that opened it, so the worker thread would see a blank
    schema (measured: ``no such table: change_snapshots``). A real file is
    the only path-independent way to exercise the card's actual off-thread
    read.

    Pre-fix this fails: ``ConsoleTurnFileCard._load_rows`` built a
    root-keyed ``changed_by_root`` dict, so the second-recorded (post-turn)
    row's files silently overwrote the first-recorded (turn) row's files at
    that root's dict slot. The result was 2 rows total (matching the file
    COUNT by coincidence) but both were ``survivor_write.txt`` -- the turn
    window's ``turn_write.txt`` never appeared at all, and expanding either
    row served the post-turn window's diff.
    """
    from tldw_chatbook.Chat.console_agent_bridge import (
        CHANGE_KIND_SUBAGENT_POST_TURN,
        CHANGE_KIND_TURN,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.UI.Screens.change_review_screen import (
        AgentRunsChangeReviewProvider,
    )
    from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
    from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker

    root = tmp_path / "root"
    root.mkdir()
    (root / "seed.txt").write_text("seed\n")

    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    conv = "conv-1"
    run_id = db.create_run(conversation_id=conv, agent_kind="primary")

    def _record_window(kind: str, mutate) -> None:
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
                kind=kind,
            )

    # Window 1: the turn's own window -- recorded FIRST, matching
    # production's insertion order (the turn ends before any post-turn
    # window is opened).
    _record_window(
        CHANGE_KIND_TURN,
        lambda: (root / "turn_write.txt").write_text("ALPHA_TURN_MARKER\n"),
    )
    # Window 2: the post-turn window covering a surviving sub-agent's
    # writes -- same root, same run_id, recorded SECOND.
    _record_window(
        CHANGE_KIND_SUBAGENT_POST_TURN,
        lambda: (root / "survivor_write.txt").write_text("BRAVO_POST_MARKER\n"),
    )

    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id=conv
    )

    class _RealHost(App):
        CSS_PATH = [str(_SELF), str(_CSS_DIR / "tldw_cli_modular.tcss"), str(_SCOPED)]

        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                "✎ Edited 2 files  +2 −0 — review with `v`",
                run_id,
                lambda: provider,
                id="card-under-test",
            )

    async with _RealHost().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        rows: list = []
        for _ in range(120):
            rows = list(card.query(".console-turn-file-row"))
            if len(rows) >= 2:
                break
            await pilot.pause(0.02)
        assert len(rows) == 2, "row count must equal total files across BOTH windows"
        labels = [str(row.render()) for row in rows]
        assert "turn_write.txt" in labels[0], labels
        assert "survivor_write.txt" in labels[1], labels
        # No duplicates: exactly one row names each file (the root-keyed
        # bug rendered survivor_write.txt twice and turn_write.txt zero times).
        assert sum("turn_write.txt" in label for label in labels) == 1
        assert sum("survivor_write.txt" in label for label in labels) == 1

        async def _expand(index: int):
            rows[index].focus()
            await pilot.press("enter")
            body = None
            for _ in range(60):
                bodies = list(card.query(".console-turn-file-diff"))
                if bodies[index].display:
                    body = bodies[index]
                    break
                await pilot.pause(0.02)
            assert body is not None, f"diff body {index} never displayed"
            return str(body.query_one(".console-turn-file-hunk").render())

        turn_diff = await _expand(0)
        assert "ALPHA_TURN_MARKER" in turn_diff
        assert "BRAVO_POST_MARKER" not in turn_diff

        post_turn_diff = await _expand(1)
        assert "BRAVO_POST_MARKER" in post_turn_diff
        assert "ALPHA_TURN_MARKER" not in post_turn_diff


@pytest.mark.asyncio
async def test_provider_failure_degrades_to_marker_only():
    class _Broken(_FakeProvider):
        def turns(self):
            raise RuntimeError("shadow repo unavailable")

    class _BrokenHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: _Broken(), id="card-under-test"
            )

    async with _BrokenHost().run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        card = pilot.app.query_one("#card-under-test", ConsoleTurnFileCard)
        assert not list(card.query(".console-turn-file-row"))
        assert MARKER.split(" — ")[0] in str(
            card.query_one(".console-turn-file-header").render()
        )


@pytest.mark.asyncio
async def test_selected_card_uses_the_bundles_focus_background():
    """Selected-card styling must resolve the BUNDLE's $ds-focus-bg.

    Regression guard for the TASK-16811 footgun (re-raised by Qodo on
    PR #1728): a widget-local `$ds-focus-bg:` "fallback" in DEFAULT_CSS
    shadows the app bundle's token for every rule in that CSS source, so
    the selected card rendered $surface while every other selected
    transcript row rendered the focus colour. A class-toggle assertion
    cannot catch that -- only the resolved colour can.
    """
    from textual.widgets import Static

    class _ParityHost(_Host):
        def compose(self) -> ComposeResult:
            yield Static(
                "peer",
                classes="console-transcript-message-selected",
                id="selected-peer",
            )
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: _FakeProvider(),
                selected=True,
                id="card-under-test",
            )
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: _FakeProvider(),
                id="card-unselected",
            )

    async with _ParityHost().run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        peer_bg = pilot.app.query_one("#selected-peer").styles.background
        card_bg = pilot.app.query_one("#card-under-test").styles.background
        plain_bg = pilot.app.query_one("#card-unselected").styles.background
        assert card_bg == peer_bg
        assert card_bg != plain_bg


# -- Task 7: Review button, expand/collapse-all, elided paths, AC#4 guard --


class _ReviewCaptureHost(_Host):
    """`_Host`, plus a capture of every posted `ReviewRequested`."""

    def __init__(self) -> None:
        super().__init__()
        self.review_requests: list[str] = []

    @on(ConsoleTurnFileCard.ReviewRequested)
    def _capture_review_requested(
        self, event: ConsoleTurnFileCard.ReviewRequested
    ) -> None:
        self.review_requests.append(event.run_id)


@pytest.mark.asyncio
async def test_review_button_posts_review_requested_with_the_cards_run_id():
    async with _ReviewCaptureHost().run_test(size=(120, 40)) as pilot:
        await _settled_card(pilot)
        review_btn = pilot.app.query_one(".console-turn-file-review-btn", Button)
        review_btn.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.review_requests == ["run-1"]


async def _toggle_all_and_wait_expanded(pilot, card):
    """Press the header toggle-all button and wait for every body to show."""
    toggle_btn = card.query_one(".console-turn-file-toggle-all-btn", Button)
    toggle_btn.focus()
    await pilot.press("enter")
    for _ in range(120):
        bodies = list(card.query(".console-turn-file-diff"))
        if bodies and all(body.display for body in bodies):
            return bodies
        await pilot.pause(0.02)
    raise AssertionError("expand-all never showed every row's body")


@pytest.mark.asyncio
async def test_toggle_all_expands_every_row_then_collapses_them_all():
    async with _Host().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        bodies = await _toggle_all_and_wait_expanded(pilot, card)
        assert len(bodies) == 2
        for body in bodies:
            assert list(body.query(".console-turn-file-hunk")), (
                "expand-all must mount every row's hunk blocks, not just "
                "flip display"
            )
        toggle_btn = card.query_one(".console-turn-file-toggle-all-btn", Button)
        toggle_btn.focus()
        await pilot.press("enter")
        await pilot.pause(0.2)
        bodies = list(card.query(".console-turn-file-diff"))
        assert not any(body.display for body in bodies), (
            "collapse-all must hide every row's body"
        )
        # display-managed, never unmounted (parity with the single-row
        # collapse/re-expand cache contract).
        assert all(body.is_mounted for body in bodies)


@pytest.mark.asyncio
async def test_toggle_all_loads_uncached_diffs_serialized_never_concurrently():
    """Hard constraint (spec §5): expand-all must load uncached diffs
    SERIALIZED in one worker, never N concurrent git subprocesses. A
    provider that tracks its own concurrent-call high-water mark, with a
    real sleep inside the off-thread read to widen the race window, would
    catch a switch to `asyncio.gather`-style concurrent loading.
    """

    class _ConcurrencyTrackingProvider(_FakeProvider):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.max_active = 0

        def diff_text(self, row, path):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.05)
                return "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"
            finally:
                self.active -= 1

    provider = _ConcurrencyTrackingProvider()

    class _ConcurrencyHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: provider, id="card-under-test"
            )

    async with _ConcurrencyHost().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        await _toggle_all_and_wait_expanded(pilot, card)
        assert provider.max_active == 1, (
            "expand-all loaded more than one diff concurrently"
        )


@pytest.mark.asyncio
async def test_toggle_all_label_reflects_manually_expanded_rows_not_just_toggle_state():
    """TASK-17611 (AC#2): the header's expand/collapse-all label must
    derive from the LIVE DOM state whenever it changes, not only when the
    toggle button itself is pressed -- expanding every row one at a time
    (via each row's own button) must flip the header to "collapse" too,
    and collapsing one of them back must flip it back to "expand".
    """
    # Raw chevron literals (not imported from the module under test): the
    # card is running in the DEFAULT (non-ASCII) glyph mode here, where
    # `resolve_glyph` is identity, so the rendered label carries these
    # unicode characters verbatim.
    _CHEVRON_CLOSED = "▸"
    _CHEVRON_OPEN = "▾"

    async with _Host().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        toggle_btn = card.query_one(".console-turn-file-toggle-all-btn", Button)

        # Baseline: nothing expanded yet -- collapsed/"expand all" state.
        assert str(toggle_btn.render()).startswith(_CHEVRON_CLOSED)
        assert toggle_btn.tooltip == "Expand every file's diff"

        rows = list(card.query(".console-turn-file-row"))
        assert len(rows) == 2

        # Expand the FIRST row individually (never touching the header
        # toggle) -- one of two rows expanded is still a "mixed" state,
        # so the header must still read "expand all".
        rows[0].focus()
        await pilot.press("enter")
        for _ in range(60):
            bodies = list(card.query(".console-turn-file-diff"))
            if bodies and bodies[0].display:
                break
            await pilot.pause(0.02)
        assert str(toggle_btn.render()).startswith(_CHEVRON_CLOSED), (
            "one of two rows expanded is still a mixed state -- header "
            "must not claim everything is expanded"
        )

        # Expand the SECOND row individually too -- now every row is
        # expanded, and the header must reflect that live DOM state even
        # though the toggle button itself was never pressed.
        rows[1].focus()
        await pilot.press("enter")
        for _ in range(60):
            bodies = list(card.query(".console-turn-file-diff"))
            if bodies and all(body.display for body in bodies):
                break
            await pilot.pause(0.02)
        assert str(toggle_btn.render()).startswith(_CHEVRON_OPEN), (
            "every row expanded individually must flip the header to "
            "'collapse all', not leave it stuck on 'expand all'"
        )
        assert toggle_btn.tooltip == "Collapse every file's diff"

        # Collapse the first row back individually -- back to mixed, so
        # the header must flip back to "expand all" too.
        rows[0].focus()
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert str(toggle_btn.render()).startswith(_CHEVRON_CLOSED), (
            "collapsing one row individually must un-stick the header "
            "from a stale 'collapse all' state"
        )
        assert toggle_btn.tooltip == "Expand every file's diff"


@pytest.mark.asyncio
async def test_row_tooltip_carries_the_full_unelided_path():
    async with _Host().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        rows = list(card.query(".console-turn-file-row"))
        assert rows[0].tooltip == "a.py"
        assert rows[1].tooltip == "b.md"


@pytest.mark.asyncio
async def test_narrow_card_elides_a_long_path_but_tooltip_keeps_it_whole():
    long_path = (
        "very/deeply/nested/directory/structure/that/is/quite/long/module.py"
    )

    class _LongPathProvider(_FakeProvider):
        def changed_files(self, row):
            from tldw_chatbook.Workspaces.change_tracking import ChangedFile

            return [ChangedFile(path=long_path, status="M", adds=1, dels=1)]

    provider = _LongPathProvider()

    class _NarrowHost(App):
        CSS_PATH = [str(_SELF), str(_CSS_DIR / "tldw_cli_modular.tcss"), str(_SCOPED)]

        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: provider, id="card-under-test"
            )

    async with _NarrowHost().run_test(size=(40, 20)) as pilot:
        card = await _settled_card(pilot)
        row = card.query_one(".console-turn-file-row", Button)
        rendered = str(row.render())
        assert long_path not in rendered, (
            "a long path in a narrow card must be elided, not shown whole"
        )
        assert "…" in rendered
        assert row.tooltip == long_path, (
            "the FULL path must still be reachable via the row's tooltip"
        )
