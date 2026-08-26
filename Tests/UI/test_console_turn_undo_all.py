"""TASK-22305: safe direct Undo All planning for Console turn cards."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_console_narrow_layout import _compositor_text
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.change_review_screen import (
    AgentRunsChangeReviewProvider,
    ChangeRevertConfirmModal,
    ReviewTurn,
)
from tldw_chatbook.Widgets.Console.console_turn_file_card import ConsoleTurnFileCard
from tldw_chatbook.Workspaces.change_revert import (
    RevertOutcome,
    RevertPreflight,
)
from tldw_chatbook.Workspaces.change_tracking import ChangedFile, ShadowRepoService
from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker


class _Provider:
    """Small provider double over the real ReviewTurn/ChangedFile shapes."""

    def __init__(self, rows: list[dict], files_by_row: dict[int, list[ChangedFile]]):
        self.turn = ReviewTurn("run-1", "turn", tuple(rows))
        self.files_by_row = files_by_row
        self.preflight_by_row: dict[int, list[str]] = {}
        self.disk_state_by_row: dict[int, dict[str, str]] = {}
        self.changed_calls: list[int] = []
        self.preflight_calls: list[tuple[int, tuple[str, ...]]] = []
        self.revert_calls: list[tuple[int, tuple[str, ...]]] = []
        self.outcomes_by_row: dict[int, list[RevertOutcome]] = {}
        self.revert_error_by_row: dict[int, Exception] = {}
        self.revert_error: Exception | None = None
        self.active = False
        self.active_roots: set[str] = set()

    def turn_for_run(self, run_id: str):
        return self.turn if run_id == self.turn.run_id else None

    def changed_files(self, row: dict):
        row_id = int(row["id"])
        self.changed_calls.append(row_id)
        return self.files_by_row[row_id]

    def preflight_revert(self, row: dict, paths: list[str]):
        row_id = int(row["id"])
        self.preflight_calls.append((row_id, tuple(paths)))
        return RevertPreflight(
            list(self.preflight_by_row.get(row_id, [])),
            dict(self.disk_state_by_row.get(row_id, {})),
        )

    def revert(self, row: dict, paths: list[str]):
        row_id = int(row["id"])
        self.revert_calls.append((row_id, tuple(paths)))
        if row_id in self.revert_error_by_row:
            raise self.revert_error_by_row[row_id]
        if self.revert_error is not None:
            raise self.revert_error
        if row_id in self.outcomes_by_row:
            return self.outcomes_by_row[row_id]
        return [RevertOutcome(path=path, ok=True) for path in paths]

    def run_active(self):
        return self.active

    def run_active_for_root(self, root: str):
        return self.active or str(root) in self.active_roots


def _row(row_id: int, root: Path, **overrides) -> dict:
    row = {
        "id": row_id,
        "root": str(root),
        "run_id": "run-1",
        "tracking_error": "",
        "baseline_sha": f"b{row_id}",
        "end_sha": f"e{row_id}",
    }
    row.update(overrides)
    return row


def _prepare(provider: _Provider):
    prepare = getattr(chat_screen_module, "_prepare_console_turn_undo", None)
    assert callable(prepare), "turn-level Undo All planner is missing"
    return prepare(provider, "run-1")


def _apply(provider: _Provider, plan):
    apply_plan = getattr(chat_screen_module, "_apply_console_turn_undo", None)
    assert callable(apply_plan), "turn-level Undo All executor is missing"
    return apply_plan(provider, plan)


def test_prepare_supports_one_window_per_root_and_labels_edited_paths(tmp_path):
    root_a = tmp_path / "alpha"
    root_b = tmp_path / "beta"
    rows = [_row(1, root_a), _row(2, root_b)]
    provider = _Provider(
        rows,
        {
            1: [ChangedFile(path="same.txt", status="M", adds=1, dels=1)],
            2: [ChangedFile(path="same.txt", status="A", adds=2, dels=0)],
        },
    )
    provider.preflight_by_row = {1: ["same.txt"], 2: ["same.txt"]}

    plan = _prepare(provider)

    assert plan.total_files == 2
    assert [(int(row["id"]), paths) for row, paths in plan.rows_paths] == [
        (1, ("same.txt",)),
        (2, ("same.txt",)),
    ]
    assert plan.edited_since == ("alpha/same.txt", "beta/same.txt")


def test_multi_root_labels_remain_distinct_when_root_leaf_names_match(tmp_path):
    root_a = tmp_path / "alpha" / "workspace"
    root_b = tmp_path / "beta" / "workspace"
    provider = _Provider(
        [_row(1, root_a), _row(2, root_b)],
        {
            1: [ChangedFile(path="same.txt", status="M", adds=1, dels=1)],
            2: [ChangedFile(path="same.txt", status="M", adds=1, dels=1)],
        },
    )
    provider.preflight_by_row = {1: ["same.txt"], 2: ["same.txt"]}

    plan = _prepare(provider)

    assert plan.edited_since == (
        f"{root_a}/same.txt",
        f"{root_b}/same.txt",
    )


def test_prepare_refuses_duplicate_root_windows_before_any_diff_or_preflight(tmp_path):
    root = tmp_path / "workspace"
    rows = [_row(1, root), _row(2, root)]
    provider = _Provider(
        rows,
        {
            1: [ChangedFile(path="first.txt", status="A", adds=1, dels=0)],
            2: [ChangedFile(path="second.txt", status="A", adds=1, dels=0)],
        },
    )
    ambiguous = getattr(chat_screen_module, "_ConsoleTurnUndoAmbiguousError", None)
    assert ambiguous is not None, "ambiguous-window refusal type is missing"

    with pytest.raises(ambiguous, match="multiple change windows"):
        _prepare(provider)

    assert provider.changed_calls == []
    assert provider.preflight_calls == []
    assert provider.revert_calls == []


def test_prepare_refuses_tracking_error_and_active_run_without_mutation(tmp_path):
    root = tmp_path / "workspace"
    unavailable = getattr(chat_screen_module, "_ConsoleTurnUndoUnavailableError", None)
    assert unavailable is not None, "safe Undo All refusal type is missing"

    errored = _Provider(
        [_row(1, root, tracking_error="snapshot failed")],
        {1: []},
    )
    with pytest.raises(unavailable, match="snapshot failed"):
        _prepare(errored)
    assert errored.changed_calls == []

    active = _Provider(
        [_row(1, root)],
        {1: [ChangedFile(path="a.txt", status="M", adds=1, dels=1)]},
    )
    active.active = True
    with pytest.raises(unavailable, match="finish or stop"):
        _prepare(active)
    assert active.changed_calls == []
    assert active.revert_calls == []


def test_prepare_refuses_a_background_run_targeting_any_turn_root(tmp_path):
    root_a = tmp_path / "alpha"
    root_b = tmp_path / "beta"
    provider = _Provider(
        [_row(1, root_a), _row(2, root_b)],
        {
            1: [ChangedFile(path="a.txt", status="M", adds=1, dels=1)],
            2: [ChangedFile(path="b.txt", status="M", adds=1, dels=1)],
        },
    )
    provider.active_roots.add(str(root_b))
    unavailable = chat_screen_module._ConsoleTurnUndoUnavailableError

    with pytest.raises(unavailable, match="finish or stop"):
        _prepare(provider)

    assert provider.changed_calls == []
    assert provider.preflight_calls == []
    assert provider.revert_calls == []


def test_apply_rechecks_preflight_and_refuses_new_edits_before_any_revert(tmp_path):
    root = tmp_path / "workspace"
    row = _row(1, root)
    provider = _Provider(
        [row],
        {1: [ChangedFile(path="a.txt", status="M", adds=1, dels=1)]},
    )
    plan = _prepare(provider)
    provider.preflight_by_row = {1: ["a.txt"]}
    stale = getattr(chat_screen_module, "_ConsoleTurnUndoStalePreflightError", None)
    assert stale is not None, "stale-confirmation refusal type is missing"

    with pytest.raises(stale, match="changed while the confirmation was open"):
        _apply(provider, plan)

    assert provider.revert_calls == []


def test_apply_refuses_when_an_already_warned_file_changes_again(tmp_path):
    root = tmp_path / "workspace"
    row = _row(1, root)
    provider = _Provider(
        [row],
        {1: [ChangedFile(path="a.txt", status="M", adds=1, dels=1)]},
    )
    provider.preflight_by_row = {1: ["a.txt"]}
    provider.disk_state_by_row = {1: {"a.txt": "sha256:first"}}
    plan = _prepare(provider)
    provider.disk_state_by_row = {1: {"a.txt": "sha256:second"}}
    stale = chat_screen_module._ConsoleTurnUndoStalePreflightError

    with pytest.raises(stale, match="changed while the confirmation was open"):
        _apply(provider, plan)

    assert provider.revert_calls == []


def test_apply_reports_all_per_path_outcomes_for_ordinary_multi_root(tmp_path):
    root_a = tmp_path / "alpha"
    root_b = tmp_path / "beta"
    provider = _Provider(
        [_row(1, root_a), _row(2, root_b)],
        {
            1: [ChangedFile(path="a.txt", status="M", adds=1, dels=1)],
            2: [ChangedFile(path="b.txt", status="A", adds=1, dels=0)],
        },
    )
    plan = _prepare(provider)

    outcomes = _apply(provider, plan)

    assert [outcome.path for outcome in outcomes] == ["alpha/a.txt", "beta/b.txt"]
    assert provider.revert_calls == [(1, ("a.txt",)), (2, ("b.txt",))]


def test_apply_preserves_success_and_marks_every_path_after_a_later_root_error(
    tmp_path,
):
    roots = [tmp_path / name for name in ("alpha", "beta", "gamma")]
    provider = _Provider(
        [_row(index, root) for index, root in enumerate(roots, start=1)],
        {
            1: [ChangedFile(path="a.txt", status="M", adds=1, dels=1)],
            2: [ChangedFile(path="b.txt", status="M", adds=1, dels=1)],
            3: [ChangedFile(path="c.txt", status="M", adds=1, dels=1)],
        },
    )
    provider.revert_error_by_row[2] = RuntimeError("second root failed")
    plan = _prepare(provider)

    outcomes = _apply(provider, plan)

    assert [(outcome.path, outcome.ok) for outcome in outcomes] == [
        ("alpha/a.txt", True),
        ("beta/b.txt", False),
        ("gamma/c.txt", False),
    ]
    assert all(
        "not processed" in str(outcome.error) and "second root failed" in outcome.error
        for outcome in outcomes[1:]
    )
    assert provider.revert_calls == [(1, ("a.txt",)), (2, ("b.txt",))]


async def _wait_for(pilot, predicate, description: str, attempts: int = 150):
    for _ in range(attempts):
        value = predicate()
        if value:
            return value
        await pilot.pause(0.02)
    raise AssertionError(f"timed out waiting for {description}")


def _record_real_turn(db, tracker, root: Path, run_id: str) -> None:
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    (root / "edited.txt").write_text("agent edit\n")
    for record in tracker.end_turn(handle):
        db.record_change_snapshot(
            run_id=run_id,
            root=record.root,
            baseline_sha=record.baseline_sha,
            end_sha=record.end_sha,
            files_changed=record.files_changed,
            adds=record.adds,
            dels=record.dels,
            tracking_error=record.tracking_error,
            untracked_oversize=record.untracked_oversize,
            nested_repos=record.nested_repos,
        )


@pytest.mark.asyncio
async def test_mounted_turn_card_confirms_cancels_then_undoes_real_files_off_thread(
    tmp_path,
    monkeypatch,
):
    """The direct card flow names risk, restores disk, and preserves history."""
    conversation_id = "conv-turn-undo"
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "edited.txt"
    target.write_text("before\n")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="turn-undo-test")
    run_id = db.create_run(conversation_id=conversation_id, agent_kind="primary")
    _record_real_turn(db, tracker, root, run_id)

    main_thread = threading.get_ident()
    preflight_threads: list[int] = []
    revert_threads: list[int] = []
    original_preflight = AgentRunsChangeReviewProvider.preflight_revert
    original_revert = AgentRunsChangeReviewProvider.revert

    def record_preflight(self, row, paths):
        preflight_threads.append(threading.get_ident())
        return original_preflight(self, row, paths)

    def record_revert(self, row, paths):
        revert_threads.append(threading.get_ident())
        return original_revert(self, row, paths)

    monkeypatch.setattr(
        AgentRunsChangeReviewProvider, "preflight_revert", record_preflight
    )
    monkeypatch.setattr(AgentRunsChangeReviewProvider, "revert", record_revert)

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        store = screen._ensure_console_chat_store()
        session = store.create_session(session_id=conversation_id)
        session.persisted_conversation_id = conversation_id
        screen._ensure_console_chat_controller()
        screen._console_agent_bridge = ConsoleAgentBridge(
            agent_runs_db=db,
            store=store,
            provider_gateway=MagicMock(),
            change_tracker=SimpleNamespace(service=service),
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="✎ Edited 1 file  +1 −1 — review with `v`",
            change_review_run_id=run_id,
        )
        await screen._sync_native_console_chat_ui()

        card = await _wait_for(
            pilot,
            lambda: (
                list(screen.query(ConsoleTurnFileCard))[0]
                if list(screen.query(ConsoleTurnFileCard))
                else None
            ),
            "turn file card",
        )
        undo = card.query_one(".console-turn-file-undo-all-btn", Button)
        await _wait_for(
            pilot,
            lambda: undo if not undo.disabled else None,
            "enabled Undo All",
        )

        card.scroll_visible(animate=False)
        await pilot.pause()
        wide_frame = host.export_screenshot(
            title="Console turn Undo All wide", simplify=True
        )
        wide_text = _compositor_text(wide_frame)
        assert all(
            text in wide_text for text in ("edited.txt", "Undo All", "Review")
        ), wide_text
        await pilot.resize_terminal(90, 30)
        card.scroll_visible(animate=False)
        await pilot.pause()
        narrow_frame = host.export_screenshot(
            title="Console turn Undo All narrow", simplify=True
        )
        narrow_text = _compositor_text(narrow_frame)
        assert all(text in narrow_text for text in ("Undo All", "Review")), narrow_text
        await pilot.resize_terminal(160, 48)
        card.scroll_visible(animate=False)
        await pilot.pause()

        target.write_text("user edit after turn\n")
        undo.press()
        modal = await _wait_for(
            pilot,
            lambda: (
                host.screen_stack[-1]
                if isinstance(host.screen_stack[-1], ChangeRevertConfirmModal)
                else None
            ),
            "Undo All confirmation",
        )
        warning = str(
            modal.query_one("#change-revert-edited-warning", Static).renderable
        )
        assert "edited.txt" in warning and "overwrites" in warning

        await pilot.click("#change-revert-no")
        await _wait_for(
            pilot,
            lambda: undo if not undo.disabled else None,
            "Undo All restored after cancellation",
        )
        assert target.read_text() == "user edit after turn\n"

        undo.press()
        await _wait_for(
            pilot,
            lambda: (
                host.screen_stack[-1]
                if isinstance(host.screen_stack[-1], ChangeRevertConfirmModal)
                else None
            ),
            "second Undo All confirmation",
        )
        await pilot.click("#change-revert-yes")
        await _wait_for(
            pilot,
            lambda: target.read_text() == "before\n" or None,
            "baseline restored",
        )
        await _wait_for(
            pilot,
            lambda: undo if str(undo.label) == "Undone" else None,
            "card marked Undone",
        )

        assert len(card.query(".console-turn-file-row")) == 1
        assert card.query_one(".console-turn-file-review-btn", Button).disabled is False
        assert preflight_threads and all(
            tid != main_thread for tid in preflight_threads
        )
        assert revert_threads and all(tid != main_thread for tid in revert_threads)


@pytest.mark.asyncio
async def test_mounted_duplicate_root_undo_refuses_inline_and_routes_to_review(
    tmp_path,
    monkeypatch,
):
    """Ambiguous windows never preflight or mutate from the compact card."""
    root = tmp_path / "workspace"
    root.mkdir()
    provider = _Provider(
        [_row(1, root), _row(2, root)],
        {
            1: [ChangedFile(path="first.txt", status="A", adds=1, dels=0)],
            2: [ChangedFile(path="second.txt", status="A", adds=1, dels=0)],
        },
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        store = screen._ensure_console_chat_store()
        session = store.create_session(session_id="conv-ambiguous-undo")
        session.persisted_conversation_id = session.id
        screen._ensure_console_chat_controller()
        monkeypatch.setattr(screen, "_console_change_review_provider", lambda: provider)
        review_calls: list[str] = []
        monkeypatch.setattr(
            screen,
            "_open_change_review",
            lambda run_id=None, **_kwargs: review_calls.append(run_id),
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="✎ Edited 2 files  +2 −0 — review with `v`",
            change_review_run_id="run-1",
        )
        await screen._sync_native_console_chat_ui()

        card = await _wait_for(
            pilot,
            lambda: (
                list(screen.query(ConsoleTurnFileCard))[0]
                if list(screen.query(ConsoleTurnFileCard))
                else None
            ),
            "ambiguous turn file card",
        )
        undo = card.query_one(".console-turn-file-undo-all-btn", Button)
        await _wait_for(
            pilot,
            lambda: undo if not undo.disabled else None,
            "ambiguous card Undo All",
        )
        changed_before = list(provider.changed_calls)

        undo.press()
        await _wait_for(pilot, lambda: review_calls or None, "Review routing")

        assert review_calls == ["run-1"]
        assert provider.changed_calls == changed_before
        assert provider.preflight_calls == []
        assert provider.revert_calls == []
        assert undo.disabled is False


@pytest.mark.asyncio
async def test_mounted_partial_and_provider_failures_name_problem_and_allow_retry(
    tmp_path,
    monkeypatch,
):
    """Incomplete or raised reverts never claim success or strand the card."""
    root = tmp_path / "workspace"
    root.mkdir()
    provider = _Provider(
        [_row(1, root)],
        {1: [ChangedFile(path="a.txt", status="M", adds=1, dels=1)]},
    )
    provider.outcomes_by_row = {
        1: [RevertOutcome(path="a.txt", ok=False, error="restore failed")]
    }

    app = _build_test_app()
    notices: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notices.append((message, kwargs.get("severity"))),
    )
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        store = screen._ensure_console_chat_store()
        session = store.create_session(session_id="conv-partial-undo")
        session.persisted_conversation_id = session.id
        screen._ensure_console_chat_controller()
        monkeypatch.setattr(screen, "_console_change_review_provider", lambda: provider)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="✎ Edited 1 file  +1 −1 — review with `v`",
            change_review_run_id="run-1",
        )
        await screen._sync_native_console_chat_ui()

        card = await _wait_for(
            pilot,
            lambda: (
                list(screen.query(ConsoleTurnFileCard))[0]
                if list(screen.query(ConsoleTurnFileCard))
                else None
            ),
            "partial-failure turn file card",
        )
        undo = card.query_one(".console-turn-file-undo-all-btn", Button)
        await _wait_for(
            pilot,
            lambda: undo if not undo.disabled else None,
            "partial-failure Undo All",
        )

        undo.press()
        await _wait_for(
            pilot,
            lambda: (
                host.screen_stack[-1]
                if isinstance(host.screen_stack[-1], ChangeRevertConfirmModal)
                else None
            ),
            "partial-failure confirmation",
        )
        await pilot.click("#change-revert-yes")
        await _wait_for(
            pilot,
            lambda: undo if not undo.disabled else None,
            "Undo All retry after partial failure",
        )
        assert any(
            "0 file(s) reverted; 1 file(s) not reverted" in message
            and "a.txt (restore failed)" in message
            and severity == "error"
            for message, severity in notices
        )
        assert str(undo.label) == "Undo All"

        provider.revert_error = RuntimeError("provider exploded")
        undo.press()
        await _wait_for(
            pilot,
            lambda: (
                host.screen_stack[-1]
                if isinstance(host.screen_stack[-1], ChangeRevertConfirmModal)
                else None
            ),
            "provider-failure confirmation",
        )
        await pilot.click("#change-revert-yes")
        await _wait_for(
            pilot,
            lambda: undo if not undo.disabled else None,
            "Undo All retry after provider failure",
        )
        assert any(
            "provider exploded" in message and severity == "error"
            for message, severity in notices
        )
        assert str(undo.label) == "Undo All"
