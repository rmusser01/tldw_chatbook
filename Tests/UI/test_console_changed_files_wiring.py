"""TASK-18060 Task 5 (review-rail spec §2): screen wiring for the rail's
"Changed files" section -- cache, guard, off-thread worker, mount,
click-through, invalidation, config gate.

Mounted Console harness -- the same host `test_console_turn_file_card_
factory.py` uses (``ConsoleHarness``/``_build_test_app``/
``_wait_for_selector`` from the destination-shells + gate-1 test modules).
A file-backed ``AgentRunsDB`` + real ``ShadowRepoService``/
``ChangeTurnTracker`` back a REAL ``AgentRunsChangeReviewProvider`` --
the fixture-invented-shapes trap has bitten this repo five separate
times, so no fake provider shapes are hand-rolled here. The screen's own
``ConsoleAgentBridge`` is constructed for REAL too (the mounted screen's
0.2s-tick-adjacent sync path reaches several of its OTHER methods --
``subagent_counts``, ``live_snapshot`` -- so a thin hand-rolled double
broke on those unrelated seams; only ``change_tracker`` is a minimal
``SimpleNamespace(service=...)`` double, since ``change_review_provider``
reads only its ``.service`` attribute).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.change_review_screen import (
    AgentRunsChangeReviewProvider,
    ChangeReviewScreen,
)
from tldw_chatbook.Widgets.Console.console_changed_files_section import (
    ConsoleChangedFilesSection,
)
from tldw_chatbook.Widgets.Console.console_turn_file_card import ConsoleTurnFileCard
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker
import tldw_chatbook.Widgets.Console.console_transcript as console_transcript_module

MARKER = "✎ Edited 1 file  +1 −1 — review with `v`"
CONV_ID = "conv-wiring-1"


def _disable_turn_file_cards(monkeypatch) -> None:
    """Keep a change-summary marker plain (no mounted `ConsoleTurnFileCard`).

    Fix round: the card's own `_load_rows` worker calls the SAME
    `AgentRunsChangeReviewProvider.changed_files` a per-row-cache spy
    patches at the class level -- an unrelated mounted card would
    contaminate the git-call count under test (empirically caught: a
    stray card's own row-load added exactly one uncounted-for call).
    Matches `test_console_turn_file_card_factory.py`'s own technique for
    the config-OFF case.
    """
    monkeypatch.setattr(
        console_transcript_module,
        "get_cli_setting",
        lambda section, key, default=None: (
            False
            if (section, key) == ("console", "turn_file_cards")
            else default
        ),
    )


def _build_real_bridge(store, db: AgentRunsDB, service: ShadowRepoService):
    """A REAL ``ConsoleAgentBridge`` wired to this test's real DB/service.

    ``change_review_provider(conversation_id)`` -- the seam under test --
    is the bridge's own unmodified implementation; it reads only
    ``self._db`` and ``self._change_tracker.service``, both real here.
    """
    return ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=MagicMock(),
        change_tracker=SimpleNamespace(service=service),
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


class _Workspace:
    def __init__(self, root, service, tracker, db) -> None:
        self.root = root
        self.service = service
        self.tracker = tracker
        self.db = db


@pytest.fixture()
def workspace_fixture(tmp_path) -> _Workspace:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.py").write_text("line1\nline2\n")
    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    return _Workspace(root, service, tracker, db)


async def _mount_console_session(pilot, console, store, ws: _Workspace, *, conv_id=CONV_ID):
    """Wait for the composer, create + activate a persisted-looking session,
    wire a REAL provider bridge, return the session.
    """
    await _wait_for_selector(console, pilot, "#console-native-composer")
    session = store.create_session(session_id=conv_id)
    session.persisted_conversation_id = conv_id
    console._ensure_console_chat_controller()
    console._console_agent_bridge = _build_real_bridge(store, ws.db, ws.service)
    return session


async def _wait_for_changed_files_state(console, pilot, predicate, *, attempts=150):
    section = console.query_one(
        "#console-changed-files-section", ConsoleChangedFilesSection
    )
    for _ in range(attempts):
        if predicate(section.state):
            return section
        await pilot.pause(0.02)
    raise AssertionError(
        f"changed-files section state never matched -- last state: {section.state!r}"
    )


@pytest.mark.asyncio
async def test_section_renders_aggregated_entries_after_worker_lands(workspace_fixture):
    ws = workspace_fixture
    run_id = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
    _record_turn(
        ws.db, ws.tracker, ws.root, run_id,
        lambda: (ws.root / "a.py").write_text("line1\nCHANGED\n"),
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()
        session = await _mount_console_session(pilot, console, store, ws)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run_id,
        )

        await console._sync_native_console_chat_ui()

        section = await _wait_for_changed_files_state(
            console, pilot, lambda state: bool(state.entries)
        )
        assert section.state.entries[0].path == "a.py"
        assert section.state.entries[0].run_id == run_id
        assert section.display is True


@pytest.mark.asyncio
async def test_guard_skips_idle_ticks_and_recomputes_once_on_new_marker(
    workspace_fixture, monkeypatch
):
    ws = workspace_fixture
    run1 = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
    _record_turn(
        ws.db, ws.tracker, ws.root, run1,
        lambda: (ws.root / "a.py").write_text("line1\nCHANGED\n"),
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()
        session = await _mount_console_session(pilot, console, store, ws)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run1,
        )

        await console._sync_native_console_chat_ui()
        await _wait_for_changed_files_state(
            console, pilot, lambda state: bool(state.entries)
        )

        # Everything above was the legitimate FIRST recompute. Install the
        # counting wrapper on the CLASS -- `_RealProviderBridge` builds a
        # fresh provider instance per call, so an instance-level patch
        # would miss every call after the first.
        calls: list[int] = []
        original = AgentRunsChangeReviewProvider.conversation_changed_files

        def counting(self, row_cache=None):
            calls.append(1)
            return original(self, row_cache=row_cache)

        monkeypatch.setattr(
            AgentRunsChangeReviewProvider, "conversation_changed_files", counting
        )

        for _ in range(6):
            await console._sync_native_console_chat_ui()
            await pilot.pause(0.02)
        assert calls == [], (
            "an unchanged (conversation_id, newest run_id) scope must never "
            f"re-call the provider -- got {len(calls)} call(s)"
        )

        run2 = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
        _record_turn(
            ws.db, ws.tracker, ws.root, run2,
            lambda: (ws.root / "a.py").write_text("line1\nCHANGED AGAIN\n"),
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run2,
        )

        for _ in range(5):
            await console._sync_native_console_chat_ui()
            await pilot.pause(0.02)
        for _ in range(100):
            if calls:
                break
            await pilot.pause(0.02)
        assert len(calls) == 1, (
            f"a new marker's scope change must trigger exactly ONE recompute, "
            f"got {len(calls)}"
        )


@pytest.mark.asyncio
async def test_notes_changed_message_resets_guard_without_clearing_summary(
    workspace_fixture,
):
    ws = workspace_fixture
    run_id = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
    _record_turn(
        ws.db, ws.tracker, ws.root, run_id,
        lambda: (ws.root / "a.py").write_text("line1\nCHANGED\n"),
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()
        session = await _mount_console_session(pilot, console, store, ws)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run_id,
        )

        await console._sync_native_console_chat_ui()
        await _wait_for_changed_files_state(
            console, pilot, lambda state: bool(state.entries)
        )
        before_summary = console._console_changed_files_summary
        assert before_summary

        # `_dispatch_console_changed_files_worker` only STARTS a
        # `thread=True` worker; nothing has been awaited yet, so this
        # assertion right after the direct handler call catches the guard
        # reset's SYNCHRONOUS half -- the note-mutation reset must not have
        # taken the "conversation switch" branch that clears the cache
        # (see `_last_console_changed_files_conversation_id`'s docstring).
        console.handle_console_turn_file_card_notes_changed(
            ConsoleTurnFileCard.NotesChanged(run_id)
        )
        assert console._console_changed_files_summary == before_summary
        assert console._last_console_changed_files_scope is not None

        calls: list[int] = []
        original = AgentRunsChangeReviewProvider.conversation_changed_files

        def counting(self, row_cache=None):
            calls.append(1)
            return original(self, row_cache=row_cache)

        import tldw_chatbook.UI.Screens.change_review_screen as cr_module

        cr_module.AgentRunsChangeReviewProvider.conversation_changed_files = counting
        try:
            console._last_console_changed_files_scope = None
            console.handle_console_turn_file_card_notes_changed(
                ConsoleTurnFileCard.NotesChanged(run_id)
            )
            for _ in range(100):
                if calls:
                    break
                await pilot.pause(0.02)
            assert calls, "a note-mutation reset must trigger one recompute"
        finally:
            cr_module.AgentRunsChangeReviewProvider.conversation_changed_files = (
                original
            )


@pytest.mark.asyncio
async def test_conversation_switch_clears_memo_and_summary_synchronously(
    workspace_fixture,
):
    ws = workspace_fixture
    run_id = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
    _record_turn(
        ws.db, ws.tracker, ws.root, run_id,
        lambda: (ws.root / "a.py").write_text("line1\nCHANGED\n"),
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()
        session1 = await _mount_console_session(pilot, console, store, ws)
        store.append_message(
            session1.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run_id,
        )

        await console._sync_native_console_chat_ui()
        await _wait_for_changed_files_state(
            console, pilot, lambda state: bool(state.entries)
        )
        assert console._console_changed_files_summary

        session2 = store.create_session(session_id="conv-wiring-2")
        session2.persisted_conversation_id = "conv-wiring-2"

        # Synchronous: the clear happens on the SAME call that notices the
        # scope changed, before the off-thread worker has any chance to run.
        console._sync_console_changed_files_if_scope_changed()
        assert console._console_changed_files_summary is None
        assert console._console_changed_files_pruned_rows == 0


@pytest.mark.asyncio
async def test_stale_worker_land_after_conversation_switch_does_not_clobber_summary(
    workspace_fixture,
):
    """Fix 4(c) (final review): `_sync_console_changed_files_if_scope_
    changed` advances `_last_console_changed_files_scope` to the NEW scope
    BEFORE dispatching the recompute worker for it -- so a worker started
    for an OLDER conversation that lands late must not be trusted to still
    be relevant just because "the guard hasn't changed since" (it HAS,
    the guard moved on before dispatch, not after landing). Textual's
    exclusive worker group only guarantees the run_worker() call cancels
    the PRIOR WORKER TASK; a `call_from_thread` callback that worker had
    already queued on the main loop before that cancellation still runs.
    `_land_console_changed_files`/`_land_console_changed_files_empty`
    carry the dispatch-time conversation id and must no-op when it no
    longer matches the CURRENT one."""
    ws = workspace_fixture
    run_id = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
    _record_turn(
        ws.db, ws.tracker, ws.root, run_id,
        lambda: (ws.root / "a.py").write_text("line1\nCHANGED\n"),
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()
        session1 = await _mount_console_session(pilot, console, store, ws)
        store.append_message(
            session1.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run_id,
        )

        await console._sync_native_console_chat_ui()
        await _wait_for_changed_files_state(
            console, pilot, lambda state: bool(state.entries)
        )
        live_summary = console._console_changed_files_summary
        assert live_summary

        # A second conversation becomes active -- as far as the screen is
        # concerned, this is what happens WHILE an older worker (dispatched
        # for CONV_ID) is still in flight.
        session2 = store.create_session(session_id="conv-wiring-stale-2")
        session2.persisted_conversation_id = "conv-wiring-stale-2"
        assert (
            console._character._current_console_rail_conversation_id()
            == "conv-wiring-stale-2"
        )

        # The stale worker's callback, captured at DISPATCH time against
        # CONV_ID, finally lands -- after the conversation has moved on.
        console._land_console_changed_files(CONV_ID, [], 0)
        assert console._console_changed_files_summary == live_summary, (
            "a stale worker land overwrote a newer conversation's summary"
        )

        console._land_console_changed_files_empty(CONV_ID)
        assert console._console_changed_files_summary == live_summary, (
            "a stale empty-land overwrote a newer conversation's summary"
        )

        # A land/empty-land for the CURRENT conversation must still apply.
        console._land_console_changed_files_empty("conv-wiring-stale-2")
        assert console._console_changed_files_summary == ()


@pytest.mark.asyncio
async def test_file_selected_opens_review_screen_with_matching_initials(
    workspace_fixture,
):
    ws = workspace_fixture
    run_id = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
    _record_turn(
        ws.db, ws.tracker, ws.root, run_id,
        lambda: (ws.root / "a.py").write_text("line1\nCHANGED\n"),
    )
    provider = AgentRunsChangeReviewProvider(
        db=ws.db, service=ws.service, conversation_id=CONV_ID
    )
    entries, _pruned = provider.conversation_changed_files()
    entry = entries[0]

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()
        await _mount_console_session(pilot, console, store, ws)

        console.handle_console_changed_files_selected(
            ConsoleChangedFilesSection.FileSelected(
                entry.run_id, entry.snapshot_id, entry.path, entry.root
            )
        )
        for _ in range(50):
            if isinstance(host.screen_stack[-1], ChangeReviewScreen):
                break
            await pilot.pause(0.02)

        pushed = host.screen_stack[-1]
        assert isinstance(pushed, ChangeReviewScreen)
        assert pushed._initial_run_id == entry.run_id
        assert pushed._initial_path == entry.path
        assert pushed._initial_snapshot_id == entry.snapshot_id


@pytest.mark.asyncio
async def test_config_off_renders_nothing_and_never_dispatches_worker(
    workspace_fixture, monkeypatch
):
    ws = workspace_fixture
    run_id = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
    _record_turn(
        ws.db, ws.tracker, ws.root, run_id,
        lambda: (ws.root / "a.py").write_text("line1\nCHANGED\n"),
    )

    calls: list[int] = []
    original = AgentRunsChangeReviewProvider.conversation_changed_files

    def counting(self, row_cache=None):
        calls.append(1)
        return original(self, row_cache=row_cache)

    monkeypatch.setattr(
        AgentRunsChangeReviewProvider, "conversation_changed_files", counting
    )
    # Surgical: only the section's own gate, never the whole module's
    # `get_cli_setting` (dozens of unrelated lookups run in the same tick).
    monkeypatch.setattr(
        ChatScreen, "_console_changed_files_section_enabled", staticmethod(lambda: False)
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()
        session = await _mount_console_session(pilot, console, store, ws)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run_id,
        )

        for _ in range(5):
            await console._sync_native_console_chat_ui()
            await pilot.pause(0.02)

        assert calls == [], "OFF must never dispatch the recompute worker"
        section = console.query_one(
            "#console-changed-files-section", ConsoleChangedFilesSection
        )
        assert section.display is False
        assert not section.state.entries


@pytest.mark.asyncio
async def test_row_cache_skips_git_for_already_seen_rows_on_note_mutation_refresh(
    workspace_fixture, monkeypatch
):
    """Fix round (reviewer-benchmarked: ~18ms/row-pair, ~900ms at turn 50,
    quadratic cumulative without this): a note-mutation-forced refresh
    reuses the screen's row-cache for every row it has already diffed --
    it does not re-run git on them, only the notes join reruns. A NEW
    turn's refresh still pays git cost, but ONLY for its own new row(s).
    """
    _disable_turn_file_cards(monkeypatch)
    ws = workspace_fixture
    run1 = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
    _record_turn(
        ws.db, ws.tracker, ws.root, run1,
        lambda: (ws.root / "a.py").write_text("line1\nCHANGED\n"),
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()
        session = await _mount_console_session(pilot, console, store, ws)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run1,
        )

        await console._sync_native_console_chat_ui()
        await _wait_for_changed_files_state(
            console, pilot, lambda state: bool(state.entries)
        )
        assert console._console_changed_files_row_cache, (
            "the first recompute must have populated the screen's per-row memo"
        )

        recompute_calls: list[int] = []
        row_calls: list[int | None] = []
        original_conversation_changed_files = (
            AgentRunsChangeReviewProvider.conversation_changed_files
        )
        original_changed_files = AgentRunsChangeReviewProvider.changed_files

        def counting_conversation_changed_files(self, row_cache=None):
            # Appended AFTER the real call returns (not before): the poll
            # loops below wait for this list to signal COMPLETION of the
            # recompute (including any per-row `changed_files` calls it
            # makes), not merely that the call was entered -- an
            # append-before-call signal fired too early once and produced
            # a real, reproduced flake (the row-cache assertions below ran
            # against a still-in-flight recompute).
            result = original_conversation_changed_files(
                self, row_cache=row_cache
            )
            recompute_calls.append(1)
            return result

        def counting_changed_files(self, row):
            row_calls.append(row.get("id"))
            return original_changed_files(self, row)

        monkeypatch.setattr(
            AgentRunsChangeReviewProvider,
            "conversation_changed_files",
            counting_conversation_changed_files,
        )
        monkeypatch.setattr(
            AgentRunsChangeReviewProvider, "changed_files", counting_changed_files
        )

        # Note-mutation-forced refresh: the guard resets, but the row cache
        # must survive it (only a genuine conversation switch clears that)
        # -- this recompute should hit git for ZERO rows.
        console.handle_console_turn_file_card_notes_changed(
            ConsoleTurnFileCard.NotesChanged(run1)
        )
        for _ in range(150):
            if recompute_calls:
                break
            await pilot.pause(0.02)
        assert recompute_calls, "the note-mutation reset never triggered a recompute"
        assert row_calls == [], (
            "a note-mutation refresh must reuse the row cache for every "
            f"already-seen row, got {len(row_calls)} git call(s): {row_calls}"
        )

        # A NEW turn's refresh must compute ONLY the new row.
        run2 = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
        _record_turn(
            ws.db, ws.tracker, ws.root, run2,
            lambda: (ws.root / "a.py").write_text("line1\nCHANGED AGAIN\n"),
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run2,
        )
        recompute_calls.clear()
        await console._sync_native_console_chat_ui()
        for _ in range(150):
            if recompute_calls:
                break
            await pilot.pause(0.02)
        assert recompute_calls, "the new marker never triggered a recompute"
        assert len(row_calls) == 1, (
            f"a new turn's refresh must compute only the NEW row, got "
            f"{len(row_calls)}: {row_calls}"
        )


@pytest.mark.asyncio
async def test_worker_failure_leaves_stale_state_and_sync_loop_recovers(
    workspace_fixture, monkeypatch
):
    """A raising `conversation_changed_files` inside the worker: the
    failure is logged, the summary/section keep their last-known (stale)
    state rather than being clobbered, and the sync loop itself is
    unharmed -- a SUBSEQUENT guard-triggered refresh (a new marker) works
    normally once the provider is healthy again.
    """
    from loguru import logger as loguru_logger

    _disable_turn_file_cards(monkeypatch)
    ws = workspace_fixture
    run1 = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
    _record_turn(
        ws.db, ws.tracker, ws.root, run1,
        lambda: (ws.root / "a.py").write_text("line1\nCHANGED\n"),
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()
        session = await _mount_console_session(pilot, console, store, ws)
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run1,
        )

        await console._sync_native_console_chat_ui()
        await _wait_for_changed_files_state(
            console, pilot, lambda state: bool(state.entries)
        )
        stale_summary = console._console_changed_files_summary
        assert stale_summary

        def boom(self, row_cache=None):
            raise RuntimeError("simulated provider failure")

        monkeypatch.setattr(
            AgentRunsChangeReviewProvider, "conversation_changed_files", boom
        )

        run2 = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
        _record_turn(
            ws.db, ws.tracker, ws.root, run2,
            lambda: (ws.root / "a.py").write_text("line1\nCHANGED AGAIN\n"),
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run2,
        )

        warnings: list[str] = []
        sink_id = loguru_logger.add(
            lambda message: warnings.append(message.record["message"]),
            level="WARNING",
        )
        try:
            await console._sync_native_console_chat_ui()
            for _ in range(150):
                if warnings:
                    break
                await pilot.pause(0.02)
        finally:
            loguru_logger.remove(sink_id)

        assert any("recompute failed" in w for w in warnings), (
            f"the worker failure must be logged, got: {warnings}"
        )
        # Stale state survives -- neither the cache nor the mounted section
        # was clobbered by the failed recompute.
        assert console._console_changed_files_summary == stale_summary
        section = console.query_one(
            "#console-changed-files-section", ConsoleChangedFilesSection
        )
        assert section.state.entries == stale_summary

        # The sync loop itself survived the exception (it was caught inside
        # the worker, never escaped to Textual's worker-error handling) --
        # prove the NEXT guard-triggered recompute still works once the
        # provider is healthy again.
        monkeypatch.undo()
        run3 = ws.db.create_run(conversation_id=CONV_ID, agent_kind="primary")
        _record_turn(
            ws.db, ws.tracker, ws.root, run3,
            lambda: (ws.root / "a.py").write_text("line1\nTHIRD CHANGE\n"),
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content=MARKER,
            change_review_run_id=run3,
        )
        await console._sync_native_console_chat_ui()
        section = await _wait_for_changed_files_state(
            console,
            pilot,
            lambda state: bool(state.entries) and state.entries[0].run_id == run3,
        )
        assert section.state.entries[0].run_id == run3
