"""TASK-1971: B/E turn snapshots around agent runs + change_snapshots schema.

Tracker and bridge tests run against REAL git (no mocks — TASK-1970's rule).
The bridge tests drive the real run loop with a scripted gateway whose
streaming callback writes files mid-turn: that is literally the run-window
side effect the feature exists to catch.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker


@pytest.fixture()
def root(tmp_path) -> Path:
    r = tmp_path / "root"
    r.mkdir()
    (r / "seed.txt").write_text("seed\n")
    return r


@pytest.fixture()
def tracker(tmp_path) -> ChangeTurnTracker:
    return ChangeTurnTracker(
        service=ShadowRepoService(data_dir=tmp_path / "appdata")
    )


# -- tracker level ----------------------------------------------------------


def test_a_turn_records_disk_truth(tracker, root):
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    (root / "new.txt").write_text("created\n")
    (root / "seed.txt").write_text("edited\n")

    records = tracker.end_turn(handle)

    assert len(records) == 1
    rec = records[0]
    assert rec.root == str(root)
    assert rec.tracking_error == ""
    assert rec.files_changed == 2
    assert rec.adds >= 2 and rec.baseline_sha != rec.end_sha


def test_a_clean_turn_yields_no_records(tracker, root):
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    records = tracker.end_turn(handle)
    assert records == []


def test_begin_is_nonblocking_and_await_gates(tmp_path, root):
    """B must ride the model's first-token latency: begin_turn returns
    while the snapshot is still running; await_baseline blocks until done.
    """
    events: list[str] = []

    class _SlowService(ShadowRepoService):
        def repo_for_root(self, r):
            repo = super().repo_for_root(r)
            original = repo.snapshot

            def slow_snapshot(message: str) -> str:
                time.sleep(0.4)
                events.append("baseline-finished")
                return original(message)

            repo.snapshot = slow_snapshot  # type: ignore[method-assign]
            return repo

    tracker = ChangeTurnTracker(service=_SlowService(data_dir=tmp_path / "app"))
    started = time.monotonic()
    handle = tracker.begin_turn([root])
    events.append("begin-returned")
    begin_elapsed = time.monotonic() - started

    handle.await_baseline()
    events.append("await-returned")

    assert begin_elapsed < 0.3, "begin_turn blocked on the snapshot"
    assert events == ["begin-returned", "baseline-finished", "await-returned"]


def test_force_add_carveout_for_tool_touched_ignored_paths(tracker, root):
    """A tool write to a .gitignore'd path (.env is the canonical case) must
    surface; a SCRIPT write into an ignored directory stays a documented
    blind spot (force-adding everything would false-positive pre-existing
    ignored files as Added).
    """
    (root / ".gitignore").write_text(".env\nignored_dir/\n")
    ignored_dir = root / "ignored_dir"
    ignored_dir.mkdir()

    handle = tracker.begin_turn([root])
    handle.await_baseline()
    (root / ".env").write_text("SECRET=1\n")
    (ignored_dir / "side_effect.txt").write_text("script wrote this\n")

    records = tracker.end_turn(handle, touched_paths=[str(root / ".env")])

    assert len(records) == 1
    changed = tracker.service.repo_for_root(root).changed_files(
        records[0].baseline_sha, records[0].end_sha
    )
    paths = [c.path for c in changed]
    assert ".env" in paths, "the tool-touched ignored file is invisible"
    assert not any("side_effect" in p for p in paths), (
        "script writes into ignored dirs are OUT of scope by design"
    )


def test_tracking_failure_yields_error_records_never_raises(tmp_path, root):
    tracker = ChangeTurnTracker(
        service=ShadowRepoService(
            data_dir=tmp_path / "app", git_executable="/nonexistent/git"
        )
    )
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    records = tracker.end_turn(handle)
    assert len(records) == 1
    assert records[0].tracking_error != ""
    assert records[0].end_sha == ""


def test_tool_touched_paths_reads_write_tools_only():
    class _Step:
        def __init__(self, tool_name, args):
            self.tool_name = tool_name
            self.args = args

    steps = [
        _Step("write_file", {"file_path": "/w/a.txt", "content": "x"}),
        _Step("read_file", {"file_path": "/w/read-only.txt"}),
        _Step("calculator", {"expression": "1+1"}),
        _Step("write_file", {"file_path": "/w/b.txt", "content": "y"}),
    ]
    touched = ChangeTurnTracker.tool_touched_paths(steps)
    assert touched == ["/w/a.txt", "/w/b.txt"], (
        "read touches would force-add pre-existing ignored files and lie "
        f"an Added row: {touched}"
    )


# -- DB level ---------------------------------------------------------------


def test_v2_database_gains_the_change_snapshots_table_on_open(tmp_path):
    """The DB has no migration framework by design: CREATE IF NOT EXISTS on
    every open IS the mechanism. A file created at v2 must gain the table."""
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL);
        INSERT INTO schema_version (version) VALUES (2);
        CREATE TABLE agent_runs (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
            parent_run_id TEXT, agent_kind TEXT NOT NULL, task TEXT,
            status TEXT NOT NULL, steps TEXT NOT NULL DEFAULT '[]',
            result TEXT, budget TEXT, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, assistant_message_id TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    db = AgentRunsDB(db_path, client_id="t")
    run_id = db.create_run(conversation_id="c1", agent_kind="primary")
    db.record_change_snapshot(
        run_id=run_id,
        root="/w/root",
        baseline_sha="b" * 8,
        end_sha="e" * 8,
        files_changed=2,
        adds=3,
        dels=1,
    )
    rows = db.change_snapshots_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["root"] == "/w/root"
    assert rows[0]["adds"] == 3

    by_conv = db.change_snapshots_for_conversation("c1")
    assert [r["run_id"] for r in by_conv] == [run_id]


# -- bridge level -----------------------------------------------------------


class _SideEffectGateway:
    """Streams a scripted reply and, mid-stream, runs a side-effect callback
    — the exact run-window write the tracker must attribute to the turn.

    Matches the real gateway contract (async generator, positional
    resolution/messages) — the first version was a sync generator and every
    bridge test failed with an empty run.
    """

    def __init__(
        self, scripts, side_effect=None, explode=False, side_effect_on_call=1
    ):
        self._scripts = list(scripts)
        self._side_effect = side_effect
        self._explode = explode
        self._side_effect_on_call = side_effect_on_call
        self._calls = 0

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        script = self._scripts[min(self._calls, len(self._scripts) - 1)]
        self._calls += 1
        if self._side_effect is not None and self._calls >= self._side_effect_on_call:
            self._side_effect()
            self._side_effect = None
        for chunk in script:
            yield chunk
        if self._explode and self._calls >= len(self._scripts):
            raise RuntimeError("provider died mid-turn")


def _bridge_with(tmp_path, gateway, tracker):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=gateway,
        change_tracker=tracker,
    )
    return bridge, db, store, session, assistant.id


def _run(bridge, session, assistant_id, root, **over):
    kwargs = dict(
        conversation_id="conv-1",
        session_id=session.id,
        resolution=object(),
        assistant_message_id=assistant_id,
        model="test-model",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
        change_roots=[root],
    )
    kwargs.update(over)
    return bridge.run_reply(**kwargs)


def _calc_fence() -> str:
    return (
        f"{FENCE_OPEN}\n"
        + json.dumps({"name": "calculator", "arguments": {"expression": "6*7"}})
        + "\n```"
    )


def test_bridge_run_records_a_change_row_matching_disk(tmp_path, root, tracker):
    """The side effect fires on the SECOND provider call -- i.e. after the
    first tool batch has passed the await-B gate. Writing during the FIRST
    provider stream would race the baseline thread (warm process: gateway
    wins, the write lands inside B and vanishes) -- a window that exists
    only for writers that bypass tools, which production has none of: every
    writer, scripts included, is a tool behind the gate. The first version
    of this test wrote pre-gate and passed only by cold-start luck.
    """
    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made_by_run.txt").write_text("hello\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    run_id, outcome = _run(bridge, session, aid, root)

    rows = db.change_snapshots_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["files_changed"] == 1
    changed = tracker.service.repo_for_root(root).changed_files(
        rows[0]["baseline_sha"], rows[0]["end_sha"]
    )
    assert [c.path for c in changed] == ["made_by_run.txt"]


def test_bridge_run_with_no_changes_records_no_row(tmp_path, root, tracker):
    gateway = _SideEffectGateway([["done."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    run_id, _outcome = _run(bridge, session, aid, root)
    assert db.change_snapshots_for_run(run_id) == []


def test_failed_run_still_records_its_end_snapshot(tmp_path, root, tracker):
    """A run that died halfway through editing is when review matters MOST."""
    gateway = _SideEffectGateway(
        [[_calc_fence()], ["partial"]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "half_done.txt").write_text("partial\n"),
        explode=True,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    run_id, outcome = _run(bridge, session, aid, root)

    assert outcome.status != "completed"
    rows = db.change_snapshots_for_run(run_id)
    assert len(rows) == 1, "the failed run's half-finished edits are unreviewable"


def test_tracking_never_blocks_the_reply(tmp_path, root):
    """Spec failure posture: git broken -> the agent reply still completes."""
    broken = ChangeTurnTracker(
        service=ShadowRepoService(
            data_dir=tmp_path / "app", git_executable="/nonexistent/git"
        )
    )
    gateway = _SideEffectGateway([["fine."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, broken)

    run_id, outcome = _run(bridge, session, aid, root)

    assert outcome.final_text.strip() == "fine."
    rows = db.change_snapshots_for_run(run_id)
    assert len(rows) == 1 and rows[0]["tracking_error"] != ""


def test_baseline_completes_before_the_first_tool_executes(tmp_path, root):
    """The spec's ordering contract: B rides first-token latency but MUST be
    done before any tool touches disk — otherwise the tool's own write races
    into the baseline and vanishes from the diff.
    """
    events: list[str] = []

    class _SlowService(ShadowRepoService):
        def repo_for_root(self, r):
            repo = super().repo_for_root(r)
            original = repo.snapshot

            def slow_snapshot(message: str) -> str:
                if "baseline" in message:
                    time.sleep(0.5)
                    events.append("baseline-finished")
                return original(message)

            repo.snapshot = slow_snapshot  # type: ignore[method-assign]
            return repo

    tracker = ChangeTurnTracker(service=_SlowService(data_dir=tmp_path / "app"))
    fence = (
        f"{FENCE_OPEN}\n"
        + json.dumps({"name": "calculator", "arguments": {"expression": "6*7"}})
        + "\n```"
    )
    gateway = _SideEffectGateway([[fence], ["42."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    def probe_review(calls):
        events.append("review-called")
        return {}

    run_id, outcome = _run(
        bridge, session, aid, root, review_tool_calls=probe_review
    )

    assert "baseline-finished" in events and "review-called" in events
    assert events.index("baseline-finished") < events.index("review-called"), (
        f"a tool could execute before B settled: {events}"
    )


# -- wiring: roots resolution + registration hook ---------------------------


def test_folder_binding_roots_includes_ro_and_never_sandbox(tmp_path, monkeypatch):
    """Tracking is about what happened on DISK: a script can write into a
    read-only root even though the file tools cannot, so ro bindings are in.
    The sandbox is app-managed scratch and would be pure review noise.
    """
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Tools import workspace_file_roots as wfr
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="t")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="A")
    rw = tmp_path / "rw"
    rw.mkdir()
    ro = tmp_path / "ro"
    ro.mkdir()
    registry.add_folder_binding("ws-a", rw, allow_write=True)
    registry.add_folder_binding("ws-a", ro)
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)

    roots = wfr.folder_binding_roots("ws-a")

    assert set(roots) == {rw.resolve(), ro.resolve()}
    assert wfr.folder_binding_roots(None) == ()


def test_adding_a_folder_binding_snapshots_it_in_the_background(tmp_path):
    """Spec §2: the FIRST snapshot happens at registration, so the first
    send never absorbs the cost of hashing a whole tree. The hook is
    best-effort and must not slow or fail registration itself.
    """
    import time as _time

    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="t")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="A")
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "code.py").write_text("x = 1\n")

    registry.add_folder_binding("ws-a", folder)

    # The hook's default-constructed service resolves the same isolated app
    # data dir this test process sees, so a fresh service finds its tip.
    service = ShadowRepoService()
    deadline = _time.monotonic() + 15.0
    tip = None
    while _time.monotonic() < deadline:
        tip = service.repo_for_root(folder).tip()
        if tip:
            break
        _time.sleep(0.05)
    assert tip, "the registered root never received its initial snapshot"


def test_carveout_survives_a_symlink_spelled_root(tmp_path):
    """Review finding: `_paths_within` resolves each touched path but the
    roots were stored UNRESOLVED — a run whose root arrived spelled through
    a symlink made `relative_to` fail, silently skipping the force-add: the
    `.env` carve-out dying without a trace.
    """
    real_root = tmp_path / "real_root"
    real_root.mkdir()
    (real_root / ".gitignore").write_text(".env\n")
    link = tmp_path / "root_link"
    try:
        link.symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unsupported on this platform/permission level")

    tracker = ChangeTurnTracker(
        service=ShadowRepoService(data_dir=tmp_path / "appdata")
    )
    handle = tracker.begin_turn([link])  # the SYMLINK spelling
    handle.await_baseline()
    (real_root / ".env").write_text("SECRET=1\n")

    records = tracker.end_turn(
        handle, touched_paths=[str(real_root / ".env")]
    )

    assert len(records) == 1 and not records[0].tracking_error
    changed = tracker.service.repo_for_root(real_root).changed_files(
        records[0].baseline_sha, records[0].end_sha
    )
    assert ".env" in [c.path for c in changed], (
        "the carve-out silently died for a symlink-spelled root"
    )


# -- TASK-1972: the transcript summary row ----------------------------------


def _tool_rows(store, session):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    return [
        m
        for m in store.messages_for_session(session.id)
        if m.role is ConsoleMessageRole.TOOL
    ]


def test_a_change_turn_emits_the_summary_row_with_real_counts(
    tmp_path, root, tracker
):
    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("one\ntwo\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    run_id, _ = _run(bridge, session, aid, root)

    rows = [m for m in _tool_rows(store, session) if m.content.startswith("✎")]
    assert len(rows) == 1
    assert "1 file" in rows[0].content
    assert "+2" in rows[0].content
    assert rows[0].change_review_run_id == run_id, (
        "the row does not know WHICH turn it reviews"
    )


def test_a_clean_turn_emits_no_summary_row(tmp_path, root, tracker):
    gateway = _SideEffectGateway([["done."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    _run(bridge, session, aid, root)
    assert not [
        m for m in _tool_rows(store, session) if m.content.startswith("✎")
    ]


def test_tracking_failure_emits_the_warning_row(tmp_path, root):
    broken = ChangeTurnTracker(
        service=ShadowRepoService(
            data_dir=tmp_path / "app", git_executable="/nonexistent/git"
        )
    )
    gateway = _SideEffectGateway([["fine."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, broken)
    _run(bridge, session, aid, root)
    warns = [
        m
        for m in _tool_rows(store, session)
        if "change tracking failed" in m.content
    ]
    assert len(warns) == 1, "a tracking failure must be DISCLOSED in the transcript"


def test_summary_row_survives_the_next_message(tmp_path, root, tracker):
    """TASK-1842's whole arc: display-only rows must survive recompute."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    _run(bridge, session, aid, root)
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="follow-up"
    )
    rows = [m for m in _tool_rows(store, session) if m.content.startswith("✎")]
    assert len(rows) == 1, "the summary row was destroyed by the next message"


def test_resume_re_derives_the_summary_row_byte_identical(tmp_path, root, tracker):
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    run_id, _ = _run(bridge, session, aid, root)
    live = [
        m for m in _tool_rows(store, session) if m.content.startswith("✎")
    ]
    assert live, "precondition"

    fresh = ConsoleAgentBridge(
        agent_runs_db=db, store=None, provider_gateway=None
    )
    resumed = [
        m
        for _anchor, block in fresh.resume_marker_messages("conv-1")
        for m in block
        if m.content.startswith("✎")
    ]
    assert [m.content for m in resumed] == [m.content for m in live]
    assert resumed[0].change_review_run_id == run_id


def test_review_changes_action_offered_only_for_summary_rows():
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Chat.console_message_actions import (
        ConsoleMessageActionService,
    )

    summary = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="✎ Edited 1 file  +2 −0 — review with `v`",
        change_review_run_id="run-1",
    )
    plain_marker = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL, content="⚙ calculator → 42"
    )
    svc = ConsoleMessageActionService()
    assert "review-changes" in [
        a.action_id for a in svc.available_actions(summary)
    ]
    assert "review-changes" not in [
        a.action_id for a in svc.available_actions(plain_marker)
    ]


def test_bridge_exposes_a_provider_for_the_review_screen(
    tmp_path, root, tracker
):
    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    run_id, _ = _run(bridge, session, aid, root)

    provider = bridge.change_review_provider("conv-1")
    assert provider is not None
    turns = provider.turns()
    assert [t.run_id for t in turns] == [run_id]

    untracked = ConsoleAgentBridge = None  # noqa: F841 -- reuse import below
    from tldw_chatbook.Chat.console_agent_bridge import (
        ConsoleAgentBridge as _B,
    )

    no_tracker = _B(agent_runs_db=db, store=store, provider_gateway=gateway)
    assert no_tracker.change_review_provider("conv-1") is None


@pytest.mark.asyncio
async def test_the_opener_pushes_the_screen_and_selects_the_turn(
    tmp_path, root, tracker
):
    """The `v`/inspector opener on the PRODUCTION ChatScreen: derives the
    run-store conversation id, builds the provider through the bridge, pushes
    the Review screen, and selects THAT turn. The opener is where an invented
    method name already slipped in once during this task -- it needs a test
    on the real screen object, not a reading.
    """
    from Tests.UI.app_factory import _build_test_app
    from textual.app import App

    from tldw_chatbook.UI.Screens.change_review_screen import ChangeReviewScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    # run_reply spins its own event loop; inside an async test that loop
    # collides with pytest-asyncio's ("Cannot run the event loop while
    # another loop is running"). Production calls it via asyncio.to_thread
    # -- so does this test.
    import asyncio as _asyncio

    run_id, outcome = await _asyncio.to_thread(_run, bridge, session, aid, root)
    assert outcome.status not in ("error",), outcome.steps
    assert db.change_snapshots_for_run(run_id), "precondition: the run recorded rows"

    class _ConsoleHarness(App):
        def __init__(self, app_instance):
            super().__init__()
            self.app_instance = app_instance

        async def on_mount(self) -> None:
            await self.push_screen(ChatScreen(self.app_instance))

    app = _build_test_app()
    # Same native-ready configuration the workbench harness applies -- the
    # Console controller is built lazily and stays None without it.
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "local-model",
            },
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    harness = _ConsoleHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        chat_screen = harness.screen_stack[-1]
        assert isinstance(chat_screen, ChatScreen)
        # The harness app has no chachanotes db, so its own bridge factory
        # returns None by design -- substitute THIS test's real bridge
        # (real tracker, real db, real turn) at the accessor seam.
        chat_screen._ensure_console_agent_bridge = lambda: bridge
        chat_screen._ensure_console_chat_controller()
        controller = chat_screen._console_chat_controller
        assert controller is not None
        controller.store.ensure_session()
        # The run-store id for the harness's session is the session id
        # itself (no persisted conversation) -- point the bridge's provider
        # at the id the run actually recorded under instead.
        chat_screen._console_chat_controller._agent_conversation_id = (
            lambda _sid: "conv-1"
        )

        chat_screen._open_change_review(run_id)
        review = await _wait_for_screen(harness, pilot, ChangeReviewScreen)
        assert review is not None, "the opener never pushed the Review screen"

        turns = review._provider.turns()
        assert [t.run_id for t in turns] == [run_id]


@pytest.mark.asyncio
async def test_the_summary_rows_own_review_action_opens_the_screen(
    tmp_path, root, tracker
):
    """TASK-2030 (live-UAT headline defect): selecting the rendered ✎ row
    and invoking its review action must open the Review screen.

    The defect class: TOOL markers are display-only rows, deliberately NOT
    tree nodes, so `store.get_message(marker_id)` ALWAYS raises -- and the
    action handler resolved the store row before dispatching, killing the
    row's own advertised affordance ("review with `v`") on the live app
    while the direct-call opener test stayed green. This test goes through
    the REAL chain the user does: transcript render -> row selection ->
    `invoke_selected_action("review-changes")` -> button dispatch ->
    handler -> pushed screen.
    """
    from Tests.UI.app_factory import _build_test_app
    from textual.app import App

    from tldw_chatbook.UI.Screens.change_review_screen import ChangeReviewScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_transcript import (
        ConsoleTranscript,
    )

    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    import asyncio as _asyncio

    run_id, outcome = await _asyncio.to_thread(_run, bridge, session, aid, root)
    assert outcome.status not in ("error",), outcome.steps
    marker = next(
        m for m in _tool_rows(store, session) if m.content.startswith("✎")
    )
    assert marker.change_review_run_id == run_id

    class _ConsoleHarness(App):
        def __init__(self, app_instance):
            super().__init__()
            self.app_instance = app_instance

        async def on_mount(self) -> None:
            await self.push_screen(ChatScreen(self.app_instance))

    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "local-model"},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "local-model",
            },
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "local-model"
    harness = _ConsoleHarness(app)
    async with harness.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        chat_screen = harness.screen_stack[-1]
        assert isinstance(chat_screen, ChatScreen)
        # The run's rows live in THIS test's store -- make it the screen's
        # store before the lazy controller builds, then render it for real.
        chat_screen._console_chat_store = store
        chat_screen._ensure_console_agent_bridge = lambda: bridge
        chat_screen._ensure_console_chat_controller()
        controller = chat_screen._console_chat_controller
        assert controller is not None
        chat_screen._console_chat_controller._agent_conversation_id = (
            lambda _sid: "conv-1"
        )
        await chat_screen._sync_native_console_transcript()
        await pilot.pause()

        transcript = chat_screen.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        rendered = [
            m
            for m in transcript._messages
            if str(getattr(m, "content", "")).startswith("✎")
        ]
        assert rendered, "precondition: the ✎ row rendered in the transcript"
        transcript.select_message(rendered[0].id)
        await pilot.pause()
        transcript.action_invoke_selected_action("review-changes")
        await pilot.pause()

        review = await _wait_for_screen(harness, pilot, ChangeReviewScreen)
        assert review is not None, (
            "the ✎ row's own review action never opened the Review screen"
        )
        turns = review._provider.turns()
        assert [t.run_id for t in turns] == [run_id]

        # AC#3: a genuinely-unknown target still gets the failure toast --
        # the display-model path is not a blanket bypass of resolution.
        harness.pop_screen()
        await pilot.pause()
        toasts: list[str] = []
        app.notify = lambda msg, **kw: toasts.append(str(msg))

        class _Btn:
            id = "console-message-action-review-changes-not-a-row"

        class _Ev:
            button = _Btn()

            def stop(self) -> None:
                pass

        handled = await chat_screen.handle_console_message_action(_Ev())
        assert handled is True
        assert any("no longer exists" in t for t in toasts), toasts
        assert not isinstance(harness.screen, ChangeReviewScreen)


async def _wait_for_screen(harness, pilot, screen_type, timeout: float = 8.0):
    import time as _t

    deadline = _t.monotonic() + timeout
    while _t.monotonic() < deadline:
        if isinstance(harness.screen, screen_type):
            return harness.screen
        await pilot.pause(0.05)
    return None


def test_resume_re_derives_the_failure_row_too(tmp_path, root):
    """Review finding: live emitted the ⚠ tracking-failed row but resume
    did not -- a resumed transcript silently hid that a turn's tracking
    failed, breaking the byte-identical marker parity rule."""
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    broken = ChangeTurnTracker(
        service=ShadowRepoService(
            data_dir=tmp_path / "app", git_executable="/nonexistent/git"
        )
    )
    gateway = _SideEffectGateway([["fine."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, broken)
    _run(bridge, session, aid, root)
    live = [
        m.content
        for m in _tool_rows(store, session)
        if "change tracking failed" in m.content
    ]
    assert live, "precondition: the live run disclosed the failure"

    fresh = ConsoleAgentBridge(
        agent_runs_db=db, store=None, provider_gateway=None
    )
    resumed = [
        m.content
        for _anchor, block in fresh.resume_marker_messages("conv-1")
        for m in block
        if "change tracking failed" in m.content
    ]
    assert resumed == live, "the failure disclosure vanished on resume"
