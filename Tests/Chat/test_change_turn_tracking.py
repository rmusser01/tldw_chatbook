"""TASK-1971: B/E turn snapshots around agent runs + change_snapshots schema.

Tracker and bridge tests run against REAL git (no mocks — TASK-1970's rule).
The bridge tests drive the real run loop with a scripted gateway whose
streaming callback writes files mid-turn: that is literally the run-window
side effect the feature exists to catch.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from Tests.Agents.test_agent_service import SUBAGENT_PROMPT_PREFIX
from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Chat.console_agent_bridge import (
    CHANGE_REVIEW_BASELINE_BYPASS_TOOLS,
    CHANGE_REVIEW_BASELINE_WAIT_SECONDS,
    ConsoleAgentBridge,
    build_change_review_dispatch_gate,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
from tldw_chatbook.Workspaces.change_review_finalization import (
    ChangeReviewFinalizationCoordinator,
    ChangeReviewFinalizeResult,
)
from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker, TurnHandle
from tldw_chatbook.Workspaces.change_review_consent import SkippedReviewRoot

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp


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


def test_change_review_dispatch_gate_bypasses_only_fixed_pure_runtime_tools():
    waits: list[float] = []
    gate = build_change_review_dispatch_gate(
        lambda timeout: waits.append(timeout) or True
    )

    gate(
        [ToolCall(name=name, args={}) for name in CHANGE_REVIEW_BASELINE_BYPASS_TOOLS],
        CHANGE_REVIEW_BASELINE_BYPASS_TOOLS,
    )
    assert waits == []

    for name in (
        "spawn_subagent",
        "install_skill",
        "run_skill_script",
        "send_to_agent",
        "provider_tool",
        "unknown_tool",
    ):
        gate = build_change_review_dispatch_gate(
            lambda timeout: waits.append(timeout) or True
        )
        gate([ToolCall(name=name, args={})], frozenset())

    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS] * 6

    collision_gate = build_change_review_dispatch_gate(
        lambda timeout: waits.append(timeout) or True
    )
    collision_gate(
        [ToolCall(name="skill_file", args={})],
        frozenset({"find_tools", "load_tools"}),
    )
    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS] * 7


def test_change_review_dispatch_gate_waits_for_mixed_batch_and_warns():
    waits: list[float] = []
    warnings: list[bool] = []
    gate = build_change_review_dispatch_gate(
        lambda timeout: waits.append(timeout) or False,
        on_timeout=lambda: warnings.append(True),
    )

    gate(
        [
            ToolCall(name="find_tools", args={}),
            ToolCall(name="provider_tool", args={}),
        ],
        CHANGE_REVIEW_BASELINE_BYPASS_TOOLS,
    )

    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS]
    assert warnings == [True]

    gate(
        [ToolCall(name="provider_tool", args={})],
        CHANGE_REVIEW_BASELINE_BYPASS_TOOLS,
    )
    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS]
    assert warnings == [True]


def test_change_review_dispatch_gate_coalesces_concurrent_waiters():
    entered = threading.Event()
    release = threading.Event()
    waits: list[float] = []

    def await_baseline(timeout: float) -> bool:
        waits.append(timeout)
        entered.set()
        assert release.wait(timeout=1)
        return False

    gate = build_change_review_dispatch_gate(await_baseline)
    calls = [ToolCall(name="provider_tool", args={})]
    first = threading.Thread(target=gate, args=(calls, frozenset()))
    second = threading.Thread(target=gate, args=(calls, frozenset()))
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive() and not second.is_alive()
    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS]


def test_tracker_supports_a_caller_owned_synchronous_lifecycle(tracker, root):
    """The app-owned coordinator must be the only owner of worker threads."""
    handle = tracker.new_turn_handle([root])

    tracker.populate_baseline(handle)
    assert handle.await_baseline(timeout=0) is True

    (root / "caller-owned.txt").write_text("changed\n")
    records = tracker.finish_turn(handle)

    assert len(records) == 1
    assert records[0].root == str(root)
    assert records[0].files_changed == 1


def test_timed_out_baseline_rejects_late_success(tmp_path, root):
    entered = threading.Event()
    release = threading.Event()

    class _HeldService(ShadowRepoService):
        def repo_for_root(self, r):
            repo = super().repo_for_root(r)
            original = repo.snapshot

            def held_snapshot(message: str) -> str:
                if message == "turn baseline":
                    entered.set()
                    release.wait(timeout=2)
                return original(message)

            repo.snapshot = held_snapshot  # type: ignore[method-assign]
            return repo

    tracker = ChangeTurnTracker(service=_HeldService(data_dir=tmp_path / "app"))
    handle = tracker.new_turn_handle([root])
    worker = threading.Thread(target=tracker.populate_baseline, args=(handle,))
    worker.start()
    assert entered.wait(timeout=1)

    assert handle.await_baseline(timeout=0.01) is False
    release.set()
    worker.join(timeout=2)

    records = tracker.finish_turn(handle)
    assert len(records) == 1
    assert "baseline snapshot still running" in records[0].tracking_error
    assert records[0].baseline_sha == ""


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


def test_change_snapshot_batch_commits_one_complete_window(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(conversation_id="c1", agent_kind="primary")
    records = [
        {
            "root": "/w/one",
            "baseline_sha": "b1",
            "end_sha": "e1",
            "files_changed": 1,
            "adds": 2,
            "dels": 0,
        },
        {
            "root": "/w/two",
            "baseline_sha": "b2",
            "end_sha": "e2",
            "tracking_error": "snapshot failed",
        },
    ]

    db.record_change_snapshots_batch(run_id=run_id, records=records, kind="turn")

    rows = db.change_snapshots_for_run(run_id)
    assert [(row["root"], row["tracking_error"]) for row in rows] == [
        ("/w/one", ""),
        ("/w/two", "snapshot failed"),
    ]


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


def _bridge_with(tmp_path, gateway, tracker, coordinator=None):
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
        change_finalization_coordinator=coordinator,
    )
    return bridge, db, store, session, assistant.id


def _run(bridge, session, assistant_id, root, **over):
    kwargs = dict(
        conversation_id="conv-1",
        session_id=session.id,
        resolution=ConsoleProviderResolution(
            provider="llama_cpp",
            base_url="",
            model="test-model",
            ready=True,
            execution_key="llama_cpp",
        ),
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


def test_bridge_returns_before_coordinated_end_snapshot_finishes(
    tmp_path, root
):
    end_entered = threading.Event()
    release_end = threading.Event()

    class _HeldEndTracker(ChangeTurnTracker):
        def finish_turn(self, handle, touched_paths=(), *, end_shas=None):
            end_entered.set()
            release_end.wait(timeout=2)
            return super().finish_turn(
                handle, touched_paths=touched_paths, end_shas=end_shas
            )

    tracker = _HeldEndTracker(
        service=ShadowRepoService(data_dir=tmp_path / "appdata")
    )
    gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (root / "made_by_run.txt").write_text("hello\n"),
    )
    publications = []
    db_holder = {}

    def publish(item):
        publications.append(item)
        db_holder["db"].record_change_snapshots_batch(
            run_id=item.run_id,
            records=[record.__dict__ for record in item.records],
            kind=item.kind,
        )

    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publish,
        worker_count=1,
        capacity=4,
    )
    bridge, db, store, session, aid = _bridge_with(
        tmp_path, gateway, tracker, coordinator
    )
    db_holder["db"] = db

    run_id, outcome = _run(bridge, session, aid, root)

    assert outcome.final_text.strip() == "done."
    assert end_entered.wait(timeout=1)
    assert db.change_snapshots_for_run(run_id) == []
    release_end.set()
    assert coordinator.wait_idle(timeout=2)
    assert len(db.change_snapshots_for_run(run_id)) == 1
    coordinator.shutdown(timeout=1)


def test_bridge_surfaces_capacity_error_when_error_channel_is_saturated(
    tmp_path, root, tracker
):
    class _Reservation:
        roots = (str(root),)
        admission_error = "change-review error publication channel is at capacity"

        @staticmethod
        def await_baseline(timeout=120.0):
            del timeout
            return True

    class _SaturatedCoordinator:
        @staticmethod
        def register(_roots, *, survivor_key=""):
            return _Reservation()

        @staticmethod
        def finalize(_reservation, **_kwargs):
            return ChangeReviewFinalizeResult.OVERLOAD_VISIBLE

    coordinator = _SaturatedCoordinator()
    bridge, _db, store, session, aid = _bridge_with(
        tmp_path,
        _SideEffectGateway([["done."]]),
        tracker,
        coordinator,
    )

    _run(
        bridge,
        session,
        aid,
        root,
        change_root_aliases=["folder-safe"],
    )

    failures = [
        message.content
        for message in _tool_rows(store, session)
        if "change tracking failed" in message.content
    ]
    assert len(failures) == 1
    assert "error publication channel is at capacity" in failures[0]
    assert "folder-safe" in failures[0]
    assert str(root.resolve()) not in failures[0]


def test_bridge_does_not_append_capacity_marker_after_coordinator_shutdown(
    tmp_path, root, tracker
):
    class _Reservation:
        roots = (str(root),)
        admission_error = "change-review coordinator is at capacity"

        @staticmethod
        def await_baseline(timeout=120.0):
            del timeout
            return True

    class _StoppedCoordinator:
        @staticmethod
        def register(_roots, *, survivor_key=""):
            return _Reservation()

        @staticmethod
        def finalize(_reservation, **_kwargs):
            return ChangeReviewFinalizeResult.REJECTED

    bridge, _db, store, session, aid = _bridge_with(
        tmp_path,
        _SideEffectGateway([["done."]]),
        tracker,
        _StoppedCoordinator(),
    )

    _run(bridge, session, aid, root)

    assert not [
        message
        for message in _tool_rows(store, session)
        if "change tracking failed" in message.content
    ]


def test_third_turn_starts_while_second_review_finalization_is_held(
    tmp_path, root
):
    second_end_entered = threading.Event()
    release_second_end = threading.Event()

    class _HoldSecondEndTracker(ChangeTurnTracker):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ends = 0

        def finish_turn(self, handle, touched_paths=(), *, end_shas=None):
            self.ends += 1
            if self.ends == 2:
                second_end_entered.set()
                release_second_end.wait(timeout=3)
            return super().finish_turn(
                handle, touched_paths=touched_paths, end_shas=end_shas
            )

    class _ThreeTurnGateway(_SideEffectGateway):
        def __init__(self):
            super().__init__([["one"], ["two"], ["three"]])
            self.third_started = threading.Event()

        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            if self._calls == 2:
                self.third_started.set()
            async for chunk in super().stream_chat(
                resolution, messages, tools=tools, **kwargs
            ):
                yield chunk

    tracker = _HoldSecondEndTracker(
        service=ShadowRepoService(data_dir=tmp_path / "appdata")
    )
    gateway = _ThreeTurnGateway()
    db_holder = {}

    def publish(item):
        db_holder["db"].record_change_snapshots_batch(
            run_id=item.run_id,
            records=[record.__dict__ for record in item.records],
            kind=item.kind,
        )

    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publish,
        worker_count=1,
        capacity=4,
    )
    bridge, db, store, session, first_assistant = _bridge_with(
        tmp_path, gateway, tracker, coordinator
    )
    db_holder["db"] = db

    _run(bridge, session, first_assistant, root)
    assert coordinator.wait_idle(timeout=2)

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="two")
    second_assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    second_run_id, _second_outcome = _run(
        bridge, session, second_assistant.id, root
    )
    bridge.record_run_assistant_message(second_run_id, "persisted-second")
    assert db.get_run(second_run_id)["assistant_message_id"] == "persisted-second"
    assert second_end_entered.wait(timeout=1)

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="three")
    third_assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    third_result = []
    third_thread = threading.Thread(
        target=lambda: third_result.append(
            _run(bridge, session, third_assistant.id, root)
        )
    )
    third_thread.start()

    assert gateway.third_started.wait(timeout=1), (
        "turn three remained blocked behind turn two's file-review E snapshot"
    )
    release_second_end.set()
    third_thread.join(timeout=3)
    assert third_result and third_result[0][1].final_text.strip() == "three"
    assert coordinator.wait_idle(timeout=3)
    coordinator.shutdown(timeout=1)


def test_cancelled_turn_still_schedules_coordinated_finalization(
    tmp_path, root
):
    tracker = ChangeTurnTracker(
        service=ShadowRepoService(data_dir=tmp_path / "appdata")
    )
    gateway = _SideEffectGateway([["never used"]])
    publications = []
    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publications.append,
        worker_count=1,
        capacity=2,
    )
    bridge, _db, _store, session, aid = _bridge_with(
        tmp_path, gateway, tracker, coordinator
    )
    scheduled = []
    original_finalize = coordinator.finalize

    def recording_finalize(*args, **kwargs):
        scheduled.append(kwargs["run_id"])
        return original_finalize(*args, **kwargs)

    coordinator.finalize = recording_finalize  # type: ignore[method-assign]

    run_id, outcome = _run(
        bridge,
        session,
        aid,
        root,
        should_cancel=lambda: True,
    )

    assert outcome.status == "cancelled"
    assert scheduled == [run_id]
    assert coordinator.wait_idle(timeout=2)
    coordinator.shutdown(timeout=1)


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


def test_review_runs_before_baseline_gate_and_tool_dispatch_waits(tmp_path, root):
    """Permission review precedes the bounded B gate; invocation follows B."""
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
    gateway = _SideEffectGateway(
        [[fence], ["42."]],
        side_effect=lambda: events.append("tool-finished"),
        side_effect_on_call=2,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    # PR2a Task 5: an AgentService-wired hook takes `(calls, run_id)` --
    # the change-tracker wrapper passes it straight through.
    def probe_review(calls, run_id):
        events.append("review-called")
        return {}

    run_id, outcome = _run(
        bridge, session, aid, root, review_tool_calls=probe_review
    )

    assert "baseline-finished" in events and "review-called" in events
    assert events.index("review-called") < events.index("baseline-finished"), (
        f"permission review did not precede the baseline gate: {events}"
    )
    assert events.index("baseline-finished") < events.index("tool-finished"), (
        f"tool dispatch raced ahead of the baseline: {events}"
    )


def test_bridge_timeout_continues_dispatch_and_warns_with_root_alias(
    tmp_path, root, tracker
):
    waits: list[float] = []

    class _Reservation:
        def __init__(self) -> None:
            self.roots = (str(root.resolve()),)
            self.admission_error = ""
            self._handle = TurnHandle([root.resolve()])

    class _TimeoutCoordinator:
        @staticmethod
        def register(_roots, *, survivor_key=""):
            return _Reservation()

        @staticmethod
        def await_baseline(reservation, timeout):
            waits.append(timeout)
            reservation._handle.errors[str(root.resolve())] = (
                "baseline snapshot still running after 3s"
            )
            return False

        @staticmethod
        def finalize(_reservation, **_kwargs):
            return ChangeReviewFinalizeResult.SCHEDULED

    bridge, _db, store, session, aid = _bridge_with(
        tmp_path,
        _SideEffectGateway([[_calc_fence()], ["done."]]),
        tracker,
        _TimeoutCoordinator(),
    )

    _run(
        bridge,
        session,
        aid,
        root,
        change_root_aliases=["folder-safe"],
    )

    warnings = [
        row.content
        for row in _tool_rows(store, session)
        if row.content.startswith("⚠ change review skipped")
    ]
    assert waits == [CHANGE_REVIEW_BASELINE_WAIT_SECONDS]
    assert warnings == [
        "⚠ change review skipped folder-safe: baseline timed out; "
        "this turn's changes are not tracked"
    ]
    assert str(root.resolve()) not in warnings[0]


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

    registry.set_change_review_enabled("ws-a", True)
    roots = wfr.folder_binding_roots("ws-a")

    assert set(roots) == {rw.resolve(), ro.resolve()}
    assert wfr.folder_binding_roots(None) == ()


def test_app_owner_snapshots_an_enabled_folder_binding_in_background(tmp_path):
    """The attached bounded owner, not registry persistence, prepares roots."""
    import time as _time

    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces import (
        ChangeReviewConsentService,
        LocalWorkspaceRegistryService,
    )

    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="t")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="A")
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "code.py").write_text("x = 1\n")

    review = ChangeReviewConsentService(registry)
    registry.attach_change_review_consent_service(review)
    registry.set_change_review_enabled("ws-a", True)
    try:
        registry.add_folder_binding("ws-a", folder)

        service = ShadowRepoService()
        deadline = _time.monotonic() + 15.0
        tip = None
        while _time.monotonic() < deadline:
            tip = service.repo_for_root(folder).tip()
            if tip:
                break
            _time.sleep(0.05)
        assert tip, "the registered root never received its initial snapshot"
    finally:
        review.shutdown(timeout=1.0)


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


@pytest.mark.parametrize(
    ("alias", "reason"),
    [
        ("folder-preparing", "Preparing change history"),
        ("folder-failed", "Change history preparation failed"),
    ],
)
def test_skipped_review_root_emits_alias_only_warning_without_snapshot_state(
    tmp_path, tracker, alias, reason
):
    """Readiness warnings never masquerade as canonical-root snapshots."""
    gateway = _SideEffectGateway([["done."]])
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    run_id, outcome = _run(
        bridge,
        session,
        aid,
        tmp_path / "unused-root",
        change_roots=[],
        change_review_skipped_roots=(
            SkippedReviewRoot(alias=alias, reason=reason),
        ),
    )

    assert outcome.status == "done"
    warnings = [
        row
        for row in _tool_rows(store, session)
        if "change review skipped" in row.content.lower()
    ]
    assert [row.content for row in warnings] == [
        f"⚠ change review skipped {alias}: {reason}"
    ]
    assert db.change_snapshots_for_run(run_id) == []
    assert db.roots_with_change_snapshots() == set()


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
    projected = [
        message
        for _anchor, block in fresh.change_review_marker_messages("conv-1")
        for message in block
    ]
    assert [message.content for message in projected] == [
        message.content for message in live
    ]
    assert all(message.change_review_run_id == run_id for message in projected)


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

    class _ConsoleHarness(ConsolidatedCSSApp):
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

    class _ConsoleHarness(ConsolidatedCSSApp):
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


# -- PR3a-1 Task 6c (audit F2): a survivor's writes vs the turn window ------
#
# PR 3a lets a sub-agent outlive the `run_reply` that spawned it. The turn's
# E snapshot is taken in that call's `finally`, so every byte a survivor
# writes afterwards falls OUTSIDE its own turn's window -- into the NEXT
# turn's baseline (invisible in every record) or into the next turn's diff
# (attributed to an agent that never made it). Both are silent.


class _FleetSurvivorGateway:
    """One primary script per turn, plus a sub-agent turn gated on an Event.

    The child's disk write fires on the child's own thread the moment the
    gate opens -- the same "run-window side effect" technique
    `_SideEffectGateway` uses for the primary, moved onto a survivor.

    The gate is awaited through ``run_in_executor`` for the reason
    `Tests/Chat/test_console_agent_bridge.py::_FleetTwoChildGateway`
    documents at length: a bare ``.wait()`` inside a coroutine blocks the
    one thread driving that loop.
    """

    def __init__(
        self,
        parent_scripts,
        gate: threading.Event,
        child_side_effect=None,
        parent_side_effect=None,
        parent_side_effect_on_call: int = 0,
        child_scripts=None,
        second_gate: "threading.Event | None" = None,
    ):
        self._parent = list(parent_scripts)
        self._child = list(child_scripts or [["child answer"]])
        self._gate = gate
        self._second_gate = second_gate
        self._child_side_effect = child_side_effect
        self._parent_side_effect = parent_side_effect
        self._parent_side_effect_on_call = parent_side_effect_on_call
        self._lock = threading.Lock()
        self.parent_calls = 0
        self.child_calls = 0
        self.child_started = threading.Event()

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        system = str(messages[0].get("content", "")) if messages else ""
        if system.startswith(SUBAGENT_PROMPT_PREFIX):
            with self._lock:
                self.child_calls += 1
                first_call = self.child_calls == 1
                chunks = (
                    self._child.pop(0) if self._child else ["child answer"]
                )
            loop = asyncio.get_running_loop()
            if first_call:
                self.child_started.set()
                await loop.run_in_executor(None, self._gate.wait)
                if self._child_side_effect is not None:
                    side_effect, self._child_side_effect = (
                        self._child_side_effect,
                        None,
                    )
                    side_effect()
            elif self._second_gate is not None:
                # The child keeps RUNNING after its write -- how a test
                # pins a window that must be closed by the next turn
                # rather than by the child finishing.
                await loop.run_in_executor(None, self._second_gate.wait)
            for chunk in chunks:
                yield chunk
            return
        with self._lock:
            assert self._parent, "parent script exhausted"
            chunks = self._parent.pop(0)
            self.parent_calls += 1
            fire = (
                self._parent_side_effect is not None
                and self.parent_calls == self._parent_side_effect_on_call
            )
        if fire:
            self._parent_side_effect()
        for chunk in chunks:
            yield chunk


def _spawn_fence(task: str) -> str:
    return (
        f"{FENCE_OPEN}\n"
        + json.dumps(
            {"name": "spawn_subagent", "arguments": {"task": task}}
        )
        + "\n```"
    )


def _join_fleet_threads(timeout: float = 5.0) -> None:
    """Block until every live fleet child thread has fully finished.

    Copied from `Tests/Chat/test_console_agent_bridge.py` for the same
    reason it exists there: a child's run row goes terminal slightly
    BEFORE its thread unwinds, so joining the thread -- not polling -- is
    what guarantees any scope wrapping that run has already exited.
    """
    for thread in list(threading.enumerate()):
        if thread.name.startswith("fleet-"):
            thread.join(timeout)


def _next_turn(store, session):
    """Append the next user/assistant pair, as a real second Send does."""
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="again")
    return store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    ).id


def test_a_survivors_write_after_its_turn_lands_in_a_change_record(
    tmp_path, root, tracker
):
    """A child released after its turn returned writes to disk. That write
    must be reviewable SOMEWHERE -- today it is in no record at all.
    """
    gate = threading.Event()
    gateway = _FleetSurvivorGateway(
        parent_scripts=[[_spawn_fence("long job")], ["turn 1 final"]],
        gate=gate,
        child_side_effect=lambda: (root / "survivor.txt").write_text(
            "written after the turn returned\n"
        ),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_id, outcome = _run(bridge, session, aid, root)
        assert outcome.status == "done"
        assert gateway.child_started.wait(5), "the child never started"
        # Nothing has changed on disk yet: the turn's own record is empty,
        # which is correct and is the state the survivor writes into.
        assert db.change_snapshots_for_run(run_id) == []
    finally:
        gate.set()
    _join_fleet_threads()

    assert (root / "survivor.txt").exists(), "precondition: the child wrote"
    rows = db.change_snapshots_for_run(run_id)
    assert len(rows) == 1, (
        "the survivor's write landed in NO change record: "
        f"{rows}"
    )
    row = rows[0]
    assert row["files_changed"] == 1, row
    changed = tracker.service.repo_for_root(root).changed_files(
        row["baseline_sha"], row["end_sha"]
    )
    assert [c.path for c in changed] == ["survivor.txt"], changed
    marker = [
        m
        for m in _tool_rows(store, session)
        if m.content.startswith("✎") and "sub-agent" in m.content
    ]
    assert len(marker) == 1, (
        "nothing in the transcript says a sub-agent changed files after "
        f"the turn: {[m.content for m in _tool_rows(store, session)]}"
    )
    assert marker[0].change_review_run_id == run_id


def test_a_survivors_write_during_the_next_turn_is_disclosed_on_it(
    tmp_path, root, tracker
):
    """The survivor writes strictly AFTER turn 2's baseline settled, so the
    write is inside turn 2's diff -- a record attributing one turn's file
    writes to another. The tracker is a working-tree differ and cannot
    un-mix concurrent writers, so the record must SAY so.
    """
    gate = threading.Event()

    def release_and_join():
        gate.set()
        _join_fleet_threads()

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],  # turn 1, call 1
            ["turn 1 final"],  # turn 1, call 2
            [_calc_fence()],  # turn 2, call 1 -- a tool, so B2 is awaited
            ["turn 2 final"],  # turn 2, call 2 -- fires the release
        ],
        gate=gate,
        child_side_effect=lambda: (root / "survivor.txt").write_text(
            "written during turn 2\n"
        ),
        parent_side_effect=release_and_join,
        parent_side_effect_on_call=4,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_1, outcome_1 = _run(bridge, session, aid, root)
        assert outcome_1.status == "done"
        assert gateway.child_started.wait(5), "the child never started"
        run_2, outcome_2 = _run(bridge, session, _next_turn(store, session), root)
        assert outcome_2.status == "done"
    finally:
        gate.set()
    _join_fleet_threads()

    # Characterisation: the survivor's file IS inside turn 2's diff. That
    # is not fixable (one tree, two writers) -- what follows is.
    rows_2 = db.change_snapshots_for_run(run_2)
    assert len(rows_2) == 1, rows_2
    changed = tracker.service.repo_for_root(root).changed_files(
        rows_2[0]["baseline_sha"], rows_2[0]["end_sha"]
    )
    assert [c.path for c in changed] == ["survivor.txt"], changed

    assert rows_2[0]["kind"] == "turn_concurrent_subagent", (
        "turn 2's record does not record that an earlier turn's sub-agent "
        "was writing during it"
    )
    disclosures = [
        m
        for m in _tool_rows(store, session)
        if "earlier turn" in m.content and "sub-agent" in m.content
    ]
    assert len(disclosures) == 1, (
        "turn 2's changes silently include a sub-agent's: "
        f"{[m.content for m in _tool_rows(store, session)]}"
    )


def test_a_survivors_write_racing_the_next_baseline_is_still_reviewable(
    tmp_path, root
):
    """The audit's second half: a survivor's tool dispatch passes turn 1's
    ALREADY-SATISFIED `await_baseline()`, so it is gated on nothing while
    turn 2's baseline is being taken -- and a write during that window is
    swallowed into B2 and vanishes from turn 2's diff.

    The baseline is made slow so the window is deterministic rather than a
    coin flip, exactly as `test_baseline_completes_before_the_first_tool_
    executes` does.
    """
    events: list[str] = []

    class _SlowService(ShadowRepoService):
        def repo_for_root(self, r):
            repo = super().repo_for_root(r)
            original = repo.snapshot

            def slow_snapshot(message: str) -> str:
                if "baseline" in message:
                    time.sleep(0.6)
                    events.append("baseline-finished")
                return original(message)

            repo.snapshot = slow_snapshot  # type: ignore[method-assign]
            return repo

    tracker = ChangeTurnTracker(service=_SlowService(data_dir=tmp_path / "app"))
    gate = threading.Event()

    def release_into_the_baseline_window():
        gate.set()
        _join_fleet_threads()

    def child_writes():
        # Recorded HERE, at the write itself -- an event appended after
        # joining the child's thread would time the JOIN instead, and the
        # join now waits for the very baseline this test is ordering
        # against.
        (root / "raced.txt").write_text(
            "written while turn 2's baseline was still snapshotting\n"
        )
        events.append("survivor-wrote")

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],  # turn 1, call 1
            ["turn 1 final"],  # turn 1, call 2
            ["turn 2 final"],  # turn 2, call 1 -- fires while B2 runs
        ],
        gate=gate,
        child_side_effect=child_writes,
        parent_side_effect=release_into_the_baseline_window,
        parent_side_effect_on_call=3,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_1, outcome_1 = _run(bridge, session, aid, root)
        assert outcome_1.status == "done"
        assert gateway.child_started.wait(5), "the child never started"
        events.clear()  # drop turn 1's baseline; only turn 2's matters now
        run_2, outcome_2 = _run(bridge, session, _next_turn(store, session), root)
        assert outcome_2.status == "done"
    finally:
        gate.set()
    _join_fleet_threads()

    assert "survivor-wrote" in events and "baseline-finished" in events, events
    assert events.index("survivor-wrote") < events.index("baseline-finished"), (
        "the survivor did NOT write inside turn 2's baseline window, so "
        f"this test proves nothing: {events}"
    )
    # Turn 2's own diff cannot see it -- B2 swallowed it.
    assert db.change_snapshots_for_run(run_2) == [], (
        "expected the raced write to be inside turn 2's baseline"
    )
    rows_1 = db.change_snapshots_for_run(run_1)
    assert len(rows_1) == 1, (
        "a write that raced the next turn's baseline is in NO record: "
        f"{rows_1}"
    )
    changed = tracker.service.repo_for_root(root).changed_files(
        rows_1[0]["baseline_sha"], rows_1[0]["end_sha"]
    )
    assert [c.path for c in changed] == ["raced.txt"], changed


def test_a_survivors_tool_dispatch_is_gated_on_nothing_across_turns(
    tmp_path, root
):
    """CHARACTERISATION (green before and after this task's fix): the
    mechanism behind the test above.

    A survivor's tool batch still calls TURN 1's `before_tool_dispatch`
    gate, whose baseline was satisfied before turn 1 even answered. So the
    survivor dispatches a tool while turn 2's baseline is still being taken
    -- the exact "a tool writing before B settles races its own change into
    the baseline" hazard the gate exists to prevent, now reachable across
    turns and gated by nothing.

    This is NOT fixed by re-gating (a survivor must not block on an
    unrelated turn's snapshot); it is made harmless by the windows sharing
    a boundary sha, which the test above pins.
    """
    events: list[str] = []
    child_run_ids: list[str] = []

    class _SlowService(ShadowRepoService):
        def repo_for_root(self, r):
            repo = super().repo_for_root(r)
            original = repo.snapshot

            def slow_snapshot(message: str) -> str:
                if "baseline" in message:
                    time.sleep(0.6)
                    events.append("baseline-finished")
                return original(message)

            repo.snapshot = slow_snapshot  # type: ignore[method-assign]
            return repo

    tracker = ChangeTurnTracker(service=_SlowService(data_dir=tmp_path / "app"))
    gate = threading.Event()
    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],
            ["turn 1 final"],
            ["turn 2 final"],
        ],
        gate=gate,
        # The child calls a REAL tool once released, so its batch goes
        # through the review hook (i.e. through turn 1's baseline gate).
        child_scripts=[[_calc_fence()], ["child answer"]],
        parent_side_effect=lambda: (gate.set(), _join_fleet_threads()),
        parent_side_effect_on_call=3,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)

    def review(calls, run_id):
        events.append(f"review:{run_id}")
        return {}

    try:
        run_1, outcome_1 = _run(
            bridge, session, aid, root, review_tool_calls=review
        )
        assert outcome_1.status == "done"
        assert gateway.child_started.wait(5), "the child never started"
        child_run_ids.extend(
            row["id"]
            for row in db.list_runs("conv-1")
            if row["agent_kind"] == "subagent"
        )
        assert child_run_ids, "no sub-agent run row"
        events.clear()
        _run(
            bridge,
            session,
            _next_turn(store, session),
            root,
            review_tool_calls=review,
        )
    finally:
        gate.set()
    _join_fleet_threads()

    child_reviews = [
        index
        for index, event in enumerate(events)
        if event == f"review:{child_run_ids[0]}"
    ]
    assert child_reviews, (
        f"the survivor never dispatched a tool after turn 1: {events}"
    )
    baseline_done = events.index("baseline-finished")
    assert child_reviews[0] < baseline_done, (
        "the survivor's tool batch waited for turn 2's baseline (it does "
        f"not, and this test exists to say so out loud): {events}"
    )


def test_the_survivor_window_ends_exactly_where_the_next_turn_begins(
    tmp_path, root, tracker
):
    """The load-bearing invariant of the fix: the two windows ABUT.

    A survivor's window is closed at the NEXT turn's baseline sha rather
    than at a snapshot of its own, so the disk history is partitioned
    (B1..E1, E1..B2, B2..E2) with no crack for a write to fall into and no
    overlap for one to be counted twice in.
    """
    gate = threading.Event()
    keep_running = threading.Event()
    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],
            ["turn 1 final"],
            [_calc_fence()],
            ["turn 2 final"],
        ],
        gate=gate,
        # Write, then keep working: the window must be closed by turn 2,
        # not by the child finishing.
        child_scripts=[[_calc_fence()], ["child answer"]],
        second_gate=keep_running,
        child_side_effect=lambda: (root / "survivor.txt").write_text("a\n"),
        # Turn 2 writes its OWN file, after its baseline settled (its
        # first call was a tool, so the gate has been awaited) -- without
        # a change of its own turn 2 records no row to abut.
        parent_side_effect=lambda: (root / "by_turn_2.txt").write_text("b\n"),
        parent_side_effect_on_call=4,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_1, _ = _run(bridge, session, aid, root)
        assert gateway.child_started.wait(5), "the child never started"
        gate.set()
        deadline = time.monotonic() + 5
        while not (root / "survivor.txt").exists():
            assert time.monotonic() < deadline, "the survivor never wrote"
            time.sleep(0.02)
        run_2, _ = _run(bridge, session, _next_turn(store, session), root)
        # Recorded by the END of turn 2, while the child is still working:
        # the window's content is settled the moment turn 2's baseline
        # exists, so waiting for a survivor that may run for an hour
        # before showing the user anything would be a choice, not a
        # necessity.
        post_turn = [
            r
            for r in db.change_snapshots_for_run(run_1)
            if r["kind"] == "subagent_post_turn"
        ]
        assert post_turn, (
            "the survivor's window was still open after the next turn "
            "ended, so its record waits on a child that need never finish"
        )
    finally:
        gate.set()
        keep_running.set()
    _join_fleet_threads()

    post_turn = [
        r
        for r in db.change_snapshots_for_run(run_1)
        if r["kind"] == "subagent_post_turn"
    ]
    assert len(post_turn) == 1, db.change_snapshots_for_run(run_1)
    assert post_turn[0]["files_changed"] == 1, post_turn
    turn_2_rows = [
        r
        for r in db.change_snapshots_for_conversation("conv-1")
        if r["run_id"] == run_2
    ]
    assert turn_2_rows, "turn 2 recorded no row to abut"
    assert post_turn[0]["end_sha"] == turn_2_rows[0]["baseline_sha"], (
        "the survivor's window and turn 2's window do not share a "
        "boundary, so a write between them belongs to neither"
    )
    # ... and therefore the write is in exactly ONE record.
    changed_2 = tracker.service.repo_for_root(root).changed_files(
        turn_2_rows[0]["baseline_sha"], turn_2_rows[0]["end_sha"]
    )
    assert "survivor.txt" not in [c.path for c in changed_2], (
        "the same write is counted in both windows"
    )
    # The survivor's row must also be emitted in the position resume will
    # re-derive it in -- BEFORE the next turn's own row, since it belongs
    # to the earlier turn's block. (Closing the window as a side effect of
    # opening the next one produces the same record in the wrong place.)
    live = [
        m.content
        for m in _tool_rows(store, session)
        if m.content.startswith("✎")
    ]
    fresh = ConsoleAgentBridge(
        agent_runs_db=db, store=None, provider_gateway=None
    )
    resumed = [
        m.content
        for _anchor, block in fresh.resume_marker_messages("conv-1")
        for m in block
        if m.content.startswith("✎")
    ]
    assert resumed == live, (
        f"live and resumed transcripts disagree: {live} vs {resumed}"
    )


def test_a_child_that_finishes_inside_its_turn_opens_no_survivor_window(
    tmp_path, root, tracker
):
    """Negative control: the post-turn window exists for survivors, not
    for every fleet turn. A child collected before the turn answers is
    fully inside the turn's own window and must add nothing."""
    gate = threading.Event()
    gate.set()  # the child never blocks
    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("quick job")],
            [_calc_fence()],
            ["turn 1 final"],
        ],
        gate=gate,
        child_side_effect=lambda: (root / "by_the_child.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    run_id, outcome = _run(bridge, session, aid, root)
    _join_fleet_threads()
    assert outcome.status == "done"

    rows = db.change_snapshots_for_run(run_id)
    assert [r["kind"] for r in rows] == ["turn"], rows
    assert not [
        m
        for m in _tool_rows(store, session)
        if "after this turn" in m.content
    ], "a survivor row was emitted for a child that never survived"


def test_a_turn_without_a_foreign_survivor_is_not_stamped_concurrent(
    tmp_path, root, tracker
):
    """Negative control for the disclosure: a turn whose own child is the
    only sub-agent must NOT claim someone else's writes may be in it."""
    gate = threading.Event()
    gate.set()
    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("quick job")],
            [_calc_fence()],
            ["turn 1 final"],
        ],
        gate=gate,
        child_side_effect=lambda: (root / "by_the_child.txt").write_text("x\n"),
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    run_id, _ = _run(bridge, session, aid, root)
    _join_fleet_threads()

    assert [r["kind"] for r in db.change_snapshots_for_run(run_id)] == ["turn"]
    assert not [
        m for m in _tool_rows(store, session) if "earlier turn" in m.content
    ]


def test_resume_re_derives_the_survivor_rows_byte_identical(
    tmp_path, root, tracker
):
    """Both new transcript rows are re-derived from the stored `kind`.

    Without the column a post-turn row and a turn row are indistinguishable
    and resume would collapse them into ONE summary showing a turn that
    never happened -- the same parity rule TASK-1972 set for the summary
    and failure rows.
    """
    gate = threading.Event()

    def release_and_join():
        gate.set()
        _join_fleet_threads()

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],
            ["turn 1 final"],
            [_calc_fence()],
            ["turn 2 final"],
        ],
        gate=gate,
        child_side_effect=lambda: (root / "survivor.txt").write_text("a\n"),
        parent_side_effect=release_and_join,
        parent_side_effect_on_call=4,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        run_1, _ = _run(bridge, session, aid, root)
        assert gateway.child_started.wait(5), "the child never started"
        # Turn 2 both closes turn 1's window (at its baseline) AND absorbs
        # the survivor's write, so this run exercises both new rows.
        (root / "by_turn_1.txt").write_text("edited before turn 2\n")
        run_2, _ = _run(bridge, session, _next_turn(store, session), root)
    finally:
        gate.set()
    _join_fleet_threads()

    live = [
        m.content
        for m in _tool_rows(store, session)
        if m.content.startswith("✎") or "earlier turn" in m.content
    ]
    assert len(live) >= 2, live

    fresh = ConsoleAgentBridge(
        agent_runs_db=db, store=None, provider_gateway=None
    )
    resumed = [
        m.content
        for _anchor, block in fresh.resume_marker_messages("conv-1")
        for m in block
        if m.content.startswith("✎") or "earlier turn" in m.content
    ]
    assert resumed == live, (
        "the survivor rows did not survive resume byte-identical"
    )


def test_opening_a_window_whose_last_child_already_left_closes_it_at_once(
    tmp_path, root, tracker
):
    """The sliver the open path exists to cover, probed directly.

    A child can finish between the turn's E snapshot and the moment the
    window is installed. Its final writes are then after E and before the
    window exists, and nothing else will ever close that window -- the
    last-child signal has already fired. Opening therefore re-checks the
    live count and closes immediately.

    Driven through the bridge's own methods rather than a scenario,
    because the race is microseconds wide and no scripted run can land in
    it reliably (see PR3a-1 Task 6b's P4: a branch no test can kill should
    be probed directly, not shipped untested).
    """
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=None,
        change_tracker=tracker,
    )
    run_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    tracker.end_turn(handle)  # the turn's own E; nothing changed
    (root / "written_in_the_sliver.txt").write_text("x\n")

    bridge._open_post_turn_change_window(
        "conv-1", run_id=run_id, session_id=session.id, handle=handle
    )

    assert bridge._post_turn_change_windows.get("conv-1") is None, (
        "a window nobody will ever close was left open"
    )
    rows = db.change_snapshots_for_run(run_id)
    assert [r["kind"] for r in rows] == ["subagent_post_turn"], rows
    assert rows[0]["files_changed"] == 1, rows


def test_a_survivor_finishing_mid_turn_is_counted_in_exactly_one_window(
    tmp_path, root, tracker
):
    """The other half of the abutment rule: a survivor that finishes
    DURING the next turn must not have its write counted twice.

    Its window cannot end at a snapshot of its own once the next turn's
    baseline exists -- that would overlap the turn's window, and the same
    file would appear on two cards as if it had been written twice.
    """
    gate = threading.Event()

    def release_and_join():
        gate.set()
        _join_fleet_threads()

    gateway = _FleetSurvivorGateway(
        parent_scripts=[
            [_spawn_fence("long job")],
            ["turn 1 final"],
            [_calc_fence()],  # turn 2 awaits B2 here
            ["turn 2 final"],  # ... then the survivor is released
        ],
        gate=gate,
        child_side_effect=lambda: (root / "survivor.txt").write_text("a\n"),
        parent_side_effect=release_and_join,
        parent_side_effect_on_call=4,
    )
    bridge, db, store, session, aid = _bridge_with(tmp_path, gateway, tracker)
    try:
        _run(bridge, session, aid, root)
        assert gateway.child_started.wait(5), "the child never started"
        _run(bridge, session, _next_turn(store, session), root)
    finally:
        gate.set()
    _join_fleet_threads()

    holding = [
        row
        for row in db.change_snapshots_for_conversation("conv-1")
        if "survivor.txt"
        in [
            c.path
            for c in tracker.service.repo_for_root(root).changed_files(
                row["baseline_sha"], row["end_sha"]
            )
        ]
    ]
    assert len(holding) == 1, (
        "the survivor's single write is on "
        f"{len(holding)} change records: {[r['kind'] for r in holding]}"
    )
