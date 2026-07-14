"""Agent rail section + [N Sub-Agents] badge render via a real App (run_test)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_agent_bridge import (
    AgentLiveSnapshot, AgentLiveStep, ConsoleAgentBridge, SubAgentSummary,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow, build_console_conversation_browser_state,
)


def _all_rows(state):
    rows = []
    for section in state.sections:
        rows.extend(section.rows)
        for group in section.groups:
            rows.extend(group.rows)
    return rows


def test_conversation_row_carries_badge_count_for_render():
    row = ConsoleConversationBrowserInputRow(
        row_key="c1", conversation_id="c1", native_session_id=None, title="Alpha",
        scope_type="global", workspace_id=None, workspace_label="",
        updated_sort="2026-07-13T00:00:00Z")
    state = build_console_conversation_browser_state(
        rows=[row], active_workspace_id=None, subagent_counts={"c1": 2})
    assert _all_rows(state)[0].subagent_count == 2


def test_conversation_row_badge_label_is_escaped_and_present():
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        format_console_conversation_row_label,
    )
    label = format_console_conversation_row_label("Beta [x]", subagent_count=3)
    assert "3 Sub-Agents" in label
    assert "\\[x]" in label or "[x]" not in label.replace("Sub-Agents", "")


def test_no_badge_when_subagent_count_is_zero():
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        format_console_conversation_row_label,
    )
    label = format_console_conversation_row_label("Beta", subagent_count=0)
    assert "Sub-Agents" not in label
    assert label == "Beta"


@pytest.mark.asyncio
async def test_agent_rail_section_mounts():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        # Navigating to Console mounts the rail; the Agent header exists.
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        assert console.query_one("#console-rail-section-header-agent")
        assert console.query_one("#console-agent-section-status")
        assert console.query_one("#console-agent-section-steps")
        assert console.query_one("#console-agent-section-subagents")


def test_resume_rederives_subagent_data_from_durable_run_store(tmp_path):
    """A fresh bridge over the same durable AgentRunsDB file reproduces the
    same badge count + sub-agent listing after a "restart" -- the run store,
    not any in-memory/session state, is the source of truth for resume."""
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db_path = tmp_path / "agent_runs.db"
    db = AgentRunsDB(db_path, client_id="t")
    primary_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    sub_id = db.create_run(
        conversation_id="conv-1", agent_kind="subagent",
        task="research pricing", parent_run_id=primary_id,
    )
    db.set_status(sub_id, "done", result="done researching")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    assert bridge.subagent_count("conv-1") == 1

    # Simulate resume: a brand-new bridge/DB handle over the same file, with
    # no in-memory live-snapshot state carried over.
    fresh_db = AgentRunsDB(db_path, client_id="t")
    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=fresh_db, store=None, provider_gateway=None)

    assert fresh_bridge.subagent_count("conv-1") == 1
    runs = fresh_bridge.subagent_runs("conv-1")
    assert len(runs) == 1
    assert runs[0]["id"] == sub_id
    assert runs[0]["status"] == "done"
    record = fresh_bridge.subagent_run(sub_id)
    assert record is not None
    assert record["task"] == "research pricing"
    # No live-run activity was replayed -- the rail's "running" state does
    # not leak across a resume.
    assert fresh_bridge.live_snapshot("conv-1").status == "idle"


# --- Finding A: batched sub-agent badge counts (one DB query, not one per
# conversation row), gated so the poll tick doesn't refresh unconditionally. ---

def test_bridge_subagent_counts_batches_in_one_db_call(tmp_path, monkeypatch):
    db = AgentRunsDB(tmp_path / "agent_runs.db", client_id="t")
    parent = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.create_run(conversation_id="conv-1", agent_kind="subagent",
                  task="x", parent_run_id=parent)
    db.create_run(conversation_id="conv-2", agent_kind="primary")

    calls = []
    original = db.count_subagents_by_conversation

    def spy(ids):
        calls.append(list(ids))
        return original(ids)

    monkeypatch.setattr(db, "count_subagents_by_conversation", spy)
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)

    counts = bridge.subagent_counts(["conv-1", "conv-2", "conv-3"])

    assert counts == {"conv-1": 1}   # conv-2/conv-3 absent -- zero sub-agents
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_subagent_counts_are_batched_and_gated_not_refreshed_every_tick(monkeypatch):
    """Finding A: the screen's badge-count refresh issues one batched DB
    call per row set (not N calls, one per row), and skips re-querying
    when neither the row set, an active run, nor cache age justify it."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        calls = []

        class _FakeBridge:
            def subagent_counts(self, conversation_ids):
                calls.append(list(conversation_ids))
                return {cid: 2 for cid in conversation_ids}

        bridge = _FakeBridge()
        rows = tuple(
            ConsoleConversationBrowserInputRow(
                row_key=f"c{i}", conversation_id=f"c{i}", native_session_id=None,
                title=f"Conv {i}", scope_type="global", workspace_id=None,
                workspace_label="", updated_sort="2026-07-13T00:00:00Z")
            for i in range(5)
        )

        fake_time = {"t": 0.0}
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.chat_screen.time.monotonic",
            lambda: fake_time["t"])

        counts = console._console_subagent_counts_for_rows(bridge, rows)
        assert counts == {f"c{i}": 2 for i in range(5)}
        assert len(calls) == 1   # one batched call for 5 rows, not 5 calls

        # Same row set, no active run, within TTL: cache reused verbatim --
        # this is the "0.2s poll tick with nothing sub-agent related
        # changed" case that previously re-queried every tick.
        console._console_subagent_counts_for_rows(bridge, rows)
        assert len(calls) == 1

        # The visible row set changes (conversation list rebuilt) -> refresh.
        console._console_subagent_counts_for_rows(bridge, rows[:2])
        assert len(calls) == 2

        # Same (smaller) row set again, still within TTL, no active run:
        # cached, no extra call.
        console._console_subagent_counts_for_rows(bridge, rows[:2])
        assert len(calls) == 2

        # Cache TTL elapses -> refresh even though nothing else changed.
        fake_time["t"] += 5.0
        console._console_subagent_counts_for_rows(bridge, rows[:2])
        assert len(calls) == 3

        # An active run forces a refresh even inside the TTL window, so a
        # just-spawned sub-agent's count shows up promptly.
        original_controller = console._console_chat_controller
        try:
            console._console_chat_controller = SimpleNamespace(
                run_state=SimpleNamespace(status=ConsoleRunStatus.STREAMING))
            console._console_subagent_counts_for_rows(bridge, rows[:2])
            assert len(calls) == 4
        finally:
            console._console_chat_controller = original_controller


# --- Finding B: rail Agent-section lines render into markup=False Statics
# -- escaping bracket text there produces literal backslashes, not markup
# protection, so this text must stay raw. ---

def test_summarize_step_does_not_escape_markup_brackets():
    from tldw_chatbook.Agents.agent_models import STEP_TOOL_RESULT, AgentStep

    step = AgentStep(index=0, kind=STEP_TOOL_RESULT, tool_name="fetch_docs",
                      result="fetch [docs] ok")
    text = ConsoleAgentBridge._summarize(step)
    assert text == "fetch [docs] ok"
    assert "\\[" not in text


def test_spawn_subagent_summary_does_not_escape_markup_brackets(tmp_path):
    import json

    from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN

    def _fence(name, args):
        return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'

    scripts = [
        [_fence("spawn_subagent", {"task": "fetch [docs] and summarize"})],
        ["done."],
        ["Done: 1."],
    ]

    class _ChunkGateway:
        def __init__(self, scripts):
            self._scripts = list(scripts)
            self.calls = 0

        async def stream_chat(self, resolution, messages):
            chunks = self._scripts[self.calls]
            self.calls += 1
            for chunk in chunks:
                yield chunk

    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts))
    bridge.run_reply(
        conversation_id="conv-1", session_id=session.id, resolution=object(),
        assistant_message_id=assistant.id, model="test-model",
        session_system_prompt="", agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False)

    snap = bridge.live_snapshot("conv-1")
    assert snap.subagents, "expected a recorded sub-agent summary"
    subagent_text = snap.subagents[0].text
    assert "fetch [docs] and summarize" in subagent_text
    assert "\\[" not in subagent_text


@pytest.mark.asyncio
async def test_agent_section_lines_render_brackets_literally_not_escaped():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        class _FakeBridge:
            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot(
                    status="running", step=1,
                    steps=(AgentLiveStep("tool_result", "fetch [docs] ok", "primary"),),
                    subagents=(SubAgentSummary("spawn [docs] task"),),
                )

            def subagent_run(self, run_id):
                return None

            def subagent_runs(self, conversation_id):
                return []

        console._console_agent_bridge = _FakeBridge()
        console._console_agent_drilldown_run_id = None
        status_line, steps_text, subagents_text = console._console_agent_section_lines()

        assert "fetch [docs] ok" in steps_text
        assert "\\[" not in steps_text
        assert "spawn [docs] task" in subagents_text
        assert "\\[" not in subagents_text


# --- Gate Finding 2: the top-level Agent summary must re-derive from
# AgentRunsDB (via bridge.historical_snapshot) when live_snapshot is idle
# (e.g. right after an app restart), instead of showing "Agent: idle"
# forever until the next live run in this process. ---


@pytest.mark.asyncio
async def test_agent_section_falls_back_to_historical_snapshot_when_live_is_idle():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        class _FakeBridge:
            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot()   # idle -- simulates a fresh process

            def historical_snapshot(self, conversation_id):
                return AgentLiveSnapshot(
                    status="done", step=1,
                    steps=(AgentLiveStep("model", "The capital of France is Paris.",
                                          "primary"),),
                    subagents=(SubAgentSummary("research pricing", status="done"),),
                )

            def subagent_run(self, run_id):
                return None

            def subagent_runs(self, conversation_id):
                return []

        console._console_agent_bridge = _FakeBridge()
        console._console_agent_drilldown_run_id = None
        status_line, steps_text, subagents_text = console._console_agent_section_lines()

        assert status_line == "Agent: done"
        assert "Paris" in steps_text
        assert "research pricing" in subagents_text


@pytest.mark.asyncio
async def test_agent_section_prefers_live_snapshot_over_historical_when_present():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        calls = []

        class _FakeBridge:
            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot(status="running", step=2)

            def historical_snapshot(self, conversation_id):
                calls.append(conversation_id)
                return AgentLiveSnapshot(status="done")

            def subagent_run(self, run_id):
                return None

            def subagent_runs(self, conversation_id):
                return []

        console._console_agent_bridge = _FakeBridge()
        console._console_agent_drilldown_run_id = None
        status_line, _steps, _subagents = console._console_agent_section_lines()

        assert status_line == "Agent: running · step 2"
        assert calls == []   # historical_snapshot must not even be consulted


def test_resume_rederives_top_level_agent_summary_from_durable_run_store(tmp_path):
    """Full-stack (real ConsoleAgentBridge + real AgentRunsDB) version of the
    same gate finding: a fresh bridge over a durable DB with a completed
    primary+subagent run reports that history via historical_snapshot, not
    the idle default -- matching the badge/drill-in's existing durability."""
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db_path = tmp_path / "agent_runs.db"
    db = AgentRunsDB(db_path, client_id="t")
    primary_id = db.create_run(conversation_id="conv-1", agent_kind="primary")
    db.append_steps(primary_id, [
        {"index": 0, "kind": "model", "summary": "final answer",
         "tool_name": "", "args": None, "result": "", "created_at": ""},
    ])
    db.set_status(primary_id, "done", result="final answer")
    sub_id = db.create_run(
        conversation_id="conv-1", agent_kind="subagent",
        task="research pricing", parent_run_id=primary_id)
    db.set_status(sub_id, "done", result="done researching")

    # Simulate resume: a brand-new bridge/DB handle over the same file.
    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(db_path, client_id="t"), store=None,
        provider_gateway=None)

    assert fresh_bridge.live_snapshot("conv-1").status == "idle"
    historical = fresh_bridge.historical_snapshot("conv-1")
    assert historical.status == "done"
    assert historical.subagents and historical.subagents[0].text == "research pricing"


# --- Finding C: a sub-agent drill-in is scoped to the conversation active
# when the user drilled in -- switching conversations must drop back to
# the overview instead of showing a foreign conversation's sub-agent. ---

@pytest.mark.asyncio
async def test_drilldown_falls_back_to_overview_after_conversation_switch():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        class _FakeBridge:
            def __init__(self):
                self.active_conversation_id = "conv-A"

            def subagent_runs(self, conversation_id):
                return [{"id": "run-1", "conversation_id": "conv-A",
                          "status": "done", "task": "t", "steps": []}]

            def subagent_run(self, run_id):
                if run_id == "run-1":
                    return {"id": "run-1", "conversation_id": "conv-A",
                            "status": "done", "task": "t", "steps": []}
                return None

            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot()

        fake_bridge = _FakeBridge()
        console._console_agent_bridge = fake_bridge
        console._current_console_rail_conversation_id = (
            lambda: fake_bridge.active_conversation_id)

        # Drill into the (only) sub-agent run of conv-A.
        console._toggle_console_agent_drilldown_from_subagents_click()
        assert console._console_agent_drilldown_run_id == "run-1"
        status_line, _steps, _subagents = console._console_agent_section_lines()
        assert status_line.startswith("Sub-agent ·")

        # Switch to a different conversation -- the drill-in must not
        # survive, even though bridge.subagent_run("run-1") would still
        # happily return the (now-foreign) record.
        fake_bridge.active_conversation_id = "conv-B"
        status_line, _steps, _subagents = console._console_agent_section_lines()
        assert console._console_agent_drilldown_run_id is None
        assert not status_line.startswith("Sub-agent ·")
        assert status_line.startswith("Agent:")


@pytest.mark.asyncio
async def test_drilldown_render_path_rejects_record_from_other_conversation():
    """Even if the drill-down id itself weren't cleared on switch, the
    render path independently verifies the record's own conversation_id
    before showing it -- a second, defensive guard."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        class _FakeBridge:
            def subagent_run(self, run_id):
                return {"id": run_id, "conversation_id": "conv-other",
                        "status": "done", "task": "t", "steps": []}

            def subagent_runs(self, conversation_id):
                return []

            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot()

        console._console_agent_bridge = _FakeBridge()
        console._current_console_rail_conversation_id = lambda: "conv-active"
        # Bypass the click-handler's own conversation tracking to isolate
        # the render path's independent conversation_id verification.
        console._console_agent_drilldown_run_id = "run-x"
        console._console_agent_drilldown_conversation_id = "conv-active"

        status_line, _steps, _subagents = console._console_agent_section_lines()
        assert console._console_agent_drilldown_run_id is None
        assert not status_line.startswith("Sub-agent ·")


@pytest.mark.asyncio
async def test_activate_native_console_session_clears_stale_drilldown():
    """The shared session-activation path (tab click / Ctrl+K / Alt+1..9)
    clears the drill-down immediately on switch, not just on the next
    rail render."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()
        # ``create_session`` immediately activates the new session, so
        # create a second session (now active) then switch back to the
        # first -- otherwise active_session_id would already equal the
        # target and the switch branch (where the clear lives) would
        # never run.
        first_session = store.ensure_session(title="First")
        store.create_session(title="Other")
        console._console_agent_drilldown_run_id = "run-1"

        await console._activate_native_console_session(first_session.id)

        assert console._console_agent_drilldown_run_id is None


# --- Finding D: repeated clicks on the combined sub-agents rail line must
# reach every sub-agent run, not just the newest one. ---

@pytest.mark.asyncio
async def test_subagents_click_cycles_through_every_run_then_overview():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        class _FakeBridge:
            _RUNS = [
                {"id": "run-newest", "conversation_id": "conv-A", "status": "done",
                 "task": "t", "steps": []},
                {"id": "run-mid", "conversation_id": "conv-A", "status": "done",
                 "task": "t", "steps": []},
                {"id": "run-oldest", "conversation_id": "conv-A", "status": "done",
                 "task": "t", "steps": []},
            ]

            def subagent_runs(self, conversation_id):
                return list(self._RUNS)

            def subagent_run(self, run_id):
                return next((r for r in self._RUNS if r["id"] == run_id), None)

            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot()

        console._console_agent_bridge = _FakeBridge()
        console._current_console_rail_conversation_id = lambda: "conv-A"

        clicked_sequence = []
        for _ in range(5):
            console._toggle_console_agent_drilldown_from_subagents_click()
            clicked_sequence.append(console._console_agent_drilldown_run_id)
            await pilot.pause()   # drain the background rail-sync worker

        # newest -> mid -> oldest -> overview -> newest again (wraps).
        assert clicked_sequence == [
            "run-newest", "run-mid", "run-oldest", None, "run-newest",
        ]
