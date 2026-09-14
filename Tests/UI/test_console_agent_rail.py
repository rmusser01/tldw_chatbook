"""Agent rail section + [N Sub-Agents] badge render via a real App (run_test)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_agent_bridge import (
    AgentLiveSnapshot,
    AgentLiveStep,
    ConsoleAgentBridge,
    SubAgentSummary,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Agents.fleet_coordinator import FleetHandle
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Widgets.Console.console_run_log_modal import ConsoleRunLogModal
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
    build_console_conversation_browser_state,
)


def _select_test_log_run(controller, target):
    """Give focused log lifecycle tests a UI selection and matching metadata seam."""
    controller._capture_run_log_selection = lambda bridge: ("test-conversation", target(), (None, None))
    controller._console_agent_bridge.resolve_run_log_target = lambda conversation, drill: drill


def _all_rows(state):
    rows = []
    for section in state.sections:
        rows.extend(section.rows)
        for group in section.groups:
            rows.extend(group.rows)
    return rows


def test_conversation_row_carries_badge_count_for_render():
    row = ConsoleConversationBrowserInputRow(
        row_key="c1",
        conversation_id="c1",
        native_session_id=None,
        title="Alpha",
        scope_type="global",
        workspace_id=None,
        workspace_label="",
        updated_sort="2026-07-13T00:00:00Z",
    )
    state = build_console_conversation_browser_state(
        rows=[row], active_workspace_id=None, subagent_counts={"c1": 2}
    )
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


def test_badge_renders_on_its_own_line_regardless_of_secondary_length():
    """Regression for task-226: the conversation-row badge used to share the
    row's *last existing line* with the unbounded secondary-detail text
    (workspace label - status - age). A long secondary line pushed the
    trailing `[N Sub-Agents]` badge past the rail's rendered width, clipping
    it to a bare `[1` (per the agent-runtime live-gate capture). The badge
    must render on its own dedicated line so its visibility never depends on
    how long the title or secondary line happen to be."""
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        format_console_conversation_row_label,
    )

    long_secondary = (
        "a-really-long-workspace-label-that-keeps-going - saved chat - 3h ago"
    )
    composed = f"  Some Title\n  {long_secondary}"
    label = format_console_conversation_row_label(composed, subagent_count=1)
    lines = label.splitlines()
    # The full badge text is present, unclipped, and is the ENTIRE final
    # line -- not sharing that line with any of the long secondary text.
    assert lines[-1] == "[dim]\\[1 Sub-Agents][/dim]"
    assert long_secondary not in lines[-1]
    assert "Sub-Agents" not in "\n".join(lines[:-1])


def test_wrapped_title_still_pairs_with_full_badge():
    """Long titles now wrap to two budget-width lines; that wrapping must
    never interact with -- or swallow -- the badge, which lives on an
    entirely separate line."""
    from rich.cells import cell_len

    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        format_console_conversation_row_label,
        wrap_console_conversation_title,
    )

    name_lines = wrap_console_conversation_title("A" * 50, 20)
    assert name_lines == ("A" * 20, "A" * 19 + "…")
    assert all(cell_len(line) <= 20 for line in name_lines)

    composed = "\n".join((*name_lines, "saved chat - 2m"))
    label = format_console_conversation_row_label(composed, subagent_count=5)
    lines = label.splitlines()
    assert lines[0] == name_lines[0]
    assert lines[-1] == "[dim]\\[5 Sub-Agents][/dim]"
    assert "Sub-Agents" not in "\n".join(lines[:-1])


def test_short_title_without_badge_is_unchanged():
    """Badge-less rows and short titles get no extra lines or ellipsis."""
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        format_console_conversation_row_label,
        wrap_console_conversation_title,
    )

    assert wrap_console_conversation_title("Short title", 20) == ("Short title",)

    composed = "Short title\nsaved chat - 2m"
    label = format_console_conversation_row_label(composed, subagent_count=0)
    assert label == composed
    assert label.count("\n") == 1


def test_conversation_row_height_tracks_name_lines_and_badge():
    """Row height = name lines + metadata line, plus one line only when a
    badge will actually render."""
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        ConsoleWorkspaceContextTray,
    )

    badge_button = ConsoleWorkspaceContextTray._conversation_button(
        "Title\nsecondary",
        id="row-badge",
        conversation_id="c1",
        subagent_count=2,
    )
    plain_button = ConsoleWorkspaceContextTray._conversation_button(
        "Title\nsecondary",
        id="row-plain",
        conversation_id="c2",
        subagent_count=0,
    )
    wrapped_button = ConsoleWorkspaceContextTray._conversation_button(
        "Title line one\nline two\nsecondary",
        id="row-wrapped",
        conversation_id="c3",
        subagent_count=0,
        name_line_count=2,
    )
    wrapped_badge_button = ConsoleWorkspaceContextTray._conversation_button(
        "Title line one\nline two\nsecondary",
        id="row-wrapped-badge",
        conversation_id="c4",
        subagent_count=1,
        name_line_count=2,
    )
    assert int(badge_button.styles.height.value) == 3
    assert int(plain_button.styles.height.value) == 2
    assert int(wrapped_button.styles.height.value) == 3
    assert int(wrapped_badge_button.styles.height.value) == 4


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
        conversation_id="conv-1",
        agent_kind="subagent",
        task="research pricing",
        parent_run_id=primary_id,
    )
    db.set_status(sub_id, "done", result="done researching")

    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    assert bridge.subagent_count("conv-1") == 1

    # Simulate resume: a brand-new bridge/DB handle over the same file, with
    # no in-memory live-snapshot state carried over.
    fresh_db = AgentRunsDB(db_path, client_id="t")
    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=fresh_db, store=None, provider_gateway=None
    )

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
    db.create_run(
        conversation_id="conv-1", agent_kind="subagent", task="x", parent_run_id=parent
    )
    db.create_run(conversation_id="conv-2", agent_kind="primary")

    calls = []
    original = db.count_subagents_by_conversation

    def spy(ids):
        calls.append(list(ids))
        return original(ids)

    monkeypatch.setattr(db, "count_subagents_by_conversation", spy)
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)

    counts = bridge.subagent_counts(["conv-1", "conv-2", "conv-3"])

    assert counts == {"conv-1": 1}  # conv-2/conv-3 absent -- zero sub-agents
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_subagent_counts_are_batched_and_gated_not_refreshed_every_tick(
    monkeypatch,
):
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
                row_key=f"c{i}",
                conversation_id=f"c{i}",
                native_session_id=None,
                title=f"Conv {i}",
                scope_type="global",
                workspace_id=None,
                workspace_label="",
                updated_sort="2026-07-13T00:00:00Z",
            )
            for i in range(5)
        )

        fake_time = {"t": 0.0}
        monkeypatch.setattr(
            "tldw_chatbook.UI.Screens.chat_screen.time.monotonic",
            lambda: fake_time["t"],
        )

        counts = console._agent._console_subagent_counts_for_rows(bridge, rows)
        assert counts == {f"c{i}": 2 for i in range(5)}
        assert len(calls) == 1  # one batched call for 5 rows, not 5 calls

        # Same row set, no active run, within TTL: cache reused verbatim --
        # this is the "0.2s poll tick with nothing sub-agent related
        # changed" case that previously re-queried every tick.
        console._agent._console_subagent_counts_for_rows(bridge, rows)
        assert len(calls) == 1

        # The visible row set changes (conversation list rebuilt) -> refresh.
        console._agent._console_subagent_counts_for_rows(bridge, rows[:2])
        assert len(calls) == 2

        # Same (smaller) row set again, still within TTL, no active run:
        # cached, no extra call.
        console._agent._console_subagent_counts_for_rows(bridge, rows[:2])
        assert len(calls) == 2

        # Cache TTL elapses -> refresh even though nothing else changed.
        fake_time["t"] += 5.0
        console._agent._console_subagent_counts_for_rows(bridge, rows[:2])
        assert len(calls) == 3

        # An active run forces a refresh even inside the TTL window, so a
        # just-spawned sub-agent's count shows up promptly.
        original_controller = console._console_chat_controller
        try:
            console._console_chat_controller = SimpleNamespace(
                run_state=SimpleNamespace(status=ConsoleRunStatus.STREAMING)
            )
            console._agent._console_subagent_counts_for_rows(bridge, rows[:2])
            assert len(calls) == 4
        finally:
            console._console_chat_controller = original_controller


# --- Finding B: rail Agent-section lines render into markup=False Statics
# -- escaping bracket text there produces literal backslashes, not markup
# protection, so this text must stay raw. ---


def test_summarize_step_does_not_escape_markup_brackets():
    from tldw_chatbook.Agents.agent_models import STEP_TOOL_RESULT, AgentStep

    step = AgentStep(
        index=0, kind=STEP_TOOL_RESULT, tool_name="fetch_docs", result="fetch [docs] ok"
    )
    text = ConsoleAgentBridge._summarize(step)
    assert text == "fetch [docs] ok"
    assert "\\[" not in text


def test_spawn_subagent_summary_does_not_escape_markup_brackets(tmp_path):
    import json

    from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN

    def _fence(name, args):
        return f"{FENCE_OPEN}\n{json.dumps({'name': name, 'arguments': args})}\n```"

    scripts = [
        [_fence("spawn_subagent", {"task": "fetch [docs] and summarize"})],
        ["done."],
        ["Done: 1."],
    ]

    class _ChunkGateway:
        def __init__(self, scripts):
            self._scripts = list(scripts)
            self.calls = 0

        async def stream_chat(self, resolution, messages, **_kwargs):
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
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts)
    )
    bridge.run_reply(
        conversation_id="conv-1",
        session_id=session.id,
        resolution=ConsoleProviderResolution(
            provider="Groq",
            base_url="",
            model="test-model",
            ready=True,
            execution_key="groq",
        ),
        assistant_message_id=assistant.id,
        model="test-model",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
    )

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
                    status="running",
                    step=1,
                    steps=(AgentLiveStep("tool_result", "fetch [docs] ok", "primary"),),
                    subagents=(SubAgentSummary("spawn [docs] task"),),
                )

            def subagent_run(self, run_id):
                return None

            def subagent_runs(self, conversation_id):
                return []

        console._console_agent_bridge = _FakeBridge()
        console._console_agent_drilldown_run_id = None
        status_line, steps_text, subagents_text = (
            console._agent._console_agent_section_lines()
        )

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
                return AgentLiveSnapshot()  # idle -- simulates a fresh process

            def historical_snapshot(self, conversation_id):
                return AgentLiveSnapshot(
                    status="done",
                    step=1,
                    steps=(
                        AgentLiveStep(
                            "model", "The capital of France is Paris.", "primary"
                        ),
                    ),
                    subagents=(SubAgentSummary("research pricing", status="done"),),
                )

            def subagent_run(self, run_id):
                return None

            def subagent_runs(self, conversation_id):
                return []

        console._console_agent_bridge = _FakeBridge()
        console._console_agent_drilldown_run_id = None
        status_line, steps_text, subagents_text = (
            console._agent._console_agent_section_lines()
        )

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
        status_line, _steps, _subagents = console._agent._console_agent_section_lines()

        assert status_line == "Agent: running · step 2"
        assert calls == []  # historical_snapshot must not even be consulted


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
    db.append_steps(
        primary_id,
        [
            {
                "index": 0,
                "kind": "model",
                "summary": "final answer",
                "tool_name": "",
                "args": None,
                "result": "",
                "created_at": "",
            },
        ],
    )
    db.set_status(primary_id, "done", result="final answer")
    sub_id = db.create_run(
        conversation_id="conv-1",
        agent_kind="subagent",
        task="research pricing",
        parent_run_id=primary_id,
    )
    db.set_status(sub_id, "done", result="done researching")

    # Simulate resume: a brand-new bridge/DB handle over the same file.
    fresh_bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(db_path, client_id="t"),
        store=None,
        provider_gateway=None,
    )

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
                return [
                    {
                        "id": "run-1",
                        "conversation_id": "conv-A",
                        "status": "done",
                        "task": "t",
                        "steps": [],
                    }
                ]

            def subagent_run(self, run_id):
                if run_id == "run-1":
                    return {
                        "id": "run-1",
                        "conversation_id": "conv-A",
                        "status": "done",
                        "task": "t",
                        "steps": [],
                    }
                return None

            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot()

        fake_bridge = _FakeBridge()
        console._console_agent_bridge = fake_bridge
        console._character._current_console_rail_conversation_id = lambda: (
            fake_bridge.active_conversation_id
        )

        # Drill into the (only) sub-agent run of conv-A. TASK-4: the old
        # cycling toggle is gone -- a row now resolves directly to its own
        # run id (here that id is already known, matching a real click on
        # that row).
        console._agent._drill_into_console_agent_subagent("run-1")
        assert console._console_agent_drilldown_run_id == "run-1"
        status_line, _steps, _subagents = console._agent._console_agent_section_lines()
        assert status_line.startswith("Sub-agent ·")

        # Switch to a different conversation -- the drill-in must not
        # survive, even though bridge.subagent_run("run-1") would still
        # happily return the (now-foreign) record.
        fake_bridge.active_conversation_id = "conv-B"
        status_line, _steps, _subagents = console._agent._console_agent_section_lines()
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
                return {
                    "id": run_id,
                    "conversation_id": "conv-other",
                    "status": "done",
                    "task": "t",
                    "steps": [],
                }

            def subagent_runs(self, conversation_id):
                return []

            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot()

        console._console_agent_bridge = _FakeBridge()
        console._character._current_console_rail_conversation_id = lambda: "conv-active"
        # Bypass the click-handler's own conversation tracking to isolate
        # the render path's independent conversation_id verification.
        console._console_agent_drilldown_run_id = "run-x"
        console._agent._console_agent_drilldown_conversation_id = "conv-active"

        status_line, _steps, _subagents = console._agent._console_agent_section_lines()
        assert console._console_agent_drilldown_run_id is None
        assert not status_line.startswith("Sub-agent ·")


# -- Review finding A: the rail must not re-slice an already-capped step
# summary down to a hardcoded 80 characters -- that silently overrides any
# configured value above 80 with no visible effect, defeating TASK-870's
# whole point. Covers both render paths: the live/historical overview
# (`snapshot.steps`) and the drilled-in sub-agent path (`record["steps"]`,
# which used to bypass `_summarize_persisted_step` entirely). --

_A_LONG_RESULT = (
    "The traditional rollback procedure requires draining every in-flight "
    "request before the schema migration begins, otherwise a half-applied "
    "column default can leave orphaned rows that the backfill job never "
    "revisits, which is exactly the failure mode this runbook exists to "
    "prevent for anyone paging through it at 3am."
)


@pytest.mark.asyncio
async def test_drilldown_step_text_grows_with_a_configured_cap_above_eighty(
    monkeypatch,
):
    monkeypatch.delenv("TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS", raising=False)
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        class _FakeBridge:
            def subagent_run(self, run_id):
                return {
                    "id": run_id,
                    "conversation_id": "conv-A",
                    "status": "done",
                    "task": "t",
                    "steps": [
                        {"kind": "tool_result", "summary": _A_LONG_RESULT},
                    ],
                }

        console._console_agent_bridge = _FakeBridge()
        console._character._current_console_rail_conversation_id = lambda: "conv-A"
        console._console_agent_drilldown_run_id = "run-x"
        console._agent._console_agent_drilldown_conversation_id = "conv-A"

        monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda *a, **k: 40)
        _status, steps_at_40, _subagents = console._agent._console_agent_section_lines()

        monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", lambda *a, **k: 300)
        _status, steps_at_300, _subagents = (
            console._agent._console_agent_section_lines()
        )

        # A bare `[:80]` slice (the pre-fix behavior) would make BOTH of
        # these identical (both capped at 80) regardless of the configured
        # value -- raising the cap from well below 80 to well above it
        # must visibly show more text.
        assert len(steps_at_300) > len(steps_at_40)
        assert len(steps_at_300) > 80
        assert "(+" in steps_at_40  # still truncated at the lower cap


# -- Review finding D: the "View full log" affordance's availability check
# (a SQLite lookup to resolve the target run id, plus a filesystem probe)
# must not run on every 0.2s rail tick -- cached per target run id, and
# skipped entirely while the Agent section is collapsed. --


@pytest.mark.asyncio
async def test_full_log_probe_never_touches_the_bridge_while_collapsed():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        probe_calls = []

        class _FakeBridge:
            def subagent_counts(self, conversation_ids):
                return {}

            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot(status="done")

            def subagent_run(self, run_id):
                return None

            def subagent_runs(self, conversation_id):
                return []

            def resolve_run_log_target(self, conversation_id, drill_id):
                return drill_id or self.latest_primary_run_id(conversation_id)

            def run_log_target_token(self, conversation_id):
                return ("turn", getattr(self, "target_run_id", "run-1"))

            def latest_primary_run_id(self, conversation_id):
                return "run-1"

            def run_log_available(self, run_id, *, cancelled=None):
                probe_calls.append(run_id)
                return True

        console._console_agent_bridge = _FakeBridge()
        console._console_agent_drilldown_run_id = None
        console._character._current_console_rail_conversation_id = lambda: "conv-A"
        console._agent._console_agent_drilldown_conversation_id = "conv-A"
        console._current_console_rail_state = lambda: SimpleNamespace(agent_open=False)

        # Several "ticks" while collapsed: not even the run-id resolution
        # -- let alone `run_log_available` -- may run.
        console._sync_console_agent_section()
        console._sync_console_agent_section()
        console._sync_console_agent_section()

        assert probe_calls == []


@pytest.mark.asyncio
async def test_full_log_probe_is_cached_per_run_id_while_open():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        probe_calls = []

        class _FakeBridge:
            def subagent_counts(self, conversation_ids):
                return {}

            def __init__(self):
                self.target_run_id = "run-1"

            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot(status="done")

            def subagent_run(self, run_id):
                return None

            def subagent_runs(self, conversation_id):
                return []

            def resolve_run_log_target(self, conversation_id, drill_id):
                return drill_id or self.latest_primary_run_id(conversation_id)

            def run_log_target_token(self, conversation_id):
                return ("turn", getattr(self, "target_run_id", "run-1"))

            def latest_primary_run_id(self, conversation_id):
                return self.target_run_id

            def run_log_available(self, run_id, *, cancelled=None):
                probe_calls.append(run_id)
                return True

        fake_bridge = _FakeBridge()
        console._console_agent_bridge = fake_bridge
        console._console_agent_drilldown_run_id = None
        console._character._current_console_rail_conversation_id = lambda: "conv-A"
        console._agent._console_agent_drilldown_conversation_id = "conv-A"
        console._current_console_rail_state = lambda: SimpleNamespace(agent_open=True)

        # Steady state (same target run id across ticks): probed once, then
        # every later tick is a cache hit.
        console._sync_console_agent_section()
        console._sync_console_agent_section()
        console._sync_console_agent_section()
        await host.workers.wait_for_complete()
        assert probe_calls == ["run-1"]

        # The target run changes (e.g. a new primary run started) -- the
        # cache must invalidate and re-probe exactly once for the new id.
        fake_bridge.target_run_id = "run-2"
        console._sync_console_agent_section()
        console._sync_console_agent_section()
        await host.workers.wait_for_complete()
        assert probe_calls == ["run-1", "run-2"]


# -- Review finding C: the full-log load (filesystem read + record parse +
# formatting) must happen off the UI thread -- the modal is only ever
# pushed once control is back on it. --


@pytest.mark.asyncio
async def test_view_full_log_loads_off_thread_then_opens_the_modal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        class _FakeBridge:
            def subagent_counts(self, conversation_ids):
                return {}

            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot(status="done")

            def subagent_run(self, run_id):
                return None

            def subagent_runs(self, conversation_id):
                return []

            def resolve_run_log_target(self, conversation_id, drill_id):
                return drill_id or self.latest_primary_run_id(conversation_id)

            def run_log_target_token(self, conversation_id):
                return ("turn", getattr(self, "target_run_id", "run-1"))

            def latest_primary_run_id(self, conversation_id):
                return "run-1"

            def run_log_available(self, run_id, *, cancelled=None):
                return True

            def load_run_log_page(self, run_id, *, cursor=None):
                from Tests.UI.test_console_run_log_paging import page
                return page(0, "full untruncated log text for run-1")

        console._console_agent_bridge = _FakeBridge()
        console._console_agent_drilldown_run_id = None
        console._character._current_console_rail_conversation_id = lambda: "conv-A"
        console._agent._console_agent_drilldown_conversation_id = "conv-A"

        stack_len_before = len(host.screen_stack)
        console._agent._open_console_agent_run_log_viewer()
        # The call itself only dispatches a worker -- it must return without
        # blocking and without having pushed the modal yet.
        assert len(host.screen_stack) == stack_len_before

        await host.workers.wait_for_complete()
        await pilot.pause()

        assert len(host.screen_stack) == stack_len_before + 1
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleRunLogModal)
        assert modal._run_id == "run-1"
        assert modal._page.slices[0].record.content == "full untruncated log text for run-1"


@pytest.mark.asyncio
async def test_view_full_log_no_ops_when_there_is_no_target_run():
    """No bridge/conversation/target -- must not dispatch a worker or push anything."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        console._console_agent_bridge = None  # no agent runtime available
        console._console_agent_drilldown_run_id = None

        stack_len_before = len(host.screen_stack)
        console._agent._open_console_agent_run_log_viewer()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert len(host.screen_stack) == stack_len_before


@pytest.mark.asyncio
async def test_activate_native_console_session_clears_stale_drilldown():
    """The shared session-activation path (tab click / Ctrl+K / Alt+1..9)
    clears the drill-down immediately on switch, not just on the next
    rail render."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)):
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

        await console._session._activate_native_console_session(first_session.id)

        assert console._console_agent_drilldown_run_id is None


# --- Finding D (original): repeated clicks on the combined sub-agents rail
# line had to reach every sub-agent run, not just the newest one -- the old
# mechanism was a cycling toggle that stepped one run per click. TASK-4
# (PR2b supervisor fleet) replaced that cycle with per-row click routing: a
# specific row's own id now resolves directly to its own run, so "every run
# is reachable" is proven by clicking rows out of order (not sequentially)
# and confirming each lands on exactly the run it names -- the meaning
# Finding D pinned, preserved through a different mechanism. ---


@pytest.mark.asyncio
async def test_clicking_a_specific_subagent_row_drills_into_that_run_directly():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        class _FakeBridge:
            _RUNS = [
                {
                    "id": "run-newest",
                    "conversation_id": "conv-A",
                    "status": "done",
                    "task": "t",
                    "steps": [],
                },
                {
                    "id": "run-mid",
                    "conversation_id": "conv-A",
                    "status": "done",
                    "task": "t",
                    "steps": [],
                },
                {
                    "id": "run-oldest",
                    "conversation_id": "conv-A",
                    "status": "done",
                    "task": "t",
                    "steps": [],
                },
            ]

            def subagent_runs(self, conversation_id):
                return list(self._RUNS)

            def subagent_counts(self, conversation_ids):
                return (
                    {"conv-A": len(self._RUNS)} if "conv-A" in conversation_ids else {}
                )

            def subagent_run(self, run_id):
                return next((r for r in self._RUNS if r["id"] == run_id), None)

            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot()

            def historical_snapshot(self, conversation_id):
                # Mirrors the real bridge's `_derive_historical_snapshot`
                # (PR2b Task 4 fix): each row's `run_id` is the record's own
                # permanent id, exactly what a real resumed conversation's
                # rows now carry.
                return AgentLiveSnapshot(
                    subagents=tuple(
                        SubAgentSummary(
                            text=r["task"], status=r["status"], run_id=r["id"]
                        )
                        for r in self._RUNS
                    )
                )

        console._console_agent_bridge = _FakeBridge()
        console._character._current_console_rail_conversation_id = lambda: "conv-A"

        rows = console._agent._console_agent_fleet_rows()
        assert [row.row_id for row in rows] == [
            "run-newest",
            "run-mid",
            "run-oldest",
        ]

        # Click rows out of order -- proves each row resolves DIRECTLY to
        # its own run, not by stepping through a shared cursor.
        console._agent._drill_into_console_agent_subagent(rows[2].row_id)
        await pilot.pause()  # drain the background rail-sync worker
        assert console._console_agent_drilldown_run_id == "run-oldest"

        console._agent._drill_into_console_agent_subagent(rows[0].row_id)
        await pilot.pause()
        assert console._console_agent_drilldown_run_id == "run-newest"

        console._agent._drill_into_console_agent_subagent(rows[1].row_id)
        await pilot.pause()
        assert console._console_agent_drilldown_run_id == "run-mid"

        # The dedicated Back button (not a row) always returns to the
        # overview directly, regardless of which row was last drilled into.
        console._console_agent_drilldown_run_id = None
        assert console._console_agent_drilldown_run_id is None


# -- PR2b Task 5: fleet token rollup + per-row cancel (controller unit) ---


@pytest.mark.asyncio
async def test_console_agent_fleet_token_total_sums_live_handles():
    """Sums `FleetHandle.total_tokens` off the LIVE fleet snapshot -- the
    same source `_console_agent_fleet_rows` itself reads for the row
    builders, so the ticker's aggregate and each row's own token segment
    can never disagree. A still-running handle's 0 naturally contributes
    nothing (no separate "terminal only" filter needed)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        handles = (
            FleetHandle(
                handle_id="h1",
                run_id="run-1",
                agent="a",
                task="t1",
                status="done",
                total_tokens=100,
            ),
            FleetHandle(
                handle_id="h2",
                run_id="run-2",
                agent="a",
                task="t2",
                status="running",
                total_tokens=0,
            ),
            FleetHandle(
                handle_id="h3",
                run_id="run-3",
                agent="a",
                task="t3",
                status="done",
                total_tokens=250,
            ),
        )

        class _FakeBridge:
            def fleet_snapshot(self, conversation_id):
                return list(handles) if conversation_id == "conv-A" else []

        console._console_agent_bridge = _FakeBridge()
        console._character._current_console_rail_conversation_id = lambda: "conv-A"

        assert console._agent._console_agent_fleet_token_total() == 350
        # An unrelated conversation id never sees this fleet's spend.
        console._character._current_console_rail_conversation_id = lambda: "conv-other"
        assert console._agent._console_agent_fleet_token_total() == 0


@pytest.mark.asyncio
async def test_console_agent_fleet_token_total_is_zero_with_no_bridge():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        console._console_agent_bridge = None
        assert console._agent._console_agent_fleet_token_total() == 0


@pytest.mark.asyncio
async def test_cancel_console_agent_fleet_row_delegates_to_the_bridge():
    """`_cancel_console_agent_fleet_row` forwards the ACTIVE conversation
    id plus the row's own id straight through to `ConsoleAgentBridge.
    cancel_subagent` -- no resolution step, matching how a live row's
    `row_id` already IS the handle id (`_fleet_row_from_handle`)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        calls = []

        class _FakeBridge:
            def cancel_subagent(self, conversation_id, handle_id):
                calls.append((conversation_id, handle_id))
                return True

        console._console_agent_bridge = _FakeBridge()
        console._character._current_console_rail_conversation_id = lambda: "conv-A"

        assert console._agent._cancel_console_agent_fleet_row("h1") is True
        assert calls == [("conv-A", "h1")]


@pytest.mark.asyncio
async def test_cancel_console_agent_fleet_row_returns_false_with_no_bridge():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        console._console_agent_bridge = None
        assert console._agent._cancel_console_agent_fleet_row("h1") is False


@pytest.mark.asyncio
async def test_cancel_console_agent_fleet_row_returns_false_with_no_row_id():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        class _FakeBridge:
            def subagent_counts(self, conversation_ids):
                return {}

            def cancel_subagent(self, conversation_id, handle_id):
                raise AssertionError("must not be called with an empty row id")

        console._console_agent_bridge = _FakeBridge()
        console._character._current_console_rail_conversation_id = lambda: "conv-A"
        assert console._agent._cancel_console_agent_fleet_row("") is False


# -- PR3a-1 Task 6b (audit F1): a LIVE child's steps exist nowhere but the
# -- bridge's own per-run slot
#
# `AgentService` persists a run's steps to `AgentRunsDB` once, at the end
# (`_persist`), so `subagent_run`'s record carries an EMPTY step list for
# the whole time a child is actually working -- and a fleet child now works
# on past the turn that spawned it, so "the whole time" can be minutes with
# no run in flight at all. Keying `_live` per run (the F1 fix) is what makes
# that progress addressable; this is the surface it is addressed FROM.


@pytest.mark.asyncio
async def test_drilldown_shows_a_live_childs_steps_before_they_reach_the_db():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        class _FakeBridge:
            def subagent_run(self, run_id):
                # Row exists from `create_run`; steps land only at the end.
                return {
                    "id": run_id,
                    "conversation_id": "conv-A",
                    "status": "running",
                    "task": "long job",
                    "steps": [],
                }

            def subagent_runs(self, conversation_id):
                return []

            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot()

            def live_run_snapshot(self, conversation_id, run_id):
                return AgentLiveSnapshot(
                    status="running",
                    step=2,
                    steps=(AgentLiveStep("tool_result", "read notes.md", "subagent"),),
                )

        console._console_agent_bridge = _FakeBridge()
        console._character._current_console_rail_conversation_id = lambda: "conv-A"
        console._console_agent_drilldown_run_id = "run-child"
        console._agent._console_agent_drilldown_conversation_id = "conv-A"

        status_line, steps_text, task_text = (
            console._agent._console_agent_section_lines()
        )

        assert status_line.startswith("Sub-agent ·")
        assert "read notes.md" in steps_text, (
            "a live child's steps were dropped: the DB has none yet"
        )
        assert task_text == "long job"


@pytest.mark.asyncio
async def test_drilldown_still_prefers_the_persisted_steps_once_they_exist():
    """The DB record is COMPLETE; the live slot only keeps the last few.

    So the live feed fills the gap and never replaces the record -- a
    finished child must still drill in to its full persisted history.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        class _FakeBridge:
            def subagent_run(self, run_id):
                return {
                    "id": run_id,
                    "conversation_id": "conv-A",
                    "status": "done",
                    "task": "long job",
                    "steps": [{"kind": "tool_result", "summary": "persisted step"}],
                }

            def subagent_runs(self, conversation_id):
                return []

            def live_snapshot(self, conversation_id):
                return AgentLiveSnapshot()

            def live_run_snapshot(self, conversation_id, run_id):
                return AgentLiveSnapshot(
                    status="done",
                    step=1,
                    steps=(AgentLiveStep("tool_result", "live step", "subagent"),),
                )

        console._console_agent_bridge = _FakeBridge()
        console._character._current_console_rail_conversation_id = lambda: "conv-A"
        console._console_agent_drilldown_run_id = "run-child"
        console._agent._console_agent_drilldown_conversation_id = "conv-A"

        _status, steps_text, _task = console._agent._console_agent_section_lines()

        assert "persisted step" in steps_text
        assert "live step" not in steps_text


@pytest.mark.asyncio
async def test_full_log_slow_probe_is_off_thread_and_rejects_stale_target():
    import threading

    from Tests.UI.test_console_run_log_paging import settled

    gate, entered = threading.Event(), threading.Event()
    ui_thread = threading.get_ident()
    calls = []
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._agent
        target = ["run-a"]

        class Bridge:
            def subagent_counts(self, conversation_ids):
                return {}

            def run_log_available(self, run_id, *, cancelled=None):
                calls.append((run_id, threading.get_ident()))
                entered.set()
                gate.wait(3)
                return True

        bridge = Bridge()
        controller._console_agent_bridge = bridge
        _select_test_log_run(controller, lambda: target[0])
        published = []
        console._sync_console_agent_section = lambda: published.append(
            controller._console_agent_full_log_cache_available
        )
        assert controller._console_agent_full_log_available() is False
        await settled(pilot, entered.is_set)
        assert calls == [("run-a", calls[0][1])]
        assert calls[0][1] != ui_thread
        target[0] = "run-b"
        gate.set()
        await host.workers.wait_for_complete()
        assert published == []
        assert controller._console_agent_full_log_cache_available is False
        assert controller._console_agent_full_log_available() is False
        await host.workers.wait_for_complete()
        assert published == [True]


@pytest.mark.asyncio
async def test_negative_availability_retries_after_real_first_record_append(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.UI.Console_Modules import agent as agent_module

    monkeypatch.setattr(run_log, "resolve_log_root", lambda: tmp_path)
    clock = [100.0]
    monkeypatch.setattr(agent_module, "_run_log_clock", lambda: clock[0], raising=False)
    db = AgentRunsDB(tmp_path / "append.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._agent
        controller._console_agent_bridge = bridge
        _select_test_log_run(controller, lambda: "append-run")
        published = []
        console._sync_console_agent_section = lambda: published.append(
            controller._console_agent_full_log_cache_available
        )
        assert controller._console_agent_full_log_available() is False
        await host.workers.wait_for_complete()
        assert published == [False]
        writer = run_log.RunLogWriter()
        writer.bind("append-run")
        writer.append(
            run_id="append-run",
            kind="primary",
            type="model",
            content="First real record",
        )
        assert controller._console_agent_full_log_available() is False
        clock[0] += 2
        assert controller._console_agent_full_log_available(allow_probe=False) is False
        assert published == [False]
        assert controller._console_agent_full_log_available() is False
        await host.workers.wait_for_complete()
        assert published == [False, True]
        assert controller._console_agent_full_log_available() is True


@pytest.mark.asyncio
async def test_pending_negative_probe_never_restarts_and_updates_real_affordance(
    monkeypatch,
):
    import threading

    from textual.widgets import Button

    from Tests.UI.test_console_run_log_paging import settled
    from tldw_chatbook.UI.Console_Modules import agent as agent_module

    clock = [100.0]
    monkeypatch.setattr(agent_module, "_run_log_clock", lambda: clock[0])
    gate, entered = threading.Event(), threading.Event()
    calls = []

    class Bridge:
        def subagent_counts(self, conversation_ids):
            return {}

        def run_log_available(self, run_id, *, cancelled=None):
            calls.append(run_id)
            entered.set()
            gate.wait(3)
            return True

    bridge = Bridge()
    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._agent
        payload = list(controller._console_agent_section_payload())
        controller._console_agent_bridge = bridge
        _select_test_log_run(controller, lambda: "held-run")
        # Keep the real DOM callback; isolate unrelated fleet payload construction.
        payload[5] = True
        controller._console_agent_section_payload = lambda: tuple(
            payload[:6] + [controller._console_agent_full_log_available()] + payload[7:]
        )
        console._sync_console_agent_section()
        await settled(pilot, entered.is_set)
        assert not console.query_one("#console-agent-view-full-log", Button).display
        for _ in range(5):
            clock[0] += 10
            assert controller._console_agent_full_log_available() is False
            assert (
                controller._console_agent_full_log_available(allow_probe=False) is False
            )
        assert calls == ["held-run"]
        gate.set()
        await host.workers.wait_for_complete()
        assert console.query_one("#console-agent-view-full-log", Button).display


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_bridge", [False, True])
async def test_initial_page_rejects_target_change_without_availability_scan(
    replace_bridge,
):
    import threading

    from Tests.UI.test_console_run_log_paging import page, settled

    gate, entered = threading.Event(), threading.Event()
    ui_thread = threading.get_ident()
    reads = []

    class Bridge:
        def subagent_counts(self, conversation_ids):
            return {}

        def run_log_available(self, *args, **kwargs):
            pytest.fail("initial viewer must not scan availability")

        def load_run_log_page(self, run_id, *, cursor=None):
            reads.append(threading.get_ident())
            entered.set()
            gate.wait(3)
            return page(0, "STALE INITIAL PAGE")

    current = [Bridge()]
    target = ["initial-run"]
    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._agent
        controller._console_agent_bridge = current[0]
        _select_test_log_run(controller, lambda: target[0])
        controller._open_console_agent_run_log_viewer()
        await settled(pilot, entered.is_set)
        if replace_bridge:
            current[0] = Bridge()
            controller._console_agent_bridge = current[0]
        else:
            target[0] = "replacement-run"
        gate.set()
        await host.workers.wait_for_complete()
        assert reads and all(thread != ui_thread for thread in reads)
        assert not any(
            isinstance(screen, ConsoleRunLogModal) for screen in host.screen_stack
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["absent", "empty-end", "empty-continuation"])
async def test_initial_empty_log_semantics(state):
    from tldw_chatbook.Agents.run_log_paging import RunLogPage, RunLogPageCursor

    first = (
        None
        if state == "absent"
        else RunLogPage(
            (),
            RunLogPageCursor(0, 0),
            RunLogPageCursor(0, 100) if state == "empty-continuation" else None,
            100,
        )
    )

    class Bridge:
        def subagent_counts(self, conversation_ids):
            return {}

        def load_run_log_page(self, run_id, *, cursor=None):
            return first

    bridge = Bridge()
    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._agent
        controller._console_agent_bridge = bridge
        _select_test_log_run(controller, lambda: "empty-run")
        controller._open_console_agent_run_log_viewer()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(host.screen_stack[-1], ConsoleRunLogModal) is (
            state == "empty-continuation"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_log_probe_settles_stale_or_cancelled_generation(cancel):
    import threading

    from Tests.UI.test_console_run_log_paging import settled

    entered, gate = threading.Event(), threading.Event()
    calls = []

    class Bridge:
        def subagent_counts(self, conversation_ids):
            return {}

        def run_log_available(self, run_id, *, cancelled=None):
            calls.append(run_id)
            entered.set()
            gate.wait(3)
            return len(calls) > 1

    bridge = Bridge()
    target = ["A"]
    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(180, 48)) as pilot:
        await _wait_for_selector(
            host.screen_stack[-1], pilot, "#console-rail-section-header-agent"
        )
        controller = host.screen_stack[-1]._agent
        controller._console_agent_bridge = bridge
        _select_test_log_run(controller, lambda: target[0])
        published = []
        controller._screen._sync_console_agent_section = lambda: published.append(True)
        assert not controller._console_agent_full_log_available()
        await settled(pilot, entered.is_set)
        if cancel:
            next(
                worker
                for worker in host.workers
                if worker.group == "run-log-availability"
            ).cancel()
        else:
            target[0] = "B"
            assert not controller._console_agent_full_log_available(allow_probe=False)
        gate.set()
        await settled(
            pilot, lambda: controller._console_agent_full_log_probe_pending is None
        )
        assert not published
        target[0] = "A"
        assert not controller._console_agent_full_log_available()
        await host.workers.wait_for_complete()
        assert calls == ["A", "A"]
        assert published == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("open_viewer", [False, True])
async def test_log_target_metadata_resolution_is_off_thread_and_stale_safe(open_viewer):
    import threading

    from Tests.UI.test_console_run_log_paging import page, settled

    gate, entered = threading.Event(), threading.Event()
    ui_thread = threading.get_ident()
    reads = []

    class Bridge:
        def subagent_counts(self, conversation_ids):
            return {}

        def run_log_target_token(self, conversation_id):
            return ("turn", "run-a")

        def resolve_run_log_target(self, conversation_id, drill_id):
            reads.append(threading.get_ident())
            entered.set()
            gate.wait(3)
            return "run-a"

        def latest_primary_run_id(self, conversation_id):
            return self.resolve_run_log_target(conversation_id, None)

        def run_log_available(self, run_id, *, cancelled=None):
            return True

        def load_run_log_page(self, run_id, *, cursor=None):
            return page(0, "UNEXPECTED OLD TARGET")

    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._agent
        bridge = Bridge()
        conversation = ["A"]
        controller._console_agent_bridge = bridge
        controller._current_rail_conversation_id = lambda: conversation[0]
        published = []
        console._sync_console_agent_section = lambda: published.append(True)
        if open_viewer:
            controller._open_console_agent_run_log_viewer()
        else:
            assert not controller._console_agent_full_log_available()
        await settled(pilot, entered.is_set)
        assert reads and all(thread != ui_thread for thread in reads)
        conversation[0] = "B"
        gate.set()
        await host.workers.wait_for_complete()
        assert not published
        assert not any(
            isinstance(screen, ConsoleRunLogModal) for screen in host.screen_stack
        )


def test_log_target_metadata_and_binding_token(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from functools import partial

    from tldw_chatbook.Agents.run_log import RunLogWriter

    db = AgentRunsDB(tmp_path / "target.db", client_id="t")
    bridge = ConsoleAgentBridge(agent_runs_db=db, store=None, provider_gateway=None)
    primary = db.create_run(conversation_id="A", agent_kind="primary")
    child = db.create_run(
        conversation_id="A", agent_kind="subagent", parent_run_id=primary
    )
    monkeypatch.setattr(
        db, "get_run", lambda *_: pytest.fail("log target must not hydrate steps")
    )
    assert bridge.resolve_run_log_target("A", None) == primary
    assert bridge.resolve_run_log_target("A", child) == child
    assert bridge.resolve_run_log_target("B", child) is None
    assert bridge.resolve_run_log_target("A", primary) is None
    before = bridge.run_log_target_token("A")
    bridge._publish_live(
        "A", "new-turn", AgentLiveSnapshot(status="running"), primary=True
    )
    started = bridge.run_log_target_token("A")
    assert started != before
    writer = RunLogWriter(
        root=tmp_path,
        on_bound=partial(
            bridge._remember_run_log_authority,
            session_id="session",
            conversation_id="A",
            access_scope=lambda: nullcontext(tmp_path),
        ),
    )
    writer.bind(primary)
    assert bridge.run_log_target_token("A") == ("new-turn", primary)
    assert bridge.run_log_target_token("A") != started


@pytest.mark.asyncio
async def test_log_path_does_not_initialize_an_absent_runtime(monkeypatch):
    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._agent
        with monkeypatch.context() as patch:
            patch.setattr(console, "_console_runtime_ref", None)
            patch.setattr(
                console,
                "_console_runtime",
                lambda: pytest.fail("log path created runtime"),
            )
            assert not controller._console_agent_full_log_available()
            controller._open_console_agent_run_log_viewer()
            assert not controller._run_log_target_matches(
                object(), ("A", None, (None, None))
            )


@pytest.mark.asyncio
async def test_old_log_probe_cannot_settle_newer_pending_generation():
    import threading

    from Tests.UI.test_console_run_log_paging import settled

    gates = {run: threading.Event() for run in ("A", "B")}
    entered = {run: threading.Event() for run in ("A", "B")}

    class Bridge:
        def subagent_counts(self, conversation_ids):
            return {}

        def run_log_available(self, run_id, *, cancelled=None):
            entered[run_id].set()
            gates[run_id].wait(3)
            return True

    bridge = Bridge()
    target = ["A"]
    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._agent
        controller._console_agent_bridge = bridge
        _select_test_log_run(controller, lambda: target[0])
        console._sync_console_agent_section = lambda: None
        completed = []
        publish = controller._publish_console_agent_log_availability

        def recording_publish(*args):
            publish(*args)
            completed.append(args[-2])

        controller._publish_console_agent_log_availability = recording_publish
        assert not controller._console_agent_full_log_available()
        await settled(pilot, entered["A"].is_set)
        first_generation = controller._console_agent_full_log_probe_pending
        target[0] = "B"
        assert not controller._console_agent_full_log_available()
        await settled(pilot, entered["B"].is_set)
        second_generation = controller._console_agent_full_log_probe_pending
        assert first_generation != second_generation
        gates["A"].set()
        await settled(pilot, lambda: first_generation in completed)
        assert controller._console_agent_full_log_probe_pending == second_generation
        assert not controller._console_agent_full_log_cache_available
        gates["B"].set()
        await settled(pilot, lambda: second_generation in completed)
        assert controller._console_agent_full_log_probe_pending is None
        assert controller._console_agent_full_log_cache_available


@pytest.mark.asyncio
async def test_modal_log_predicate_uses_turn_token_without_metadata_reads():
    import threading

    from textual.widgets import TextArea

    from Tests.UI.test_console_run_log_paging import page, settled

    ui_thread = threading.get_ident()
    token = ["turn-a"]
    metadata_reads, page_reads = [], []

    class Bridge:
        def subagent_counts(self, conversation_ids):
            return {}

        def run_log_target_token(self, conversation_id):
            return (token[0], "run-a")

        def resolve_run_log_target(self, conversation_id, drill_id):
            metadata_reads.append(threading.get_ident())
            assert threading.get_ident() != ui_thread
            return "run-a"

        def load_run_log_page(self, run_id, *, cursor=None):
            page_reads.append(threading.get_ident())
            return page(1 if cursor else 0, "SECOND" if cursor else "FIRST")

    bridge = Bridge()
    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._agent
        controller._console_agent_bridge = bridge
        controller._current_rail_conversation_id = lambda: "A"
        console._sync_console_agent_section = lambda: None
        controller._open_console_agent_run_log_viewer()
        await settled(
            pilot, lambda: isinstance(host.screen_stack[-1], ConsoleRunLogModal)
        )
        modal = host.screen_stack[-1]
        await pilot.click("#console-run-log-next")
        await settled(pilot, lambda: "SECOND" in modal.query_one(TextArea).text)
        assert len(metadata_reads) == 1
        assert len(page_reads) == 2 and all(
            thread != ui_thread for thread in page_reads
        )
        token[0] = "replacement-turn"
        await pilot.click("#console-run-log-next")
        await pilot.pause()
        assert len(metadata_reads) == 1
        assert len(page_reads) == 2


@pytest.mark.asyncio
async def test_log_probe_cancelled_before_entry_retries_on_existing_tick(monkeypatch):
    from Tests.UI.test_console_run_log_paging import settled

    calls = []

    class Bridge:
        def subagent_counts(self, conversation_ids):
            return {}

        def run_log_available(self, run_id, *, cancelled=None):
            calls.append(run_id)
            return True

    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")
        controller = console._agent
        controller._console_agent_bridge = Bridge()
        _select_test_log_run(controller, lambda: "A")
        console._sync_console_agent_section = lambda: None
        original = console.run_worker
        dispatched = []

        def queued_once(work, **kwargs):
            if kwargs.get("group") != "run-log-availability":
                return original(work, **kwargs)
            worker = original(work, start=bool(dispatched), **kwargs)
            dispatched.append(worker)
            return worker

        monkeypatch.setattr(console, "run_worker", queued_once)
        assert not controller._console_agent_full_log_available()
        assert not calls
        dispatched[0].cancel()
        assert not controller._console_agent_full_log_available()
        await settled(pilot, lambda: controller._console_agent_full_log_cache_available)
        assert calls == ["A"]
        assert len(dispatched) == 2
        assert controller._console_agent_full_log_probe_pending is None
