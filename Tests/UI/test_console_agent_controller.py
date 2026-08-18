"""Characterisation of the Console agent cluster (wave-4 decomposition, task 3).

Written and run green BEFORE `ConsoleAgentController` existed, driven
entirely through `ChatScreen`'s own names -- so this file is the
extraction's before/after equivalence check, not a description of wherever
the code happens to live afterwards.

The extraction then repointed the calls the screen no longer answers (it
kept only `_ensure_console_agent_bridge`, which three other test files
replace by name), exactly as it repointed the pre-existing suite. What did
NOT change is every assertion: same seeded run store, same expected text,
same painted widgets, same call counts.

What the pre-existing suite already covers, and this file therefore does not
duplicate:

- `Tests/UI/test_console_agent_rail.py` -- the rail's compose-time widgets,
  bracket/escaping behaviour, the drill-in's conversation scoping, the
  full-log probe cache, and the off-thread full-log load + modal push.
- `Tests/UI/test_console_parallel_runs.py` -- the fleet summary line and
  `_apply_fleet_agent_section_auto_open`'s sticky-dismissal window.
- `Tests/UI/test_console_native_chat_flow.py::test_resume_wiring_injects_
  agent_markers_from_agent_runs_db` -- `_inject_resume_agent_markers` over a
  real sibling `AgentRunsDB`.

What was NOT covered, and is pinned here:

1. The whole persisted-state -> painted-rail path in one go: rows in a real
   `AgentRunsDB`, read through a real `ConsoleAgentBridge`, rendered by
   `_sync_console_agent_section` into the really-mounted Statics.
2. That the sync's equality guard is observable behaviour (a second tick
   really does skip the `Static.update()` calls), not just an internal memo.
3. Drilling directly into a *persisted* sub-agent run by its own row id
   (PR2b Task 4 replaced the old cycling toggle with per-row click
   routing), with `_console_agent_full_log_run_id` tracking drill-in vs
   overview against those same real records.
4. `_console_subagent_counts_for_rows`' batching and its row-set cache
   invalidation, measured against a real DB rather than a fake bridge.
5. `_ensure_console_agent_bridge`'s durable-vs-`:memory:` fork and its
   memoization.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
)

#: Terminal size the agent-section tests mount at -- wide enough that the
#: Agent rail section is expanded rather than collapsed by the responsive
#: shell. File-local on purpose; see the note in
#: `test_console_button_routing.py` for why there is no shared constant.
_AGENT_SECTION_SIZE = (180, 48)


def _bridge_over(db_path) -> ConsoleAgentBridge:
    """A real bridge over a real durable run store -- no fakes anywhere."""
    return ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(db_path, client_id="t"),
        store=None,
        provider_gateway=None,
    )


def _seed_done_primary_with_subagents(db_path, *, conversation_id="conv-A", tasks=()):
    """Persist one finished primary run plus a sub-agent run per ``tasks``."""
    db = AgentRunsDB(db_path, client_id="t")
    primary_id = db.create_run(conversation_id=conversation_id, agent_kind="primary")
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
    sub_ids = []
    for task in tasks:
        sub_id = db.create_run(
            conversation_id=conversation_id,
            agent_kind="subagent",
            task=task,
            parent_run_id=primary_id,
        )
        db.set_status(sub_id, "done", result=f"done {task}")
        sub_ids.append(sub_id)
    return primary_id, sub_ids


def _static_text(console, widget_id: str) -> str:
    from textual.widgets import Static

    return str(console.query_one(widget_id, Static).renderable)


@pytest.mark.asyncio
async def test_persisted_run_state_reaches_the_mounted_agent_rail_statics(tmp_path):
    """Rows in a real ``AgentRunsDB`` -> real bridge -> painted rail Statics.

    The end-to-end path the cluster exists for. Nothing live has ever run in
    this process, so every line below is re-derived from the durable store
    (``historical_snapshot``), which is exactly the resumed-after-restart
    case.
    """
    db_path = tmp_path / "agent_runs.db"
    _seed_done_primary_with_subagents(
        db_path, tasks=("research pricing", "summarize docs")
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        bridge = _bridge_over(db_path)
        console._console_agent_bridge = bridge
        console._console_agent_drilldown_run_id = None
        console._current_console_rail_conversation_id = lambda: "conv-A"
        console._agent._console_agent_drilldown_conversation_id = "conv-A"

        # Precondition: nothing live -- the text below can only come from the
        # persisted run store.
        assert bridge.live_snapshot("conv-A").status == "idle"

        status_line, steps_text, subagents_text = (
            console._agent._console_agent_section_lines()
        )
        assert status_line == "Agent: done"
        assert "final answer" in steps_text
        assert "research pricing" in subagents_text
        assert "summarize docs" in subagents_text

        console._sync_console_agent_section()
        # The fleet mini-section goes from 0 rows (nothing set up yet at
        # initial compose) to 2 -- a structural change, so `sync_state`
        # schedules a `refresh(recompose=True)` rather than patching in
        # place (`ConsoleInspectorSection.sync_state`'s own discipline,
        # Task 3). The row Statics queried below don't exist until that
        # recompose actually runs.
        await pilot.pause()

        assert _static_text(console, "#console-agent-section-status") == "Agent: done"
        assert "final answer" in _static_text(console, "#console-agent-section-steps")
        # PR2b Task 4: the joined-string Static that used to live at this id
        # is now a `ConsoleInspectorSection` -- read its mounted rows'
        # primary/secondary text instead of a single Static's renderable.
        fleet_section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        painted_subagents = " ".join(
            f"{row.primary_text} {row.secondary_text}" for row in fleet_section.rows
        )
        assert "research pricing" in painted_subagents
        assert "summarize docs" in painted_subagents
        # Review round 2 (Task 4 approval, one Medium finding): the check
        # above reads `InspectorSectionRow` value objects -- plain Python
        # attributes the controller built, not what the compositor actually
        # painted. Blanking every row's rendered Static content left this
        # test green (the reviewer proved it; mutation-verified again
        # below). This IS the historical/resumed path (nothing live has
        # ever run in this process -- see the docstring), the one path
        # whose real DOM rendering was otherwise unguarded anywhere in the
        # suite: `test_state_2_expanded_rows_render_two_painted_lines_per_
        # child` (`Tests/UI/test_console_fleet_panel.py`) only exercises the
        # LIVE-handle row builder. Read the REAL mounted row Statics too.
        painted_row_statics = " ".join(
            f"{_static_text(console, f'#console-inspector-section-agent-fleet-row-{i}-primary')} "
            f"{_static_text(console, f'#console-inspector-section-agent-fleet-row-{i}-secondary')}"
            for i in range(len(fleet_section.rows))
        )
        assert "research pricing" in painted_row_statics
        assert "summarize docs" in painted_row_statics


@pytest.mark.asyncio
async def test_agent_section_sync_skips_repainting_an_unchanged_payload(tmp_path):
    """TASK-251's equality guard is observable, not just an internal memo.

    Overwriting a Static by hand and re-syncing must leave the hand-written
    text in place: the second tick recognises an unchanged payload and never
    reaches ``Static.update()`` at all.
    """
    db_path = tmp_path / "agent_runs.db"
    _primary_id, sub_ids = _seed_done_primary_with_subagents(
        db_path, tasks=("research pricing",)
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        from textual.widgets import Static

        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        console._console_agent_bridge = _bridge_over(db_path)
        console._console_agent_drilldown_run_id = None
        console._current_console_rail_conversation_id = lambda: "conv-A"
        console._agent._console_agent_drilldown_conversation_id = "conv-A"

        console._sync_console_agent_section()
        assert _static_text(console, "#console-agent-section-status") == "Agent: done"

        console.query_one("#console-agent-section-status", Static).update("SENTINEL")
        console._sync_console_agent_section()
        assert _static_text(console, "#console-agent-section-status") == "SENTINEL"

        # ...and a genuinely changed payload does repaint: drilling in flips
        # the status line, so the guard is a guard and not a one-shot.
        # TASK-4: drills directly into the seeded sub-agent's own row id
        # (the old cycling toggle stepped to it via `runs[0]`; a click on
        # its row now resolves the same target directly).
        console._agent._drill_into_console_agent_subagent(sub_ids[0])
        await pilot.pause()
        console._sync_console_agent_section()
        assert _static_text(console, "#console-agent-section-status").startswith(
            "Sub-agent · done"
        )


@pytest.mark.asyncio
async def test_drilldown_row_click_retargets_the_full_log_to_that_run(
    tmp_path,
):
    """A specific row's drill-in retargets "View full log" to THAT run --
    overview targets the latest primary run, drilling into a sub-agent row
    targets that row's own run, directly (not via a shared cycling cursor;
    TASK-4 replaced the old step-through-every-run toggle with per-row
    click routing -- see ``test_console_agent_rail.py``'s
    ``test_clicking_a_specific_subagent_row_drills_into_that_run_directly``
    for the row-derivation half of this same replacement)."""
    db_path = tmp_path / "agent_runs.db"
    primary_id, sub_ids = _seed_done_primary_with_subagents(
        db_path, tasks=("oldest task", "newest task")
    )
    oldest_sub, newest_sub = sub_ids

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        console._console_agent_bridge = _bridge_over(db_path)
        console._console_agent_drilldown_run_id = None
        console._current_console_rail_conversation_id = lambda: "conv-A"
        console._agent._console_agent_drilldown_conversation_id = "conv-A"

        # Overview: the affordance targets the conversation's latest primary.
        #
        # `_console_agent_full_log_run_id` is the ONE name in this file that
        # had to follow the wave-4 task-3 extraction: it has no consumer
        # outside the agent cluster, so it moved to `ConsoleAgentController`
        # with no screen-level delegation. Every other assertion here still
        # reads through `ChatScreen`'s own names, unchanged from the
        # pre-move run.
        full_log_run_id = console._agent._console_agent_full_log_run_id
        assert full_log_run_id() == primary_id

        # Drill into the OLDEST sub-agent's row directly -- not "the first
        # one a cycling cursor would reach".
        console._agent._drill_into_console_agent_subagent(oldest_sub)
        await pilot.pause()
        assert console._console_agent_drilldown_run_id == oldest_sub
        assert full_log_run_id() == oldest_sub

        # Click a DIFFERENT row next, out of any sequential order -- proves
        # each row resolves to its own run independently of drill history.
        console._agent._drill_into_console_agent_subagent(newest_sub)
        await pilot.pause()
        assert console._console_agent_drilldown_run_id == newest_sub
        assert full_log_run_id() == newest_sub

        # Back to the overview (the dedicated Back button's own effect,
        # not a row) -- the affordance reverts to the latest primary run.
        console._console_agent_drilldown_run_id = None
        assert full_log_run_id() == primary_id


@pytest.mark.asyncio
async def test_subagent_badge_counts_batch_once_and_cache_until_the_row_set_changes(
    tmp_path,
):
    """One batched DB query per refresh, and no re-query while the visible
    row set is unchanged (the 0.2s poll tick calls this every time)."""
    db_path = tmp_path / "agent_runs.db"
    _seed_done_primary_with_subagents(db_path, tasks=("t1", "t2"))
    _seed_done_primary_with_subagents(
        tmp_path / "agent_runs.db", conversation_id="conv-B", tasks=("t3",)
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        bridge = _bridge_over(db_path)
        calls: list[list[str]] = []
        original = bridge._db.count_subagents_by_conversation

        def _spy(ids):
            calls.append(sorted(ids))
            return original(ids)

        bridge._db.count_subagents_by_conversation = _spy

        rows_ab = [
            SimpleNamespace(conversation_id="conv-A"),
            SimpleNamespace(conversation_id="conv-B"),
        ]

        counts = console._agent._console_subagent_counts_for_rows(bridge, rows_ab)
        assert counts == {"conv-A": 2, "conv-B": 1}
        assert calls == [["conv-A", "conv-B"]]

        # Same visible rows, next tick: served from cache, no second query.
        assert console._agent._console_subagent_counts_for_rows(bridge, rows_ab) == {
            "conv-A": 2,
            "conv-B": 1,
        }
        assert len(calls) == 1

        # The visible row set changed -- one more batched query, still one.
        assert console._agent._console_subagent_counts_for_rows(
            bridge, rows_ab[:1]
        ) == {"conv-A": 2}
        assert calls[-1] == ["conv-A"]
        assert len(calls) == 2

        # A bridge-less screen (no agent runtime) yields no counts and never
        # touches the DB.
        assert console._agent._console_subagent_counts_for_rows(None, rows_ab) == {}
        assert len(calls) == 2


def test_agent_bridge_is_built_from_the_sibling_run_store_and_memoized(tmp_path):
    """``_ensure_console_agent_bridge`` keys the run store off the durable
    ChaChaNotes path, sees rows another handle persisted there, and hands
    back the same instance on every later call."""
    db_path = tmp_path / "agent_runs.db"
    _, sub_ids = _seed_done_primary_with_subagents(db_path, tasks=("research",))

    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = SimpleNamespace(
        db_path=str(tmp_path / "chacha.db")
    )

    bridge = screen._ensure_console_agent_bridge()
    assert bridge is not None
    assert bridge.subagent_count("conv-A") == 1
    record = bridge.subagent_run(sub_ids[0])
    assert record is not None and record["task"] == "research"

    assert screen._ensure_console_agent_bridge() is bridge


def test_agent_bridge_is_absent_without_a_durable_run_store(tmp_path):
    """An in-memory harness has nowhere to key the sibling run store off, so
    there is no agent runtime.

    The ``None`` verdict is deliberately NOT memoized (only a real bridge
    is): the memo check is ``is not None``, so a screen whose durable DB
    arrives later still gets a bridge on the next call. Pinned because a
    "tidy" rewrite of that guard into a sentinel-based memo would silently
    strand every such screen without an agent runtime for its whole life.
    """
    screen = ChatScreen(_build_test_app())
    screen.app_instance.chachanotes_db = SimpleNamespace(db_path=":memory:")

    assert screen._ensure_console_agent_bridge() is None
    assert screen._console_agent_bridge is None

    # No db at all -- same verdict, still no crash.
    screen.app_instance.chachanotes_db = None
    assert screen._ensure_console_agent_bridge() is None

    # A durable path appears afterwards: the next call builds for real.
    screen.app_instance.chachanotes_db = SimpleNamespace(
        db_path=str(tmp_path / "chacha.db")
    )
    assert screen._ensure_console_agent_bridge() is not None


@pytest.mark.asyncio
async def test_drilldown_header_names_the_resumed_from_run(tmp_path):
    """PR3b Task 4: a resumed sub-agent's drill-in header carries its
    lineage -- ``resumed from <id>`` -- read straight off the run row's
    ``resumed_from_run_id`` column (SELECT * flows it here for free); a
    row without one keeps the exact pre-existing header."""
    db_path = tmp_path / "agent_runs.db"
    _primary_id, sub_ids = _seed_done_primary_with_subagents(
        db_path, tasks=("original attempt",)
    )
    original_sub = sub_ids[0]
    db = AgentRunsDB(db_path, client_id="t")
    resumed_sub = db.create_run(
        conversation_id="conv-A",
        agent_kind="subagent",
        task="original attempt",
        parent_run_id=_primary_id,
        resumed_from_run_id=original_sub,
    )
    db.set_status(resumed_sub, "done", result="second pass")

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-rail-section-header-agent")

        console._console_agent_bridge = _bridge_over(db_path)
        console._current_console_rail_conversation_id = lambda: "conv-A"
        console._agent._console_agent_drilldown_conversation_id = "conv-A"

        console._agent._drill_into_console_agent_subagent(resumed_sub)
        await pilot.pause()
        console._sync_console_agent_section()
        status = _static_text(console, "#console-agent-section-status")
        assert status.startswith("Sub-agent · done")
        assert f"resumed from {original_sub}" in status

        # A NON-resumed row's header is byte-identical to before.
        console._agent._drill_into_console_agent_subagent(original_sub)
        await pilot.pause()
        console._sync_console_agent_section()
        assert (
            _static_text(console, "#console-agent-section-status")
            == "Sub-agent · done (Back)"
        )
