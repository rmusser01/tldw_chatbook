"""Fleet preview, full history paging, and selection through the real Console."""

import threading
from dataclasses import replace

import pytest
from textual.widgets import Button, DataTable, Static

from Tests.UI.test_console_fleet_budget_history import _bridge, _painted_text
from Tests.UI.test_console_fleet_historical_detail import _StyledConsoleHarness
from Tests.UI.test_console_fleet_panel import _scroll_into_view, _setup_console
from Tests.UI.test_console_inspector_section import _rows, _SectionHarness
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
from Tests.UI.test_console_parallel_runs import _assert_painted_at_own_region
from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionRow,
    ConsoleInspectorSectionState,
)


@pytest.mark.asyncio
async def test_preview_bounds_mounts_and_hidden_changes_do_not_recompose():
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        rows=_rows(12),
        summary="12 done",
        max_visible_rows=4,
        view_all_label="View all runs",
    )
    host = _SectionHarness(section)
    async with host.run_test(size=(70, 20)) as pilot:
        await pilot.pause()
        assert len(section.query(ConsoleInspectorSectionRow)) == 4
        assert len(section.rows) == 12
        updated = list(section.rows)
        updated[-1] = replace(updated[-1], secondary_text="hidden update")
        section.sync_state(
            ConsoleInspectorSectionState(rows=tuple(updated), summary="12 done")
        )
        await pilot.pause()
        assert section.recompose_count == 0
        assert len(section.query(ConsoleInspectorSectionRow)) == 4


async def _loaded(pilot, modal):
    for _ in range(80):
        if not modal.query_one(DataTable).disabled:
            return
        await pilot.pause(0.05)
    raise AssertionError("history page did not finish loading")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(180, 48), (120, 35)])
async def test_view_all_pages_to_old_child_and_drills_into_it(tmp_path, size):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="history")
    try:
        parent = db.create_run(conversation_id="conv-A", agent_kind="primary")
        for index in range(55):
            child = db.create_run(
                conversation_id="conv-A",
                agent_kind="subagent",
                parent_run_id=parent,
                task=f"Child {index:02d}",
            )
            db.set_status(
                child, "done", result=f"Answer {index:02d}", budget_tokens=index
            )
        db.set_status(parent, "done")
        bridge = _bridge(db)
        app = _build_test_app()
        _configure_native_ready_console(app)
        host = _StyledConsoleHarness(app)
        async with host.run_test(size=size) as pilot:
            console = await _setup_console(pilot, host, bridge)
            section = console.query_one(
                "#console-agent-section-subagents", ConsoleInspectorSection
            )
            # Expansion is a real click. Its own handler scrolls the new preview.
            assert await pilot.click("#console-inspector-section-agent-fleet-toggle")
            await pilot.pause()
            assert section.open
            assert len(section.query(ConsoleInspectorSectionRow)) == 4
            tail = console.query_one(
                "#console-inspector-section-agent-fleet-view-all", Button
            )
            _assert_painted_at_own_region(host, tail)
            await pilot.click(tail)
            await pilot.pause()
            from tldw_chatbook.Widgets.Console.console_agent_history_modal import (
                ConsoleAgentHistoryModal,
            )

            modal = host.screen
            assert isinstance(modal, ConsoleAgentHistoryModal)
            await _loaded(pilot, modal)
            table = modal.query_one(DataTable)
            assert table.row_count == 50
            await pilot.click("#agent-history-next")
            await _loaded(pilot, modal)
            assert table.row_count == 5
            await pilot.click("#agent-history-previous")
            await _loaded(pilot, modal)
            assert table.row_count == 50
            await pilot.click("#agent-history-next")
            await _loaded(pilot, modal)
            assert table.row_count == 5
            target = str(table.ordered_rows[-1].key.value)
            table.focus()
            await pilot.press("pagedown")
            await pilot.pause()
            assert (
                str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)
                == target
            )
            task = db.get_run(target)["task"]
            assert task in _painted_text(host, table)
            if size == (120, 35):
                # Hit-test the painted final row, then select it with a real click.
                cell = table._get_cell_region(table.cursor_coordinate)
                x = table.content_region.x + 1
                y = table.content_region.y + cell.y - int(table.scroll_y)
                assert host.screen.get_widget_at(x, y)[0] is table
                assert await pilot.click(offset=(x, y))
            else:
                await pilot.press("enter")
            await pilot.pause()
            assert host.screen is console
            assert console._console_agent_drilldown_run_id == target
            await _scroll_into_view(pilot, console, "#console-agent-section-steps")
            assert "This run:" in _painted_text(
                host, console.query_one("#console-agent-section-steps")
            )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_history_error_is_recoverable_and_dismissed_load_is_safe():
    from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
    from tldw_chatbook.Widgets.Console.console_agent_history_modal import (
        ConsoleAgentHistoryModal,
    )

    attempts = 0
    entered = threading.Event()
    release = threading.Event()

    def loader(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("private DB details")
        entered.set()
        assert release.wait(5)
        return []

    host = ConsolidatedCSSApp(css_path=BUNDLED_STYLESHEET)
    async with host.run_test(size=(100, 30)) as pilot:
        modal = ConsoleAgentHistoryModal(load_page=loader)
        await host.push_screen(modal)
        await pilot.pause(0.2)
        assert "Refresh to retry" in str(
            modal.query_one("#agent-history-state", Static).renderable
        )
        assert "private DB" not in str(
            modal.query_one("#agent-history-state", Static).renderable
        )
        try:
            await pilot.click("#agent-history-refresh")
            for _ in range(50):
                if entered.is_set():
                    break
                await pilot.pause(0.02)
            assert entered.is_set()
            await pilot.press("escape")
            await pilot.pause()
            assert host.screen is not modal
        finally:
            release.set()
            await pilot.pause(0.1)


@pytest.mark.asyncio
async def test_history_remains_reachable_after_a_childless_primary_run(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="history")
    try:
        parent = db.create_run(conversation_id="conv-A", agent_kind="primary")
        child = db.create_run(
            conversation_id="conv-A",
            agent_kind="subagent",
            parent_run_id=parent,
            task="Earlier child",
        )
        db.set_status(child, "done", budget_tokens=10)
        db.set_status(parent, "done")
        newer = db.create_run(conversation_id="conv-A", agent_kind="primary")
        db.set_status(newer, "done")
        app = _build_test_app()
        _configure_native_ready_console(app)
        host = _StyledConsoleHarness(app)
        async with host.run_test(size=(120, 35)) as pilot:
            console = await _setup_console(pilot, host, _bridge(db))
            section = console.query_one(
                "#console-agent-section-subagents", ConsoleInspectorSection
            )
            assert not section.rows
            assert section.styles.display == "block"
            assert await pilot.click("#console-inspector-section-agent-fleet-toggle")
            await pilot.pause()
            tail = section.query_one("#console-inspector-section-agent-fleet-view-all")
            _assert_painted_at_own_region(host, tail)
            assert await pilot.click(tail)
            await _loaded(pilot, host.screen)
            assert host.screen.query_one(DataTable).row_count == 1
            await pilot.press("enter")
            await pilot.pause()
            assert console._console_agent_drilldown_run_id == child
    finally:
        db.close()


@pytest.mark.asyncio
async def test_empty_history_paints_recovery_controls():
    from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
    from tldw_chatbook.Widgets.Console.console_agent_history_modal import (
        ConsoleAgentHistoryModal,
    )

    host = ConsolidatedCSSApp(css_path=BUNDLED_STYLESHEET)
    async with host.run_test(size=(100, 30)) as pilot:
        modal = ConsoleAgentHistoryModal(load_page=lambda **kwargs: [])
        await host.push_screen(modal)
        await _loaded(pilot, modal)
        state = modal.query_one("#agent-history-state", Static)
        assert "No saved sub-agent runs" in _painted_text(host, state)
        assert modal.query_one("#agent-history-previous", Button).disabled
        assert modal.query_one("#agent-history-next", Button).disabled
        assert await pilot.click("#agent-history-close")
        await pilot.pause()
        assert host.screen is not modal


@pytest.mark.asyncio
async def test_history_selection_rechecks_conversation_and_agent_kind(
    tmp_path, monkeypatch
):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="history")
    try:
        parent = db.create_run(conversation_id="conv-A", agent_kind="primary")
        child = db.create_run(
            conversation_id="conv-A", agent_kind="subagent", parent_run_id=parent
        )
        foreign = db.create_run(conversation_id="conv-B", agent_kind="subagent")
        bridge = _bridge(db)
        app = _build_test_app()
        _configure_native_ready_console(app)
        host = _StyledConsoleHarness(app)
        async with host.run_test(size=(180, 48)) as pilot:
            console = await _setup_console(pilot, host, bridge)
            callbacks = []
            monkeypatch.setattr(
                host, "push_screen", lambda modal, callback: callbacks.append(callback)
            )
            console._agent.open_fleet_history()
            callbacks[0](foreign)
            callbacks[0](parent)
            assert console._console_agent_drilldown_run_id is None
            console._current_console_rail_conversation_id = lambda: "conv-B"
            callbacks[0](child)
            assert console._console_agent_drilldown_run_id is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_expansion_scrolls_once_and_in_place_refresh_preserves_reading_position():
    from Tests.UI.test_console_inspector_section import _ScrolledSectionHarness

    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        rows=_rows(12),
        summary="12 done",
        max_visible_rows=4,
        view_all_label="View all runs",
        open=False,
        scroll_on_expand=True,
    )
    host = _ScrolledSectionHarness(section)
    async with host.run_test(size=(70, 20)) as pilot:
        section.set_open(True)
        await pilot.pause(0.2)
        tail = section.query_one("#console-inspector-section-agents-view-all")
        _assert_painted_at_own_region(host, tail)
        scroll = host.query_one("#scroll")
        assert scroll.scroll_y > 0
        scroll.scroll_to(y=0, animate=False)
        await pilot.pause()
        changed = list(section.rows)
        changed[0] = replace(changed[0], secondary_text="new detail")
        section.sync_state(
            ConsoleInspectorSectionState(rows=tuple(changed), summary="12 done")
        )
        await pilot.pause(0.2)
        assert scroll.scroll_y == 0
        assert section.recompose_count == 0
