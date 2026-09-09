"""Saved automatic-work pauses stay visible and use existing manual actions."""

from __future__ import annotations

import asyncio
import threading

import pytest
from textual.widgets import DataTable

from Tests.Chat.test_console_fleet_wake import (
    _drain,
    _RecordingWakeGateway,
    _survivor,
    _terminal_subagent_run,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_console_fleet_panel import _scroll_into_view
from Tests.UI.test_console_fleet_wake_ui_freshness import _settle
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_inspector_section import _rows, _SectionHarness
from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_console_session_tab_strip import _rendered_text
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunMarker
from tldw_chatbook.Chat.console_fleet_attention import clear_fleet_unseen_completion
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Widgets.Console.console_agent_history_modal import (
    ConsoleAgentHistoryModal,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_inspector_section import (
    ConsoleInspectorSection,
    ConsoleInspectorSectionState,
)


class _ConcurrentGateway(_RecordingWakeGateway):
    def __init__(self):
        super().__init__(reply="Manual continuation completed")
        self.thread_gate = None

    async def stream_chat(self, resolution, messages, **kwargs):
        self.payloads.append([dict(message) for message in messages])
        while self.thread_gate is not None and not self.thread_gate.is_set():
            await asyncio.sleep(0.01)
        yield self.reply


def _painted_notice(host, console):
    notice = console.query_one("#console-inspector-section-agent-fleet-notice")
    region = notice.region
    strips = host.screen._compositor.render_strips()
    return " ".join(
        strip.crop(region.x, region.right).text.strip()
        for strip in strips[max(0, region.y) : region.bottom]
    )


class _PauseHarness(ConsoleHarness):
    CSS_PATH = BUNDLED_STYLESHEET


def _app(tmp_path):
    app = _build_test_app()
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    _attach_real_dbs(app, tmp_path)
    return app


async def _rig(host, pilot):
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    controller = console._ensure_console_chat_controller()
    bridge = console._ensure_console_agent_bridge()
    assert bridge is not None
    _select_llamacpp_console(console)
    gateway = _ConcurrentGateway()
    console._console_provider_gateway = gateway
    controller.provider_gateway = gateway
    bridge._gateway = gateway
    store = console._ensure_console_chat_store()
    session = store.ensure_session()
    store.set_session_project_instruction_state(
        session.id, ProjectInstructionControlState.legacy_disabled()
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="Research topic", persist=True
    )
    assert session.persisted_conversation_id
    await _settle(pilot, lambda: controller.fleet_wake._recovery_ready)
    return console, controller, bridge, gateway, store, session


async def _saved_pause(bridge, controller, session, reason):
    cid = session.persisted_conversation_id
    db = bridge.runs_db
    parent, child = _terminal_subagent_run(
        db, cid, result="Saved research result", work_chain_id=None
    )
    chain = None
    if reason != "legacy_lineage":
        chain = db.automatic_work.create_chain(cid, root_submission_id="original-work")
        db.automatic_work.attach_run(parent, chain)
        db.automatic_work.attach_run(child, chain)
        db.automatic_work.pause(
            chain, reason, review_required=reason != "generation_budget"
        )
    await controller.fleet_wake.recover()
    assert controller.fleet_wake.pause_reason(cid) == reason
    assert controller.fleet_wake.has_pending(cid)
    return cid, chain, child


@pytest.mark.parametrize(
    ("reason", "explanation", "next_step"),
    [
        ("generation_budget", "turn limit", "Send a message to continue"),
        ("model_call_budget", "call limit", "Send a message to continue"),
        ("output_tokens_budget", "reply limit", "Send a message to continue"),
        ("autowake_disabled", "auto off", "Send a message to continue"),
        ("interrupted_work", "work may have run", "Review Run history"),
        ("legacy_lineage", "earlier work", "Review Run history"),
        ("history_unavailable", "history error", "Retry Run history"),
    ],
)
async def test_pause_copy_is_painted_with_no_live_children_and_survives_view_clear(
    tmp_path, reason, explanation, next_step
):
    app = _app(tmp_path)
    host = _PauseHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console, controller, bridge, gateway, _store, session = await _rig(host, pilot)
        cid, _chain, child = await _saved_pause(bridge, controller, session, reason)
        assert bridge.fleet_snapshot(cid) == []
        console._set_console_rail_preference(
            section_updates={"agent": True}, notify_on_failure=False
        )
        console._sync_console_agent_section()
        await pilot.pause()
        section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        section.set_open(False)
        await _scroll_into_view(
            pilot, console, "#console-inspector-section-agent-fleet-toggle"
        )
        await pilot.click("#console-inspector-section-agent-fleet-toggle")
        await pilot.pause()

        painted = _painted_notice(host, console)
        assert "Paused:" in painted, (
            console._agent._console_agent_fleet_section_state(),
            painted,
        )
        assert explanation in painted
        assert "Results saved" in painted
        assert next_step in painted
        assert gateway.payloads == []
        assert bridge.runs_db.get_run(child)["wake_delivered_at"] is None

        await console._sync_console_native_session_tabs()
        assert cid not in console._fleet._console_fleet_unseen_ids()
        marker = console._fleet._console_run_marker_with_unseen(
            controller, session, console._fleet._console_fleet_unseen_ids()
        )
        assert marker is ConsoleRunMarker.NONE
        assert controller.fleet_wake.pause_reason(cid) == reason
        assert controller.fleet_wake.has_pending(cid)
        assert bridge.runs_db.get_run(child)["result"] == "Saved research result"
        assert bridge.runs_db.get_run(child)["wake_delivered_at"] is None
        assert _painted_notice(host, console) == painted


async def test_run_history_and_manual_send_leave_old_paused_chain_unchanged(tmp_path):
    app = _app(tmp_path)
    host = _PauseHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console, controller, bridge, gateway, store, session = await _rig(host, pilot)
        cid, chain, child = await _saved_pause(
            bridge, controller, session, "generation_budget"
        )
        before = bridge.runs_db.automatic_work.snapshot(chain)
        console._set_console_rail_preference(
            section_updates={"agent": True}, notify_on_failure=False
        )
        console._sync_console_agent_section()
        await pilot.pause()
        await _scroll_into_view(
            pilot, console, "#console-inspector-section-agent-fleet-view-all"
        )
        await pilot.click("#console-inspector-section-agent-fleet-view-all")
        assert await _settle(
            pilot, lambda: isinstance(host.screen, ConsoleAgentHistoryModal)
        )
        assert await _settle(
            pilot, lambda: host.screen.query_one(DataTable).row_count == 1
        )
        await pilot.press("enter")
        assert await _settle(pilot, lambda: host.screen is console)
        assert console._agent._console_agent_drilldown_run_id == child
        assert bridge.runs_db.get_run(child)["result"] == "Saved research result"
        assert bridge.runs_db.automatic_work.snapshot(chain) == before

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("Continue using the saved research result")
        composer.focus()
        await pilot.press("enter")
        assert await _settle(pilot, lambda: bool(gateway.payloads)), (
            type(host.screen).__name__,
            controller.run_state_for(session.id),
            _rendered_text(host),
        )
        assert await _settle(pilot, lambda: controller.in_flight_run_count() == 0)
        assert any(
            message.content == "Manual continuation completed"
            for message in store.messages_for_session(session.id)
        )
        assert len(gateway.payloads) == 1
        assert bridge.runs_db.automatic_work.snapshot(chain) == before
        assert bridge.runs_db.get_run(child)["wake_delivered_at"] is None
        assert controller.fleet_wake.has_pending(cid)
        with bridge.runs_db.connection() as conn:
            chains = [
                dict(row)
                for row in conn.execute(
                    "SELECT id, status FROM automatic_work_chains WHERE conversation_id = ?",
                    (cid,),
                )
            ]
        assert len(chains) == 2
        assert next(row for row in chains if row["id"] != chain)["status"] == "active"


async def test_mount_seed_discovers_saved_results_without_badges(tmp_path):
    app = _app(tmp_path)
    host = _PauseHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console, controller, bridge, _gateway, _store, session = await _rig(host, pilot)
        cid = session.persisted_conversation_id
        _terminal_subagent_run(bridge.runs_db, cid, work_chain_id=None)
        clear_fleet_unseen_completion(app, cid)
        assert not controller.fleet_wake.has_pending(cid)
        console._fleet._claim_console_fleet_wake_marks()
        assert await _settle(pilot, lambda: controller.fleet_wake.has_pending(cid))


async def test_both_concurrent_wake_sessions_are_recognized_as_active(tmp_path):
    app = _app(tmp_path)
    host = _PauseHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console, controller, bridge, gateway, store, first = await _rig(host, pilot)
        second = store.create_session(title="Other work", settings=first.settings)
        store.set_session_project_instruction_state(
            second.id, ProjectInstructionControlState.legacy_disabled()
        )
        store.append_message(
            second.id, role=ConsoleMessageRole.USER, content="Other topic", persist=True
        )
        gate = threading.Event()
        gateway.thread_gate = gate
        try:
            for session in (first, second):
                cid = session.persisted_conversation_id or session.id
                chain = bridge.runs_db.automatic_work.create_chain(
                    cid, root_submission_id=session.id
                )
                _parent, child = _terminal_subagent_run(
                    bridge.runs_db, cid, work_chain_id=chain
                )
                controller.fleet_wake.on_fleet_drained(
                    _drain(cid, _survivor(child, session_id=session.id))
                )
            assert await _settle(
                pilot, lambda: len(controller.fleet_wake.delivering_session_ids()) == 2
            ), (
                gateway.payloads,
                controller.fleet_wake._paused,
                controller.fleet_wake.pending_conversation_ids(),
            )
            assert console._fleet._console_wake_turn_active(first.id)
            assert console._fleet._console_wake_turn_active(second.id)
        finally:
            gate.set()
            await _settle(
                pilot, lambda: not controller.fleet_wake.delivering_session_ids()
            )


async def test_legacy_completion_intake_repaints_the_pause_without_recovery(tmp_path):
    app = _app(tmp_path)
    host = _PauseHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console, controller, bridge, gateway, _store, session = await _rig(host, pilot)
        cid = session.persisted_conversation_id
        console._set_console_rail_preference(
            section_updates={"agent": True}, notify_on_failure=False
        )
        console._sync_console_agent_section()
        _parent, child = _terminal_subagent_run(bridge.runs_db, cid, work_chain_id=None)
        controller.fleet_wake.on_fleet_drained(
            _drain(cid, _survivor(child, session_id=session.id))
        )
        assert await _settle(
            pilot,
            lambda: bool(
                console.query("#console-inspector-section-agent-fleet-notice")
            ),
        )
        assert await _settle(
            pilot, lambda: not controller.fleet_wake.delivering_session_ids()
        )
        section = console.query_one(
            "#console-agent-section-subagents", ConsoleInspectorSection
        )
        section.set_open(False)
        await pilot.pause()
        await _scroll_into_view(
            pilot, console, "#console-inspector-section-agent-fleet-toggle"
        )
        assert await pilot.click("#console-inspector-section-agent-fleet-toggle")
        await pilot.pause()
        assert section.open
        painted = _painted_notice(host, console)
        assert "Paused: earlier work" in painted
        assert "Results saved" in painted
        assert "Review Run history" in painted
        assert controller.fleet_wake.pause_reason(cid) == "legacy_lineage"
        assert controller.fleet_wake.has_pending(cid)
        assert gateway.payloads == []
        assert bridge.runs_db.get_run(child)["wake_delivered_at"] is None


async def test_wrapped_notice_updates_in_place_and_clears_with_the_section_state():
    rows = _rows(2)
    section = ConsoleInspectorSection(
        title="Agents",
        section_id="agents",
        rows=rows,
        summary="Paused",
        notice="Paused: turn limit. Results saved. Send a message to continue.",
        max_visible_rows=1,
    )
    host = _SectionHarness(section)
    async with host.run_test(size=(35, 16)) as pilot:
        await pilot.pause()
        notice = section.query_one("#console-inspector-section-agents-notice")
        first_row = section.query_one("#console-inspector-section-agents-row-0")
        assert "Results saved." in _rendered_text(host)
        assert len(section.rows) == 2
        assert not section.query("#console-inspector-section-agents-row-1")
        count = section.recompose_count
        section.sync_state(
            ConsoleInspectorSectionState(
                rows=rows,
                summary="Paused",
                notice="Paused: usage unclear. Results saved. Review Run history.",
            )
        )
        await pilot.pause()
        assert section.query_one("#console-inspector-section-agents-notice") is notice
        assert section.query_one("#console-inspector-section-agents-row-0") is first_row
        assert section.recompose_count == count
        assert "usage unclear" in _rendered_text(host)
        section.sync_state(ConsoleInspectorSectionState(rows=rows, summary="2 done"))
        await pilot.pause()
        assert not section.query("#console-inspector-section-agents-notice")
        assert "Results saved." not in _rendered_text(host)
        assert section.rows == rows


async def test_unreadable_history_does_not_claim_unverified_results_are_saved(tmp_path):
    app = _app(tmp_path)
    host = _PauseHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console, controller, bridge, gateway, _store, session = await _rig(host, pilot)
        cid = session.persisted_conversation_id
        assert not controller.fleet_wake.has_pending(cid)
        with bridge.runs_db.connection() as conn:
            conn.execute(
                "ALTER TABLE automatic_wake_attempts RENAME TO unavailable_attempts"
            )
        try:
            await controller.fleet_wake.recover()
            assert controller.fleet_wake.pause_reason(cid) == "history_unavailable"
            console._set_console_rail_preference(
                section_updates={"agent": True}, notify_on_failure=False
            )
            console._sync_console_agent_section()
            await pilot.pause()
            section = console.query_one(
                "#console-agent-section-subagents", ConsoleInspectorSection
            )
            section.set_open(False)
            await pilot.pause()
            await _scroll_into_view(
                pilot, console, "#console-inspector-section-agent-fleet-toggle"
            )
            assert await pilot.click("#console-inspector-section-agent-fleet-toggle")
            await pilot.pause()
            painted = _painted_notice(host, console)
            assert "Paused: history error" in painted
            assert "Retry Run history" in painted
            assert "Results saved" not in painted
            assert not controller.fleet_wake.has_pending(cid)
            assert gateway.payloads == []
        finally:
            with bridge.runs_db.connection() as conn:
                conn.execute(
                    "ALTER TABLE unavailable_attempts RENAME TO automatic_wake_attempts"
                )
