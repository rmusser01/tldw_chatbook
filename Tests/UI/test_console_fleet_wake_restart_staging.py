"""task-15864: restart-staged wakes must be visible and deliverable.

PR3a-2 Task 7's live restart (scenario 5) found the durable layer intact
-- mark and owed ledger survived SIGKILL -- while the STAGING around it
failed twice, plus one recorded window needing a ruling:

1. (AC#1) the marked conversation's sidebar row rendered NO ◈: the
   unseen-badge derivation lived only on the open-session (native) row
   path, and session tabs do not restore across restart, so the
   no-open-session persisted/membership row is the NORMAL restart shape;
2. (AC#2) opening the marked conversation created its session but did
   NOT deliver -- session-open was missing from the wake retry-trigger
   list, so the owed wake sat pending until an unrelated composer
   keystroke;
3. (AC#3) a wake deferred while its conversation is VIEWED view-cleared
   the mark while the ledger still owed -- restart in that window leaves
   an owed, UNMARKED run, and the mount-claim is marks-indexed
   (``seed_from_marks``: the mark names WHICH conversations to claim),
   so it never seeds. Verified here as a limit of the marks-indexed
   claim; fixed by making the view-clear YIELD while the coordinator
   still owes the conversation, so the mark -- the restart staging bit --
   survives every deferral window.
"""

from __future__ import annotations

import time

import pytest

from Tests.Chat.test_console_fleet_wake import (
    _RecordingWakeGateway,
    _controller_rig,
    _drain,
    _survivor,
    _terminal_subagent_run,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_panel import _AGENT_SECTION_SIZE
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import (
    SearchableConversationService,
    _click_console_workspace_conversation_for_id,
    _configure_grouped_browser_workspaces,
    _configure_native_ready_console,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_fleet_attention import bump_fleet_unseen_revision
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.config import load_settings, save_setting_to_cli_config


async def _settle(pilot, predicate, seconds: float = 8.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return bool(predicate())


# ---------------------------------------------------------------------------
# AC#1: the restart shape -- a marked conversation with NO open session
# must still show the ◈ on its sidebar browser row.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unseen_mark_reaches_membership_rows_without_a_session(tmp_path):
    """A marked workspace conversation with no native session (the normal
    restart shape -- session tabs do not restore) must carry the ◈ on its
    membership browser row, exactly as an open session's tab does."""
    app = _build_test_app()
    marks = _attach_real_dbs(app, tmp_path)
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-marked",
        role="workspace-thread",
        title="Marked while away",
    )
    marks.set_mark("conv-marked", ConversationLocalMarksService.FLEET_UNSEEN)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        from Tests.UI.test_destination_shells import _wait_for_selector

        await _wait_for_selector(console, pilot, "#console-session-surface")
        bump_fleet_unseen_revision(app)
        rows = console._workspace._membership_console_browser_rows()
        by_conversation = {row.conversation_id: row for row in rows}
        assert "conv-marked" in by_conversation, f"membership rows: {rows!r}"
        assert by_conversation["conv-marked"].run_marker == "◈", (
            "task-15864 AC#1: a marked conversation with no open session "
            "must show the unseen badge on its sidebar row "
            f"(got {by_conversation['conv-marked'].run_marker!r})"
        )


@pytest.mark.asyncio
async def test_unseen_mark_reaches_persisted_rows_without_a_session(tmp_path):
    """Same restart shape through the persisted-conversation listing path
    (the grouped browser's saved-conversation rows)."""

    class _PersistedListingService:
        async def list_conversations(self, **kwargs):
            if kwargs.get("scope_type") == "global":
                return {
                    "items": [{"id": "conv-marked", "title": "Marked saved"}],
                    "total": 1,
                }
            return {"items": [], "total": 0}

    app = _build_test_app()
    marks = _attach_real_dbs(app, tmp_path)
    app.chat_conversation_scope_service = _PersistedListingService()
    marks.set_mark("conv-marked", ConversationLocalMarksService.FLEET_UNSEEN)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        from Tests.UI.test_destination_shells import _wait_for_selector

        await _wait_for_selector(console, pilot, "#console-session-surface")
        bump_fleet_unseen_revision(app)
        rows, _total, error = await console._workspace._persisted_console_browser_rows(
            ""
        )
        assert not error
        by_conversation = {row.conversation_id: row for row in rows}
        assert "conv-marked" in by_conversation, f"persisted rows: {rows!r}"
        assert by_conversation["conv-marked"].run_marker == "◈", (
            "task-15864 AC#1: a marked persisted conversation with no open "
            "session must show the unseen badge on its sidebar row "
            f"(got {by_conversation['conv-marked'].run_marker!r})"
        )


# ---------------------------------------------------------------------------
# AC#2: opening the marked conversation IS a retry trigger.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opening_a_marked_conversation_delivers_without_a_keystroke(
    tmp_path,
):
    """The live scenario-5 gap, end to end: durable mark + owed ledger
    survive 'restart' (a fresh mount over the same DBs); the mount-claim
    seeds pending; opening the marked conversation from the sidebar
    creates its session -- and that alone must deliver the owed wake.
    Live, the wake sat pending until an unrelated composer keystroke."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    assert save_setting_to_cli_config("console", "agent_runtime", False)
    assert save_setting_to_cli_config(
        "console.conversation_browser", "expanded_workspace_ids", ["ws-a"]
    )
    app.app_config = load_settings()
    marks = _attach_real_dbs(app, tmp_path)
    # The conversation genuinely exists in ChaChaNotes (production shape:
    # it persisted before the restart) so the wake's SYSTEM notice row can
    # persist into it.
    app.chachanotes_db.add_conversation(
        {"id": "conv-marked", "title": "Marked while away"}
    )
    app.chachanotes_db.add_message(
        {
            "id": "message-1",
            "conversation_id": "conv-marked",
            "sender": "user",
            "content": "please research this in the background",
        }
    )
    service = _configure_grouped_browser_workspaces(app)
    service.link_membership(
        "ws-a",
        item_type="conversation",
        item_id="conv-marked",
        role="workspace-thread",
        title="Marked while away",
    )
    app.chat_conversation_scope_service = SearchableConversationService(
        {
            "conv-marked": {
                "conversation": {
                    "id": "conv-marked",
                    "title": "Marked while away",
                    "scope_type": "workspace",
                    "workspace_id": "ws-a",
                },
                "root_threads": [
                    {
                        "id": "message-1",
                        "role": "user",
                        "content": "please research this in the background",
                    }
                ],
            },
        }
    )
    # The durable state a settled-while-away survivor leaves behind, seeded
    # into the SAME sibling runs DB the screen's bridge will open.
    runs_db = AgentRunsDB(tmp_path / "agent_runs.db", client_id="seed")
    _parent, run_id = _terminal_subagent_run(
        runs_db, "conv-marked", result="staged background answer"
    )
    runs_db.close()
    marks.set_mark("conv-marked", ConversationLocalMarksService.FLEET_UNSEEN)

    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        from Tests.UI.test_destination_shells import _wait_for_selector

        await _wait_for_selector(console, pilot, "#console-session-surface")
        controller = console._ensure_console_chat_controller()
        gateway = _RecordingWakeGateway(reply="acting on the staged result")
        controller.provider_gateway = gateway
        assert controller.fleet_wake.has_pending("conv-marked"), (
            "precondition: the mount-claim must have seeded the owed wake "
            "from mark + ledger"
        )

        await _click_console_workspace_conversation_for_id(
            console, pilot, "conv-marked"
        )
        delivered = await _settle(pilot, lambda: gateway.payloads)
        assert delivered, (
            "task-15864 AC#2: opening the marked conversation must deliver "
            "the staged wake -- live it waited for a composer keystroke"
        )
        assert len(gateway.payloads) == 1
        assert "staged background answer" in gateway.payloads[0][-1]["content"]
        bridge = console._ensure_console_agent_bridge()
        stamped = await _settle(
            pilot,
            lambda: bool(
                (bridge.runs_db.get_run(run_id) or {}).get("wake_delivered_at")
            ),
        )
        assert stamped, "the delivered wake must stamp the per-run ledger"


# ---------------------------------------------------------------------------
# AC#3: the owed-but-unmarked window.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_view_clear_yields_while_a_wake_is_still_owed(tmp_path):
    """The window's entry point, closed: viewing a conversation whose wake
    is deferred (owed in the coordinator's pending set) must NOT view-clear
    the durable mark -- the mark is the restart staging bit the
    marks-indexed mount-claim depends on. Once nothing is owed, viewing
    clears it exactly as before (Task 4's behaviour, preserved)."""
    app = _build_test_app()
    marks = _attach_real_dbs(app, tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        from Tests.UI.test_destination_shells import _wait_for_selector

        await _wait_for_selector(console, pilot, "#console-session-surface")
        controller = console._ensure_console_chat_controller()
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        wake = controller.fleet_wake
        # The live shape: a due wake deferred while its conversation is
        # viewed (scenario 4's draft deferral). The probe holds it.
        controller.wake_user_priority_probe = lambda sid: True
        wake.on_fleet_drained(
            _drain(session.id, _survivor("run-owed", session_id=session.id))
        )
        marks.set_mark(session.id, ConversationLocalMarksService.FLEET_UNSEEN)
        bump_fleet_unseen_revision(app)

        await console._sync_console_native_session_tabs()
        await pilot.pause()
        assert wake.has_pending(session.id), "precondition: the wake is owed"
        assert marks.has_mark(session.id, ConversationLocalMarksService.FLEET_UNSEEN), (
            "task-15864 AC#3: viewing must not clear the mark while the "
            "wake is still owed -- a restart in that window leaves an owed, "
            "unmarked run the marks-indexed mount-claim never seeds"
        )

        # Nothing owed anymore (the delivery committed): viewing clears.
        with wake._registry_lock:
            wake._pending.pop(session.id, None)
        await console._sync_console_native_session_tabs()
        await pilot.pause()
        assert not marks.has_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        ), "with nothing owed, viewing IS the clear (Task 4, preserved)"


@pytest.mark.asyncio
async def test_marks_indexed_mount_claim_alone_misses_an_unmarked_owed_run(
    tmp_path,
):
    """The verified limit the AC#3 ruling records: ``seed_from_marks`` is
    marks-INDEXED -- the ledger defines WHAT is owed, but only for
    conversations the mark names. An owed, unmarked run seeds nothing.
    This is why the fix above keeps the mark alive through every deferral
    window instead of trying to claim from the ledger globally (which
    would also sweep in restart-orphans the corrected spec §3 deliberately
    leaves to next-turn handling)."""
    chacha, app, runs_db, store, session, gateway, bridge, controller = _controller_rig(
        tmp_path
    )
    try:
        _parent, _run_id = _terminal_subagent_run(runs_db, "conv-unmarked")
        assert controller.fleet_wake.seed_from_marks() == 0, (
            "documented limit: an owed but UNMARKED conversation is "
            "invisible to the marks-indexed mount-claim"
        )
        assert not controller.fleet_wake.has_pending("conv-unmarked")
    finally:
        chacha.close()
