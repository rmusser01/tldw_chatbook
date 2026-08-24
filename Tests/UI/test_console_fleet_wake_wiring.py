"""PR 3a-2 Task 5: the screen half of auto-wake.

The coordinator's behaviour (gating, coalescing, exactly-once, the kill
switch) is pinned at the controller level in
``Tests/Chat/test_console_fleet_wake.py``; THIS file pins what only the
real ``ChatScreen`` can break:

1. **the mount ordering hazard** (Task 4's stated trap): the wake
   mount-claim reads the durable marks BEFORE the first
   ``_sync_console_native_session_tabs`` can view-clear the ACTIVE
   conversation's mark -- claim-before-sync is asserted on a real mount,
   by recorded call order, not by reading the source;
2. **the mount-claim seam does real work**: a marked conversation whose
   session is open and whose terminal survivor run sits in the real
   bridge's ``agent_runs`` DB seeds the coordinator's pending set;
3. **user-wins-ties wiring**: the probe the controller consults is the
   screen's composer-draft read, live against the real composer widget;
4. **the composer-empty poke**: clearing the draft through the composer's
   own mutation path retries a deferred wake.
"""

from __future__ import annotations

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_panel import _AGENT_SECTION_SIZE
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.fleet import (
    ConsoleFleetLifecycleController,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _attach_real_dbs(app, tmp_path):
    """Real marks service AND a real on-disk ChaChaNotes DB handle, so the
    screen's lazy agent-bridge construction (which keys the sibling
    ``agent_runs.db`` off ``chachanotes_db.db_path``) actually builds."""
    db = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="ui-test")
    app.conversation_local_marks_service = ConversationLocalMarksService(db)
    app.chachanotes_db = db
    return app.conversation_local_marks_service


@pytest.mark.asyncio
async def test_mount_claims_wake_marks_before_the_first_tab_sync(tmp_path):
    """The ordering the whole staged-wake design leans on: the first tab
    sync view-clears the ACTIVE conversation's mark, so a claim running
    after it reads nothing. Recorded on a REAL mount of the real screen
    class."""
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    order: list[str] = []
    real_claim = ConsoleFleetLifecycleController._claim_console_fleet_wake_marks
    real_sync = ChatScreen._sync_console_native_session_tabs

    def recording_claim(self):
        order.append("claim")
        return real_claim(self)

    async def recording_sync(self):
        order.append("sync")
        return await real_sync(self)

    ConsoleFleetLifecycleController._claim_console_fleet_wake_marks = recording_claim
    ChatScreen._sync_console_native_session_tabs = recording_sync
    try:
        host = ConsoleHarness(app)
        async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
            console = host.screen_stack[-1]
            from Tests.UI.test_destination_shells import _wait_for_selector

            await _wait_for_selector(console, pilot, "#console-session-surface")
            await pilot.pause()
    finally:
        ConsoleFleetLifecycleController._claim_console_fleet_wake_marks = real_claim
        ChatScreen._sync_console_native_session_tabs = real_sync
    assert "claim" in order, "the mount never ran the wake mark claim"
    assert "sync" in order, (
        "harness never reached a tab sync; the ordering was not exercised"
    )
    assert order.index("claim") < order.index("sync"), (
        "the wake mark claim must run BEFORE the first tab sync's "
        f"view-clear can consume the active conversation's mark: {order}"
    )


@pytest.mark.asyncio
async def test_the_mount_claim_seeds_pending_from_mark_and_runs_db(tmp_path):
    """The claim seam, driven with the exact durable state a
    settled-while-unmounted survivor leaves behind: the FLEET_UNSEEN mark
    plus a terminal subagent row in the real bridge's runs DB. The
    coordinator's pending set gains the conversation (delivery itself is
    gated behind provider readiness and is pinned controller-level)."""
    app = _build_test_app()
    marks = _attach_real_dbs(app, tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        from Tests.UI.test_destination_shells import _wait_for_selector

        await _wait_for_selector(console, pilot, "#console-session-surface")
        controller = console._ensure_console_chat_controller()
        bridge = console._ensure_console_agent_bridge()
        assert bridge is not None, (
            "harness must build the real bridge (chachanotes_db path wired)"
        )
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        runs_db = bridge.runs_db
        parent_id = runs_db.create_run(conversation_id=session.id, agent_kind="primary")
        runs_db.set_status(parent_id, "done", "turn final")
        child_id = runs_db.create_run(
            conversation_id=session.id,
            agent_kind="subagent",
            task="long job",
            parent_run_id=parent_id,
        )
        marks.set_mark(session.id, ConversationLocalMarksService.FLEET_UNSEEN)
        runs_db.set_status(child_id, "done", "staged answer")

        console._fleet._claim_console_fleet_wake_marks()
        assert controller.fleet_wake.has_pending(session.id), (
            "the mount-claim seam must turn mark + runs-DB state into a pending wake"
        )


@pytest.mark.asyncio
async def test_user_priority_probe_reads_the_live_composer_draft(tmp_path):
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        from Tests.UI.test_destination_shells import _wait_for_selector
        from tldw_chatbook.Widgets.Console.console_composer_bar import (
            ConsoleComposerBar,
        )

        await _wait_for_selector(console, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        probe = controller.wake_user_priority_probe
        assert callable(probe), "the screen must wire the user-wins-ties probe"
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        session_id = console._ensure_console_chat_store().ensure_session().id
        assert probe(session_id) is False, "an empty composer holds no claim"
        composer.load_draft("half-typed message")
        await pilot.pause()
        assert probe(session_id) is True, (
            "a non-empty draft is the user's sending claim; the wake must see it"
        )
        composer.load_draft("")
        await pilot.pause()
        assert probe(session_id) is False


@pytest.mark.asyncio
async def test_emptying_the_composer_pokes_the_wake_retry(tmp_path):
    """The deferral's exit: the moment the user's draft claim ends --
    through the composer's own every-mutation change signal, so paste/
    clear/load_draft count, not just keystrokes -- the deferred wake gets
    its retry."""
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    host = ConsoleHarness(app)
    async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
        console = host.screen_stack[-1]
        from Tests.UI.test_destination_shells import _wait_for_selector
        from tldw_chatbook.Widgets.Console.console_composer_bar import (
            ConsoleComposerBar,
        )

        await _wait_for_selector(console, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        pokes: list[str] = []
        controller.fleet_wake.retry_soon = lambda: pokes.append("poke")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("about to change my mind")
        await pilot.pause()
        emptied_before = len(pokes)
        composer.load_draft("")
        await pilot.pause()
        assert len(pokes) > emptied_before, (
            "clearing the composer must retry a deferred wake"
        )
