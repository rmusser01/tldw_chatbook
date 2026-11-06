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


# The helper owns precisely the on-disk DB it injects, including worker handles.
_attached_real_databases = []


async def drain_attached_real_runtimes() -> None:
    """Retire the runtimes borrowing the helper's injected databases."""
    from tldw_chatbook.Chat.console_runtime import dispose_console_runtime

    for app, _database in _attached_real_databases:
        await dispose_console_runtime(app)


def drain_attached_real_databases() -> None:
    """Quiesce precisely the helper-owned DBs, after any runtime retirement."""
    from Tests.conftest import _close_database_instance

    while _attached_real_databases:
        _app, database = _attached_real_databases[-1]
        _close_database_instance(database)
        _attached_real_databases.pop()


def _attach_real_dbs(app, tmp_path):
    """Real marks service AND a real on-disk ChaChaNotes DB handle, so the
    screen's lazy agent-bridge construction (which keys the sibling
    ``agent_runs.db`` off ``chachanotes_db.db_path``) actually builds."""
    db = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="ui-test")
    _attached_real_databases.append((app, db))
    app.conversation_local_marks_service = ConversationLocalMarksService(db)
    app.chachanotes_db = db
    # ConsoleHarness bypasses application startup and mounts its ready screen.
    app._ui_ready = True
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    app.local_chat_conversation_service = ChatConversationService(db)
    from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
    app.local_chat_dictionary_service = LocalChatDictionaryService(db)
    if getattr(app, "chat_dictionary_scope_service", None) is not None:
        app.chat_dictionary_scope_service.local_service = app.local_chat_dictionary_service
    return app.conversation_local_marks_service


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_no_tab_sync_before_the_mount_claim_consumes_the_active_chats_wake_mark(
    tmp_path,
):
    """The hazard the staged-wake design leans on: a tab sync view-clears the
    ACTIVE conversation's mark, so a claim reading after it finds nothing.

    Since 3e618d7119 the readiness projection publishes a whole sync from a
    worker that can run before ``ChatScreen.on_mount``, so tab syncs now run
    before the claim in about half the mounts, and a strict claim-first
    order is no longer what holds. What must hold is the outcome: a relaunch
    with a staged wake on the active chat keeps its mark through every sync
    before the claim, and the claim seeds the wake (TASK-39044). The test
    runs one real tab sync before the claim on every mount, so the hazard is
    always exercised. Recorded on a REAL mount of the real screen class.
    """
    from pathlib import Path

    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_runtime import ensure_console_runtime
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    app = _build_test_app()
    marks = _attach_real_dbs(app, tmp_path)
    store = ensure_console_runtime(app).ensure_chat_store()
    session = store.ensure_session()
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="Research", persist=True
    )
    cid = session.persisted_conversation_id
    assert cid
    runs = AgentRunsDB(Path(app.chachanotes_db.db_path).parent / "agent_runs.db")
    try:
        parent = runs.create_run(conversation_id=cid, agent_kind="primary")
        runs.set_status(parent, "done", "turn final")
        child = runs.create_run(
            conversation_id=cid,
            agent_kind="subagent",
            task="long job",
            parent_run_id=parent,
        )
        marks.set_mark(cid, ConversationLocalMarksService.FLEET_UNSEEN)
        runs.set_status(child, "done", "staged answer")
    finally:
        runs.close()

    def marked() -> bool:
        return marks.has_mark(cid, ConversationLocalMarksService.FLEET_UNSEEN)

    import asyncio

    order: list[tuple[str, bool]] = []
    forced: list[asyncio.Task] = []
    real_claim = ConsoleFleetLifecycleController._claim_console_fleet_wake_marks
    real_sync = ChatScreen._sync_console_native_session_tabs

    async def recording_sync(self):
        order.append(("sync", marked()))
        result = await real_sync(self)
        order.append(("synced", marked()))
        return result

    def recording_claim(self):
        if forced:
            order.append(("claim", marked()))
            return real_claim(self)
        # Every mount meets the hazard: one real tab sync runs before the
        # mount's claim reads the marks, not only the mounts where a
        # readiness worker happens to win.
        screen = host.screen

        async def sync_then_claim():
            await recording_sync(screen)
            order.append(("claim", marked()))
            real_claim(self)

        forced.append(asyncio.get_running_loop().create_task(sync_then_claim()))
        return None

    host = ConsoleHarness(app)
    ConsoleFleetLifecycleController._claim_console_fleet_wake_marks = recording_claim
    ChatScreen._sync_console_native_session_tabs = recording_sync
    try:
        async with host.run_test(size=_AGENT_SECTION_SIZE) as pilot:
            console = host.screen_stack[-1]
            from Tests.UI.test_console_fleet_wake_ui_freshness import _settle
            from Tests.UI.test_destination_shells import _wait_for_selector

            await _wait_for_selector(console, pilot, "#console-session-surface")
            assert forced and await _settle(pilot, forced[0].done), (
                "the forced pre-claim sync never finished"
            )
            forced[0].result()
            wake = console._ensure_console_chat_controller().fleet_wake
            assert await _settle(
                pilot,
                lambda: wake.has_pending(cid) or bool(wake.delivering_conversation_ids()),
            ), f"the staged wake on the active chat was never seeded: {order}"
    finally:
        ConsoleFleetLifecycleController._claim_console_fleet_wake_marks = real_claim
        ChatScreen._sync_console_native_session_tabs = real_sync
    names = [name for name, _ in order]
    assert "claim" in names, "the mount never ran the wake mark claim"
    assert "sync" in names, (
        "harness never reached a tab sync; the hazard was not exercised"
    )
    before_claim = order[: names.index("claim") + 1]
    assert all(seen for _, seen in before_claim), (
        "a tab sync consumed the active conversation's wake mark before the "
        f"mount claim read it: {order}"
    )


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
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
        from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
        store.append_message(session.id, role=ConsoleMessageRole.USER, content="Research", persist=True)
        cid = session.persisted_conversation_id
        assert cid
        runs_db = bridge.runs_db
        parent_id = runs_db.create_run(conversation_id=cid, agent_kind="primary")
        runs_db.set_status(parent_id, "done", "turn final")
        child_id = runs_db.create_run(
            conversation_id=cid,
            agent_kind="subagent",
            task="long job",
            parent_run_id=parent_id,
        )
        marks.set_mark(cid, ConversationLocalMarksService.FLEET_UNSEEN)
        runs_db.set_status(child_id, "done", "staged answer")

        console._fleet._claim_console_fleet_wake_marks()
        from Tests.UI.test_console_fleet_wake_ui_freshness import _settle
        assert await _settle(pilot, lambda: controller.fleet_wake.has_pending(cid)), (
            "the mount-claim seam must turn mark + runs-DB state into a pending wake"
        )


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
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
@pytest.mark.bootstrap_profile
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
