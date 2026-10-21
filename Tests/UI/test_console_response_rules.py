"""Native composer commands over the actual Console screen."""

import asyncio

import pytest
from textual.widgets import Button

from Tests.UI.response_rules_fixtures import mounted_rules_console, seed_answer
from Tests.UI.test_console_native_chat_flow import (
    _build_console_send_test_app,
    _configure_native_ready_console,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_command_grammar import default_console_registry
from tldw_chatbook.Chat.console_command_suggestions import _COMMAND_DESCRIPTIONS
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleStatusChips,
    ConsoleStatusText,
)
from tldw_chatbook.Widgets.Console.response_rules_modal import ResponseRulesModal

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


# TASK-35160: the run chip is the status line's "run" segment; these read
# the strip's public seam where the chip widget used to be queried.
def _run_visible(console) -> bool:
    return console.query_one(
        "#console-status-chips", ConsoleStatusChips
    ).segment_visible("run")


def _run_text(console) -> str:
    return console.query_one(
        "#console-status-chips", ConsoleStatusChips
    ).segment_text("run")



@pytest.mark.parametrize(
    "name,handler",
    [("omfg", "learn-response-rule"), ("rules", "manage-response-rules")],
)
def test_commands_are_registered_once_and_have_help(name, handler):
    registry = default_console_registry()
    assert registry.parse(f"/{name} problem").kind == "command"
    commands = [c for c in registry.commands() if c.name == name]
    assert len(commands) == 1 and commands[0].handler_id == handler
    assert _COMMAND_DESCRIPTIONS[name]


@pytest.mark.asyncio
async def test_actual_composer_refusal_preserves_draft_and_rules_is_native(tmp_path):
    app = _build_console_send_test_app()
    from Tests.conftest import _close_database_instance
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    _close_database_instance(app.chachanotes_db)
    app.chachanotes_db = CharactersRAGDB(tmp_path / "ui-rules.sqlite", "ui-rules")
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        gateway = console._ensure_console_chat_controller().provider_gateway
        calls = []

        async def ordinary(*args, **kwargs):
            calls.append(args)
            yield "unexpected ordinary response"

        gateway.stream_chat = ordinary
        for draft in ("/omfg", "/omfg Include evidence"):
            composer.load_draft(draft)
            console.query_one("#console-send-message", Button).press()
            for _ in range(12):
                await pilot.pause()
            assert composer.draft_text() == draft
            assert calls == []
        composer.load_draft("/rules")
        console.query_one("#console-send-message", Button).press()
        for _ in range(12):
            await pilot.pause()
            if isinstance(host.screen, ResponseRulesModal):
                break
        assert isinstance(host.screen, ResponseRulesModal)
        assert host.screen.scope.kind == "chat"
        assert calls == []
        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()
            if host.screen is console:
                break
        assert host.screen is console
        assert host.focused is composer


@pytest.mark.asyncio
async def test_actual_stop_after_generation_cancels_rule_check(tmp_path):
    async with mounted_rules_console(tmp_path) as case:
        seed_answer(case)
        learned = await case.rules.learn(case.session_id, "The answer omitted evidence")
        assert learned.state == "active", (learned.reason, case.failures)
        entered, release = asyncio.Event(), asyncio.Event()
        assess = case.rules.evaluator.assess

        async def held(*args, **kwargs):
            entered.set()
            await release.wait()
            return await assess(*args, **kwargs)

        case.rules.evaluator.assess = held
        case.composer.load_draft("Explain another result")
        case.console.query_one("#console-send-message", Button).press()
        for _ in range(60):
            await case.pilot.pause()
            if entered.is_set():
                break
        assert entered.is_set(), case.controller.run_state
        before = len(case.requests)
        assert (
            case.chats.get_message(case.chats.active_leaf(case.session_id)).status
            == "complete"
        )
        assert await case.pilot.click("#console-stop-generation")
        for _ in range(10):
            await case.pilot.pause()
        release.set()
        assert case.rules.state(case.session_id).phase == "idle"
        assert len(case.requests) == before
        assert case.host.focused is case.composer


@pytest.mark.asyncio
async def test_workspace_command_opens_same_manager_in_registered_scope(tmp_path):
    async with mounted_rules_console(tmp_path) as case:
        workspace = case.app.workspace_registry_service.create_workspace(
            workspace_id="rules-workspace",
            name="Rules workspace",
            assistant_defaults=None,
        )
        case.console._workspace.activate_workspace_id(workspace.workspace_id)
        await case.pilot.pause()
        await case.console.workers.wait_for_complete()
        session = next(
            s for s in case.chats.sessions() if s.id == case.chats.active_session_id
        )
        case.session_id = session.id
        await case.pilot.pause()
        case.composer.load_draft("/rules workspace")
        await case.pilot.pause()
        assert await case.pilot.click("#console-send-message")
        for _ in range(15):
            await case.pilot.pause()
            if isinstance(case.host.screen, ResponseRulesModal):
                break
        assert isinstance(case.host.screen, ResponseRulesModal), (
            case.chats.active_session_id,
            case.rules.scopes(session.id),
            case.composer.draft_text(),
            case.controller.run_state,
        )
        assert case.host.screen.scope == case.rules.scopes(session.id)[1]
        assert case.requests == [] and case.transport.requests == []


@pytest.mark.asyncio
async def test_chat_and_canonical_settings_open_the_same_scoped_manager(tmp_path):
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    async with mounted_rules_console(tmp_path, (120, 35)) as case:
        case.console.action_open_console_session_settings()
        for _ in range(30):
            await case.pilot.pause()
            if case.host.screen.query("#console-settings-response-rules"):
                break
        case.host.screen.query_one("#console-settings-response-rules", Button).press()
        for _ in range(10):
            await case.pilot.pause()
        assert isinstance(case.host.screen, ResponseRulesModal)
        assert case.host.screen.scope == case.rules.scopes(case.session_id)[0]
        await case.pilot.press("escape")
        await case.pilot.pause()
        await case.host.pop_screen()
        settings = SettingsScreen(case.app)
        settings.apply_navigation_context({"category": "hooks"})
        await case.host.push_screen(settings)
        for _ in range(20):
            await case.pilot.pause()
            if settings.query("#settings-response-rules"):
                break
        settings.query_one("#settings-response-rules", Button).press()
        for _ in range(10):
            await case.pilot.pause()
        assert isinstance(case.host.screen, ResponseRulesModal)
        assert case.host.screen.scope.kind == "global"
        assert case.host.screen.runtime is case.rules


@pytest.mark.asyncio
async def test_canonical_settings_opens_global_rules_before_console_visit(tmp_path):
    from Tests.UI.test_destination_shells import _build_test_app
    from Tests.UI.test_response_rules_modal import ManagerHost
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    app = _build_test_app()
    app.chachanotes_db = CharactersRAGDB(
        tmp_path / "settings-rules.sqlite", "settings-rules"
    )
    settings = SettingsScreen(app)
    settings.apply_navigation_context({"category": "hooks"})
    try:
        async with ManagerHost(settings).run_test(size=(120, 35)) as pilot:
            for _ in range(25):
                await pilot.pause()
                if settings.query("#settings-response-rules"):
                    break
            settings.query_one("#settings-response-rules", Button).press()
            for _ in range(30):
                await pilot.pause()
                if isinstance(settings.app.screen, ResponseRulesModal):
                    break
            assert isinstance(settings.app.screen, ResponseRulesModal)
            modal = settings.app.screen
            assert modal.scope.kind == "global"
            assert not modal.runtime.chat_store.sessions()
            assert app.console_runtime._chat_controller is None
    finally:
        owner = getattr(app, "console_runtime", None)
        if owner is not None:
            await owner.dispose()
        app.chachanotes_db.close()


@pytest.mark.asyncio
async def test_active_run_copy_takes_priority_over_an_earlier_rule_verdict(tmp_path):
    async with mounted_rules_console(tmp_path) as case:
        seed_answer(case)
        learned = await case.rules.learn(case.session_id, "The answer omitted evidence")
        assert learned.state == "active", (learned.reason, case.failures)
        entered, release = asyncio.Event(), asyncio.Event()

        case.primary_entered, case.primary_release = entered, release
        case.composer.load_draft("Explain another result")
        case.console.query_one("#console-send-message", Button).press()
        try:
            for _ in range(60):
                await case.pilot.pause()
                if entered.is_set():
                    break
            assert entered.is_set(), case.controller.run_state
            assert (
                case.console._console_active_run_copy()
                == case.controller.run_state.visible_copy
            )
            helpers = len(case.transport.requests)
            primary_calls = len(case.requests)
            case.composer.load_draft("/omfg Add evidence")
            case.console.query_one("#console-send-message", Button).press()
            for _ in range(8):
                await case.pilot.pause()
            assert case.composer.draft_text() == "/omfg Add evidence"
            assert len(case.transport.requests) == helpers
            assert len(case.requests) == primary_calls
        finally:
            release.set()


@pytest.mark.asyncio
async def test_violation_status_opens_rules_with_disable_action(tmp_path):
    async with mounted_rules_console(tmp_path, (120, 35)) as case:
        seed_answer(case)
        case.reply = "Still missing proof"
        assert (
            await case.rules.learn(case.session_id, "The answer omitted evidence")
        ).state == "active"
        assert case.rules.state(case.session_id).assessment.outcome == "violation"
        case.console._sync_console_mode_bar()
        # TASK-35160: the run chip's activation route is the line's segment
        # action now (it posts the same ConsoleRunChip.OpenRequested).
        case.console.query_one(
            "#console-status-text", ConsoleStatusText
        ).action_open_segment("run")
        for _ in range(12):
            await case.pilot.pause()
        assert isinstance(case.host.screen, ResponseRulesModal)
        assert not case.host.screen.query_one("#rr-disable", Button).disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_mapped_learning_phase_and_stop_follow_live_helper(tmp_path, size):
    async with mounted_rules_console(tmp_path, size) as case:
        seed_answer(case)
        case.helper_entered, case.helper_release = asyncio.Event(), asyncio.Event()
        complaint = "/omfg Include evidence in the answer"
        case.composer.load_draft(complaint)
        await case.pilot.pause()
        assert not case.console._console_transcript_poll_needed()
        case.console.query_one("#console-send-message", Button).press()
        try:
            async with asyncio.timeout(5):
                await case.helper_entered.wait()
            await case.pilot.pause()
            assert case.rules.state(case.session_id).phase == "drafting"
            assert not case.console._console_transcript_poll_needed()
            assert _run_visible(case.console)
            assert "Drafting rule" in _run_text(case.console)
            stop = case.console.query_one("#console-stop-generation", Button)
            assert stop.display and not stop.disabled
            assert await case.pilot.click("#console-stop-generation")
            await case.pilot.pause()
            assert case.composer.draft_text() == complaint
            assert case.rules.state(case.session_id).phase == "idle"
            assert case.rules.helpers.unsettled_count == 1
            assert "Drafting rule" not in _run_text(case.console)
        finally:
            case.helper_release.set()
            async with asyncio.timeout(5):
                await case.rules.helpers.wait_for_cleanup()
            await case.console.workers.wait_for_complete()
        assert case.rules.helpers.unsettled_count == 0
        assert case.composer.draft_text() == complaint
        assert case.learning_results and case.learning_results[-1][0] != "active"
        assert case.rules.state(case.session_id).phase == "idle"
        assert "Drafting rule" not in _run_text(case.console)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deferral", ["none", "rebuild", "maintenance", "no-controller"]
)
async def test_rule_phase_events_project_only_the_mounted_current_session(
    tmp_path, deferral, monkeypatch
):
    if deferral == "no-controller":
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        app = _build_console_send_test_app()
        console = ChatScreen(app)
        owner = console._console_runtime()
        assert owner.chat_controller is None
        notices = []
        app.notify = lambda message, **kwargs: notices.append(message)
        try:
            await console.action_stop_console_run()
            assert owner.chat_controller is None
            assert owner.chat_store is None
            assert notices == ["No active Console run to stop."]
        finally:
            await owner.dispose()
        return
    async with mounted_rules_console(tmp_path) as case:
        other_session_id = case.chats.create_session(activate=False).id
        if deferral != "none":
            import sys
            import threading
            import time
            import traceback

            from tldw_chatbook import config

            case.host.app_config = config.load_settings()
            assert case.console._console_config_snapshot_is_disk_loaded(
                case.host.app_config
            )
            seed_answer(case)
            case.helper_entered, case.helper_release = asyncio.Event(), asyncio.Event()
            complaint = "/omfg Include evidence in the answer"
            case.composer.load_draft(complaint)
            case.console.query_one("#console-send-message", Button).press()
            async with asyncio.timeout(5):
                await case.helper_entered.wait()
            await case.pilot.pause()
            async with asyncio.timeout(2):
                while (
                    not case.console._console_attach_reconciled
                    or case.console._console_control_bar_sync_scheduled
                ):
                    await case.pilot.pause(0.01)
                # Attach readiness precedes its queued full-refresh callback.
                refreshed = asyncio.Event()
                case.console.call_after_refresh(refreshed.set)
                await refreshed.wait()
            assert case.console._console_credential_poll_timer is not None
            source = case.rules._sources[case.session_id]
            assert case.rules.acceptance_current(source)
            stop = case.console.query_one("#console-stop-generation", Button)
            assert stop.display and not stop.disabled
            held, release, expired = (
                threading.Event(),
                threading.Event(),
                threading.Event(),
            )
            errors = []
            blocked_stack = []

            def hold_rebuild():
                try:
                    with config._settings_rebuild_lock():
                        config._invalidate_config_caches()
                        held.set()
                        assert release.wait(5), "config holder was not released"
                except BaseException as error:  # noqa: BLE001 - asserted after thread retirement.
                    errors.append(error)

            def unblock_failure():
                blocked_stack.extend(
                    traceback.format_stack(
                        sys._current_frames()[threading.main_thread().ident]
                    )
                )
                expired.set()
                release.set()

            sync_results = []
            sync_control_bar = case.console._sync_console_control_bar

            def observed_sync(*args, **kwargs):
                result = sync_control_bar(*args, **kwargs)
                sync_results.append(result)
                return result

            holder = threading.Thread(target=hold_rebuild)
            watchdog = threading.Timer(2, unblock_failure)
            if deferral == "maintenance":
                case.console._console_sync_maintenance_close_admission()
            try:
                holder.start()
                assert held.wait(3)
                watchdog.start()
                with monkeypatch.context() as observer:
                    observer.setattr(case.console, "_sync_console_control_bar", observed_sync)
                    started = time.monotonic()
                    case.rules._project(case.session_id, phase="testing")
                    elapsed = time.monotonic() - started
                assert not expired.is_set(), (
                    f"rule event blocked on the config rebuild for {elapsed:.3f}s"
                )
                assert _run_visible(case.console)
                assert "Testing rule" in _run_text(case.console)
                assert len(sync_results) == 1
                assert type(sync_results[0]) is bool
                if deferral == "maintenance":
                    assert sync_results[0] is False
                    assert case.console._console_control_bar_sync_scheduled
                elif sync_results[0] is False:
                    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
                        ConsoleReadinessConfigProjection,
                    )

                    projection = ConsoleReadinessConfigProjection.for_screen(case.console)
                    assert case.console._console_control_bar_sync_scheduled or (
                        projection.pending
                        and any(
                            worker.node is case.console
                            and worker.group == "console-readiness-config"
                            and not worker.is_finished
                            for worker in case.console.workers
                        )
                    )
                case.rules._project(other_session_id, phase="repairing")
                assert "Testing rule" in _run_text(case.console)
                # Exercise the armed timer before Stop, without relying on cadence.
                await case.console._console_credential_poll_timer._tick(
                    next_timer=time.monotonic(), count=0
                )
                assert not expired.is_set(), "credential timer blocked the UI loop"
                stop.press()
                async with asyncio.timeout(1):
                    while case.rules.state(case.session_id).phase != "idle" or (
                        _run_visible(case.console)
                        and "Testing rule" in _run_text(case.console)
                    ):
                        await asyncio.sleep(0.01)
                assert not expired.is_set() and not release.is_set()
                assert holder.is_alive()
                assert not _run_visible(case.console) or (
                    "Testing rule" not in _run_text(case.console)
                )
                assert case.rules.state(other_session_id).phase == "repairing"
                assert not case.rules.acceptance_current(source)
                assert case.rules.helpers.unsettled_count == 1
                assert case.composer.draft_text() == complaint
            finally:
                watchdog.cancel()
                release.set()
                holder.join(3)
                if watchdog.ident is not None:
                    watchdog.join(3)
                case.console._console_sync_maintenance_resume()
                assert not holder.is_alive() and not watchdog.is_alive()
                assert not errors
                if blocked_stack:
                    print("Blocked UI stack:\n" + "".join(blocked_stack))
                case.helper_release.set()
                async with asyncio.timeout(5):
                    await case.rules.helpers.wait_for_cleanup()
                from textual.worker import WorkerCancelled

                for worker in tuple(case.console.workers):
                    try:
                        await worker.wait()
                    except WorkerCancelled:
                        pass  # Stop/resume can replace a coalesced UI sync worker.
            assert case.rules.helpers.unsettled_count == 0
            assert case.composer.draft_text() == complaint
            assert case.learning_results and case.learning_results[-1][0] != "active"
            async with asyncio.timeout(5):
                while case.console._console_control_bar_sync_scheduled:
                    await case.pilot.pause(0.01)
            assert "Testing rule" not in _run_text(case.console)
            assert not stop.display
            assert case.rules.state(case.session_id).phase == "idle"
            # The following source-event checks start without a learning result.
            case.rules._project(case.session_id, phase="idle", learning=None)
        for phase, label in (
            ("checking", "Checking rules"),
            ("drafting", "Drafting rule"),
            ("testing", "Testing rule"),
            ("repairing", "Repairing rule violation"),
        ):
            # Exercise the runtime's source notification boundary, not widget writers.
            case.rules._project(case.session_id, phase=phase)
            assert _run_visible(case.console)
            assert label in _run_text(case.console)
            case.rules._project(other_session_id, phase="idle")
            assert label in _run_text(case.console)
        case.rules._project(case.session_id, phase="idle")
        assert not _run_visible(case.console)
        # TASK-35160: pop_screen empties the DOM at every level (the
        # screen's children AND the strip's), so no widget query can answer
        # for a detached screen -- the chip era held a bare chip widget and
        # read its own label. The segment equivalent of that bare-widget
        # read is the strip's own run-segment STATE: asserting it does not
        # move when phases project onto the detached screen is the same
        # contract with no DOM dependency.
        strip = case.console.query_one("#console-status-chips", ConsoleStatusChips)
        run_state_before = strip._run_chip_state
        await case.host.pop_screen()
        await case.pilot.pause()
        assert not case.console.is_attached
        snapshot = case.console._console_credential_snapshot
        case.console._poll_console_credential_readiness()
        assert case.console._console_credential_snapshot == snapshot
        case.rules._project(case.session_id, phase="checking")
        assert strip._run_chip_state == run_state_before


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", ["recovery-and-resume", "full-ui", "full-ui-held"])
async def test_console_syncs_never_wait_for_a_config_rebuild(tmp_path, monkeypatch, sync):
    """TASK-35444, TASK-35645: display syncs never wait while a rebuild holds config.

    The phase test above failed in CI when a sync during Stop rebuilt the
    provider selection with ``load_settings()`` on the UI loop while a config
    rebuild held its lock: first through the transcript's recovery
    placeholder, then through the full UI sync's transcript step and its rail
    visibility pass, then its core-state push and the transcript's row
    reconcile, and the guidance the attach reconcile's screen resume
    refreshes. Here the rebuild starts as the transcript step begins, which
    then reconciles the seeded rows ("full-ui"), before the whole sync
    ("full-ui-held"), or before the recovery sync and a screen resume
    ("recovery-and-resume", one mount for both). Queued refreshes get loop
    turns inside the hold too. The watchdog records where the UI thread was
    waiting.
    """
    import sys
    import threading
    import time
    import traceback

    from tldw_chatbook import config

    async with mounted_rules_console(tmp_path) as case:
        case.host.app_config = config.load_settings()
        assert case.console._console_config_snapshot_is_disk_loaded(
            case.host.app_config
        )
        region = case.console._console_transcript_region_or_none()
        assert region is not None and region.is_mounted
        hold, held, release = threading.Event(), threading.Event(), threading.Event()
        blocked = []

        def hold_rebuild():
            hold.wait(5)
            with config._settings_rebuild_lock():
                config._invalidate_config_caches()
                held.set()
                release.wait(5)

        def expire():
            frame = sys._current_frames()[threading.main_thread().ident]
            blocked.extend(traceback.format_stack(frame))
            release.set()

        def start_rebuild():
            hold.set()
            assert held.wait(3)
            started.append(time.monotonic())
            watchdog.start()

        holder = threading.Thread(target=hold_rebuild)
        watchdog = threading.Timer(2, expire)
        started = []  # when the hold began
        holder.start()
        try:
            if sync == "recovery-and-resume":
                await settle(case.console)
                start_rebuild()
                region.sync_recovery()
                case.console.on_screen_resume()
            else:
                seed_answer(case)
                transcript = case.console.query_one("#console-native-transcript")
                reconcile_rows, reconciled = transcript._reconcile_rows, []

                async def counted_reconcile(*args, **kwargs):
                    if held.is_set():
                        reconciled.append(True)
                    return await reconcile_rows(*args, **kwargs)

                monkeypatch.setattr(transcript, "_reconcile_rows", counted_reconcile)
                transcript_sync = case.console._sync_native_console_transcript
                test_task = asyncio.current_task()

                async def rebuild_then_transcript(*args, **kwargs):
                    # Only this test's sync: the app's own syncs pass through.
                    if asyncio.current_task() is not test_task:
                        return await transcript_sync(*args, **kwargs)
                    if sync == "full-ui":
                        start_rebuild()
                    # Past the TASK-34958 tick gate and the refresh key: the
                    # whole pipeline runs, through the rows' reconcile.
                    case.console._last_native_transcript_refresh_key = None
                    await transcript_sync(force=True)

                monkeypatch.setattr(
                    case.console, "_sync_native_console_transcript", rebuild_then_transcript
                )
                if sync == "full-ui-held":
                    await settle(case.console)
                    start_rebuild()
                    await case.console._sync_native_console_chat_ui()
                else:
                    # Before the rebuild starts, an ordinary deferral (a cold
                    # display projection, say) can still end the sync ahead
                    # of its transcript step, under load on dev too; retry
                    # until this test's own sync reaches it.
                    for _ in range(5):
                        await settle(case.console)
                        await case.console._sync_native_console_chat_ui()
                        if held.is_set():
                            break
                for _ in range(10):
                    await asyncio.sleep(0.02)
                assert held.is_set(), "the full sync never reached its transcript step"
                if sync == "full-ui":
                    assert reconciled, "no transcript rows reconciled under the rebuild"
            elapsed = time.monotonic() - started[0]
            assert not release.is_set(), (
                f"{sync} sync waited {elapsed:.3f}s for the config rebuild:\n"
                + "".join(blocked[-14:])
            )
        finally:
            hold.set()
            watchdog.cancel()
            release.set()
            holder.join(3)
        assert not holder.is_alive()
        if sync.startswith("full-ui"):
            # Deferred, not dropped: any whole-sync replay finishes after release.
            await settle(case.console)
            await _wait_for_text(case.console, case.pilot, "Missing proof")


async def settle(console):
    """Wait out the app's own sync and replay, and the attach reconcile.

    The sync and replay are two of the flags the entry guard checks (nothing
    awaits between this check and the next sync's entry). The attach
    reconcile's resume is pinned by the recovery-and-resume case; the mount's
    own refreshes by the test below (TASK-35836).
    """
    async with asyncio.timeout(5):
        while (
            console._console_sync_in_progress
            or getattr(console, "_console_control_bar_replay_whole_sync", False)
            or not console._console_attach_reconciled
        ):
            await asyncio.sleep(0.01)


_MOUNT_REBUILD_STARTS = {
    "mount": "on_mount",
    "view-start": "_start_console_view_after_reconciliation",
    "roleplay": "_dispatch_active_console_roleplay_refresh",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("start", list(_MOUNT_REBUILD_STARTS))
async def test_console_mount_refreshes_never_wait_for_a_config_rebuild(
    tmp_path, monkeypatch, start
):
    """TASK-35836: the refreshes a Console mount schedules never wait for a rebuild.

    A config rebuild starts as the mount reaches one step and holds the config
    while the refreshes after it run: ``on_mount`` itself (the width rail rules
    of its first resize), the attach reconcile's view start (skill discovery),
    or the first roleplay refresh (the identity surfaces). Each read config on
    the UI loop before this task; a rebuild from the start of the mount defers
    the full sync, and with it the later two, so they get their own start.
    The watchdog records where the UI thread was waiting. After release, every
    deferred refresh finishes.
    """
    import functools
    import sys
    import threading
    import traceback

    from Tests.UI.test_destination_shells import _build_test_app
    from tldw_chatbook import config
    from tldw_chatbook.config import save_setting_to_cli_config
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app = _build_test_app()
    app.chachanotes_db = CharactersRAGDB(tmp_path / "console-mount.sqlite", "mount-ui")
    _configure_native_ready_console(app, model="test-model")
    assert save_setting_to_cli_config("console", "agent_runtime", False)
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    hold, held, release = threading.Event(), threading.Event(), threading.Event()
    blocked = []

    def hold_rebuild():
        hold.wait(10)
        with config._settings_rebuild_lock():
            config._invalidate_config_caches()
            held.set()
            release.wait(10)

    def expire():
        frame = sys._current_frames()[threading.main_thread().ident]
        blocked.extend(traceback.format_stack(frame))
        release.set()

    step = getattr(ChatScreen, _MOUNT_REBUILD_STARTS[start])

    @functools.wraps(step)  # Textual reads a handler's signature to pass its event
    def rebuild_then_step(self, *args, **kwargs):
        if not hold.is_set():
            hold.set()
            assert held.wait(3)
            watchdog.start()
        return step(self, *args, **kwargs)

    holder = threading.Thread(target=hold_rebuild)
    # 5 s, not 2: from on_mount the rest of startup runs inside the hold, and a
    # loaded runner can take seconds; a UI thread that waits stays blocked
    # until this fires, since the test cannot run to release it.
    watchdog = threading.Timer(5, expire)
    monkeypatch.setattr(ChatScreen, _MOUNT_REBUILD_STARTS[start], rebuild_then_step)
    fetched_after_release = []  # skill discovery's context fetches
    service = app.skills_scope_service
    get_context = service.get_context

    async def observed_get_context(*args, **kwargs):
        fetched_after_release.append(release.is_set())
        return await get_context(*args, **kwargs)

    monkeypatch.setattr(service, "get_context", observed_get_context)
    holder.start()
    try:
        async with ConsoleHarness(app).run_test(size=(120, 35)) as pilot:
            console = pilot.app.screen_stack[-1]
            async with asyncio.timeout(5):
                while not held.is_set():
                    await asyncio.sleep(0.01)
            deferred, discovery_waited = set(), False
            for _ in range(15):  # the mount's timers (<= 0.3 s), workers and polls
                deferred.update(getattr(console, "_console_config_retries", None) or ())
                discovery_waited = discovery_waited or any(
                    worker.group == "console-skill-discovery" and not worker.is_finished
                    for worker in console.workers
                )
                await asyncio.sleep(0.05)
            waited = release.is_set()
            release.set()
            holder.join(3)
            assert not waited, "the Console mount waited for the config rebuild:\n" + "".join(
                blocked[-14:]
            )
            # The step's refresh deferred under the hold: a decorator retry, or
            # skill discovery waiting without having fetched.
            expected = {
                "mount": "_adapt_console_workspace_to_width",
                "roleplay": "_sync_console_identity_surfaces",
            }.get(start)
            assert expected is None or expected in deferred, deferred
            assert discovery_waited or start != "view-start"
            assert False not in fetched_after_release, "skill discovery fetched in the hold"
            # Deferred, not dropped: every retry runs and discovery fetches.
            async with asyncio.timeout(5):
                while (
                    getattr(console, "_console_config_retries", None)
                    or not console._console_attach_reconciled
                    or True not in fetched_after_release
                    or any(
                        worker.group == "console-skill-discovery" and not worker.is_finished
                        for worker in console.workers
                    )
                ):
                    await asyncio.sleep(0.02)
            assert console._last_console_workspace_width_band is not None
            await console._console_runtime().dispose()
    finally:
        hold.set()
        watchdog.cancel()
        release.set()
        holder.join(3)
    assert not holder.is_alive()


def test_a_deferred_refresh_runs_once_with_its_latest_call(monkeypatch):
    """TASK-35836: retry_after_config_rebuild defers, then runs; it never drops."""
    from tldw_chatbook import config
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        retry_after_config_rebuild,
    )

    would_wait = [True]
    monkeypatch.setattr(config, "config_read_would_wait", lambda: would_wait[0])
    timers, calls = [], []

    class Screen:
        def set_timer(self, delay, callback):
            timers.append((delay, callback))

    @retry_after_config_rebuild
    def refresh(screen, value):
        calls.append(value)

    screen = Screen()
    refresh(screen, "first")
    refresh(screen, "latest")
    assert calls == [] and [delay for delay, _ in timers] == [0.2]
    timers.pop()[1]()  # the rebuild still holds: retried again, nothing ran
    assert calls == [] and len(timers) == 1
    would_wait[0] = False
    timers.pop()[1]()
    assert calls == ["latest"] and not timers and not screen._console_config_retries
    refresh(screen, "now")
    assert calls == ["latest", "now"]
    screen._closing = True  # a torn-down screen's retry renders nothing
    would_wait[0] = True
    refresh(screen, "late")
    timers.pop()[1]()
    assert calls == ["latest", "now"]
