"""Native composer commands over the actual Console screen."""

import asyncio
import sys
import threading
import time
import traceback

import pytest
from textual.errors import NoWidget
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


#: TASK-39166: patience for an event or flag a test reads next. A bound on
#: state, not a latency budget: a loaded host only makes the wait longer.
_STATE_SECONDS = 30.0
#: How long a config holder holds without its test's release. Every test
#: releases in ``finally``; this only ends the thread if one cannot.
_HOLD_SECONDS = 60.0
#: How long the UI loop may go without running a posted callback before a
#: config-hold test calls it blocked (TASK-39166).
_BLOCKED_LOOP_SECONDS = 10.0


async def _until(predicate, what, step=None, *, timeout=_STATE_SECONDS):
    """Wait, bounded, until ``predicate()``; fail naming ``what`` (TASK-39166)."""
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await (step() if step is not None else asyncio.sleep(0.01))
    except TimeoutError:
        pytest.fail(f"{what}: not within {timeout:g} s")


async def _until_set(event, what):
    await _until(event.is_set, what)


class _UiLoopWatch:
    """Release a config hold only when the UI loop has stopped running.

    TASK-39166. These tests released the hold from a timer started with it
    (2 s, or 5 s for a mount) and failed if it fired. Under load it fired while
    the loop still ran: at expiry the main thread sat in ``selectors.select``,
    idle, because the sync was slow, not waiting. A UI thread that waits for
    the rebuild runs no callback until the hold ends, so this posts one every
    50 ms and fires when the loop has not run it in ``_BLOCKED_LOOP_SECONDS``:
    it records the loop thread's stack, sets ``fired`` and calls ``on_block``.
    """

    def __init__(self, on_block, *, blocked_seconds=_BLOCKED_LOOP_SECONDS):
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self._on_block = on_block
        self._blocked_seconds = blocked_seconds
        self._done = threading.Event()
        self.fired = threading.Event()
        self.stack: list[str] = []
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 3) -> None:
        self._done.set()
        if self._thread.ident is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _watch(self) -> None:
        while not self._done.is_set():
            ran = threading.Event()
            try:
                self._loop.call_soon_threadsafe(ran.set)
            except RuntimeError:  # the loop closed with the test
                return
            deadline = time.monotonic() + self._blocked_seconds
            while not ran.wait(0.05):
                if self._done.is_set():
                    return
                if time.monotonic() >= deadline:
                    frame = sys._current_frames()[self._loop_thread]
                    self.stack.extend(traceback.format_stack(frame))
                    self.fired.set()
                    self._on_block()
                    return
            self._done.wait(0.05)

    def failure(self, what: str) -> str:
        return (
            f"{what}: the UI loop ran no callback for {self._blocked_seconds:g} s"
            " while the rebuild held config:\n" + "".join(self.stack[-14:])
        )



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
        # TASK-39166: bounded waits on each step's state (the modal mounted,
        # not just pushed; then the Console and its composer focus back).
        await _until(
            lambda: isinstance(host.screen, ResponseRulesModal)
            and host.screen.is_mounted,
            "/rules mounting the Response rules manager",
            lambda: pilot.pause(0.01),
        )
        assert host.screen.scope.kind == "chat"
        assert calls == []
        await pilot.press("escape")
        await _until(
            lambda: host.screen is console and host.focused is composer,
            "Escape returning to the Console with the composer focused",
            lambda: pilot.pause(0.01),
        )


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
        # TASK-39166: bounded waits on each step's own state, not 60 and 10
        # pilot turns.
        await _until_set(entered, "the held rule check's assessment")
        before = len(case.requests)
        assert (
            case.chats.get_message(case.chats.active_leaf(case.session_id)).status
            == "complete"
        )
        stop = case.console.query_one("#console-stop-generation", Button)

        def stop_under_pointer():
            # pilot.click aims at the region's origin and reports whether the
            # button was the widget there; a Stop not laid out yet misses.
            if not (stop.display and not stop.disabled and stop.region.area):
                return False
            try:
                return case.host.get_widget_at(*stop.region.offset)[0] is stop
            except NoWidget:
                return False

        await _until(
            stop_under_pointer,
            "the Stop button laid out under the pointer",
            lambda: case.pilot.pause(0.01),
        )
        assert await case.pilot.click("#console-stop-generation")
        await _until(
            lambda: case.rules.state(case.session_id).phase == "idle",
            "Stop cancelling the held rule check (phase idle)",
        )
        release.set()
        assert case.rules.state(case.session_id).phase == "idle"
        assert len(case.requests) == before
        await _until(
            lambda: case.host.focused is case.composer,
            "focus returning to the composer after Stop",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["primary", "helper"])
async def test_fixture_failure_retires_held_transport(tmp_path, transport):
    """A setup failure preserves its assertion and retires the actual HTTP task."""

    class ObservedRelease(asyncio.Event):
        task = None

        async def wait(self):
            self.task = asyncio.current_task()
            return await super().wait()

    entered, release = asyncio.Event(), ObservedRelease()
    failure = AssertionError("response-rule setup failed before its release block")
    rescued = []

    async def release_after_bound():
        # TASK-39166: bound the teardown that must retire the held transport.
        # A cancel cannot: ConsoleRuntime.dispose() waits out a helper's
        # physical exit by design and re-raises the cancel only afterwards,
        # so an unreleased helper hold ran to the 300 s pytest timeout. Past
        # the bound the test frees the hold itself and fails naming it.
        await asyncio.sleep(_STATE_SECONDS)
        if not release.is_set():
            rescued.append(release.task)
            release.set()

    rescue = None
    try:
        with pytest.raises(AssertionError) as caught:
            async with mounted_rules_console(tmp_path) as case:
                if transport == "helper":
                    seed_answer(case)
                    case.helper_entered, case.helper_release = entered, release
                    draft = "/omfg Include evidence in the answer"
                else:
                    case.primary_entered, case.primary_release = entered, release
                    draft = "Explain another result"
                case.composer.load_draft(draft)
                case.console.query_one("#console-send-message", Button).press()
                await _until_set(entered, f"the {transport} transport's request")
                assert release.task is not None and not release.task.done()
                assert not release.is_set()
                rescue = asyncio.create_task(release_after_bound())
                raise failure
        assert not rescued, (
            f"the fixture's teardown did not retire the held {transport} transport "
            f"within {_STATE_SECONDS:g} s; the test released it (task={rescued[0]!r})"
        )
        assert caught.value is failure
        assert release.is_set()
        assert release.task.done()
        assert case.rules.helpers.unsettled_count == 0
    finally:
        if rescue is not None:
            rescue.cancel()


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
        # TASK-39166: wait for these workers to finish, a cancelled one included.
        # wait_for_complete() raised WorkerCancelled when a newer sync replaced
        # one (2 of 10 whole-file runs at load 44-80).
        activation = tuple(case.console.workers)
        await _until(
            lambda: all(worker.is_finished for worker in activation),
            "the workspace activation's workers finishing",
        )
        session = next(
            s for s in case.chats.sessions() if s.id == case.chats.active_session_id
        )
        case.session_id = session.id
        await case.pilot.pause()
        case.composer.load_draft("/rules workspace")
        await case.pilot.pause()
        assert await case.pilot.click("#console-send-message")
        # Mounted, not just pushed: is_mounted turns True only after on_mount's
        # awaited reload, so the test cannot end with the modal mid-mount.
        await _until(
            lambda: isinstance(case.host.screen, ResponseRulesModal)
            and case.host.screen.is_mounted,
            "the /rules workspace command's ResponseRulesModal mounting",
            lambda: case.pilot.pause(0.01),
        )
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
        # TASK-39166: bounded waits on each step's state, not pilot-turn counts.
        step = lambda: case.pilot.pause(0.01)  # noqa: E731
        await _until(
            lambda: bool(case.host.screen.query("#console-settings-response-rules")),
            "the session settings' Response rules button",
            step,
        )
        case.host.screen.query_one("#console-settings-response-rules", Button).press()
        await _until(
            lambda: isinstance(case.host.screen, ResponseRulesModal)
            and case.host.screen.is_mounted,
            "the chat-scoped Response rules manager mounting",
            step,
        )
        assert case.host.screen.scope == case.rules.scopes(case.session_id)[0]
        await case.pilot.press("escape")
        await _until(
            lambda: not isinstance(case.host.screen, ResponseRulesModal),
            "Escape closing the chat-scoped manager",
            step,
        )
        await case.host.pop_screen()
        settings = SettingsScreen(case.app)
        settings.apply_navigation_context({"category": "hooks"})
        await case.host.push_screen(settings)
        await _until(
            lambda: bool(settings.query("#settings-response-rules")),
            "the Hooks category's Response rules button",
            step,
        )
        settings.query_one("#settings-response-rules", Button).press()
        await _until(
            lambda: isinstance(case.host.screen, ResponseRulesModal)
            and case.host.screen.is_mounted,
            "the global Response rules manager mounting",
            step,
        )
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
            # TASK-39166: bounded waits on each step's state. The modal must be
            # mounted, not just pushed: under load the test ended with its
            # Selects still mounting, and teardown raised NoMatches from them.
            await _until(
                lambda: bool(settings.query("#settings-response-rules")),
                "the Hooks category's Response rules button",
                lambda: pilot.pause(0.01),
            )
            settings.query_one("#settings-response-rules", Button).press()
            await _until(
                lambda: isinstance(settings.app.screen, ResponseRulesModal)
                and settings.app.screen.is_mounted,
                "the Response rules manager mounting",
                lambda: pilot.pause(0.01),
            )
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
    from tldw_chatbook.Chat.console_configuration_preparation import (
        standard_console_configuration_sources,
    )

    async with mounted_rules_console(tmp_path) as case:
        # The seeded history skips the send that would prepare these sources.
        trust = await case.app.ensure_local_skill_trust_service()
        assert case.app.local_skills_service.trust_service is trust
        assert standard_console_configuration_sources(
            case.app, case.chats, case.controller, session_id=case.session_id
        )
        seed_answer(case)
        learned = await case.rules.learn(case.session_id, "The answer omitted evidence")
        assert learned.state == "active", (learned.reason, case.failures)
        assert case.rules.state(case.session_id).assessment is not None
        entered, release = asyncio.Event(), asyncio.Event()

        case.primary_entered, case.primary_release = entered, release
        case.composer.load_draft("Explain another result")
        case.console.query_one("#console-send-message", Button).press()
        try:
            await _until_set(entered, "the held primary request")  # TASK-39166
            # The held provider has sent no chunk yet, so the run's own copy
            # is the pre-first-chunk wording (077e5bfa09 moved streaming copy
            # to Agent.console_run_feedback_copy), never the rule verdict.
            assert case.console._console_active_run_copy() == "Waiting for reply…"
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
        # TASK-39166: mounted means on_mount's reload set the Disable action.
        await _until(
            lambda: isinstance(case.host.screen, ResponseRulesModal)
            and case.host.screen.is_mounted,
            "the run segment opening the Response rules manager",
            lambda: case.pilot.pause(0.01),
        )
        assert not case.host.screen.query_one("#rr-disable", Button).disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_mapped_learning_phase_and_stop_follow_live_helper(tmp_path, size):
    from textual.worker import WorkerCancelled, WorkerState

    from tldw_chatbook.UI.Console_Modules.command_handoff import COMMAND_WORKER_GROUP

    async with mounted_rules_console(tmp_path, size) as case:
        seed_answer(case)
        case.helper_entered, case.helper_release = asyncio.Event(), asyncio.Event()
        complaint = "/omfg Include evidence in the answer"
        case.composer.load_draft(complaint)
        await case.pilot.pause()
        assert not case.console._console_transcript_poll_needed()
        case.console.query_one("#console-send-message", Button).press()
        learning_workers = []
        try:
            async with asyncio.timeout(5):
                await case.helper_entered.wait()
            learning_workers = [
                worker
                for worker in case.console.workers
                if worker.node is case.console and worker.group == COMMAND_WORKER_GROUP
            ]
            assert len(learning_workers) == 1
            await case.pilot.pause()
            assert case.rules.state(case.session_id).phase == "drafting"
            assert not case.console._console_transcript_poll_needed()
            assert _run_visible(case.console)
            assert "Drafting rule" in _run_text(case.console)
            stop = case.console.query_one("#console-stop-generation", Button)
            # Rule copy publishes immediately; checked controls may replay later.
            # Pilot caches the click origin before its own layout pause.
            async with asyncio.timeout(5):
                while True:
                    region = stop.region
                    if (
                        stop.is_mounted
                        and stop.display
                        and stop.visible
                        and not stop.disabled
                        and region.width > 0
                        and region.height > 0
                    ):
                        try:
                            hit, _ = case.host.screen.get_widget_at(region.x, region.y)
                        except NoWidget:
                            pass
                        else:
                            if hit is stop:
                                break
                    await asyncio.sleep(0.01)
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
            unrelated = case.console.run_worker(
                asyncio.Event().wait(), group="unrelated-stop-test-worker"
            )
            asyncio.get_running_loop().call_soon(unrelated.cancel)
            async with asyncio.timeout(5):
                for worker in learning_workers:
                    await worker.wait()
                    assert worker.state is WorkerState.SUCCESS
            with pytest.raises(WorkerCancelled):
                await unrelated.wait()
            assert unrelated.is_finished and unrelated.is_cancelled
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
        # Screen assembly may create a passive store; Stop must also work without it.
        original_store = owner.chat_store
        owner.set_chat_store(None)
        assert owner.chat_store is None
        notices = []
        app.notify = lambda message, **kwargs: notices.append(message)
        try:
            await console.action_stop_console_run()
            assert owner.chat_controller is None
            assert owner.chat_store is None
            assert notices == ["No active Console run to stop."]
        finally:
            owner.set_chat_store(original_store)
            await owner.dispose()
        return

    startup_sync_dispatched = set()
    if deferral != "none":
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        call_after_refresh = ChatScreen.call_after_refresh

        def observe_startup_sync(screen, callback, *args, **kwargs):
            if (
                callback == screen.run_worker
                and args == (screen._sync_native_console_chat_ui,)
                and kwargs.get("group") == "console-sync"
                and kwargs.get("exclusive") is False
            ):

                def dispatch():
                    callback(*args, **kwargs)
                    startup_sync_dispatched.add(screen)

                return call_after_refresh(screen, dispatch)
            return call_after_refresh(screen, callback, *args, **kwargs)

        monkeypatch.setattr(ChatScreen, "call_after_refresh", observe_startup_sync)

    async with mounted_rules_console(tmp_path) as case:
        other_session_id = case.chats.create_session(activate=False).id
        if deferral != "none":
            from tldw_chatbook import config

            case.host.app_config = config.load_settings()
            assert case.console._console_config_snapshot_is_disk_loaded(
                case.host.app_config
            )
            # Establish the mounted state before sending; startup and post-send
            # control-bar settling share the original two-second allowance.
            loop = asyncio.get_running_loop()
            readiness_deadline = loop.time() + 2
            try:
                async with asyncio.timeout_at(readiness_deadline):
                    # The attach flag can precede its startup worker dispatch.
                    while (
                        not case.console._console_attach_reconciled
                        or case.console._console_control_bar_sync_scheduled
                        or case.console not in startup_sync_dispatched
                    ):
                        await asyncio.sleep(0.01)
            except TimeoutError as error:
                error.add_note(
                    "Console startup readiness: "
                    f"attach_reconciled={case.console._console_attach_reconciled}, "
                    f"control_sync_scheduled={case.console._console_control_bar_sync_scheduled}, "
                    f"startup_sync_dispatched={case.console in startup_sync_dispatched}"
                )
                raise
            remaining = readiness_deadline - loop.time()
            seed_answer(case)
            case.helper_entered, case.helper_release = asyncio.Event(), asyncio.Event()
            complaint = "/omfg Include evidence in the answer"
            case.composer.load_draft(complaint)
            case.console.query_one("#console-send-message", Button).press()
            async with asyncio.timeout(5):
                await case.helper_entered.wait()
            await case.pilot.pause()
            await _until(
                lambda: not case.console._console_control_bar_sync_scheduled,
                "Console control-bar sync settling before the rebuild hold",
                timeout=remaining,
            )
            assert case.console._console_credential_poll_timer is not None
            source = case.rules._sources[case.session_id]
            assert case.rules.acceptance_current(source)
            stop = case.console.query_one("#console-stop-generation", Button)
            assert stop.display and not stop.disabled
            held, release = threading.Event(), threading.Event()
            errors = []

            def hold_rebuild():
                try:
                    with config._settings_rebuild_lock():
                        config._invalidate_config_caches()
                        held.set()
                        assert release.wait(5), "config holder was not released"
                except BaseException as error:  # noqa: BLE001 - asserted after thread retirement.
                    errors.append(error)

            sync_results = []
            sync_control_bar = case.console._sync_console_control_bar

            def observed_sync(*args, **kwargs):
                result = sync_control_bar(*args, **kwargs)
                sync_results.append(result)
                return result

            holder = threading.Thread(target=hold_rebuild)
            watch = _UiLoopWatch(release.set, blocked_seconds=2)
            expired = watch.fired
            if deferral == "maintenance":
                case.console._console_sync_maintenance_close_admission()
            try:
                holder.start()
                assert held.wait(3), "the holder never took the rebuild lock"
                watch.start()
                with monkeypatch.context() as observer:
                    observer.setattr(case.console, "_sync_console_control_bar", observed_sync)
                    started = time.monotonic()
                    case.rules._project(case.session_id, phase="testing")
                    elapsed = time.monotonic() - started
                assert not expired.is_set(), watch.failure(
                    f"the rule event ({elapsed:.3f}s)"
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
                assert not expired.is_set(), watch.failure("the credential timer")
                # Exercise passive recovery even before its periodic refresh fires.
                region = case.console._console_transcript_region_or_none()
                assert region is not None
                region.sync_recovery()
                assert not expired.is_set(), watch.failure("passive recovery")
                stop.press()
                await _until(
                    lambda: case.rules.state(case.session_id).phase == "idle"
                    and not (
                        _run_visible(case.console)
                        and "Testing rule" in _run_text(case.console)
                    ),
                    "Stop returning the rule phase to idle and clearing 'Testing rule'",
                    timeout=1,
                )
                assert not expired.is_set(), watch.failure("Stop")
                assert not release.is_set()
                assert holder.is_alive()
                assert not _run_visible(case.console) or (
                    "Testing rule" not in _run_text(case.console)
                )
                assert case.rules.state(other_session_id).phase == "repairing"
                assert not case.rules.acceptance_current(source)
                assert case.rules.helpers.unsettled_count == 1
                assert case.composer.draft_text() == complaint
            finally:
                watch.stop()
                release.set()
                holder.join(3)
                case.console._console_sync_maintenance_resume()
                assert not holder.is_alive() and not watch.is_alive()
                assert not errors
                if watch.stack:
                    print("Blocked UI stack:\n" + "".join(watch.stack))
                case.helper_release.set()
                try:
                    async with asyncio.timeout(5):
                        await case.rules.helpers.wait_for_cleanup()
                except TimeoutError:
                    pytest.fail(
                        "the response-rule helpers did not clean up within 5 s"
                    )
                from textual.worker import WorkerCancelled

                for worker in tuple(case.console.workers):
                    try:
                        await worker.wait()
                    except WorkerCancelled:
                        pass  # Stop/resume can replace a coalesced UI sync worker.
            assert case.rules.helpers.unsettled_count == 0
            assert case.composer.draft_text() == complaint
            assert case.learning_results and case.learning_results[-1][0] != "active"
            # A later sync can still own the rendered Stop state and rule text.
            async with asyncio.timeout(5):
                while (
                    case.console._console_control_bar_sync_scheduled
                    or stop.display
                    or "Testing rule" in _run_text(case.console)
                ):
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
@pytest.mark.parametrize(
    "sync",
    [
        "recovery-and-resume",
        "full-ui",
        "full-ui-held",
        "coachmark",
        "coachmark-dismissed",
    ],
)
async def test_console_syncs_never_wait_for_a_config_rebuild(
    tmp_path, monkeypatch, sync
):
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
    turns inside the hold too. The watch (TASK-39166) records where the UI
    thread was waiting. "coachmark" (TASK-39166): a tab sync that sees the
    session count reach two shows the fleet coach-mark, whose copy reads the
    parallel-run cap; it waited in ``load_settings()`` through
    ``_ensure_console_chat_controller`` under load. "coachmark-dismissed": a
    dismissal while that show is deferred keeps it from appearing afterwards.
    """
    from tldw_chatbook import config

    async with mounted_rules_console(tmp_path) as case:
        case.host.app_config = config.load_settings()
        assert case.console._console_config_snapshot_is_disk_loaded(
            case.host.app_config
        )
        region = case.console._console_transcript_region_or_none()
        assert region is not None and region.is_mounted
        held, release = threading.Event(), threading.Event()

        def hold_rebuild():
            with config._settings_rebuild_lock():
                config._invalidate_config_caches()
                held.set()
                release.wait(_HOLD_SECONDS)

        def start_rebuild():
            holder.start()
            assert held.wait(_STATE_SECONDS), "the holder never took the rebuild lock"
            started.append(time.monotonic())
            watch.start()

        holder = threading.Thread(target=hold_rebuild)
        watch = _UiLoopWatch(release.set)
        started = []  # when the hold began
        try:
            if sync.startswith("coachmark"):
                await settle(case.console)
                await (
                    case.console._sync_console_native_session_tabs()
                )  # seeds the count
                assert len(case.chats.sessions()) == 1
                case.chats.create_session(activate=False)
                start_rebuild()
                await case.console._sync_console_native_session_tabs()
                if sync == "coachmark-dismissed":
                    assert "_show_console_fleet_coachmark" in (
                        getattr(case.console, "_console_config_retries", None) or {}
                    ), "the coach-mark show was not deferred"
                    case.console._record_console_fleet_coachmark_dismissed()
            elif sync == "recovery-and-resume":
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
                    case.console,
                    "_sync_native_console_transcript",
                    rebuild_then_transcript,
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
            assert not watch.fired.is_set(), watch.failure(
                f"the {sync} sync ({elapsed:.3f}s into the hold)"
            )
        finally:
            watch.stop()
            release.set()
            if holder.ident is not None:
                holder.join(3)
        assert not holder.is_alive() and not watch.is_alive()
        banner = case.console.query_one("#console-fleet-coachmark")
        if sync == "coachmark-dismissed":
            await _until(
                lambda: (
                    "_show_console_fleet_coachmark"
                    not in (
                        getattr(case.console, "_console_config_retries", None) or {}
                    )
                ),
                "the deferred coach-mark show running after the release",
            )
            assert banner.styles.display != "block", (
                "the dismissed coach-mark reappeared"
            )
        if sync == "coachmark":
            # Deferred, not dropped: the coach-mark shows once the hold ends.
            await _until(
                lambda: banner.styles.display == "block",
                "the fleet coach-mark showing after the release",
            )
            assert "in parallel" in str(
                case.console.query_one("#console-fleet-coachmark-text").renderable
            )
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
    await _until(
        lambda: not (
            console._console_sync_in_progress
            or getattr(console, "_console_control_bar_replay_whole_sync", False)
            or not console._console_attach_reconciled
        ),
        "the app's own sync, its replay and the attach reconcile settling",
    )


_MOUNT_REBUILD_STARTS = {
    "mount": "on_mount",
    "view-start": "_start_console_view_after_reconciliation",
    "roleplay": "_sync_console_identity_surfaces",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("start", list(_MOUNT_REBUILD_STARTS))
async def test_console_mount_refreshes_never_wait_for_a_config_rebuild(
    tmp_path, start, monkeypatch
):
    """TASK-35836: the refreshes a Console mount schedules never wait for a rebuild.

    A config rebuild starts as the mount reaches one step and holds the config
    while the refreshes after it run: ``on_mount`` itself (the width rail rules
    of its first resize), the attach reconcile's view start (skill discovery),
    or the identity-surfaces entry reached by roleplay refresh. Each read config on
    the UI loop before this task; a rebuild from the start of the mount defers
    the full sync, and with it the later two, so they get their own start.
    The watchdog records where the UI thread was waiting. After release, every
    deferred refresh finishes.
    """
    import inspect
    import itertools

    from Tests.UI.test_destination_shells import _build_test_app
    from tldw_chatbook import config
    from tldw_chatbook.config import save_setting_to_cli_config
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets import compact_model_bar

    app = _build_test_app()
    app.chachanotes_db = CharactersRAGDB(tmp_path / "console-mount.sqlite", "mount-ui")
    _configure_native_ready_console(app, model="test-model")
    assert save_setting_to_cli_config("console", "agent_runtime", False)
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    # TASK-39166: ConsoleHarness hosts ChatScreen on its own App, so the compact
    # model bar's checked setup refuses (`widget.app` is not its app_instance)
    # and the bar takes its direct-read compatibility route in compose and
    # on_mount. Under load the roleplay start's hold began before that on_mount,
    # which then waited for the rebuild in load_settings(); production's bar reads
    # off the loop. Those two harness-only reads get the providers read here.
    read_providers = compact_model_bar.get_cli_providers_and_models
    providers = read_providers()
    harness_reads = {
        compact_model_bar.CompactModelBar.compose.__code__,
        compact_model_bar.CompactModelBar.on_mount.__code__,
    }

    def providers_for_harness_bar():
        if sys._getframe(1).f_code in harness_reads:
            return {name: list(models) for name, models in providers.items()}
        return read_providers()

    monkeypatch.setattr(
        compact_model_bar, "get_cli_providers_and_models", providers_for_harness_bar
    )
    hold, held, release = threading.Event(), threading.Event(), threading.Event()

    def hold_rebuild():
        hold.wait()
        if release.is_set():
            return
        with config._settings_rebuild_lock():
            config._invalidate_config_caches()
            held.set()
            release.wait(_HOLD_SECONDS)

    step = getattr(ChatScreen, _MOUNT_REBUILD_STARTS[start])
    original_step = inspect.unwrap(step)
    if start == "roleplay":
        # A lazy mount may have no active chat. Give the native roleplay
        # projection a real session to refresh instead of observing a no-op.
        from tldw_chatbook.Chat.console_runtime import ensure_console_runtime

        ensure_console_runtime(app).ensure_chat_store().create_session()

    holder = threading.Thread(target=hold_rebuild)
    # From on_mount the rest of startup runs inside the hold, and a loaded
    # runner can take seconds; a UI thread that waits stays blocked until the
    # watch fires, since the test cannot run to release it (TASK-39166).
    watch = _UiLoopWatch(release.set)
    fetches = []  # when skill discovery fetched its context: before, in or after the hold
    service = app.skills_scope_service
    get_context = service.get_context

    context_code = get_context.__func__.__code__
    body_codes = {
        inspect.unwrap(getattr(ChatScreen, name)).__code__: name
        for name in ("_sync_console_identity_surfaces", "_adapt_console_workspace_to_width")
    }
    bodies = []
    monitor = None
    for candidate in range(5, 0, -1):
        try:
            sys.monitoring.use_tool_id(candidate, "console-mount-config-rebuild")
        except ValueError:
            continue
        monitor = candidate
        break
    assert monitor is not None

    observed_entry = []

    def observe(code, _offset):
        # Observe just these original entries. Replacing their methods changes
        # native qualification; a process-wide profiler distorts mount timing.
        frame = sys._getframe(1)
        if (
            code is step.__code__ and not hold.is_set()
            and frame.f_locals.get("function", step) is original_step
        ):
            observed_entry.append(frame.f_locals.get("screen", frame.f_locals.get("self")))
            hold.set()
            assert held.wait(_STATE_SECONDS), "the holder never took the rebuild lock"
            watch.start()
        if code in body_codes and observed_entry and frame.f_locals.get("self") is observed_entry[0]:
            bodies.append((body_codes[code], "after" if release.is_set() else "hold"))
        if code is context_code and frame.f_locals.get("self") is service:
            fetches.append("after" if release.is_set() else "hold" if held.is_set() else "before")

    entry_event = sys.monitoring.events.PY_START
    sys.monitoring.register_callback(monitor, entry_event, observe)
    observed_codes = {step.__code__, context_code, *body_codes}
    for code in observed_codes:
        sys.monitoring.set_local_events(monitor, code, entry_event)
    async def coordinate_hold():
        # run_test may wait for readiness work blocked by this artificial hold.
        # Observe loop progress and release independently of entering its body.
        try:
            await _until_set(held, f"the {start} step's entry (the hold's start)")
            console = observed_entry[0]
            expected = {
                "mount": "_adapt_console_workspace_to_width",
                "roleplay": "_sync_console_identity_surfaces",
            }.get(start)
            deferred, discovery_waited = set(), False
            heartbeats = 0
            # TASK-39166: observe the hold until it has seen 15 loop turns and
            # the deferral this start expects, not for a fixed 2 s: under load
            # 2 s held 7 turns. The watch detects a loop that stopped.
            deadline = asyncio.get_running_loop().time() + _STATE_SECONDS
            for tick in itertools.count():
                if held.is_set() and not release.is_set():
                    heartbeats += 1
                deferred.update(getattr(console, "_console_config_retries", None) or ())
                discovery_waited = discovery_waited or any(
                    worker.group == "console-skill-discovery" and not worker.is_finished
                    for worker in console.workers
                )
                seen = (expected is None or expected in deferred) and (
                    discovery_waited or start != "view-start"
                )
                if (
                    (heartbeats >= 15 and seen)
                    or release.is_set()
                    or asyncio.get_running_loop().time() > deadline
                ):
                    break
                await asyncio.sleep(0.05)
            assert not watch.fired.is_set(), watch.failure("the Console mount")
            assert not release.is_set()
            assert heartbeats >= 15, "the UI loop did not progress during the hold"
            assert expected is None or expected in deferred, deferred
            assert discovery_waited or start != "view-start"
            assert "hold" not in fetches, "skill discovery fetched in the hold"
            assert not bodies, "a display refresh entered its body during the hold"
            return deferred
        finally:
            release.set()
            watch.stop()

    holder.start()
    coordinator = asyncio.create_task(coordinate_hold())
    try:
        await asyncio.sleep(0)  # the heartbeat runs before mount readiness waits
        async with ConsoleHarness(app).run_test(size=(120, 35)) as pilot:
            console = pilot.app.screen_stack[-1]
            deferred = await coordinator
            holder.join(3)
            assert not holder.is_alive()
            assert observed_entry == [console], "the intended native refresh never started"
            # Native startup work can reacquire the locks after our holder
            # releases. Attach, discovery, and readiness own that lifecycle;
            # their completion is not the display callback delivery budget.
            # Re-read readiness because startup may issue another projection.
            await _until(
                lambda: console._console_attach_reconciled
                and bool({"before", "after"} & set(fetches))
                and not any(
                    worker.group == "console-skill-discovery" and not worker.is_finished
                    for worker in console.workers
                )
                and not (
                    (projection := getattr(console, "_console_readiness_config_projection", None))
                    is not None and projection.pending
                ),
                "attach, skill discovery and readiness settling after the release",
            )
            # Deferred, not dropped: every original display retry must run
            # within the unchanged five-second delivery budget.
            try:
                async with asyncio.timeout(5):
                    while (
                        getattr(console, "_console_config_retries", None)
                        or not deferred.issubset({name for name, phase in bodies if phase == "after"})
                    ):
                        await asyncio.sleep(0.02)
            except TimeoutError:
                pytest.fail(
                    "after-release refreshes did not settle: "
                    f"retries={getattr(console, '_console_config_retries', None)!r}, "
                    f"bodies={bodies!r}, "
                    f"reconciled={console._console_attach_reconciled}, "
                    f"fetches={fetches!r}, "
                    f"skill_workers={[str(worker) for worker in console.workers if worker.group == 'console-skill-discovery']!r}, "
                    f"locks={(config._settings_rebuild_lock(), config._config_file_lock())!r}, "
                    f"timer_count={len(console._timers)}"
                )
            assert console._last_console_workspace_width_band is not None
            await console._console_runtime().dispose()
    finally:
        coordinator.cancel()
        await asyncio.gather(coordinator, return_exceptions=True)
        for code in observed_codes:
            sys.monitoring.set_local_events(monitor, code, 0)
        sys.monitoring.register_callback(monitor, entry_event, None)
        sys.monitoring.free_tool_id(monitor)
        watch.stop()
        release.set()
        hold.set()
        holder.join(3)
    assert not holder.is_alive() and not watch.is_alive()


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
    # A fresh call after unlock must retire the older queued arguments.
    would_wait[0] = True
    refresh(screen, "stale")
    would_wait[0] = False
    refresh(screen, "fresh")
    timers.pop()[1]()
    assert calls == ["latest", "now", "fresh"]
    screen._closing = True  # a torn-down screen's retry renders nothing
    would_wait[0] = True
    refresh(screen, "late")
    timers.pop()[1]()
    assert calls == ["latest", "now", "fresh"]

    # A rebuild may win after the optimistic cache/probe check. The actual
    # render must acquire both native locks without waiting and retain them.
    import threading

    locks = (threading.RLock(), threading.RLock())
    monkeypatch.setattr(config, "_settings_rebuild_lock", lambda: locks[0])
    monkeypatch.setattr(config, "_config_file_lock", lambda: locks[1])
    would_wait[0] = False
    screen._closing = False
    held, release = threading.Event(), threading.Event()

    def hold_file_lock():
        with locks[1]:
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold_file_lock)
    holder.start()
    try:
        assert held.wait(3)
        refresh(screen, "raced")
        assert calls == ["latest", "now", "fresh"]
        assert len(timers) == 1
        assert not locks[0]._is_owned(), "partial acquisition leaked"
    finally:
        release.set()
        holder.join(3)
    assert not holder.is_alive()
    timers.pop()[1]()
    assert calls == ["latest", "now", "fresh", "raced"]

    acquired_during_render = []

    def try_rebuild():
        for lock in locks:
            acquired = lock.acquire(blocking=False)
            acquired_during_render.append(acquired)
            if acquired:
                lock.release()

    @retry_after_config_rebuild
    def protected_refresh(screen):
        contender = threading.Thread(target=try_rebuild)
        contender.start()
        contender.join(3)
        assert not contender.is_alive()

    protected_refresh(screen)
    assert acquired_during_render == [False, False]
    contender = threading.Thread(target=try_rebuild)
    contender.start()
    contender.join(3)
    assert acquired_during_render == [False, False, True, True]
