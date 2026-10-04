"""Native composer commands over the actual Console screen."""

import pytest
import asyncio
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import (
    _build_console_send_test_app,
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_command_grammar import default_console_registry
from tldw_chatbook.Chat.console_command_suggestions import _COMMAND_DESCRIPTIONS
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.response_rules_modal import ResponseRulesModal
from Tests.UI.response_rules_fixtures import mounted_rules_console, seed_answer

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


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
        session = case.chats.create_session(workspace_id=workspace.workspace_id)
        case.session_id = session.id
        await case.pilot.pause()
        case.composer.load_draft("/rules workspace")
        await case.pilot.pause()
        assert await case.pilot.click("#console-send-message")
        for _ in range(15):
            await case.pilot.pause()
            if isinstance(case.host.screen, ResponseRulesModal):
                break
        assert isinstance(case.host.screen, ResponseRulesModal)
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
    from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleRunChip

    async with mounted_rules_console(tmp_path, (120, 35)) as case:
        seed_answer(case)
        case.reply = "Still missing proof"
        assert (
            await case.rules.learn(case.session_id, "The answer omitted evidence")
        ).state == "active"
        assert case.rules.state(case.session_id).assessment.outcome == "violation"
        case.console._sync_console_mode_bar()
        case.console.query_one(
            "#console-run-chip", ConsoleRunChip
        ).action_open_inspector()
        for _ in range(12):
            await case.pilot.pause()
        assert isinstance(case.host.screen, ResponseRulesModal)
        assert not case.host.screen.query_one("#rr-disable", Button).disabled
