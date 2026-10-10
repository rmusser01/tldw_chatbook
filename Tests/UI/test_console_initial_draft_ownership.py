"""The initial composer owns its draft before a deferred first UI refresh."""

from types import SimpleNamespace

import pytest
from textual.events import Key, Mount

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_console_session_tab_close import ProductionConsoleHarness
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile


class DeferredInitialConsole(ProductionConsoleHarness):
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, app_instance, initial):
        super().__init__(app_instance)
        self.app_instance = app_instance
        self.initial = initial
        self.changed_owners = []

    async def on_mount(self, event: Mount):
        # This specialized mount replaces the base harness's normal Console.
        event.prevent_default()
        self.app_instance._ui_ready = True
        screen = ChatScreen(self.app_instance)
        store = screen._ensure_console_chat_store()
        session = store.ensure_session(title="Initial draft")
        store.set_session_draft(session.id, self.initial)
        original_changed = screen._session._on_draft_session_changed

        def record_changed():
            self.changed_owners.append(store.active_session_id)
            return original_changed()

        screen._session._on_draft_session_changed = record_changed
        # Exercise the original full-sync deferral, not a replaced callback.
        screen._console_sync_maintenance_paused = True
        await self.push_screen(screen)


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", ["", "restored draft "])
async def test_initial_typing_survives_first_deferred_draft_refresh(initial):
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = DeferredInitialConsole(app, initial)
    async with host.run_test(size=(160, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        store = screen._ensure_console_chat_store()
        owner = store.active_session_id
        assert composer.draft_text() == initial
        assert composer.handle_console_key(Key("x", "x"))
        authored = composer.capture_draft_snapshot()
        history = composer.export_undo_history()
        assert composer.draft_text() == initial + "x"

        generation = screen._hooks._generation
        assert screen._session._console_visible_draft_session_id == owner
        assert store.active_session_id == owner
        assert screen._session._initial_composer_sync_pending
        screen._console_sync_maintenance_paused = False
        screen._session._sync_console_session_draft()
        assert (
            screen._hooks._generation == generation
        ), "First same-owner sync cancelled the hook Send"
        screen._session._sync_console_session_draft()
        assert (
            screen._hooks._generation == generation
        ), "Repeat same-owner sync cancelled the hook Send"

        assert host.changed_owners == [owner]
        assert composer.draft_text() == initial + "x"
        assert store.session_draft(owner) == initial + "x"
        assert screen._console_visible_draft_session_id == owner
        assert composer.capture_draft_snapshot() == authored
        assert composer.export_undo_history() == history
        assert composer.undo()
        assert composer.draft_text() == initial
        store.set_session_draft(owner, "external revision")
        screen._session._sync_console_session_draft()
        assert composer.draft_text() == "external revision"


@pytest.mark.asyncio
async def test_initial_raw_paste_keeps_provenance_through_first_draft_refresh():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = DeferredInitialConsole(app, "")
    async with host.run_test(size=(160, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        assert composer.handle_console_key(Key("exclamation_mark", "!"))
        assert composer.handle_console_key(Key("space", " "))
        composer.insert_pasted_text("pwd")
        authored = composer.capture_draft_snapshot()
        captured = composer.capture_draft_for_send()
        assert captured.raw_cli_prefix_typed and captured.has_paste

        screen._console_sync_maintenance_paused = False
        screen._session._sync_console_session_draft()

        assert composer.draft_text() == "! pwd"
        assert composer.capture_draft_snapshot() == authored
        after = composer.capture_draft_for_send()
        assert after.raw_cli_prefix_typed and after.has_paste
        assert after.segments == captured.segments


@pytest.mark.parametrize("edit_after_binding", [False, True])
def test_initial_handoff_receipt_only_consumes_its_original_draft(edit_after_binding):
    session = SimpleNamespace(
        id="handoff",
        incarnation_id="incarnation",
        agent_handoff_revision=3,
        agent_handoff_state="pending",
    )
    store = SimpleNamespace(
        session_draft=lambda _id: "handoff draft",
        session_input_snapshot=lambda _id: SimpleNamespace(draft_revision=7),
        sessions=lambda: [session],
    )
    controller = SimpleNamespace(
        _console_chat_store=store,
        _active_native_console_session=lambda: session,
        _console_undo_histories={},
        _screen=SimpleNamespace(),
    )
    composer = ConsoleComposerBar()
    ConsoleSessionController.initialize_composer_draft(controller, composer)
    assert controller._screen._console_visible_draft_revision == 7
    if edit_after_binding:
        composer.insert_text(" newer")
    session.agent_handoff_state = "consumed"
    session.agent_handoff_revision += 1

    ConsoleSessionController._consume_visible_agent_handoff(controller, store, composer)

    assert composer.draft_text() == (
        "handoff draft newer" if edit_after_binding else ""
    )


def test_initial_binding_without_existing_session_leaves_composer_untouched():
    controller = SimpleNamespace(
        _console_chat_store=object(),
        _active_native_console_session=lambda: None,
        _console_visible_draft_session_id=None,
    )
    composer = ConsoleComposerBar()
    composer.load_draft("pending resume")
    captured = composer.capture_draft_snapshot()

    ConsoleSessionController.initialize_composer_draft(controller, composer)

    assert composer.capture_draft_snapshot() == captured
    assert controller._console_visible_draft_session_id is None


@pytest.mark.asyncio
async def test_successor_draft_sync_invalidates_hooks_and_preserves_original_draft():
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = DeferredInitialConsole(app, "original draft ")
    async with host.run_test(size=(160, 48)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        store = screen._ensure_console_chat_store()
        owner = store.active_session_id
        assert composer.handle_console_key(Key("x", "x"))
        history = composer.export_undo_history()
        screen._console_sync_maintenance_paused = False
        screen._session._sync_console_session_draft()
        generation = screen._hooks._generation

        successor = store.create_session(title="Successor draft")
        store.set_session_draft(successor.id, "successor input")
        screen._session._sync_console_session_draft()

        assert screen._hooks._generation == generation + 1
        assert host.changed_owners == [owner, successor.id]
        assert store.session_draft(owner) == "original draft x"
        assert screen._session._console_undo_histories[owner] == history
        assert composer.draft_text() == "successor input"
        assert screen._console_visible_draft_session_id == successor.id
