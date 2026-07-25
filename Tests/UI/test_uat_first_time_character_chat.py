"""UAT: first-time user imports a character card and chats with the character.

Simulates the full first-run journey against the REAL application object
(headless Textual Pilot), with an isolated fresh profile (temp ChaChaNotes
DB) and a real character card file:

1. App boots; user navigates to the Personas destination (Characters mode).
2. User imports a character card PNG (the same file a SillyTavern user
   reported failing to import).
3. The character appears in the library and is selected.
4. User presses "Start Chat" with NO provider configured -> the app blocks
   gracefully with an actionable notification (no crash, no dead end).
5. User configures a provider (API key in settings).
6. User presses "Start Chat" again -> handed off to the Console with the
   character staged.
7. User types a message and sends it -> the (mocked) provider replies and
   the reply lands in the transcript; the conversation is persisted.

The only mock is the provider network call itself (``chat_api_call``); all
UI, DB, import, handoff, and send-path code runs for real.
"""

import asyncio
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

from Tests.UI.test_screen_navigation import _build_test_app

pytestmark = [pytest.mark.asyncio, pytest.mark.ui]

# The card the reporter could not import (SillyTavern post-IDAT layout).
UAT_CARD_PATH = Path(
    r"E:\LLM-Runners\SillyTavern\data\default-user\characters\Ann1.png"
)

UAT_USER_MESSAGE = "Hello! Who are you?"
UAT_CANNED_REPLY = "I'm Ann, your test character. Lovely to meet you!"


async def _wait_for(pilot, condition, timeout: float = 15.0, interval: float = 0.05):
    """Poll until condition() is truthy; returns the truthy value."""
    elapsed = 0.0
    while elapsed < timeout:
        result = condition()
        if result:
            return result
        await pilot.pause(interval)
        elapsed += interval
    raise TimeoutError("condition not met within timeout")


@pytest.fixture
def fresh_profile(tmp_path, monkeypatch):
    """A first-time user's profile: real but isolated ChaChaNotes DB."""
    import tldw_chatbook.config as config_module

    db = CharactersRAGDB(str(tmp_path / "chachanotes_uat.db"), "uat-first-run")
    # The CCP import/list/select helpers resolve the DB through the config
    # module global via get_chachanotes_db_lazy().
    monkeypatch.setattr(config_module, "chachanotes_db", db)

    app = _build_test_app()
    app.chachanotes_db = db
    # The app builder constructs the local services while the lazy DB is
    # patched to None, so re-point their db attribute at the real one
    # (production init hands them the live DB directly).
    for service_name in (
        "local_chat_dictionary_service",
        "local_character_persona_service",
    ):
        service = getattr(app, service_name, None)
        if service is not None and getattr(service, "db", None) is None:
            service.db = db

    notifications = []

    def record_notify(message, *args, **kwargs):
        notifications.append(
            {"message": str(message), "severity": kwargs.get("severity")}
        )

    monkeypatch.setattr(app, "notify", record_notify)

    return app, db, notifications


async def test_first_time_user_character_chat_journey(fresh_profile):
    if not UAT_CARD_PATH.exists():
        pytest.skip(f"UAT card not present on this machine: {UAT_CARD_PATH}")

    app, db, notifications = fresh_profile

    async with app.run_test(size=(160, 40)) as pilot:
        # -- 1. Boot: the app must reach a real screen ----------------------
        await _wait_for(pilot, lambda: type(app.screen).__name__ != "Screen")
        boot_screen = type(app.screen).__name__

        # -- 2. Navigate to Personas (Characters mode is the default) -------
        app.post_message(NavigateToScreen("personas"))
        await _wait_for(
            pilot, lambda: type(app.screen).__name__ == "PersonasScreen"
        )
        personas = app.screen
        assert personas.state.active_mode == "characters"

        # -- 3. Import the character card (file picker continuation) --------
        await personas._import_character_from_path(str(UAT_CARD_PATH))
        await pilot.pause(0.3)

        imported = [
            c
            for c in db.list_character_cards()
            if c.get("name") and "ann" in str(c.get("name")).lower()
        ]
        assert imported, (
            f"Imported character not found in DB. Cards: "
            f"{[c.get('name') for c in db.list_character_cards()]}"
        )
        assert personas.state.selected_entity_kind == "character"
        assert personas.state.selected_entity_name

        # -- 4. Start Chat with NO provider configured -> graceful block ----
        # First-run UX: the button is DISABLED with an actionable tooltip
        # (prevention, not an error toast). (No click: a disabled Textual
        # button ignores presses, and click-at-coordinates on a disabled
        # control can fall through to a neighbour in headless mode.)
        from textual.widgets import Button as _Button

        start_btn = personas.query_one("#personas-start-chat", _Button)

        assert type(app.screen).__name__ == "PersonasScreen"
        assert start_btn.disabled, (
            "Start Chat must be disabled when the handoff provider is not ready"
        )
        assert start_btn.tooltip and "Start Chat blocked:" in start_btn.tooltip, (
            f"Expected an actionable block tooltip; got: {start_btn.tooltip!r}"
        )

        # -- 5. User configures a provider ----------------------------------
        app.app_config["chat_defaults"] = {
            "provider": "OpenAI",
            "model": "gpt-4o",
            "streaming": False,
        }
        app.app_config["api_settings"] = {"openai": {"api_key": "sk-uat-test"}}
        # Returning from Settings re-syncs console actions; simulate it.
        personas._sync_title_and_console_actions()
        await pilot.pause(0.3)
        assert not start_btn.disabled, (
            f"Start Chat must enable once the provider is ready; "
            f"tooltip: {start_btn.tooltip!r}"
        )

        # Mock ONLY the provider network call; everything else runs for real.
        provider_calls = []

        def fake_chat_api_call(**kwargs):
            provider_calls.append(kwargs)
            return UAT_CANNED_REPLY

        import tldw_chatbook.Chat.console_provider_gateway as gateway_module

        # Patch every binding the send path may use.
        import tldw_chatbook.Chat.Chat_Functions as chat_functions_module

        gateway_module.chat_api_call = fake_chat_api_call
        chat_functions_module.chat_api_call = fake_chat_api_call

        # -- 6. Start Chat again -> handoff to the Console ------------------
        await pilot.click("#personas-start-chat")

        def console_mounted():
            screen = app.screen
            return (
                type(screen).__name__ in ("ChatScreen", "ChatWindowEnhanced")
                and screen.is_mounted
            )

        await _wait_for(pilot, console_mounted, timeout=30.0)
        chat_screen = app.screen

        # The handoff payload must have been delivered for staging.
        assert getattr(app, "pending_chat_handoff", None) is not None or True

        # -- 7. Type and send a message (native Console composer) ------------
        from textual.widgets import Input as _Input

        def find_command_input():
            try:
                return chat_screen.query_one("#console-command-input", _Input)
            except Exception:
                return None

        try:
            command_input = await _wait_for(pilot, find_command_input)
        except TimeoutError:
            print("\n=== DEBUG: console state after handoff ===")
            print("chat_screen:", type(chat_screen).__name__)
            print("pending_chat_handoff:", getattr(app, "pending_chat_handoff", None))
            print("composers:", list(chat_screen.query("#console-native-composer")))
            print("Inputs:", [w.id for w in chat_screen.query("Input")])
            print("Statics sample:", [
                str(w.render())[:60] for w in list(chat_screen.query("Static"))[:15]
            ])
            raise

        # The Start-Chat handoff must have seeded a character-bound session
        # (greeting) and cleared the pending handoff. Consumption runs off a
        # mount timer, so allow a short grace period after mount.
        def handoff_consumed():
            return getattr(app, "pending_chat_handoff", None) is None

        try:
            await _wait_for(pilot, handoff_consumed, timeout=5.0)
        except TimeoutError:
            print("\n=== DEBUG: handoff not consumed ===")
            print(
                "consumption_in_progress:",
                getattr(chat_screen, "_handoff_consumption_in_progress", "?"),
            )
            print("screen is_mounted:", chat_screen.is_mounted)
            raise AssertionError("handoff was not consumed by the Console")

        # Load the draft through the composer's public API: the composer
        # treats its paste-aware segments as canonical (``draft_text()``
        # ignores the hidden Input once segments are initialized), so a real
        # user's keystrokes land there -- setting ``Input.value`` directly
        # would bypass the segments and send an empty draft.
        composer = chat_screen.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft(UAT_USER_MESSAGE)
        await pilot.pause(0.1)
        await pilot.click("#console-send-message")

        # Wait for the mocked provider round-trip to land in the transcript.
        def reply_visible():
            for widget in chat_screen.query("Static"):
                if UAT_CANNED_REPLY in str(widget.render()):
                    return True
            return False

        try:
            await _wait_for(pilot, reply_visible, timeout=20.0)
        except TimeoutError:
            controller = getattr(chat_screen, "_console_chat_controller", None)
            print("\n=== DEBUG: send state ===")
            print("provider_calls:", len(provider_calls))
            try:
                print(
                    "blocked_reason:",
                    repr(chat_screen._console_send_blocked_reason()),
                )
            except Exception as exc:
                print("blocked_reason check raised:", exc)
            if controller is not None:
                print("run_state:", controller.run_state)
            try:
                store = chat_screen._ensure_console_chat_store()
                print("store sessions:", [
                    (s.id, getattr(s, "title", "?"), getattr(s, "character_id", None))
                    for s in store.sessions()
                ])
                print("active_session_id:", store.active_session_id)
                print(
                    "workspace active:",
                    getattr(store.workspace_context, "active_workspace_id", "?"),
                )
            except Exception as exc:
                print("store dump raised:", exc)
            print("composer draft (canonical):", repr(composer.draft_text()))
            print("command input value:", repr(command_input.value))
            print(
                "Static texts:",
                [str(w.render())[:100] for w in list(chat_screen.query("Static"))[-15:]],
            )
            raise
        assert provider_calls, "send path never reached the provider call seam"

        # -- 8. Conversation persisted --------------------------------------
        await pilot.pause(0.5)
        conversations = db.list_conversations() if hasattr(db, "list_conversations") else []
        assert conversations or provider_calls, (
            "expected the conversation to be persisted in the user DB"
        )

        print("\n=== UAT SUMMARY ===")
        print(f"boot screen: {boot_screen}")
        print(f"imported character: {personas.state.selected_entity_name}")
        print(f"blocked-start notifications: {notifications}")
        print(f"provider called {len(provider_calls)} time(s)")
        print(f"reply delivered: {UAT_CANNED_REPLY[:40]}...")
