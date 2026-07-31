"""UAT: first-time user imports a character card and chats with the character.

Simulates the full first-run journey against the REAL application object
(headless Textual Pilot), with an isolated fresh profile (temp ChaChaNotes
DB) and a SillyTavern-layout (post-IDAT ``tEXt`` chunk) character card PNG:

1. App boots; user navigates to the Personas destination (Characters mode).
2. User imports a character card PNG (the layout a SillyTavern user reported
   failing to import).
3. The character appears in the library and is selected.
4. User presses "Start Chat" with NO provider configured -> the app blocks
   gracefully with an actionable tooltip (no crash, no dead end).
5. User configures a provider (API key in settings).
6. User presses "Start Chat" again -> a character handoff payload is staged
   and the Console consumes it.
7. User types a message and sends it -> the (mocked) provider replies, the
   reply lands in the transcript, and the conversation is persisted.

The card PNG is generated inside the test (same post-IDAT chunk surgery as
``Tests/Character_Chat/test_character_card_lenient_import.py``) so the
journey runs on every machine and CI. Set ``TLDW_UAT_CARD_PATH`` to a real
card file to run the journey against it instead (a set-but-missing path is a
hard failure, never a silent skip).

The only mock is the provider network call itself (``chat_api_call``); all
UI, DB, import, handoff, and send-path code runs for real.
"""

import asyncio
import base64
import json
import os
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

from Tests.Character_Chat.test_character_card_lenient_import import (
    _v2_card,
    _write_png_with_trailing_metadata,
)
from Tests.UI.app_factory import _build_test_app

pytestmark = [pytest.mark.asyncio, pytest.mark.ui]

# Clearly-fake credential placeholder (rule: no realistic-looking secrets in
# tests). Non-empty and not in the app's placeholder-key denylist, so the
# provider-readiness gate treats it as "configured".
FAKE_UAT_API_KEY = "sk-uat-fake-placeholder-not-a-real-key"

# Optional override: run the journey against a real local card file.
UAT_CARD_ENV_VAR = "TLDW_UAT_CARD_PATH"

UAT_USER_MESSAGE = "Hello! Who are you?"
UAT_CANNED_REPLY = "I'm Ann, your test character. Lovely to meet you!"


async def _wait_for(pilot, condition, timeout: float = 15.0, interval: float = 0.05):
    """Poll until ``condition()`` is truthy and return that truthy value.

    Args:
        pilot: Textual Pilot used to yield to the app loop between polls.
        condition: Zero-arg callable evaluated each poll cycle.
        timeout: Maximum seconds to wait before failing.
        interval: Seconds to pause between polls.

    Returns:
        The first truthy value produced by ``condition``.

    Raises:
        TimeoutError: When no truthy value appears within ``timeout``.
    """
    elapsed = 0.0
    while elapsed < timeout:
        result = condition()
        if result:
            return result
        await pilot.pause(interval)
        elapsed += interval
    raise TimeoutError("condition not met within timeout")


def _resolve_uat_card_path(tmp_path: Path) -> Path:
    """Return the character card PNG to import during the journey.

    Defaults to a generated SillyTavern-layout PNG (post-IDAT ``tEXt`` chunk,
    the layout that motivated the import fix) so the UAT runs everywhere.
    ``TLDW_UAT_CARD_PATH`` overrides with a real local card; a set-but-missing
    override fails loudly instead of silently skipping the journey.

    Args:
        tmp_path: Pytest-provided temporary directory for the generated PNG.

    Returns:
        Path to the card PNG the test should import.

    Raises:
        pytest.Failed: When the env-var override points at a missing file.
    """
    override = os.environ.get(UAT_CARD_ENV_VAR, "").strip()
    if override:
        override_path = Path(override)
        if not override_path.exists():
            pytest.fail(
                f"{UAT_CARD_ENV_VAR} is set but the file does not exist: "
                f"{override_path}"
            )
        return override_path
    card = _v2_card(name="UAT Ann", first_mes="Hello, I am Ann.")
    payload = base64.b64encode(json.dumps(card).encode("utf-8")).decode("utf-8")
    return _write_png_with_trailing_metadata(
        tmp_path / "uat_card.png", {"chara": payload}
    )


@pytest.fixture
def fresh_profile(tmp_path, monkeypatch):
    """Build a first-time user's profile: real but isolated ChaChaNotes DB.

    Args:
        tmp_path: Pytest-provided temporary directory for the user database.
        monkeypatch: Pytest fixture used to point the app's lazy DB global
            and notification sink at the isolated profile.

    Returns:
        Tuple of ``(app, db, notifications)``: the real application object,
        the isolated ``CharactersRAGDB``, and a list that records every
        ``app.notify`` call as ``{"message", "severity"}`` dicts.
    """
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


async def test_first_time_user_character_chat_journey(
    fresh_profile, tmp_path, monkeypatch
):
    """Drive the full first-run journey: import a card, then chat with it.

    Args:
        fresh_profile: Isolated ``(app, db, notifications)`` profile fixture.
        tmp_path: Pytest-provided temporary directory for the generated card.
        monkeypatch: Pytest fixture that scopes the provider-call patch so
            the fake cannot leak into other tests in the session.

    Raises:
        AssertionError: When any journey step violates the expected
            first-run behavior.
    """
    card_path = _resolve_uat_card_path(tmp_path)
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
        await personas._import_character_from_path(str(card_path))
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
        app.app_config["api_settings"] = {
            "openai": {"api_key": FAKE_UAT_API_KEY}
        }
        # Returning from Settings re-syncs console actions; simulate it.
        personas._sync_title_and_console_actions()
        await pilot.pause(0.3)
        assert not start_btn.disabled, (
            f"Start Chat must enable once the provider is ready; "
            f"tooltip: {start_btn.tooltip!r}"
        )

        # Mock ONLY the provider network call; everything else runs for real.
        # monkeypatch.setattr scopes the patch to this test -- direct module
        # assignment would leak the fake into later tests in the session.
        provider_calls = []

        def fake_chat_api_call(**kwargs):
            provider_calls.append(kwargs)
            return UAT_CANNED_REPLY

        # The Console gateway resolves the provider call lazily at send time
        # (``from tldw_chatbook.Chat.Chat_Functions import chat_api_call``
        # inside ``ConsoleProviderGateway._chat_api_call``), so patching the
        # attribute on the Chat_Functions module covers the whole send path.
        import tldw_chatbook.Chat.Chat_Functions as chat_functions_module

        monkeypatch.setattr(
            chat_functions_module, "chat_api_call", fake_chat_api_call
        )

        # -- 6. Start Chat again -> handoff to the Console ------------------
        # Baseline for the persistence check: anything the handoff/send
        # persists must appear AFTER this point.
        before_conversation_ids = set(db.get_all_conversation_ids())

        await pilot.click("#personas-start-chat")

        # Capture the staged handoff payload before the Console consumes it
        # (consumption clears app.pending_chat_handoff; the local reference
        # stays valid for field assertions).
        handoff = await _wait_for(
            pilot,
            lambda: getattr(app, "pending_chat_handoff", None),
            timeout=5.0,
        )
        handoff_metadata = handoff.metadata or {}
        assert handoff_metadata.get("intent") == "start_chat", (
            f"handoff intent must be start_chat; metadata: {handoff_metadata!r}"
        )
        assert handoff_metadata.get("selected_kind") == "character", (
            f"handoff must stage a character; metadata: {handoff_metadata!r}"
        )
        assert str(handoff_metadata.get("selected_record_id") or "").strip(), (
            f"handoff must carry the character record id; "
            f"metadata: {handoff_metadata!r}"
        )

        def console_mounted():
            screen = app.screen
            return (
                type(screen).__name__ in ("ChatScreen", "ChatWindowEnhanced")
                and screen.is_mounted
            )

        await _wait_for(pilot, console_mounted, timeout=30.0)
        chat_screen = app.screen

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
        after_conversation_ids = set(db.get_all_conversation_ids())
        new_conversation_ids = after_conversation_ids - before_conversation_ids
        assert new_conversation_ids, (
            "the character chat was not persisted to the user DB "
            f"(before={len(before_conversation_ids)}, "
            f"after={len(after_conversation_ids)})"
        )
        message_counts = {
            cid: db.count_messages_for_conversation(cid)
            for cid in new_conversation_ids
        }
        assert any(count >= 2 for count in message_counts.values()), (
            "expected the new conversation to hold the user + assistant "
            f"messages; per-conversation counts: {message_counts}"
        )

        print("\n=== UAT SUMMARY ===")
        print(f"boot screen: {boot_screen}")
        print(f"card under test: {card_path}")
        print(f"imported character: {personas.state.selected_entity_name}")
        print(f"blocked-start notifications: {notifications}")
        print(f"provider called {len(provider_calls)} time(s)")
        print(f"reply delivered: {UAT_CANNED_REPLY[:40]}...")
        print(f"new conversations persisted: {sorted(message_counts.items())}")
