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
from types import SimpleNamespace
from uuid import UUID

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    export_character_card_to_json,
)
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSEventHandler,
    TTSMessageSpeechRequestEvent,
)
from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSAudioResponse,
    TTSModelInfo,
    TTSNativeCapabilitySnapshot,
    TTSProviderCatalog,
    TTSRequest,
    TTSVoiceDiscoveryResult,
)
from tldw_chatbook.TTS.playground_types import TTSRequestedSelectionSnapshot
from tldw_chatbook.TTS.profile_portability import (
    CHARACTER_CARD_TTS_EXTENSION_KEY,
    PortableTTSProfile,
)
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.TTS.profile_service import TTSProfileService
from tldw_chatbook.TTS.profile_types import CharacterRef, TTSProfileDraft
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
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

UAT_COMPLETE_WAV = (
    b"RIFF"
    b"\x24\x00\x00\x00WAVEfmt "
    b"\x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


class _AvailableAudioCppCapabilities:
    """Stable external-server capability evidence for the portability UAT."""

    revision = 3

    def __init__(self, *, model_id: str, voice_id: str) -> None:
        catalog_revision = 9
        self.snapshot = TTSNativeCapabilitySnapshot(
            provider_id="audio_cpp",
            configuration_revision=self.revision,
            state="complete",
            catalog=TTSProviderCatalog(
                provider_id="audio_cpp",
                revision=catalog_revision,
                health=ProviderHealth(state="available", fresh=True),
                models=(
                    TTSModelInfo(
                        model_id=model_id,
                        display_name=model_id,
                        family="uat",
                        upstream_mode="tts",
                        formats=("wav",),
                        voices=(),
                        supports_speed=False,
                    ),
                ),
            ),
            voice_results={
                model_id: TTSVoiceDiscoveryResult(
                    provider_id="audio_cpp",
                    model_id=model_id,
                    catalog_revision=catalog_revision,
                    voices=(voice_id,),
                    state="complete",
                )
            },
        )

    async def get_native_capability_snapshot(
        self,
        provider_id: str,
        exact_voice_model_ids,
    ) -> TTSNativeCapabilitySnapshot:
        assert provider_id == "audio_cpp"
        assert tuple(exact_voice_model_ids) in {(), (self.snapshot.catalog.models[0].model_id,)}
        return self.snapshot

    def configuration_revision(self, provider_id: str) -> int:
        assert provider_id == "audio_cpp"
        return self.revision

    async def require_current_configuration_revision(
        self,
        provider_id: str,
        expected_revision: int,
    ) -> None:
        assert (provider_id, expected_revision) == ("audio_cpp", self.revision)


class _CompleteWAVSpeechService:
    """External audio.cpp response boundary returning one complete WAV."""

    def __init__(self) -> None:
        self.exact_requests: list[TTSRequest] = []

    def preferences_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(provider_id="audio_cpp")

    async def synthesize_exact(
        self,
        request: TTSRequest,
        progress_sink=None,
    ) -> tuple[TTSAudioResponse, TTSRequestedSelectionSnapshot]:
        self.exact_requests.append(request)

        async def complete_wav_stream():
            yield UAT_COMPLETE_WAV

        return (
            TTSAudioResponse(
                provider_id="audio_cpp",
                model_id=request.model_id,
                audio_format="wav",
                content_type="audio/wav",
                byte_stream=complete_wav_stream(),
            ),
            TTSRequestedSelectionSnapshot(
                provider_id=request.provider_id,
                model_id=request.model_id,
                voice_id=request.voice,
                response_format=request.response_format,
                speed=request.speed,
                options=request.options,
                configuration_revision=3,
            ),
        )

    async def synthesize_default(self, **_kwargs):
        raise AssertionError("assigned character speech must not use global defaults")


class _UATTTSEventHandler(TTSEventHandler):
    def __init__(self, profile_service_loader) -> None:
        super().__init__(profile_service_loader=profile_service_loader)
        self.messages: list[object] = []
        self.completion_posted = asyncio.Event()

    async def post_message(self, message: object) -> None:
        self.messages.append(message)
        if isinstance(message, TTSCompleteEvent):
            self.completion_posted.set()


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
            pilot,
            lambda: (
                app.screen
                if type(app.screen).__name__ == "PersonasScreen"
                and app.screen.is_mounted
                else None
            ),
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

        # Observe the production store's public staging boundary while still
        # forwarding the exact value into the real single-slot store. The
        # store intentionally exposes no peek API, and the Console may claim
        # a handoff before the next Pilot tick.
        staged_chat_handoffs = []
        real_stage = app.pending_handoffs.stage

        def record_chat_handoff(channel, value):
            if channel is HandoffChannel.CHAT:
                detached = ChatHandoffPayload.from_dict(value)
                assert detached is not None
                staged_chat_handoffs.append(detached)
            return real_stage(channel, value)

        monkeypatch.setattr(app.pending_handoffs, "stage", record_chat_handoff)
        await pilot.click("#personas-start-chat")

        # Capture the payload at the public staging boundary; consumption
        # claims and acknowledges the store slot asynchronously.
        handoff = await _wait_for(
            pilot,
            lambda: staged_chat_handoffs[0] if staged_chat_handoffs else None,
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
            print(
                "chat_handoff_pending:",
                app.pending_handoffs.has_pending(HandoffChannel.CHAT),
            )
            print("composers:", list(chat_screen.query("#console-native-composer")))
            print("Inputs:", [w.id for w in chat_screen.query("Input")])
            print("Statics sample:", [
                str(w.render())[:60] for w in list(chat_screen.query("Static"))[:15]
            ])
            raise

        # The Start-Chat handoff must have seeded a character-bound session
        # (greeting) and acknowledged the store slot. Consumption runs off a
        # mount timer, so allow a short grace period after mount.
        def handoff_consumed():
            store = chat_screen._ensure_console_chat_store()
            return (
                not app.pending_handoffs.has_pending(HandoffChannel.CHAT)
                and not chat_screen._handoff_consumption_in_progress
                and any(
                    str(getattr(session, "character_id", ""))
                    == str(handoff_metadata["selected_record_id"])
                    for session in store.sessions()
                )
            )

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


async def test_character_voice_portability_round_trip_to_complete_wav(
    fresh_profile,
    tmp_path,
):
    """Explicit card portability survives import and drives roleplay speech."""

    app, destination_db, _notifications = fresh_profile
    model_id = "supertonic-3"
    voice_id = "M1"
    profile_id = UUID("00000000-0000-4000-8000-000000000004")
    portable = PortableTTSProfile(
        profile_id=profile_id,
        draft=TTSProfileDraft(
            display_name="Portable Ann voice",
            provider_id="audio_cpp",
            model_id=model_id,
            voice_id=voice_id,
            response_format="wav",
            speed=1.0,
            options={},
        ),
    )

    source_db = CharactersRAGDB(
        str(tmp_path / "portable_source.db"),
        "uat-portable-source",
    )
    try:
        source_character_id = source_db.add_character_card(
            {
                "name": "Portable Ann",
                "description": "A roleplay character with an opt-in voice.",
                "first_message": "Hello from Portable Ann.",
                "extensions": {"unrelated/uat": {"preserved": True}},
            }
        )
        assert type(source_character_id) is int
        exported = export_character_card_to_json(
            source_db,
            source_character_id,
            include_image=False,
            portable_tts_profile=portable,
        )
        assert exported is not None
    finally:
        source_db.close_connection()

    exported_payload = json.loads(exported)
    assert (
        exported_payload["data"]["extensions"][CHARACTER_CARD_TTS_EXTENSION_KEY]
        == {
            "schema_version": 1,
            "profile_id": str(profile_id),
            "name": "Portable Ann voice",
            "provider_id": "audio_cpp",
            "model_id": model_id,
            "voice_id": voice_id,
            "response_format": "wav",
            "speed": 1.0,
            "options": {},
        }
    )
    card_path = tmp_path / "portable_ann.json"
    card_path.write_text(exported, encoding="utf-8")

    repository = TTSProfileRepository(tmp_path / "portable_profiles.db")
    await repository.open()
    profile_service = TTSProfileService(
        repository,
        _AvailableAudioCppCapabilities(model_id=model_id, voice_id=voice_id),
    )
    # Use the isolated real repository as the app-owned store. The app closes
    # it during shutdown exactly as production does.
    app._tts_profile_repository = repository
    app._tts_profile_service = profile_service

    handler: _UATTTSEventHandler | None = None
    artifact: Path | None = None
    async with app.run_test(size=(160, 40)) as pilot:
        await _wait_for(pilot, lambda: type(app.screen).__name__ != "Screen")
        app.post_message(NavigateToScreen("personas"))
        personas = await _wait_for(
            pilot,
            lambda: (
                app.screen
                if type(app.screen).__name__ == "PersonasScreen"
                and app.screen.is_mounted
                else None
            ),
        )

        await personas._import_character_from_path(str(card_path))
        await pilot.pause(0.3)
        imported = next(
            card
            for card in destination_db.list_character_cards()
            if card.get("name") == "Portable Ann"
        )
        imported_character_id = int(imported["id"])
        stored = destination_db.get_character_card_by_id(imported_character_id)
        assert CHARACTER_CARD_TTS_EXTENSION_KEY not in stored["extensions"]
        assert stored["extensions"]["unrelated/uat"] == {"preserved": True}

        character_ref = CharacterRef(
            source="local",
            authority_id=destination_db.get_local_authority_id(),
            character_id=str(imported_character_id),
        )
        assigned = await profile_service.get_assigned_profile(character_ref)
        assert assigned.snapshot is not None
        assert assigned.snapshot.profile.profile_id == profile_id
        assert assigned.snapshot.assignment.character_ref == character_ref

        store = ConsoleChatStore()
        session = store.create_session(
            runtime_backend="local",
            assistant_kind="character",
            assistant_id=str(imported_character_id),
            assistant_authority_id=character_ref.authority_id,
            character_id=imported_character_id,
            character_name="Portable Ann",
        )
        response_text = "Portable Ann answers in her dedicated voice."
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=response_text,
        )
        snapshot = store.issue_tts_message_speech_snapshot(message.id)
        assert snapshot.character_ref == character_ref

        async def load_profile_service() -> TTSProfileService:
            return profile_service

        speech_service = _CompleteWAVSpeechService()
        handler = _UATTTSEventHandler(load_profile_service)
        handler._request_cooldown = {}
        handler._tts_service = speech_service
        await handler.handle_tts_request(
            TTSMessageSpeechRequestEvent(
                snapshot,
                store.validate_tts_message_speech_snapshot,
            )
        )
        await asyncio.wait_for(handler.completion_posted.wait(), timeout=2.0)
        completion = next(
            event for event in handler.messages if isinstance(event, TTSCompleteEvent)
        )
        artifact = completion.audio_file

        assert completion.error is None
        assert speech_service.exact_requests == [
            TTSRequest(
                provider_id="audio_cpp",
                model_id=model_id,
                text=response_text,
                voice=voice_id,
                response_format="wav",
                speed=1.0,
                options={},
            )
        ]
        assert artifact is not None
        assert artifact.suffix == ".wav"
        assert artifact.read_bytes() == UAT_COMPLETE_WAV

        await handler.cleanup_tts_resources()

    assert handler is not None
    assert artifact is not None
    assert not artifact.exists()
