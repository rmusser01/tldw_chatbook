"""Selection failures retain actionable, value-free diagnostics at UI boundaries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore, ConsoleMessageRole
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
)
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSEventHandler,
    TTSMessageSpeechRequestEvent,
)
from tldw_chatbook.TTS.effective_settings import (
    TTSEffectiveResolutionError,
    TTSSelectionSource,
)
from tldw_chatbook.TTS.playground_types import STTSPlaygroundRequest


@pytest.mark.parametrize(
    "copy", [STTSEventHandler._generation_error_copy, TTSEventHandler._tts_error_copy]
)
@pytest.mark.parametrize(
    "code,axis,source,words",
    [
        (
            "invalid_selection",
            "provider_options",
            TTSSelectionSource.STUDIO_DRAFT,
            ("options", "Speech Lab"),
        ),
        (
            "unsupported_selection",
            "voice_mode",
            TTSSelectionSource.GLOBAL,
            ("voice", "Settings"),
        ),
        (
            "invalid_selection",
            "speed",
            TTSSelectionSource.GLOBAL,
            ("speed", "Settings"),
        ),
        (
            "missing_exact",
            "voice_id",
            TTSSelectionSource.CHARACTER_PROFILE,
            ("voice", "character"),
        ),
        (
            "missing_exact",
            "model_id",
            TTSSelectionSource.DEFAULT_PROFILE,
            ("model", "default voice profile"),
        ),
        (
            "revision_incoherent",
            "studio_preferences",
            TTSSelectionSource.STUDIO_DRAFT,
            ("Studio", "reopen"),
        ),
        (
            "catalog_unavailable",
            "provider_catalog",
            TTSSelectionSource.GLOBAL,
            ("refresh", "Speech Lab"),
        ),
        (
            "provider_unknown",
            "provider_id",
            TTSSelectionSource.STUDIO_SAVED,
            ("provider", "Studio preferences"),
        ),
    ],
)
def test_resolution_copy_names_setting_scope_and_recovery(
    copy, code, axis, source, words
):
    error = TTSEffectiveResolutionError(code=code, axis=axis, source=source)
    error.args = ("PRIVATE_TEXT_OR_PROVIDER_PAYLOAD",)
    message = copy(error)
    assert all(word in message for word in words), message
    assert "PRIVATE" not in message
    if code not in {"revision_incoherent", "catalog_unavailable"}:
        assert "retry" not in message.lower()


class _App:
    def __init__(self):
        self.messages = []
        self.notices = []

    def post_message(self, message):
        self.messages.append(message)
        return True

    def notify(self, message, **kwargs):
        self.notices.append(str(message))

    def query_one(self, *args):
        raise LookupError("No Playground mounted")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["studio", "console", "automatic_preflight"])
async def test_resolution_logs_keep_code_axis_source_without_raw_exception(
    path, monkeypatch
):
    error = TTSEffectiveResolutionError(
        code="invalid_selection",
        axis="provider_options",
        source=TTSSelectionSource.STUDIO_DRAFT,
    )
    error.args = ("PRIVATE_TEXT_OR_PROVIDER_PAYLOAD",)
    app = _App()
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        if path == "studio":
            handler = STTSEventHandler(app)
            handler._stts_service = SimpleNamespace(
                synthesize_exact=AsyncMock(side_effect=error)
            )
            request = STTSPlaygroundRequest(
                operation_id="resolution-error",
                provider_id="audio_cpp",
                model_id="model",
                text="PRIVATE_SOURCE_TEXT",
                voice_id=None,
                response_format="wav",
                speed=1.0,
                options={},
            )
            await handler.handle_playground_generate(
                STTSPlaygroundGenerateEvent(request)
            )
            displayed = " ".join(app.notices)
        else:
            handler = TTSEventHandler()
            handler.app = app
            handler._tts_service = SimpleNamespace(
                preferences_snapshot=lambda: SimpleNamespace(provider_id="kokoro"),
                synthesize_default=AsyncMock(side_effect=error),
            )
            if path == "console":
                await handler._generate_tts("PRIVATE_SOURCE_TEXT", "message", None)
            else:
                store = ConsoleChatStore()
                session = store.create_session()
                row = store.append_message(
                    session.id,
                    role=ConsoleMessageRole.ASSISTANT,
                    content="PRIVATE_SOURCE_TEXT",
                )
                event = TTSMessageSpeechRequestEvent(
                    store.issue_tts_message_speech_snapshot(row.id),
                    store.validate_tts_message_speech_snapshot,
                    expected_destination_fingerprint="sha256:" + "a" * 64,
                )
                monkeypatch.setattr(
                    handler, "_destination_for_resolution", AsyncMock(side_effect=error)
                )
                await handler.handle_tts_request(event)
            completions = [
                message
                for message in app.messages
                if isinstance(message, TTSCompleteEvent)
            ]
            assert len(completions) == 1
            displayed = completions[0].error
    finally:
        logger.remove(sink)
    diagnostic = [message for message in messages if "resolution_code=" in message]
    assert len(diagnostic) == 1, messages
    assert "resolution_code=invalid_selection" in diagnostic[0]
    assert "resolution_axis=provider_options" in diagnostic[0]
    assert "resolution_source=studio_draft" in diagnostic[0]
    assert "options" in displayed and "Speech Lab" in displayed
    assert "PRIVATE" not in " ".join([*messages, displayed])
