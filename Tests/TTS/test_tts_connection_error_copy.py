"""TASK-15530: a connection failure is not a configuration failure.

The dead-port live check in TASK-15422 surfaced "TTS is not configured;
open STTS Settings" for a *reachability* failure against a fully
configured endpoint: `OpenAITTSBackend` wraps `httpx.RequestError` in a
plain `ValueError`, and the events layer buckets every `ValueError` as
configuration-invalid. The typed error stays a `ValueError` subclass so
every existing backend-stream consumer that catches `ValueError`
(audiobook, media reading, briefings) is unaffected.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler
from tldw_chatbook.TTS.base_backends import TTSBackendConnectionError

pytestmark = pytest.mark.unit


def test_connection_error_is_a_value_error_for_existing_consumers() -> None:
    """The backend failure contract ("failures are ValueError") holds.

    Returns:
        None.
    """
    assert issubclass(TTSBackendConnectionError, ValueError)


def test_connection_error_maps_to_connection_unavailable_outcome() -> None:
    """The metric outcome names reachability, not configuration.

    Returns:
        None.
    """
    error = TTSBackendConnectionError(
        "Unable to connect to TTS service. Please check your internet connection."
    )

    assert TTSEventHandler._tts_outcome_code(error) == "connection_unavailable"


def test_connection_error_copy_points_at_the_server_not_the_config() -> None:
    """The toast names the server and Base URL as the thing to check.

    Returns:
        None.
    """
    error = TTSBackendConnectionError(
        "Unable to connect to TTS service. Please check your internet connection."
    )

    assert TTSEventHandler._tts_error_copy(error) == (
        "Unable to reach the TTS server; check that it is running and "
        "the Base URL in STTS Settings"
    )


@pytest.mark.asyncio
async def test_openai_backend_raises_the_typed_error_on_connection_failure(
    monkeypatch,
) -> None:
    """The openai backend's network branch raises the typed error.

    Args:
        monkeypatch: Used to stub the HTTP client seam with a connect
            failure.

    Returns:
        None.
    """
    import httpx

    from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
    from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend

    backend = OpenAITTSBackend(
        {
            "OPENAI_BASE_URL": "http://127.0.0.1:1/v1/audio/speech",
            "OPENAI_AUTH_MODE": "none",
        }
    )
    # The constructor builds a real AsyncClient; close it before swapping in
    # the stub so the test leaks no client.
    await backend.client.aclose()

    class FailingStream:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("connection refused")

        async def __aexit__(self, *exc):
            return False

    class StubClient:
        stream = FailingStream

        async def aclose(self) -> None:
            """Keep ``backend.close()`` safe after the swap."""

    monkeypatch.setattr(backend, "client", StubClient())

    request = OpenAISpeechRequest(
        model="mock-model", input="hi", voice="mock-voice",
        response_format="wav", speed=1.0,
    )
    with pytest.raises(TTSBackendConnectionError):
        async for _chunk in backend.generate_speech_stream(request):
            pass
    await backend.close()
