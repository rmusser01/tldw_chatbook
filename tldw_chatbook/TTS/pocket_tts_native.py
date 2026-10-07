"""PocketTTS's native speech API, spoken through the OpenAI-compatible slot.

TASK-34100.8. The official pocket-tts server (Kyutai, PyPI ``pocket-tts``,
started with ``pocket-tts serve``) does not serve the OpenAI speech route. It
serves ``POST /tts`` with form fields ``text`` and an optional ``voice_url``
(a built-in voice name such as ``alba``), answers ``GET /health``, listens on
port 8000 by default, and streams ``audio/wav`` whose RIFF and data sizes are
placeholders it cannot know up front. ``POST /v1/audio/speech`` answers 404,
so the first-run PocketTTS preset that targeted that route could never speak.

A new TTS provider does not fit the backend layer cleanly (the legacy registry
is sealed and the eight built-in provider ids are pinned across Settings, the
playground and the preference contracts). PocketTTS is instead a dialect of
the existing OpenAI-compatible slot, chosen by an endpoint whose path is
exactly ``/tts``. Settings ▸ Speech & TTS configures it the same way: its
OpenAI Base URL takes the ``/tts`` address.
"""

from __future__ import annotations

from urllib.parse import urlsplit

POCKET_TTS_NATIVE_PATH = "/tts"
POCKET_TTS_DEFAULT_PORT = 8000
POCKET_TTS_DEFAULT_ENDPOINT = (
    f"http://127.0.0.1:{POCKET_TTS_DEFAULT_PORT}{POCKET_TTS_NATIVE_PATH}"
)
#: The server only ever returns WAV.
POCKET_TTS_RESPONSE_FORMAT = "wav"
#: Built-in English voices of pocket-tts 3.3.0 (``_ORIGINS_OF_PREDEFINED_VOICES``).
POCKET_TTS_VOICES: tuple[str, ...] = (
    "alba",
    "marius",
    "javert",
    "jean",
    "fantine",
    "cosette",
    "eponine",
    "azelma",
    "anna",
    "vera",
    "charles",
    "paul",
    "george",
    "mary",
    "jane",
    "michael",
    "eve",
)
POCKET_TTS_WAV_ONLY_COPY = (
    "PocketTTS returns WAV audio only. Set the output format to wav."
)


def is_pocket_tts_native_url(url: object) -> bool:
    """Return whether ``url`` targets pocket-tts's native ``/tts`` route.

    Args:
        url: A speech endpoint URL, normally already normalized.

    Returns:
        True only for an http(s) URL whose path is exactly ``/tts``.
    """
    if not isinstance(url, str):
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme in {"http", "https"} and parts.path == POCKET_TTS_NATIVE_PATH


def pocket_tts_form(text: str, voice: str) -> dict[str, str]:
    """Build the ``/tts`` form body.

    Args:
        text: The text to speak.
        voice: A built-in voice name or voice URL; blank lets the server use
            its own default voice.

    Returns:
        The form fields the server reads.
    """
    fields = {"text": text}
    voice = voice.strip()
    if voice:
        fields["voice_url"] = voice
    return fields


def repair_streamed_wav(body: bytes) -> bytes:
    """Rewrite the placeholder sizes of a streamed WAV to the bytes received.

    A streaming WAV writer cannot know the final length, so pocket-tts sends
    oversized RIFF and data sizes. The app's playable-audio check (and any
    strict reader) rejects those. Only sizes that overrun the body are
    rewritten; a well-formed WAV or a non-WAV body comes back unchanged.

    Args:
        body: The complete response body.

    Returns:
        The body with consistent RIFF and data chunk sizes.
    """
    if len(body) < 12 or body[:4] != b"RIFF" or body[8:12] != b"WAVE":
        return body
    offset = 12
    while offset + 8 <= len(body):
        chunk_id = body[offset : offset + 4]
        declared = int.from_bytes(body[offset + 4 : offset + 8], "little")
        if chunk_id == b"data":
            actual = len(body) - (offset + 8)
            riff_declared = int.from_bytes(body[4:8], "little")
            if declared <= actual and riff_declared + 8 <= len(body):
                return body
            repaired = bytearray(body)
            repaired[4:8] = (len(body) - 8).to_bytes(4, "little")
            repaired[offset + 4 : offset + 8] = actual.to_bytes(4, "little")
            return bytes(repaired)
        offset += 8 + declared + (declared & 1)
    return body


__all__ = [
    "POCKET_TTS_DEFAULT_ENDPOINT",
    "POCKET_TTS_DEFAULT_PORT",
    "POCKET_TTS_NATIVE_PATH",
    "POCKET_TTS_RESPONSE_FORMAT",
    "POCKET_TTS_VOICES",
    "POCKET_TTS_WAV_ONLY_COPY",
    "is_pocket_tts_native_url",
    "pocket_tts_form",
    "repair_streamed_wav",
]
