"""Provider-neutral realtime voice session protocol.

Defines the shapes every realtime transport/session implementation (OpenAI
first; other providers later) must produce and consume, so the rest of the
app -- the hands-free engine-resolution logic in `Chat/console_voice_input.py`
and the eventual Console wiring -- never imports a provider-specific type.

Deliberately free of any provider SDK or `websockets` import: this module is
pure dataclasses/typing, safe to import unconditionally from anywhere
(including at package-import time from `LLM_Calls/realtime/__init__.py`).
Provider transports import `websockets` (or an equivalent) lazily, inside
their own modules, only when a session is actually constructed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RealtimeSessionConfig:
    """Immutable parameters for opening one realtime voice session.

    Attributes:
        api_key: Provider API key used to authenticate the connection.
        model: Provider model identifier (e.g. `"gpt-realtime"`).
        voice: Requested output voice name, or None to use the provider's
            server-side default.
        input_sample_rate: Sample rate in Hz of audio the caller will feed
            via `RealtimeSession.append_audio`.
        output_sample_rate: Sample rate in Hz of audio delivered to
            `RealtimeCallbacks.on_audio_delta`.
        instructions: Optional system/instructions text for the session.
        turn_detection: How the provider should decide a user turn has
            ended -- `"semantic_vad"` (from the content of the speech) or
            `"server_vad"` (from an energy gate). Defaults to
            `"server_vad"`, the PROVIDER's own default, so this transport
            layer keeps describing the provider rather than editorializing;
            the app's product default is chosen one level up, by
            `Chat/console_voice_input.realtime_turn_detection()`.
        vad_threshold: Energy threshold (0-1) for `server_vad`, or None to
            let the provider choose.
        vad_silence_ms: End-of-turn silence window in milliseconds for
            `server_vad`, or None to let the provider choose.

    `vad_threshold`/`vad_silence_ms` apply to `server_vad` ONLY and are
    dropped in semantic mode: the live GA endpoint rejects them there with
    `unknown_parameter`, which fails the entire `session.update`.
    """

    api_key: str
    model: str
    voice: str | None = None
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    instructions: str | None = None
    turn_detection: str = "server_vad"
    vad_threshold: float | None = None
    vad_silence_ms: int | None = None


@dataclass
class RealtimeCallbacks:
    """Mutable bundle of optional event callbacks for a realtime session.

    All fields default to None -- callers set only the events they care
    about. Callbacks are fired from the session's receive loop thread/task,
    so consumers must be thread-safe or trampoline themselves (e.g. via
    `call_from_thread` when updating Textual widgets).

    Attributes:
        on_ready: Session is connected and ready to accept audio/text.
        on_audio_delta: A chunk of output PCM audio arrived.
        on_reply_started: The assistant began a new reply; the argument is
            the assistant item id.
        on_first_audio: The first audio chunk of the current reply arrived.
        on_reply_done: The assistant's current reply has fully completed.
        on_turn_committed: The user's input turn was committed server-side.
        on_input_transcript: A transcript of the user's spoken input.
        on_output_transcript_delta: A chunk of the assistant's output
            transcript text.
        on_speech_started: Server-side voice activity detection observed the
            user starting to speak (used for barge-in).
        on_usage: Token/billing usage information for the session.
        on_error: An error occurred; the argument is the raised exception.
        on_closed: The session closed; the argument is a reason string.
    """

    on_ready: Callable[[], None] | None = None
    on_audio_delta: Callable[[bytes], None] | None = None
    on_reply_started: Callable[[str], None] | None = None
    on_first_audio: Callable[[], None] | None = None
    on_reply_done: Callable[[], None] | None = None
    on_turn_committed: Callable[[], None] | None = None
    on_input_transcript: Callable[[str], None] | None = None
    on_output_transcript_delta: Callable[[str], None] | None = None
    on_speech_started: Callable[[], None] | None = None
    on_usage: Callable[[dict], None] | None = None
    on_error: Callable[[Exception], None] | None = None
    on_closed: Callable[[str], None] | None = None


@runtime_checkable
class RealtimeSession(Protocol):
    """Structural interface every provider-specific realtime session must
    satisfy.

    Implementations own their own transport (typically a `websockets`
    connection) and translate provider wire events into the
    `RealtimeCallbacks` fired against the instance they were constructed
    with. This is a `typing.Protocol`, not a base class: conformance is
    structural (`isinstance(obj, RealtimeSession)` works because the class
    is `@runtime_checkable`), so implementations do not need to subclass it.
    """

    async def connect(self) -> None:
        """Open the transport and complete the provider's session handshake.

        Raises:
            Exception: Whatever the transport/provider raises on failure to
                connect or authenticate.
        """
        ...

    def append_audio(self, frames: bytes) -> None:
        """Queue a chunk of input PCM audio to send to the session.

        Args:
            frames: Raw PCM audio bytes at the session's configured input
                sample rate.
        """
        ...

    def send_seed(self, items: list[tuple[str, str]], instructions: str | None) -> None:
        """Seed the session with prior conversation context before the first
        turn.

        Args:
            items: Ordered `(role, text)` pairs to seed as conversation
                history.
            instructions: Optional instructions text to set/override for the
                session.
        """
        ...

    def send_text_item(self, text: str, *, request_response: bool) -> None:
        """Send a text (non-audio) user turn to the session.

        Args:
            text: The text content of the turn.
            request_response: Whether the provider should generate a reply
                immediately after this item.
        """
        ...

    def cancel_response(self, played_ms: int) -> bool:
        """Cancel the assistant's in-progress response (barge-in).

        Args:
            played_ms: Milliseconds of the current response's audio that
                have already been played to the user, so the provider can
                truncate its record of what was actually heard.

        Returns:
            True when a cancel was actually sent, False when there was no
            active response to cancel. Implementations that cannot tell
            the difference should return True; the value exists so callers
            can RECORD which happened, never to gate behavior on.
        """
        ...

    async def close(self) -> None:
        """Close the session and release its transport.

        Safe to call even if `connect` was never called or already failed.
        """
        ...
