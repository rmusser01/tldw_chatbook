"""OpenAI Realtime API session -- implements `RealtimeSession` (Task 1's
provider-neutral protocol) on top of `transport.WsTransport`.

Live-probe ground truth (2026-08-04, `Tests/LLM_Calls/openai_realtime_probe.py`
against `wss://api.openai.com/v1/realtime?model=gpt-realtime`, live API key):

INBOUND event `type` names actually observed on the wire -- these match the
task-2 brief's a-priori "expected GA set" exactly, so no discrepancy on
event *names*:
    session.created, session.updated, response.created,
    response.output_item.added, response.output_audio.delta,
    response.output_audio_transcript.delta, response.output_audio.done,
    response.output_audio_transcript.done, response.done,
    input_audio_buffer.speech_started, input_audio_buffer.speech_stopped,
    input_audio_buffer.committed, error.
Also observed but not in the brief's list (extra lifecycle events this
session ignores, forward-compatibly, since they carry no callback this
module fires): conversation.item.added, conversation.item.done,
response.content_part.added, response.content_part.done,
response.output_item.done.
`conversation.item.input_audio_transcription.completed` (in the brief's
list) was not captured live -- the ad hoc audio-turn probe run raced a
second spurious `input_audio_buffer.speech_started` that interrupted the
response before transcription completed. Kept as the brief specifies since
nothing contradicts it and the naming convention is otherwise unchanged
for input-side events.

*** DISCREPANCY (prominent, request-shape not event-name): *** the brief's
Step 1 description ("send session.update with audio+text modalities, pcm16,
input transcription on, server VAD on") describes a request shape that the
live GA endpoint rejects outright. The actual outbound `session.update`
required by the live API is a nested-object shape, confirmed by running the
probe repeatedly against the real endpoint's `error` responses:
  - `session.type` is a required field (probe: `missing_required_parameter:
    session.type`); this session always sends `"type": "realtime"`.
  - Audio config is nested under `session.audio.input` / `session.audio.
    output`, each with `format: {"type": "audio/pcm", "rate": <hz>}`
    (flat `input_audio_format: "pcm16"` -- the old beta shape -- is
    rejected; `rate` is required even inside `format`, confirmed by a
    second live `missing_required_parameter` response).
  - `session.output_modalities` accepts exactly `["text"]` or `["audio"]`
    -- **never both together** (probe: `invalid_value: Invalid modalities:
    ['audio', 'text']. Supported combinations are: ['text'] and
    ['audio'].`). This session always requests `["audio"]` only (voice
    session), diverging from the brief's literal "audio+text modalities"
    instruction, which the live GA API does not support as a single
    request; the probe wins per this task's own governing rule.

See `.superpowers/sdd/2026-08-04-realtime-voice-engine/task-2-report.md`
for the full probe transcript and the request bodies that produced each
error above.

Also live-confirmed: a *partial* `session.update` sent mid-session (only
`{"type": "realtime", "instructions": "..."}`, no `audio` block -- the
shape `send_seed` sends) is accepted and returns `session.updated`, so it
does not need to repeat the full audio schema every time.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

from loguru import logger

from .protocol import RealtimeCallbacks, RealtimeSessionConfig
from .transport import WsTransport

_DEFAULT_URL_TEMPLATE = "wss://api.openai.com/v1/realtime?model={model}"
_TRANSCRIPTION_MODEL = "whisper-1"


class OpenAIRealtimeSession:
    """`RealtimeSession` implementation speaking the OpenAI Realtime GA
    WebSocket protocol.

    Owns one `WsTransport` connection plus two background asyncio tasks
    created in `connect()`: a receive loop that decodes inbound events and
    dispatches them to `callbacks`, and a sender loop that serially drains
    an outbound queue so audio chunks and control messages (seed items,
    cancel/truncate, ...) are never interleaved out of order regardless of
    which thread enqueued them.

    `append_audio` is the one method documented (Task 2 brief) to be called
    from a foreign thread (the future mic-tap recorder thread) -- it and
    every other synchronous public method marshal onto the session's event
    loop via `loop.call_soon_threadsafe`, which is safe to call from any
    thread, including the loop's own.
    """

    def __init__(
        self,
        config: RealtimeSessionConfig,
        callbacks: RealtimeCallbacks,
        *,
        url: str | None = None,
    ) -> None:
        """Construct a session. Does not connect -- call `connect()`.

        Args:
            config: Immutable session parameters (API key, model, voice,
                sample rates, instructions).
            callbacks: Mutable bundle of event callbacks to fire as events
                arrive. Read at dispatch time, so callers may still assign
                new callback fields after construction as long as it
                happens before the corresponding event could arrive.
            url: WebSocket URL override, for pointing at a fake server in
                tests. None uses the production OpenAI realtime endpoint
                with `?model=<config.model>`.
        """
        self._config = config
        self._callbacks = callbacks
        self._url = url
        self._transport = WsTransport()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._outbound_queue: asyncio.Queue[dict | None] | None = None
        self._recv_task: asyncio.Task | None = None
        self._sender_task: asyncio.Task | None = None

        self._current_assistant_item_id: str | None = None
        self._first_audio_fired = False
        self._closed = False

    # ------------------------------------------------------------------
    # RealtimeSession protocol
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the WebSocket connection and send the initial
        `session.update` handshake.

        Starts the background receive and sender-loop tasks. `on_ready`
        (via `callbacks`) fires later, when the server acknowledges the
        handshake with `session.updated`, not from this method directly.

        Returns:
            None.

        Raises:
            Exception: Whatever `WsTransport.connect` raises on failure to
                connect or authenticate (connection refused, handshake
                rejected by the server, DNS failure, etc).
        """
        self._loop = asyncio.get_running_loop()
        self._outbound_queue = asyncio.Queue()

        url = self._url or _DEFAULT_URL_TEMPLATE.format(model=self._config.model)
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        await self._transport.connect(url, headers)

        self._recv_task = asyncio.create_task(self._run_recv_loop())
        self._sender_task = asyncio.create_task(self._run_sender_loop())
        self._enqueue(self._build_session_update())

    def append_audio(self, frames: bytes) -> None:
        """Queue a chunk of input PCM audio to send to the session.

        Thread-safe: safe to call from any thread, including a foreign
        recorder thread with no event loop of its own -- marshals onto the
        session's event loop via `loop.call_soon_threadsafe`.

        Args:
            frames: Raw PCM16 audio bytes at `config.input_sample_rate`.

        Returns:
            None.
        """
        payload = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(frames).decode("ascii"),
        }
        self._enqueue(payload)

    def send_seed(
        self, items: list[tuple[str, str]], instructions: str | None
    ) -> None:
        """Seed the session with prior conversation history and, optionally,
        updated instructions -- without requesting a response.

        Args:
            items: Ordered `(role, text)` pairs to seed as conversation
                history; sent as `conversation.item.create` events in the
                given order.
            instructions: Optional instructions text; if not None, sent as
                a `session.update` before the seed items.

        Returns:
            None.
        """
        if instructions is not None:
            self._enqueue(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "instructions": instructions},
                }
            )
        for role, text in items:
            self._enqueue(self._build_conversation_item(role, text))

    def send_text_item(self, text: str, *, request_response: bool) -> None:
        """Send a text (non-audio) user turn to the session.

        Args:
            text: The text content of the turn.
            request_response: If True, also enqueue `response.create`
                immediately after the item, so the provider replies.

        Returns:
            None.
        """
        self._enqueue(self._build_conversation_item("user", text))
        if request_response:
            self._enqueue({"type": "response.create"})

    def cancel_response(self, played_ms: int) -> None:
        """Cancel the assistant's in-progress response (barge-in).

        Always sends `response.cancel`. If an assistant item id has been
        tracked (from a prior `response.output_item.added`), also sends
        `conversation.item.truncate` for that item so the provider's
        record matches what the user actually heard.

        Args:
            played_ms: Milliseconds of the current response's audio that
                have already been played to the user.

        Returns:
            None.
        """
        self._enqueue({"type": "response.cancel"})
        item_id = self._current_assistant_item_id
        if item_id is not None:
            self._enqueue(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": played_ms,
                }
            )

    async def close(self) -> None:
        """Close the session and release its transport. Idempotent.

        Returns:
            None.
        """
        if self._closed:
            return
        self._closed = True

        if self._loop is not None and self._outbound_queue is not None:
            self._loop.call_soon_threadsafe(self._outbound_queue.put_nowait, None)
        if self._sender_task is not None:
            try:
                await self._sender_task
            except Exception as exc:
                logger.warning(
                    f"OpenAIRealtimeSession.close: sender task ended with "
                    f"error: op=close_sender error={exc!r}"
                )

        await self._transport.close()

        if self._recv_task is not None:
            try:
                await self._recv_task
            except Exception as exc:
                logger.warning(
                    f"OpenAIRealtimeSession.close: recv task ended with "
                    f"error: op=close_recv error={exc!r}"
                )

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    def _build_session_update(self) -> dict:
        """Build the initial `session.update` payload from `self._config`.

        Returns:
            The `session.update` event dict, per the nested GA schema
            documented in this module's header comment block.
        """
        output: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": self._config.output_sample_rate}
        }
        if self._config.voice is not None:
            output["voice"] = self._config.voice

        session: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self._config.input_sample_rate,
                    },
                    "transcription": {"model": _TRANSCRIPTION_MODEL},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": output,
            },
        }
        if self._config.instructions is not None:
            session["instructions"] = self._config.instructions
        return {"type": "session.update", "session": session}

    @staticmethod
    def _build_conversation_item(role: str, text: str) -> dict:
        """Build a `conversation.item.create` event for one `(role, text)`
        turn.

        Args:
            role: `"user"`, `"assistant"`, or `"system"`.
            text: The turn's text content.

        Returns:
            The `conversation.item.create` event dict. User items use
            `input_text` content parts; non-user items use `text`, per the
            Realtime API's content-part typing.
        """
        content_type = "input_text" if role == "user" else "text"
        return {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": role,
                "content": [{"type": content_type, "text": text}],
            },
        }

    def _enqueue(self, item: dict) -> None:
        """Post `item` onto the outbound queue from any thread.

        Args:
            item: A JSON-serializable event dict to send.

        Returns:
            None.
        """
        if self._loop is None or self._outbound_queue is None:
            logger.error(
                "OpenAIRealtimeSession._enqueue called before connect(): "
                f"op=enqueue item_type={item.get('type')!r}"
            )
            return
        self._loop.call_soon_threadsafe(self._outbound_queue.put_nowait, item)

    async def _run_sender_loop(self) -> None:
        """Drain the outbound queue and send each item serially.

        A `None` item is the shutdown sentinel posted by `close()`.

        Returns:
            None.
        """
        assert self._outbound_queue is not None
        try:
            while True:
                item = await self._outbound_queue.get()
                if item is None:
                    return
                await self._transport.send_json(item)
        except Exception as exc:
            logger.error(
                f"OpenAIRealtimeSession sender loop failed: op=sender_loop "
                f"error={exc!r}"
            )
            self._safe_invoke(self._callbacks.on_error, exc, op="on_error")

    # ------------------------------------------------------------------
    # Inbound
    # ------------------------------------------------------------------

    async def _run_recv_loop(self) -> None:
        """Run the transport's receive loop and fire `on_closed` when it
        ends, whatever the reason.

        Returns:
            None.
        """
        reason = "unknown"
        try:
            reason = await self._transport.recv_loop(self._handle_event)
        except Exception as exc:
            logger.error(
                f"OpenAIRealtimeSession recv loop failed: op=recv_loop "
                f"error={exc!r}"
            )
            self._safe_invoke(self._callbacks.on_error, exc, op="on_error")
            reason = f"error: {exc}"
        finally:
            self._safe_invoke(self._callbacks.on_closed, reason, op="on_closed")

    def _handle_event(self, event: dict) -> None:
        """Dispatch one decoded inbound event to the matching handler.

        Any exception raised while parsing/dispatching (as opposed to one
        raised by a user callback, which `_safe_invoke` already isolates)
        is caught here, logged with the event type for context, and routed
        to `on_error` -- never allowed to propagate back into the
        transport's receive loop and kill it.

        Args:
            event: The decoded JSON event dict from the server.

        Returns:
            None.
        """
        event_type = event.get("type", "")
        handler = self._EVENT_HANDLERS.get(event_type)
        if handler is None:
            return
        try:
            handler(self, event)
        except Exception as exc:
            logger.error(
                f"OpenAIRealtimeSession event handling failed: "
                f"op=handle_event event_type={event_type!r} error={exc!r}"
            )
            self._safe_invoke(self._callbacks.on_error, exc, op="on_error")

    def _on_session_updated(self, _event: dict) -> None:
        """Handle `session.updated`: the server accepted our handshake.

        Args:
            _event: The decoded `session.updated` event (unused; the event
                carries no data this session needs).

        Returns:
            None.
        """
        self._safe_invoke(self._callbacks.on_ready, op="on_ready")

    def _on_output_item_added(self, event: dict) -> None:
        """Handle `response.output_item.added`: a new assistant reply item
        started -- track its id for later truncation and reset the
        first-audio-of-this-reply flag.

        Args:
            event: The decoded event; `event["item"]["id"]` is the new
                assistant item id.

        Returns:
            None.
        """
        item = event.get("item") or {}
        item_id = item.get("id")
        self._current_assistant_item_id = item_id
        self._first_audio_fired = False
        if item_id is not None:
            self._safe_invoke(
                self._callbacks.on_reply_started, item_id, op="on_reply_started"
            )

    def _on_audio_delta(self, event: dict) -> None:
        """Handle `response.output_audio.delta`: decode base64 audio and
        fire `on_audio_delta`, plus `on_first_audio` the first time this
        fires for the current reply.

        Args:
            event: The decoded event; `event["delta"]` is base64-encoded
                PCM audio.

        Returns:
            None.
        """
        delta_b64 = event.get("delta", "")
        audio_bytes = base64.b64decode(delta_b64) if delta_b64 else b""
        self._safe_invoke(
            self._callbacks.on_audio_delta, audio_bytes, op="on_audio_delta"
        )
        if not self._first_audio_fired:
            self._first_audio_fired = True
            self._safe_invoke(self._callbacks.on_first_audio, op="on_first_audio")

    def _on_output_transcript_delta(self, event: dict) -> None:
        """Handle `response.output_audio_transcript.delta`: fire
        `on_output_transcript_delta` with the transcript text chunk.

        Args:
            event: The decoded event; `event["delta"]` is the text chunk.

        Returns:
            None.
        """
        self._safe_invoke(
            self._callbacks.on_output_transcript_delta,
            event.get("delta", ""),
            op="on_output_transcript_delta",
        )

    def _on_input_transcript_completed(self, event: dict) -> None:
        """Handle `conversation.item.input_audio_transcription.completed`:
        fire `on_input_transcript` with the user's spoken-input transcript.

        Args:
            event: The decoded event; `event["transcript"]` is the text.

        Returns:
            None.
        """
        self._safe_invoke(
            self._callbacks.on_input_transcript,
            event.get("transcript", ""),
            op="on_input_transcript",
        )

    def _on_speech_started(self, _event: dict) -> None:
        """Handle `input_audio_buffer.speech_started`: fire
        `on_speech_started` (used for barge-in detection).

        Args:
            _event: The decoded event (unused; carries no data this
                session needs).

        Returns:
            None.
        """
        self._safe_invoke(self._callbacks.on_speech_started, op="on_speech_started")

    def _on_input_committed(self, _event: dict) -> None:
        """Handle `input_audio_buffer.committed`: fire `on_turn_committed`.

        Args:
            _event: The decoded event (unused; carries no data this
                session needs).

        Returns:
            None.
        """
        self._safe_invoke(self._callbacks.on_turn_committed, op="on_turn_committed")

    def _on_response_done(self, event: dict) -> None:
        """Handle `response.done`: fire `on_reply_done`, plus `on_usage` if
        the event carries usage information.

        Args:
            event: The decoded event; `event["response"]["usage"]`, if
                present, is passed to `on_usage`.

        Returns:
            None.
        """
        self._safe_invoke(self._callbacks.on_reply_done, op="on_reply_done")
        usage = (event.get("response") or {}).get("usage")
        if usage is not None:
            self._safe_invoke(self._callbacks.on_usage, usage, op="on_usage")

    def _on_error_event(self, event: dict) -> None:
        """Handle `error`: build an `Exception` from the provider's error
        payload and fire `on_error`.

        Args:
            event: The decoded event; `event["error"]["message"]`/`["code"]`
                describe what went wrong.

        Returns:
            None.
        """
        err = event.get("error") or {}
        message = err.get("message", "unknown realtime API error")
        code = err.get("code")
        exc = RuntimeError(f"OpenAI realtime error: {message} (code={code})")
        self._safe_invoke(self._callbacks.on_error, exc, op="on_error")

    _EVENT_HANDLERS: dict[str, Any] = {
        "session.updated": _on_session_updated,
        "response.output_item.added": _on_output_item_added,
        "response.output_audio.delta": _on_audio_delta,
        "response.output_audio_transcript.delta": _on_output_transcript_delta,
        "conversation.item.input_audio_transcription.completed": (
            _on_input_transcript_completed
        ),
        "input_audio_buffer.speech_started": _on_speech_started,
        "input_audio_buffer.committed": _on_input_committed,
        "response.done": _on_response_done,
        "error": _on_error_event,
    }

    def _safe_invoke(self, callback, *args: Any, op: str) -> None:
        """Invoke `callback` with `args`, isolating any exception it raises.

        Any exception raised by `callback` is logged with the failing
        operation name for context and routed to `on_error`. If `on_error`
        is the callback that failed (or raises while being invoked to
        report the original failure), the exception is logged and
        swallowed instead of recursing -- a callback exception can never
        propagate back into the receive loop.

        Args:
            callback: The user-supplied callback to invoke, or None (no-op).
            *args: Positional arguments to pass to `callback`.
            op: Short operation name for log context (e.g. "on_audio_delta").

        Returns:
            None.
        """
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:
            logger.error(
                f"OpenAIRealtimeSession callback failed: op={op} error={exc!r}"
            )
            if callback is self._callbacks.on_error:
                return
            on_error = self._callbacks.on_error
            if on_error is None:
                return
            try:
                on_error(exc)
            except Exception as exc2:
                logger.error(
                    f"OpenAIRealtimeSession on_error callback itself failed: "
                    f"op={op} error={exc2!r}"
                )
