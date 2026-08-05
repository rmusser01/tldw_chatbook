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
CORRECTION (V4 final review M9 (d)): at the time this paragraph was
written, `openai_realtime_probe.py` sent no audio at all (its docstring
still described a stale "audio+text modalities" shape it never actually
sent either), so it could not have produced any `input_audio_buffer.*`
event -- the claim above outran what the referenced script could prove.
Fixed by the task-2362/2363 follow-up: the script's docstring now matches
its real (text-turn) behavior, and a new `--audio` mode was added and run
live three times (see the USAGE ground truth section below), each of which
DID observe `input_audio_buffer.committed` -- via a manual `input_audio_
buffer.commit`, not server VAD (that mode disables `turn_detection`
entirely for probe determinism; see that mode's own docstring). `speech_
started`/`speech_stopped` remain UNCONFIRMED by any script committed to
this repo -- they are VAD-only events the manual-commit probe mode never
exercises, and their entry above traces only to the brief's a-priori
expected set, never independently re-probed.
`conversation.item.input_audio_transcription.completed` (field
`transcript`) -- in the brief's list, but not captured by this task's own
probe run (an ad hoc audio-turn script raced a second spurious
`input_audio_buffer.speech_started` that interrupted the response first)
-- was subsequently CONFIRMED live by the task-2 reviewer's own re-probe,
and again, independently, by all three task-2362/2363 `--audio` runs above.
That re-probe also discovered a sibling event this session was not told
about: `conversation.item.input_audio_transcription.delta`, which arrives
*first* (before `...completed`), carrying the same incremental text under
a `delta` field rather than `transcript`. Deliberately still ignored here
-- `on_input_transcript` fires once, from `...completed`'s full transcript,
not incrementally from the `.delta` sibling; a future task can wire the
`.delta` variant to a streaming callback if that granularity is ever
needed. Also observed but not in the brief's list (extra lifecycle events
this session ignores, forward-compatibly, since they carry no callback
this module fires): conversation.item.added, conversation.item.done,
response.content_part.added, response.content_part.done,
response.output_item.done, `rate_limits.updated`. `_TRANSCRIPTION_MODEL =
"whisper-1"` (this module's hardcoded transcription model choice) is
confirmed accepted live.

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

USAGE ground truth (2026-08-04, `Tests/LLM_Calls/openai_realtime_probe.py
--audio`, live key, same GA endpoint, three separate runs -- follow-up
task-2362/2363, closing V4 final review M9 (e) and T2-F12). Moved here from
a `Chat/provider_usage.py` comment, which claimed "live-confirmed" without
this header ever recording the probe that did it:

`response.done`'s `response.usage` -- the payload `on_usage` receives --
splits BOTH `input_token_details` and `output_token_details` (SINGULAR
"token", confirmed by three live runs; `Chat/provider_usage.py`'s
`from_provider_payload` checks this spelling as a Realtime-specific
fallback after the Responses API's plural `input_tokens_details`) into
`text_tokens`/`audio_tokens` (input also carries `image_tokens` and a
nested `cached_tokens_details` with the same three-way split; output does
not -- only input can be served from cache):
    {"total_tokens": 151, "input_tokens": 33, "output_tokens": 118,
     "input_token_details": {"text_tokens": 15, "audio_tokens": 18,
       "image_tokens": 0, "cached_tokens": 0,
       "cached_tokens_details": {"text_tokens": 0, "audio_tokens": 0,
         "image_tokens": 0}},
     "output_token_details": {"text_tokens": 28, "audio_tokens": 90}}
Realtime is billed per audio minute, not per audio token, but the API
still reports audio usage in token units -- `ProviderUsage.audio_input`/
`audio_output` capture this split distinctly from the plain uncached/cached
buckets (task-2363); no cost-catalog wiring reads them yet, so they are
inert for billing until a follow-up task adds that.

`conversation.item.input_audio_transcription.completed` carries its OWN
`usage` field, entirely separate from `response.done`'s above -- NOT a
token count at all, a plain duration of the transcribed input audio:
    {"type": "duration", "seconds": 2}
Previously invisible to `on_usage` (T2-F12: this event only ever fed
`on_input_transcript`). Delivered via the dedicated `on_transcription_
usage` callback (task-2363) so a duration payload can never be
misinterpreted as a token-usage payload by `ProviderUsage.from_provider_
payload`, which does not recognize this shape at all.

`conversation.item.input_audio_transcription.completed` was also observed,
across these three runs, to arrive both BEFORE and AFTER `response.done`
for the same turn -- the same raciness this header already noted for the
task-2 reviewer's own re-probe (an interrupting spurious `speech_started`).
The probe script's `--audio` mode keeps listening a bounded grace period
past `response.done` for exactly this reason; production's own wiring has
no such ordering dependency (`on_transcription_usage`/`on_input_transcript`
target the user's row, `on_usage` the assistant's, entirely independently).

TURN DETECTION ground truth (2026-08-04,
`Tests/LLM_Calls/openai_realtime_turn_detection_probe.py`, live key, same
GA endpoint). The server's OWN default, echoed on `session.created`:

    {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
     "silence_duration_ms": 200, "create_response": true,
     "interrupt_response": true, "idle_timeout_ms": null}

Note `silence_duration_ms: 200` -- two tenths of a second of quiet ends a
turn. That is the mechanism behind gate round 5's "it transcribes random
words": an ordinary mid-sentence pause (or a keystroke, or a cough) closes
the turn, and the transcription model is handed a fragment to hallucinate
from. Sent blocks and their verbatim verdicts:

  ACCEPTED  {"type": "server_vad"}
      -> echoed with the server's defaults filled in (above)
  ACCEPTED  {"type": "server_vad", "threshold": 0.6,
             "prefix_padding_ms": 300, "silence_duration_ms": 700}
      -> echoed back verbatim (threshold 0.6, silence 700)
  ACCEPTED  {"type": "semantic_vad"}
      -> echoed as {"type": "semantic_vad", "eagerness": "auto",
                    "create_response": true, "interrupt_response": true}
  ACCEPTED  {"type": "semantic_vad", "eagerness": "low"|"auto"|"high"}
      -> echoed back verbatim; `eagerness` IS in the accepted schema and
         defaults to "auto"
  REJECTED  {"type": "semantic_vad", "threshold": 0.6}
      -> code='unknown_parameter'
         param='session.audio.input.turn_detection.threshold'
         message="Unknown parameter:
                  'session.audio.input.turn_detection.threshold'."

So the two modes take DISJOINT fields, and mixing them is not a
best-effort degradation -- an `unknown_parameter` fails the entire
`session.update`. `_build_turn_detection` drops the server_vad knobs in
semantic mode for exactly that reason.
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

# Sentinel distinct from any real response id (including None, which is
# itself a possible -- if degenerate -- `response.id` in test doubles) so
# `_reply_started_for_response_id`'s "have we already handled this
# response's first item" guard cannot be accidentally satisfied before any
# `response.created` has ever been seen.
_UNSET = object()


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
        self._current_response_id: str | None = None
        self._response_active = False
        self._reply_started_for_response_id: object = _UNSET
        self._first_audio_fired = False
        self._closed = False
        self._teardown_done = False
        self._enqueue_error_logged = False

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

    def cancel_response(self, played_ms: int) -> bool:
        """Cancel the assistant's in-progress response (barge-in).

        Two independent messages, gated independently:

        * `response.cancel` is sent ONLY while a response is genuinely
          active (F8) -- live-confirmed, cancelling a response that has
          already ended produces an `error` event from the provider.
        * `conversation.item.truncate` is sent whenever an assistant item
          id is known, ACTIVE OR NOT. The tracked item id is deliberately
          never cleared by a completed response (see `_on_response_done`),
          because a just-completed-but-still-playing item is the normal
          barge-in case: audio outlives generation by whatever is still
          buffered, and the wiring holds the loop in `speaking` for
          exactly that window. Gating the truncate on `_response_active`
          too meant the most common barge-in of all -- the user cutting
          off a reply they can still hear -- told the provider nothing at
          all, leaving its history claiming the user heard the whole
          answer. That is precisely what `played_ms` exists to prevent
          (PR #1350 review, Q1).

          An earlier note in this module claimed the truncate would error
          for a completed response. A dedicated live probe (2026-08-04)
          disproved it: `conversation.item.truncate` against a completed
          item returns `conversation.item.truncated`, no error.

        Args:
            played_ms: Milliseconds of the current response's audio that
                have already been played to the user.

        Returns:
            True when ANYTHING was enqueued (a cancel, a truncate, or
            both), False only for the true no-op: no active response AND
            no assistant item ever tracked, so there is nothing to cancel
            and nothing to truncate. Reported so a caller can record which
            branch a barge-in took -- "the provider was told" and "there
            was nothing to tell it" are different events, and from outside
            this class they are otherwise indistinguishable (a live-gate
            incident was spent guessing which had happened).
        """
        item_id = self._current_assistant_item_id
        if not self._response_active and item_id is None:
            logger.debug(
                "OpenAIRealtimeSession.cancel_response: nothing to cancel or "
                f"truncate: op=cancel_response played_ms={played_ms}"
            )
            return False
        if self._response_active:
            self._enqueue({"type": "response.cancel"})
        else:
            logger.debug(
                "OpenAIRealtimeSession.cancel_response: response already "
                "ended; truncating the still-playing item only: "
                f"op=cancel_response played_ms={played_ms}"
            )
        if item_id is not None:
            self._enqueue(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": played_ms,
                }
            )
        return True

    async def close(self) -> None:
        """Close the session and release its transport. Idempotent.

        Sets the closed flag immediately (rejecting any further
        `_enqueue` calls, e.g. from `append_audio` racing in from a
        foreign thread during shutdown), then runs teardown at most once
        -- guarded separately from the closed flag, since
        `_run_sender_loop` also sets it after an unrecoverable send
        failure without having run teardown. The sender task's join is
        bounded to 2 seconds: a stalled connection must not hang teardown
        forever; if it doesn't finish in time, it's cancelled and the
        transport is closed anyway.

        Returns:
            None.
        """
        self._closed = True
        if self._teardown_done:
            return
        self._teardown_done = True

        if self._loop is not None and self._outbound_queue is not None:
            self._loop.call_soon_threadsafe(self._outbound_queue.put_nowait, None)
        if self._sender_task is not None:
            try:
                await asyncio.wait_for(self._sender_task, timeout=2.0)
            except TimeoutError:
                logger.warning(
                    "OpenAIRealtimeSession.close: sender task did not finish "
                    "within 2s, cancelling and closing transport anyway: "
                    "op=close_sender"
                )
                self._sender_task.cancel()
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
                    "turn_detection": self._build_turn_detection(),
                },
                "output": output,
            },
        }
        if self._config.instructions is not None:
            session["instructions"] = self._config.instructions
        return {"type": "session.update", "session": session}

    def _build_turn_detection(self) -> dict:
        """Build the `turn_detection` block for `session.audio.input`.

        The two modes take DISJOINT fields, live-confirmed: `semantic_vad`
        rejects `threshold` with `unknown_parameter` (and an
        `unknown_parameter` fails the WHOLE `session.update`, taking the
        conversation with it), so a threshold configured while semantic
        mode is selected is dropped here rather than forwarded.

        An unset knob is OMITTED rather than defaulted: the provider fills
        its own value and echoes it back, so restating today's number here
        would silently freeze it.

        Returns:
            The `turn_detection` block to send.
        """
        block: dict[str, Any] = {"type": self._config.turn_detection}
        if self._config.turn_detection != "server_vad":
            return block
        if self._config.vad_threshold is not None:
            block["threshold"] = self._config.vad_threshold
        if self._config.vad_silence_ms is not None:
            block["silence_duration_ms"] = self._config.vad_silence_ms
        return block

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

        Silently drops `item` once the session is closed (whether by an
        explicit `close()` call or because the sender loop already died --
        see `_run_sender_loop`), rather than raising into whatever thread
        called it -- `append_audio` is documented to be safe from the
        recorder thread even after shutdown races ahead of it. The
        `call_soon_threadsafe` marshal itself is also guarded: if the
        event loop closes in the narrow window between the `self._closed`
        check and this call, the resulting `RuntimeError` is caught and
        logged once (not per dropped frame, to avoid log-flooding a busy
        audio stream) rather than propagating.

        Args:
            item: A JSON-serializable event dict to send.

        Returns:
            None.
        """
        if self._closed:
            return
        if self._loop is None or self._outbound_queue is None:
            logger.error(
                "OpenAIRealtimeSession._enqueue called before connect(): "
                f"op=enqueue item_type={item.get('type')!r}"
            )
            return
        try:
            self._loop.call_soon_threadsafe(self._outbound_queue.put_nowait, item)
        except RuntimeError as exc:
            if not self._enqueue_error_logged:
                self._enqueue_error_logged = True
                logger.error(
                    "OpenAIRealtimeSession._enqueue: loop rejected marshal, "
                    "session is effectively closed (further drops logged "
                    f"once): op=enqueue item_type={item.get('type')!r} "
                    f"error={exc!r}"
                )

    async def _run_sender_loop(self) -> None:
        """Drain the outbound queue and send each item serially.

        A `None` item is the shutdown sentinel posted by `close()`. If
        sending an item raises (e.g. the connection died), the loop marks
        the session closed -- so callers stop silently no-oping into a
        queue nobody drains any more, per `_enqueue`'s guard -- fires
        `on_error` once, and exits. It does not retry or restart itself.

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
                f"OpenAIRealtimeSession sender loop failed, marking session "
                f"closed: op=sender_loop error={exc!r}"
            )
            self._closed = True
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
            event: The decoded JSON payload from the server -- typed `dict`
                because that's the well-formed case, but not guaranteed to
                actually be one: any JSON value (a bare list, string,
                number...) decodes successfully at the transport layer, so
                this method must not assume `.get` exists on it.

        Returns:
            None.
        """
        if not isinstance(event, dict):
            logger.warning(
                "OpenAIRealtimeSession dropping non-dict inbound frame: "
                f"op=handle_event received_type={type(event).__name__}"
            )
            return
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

    def _on_response_created(self, event: dict) -> None:
        """Handle `response.created`: a new response started -- mark one
        active and remember its id.

        The id lets `_on_output_item_added` tell whether a later
        `response.output_item.added` is the first item of this response
        (reset first-audio, fire `on_reply_started`) or an additional item
        within a response already underway (retarget the tracked item id
        only). `_response_active` gates `cancel_response`: cancelling with
        no active response live-produces an `error` event from the
        provider, so `cancel_response` no-ops instead of sending a stale
        `response.cancel`.

        Args:
            event: The decoded event; `event["response"]["id"]` is the new
                response id.

        Returns:
            None.
        """
        response = event.get("response") or {}
        self._response_active = True
        self._current_response_id = response.get("id")

    def _on_output_item_added(self, event: dict) -> None:
        """Handle `response.output_item.added`: track the latest assistant
        item id for later truncation, and -- only for the first assistant
        item of each response, guarded on response id rather than firing
        per item -- reset the first-audio-of-this-reply flag and fire
        `on_reply_started`.

        Non-assistant output items (e.g. function-call items, which carry
        no `role`) are ignored entirely: they are not part of the spoken
        reply this session's callbacks describe.

        Args:
            event: The decoded event; `event["item"]["id"]`/`["role"]`
                describe the new output item.

        Returns:
            None.
        """
        item = event.get("item") or {}
        if item.get("role") != "assistant":
            return
        item_id = item.get("id")
        if item_id is not None:
            self._current_assistant_item_id = item_id
        if self._reply_started_for_response_id == self._current_response_id:
            return
        self._reply_started_for_response_id = self._current_response_id
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
        """Handle `response.done`: fire `on_reply_done` unless the response
        was client-cancelled, route a descriptive error to `on_error` (in
        addition to `on_reply_done`, so downstream still unwinds) if it
        failed, and fire `on_usage` if usage info is present.

        Status handling:
          - `"cancelled"`: no `on_reply_done` -- this is the barge-in path,
            the client already handled ending the reply locally when it
            called `cancel_response`; firing `on_reply_done` too would be
            a spurious second "reply finished" signal.
          - `"failed"`: routes a `RuntimeError` built from
            `response.status_details.error.message` to `on_error`, AND
            still fires `on_reply_done` -- callers that only unwind
            "reply in progress" UI state on `on_reply_done` must still see
            it even though the reply also failed.
          - `"completed"` (or an unset/unrecognized status): fires
            `on_reply_done` as the normal case.

        Marks no response active any more, but deliberately does NOT clear
        `_current_assistant_item_id` -- a `cancel_response` call arriving
        just after a response completes (audio still playing client-side)
        must still be able to truncate that item; this is the ordinary
        successful barge-in case, live-confirmed to succeed against a
        completed-but-still-playing item.

        Args:
            event: The decoded event; `event["response"]["status"]` drives
                the branching above.

        Returns:
            None.
        """
        response = event.get("response") or {}
        self._response_active = False
        status = response.get("status", "completed")

        if status == "cancelled":
            pass
        elif status == "failed":
            details = (response.get("status_details") or {}).get("error") or {}
            message = details.get("message", "response failed")
            exc = RuntimeError(f"OpenAI realtime response failed: {message}")
            self._safe_invoke(self._callbacks.on_error, exc, op="on_error")
            self._safe_invoke(self._callbacks.on_reply_done, op="on_reply_done")
        else:
            self._safe_invoke(self._callbacks.on_reply_done, op="on_reply_done")

        usage = response.get("usage")
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
        "response.created": _on_response_created,
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
