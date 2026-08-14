"""Fake-server tests for `OpenAIRealtimeSession` (V4 task 2). See
`.superpowers/sdd/2026-08-04-realtime-voice-engine/task-2-brief.md`.

Runs a real `websockets.serve` server on `127.0.0.1:0` for each test and
drives it with a small scripted list of `("expect", predicate)` /
`("send", event_dict)` / `("expect_none", None)` / `("close", {...})` steps,
so `OpenAIRealtimeSession` is exercised against a real (if trivial) wire
protocol rather than a hand-mocked transport.

Event-name ground truth for this suite comes from the live probe run
recorded in `openai_realtime_probe.py` and `openai_session.py`'s header
comment block, not from the brief's a-priori expected list -- see that
header for the discrepancies (nested `session.update` schema, single-
modality `output_modalities` restriction).
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import Callable

import pytest

# `websockets` ships only in the `realtime` extra -- a base/dev install
# without it must skip this module's collection, not error out, matching
# the `pytest.importorskip` convention used by the ~16 other optional-dep
# test files in this suite (e.g. Tests/Subscriptions/
# test_briefing_audio_synthesis.py for `pydub`).
pytest.importorskip("websockets")
import websockets

from tldw_chatbook.LLM_Calls.realtime.openai_session import OpenAIRealtimeSession
from tldw_chatbook.LLM_Calls.realtime.protocol import (
    RealtimeCallbacks,
    RealtimeSession,
    RealtimeSessionConfig,
)

# Every case uses this module's in-process WebSocket server on 127.0.0.1.
pytestmark = pytest.mark.allow_network


_WIRE_STEP_TIMEOUT_SECONDS = 30
_SCRIPT_COMPLETION_TIMEOUT_SECONDS = 30


def _transport_safe_error(exc: Exception) -> AssertionError:
    """Copy exception diagnostics without retaining traceback locals."""
    return AssertionError(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Scripted fake server
# ---------------------------------------------------------------------------


class ScriptedServer:
    """Drives one WebSocket connection through a fixed script of steps.

    Each step is a `(kind, payload)` tuple:
      - `("expect", predicate)`: receive one frame, decode as JSON, assert
        `predicate(event)` is truthy (recording the event either way).
      - `("send", event_dict)`: send `event_dict` as JSON.
      - `("expect_none", None)`: assert no frame arrives within a short
        grace window (used to prove a message was *not* sent).
      - `("close", {"code": int, "reason": str})`: close the connection
        with the given code/reason.
    """

    def __init__(self, script: list[tuple[str, object]]) -> None:
        """Store the script to run against the first connected client.

        Args:
            script: Ordered list of `(kind, payload)` steps, see the class
                docstring for the supported kinds.
        """
        self.script = script
        self.received: list[dict] = []
        self.error: Exception | None = None
        self._done = asyncio.Event()

    async def handler(self, ws) -> None:
        """`websockets.serve` connection handler: runs `self.script`.

        Args:
            ws: The server-side connection for the newly connected client.

        Returns:
            None.
        """
        try:
            for kind, payload in self.script:
                if kind == "expect":
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=_WIRE_STEP_TIMEOUT_SECONDS
                    )
                    event = json.loads(raw)
                    self.received.append(event)
                    predicate: Callable[[dict], bool] = payload  # type: ignore[assignment]
                    if not predicate(event):
                        raise AssertionError(
                            f"scripted predicate rejected event: {event}"
                        )
                elif kind == "send":
                    await ws.send(json.dumps(payload))
                elif kind == "expect_none":
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.3)
                    except (TimeoutError, asyncio.TimeoutError):
                        continue
                    event = json.loads(raw)
                    raise AssertionError(
                        f"expected no further message, but got: {event}"
                    )
                elif kind == "close":
                    payload = payload or {}
                    await ws.close(
                        code=payload.get("code", 1000),
                        reason=payload.get("reason", ""),
                    )
                else:
                    raise ValueError(f"unknown script step kind: {kind!r}")
        except Exception as exc:  # noqa: BLE001 - captured for the test to re-raise
            # Retaining ``exc`` also retains its traceback frame and the live
            # ``ServerConnection`` local. pytest-xdist cannot serialize that
            # object when this suite runs in parallel, so preserve only the
            # diagnostic type and message in a transport-safe exception.
            self.error = _transport_safe_error(exc)
        finally:
            self._done.set()

    async def wait_done(
        self, timeout: float = _SCRIPT_COMPLETION_TIMEOUT_SECONDS
    ) -> None:
        """Wait for the script to finish and re-raise any error it hit.

        Args:
            timeout: Seconds to wait before giving up.

        Returns:
            None.

        Raises:
            Exception: Whatever the handler caught while running the
                script (assertion failures, timeouts, etc).
        """
        await asyncio.wait_for(self._done.wait(), timeout=timeout)
        if self.error is not None:
            raise self.error


@pytest.fixture
async def fake_server():
    """Yield a `(start, track)` pair for driving a scripted fake server.

    Returns:
        `start`: an async factory `start(script) -> (url, ScriptedServer)`
        that boots a fresh `websockets.serve` server on `127.0.0.1:0`.
        `track`: registers an `OpenAIRealtimeSession` for automatic
        `close()` during teardown. Both the server(s) and any tracked
        session(s) are torn down after the test.
    """
    servers: list = []
    sessions: list[OpenAIRealtimeSession] = []

    async def _start(script: list[tuple[str, object]]):
        scripted = ScriptedServer(script)
        server = await websockets.serve(scripted.handler, "127.0.0.1", 0)
        servers.append(server)
        host, port = server.sockets[0].getsockname()[:2]
        url = f"ws://{host}:{port}"
        return url, scripted

    def _track(session: OpenAIRealtimeSession) -> OpenAIRealtimeSession:
        sessions.append(session)
        return session

    yield _start, _track

    for session in sessions:
        try:
            await session.close()
        except Exception:
            pass
    for server in servers:
        server.close()
        await server.wait_closed()


def _make_is_session_update(
    input_rate: int = 24000, output_rate: int = 24000
) -> Callable[[dict], bool]:
    """Build a predicate asserting `event` is a well-formed `session.update`:
    single-modality (`["audio"]`, never `["audio", "text"]` together --
    live-confirmed the API rejects that combination), pcm16 audio in/out at
    the exact given rates (not merely present -- a swapped input/output
    rate must fail this predicate, not just a missing one), input
    transcription enabled with the exact live-confirmed model, and server
    VAD on.

    Args:
        input_rate: Expected `session.audio.input.format.rate`.
        output_rate: Expected `session.audio.output.format.rate`.

    Returns:
        A predicate `(dict) -> bool` for use as an `"expect"` script step.
    """

    def _predicate(event: dict) -> bool:
        if event.get("type") != "session.update":
            return False
        session = event.get("session", {})
        audio = session.get("audio", {})
        input_cfg = audio.get("input", {})
        output_cfg = audio.get("output", {})
        return (
            session.get("type") == "realtime"
            and session.get("output_modalities") == ["audio"]
            and input_cfg.get("format", {}).get("type") == "audio/pcm"
            and input_cfg.get("format", {}).get("rate") == input_rate
            and output_cfg.get("format", {}).get("type") == "audio/pcm"
            and output_cfg.get("format", {}).get("rate") == output_rate
            # Pinned to the literal, not to `openai_session._TRANSCRIPTION_
            # MODEL`: importing the production constant here would make
            # this assertion tautological (it would "pass" even if the
            # module quietly switched models with nothing re-confirming the
            # new one against the live API) -- see this module's header,
            # M9 (c): the fake previously only checked transcription was
            # enabled at all, not which model.
            and input_cfg.get("transcription", {}).get("model") == "whisper-1"
            # Either live-accepted mode -- this is the GENERIC handshake
            # matcher, and turn detection became configurable in gate
            # round 5. Still asserted (a missing or garbage block fails);
            # the exact block per mode is pinned by the turn-detection
            # tests at the bottom of this module.
            and input_cfg.get("turn_detection", {}).get("type")
            in {"server_vad", "semantic_vad"}
        )

    return _predicate


# Default predicate for the handshake helper and most tests: matches
# `_config()`'s default 24000/24000 rates.
_is_session_update = _make_is_session_update()


def _config(**overrides) -> RealtimeSessionConfig:
    """Build a `RealtimeSessionConfig` for tests, with sane defaults.

    Args:
        **overrides: Field overrides passed through to the constructor.

    Returns:
        A `RealtimeSessionConfig` suitable for pointing at the fake server.
    """
    defaults = dict(api_key="sk-test", model="gpt-realtime")
    defaults.update(overrides)
    return RealtimeSessionConfig(**defaults)


async def _connect_and_handshake(
    fake_server, extra_script, callbacks=None, config=None, handshake_predicate=None
):
    """Start a fake server, connect a session, and run the standard
    `session.update` -> `session.updated` handshake.

    Args:
        fake_server: The `(start, track)` tuple yielded by the `fake_server`
            fixture.
        extra_script: Script steps to run *after* the handshake pair.
        callbacks: `RealtimeCallbacks` to use, or None for a fresh empty one.
        config: `RealtimeSessionConfig` to use, or None for `_config()`.

    Returns:
        A `(session, scripted)` tuple: the connected, handshaken session
        and the `ScriptedServer` driving it.
    """
    start, track = fake_server
    callbacks = callbacks if callbacks is not None else RealtimeCallbacks()
    script = [
        ("expect", handshake_predicate or _is_session_update),
        ("send", {"type": "session.updated"}),
        *extra_script,
    ]
    url, scripted = await start(script)
    session = track(OpenAIRealtimeSession(config or _config(), callbacks, url=url))
    await session.connect()
    return session, scripted


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_transport_safe_error_discards_original_traceback():
    try:
        raise TimeoutError("wire receive exceeded its allowance")
    except TimeoutError as exc:
        assert exc.__traceback__ is not None
        safe_error = _transport_safe_error(exc)

    assert type(safe_error) is AssertionError
    assert safe_error.__traceback__ is None
    assert str(safe_error) == "TimeoutError: wire receive exceeded its allowance"


async def test_connect_sends_session_update_and_fires_ready(fake_server):
    fired = {"ready": 0}
    callbacks = RealtimeCallbacks(
        on_ready=lambda: fired.__setitem__("ready", fired["ready"] + 1)
    )
    session, scripted = await _connect_and_handshake(
        fake_server, [], callbacks=callbacks
    )
    assert isinstance(session, RealtimeSession)
    await scripted.wait_done()
    # `wait_done` only proves the server's script finished (it sent
    # session.updated); the client's recv loop processes that frame and
    # fires on_ready asynchronously, so poll briefly rather than assert
    # immediately.
    for _ in range(20):
        if fired["ready"]:
            break
        await asyncio.sleep(0.05)
    assert fired["ready"] == 1


async def test_append_audio_base64_roundtrip(fake_server):
    expected = b"\x01\x02" * 480
    session, scripted = await _connect_and_handshake(
        fake_server,
        [("expect", lambda e: e.get("type") == "input_audio_buffer.append")],
    )
    session.append_audio(expected)
    await scripted.wait_done()
    sent = scripted.received[-1]
    assert base64.b64decode(sent["audio"]) == expected


async def test_append_audio_from_foreign_thread_is_delivered(fake_server):
    """`append_audio` is called from the future mic-tap recorder thread, not
    the session's own event loop thread -- must marshal via
    `loop.call_soon_threadsafe` and still be delivered in order."""
    expected = b"\xaa\xbb" * 240
    session, scripted = await _connect_and_handshake(
        fake_server,
        [("expect", lambda e: e.get("type") == "input_audio_buffer.append")],
    )

    thread = threading.Thread(target=session.append_audio, args=(expected,))
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()

    await scripted.wait_done()
    sent = scripted.received[-1]
    assert base64.b64decode(sent["audio"]) == expected


async def test_audio_delta_decodes_to_bytes_and_first_audio_fires_once(fake_server):
    audio_calls: list[bytes] = []
    first_audio_calls = {"n": 0}
    reply_started: list[str] = []
    callbacks = RealtimeCallbacks(
        on_audio_delta=lambda b: audio_calls.append(b),
        on_first_audio=lambda: first_audio_calls.__setitem__(
            "n", first_audio_calls["n"] + 1
        ),
        on_reply_started=lambda item_id: reply_started.append(item_id),
    )
    chunk_a = base64.b64encode(b"\x01\x02\x03").decode("ascii")
    chunk_b = base64.b64encode(b"\x04\x05\x06").decode("ascii")
    _, scripted = await _connect_and_handshake(
        fake_server,
        [
            ("send", {"type": "response.created", "response": {"id": "resp-1"}}),
            (
                "send",
                {
                    "type": "response.output_item.added",
                    "item": {"id": "item-assistant-1", "role": "assistant"},
                },
            ),
            ("send", {"type": "response.output_audio.delta", "delta": chunk_a}),
            ("send", {"type": "response.output_audio.delta", "delta": chunk_b}),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert audio_calls == [b"\x01\x02\x03", b"\x04\x05\x06"]
    assert first_audio_calls["n"] == 1
    assert reply_started == ["item-assistant-1"]


async def test_transcripts_route_to_both_callbacks(fake_server):
    input_transcripts: list[str] = []
    output_deltas: list[str] = []
    callbacks = RealtimeCallbacks(
        on_input_transcript=lambda t: input_transcripts.append(t),
        on_output_transcript_delta=lambda t: output_deltas.append(t),
    )
    _, scripted = await _connect_and_handshake(
        fake_server,
        [
            (
                "send",
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "hello there",
                },
            ),
            (
                "send",
                {"type": "response.output_audio_transcript.delta", "delta": "hi "},
            ),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert input_transcripts == ["hello there"]
    assert output_deltas == ["hi "]


async def test_input_transcript_completed_with_usage_fires_on_transcription_usage(
    fake_server,
):
    """task-2363 / T2-F12: `conversation.item.input_audio_transcription.
    completed` carries its OWN `usage` field (`{"type": "duration",
    "seconds": N}`, live-confirmed -- see this module's header, USAGE
    section), entirely independent of `response.done`'s token usage. It
    previously reached nowhere at all -- `_on_input_transcript_completed`
    only ever read `transcript`."""
    usage_calls: list[dict] = []
    callbacks = RealtimeCallbacks(
        on_transcription_usage=lambda u: usage_calls.append(u),
    )
    _, scripted = await _connect_and_handshake(
        fake_server,
        [
            (
                "send",
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "hello there",
                    "usage": {"type": "duration", "seconds": 2},
                },
            ),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert usage_calls == [{"type": "duration", "seconds": 2}]


async def test_input_transcript_completed_without_usage_does_not_fire_transcription_usage(
    fake_server,
):
    """The event does not always carry `usage` -- must not fire the
    callback with None/garbage when it's simply absent."""
    usage_calls: list[dict] = []
    callbacks = RealtimeCallbacks(
        on_transcription_usage=lambda u: usage_calls.append(u),
    )
    _, scripted = await _connect_and_handshake(
        fake_server,
        [
            (
                "send",
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "hello there",
                },
            ),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert usage_calls == []


async def test_speech_started_fires_during_active_response(fake_server):
    speech_calls = {"n": 0}
    callbacks = RealtimeCallbacks(
        on_speech_started=lambda: speech_calls.__setitem__("n", speech_calls["n"] + 1)
    )
    _, scripted = await _connect_and_handshake(
        fake_server,
        [
            ("send", {"type": "response.created", "response": {"id": "resp-1"}}),
            (
                "send",
                {
                    "type": "response.output_item.added",
                    "item": {"id": "item-1", "role": "assistant"},
                },
            ),
            (
                "send",
                {
                    "type": "response.output_audio.delta",
                    "delta": base64.b64encode(b"\x00\x01").decode("ascii"),
                },
            ),
            ("send", {"type": "input_audio_buffer.speech_started"}),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert speech_calls["n"] == 1


async def test_input_audio_buffer_committed_fires_on_turn_committed(fake_server):
    """M9 (a): `_on_input_committed` dispatches `on_turn_committed` in
    production, but no fake script ever sent `input_audio_buffer.committed`
    -- deleting the dispatch line stayed green. Pinned here so it cannot."""
    committed_calls = {"n": 0}
    callbacks = RealtimeCallbacks(
        on_turn_committed=lambda: committed_calls.__setitem__(
            "n", committed_calls["n"] + 1
        )
    )
    _, scripted = await _connect_and_handshake(
        fake_server,
        [("send", {"type": "input_audio_buffer.committed"})],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert committed_calls["n"] == 1


async def test_cancel_response_sends_cancel_then_truncate_with_played_ms(fake_server):
    start, track = fake_server
    script = [
        ("expect", _is_session_update),
        ("send", {"type": "session.updated"}),
        ("send", {"type": "response.created", "response": {"id": "resp-1"}}),
        (
            "send",
            {
                "type": "response.output_item.added",
                "item": {"id": "item-current", "role": "assistant"},
            },
        ),
        ("expect", lambda e: e.get("type") == "response.cancel"),
        (
            "expect",
            lambda e: (
                e.get("type") == "conversation.item.truncate"
                and e.get("item_id") == "item-current"
                and e.get("audio_end_ms") == 1234
            ),
        ),
    ]
    url, scripted = await start(script)
    session = track(OpenAIRealtimeSession(_config(), RealtimeCallbacks(), url=url))
    await session.connect()

    # Give the recv loop a tick to process session.updated, response.created,
    # and the response.output_item.added send, so the session has both an
    # active response and the current assistant item id tracked before we
    # cancel.
    await asyncio.sleep(0.1)

    assert session.cancel_response(1234) is True

    await scripted.wait_done()


async def test_send_seed_creates_items_in_order_without_response(fake_server):
    session, scripted = await _connect_and_handshake(
        fake_server,
        [
            (
                "expect",
                lambda e: (
                    e.get("type") == "session.update"
                    # `session.type` is a REQUIRED field on every
                    # `session.update`, including this partial
                    # instructions-only one -- live-confirmed
                    # (`missing_required_parameter: session.type`, see this
                    # module's ground-truth header). Asserted here because
                    # without it the field could be deleted with this suite
                    # still green while the live endpoint rejected every
                    # seed the app ever sent (final review I3).
                    and e.get("session", {}).get("type") == "realtime"
                    and e.get("session", {}).get("instructions") == "Be nice."
                ),
            ),
            (
                "expect",
                lambda e: (
                    e.get("type") == "conversation.item.create"
                    and e["item"]["role"] == "user"
                    and e["item"]["content"][0]["text"] == "hi"
                ),
            ),
            (
                "expect",
                lambda e: (
                    e.get("type") == "conversation.item.create"
                    and e["item"]["role"] == "assistant"
                    and e["item"]["content"][0]["text"] == "hello"
                ),
            ),
            ("expect_none", None),
        ],
    )
    session.send_seed([("user", "hi"), ("assistant", "hello")], "Be nice.")
    await scripted.wait_done()


async def test_send_text_item_with_request_response_true_sends_response_create(
    fake_server,
):
    session, scripted = await _connect_and_handshake(
        fake_server,
        [
            (
                "expect",
                lambda e: (
                    e.get("type") == "conversation.item.create"
                    and e["item"]["role"] == "user"
                    and e["item"]["content"][0]["text"] == "how are you"
                ),
            ),
            ("expect", lambda e: e.get("type") == "response.create"),
        ],
    )
    session.send_text_item("how are you", request_response=True)
    await scripted.wait_done()


async def test_response_done_completed_fires_reply_done_and_usage(fake_server):
    reply_done_calls = {"n": 0}
    usage_calls: list[dict] = []
    callbacks = RealtimeCallbacks(
        on_reply_done=lambda: reply_done_calls.__setitem__(
            "n", reply_done_calls["n"] + 1
        ),
        on_usage=lambda u: usage_calls.append(u),
    )
    usage_payload = {"total_tokens": 42, "input_tokens": 10, "output_tokens": 32}
    _, scripted = await _connect_and_handshake(
        fake_server,
        [
            (
                "send",
                {
                    "type": "response.done",
                    "response": {"status": "completed", "usage": usage_payload},
                },
            ),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert reply_done_calls["n"] == 1
    assert usage_calls == [usage_payload]


async def test_response_done_cancelled_does_not_fire_reply_done(fake_server):
    """F5: a client-cancelled response (barge-in) must not also fire
    on_reply_done -- the client already ended the reply locally when it
    called cancel_response; a second "reply finished" signal is spurious."""
    reply_done_calls = {"n": 0}
    callbacks = RealtimeCallbacks(
        on_reply_done=lambda: reply_done_calls.__setitem__(
            "n", reply_done_calls["n"] + 1
        )
    )
    _, scripted = await _connect_and_handshake(
        fake_server,
        [
            ("send", {"type": "response.created", "response": {"id": "resp-1"}}),
            ("send", {"type": "response.done", "response": {"status": "cancelled"}}),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert reply_done_calls["n"] == 0


async def test_response_done_failed_routes_to_error_and_still_fires_reply_done(
    fake_server,
):
    """F5: a failed response must route a descriptive error to on_error AND
    still fire on_reply_done, since callers that only unwind "reply in
    progress" UI state on on_reply_done must still see it."""
    errors: list[Exception] = []
    reply_done_calls = {"n": 0}
    callbacks = RealtimeCallbacks(
        on_error=lambda exc: errors.append(exc),
        on_reply_done=lambda: reply_done_calls.__setitem__(
            "n", reply_done_calls["n"] + 1
        ),
    )
    _, scripted = await _connect_and_handshake(
        fake_server,
        [
            ("send", {"type": "response.created", "response": {"id": "resp-1"}}),
            (
                "send",
                {
                    "type": "response.done",
                    "response": {
                        "status": "failed",
                        "status_details": {"error": {"message": "rate limit exceeded"}},
                    },
                },
            ),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert len(errors) == 1
    assert "rate limit exceeded" in str(errors[0])
    assert reply_done_calls["n"] == 1


async def test_server_close_fires_on_closed_with_reason(fake_server):
    closed_reasons: list[str] = []
    callbacks = RealtimeCallbacks(
        on_closed=lambda reason: closed_reasons.append(reason)
    )
    _, scripted = await _connect_and_handshake(
        fake_server,
        [("close", {"code": 1001, "reason": "server-shutdown"})],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    for _ in range(20):
        if closed_reasons:
            break
        await asyncio.sleep(0.05)

    assert closed_reasons == ["server-shutdown"]


async def test_error_event_routes_to_on_error_not_crash(fake_server):
    errors: list[Exception] = []
    reply_done_calls = {"n": 0}
    callbacks = RealtimeCallbacks(
        on_error=lambda exc: errors.append(exc),
        on_reply_done=lambda: reply_done_calls.__setitem__(
            "n", reply_done_calls["n"] + 1
        ),
    )
    _, scripted = await _connect_and_handshake(
        fake_server,
        [
            (
                "send",
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "bad_thing",
                        "message": "boom",
                    },
                },
            ),
            ("send", {"type": "response.done", "response": {"status": "completed"}}),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert len(errors) == 1
    assert isinstance(errors[0], Exception)
    # The recv loop kept running after the error event -- proven by the
    # subsequent response.done still being processed.
    assert reply_done_calls["n"] == 1


async def test_callback_exception_is_isolated_and_routed_to_on_error(fake_server):
    """Context-note hard requirement: an exception in one callback must
    route to on_error and never kill the recv loop."""
    errors: list[Exception] = []

    def _boom(_b: bytes) -> None:
        raise RuntimeError("callback exploded")

    reply_done_calls = {"n": 0}
    callbacks = RealtimeCallbacks(
        on_audio_delta=_boom,
        on_error=lambda exc: errors.append(exc),
        on_reply_done=lambda: reply_done_calls.__setitem__(
            "n", reply_done_calls["n"] + 1
        ),
    )
    _, scripted = await _connect_and_handshake(
        fake_server,
        [
            (
                "send",
                {
                    "type": "response.output_audio.delta",
                    "delta": base64.b64encode(b"\x00\x01").decode("ascii"),
                },
            ),
            ("send", {"type": "response.done", "response": {"status": "completed"}}),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert len(errors) == 1
    assert "callback exploded" in str(errors[0])
    assert reply_done_calls["n"] == 1


async def test_bad_frame_does_not_kill_recv_loop_and_response_done_still_fires(
    fake_server,
):
    """F1: a well-formed JSON frame that isn't an object (e.g. a bare
    array) must not kill the recv loop with an AttributeError -- proven by
    a subsequent, well-formed response.done still firing on_reply_done."""
    reply_done_calls = {"n": 0}
    callbacks = RealtimeCallbacks(
        on_reply_done=lambda: reply_done_calls.__setitem__(
            "n", reply_done_calls["n"] + 1
        )
    )
    start, track = fake_server
    script = [
        ("expect", _is_session_update),
        ("send", {"type": "session.updated"}),
        ("send", [1, 2, 3]),
        ("send", {"type": "response.done", "response": {"status": "completed"}}),
    ]
    url, scripted = await start(script)
    session = track(OpenAIRealtimeSession(_config(), callbacks, url=url))
    await session.connect()
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert reply_done_calls["n"] == 1


async def test_append_audio_after_close_from_foreign_thread_does_not_raise_or_queue(
    fake_server,
):
    """F2: append_audio calling into a closed session from a foreign
    thread must not raise (recorder threads have no way to observe/handle
    a RuntimeError) and must not queue anything nobody will ever drain."""
    session, scripted = await _connect_and_handshake(fake_server, [])
    await scripted.wait_done()
    await session.close()

    errors: list[Exception] = []

    def _call() -> None:
        try:
            session.append_audio(b"\x01\x02")
        except Exception as exc:  # noqa: BLE001 - the property under test
            errors.append(exc)

    thread = threading.Thread(target=_call)
    thread.start()
    thread.join(timeout=5)
    # `thread.join()` only proves append_audio's own call returned (it just
    # schedules `call_soon_threadsafe` and returns immediately) -- the
    # scheduled callback itself needs the event loop to actually get a
    # turn before `qsize()` is a meaningful check, not a race that happens
    # to read the queue before the scheduled put ever runs.
    await asyncio.sleep(0.05)

    assert not thread.is_alive()
    assert errors == []
    assert session._outbound_queue is not None
    assert session._outbound_queue.qsize() == 0


async def test_connect_sends_correct_input_and_output_rates_without_swapping(
    fake_server,
):
    """F3: a config with distinct, non-default input/output rates must
    reach the wire as-is -- a swapped input/output rate bug would fail
    this predicate even though the old (rate-agnostic) predicate could
    not have caught it."""
    config = _config(input_sample_rate=16000, output_sample_rate=22050)
    start, track = fake_server
    predicate = _make_is_session_update(input_rate=16000, output_rate=22050)
    script = [("expect", predicate), ("send", {"type": "session.updated"})]
    url, scripted = await start(script)
    session = track(OpenAIRealtimeSession(config, RealtimeCallbacks(), url=url))
    await session.connect()
    await scripted.wait_done()


async def test_close_does_not_hang_when_sender_task_is_stalled(fake_server):
    """F7: close() must bound its wait for the sender task and close the
    transport regardless -- a stalled connection (send() never returns)
    must not hang teardown forever."""
    session, scripted = await _connect_and_handshake(fake_server, [])
    await scripted.wait_done()

    async def _hang_forever(_obj: dict) -> None:
        await asyncio.sleep(999)

    session._transport.send_json = _hang_forever
    session.append_audio(b"\x01\x02")
    await asyncio.sleep(0.05)  # let the sender loop pick it up and get stuck

    await asyncio.wait_for(session.close(), timeout=3.5)


async def test_sender_loop_death_marks_session_closed_and_fires_on_error_once(
    fake_server,
):
    """F6: when the sender loop dies (send() raises), the session must be
    marked closed so further sends stop silently vanishing into a queue
    nobody drains any more, and on_error must fire exactly once."""
    errors: list[Exception] = []
    callbacks = RealtimeCallbacks(on_error=lambda exc: errors.append(exc))
    session, scripted = await _connect_and_handshake(
        fake_server, [], callbacks=callbacks
    )
    await scripted.wait_done()

    async def _boom(_obj: dict) -> None:
        raise RuntimeError("transport send exploded")

    session._transport.send_json = _boom
    session.append_audio(b"\x01\x02")
    await asyncio.sleep(0.1)

    assert len(errors) == 1
    assert "transport send exploded" in str(errors[0])
    assert session._closed is True

    before = session._outbound_queue.qsize() if session._outbound_queue else 0
    session.append_audio(b"\x03\x04")
    await asyncio.sleep(0.05)
    after = session._outbound_queue.qsize() if session._outbound_queue else 0
    assert after == before


async def test_cancel_response_noops_when_no_response_active(fake_server):
    """F8: cancelling with no response ever started must not send
    response.cancel -- live-confirmed a stale/unmatched cancel produces an
    error event from the provider."""
    session, scripted = await _connect_and_handshake(
        fake_server,
        [("expect_none", None)],
    )
    # Nothing active AND no item ever tracked: nothing to cancel and
    # nothing to truncate, so this stays the true no-op. The return value
    # reports WHICH branch ran, so a caller can log "there was nothing to
    # cancel" instead of guessing; asserted here so the guard cannot be
    # neutered while the wire assertion stays green.
    assert session.cancel_response(500) is False
    await scripted.wait_done()


async def test_barge_in_after_response_done_still_truncates(fake_server):
    """A barge-in during the PLAYBACK DRAIN must still truncate.

    The wiring deliberately keeps the loop in `speaking` after
    `response.done` while the sink plays the buffered tail out (that is
    what stops the model hearing itself) -- so the most common barge-in of
    all, the user cutting off a reply they can still hear, arrives with
    `_response_active` already False. Skipping the truncate there leaves
    the provider believing the user heard the whole answer, which is the
    exact thing `played_ms` exists to prevent.

    Only `response.cancel` is skipped. This module's earlier note claimed
    the truncate would error too; a dedicated live probe (2026-08-04)
    disproved it -- truncating a just-completed item returns
    `conversation.item.truncated`, matching what `cancel_response`'s own
    docstring always said about a "just-completed-but-still-playing item".
    """
    session, scripted = await _connect_and_handshake(
        fake_server,
        [
            ("send", {"type": "response.created", "response": {"id": "resp-1"}}),
            (
                "send",
                {
                    "type": "response.output_item.added",
                    "item": {"id": "item-1", "role": "assistant"},
                },
            ),
            ("send", {"type": "response.done", "response": {"status": "completed"}}),
            (
                "expect",
                lambda e: (
                    e.get("type") == "conversation.item.truncate"
                    and e.get("item_id") == "item-1"
                    and e.get("audio_end_ms") == 500
                ),
            ),
            ("expect_none", None),
        ],
    )
    # Let the client fully process response.done (and flip _response_active
    # to False) before the drain-window barge-in.
    await asyncio.sleep(0.1)
    assert session.cancel_response(500) is True
    await scripted.wait_done()


async def test_output_item_added_ignores_non_assistant_role_items(fake_server):
    """F9: output items without role=="assistant" (e.g. function-call
    items) must not be treated as the start of a spoken reply."""
    reply_started: list[str] = []
    callbacks = RealtimeCallbacks(
        on_reply_started=lambda item_id: reply_started.append(item_id)
    )
    _, scripted = await _connect_and_handshake(
        fake_server,
        [
            ("send", {"type": "response.created", "response": {"id": "resp-1"}}),
            (
                "send",
                {
                    "type": "response.output_item.added",
                    "item": {"id": "item-fn", "type": "function_call"},
                },
            ),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert reply_started == []


async def test_output_item_added_only_resets_first_audio_once_per_response(
    fake_server,
):
    """F9: a second response.output_item.added for the SAME response must
    not re-fire on_reply_started or reset the first-audio flag, but must
    still retarget the tracked item id (so cancel_response truncates the
    item actually currently playing)."""
    reply_started: list[str] = []
    first_audio_calls = {"n": 0}
    callbacks = RealtimeCallbacks(
        on_reply_started=lambda item_id: reply_started.append(item_id),
        on_first_audio=lambda: first_audio_calls.__setitem__(
            "n", first_audio_calls["n"] + 1
        ),
    )
    chunk = base64.b64encode(b"\x00\x01").decode("ascii")
    session, scripted = await _connect_and_handshake(
        fake_server,
        [
            ("send", {"type": "response.created", "response": {"id": "resp-1"}}),
            (
                "send",
                {
                    "type": "response.output_item.added",
                    "item": {"id": "item-a", "role": "assistant"},
                },
            ),
            ("send", {"type": "response.output_audio.delta", "delta": chunk}),
            (
                "send",
                {
                    "type": "response.output_item.added",
                    "item": {"id": "item-b", "role": "assistant"},
                },
            ),
        ],
        callbacks=callbacks,
    )
    await scripted.wait_done()
    await asyncio.sleep(0.05)

    assert reply_started == ["item-a"]
    assert first_audio_calls["n"] == 1
    assert session._current_assistant_item_id == "item-b"


def test_safe_invoke_isolates_exceptions_from_on_error_itself_and_as_reporter():
    """Adjudication: _safe_invoke must isolate an exception whether it
    comes from on_error itself (case A: must not recurse) or from a
    different callback whose failure on_error itself then also fails to
    report (case B). Success is simply neither case raising."""
    session = OpenAIRealtimeSession(_config(), RealtimeCallbacks(), url="ws://unused")

    def _raise_on_error(_exc: Exception) -> None:
        raise RuntimeError("on_error exploded")

    session._callbacks.on_error = _raise_on_error
    session._safe_invoke(
        session._callbacks.on_error, RuntimeError("boom"), op="on_error"
    )

    def _raise_on_ready() -> None:
        raise RuntimeError("on_ready exploded")

    session._callbacks.on_ready = _raise_on_ready
    session._safe_invoke(session._callbacks.on_ready, op="on_ready")


# ---------------------------------------------------------------------------
# Voice (M9 (b))
#
# `_build_session_update` sends `voice` under `session.audio.output` when
# configured, but no fake script asserted it -- deleting the `output["voice"]
# = ...` line stayed green.
# ---------------------------------------------------------------------------


def _audio_output_of(event: dict) -> dict:
    return event.get("session", {}).get("audio", {}).get("output", {})


async def test_configured_voice_is_sent_under_audio_output(fake_server):
    seen: list[dict] = []

    def _capture(event: dict) -> bool:
        seen.append(_audio_output_of(event))
        return _is_session_update(event)

    session, scripted = await _connect_and_handshake(
        fake_server, [], config=_config(voice="marin"), handshake_predicate=_capture
    )
    await scripted.wait_done()
    assert seen[0].get("voice") == "marin"


async def test_unset_voice_omits_the_voice_key(fake_server):
    """An unconfigured voice is the provider's to choose -- proven by the
    key being ABSENT, not merely falsy/empty, mirroring how the
    turn-detection knobs are omitted rather than defaulted."""
    seen: list[dict] = []

    def _capture(event: dict) -> bool:
        seen.append(_audio_output_of(event))
        return _is_session_update(event)

    session, scripted = await _connect_and_handshake(
        fake_server, [], handshake_predicate=_capture
    )
    await scripted.wait_done()
    assert "voice" not in seen[0]


# ---------------------------------------------------------------------------
# Turn detection (gate round 5)
#
# Shapes established by the live probe FIRST
# (`openai_realtime_turn_detection_probe.py`, run against the GA endpoint
# with the repo key) -- see this module's ground-truth header. The two
# modes take DISJOINT fields: `semantic_vad` + `threshold` is rejected
# `unknown_parameter`, so sending the server_vad knobs in semantic mode
# would take down the whole handshake.
# ---------------------------------------------------------------------------


def _turn_detection_of(event: dict) -> dict:
    return (
        event.get("session", {})
        .get("audio", {})
        .get("input", {})
        .get("turn_detection", {})
    )


async def test_semantic_vad_sends_the_bare_semantic_block(fake_server):
    seen: list[dict] = []

    def _capture(event: dict) -> bool:
        seen.append(_turn_detection_of(event))
        return _is_session_update(event)

    session, scripted = await _connect_and_handshake(
        fake_server,
        [],
        config=_config(turn_detection="semantic_vad"),
        handshake_predicate=_capture,
    )
    await scripted.wait_done()
    assert seen[0] == {"type": "semantic_vad"}


async def test_semantic_vad_never_carries_the_server_vad_knobs(fake_server):
    """Live-rejected (`unknown_parameter`): a threshold configured while
    semantic mode is selected must be DROPPED, not forwarded."""
    seen: list[dict] = []

    def _capture(event: dict) -> bool:
        seen.append(_turn_detection_of(event))
        return _is_session_update(event)

    session, scripted = await _connect_and_handshake(
        fake_server,
        [],
        config=_config(
            turn_detection="semantic_vad", vad_threshold=0.6, vad_silence_ms=700
        ),
        handshake_predicate=_capture,
    )
    await scripted.wait_done()
    assert seen[0] == {"type": "semantic_vad"}


async def test_server_vad_sends_only_the_knobs_that_are_set(fake_server):
    seen: list[dict] = []

    def _capture(event: dict) -> bool:
        seen.append(_turn_detection_of(event))
        return _is_session_update(event)

    session, scripted = await _connect_and_handshake(
        fake_server,
        [],
        config=_config(turn_detection="server_vad", vad_silence_ms=700),
        handshake_predicate=_capture,
    )
    await scripted.wait_done()
    # `threshold` is absent, not defaulted: an unset knob is the
    # provider's to choose.
    assert seen[0] == {"type": "server_vad", "silence_duration_ms": 700}


async def test_server_vad_carries_both_knobs_when_both_are_set(fake_server):
    seen: list[dict] = []

    def _capture(event: dict) -> bool:
        seen.append(_turn_detection_of(event))
        return _is_session_update(event)

    session, scripted = await _connect_and_handshake(
        fake_server,
        [],
        config=_config(
            turn_detection="server_vad", vad_threshold=0.6, vad_silence_ms=700
        ),
        handshake_predicate=_capture,
    )
    await scripted.wait_done()
    assert seen[0] == {
        "type": "server_vad",
        "threshold": 0.6,
        "silence_duration_ms": 700,
    }


async def test_default_config_sends_the_providers_own_mode(fake_server):
    """The session layer describes the provider, it does not editorialize:
    an unconfigured `RealtimeSessionConfig` sends `server_vad`, which IS
    the provider's default. The app's product default (`semantic_vad`) is
    chosen one level up, by `console_voice_input.realtime_turn_detection()`
    -- pinned here so the two cannot be confused for each other."""
    seen: list[dict] = []

    def _capture(event: dict) -> bool:
        seen.append(_turn_detection_of(event))
        return _is_session_update(event)

    session, scripted = await _connect_and_handshake(
        fake_server, [], handshake_predicate=_capture
    )
    await scripted.wait_done()
    assert seen[0] == {"type": "server_vad"}
