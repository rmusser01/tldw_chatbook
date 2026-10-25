import asyncio
import json
import struct

import pytest

from tldw_chatbook.Canvas.control_protocol import (
    CONTROL_PROTOCOL_VERSION,
    MAX_CONTROL_FRAME_BYTES,
    CanvasControlBroker,
    CanvasControlClient,
    ControlMessage,
    ControlProtocolError,
    decode_control_frame,
    encode_control_frame,
)

pytestmark = pytest.mark.loopback_network


def test_v3_auth_codec_retains_snapshot_identity():
    message = ControlMessage(
        version=CONTROL_PROTOCOL_VERSION,
        message_type="auth.request",
        request_id="auth-1",
        deadline_ms=None,
        payload={
            "child_id": "child-a",
            "secret": "a" * 64,
            "runtime_snapshot_id": "b" * 64,
        },
    )
    decoded = decode_control_frame(encode_control_frame(message)[4:])
    assert decoded.payload["runtime_snapshot_id"] == "b" * 64


@pytest.mark.parametrize("identity", [None, "", "b" * 63, "g" * 64, 1])
def test_auth_rejects_malformed_snapshot_identity(identity):
    with pytest.raises(ControlProtocolError, match="invalid_runtime_snapshot_id"):
        _message(
            "auth.request",
            {
                "child_id": "child-a",
                "secret": "a" * 64,
                "runtime_snapshot_id": identity,
            },
        )


def test_auth_rejects_missing_identity_and_legacy_version():
    with pytest.raises(ControlProtocolError, match="missing_payload_field"):
        _message("auth.request", {"child_id": "child-a", "secret": "a" * 64})
    with pytest.raises(ControlProtocolError, match="unsupported_version"):
        ControlMessage(
            1,
            "auth.request",
            "auth-old",
            None,
            {"child_id": "child-a", "secret": "a" * 64},
        )


def test_parent_refuses_foreign_snapshot():
    async def scenario():
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(launch.environment, runtime_snapshot_id="b" * 64)
        try:
            with pytest.raises(ControlProtocolError, match="runtime_snapshot_mismatch"):
                await client.start()
        finally:
            await client.aclose()
            await broker.aclose()

    asyncio.run(scenario())


def _message(message_type: str, payload: dict, *, request_id: str = "request-1"):
    return ControlMessage(
        version=CONTROL_PROTOCOL_VERSION,
        message_type=message_type,
        request_id=request_id,
        deadline_ms=None,
        payload=payload,
    )


@pytest.mark.parametrize("generation", [None, "", 3, "x" * 257])
def test_served_selection_requires_bounded_original_generation(generation):
    payload = {
        "action": "follow",
        "expected_session_id": "session-a",
        "expected_canvas_id": "canvas-a",
        "expected_revision_id": "revision-a",
        "expected_selection_generation": generation,
    }
    with pytest.raises(ControlProtocolError, match="invalid_payload_field"):
        _message("selection.request", payload)
    payload.pop("expected_selection_generation")
    with pytest.raises(ControlProtocolError, match="missing_payload_field"):
        _message("selection.request", payload)


def test_codec_round_trips_a_typed_health_request() -> None:
    message = _message("health.request", {})

    encoded = encode_control_frame(message)

    assert decode_control_frame(encoded[4:]) == message
    assert struct.unpack(">I", encoded[:4])[0] == len(encoded) - 4


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"version": 99}, "unsupported_version"),
        ({"type": "future.request"}, "unsupported_type"),
        ({"surprise": True}, "unknown_field"),
        ({"payload": {"surprise": True}}, "unknown_payload_field"),
    ],
)
def test_codec_rejects_unknown_versions_types_and_fields(mutation, code) -> None:
    wire = {
        "version": CONTROL_PROTOCOL_VERSION,
        "type": "health.request",
        "request_id": "request-1",
        "deadline_ms": None,
        "payload": {},
    }
    wire.update(mutation)

    with pytest.raises(ControlProtocolError, match=code):
        decode_control_frame(json.dumps(wire).encode("utf-8"))


def test_codec_rejects_oversized_frame_before_json_decode() -> None:
    with pytest.raises(ControlProtocolError, match="frame_too_large"):
        decode_control_frame(b" " * (MAX_CONTROL_FRAME_BYTES + 1))


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("version", True, "unsupported_version"),
        ("type", [], "unsupported_type"),
        ("request_id", [], "invalid_request_id"),
    ],
)
def test_codec_fails_closed_on_wrong_envelope_scalar_types(field, value, code) -> None:
    wire = {
        "version": CONTROL_PROTOCOL_VERSION,
        "type": "health.request",
        "request_id": "request-1",
        "deadline_ms": None,
        "payload": {},
    }
    wire[field] = value

    with pytest.raises(ControlProtocolError, match=code):
        decode_control_frame(json.dumps(wire).encode("utf-8"))


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (
            (
                b'{"version":3,"type":"health.request","request_id":"request-1",'
                b'"deadline_ms":' + (b"9" * 5000) + b',"payload":{}}'
            ),
            "invalid_json",
        ),
        (
            (
                b'{"version":3,"type":"bridge.request","request_id":"request-1",'
                b'"deadline_ms":null,"payload":{"request":'
                + (b"[" * 2000)
                + b"null"
                + (b"]" * 2000)
                + b"}}"
            ),
            "payload_too_deep",
        ),
    ],
    ids=["oversized-integer", "deep-json"],
)
def test_decoder_maps_json_resource_failures_to_bounded_errors(body, code) -> None:
    with pytest.raises(ControlProtocolError, match=code):
        decode_control_frame(body)


def test_error_frames_are_content_free_and_bounded() -> None:
    message = _message("control.error", {"code": "deadline_exceeded"})

    assert decode_control_frame(encode_control_frame(message)[4:]) == message
    with pytest.raises(ControlProtocolError, match="unknown_payload_field"):
        encode_control_frame(
            _message(
                "control.error",
                {"code": "operation_failed", "detail": "secret source text"},
            )
        )


@pytest.mark.parametrize(
    ("message_type", "payload"),
    [
        ("health.response", {"status": 1}),
        (
            "selection.response",
            {"canvas_id": "canvas-a", "revision_id": "revision-a", "following": "yes", "selection_generation": "intent-a"},
        ),
        (
            "canvas.events",
            {
                "event_id": "event-a",
                "kind": "updated",
                "canvas_id": "canvas-a",
                "revision_id": "revision-a",
                "metadata": [],
            },
        ),
    ],
)
def test_typed_payloads_reject_values_of_the_wrong_kind(message_type, payload) -> None:
    with pytest.raises(ControlProtocolError, match="invalid_payload_field"):
        ControlMessage(
            CONTROL_PROTOCOL_VERSION,
            message_type,
            "request-1",
            None,
            payload,
        )


def test_parent_rejects_out_of_order_response_types() -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")

        async def wrong_handler(message: ControlMessage) -> ControlMessage:
            assert message.message_type == "health.request"
            return _message(
                "canvas.list.response",
                {"canvases": []},
                request_id=message.request_id,
            )

        client = CanvasControlClient(
            launch.environment, runtime_snapshot_id="a" * 64, handler=wrong_handler
        )
        await client.start()
        await broker.wait_connected("child-a", timeout=1)
        try:
            with pytest.raises(ControlProtocolError, match="out_of_order_reply"):
                await broker.request("child-a", "health.request", {}, timeout=1)
        finally:
            await client.aclose()
            await broker.aclose()

    asyncio.run(scenario())


def test_child_timeout_cancels_work_and_releases_backpressure_slot() -> None:
    async def scenario() -> None:
        cancelled = asyncio.Event()

        async def handler(message: ControlMessage) -> ControlMessage:
            if message.message_type == "health.request":
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return _message(
                "health.response",
                {"status": "ok"},
                request_id=message.request_id,
            )

        broker = CanvasControlBroker(
            runtime_snapshot_id="a" * 64, max_pending_requests=1
        )
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(
            launch.environment, runtime_snapshot_id="a" * 64, handler=handler
        )
        await client.start()
        await broker.wait_connected("child-a", timeout=1)
        try:
            with pytest.raises(ControlProtocolError, match="deadline_exceeded"):
                await broker.request("child-a", "health.request", {}, timeout=0.02)
            await asyncio.wait_for(cancelled.wait(), timeout=1)
        finally:
            await client.aclose()
            await broker.aclose()

    asyncio.run(scenario())


def test_two_children_cannot_cross_auth_or_receive_each_others_events() -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch_a = broker.issue_child("child-a")
        launch_b = broker.issue_child("child-b")

        crossed = dict(launch_a.environment)
        crossed["CHATBOOK_CANVAS_CONTROL_SECRET"] = launch_b.environment[
            "CHATBOOK_CANVAS_CONTROL_SECRET"
        ]
        attacker = CanvasControlClient(crossed, runtime_snapshot_id="a" * 64)
        with pytest.raises(ControlProtocolError, match="authentication_failed"):
            await attacker.start()

        client_a = CanvasControlClient(
            launch_a.environment, runtime_snapshot_id="a" * 64
        )
        client_b = CanvasControlClient(
            launch_b.environment, runtime_snapshot_id="a" * 64
        )
        await client_a.start()
        await client_b.start()
        await broker.wait_connected("child-a", timeout=1)
        await broker.wait_connected("child-b", timeout=1)
        try:
            await client_a.send_event(
                {
                    "event_id": "event-a",
                    "kind": "updated",
                    "canvas_id": "canvas-a",
                    "revision_id": "revision-a",
                    "metadata": {},
                }
            )
            event = await broker.next_event("child-a", timeout=1)
            assert event.payload["canvas_id"] == "canvas-a"
            with pytest.raises(asyncio.TimeoutError):
                await broker.next_event("child-b", timeout=0.02)
        finally:
            await client_a.aclose()
            await client_b.aclose()
            await broker.aclose()

    asyncio.run(scenario())


def test_child_restart_rotates_and_revokes_the_previous_secret() -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        first = broker.issue_child("child-a")
        second = broker.issue_child("child-a")
        assert (
            first.environment["CHATBOOK_CANVAS_CONTROL_SECRET"]
            != second.environment["CHATBOOK_CANVAS_CONTROL_SECRET"]
        )
        stale = CanvasControlClient(first.environment, runtime_snapshot_id="a" * 64)
        with pytest.raises(ControlProtocolError, match="authentication_failed"):
            await stale.start()
        await broker.aclose()

    asyncio.run(scenario())


def test_launch_secret_cannot_be_replayed_after_a_disconnect() -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")
        first = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
        await first.start()
        await broker.wait_connected("child-a", timeout=1)
        await first.aclose()
        await asyncio.sleep(0)

        replay = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
        with pytest.raises(ControlProtocolError, match="authentication_failed"):
            await replay.start()
        await broker.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_secret", ["", "é" * 32])
def test_consumed_launch_rejects_a_raw_malformed_secret(invalid_secret) -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")
        first = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
        await first.start()
        await broker.wait_connected("child-a", timeout=1)
        await first.aclose()
        await asyncio.sleep(0)

        reader, writer = await asyncio.open_connection(
            launch.environment["CHATBOOK_CANVAS_CONTROL_HOST"],
            int(launch.environment["CHATBOOK_CANVAS_CONTROL_PORT"]),
        )
        writer.write(
            encode_control_frame(
                _message(
                    "auth.request",
                    {
                        "child_id": "child-a",
                        "secret": invalid_secret,
                        "runtime_snapshot_id": "a" * 64,
                    },
                    request_id="auth-empty",
                )
            )
        )
        await writer.drain()
        size = struct.unpack(">I", await reader.readexactly(4))[0]
        response = decode_control_frame(await reader.readexactly(size))
        assert response.message_type == "control.error"
        assert response.payload == {"code": "authentication_failed"}
        writer.close()
        await writer.wait_closed()
        await broker.aclose()

    asyncio.run(scenario())


def test_cancelling_parent_request_cancels_child_and_releases_pending_slot() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        calls = 0

        async def handler(message: ControlMessage) -> ControlMessage:
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return _message(
                "health.response", {"status": "ok"}, request_id=message.request_id
            )

        broker = CanvasControlBroker(
            runtime_snapshot_id="a" * 64, max_pending_requests=1
        )
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(
            launch.environment, runtime_snapshot_id="a" * 64, handler=handler
        )
        await client.start()
        await broker.wait_connected("child-a", timeout=1)
        try:
            request = asyncio.create_task(
                broker.request("child-a", "health.request", {}, timeout=5)
            )
            await asyncio.wait_for(entered.wait(), timeout=1)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            await asyncio.wait_for(cancelled.wait(), timeout=1)
            response = await broker.request("child-a", "health.request", {}, timeout=1)
            assert response.payload == {"status": "ok"}
        finally:
            await client.aclose()
            await broker.aclose()

    asyncio.run(scenario())


def test_late_response_after_timeout_does_not_disconnect_child() -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(message: ControlMessage) -> ControlMessage:
            nonlocal calls
            calls += 1
            if calls == 1:
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    # A non-cooperative authority can race one last response.
                    pass
            return _message(
                "health.response", {"status": "ok"}, request_id=message.request_id
            )

        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(
            launch.environment, runtime_snapshot_id="a" * 64, handler=handler
        )
        await client.start()
        await broker.wait_connected("child-a", timeout=1)
        try:
            with pytest.raises(ControlProtocolError, match="deadline_exceeded"):
                await broker.request("child-a", "health.request", {}, timeout=0.02)
            await asyncio.sleep(0.02)
            response = await broker.request("child-a", "health.request", {}, timeout=1)
            assert response.payload == {"status": "ok"}
        finally:
            await client.aclose()
            await broker.aclose()

    asyncio.run(scenario())


RELAY_APPROVAL_RAISED_PAYLOAD = {
    "round_id": "round-a",
    "session_id": "session-a",
    "calls": [
        {
            "name": "fs_write",
            "arguments": {"path": "notes/scratch.md", "content": "hi"},
            "options": {"mode": "overwrite"},
        }
    ],
    "timeout_seconds": 300,
}


@pytest.mark.parametrize(
    ("message_type", "payload"),
    [
        ("relay.approval.raised", dict(RELAY_APPROVAL_RAISED_PAYLOAD)),
        (
            "relay.approval.cleared",
            {"round_id": "round-a", "resolution": "approved"},
        ),
        ("relay.run.status", {"session_id": "session-a", "status": "completed"}),
        (
            "relay.decide.request",
            {"round_id": "round-a", "decisions": {"call-1": "approve"}},
        ),
        ("relay.decide.response", {"round_id": "round-a", "delivered": True}),
    ],
)
def test_relay_message_types_round_trip(message_type: str, payload: dict) -> None:
    message = _message(message_type, payload)

    encoded = encode_control_frame(message)

    assert decode_control_frame(encoded[4:]) == message


def test_relay_family_rejects_unregistered_message_types() -> None:
    with pytest.raises(ControlProtocolError, match="unsupported_type"):
        _message("relay.approval.expired", {"round_id": "round-a"})


@pytest.mark.parametrize(
    ("message_type", "payload"),
    [
        ("relay.approval.raised", {**RELAY_APPROVAL_RAISED_PAYLOAD, "round_id": 7}),
        (
            "relay.approval.raised",
            {**RELAY_APPROVAL_RAISED_PAYLOAD, "session_id": None},
        ),
        ("relay.approval.raised", {**RELAY_APPROVAL_RAISED_PAYLOAD, "calls": {}}),
        ("relay.approval.raised", {**RELAY_APPROVAL_RAISED_PAYLOAD, "calls": [7]}),
        (
            "relay.approval.raised",
            {**RELAY_APPROVAL_RAISED_PAYLOAD, "calls": ["fs_write"]},
        ),
        (
            "relay.approval.raised",
            {**RELAY_APPROVAL_RAISED_PAYLOAD, "timeout_seconds": True},
        ),
        (
            "relay.approval.raised",
            {**RELAY_APPROVAL_RAISED_PAYLOAD, "timeout_seconds": -1},
        ),
        ("relay.approval.cleared", {"round_id": "round-a", "resolution": 3}),
        ("relay.run.status", {"session_id": "session-a", "status": 2}),
        ("relay.decide.request", {"round_id": "round-a", "decisions": []}),
        (
            "relay.decide.request",
            {"round_id": "round-a", "decisions": "call-1=approve"},
        ),
        ("relay.decide.response", {"round_id": "round-a", "delivered": "yes"}),
        ("relay.decide.response", {"round_id": "round-a"}),
        (
            "relay.approval.cleared",
            {"round_id": "round-a", "resolution": "approved", "detail": "extra"},
        ),
    ],
)
def test_relay_payloads_reject_wrong_or_unknown_fields(
    message_type: str, payload: dict
) -> None:
    with pytest.raises(
        ControlProtocolError,
        match="missing_payload_field|invalid_payload_field|unknown_payload_field",
    ):
        _message(message_type, payload)


def test_parent_relays_decisions_to_child_and_receives_delivery() -> None:
    async def scenario() -> None:
        seen: list[ControlMessage] = []

        async def handler(message: ControlMessage) -> ControlMessage:
            seen.append(message)
            return _message(
                "relay.decide.response",
                {"round_id": message.payload["round_id"], "delivered": True},
                request_id=message.request_id,
            )

        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(
            launch.environment, runtime_snapshot_id="a" * 64, handler=handler
        )
        await client.start()
        await broker.wait_connected("child-a", timeout=1)
        try:
            response = await broker.request(
                "child-a",
                "relay.decide.request",
                {"round_id": "round-a", "decisions": {"call-1": "approve"}},
                timeout=1,
            )
            assert response.message_type == "relay.decide.response"
            assert response.payload == {"round_id": "round-a", "delivered": True}
            assert [message.message_type for message in seen] == [
                "relay.decide.request"
            ]
        finally:
            await client.aclose()
            await broker.aclose()

    asyncio.run(scenario())


def test_relay_events_are_queued_as_child_events_like_canvas_events() -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")

        reader, writer = await asyncio.open_connection(
            launch.environment["CHATBOOK_CANVAS_CONTROL_HOST"],
            int(launch.environment["CHATBOOK_CANVAS_CONTROL_PORT"]),
            limit=MAX_CONTROL_FRAME_BYTES + 4,
        )
        writer.write(
            encode_control_frame(
                _message(
                    "auth.request",
                    {
                        "child_id": "child-a",
                        "secret": launch.environment["CHATBOOK_CANVAS_CONTROL_SECRET"],
                        "runtime_snapshot_id": "a" * 64,
                    },
                    request_id="auth-relay",
                )
            )
        )
        await writer.drain()
        size = struct.unpack(">I", await reader.readexactly(4))[0]
        auth_response = decode_control_frame(await reader.readexactly(size))
        assert auth_response.message_type == "auth.response"

        relay_frames = [
            _message(
                "relay.approval.raised",
                dict(RELAY_APPROVAL_RAISED_PAYLOAD),
                request_id="relay-raised-1",
            ),
            _message(
                "relay.approval.cleared",
                {"round_id": "round-a", "resolution": "denied"},
                request_id="relay-cleared-1",
            ),
            _message(
                "relay.run.status",
                {"session_id": "session-a", "status": "completed"},
                request_id="relay-run-1",
            ),
        ]
        for frame in relay_frames:
            writer.write(encode_control_frame(frame))
        await writer.drain()

        try:
            for expected in relay_frames:
                event = await broker.next_event("child-a", timeout=1)
                assert event.message_type == expected.message_type
                assert event.payload == expected.payload
        finally:
            writer.close()
            await writer.wait_closed()
            await broker.aclose()

    asyncio.run(scenario())


def test_child_sends_relay_events_through_the_control_client() -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
        await client.start()
        await broker.wait_connected("child-a", timeout=1)
        try:
            await client.send_relay_event(
                "relay.approval.raised", dict(RELAY_APPROVAL_RAISED_PAYLOAD)
            )
            await client.send_relay_event(
                "relay.approval.cleared",
                {"round_id": "round-a", "resolution": "approved"},
            )
            await client.send_relay_event(
                "relay.run.status", {"session_id": "session-a", "status": "completed"}
            )
            expected = [
                ("relay.approval.raised", dict(RELAY_APPROVAL_RAISED_PAYLOAD)),
                (
                    "relay.approval.cleared",
                    {"round_id": "round-a", "resolution": "approved"},
                ),
                ("relay.run.status", {"session_id": "session-a", "status": "completed"}),
            ]
            for message_type, payload in expected:
                event = await broker.next_event("child-a", timeout=1)
                assert event.message_type == message_type
                assert event.payload == payload
            # Only the relay kinds ride this path; canvas.events stays on
            # send_event, and anything else is refused before the wire.
            with pytest.raises(
                ControlProtocolError, match="unsupported_relay_type"
            ):
                await client.send_relay_event(
                    "canvas.events",
                    {
                        "event_id": "e-1",
                        "kind": "updated",
                        "canvas_id": "canvas-a",
                        "revision_id": "revision-a",
                        "metadata": {},
                    },
                )
        finally:
            await client.aclose()
            await broker.aclose()

    asyncio.run(scenario())


def test_relay_send_requires_an_authenticated_connection() -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
        try:
            with pytest.raises(ControlProtocolError, match="client_not_connected"):
                await client.send_relay_event(
                    "relay.run.status", {"session_id": "session-a", "status": "idle"}
                )
        finally:
            await client.aclose()
            await broker.aclose()

    asyncio.run(scenario())


def test_broker_aclose_completes_while_the_client_is_still_connected() -> None:
    """Regression (2c review, real bug): ``aclose`` must revoke children —
    closing their server-side writers — BEFORE awaiting the server's
    ``wait_closed``.

    Every accepted connection's read loop parks in ``_read_message`` until
    its socket reaches EOF, and ``Server.wait_closed`` does not complete
    while any connection handler is still running, so awaiting the server
    first deadlocked whenever a client had not closed on its own (the E2E
    relay test's forced client-close-first ordering was the workaround).
    This test deliberately leaves the client's socket open and bounds
    ``aclose`` with a deadline instead: pre-fix it times out, post-fix the
    revocation unblocks the parked read loop and the close completes.
    """

    async def scenario() -> None:
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
        await client.start()
        await broker.wait_connected("child-a", timeout=1)
        try:
            # NO client close first: aclose must complete on its own.
            await asyncio.wait_for(broker.aclose(), timeout=5)
        finally:
            await client.aclose()

    asyncio.run(scenario())


def test_served_handler_answers_relay_decide_via_the_console_resolver() -> None:
    """ADR-236: the served child's REAL control handler resolves
    ``relay.decide.request`` through its wired console resolver.

    A real broker + real ``CanvasControlClient`` round-trip against
    ``ServedCanvasControlHandler.handle``: a phone decision on a desktop
    round reaches the stub console controller, and the parent's response
    reports ``delivered`` from the resolver's truthiness. The branch needs
    no canvas scope (the handler is deliberately unbound here).
    """

    from tldw_chatbook.Canvas.gateway import ServedCanvasControlHandler

    class _StubConsoleController:
        def __init__(self) -> None:
            self.resolved: list[tuple[str, dict[str, str]]] = []
            self.armed = {"round-a"}

        def resolve_pending_approval(
            self, decisions: dict[str, str], *, round_id: str | None = None
        ) -> bool:
            if round_id not in self.armed:
                return False
            self.resolved.append((str(round_id), dict(decisions)))
            return True

    async def scenario() -> None:
        stub = _StubConsoleController()
        handler = ServedCanvasControlHandler(
            relay_decide=lambda round_id, decisions: (
                stub.resolve_pending_approval(decisions, round_id=round_id)
            )
        )
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(
            launch.environment, runtime_snapshot_id="a" * 64, handler=handler.handle
        )
        await client.start()
        await broker.wait_connected("child-a", timeout=1)
        try:
            response = await broker.request(
                "child-a",
                "relay.decide.request",
                {"round_id": "round-a", "decisions": {"call-1": "approve_once"}},
                timeout=1,
            )
            assert response.message_type == "relay.decide.response"
            assert response.payload == {"round_id": "round-a", "delivered": True}
            assert stub.resolved == [("round-a", {"call-1": "approve_once"})]

            # An unknown round resolves nothing and answers delivered False.
            unknown = await broker.request(
                "child-a",
                "relay.decide.request",
                {"round_id": "round-unknown", "decisions": {"call-1": "deny"}},
                timeout=1,
            )
            assert unknown.message_type == "relay.decide.response"
            assert unknown.payload == {"round_id": "round-unknown", "delivered": False}
            assert stub.resolved == [("round-a", {"call-1": "approve_once"})]
        finally:
            await client.aclose()
            await broker.aclose()

    asyncio.run(scenario())


def test_served_handler_relay_decide_never_answers_an_error_frame() -> None:
    """No resolver wired, or the resolver raises → ``delivered: false``.

    The relay decide branch must never surface a ``control.error`` frame:
    the parent treats a failed delivery as fail-closed (the child's own
    timeout/deny semantics govern), and other message types keep the
    handler's existing error semantics (scope lookup, unsupported type).
    """

    async def scenario() -> None:
        from tldw_chatbook.Canvas.gateway import ServedCanvasControlHandler

        def _raising(round_id: str, decisions: dict[str, str]) -> bool:
            raise RuntimeError("controller vanished")

        unbound = ServedCanvasControlHandler()
        raising = ServedCanvasControlHandler(relay_decide=_raising)
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        try:
            for child_id, handler in (("child-a", unbound), ("child-b", raising)):
                launch = broker.issue_child(child_id)
                client = CanvasControlClient(
                    launch.environment,
                    runtime_snapshot_id="a" * 64,
                    handler=handler.handle,
                )
                await client.start()
                await broker.wait_connected(child_id, timeout=1)
                response = await broker.request(
                    child_id,
                    "relay.decide.request",
                    {"round_id": "round-a", "decisions": {"call-1": "deny"}},
                    timeout=1,
                )
                assert response.message_type == "relay.decide.response"
                assert response.payload == {"round_id": "round-a", "delivered": False}
                await client.aclose()
            # Existing semantics for other types are untouched: an
            # unbound handler still refuses scope-requiring requests.
            launch = broker.issue_child("child-c")
            client = CanvasControlClient(
                launch.environment, runtime_snapshot_id="a" * 64, handler=unbound.handle
            )
            await client.start()
            await broker.wait_connected("child-c", timeout=1)
            with pytest.raises(ControlProtocolError, match="scope_unavailable"):
                await broker.request(
                    "child-c",
                    "canvas.list.request",
                    {},
                    timeout=1,
                )
            await client.aclose()
        finally:
            await broker.aclose()

    asyncio.run(scenario())


def test_revoked_channel_reports_disconnect_and_rejects_lost_events() -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
        await client.start()
        await broker.wait_connected("child-a", timeout=1)

        await broker.revoke_child("child-a")
        await client.wait_disconnected(timeout=1)

        with pytest.raises(ControlProtocolError, match="client_not_connected"):
            await client.send_event(
                {
                    "event_id": "event-lost",
                    "kind": "updated",
                    "canvas_id": "canvas-a",
                    "revision_id": "revision-a",
                    "metadata": {},
                }
            )
        await client.aclose()
        await broker.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("version", [1, 2])
def test_private_v3_refuses_legacy_envelopes(version):
    with pytest.raises(ControlProtocolError, match="unsupported_version"):
        ControlMessage(version, "health.request", "request-a", None, {})


@pytest.mark.parametrize(
    "method, response_field",
    [
        ("snapshot", "snapshot"),
        ("events", "batch"),
        ("ticket", "ticket"),
        ("command", "receipt"),
        ("receipt", "receipt"),
    ],
)
def test_console_private_method_envelopes_are_bounded_closed_and_refuse_unsupported(
    method, response_field
):
    request_type = f"console.{method}.request"
    response_type = f"console.{method}.response"
    request = _message(request_type, {"request": {}})
    assert decode_control_frame(encode_control_frame(request)[4:]) == request
    response = _message(response_type, {response_field: {}})
    assert decode_control_frame(encode_control_frame(response)[4:]) == response
    with pytest.raises(ControlProtocolError, match="unknown_payload_field"):
        _message(request_type, {"request": {}, "path": "/private/profile"})
    with pytest.raises(ControlProtocolError, match="invalid_payload_field"):
        _message(request_type, {"request": []})

    async def scenario():
        broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(launch.environment, runtime_snapshot_id="a" * 64)
        try:
            await client.start()
            with pytest.raises(ControlProtocolError, match="authority_unavailable"):
                await broker.request(
                    "child-a", request_type, {"request": {}}, timeout=1
                )
            assert (
                await broker.request("child-a", "health.request", {}, timeout=1)
            ).payload["status"] == "ok"
        finally:
            await client.aclose()
            await broker.aclose()

    asyncio.run(scenario())
