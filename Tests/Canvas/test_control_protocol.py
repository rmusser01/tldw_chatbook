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
                b'{"version":1,"type":"health.request","request_id":"request-1",'
                b'"deadline_ms":' + (b"9" * 5000) + b',"payload":{}}'
            ),
            "invalid_json",
        ),
        (
            (
                b'{"version":1,"type":"bridge.request","request_id":"request-1",'
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
        broker = CanvasControlBroker()
        await broker.start()
        launch = broker.issue_child("child-a")

        async def wrong_handler(message: ControlMessage) -> ControlMessage:
            assert message.message_type == "health.request"
            return _message(
                "canvas.list.response",
                {"canvases": []},
                request_id=message.request_id,
            )

        client = CanvasControlClient(launch.environment, handler=wrong_handler)
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

        broker = CanvasControlBroker(max_pending_requests=1)
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(launch.environment, handler=handler)
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
        broker = CanvasControlBroker()
        await broker.start()
        launch_a = broker.issue_child("child-a")
        launch_b = broker.issue_child("child-b")

        crossed = dict(launch_a.environment)
        crossed["CHATBOOK_CANVAS_CONTROL_SECRET"] = launch_b.environment[
            "CHATBOOK_CANVAS_CONTROL_SECRET"
        ]
        attacker = CanvasControlClient(crossed)
        with pytest.raises(ControlProtocolError, match="authentication_failed"):
            await attacker.start()

        client_a = CanvasControlClient(launch_a.environment)
        client_b = CanvasControlClient(launch_b.environment)
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
        broker = CanvasControlBroker()
        await broker.start()
        first = broker.issue_child("child-a")
        second = broker.issue_child("child-a")
        assert (
            first.environment["CHATBOOK_CANVAS_CONTROL_SECRET"]
            != second.environment["CHATBOOK_CANVAS_CONTROL_SECRET"]
        )
        stale = CanvasControlClient(first.environment)
        with pytest.raises(ControlProtocolError, match="authentication_failed"):
            await stale.start()
        await broker.aclose()

    asyncio.run(scenario())


def test_launch_secret_cannot_be_replayed_after_a_disconnect() -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker()
        await broker.start()
        launch = broker.issue_child("child-a")
        first = CanvasControlClient(launch.environment)
        await first.start()
        await broker.wait_connected("child-a", timeout=1)
        await first.aclose()
        await asyncio.sleep(0)

        replay = CanvasControlClient(launch.environment)
        with pytest.raises(ControlProtocolError, match="authentication_failed"):
            await replay.start()
        await broker.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_secret", ["", "é" * 32])
def test_consumed_launch_rejects_a_raw_malformed_secret(invalid_secret) -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker()
        await broker.start()
        launch = broker.issue_child("child-a")
        first = CanvasControlClient(launch.environment)
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
                    {"child_id": "child-a", "secret": invalid_secret},
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

        broker = CanvasControlBroker(max_pending_requests=1)
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(launch.environment, handler=handler)
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

        broker = CanvasControlBroker()
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(launch.environment, handler=handler)
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


def test_revoked_channel_reports_disconnect_and_rejects_lost_events() -> None:
    async def scenario() -> None:
        broker = CanvasControlBroker()
        await broker.start()
        launch = broker.issue_child("child-a")
        client = CanvasControlClient(launch.environment)
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
