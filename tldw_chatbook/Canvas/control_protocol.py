"""Private authenticated control channel for served Canvas sessions.

The textual-serve parent is deliberately only a bounded transport.  One
Chatbook child owns conversation scope and Canvas authority; the per-child
secret carried in its spawn environment only lets that child attach to its
matching transport slot.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import secrets
import struct
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .limits import CanvasLimits, validate_opaque_identifier

CONTROL_PROTOCOL_VERSION = 1
# A generated download may carry 10 MiB of decoded bytes as a base64 data URL.
# Keep the private frame closed to that documented V1 ceiling plus a small JSON
# envelope allowance; this is not an unbounded general-purpose transport.
MAX_CONTROL_VALUE_BYTES = (
    (CanvasLimits().download_payload_bytes + 2) // 3
) * 4 + 64 * 1024
MAX_CONTROL_FRAME_BYTES = MAX_CONTROL_VALUE_BYTES + 256 * 1024
DEFAULT_MAX_PENDING_REQUESTS = 32
DEFAULT_MAX_QUEUED_EVENTS = 64

_ENV_HOST = "CHATBOOK_CANVAS_CONTROL_HOST"
_ENV_PORT = "CHATBOOK_CANVAS_CONTROL_PORT"
_ENV_CHILD_ID = "CHATBOOK_CANVAS_CONTROL_CHILD_ID"
_ENV_SECRET = "CHATBOOK_CANVAS_CONTROL_SECRET"
_ENV_VERSION = "CHATBOOK_CANVAS_CONTROL_VERSION"

_ENVELOPE_FIELDS = frozenset(
    {"version", "type", "request_id", "deadline_ms", "payload"}
)
_REQUEST_REPLY = {
    "scope.snapshot.request": "scope.snapshot.response",
    "canvas.list.request": "canvas.list.response",
    "canvas.read.request": "canvas.read.response",
    "selection.request": "selection.response",
    "canvas.events.request": "canvas.events.response",
    "bridge.request": "bridge.response",
    "bridge.decision.request": "bridge.decision.response",
    "health.request": "health.response",
    "shutdown.request": "shutdown.response",
}
_RESPONSE_TYPES = frozenset(_REQUEST_REPLY.values())
_REQUEST_TYPES = frozenset(_REQUEST_REPLY)
_MESSAGE_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "auth.request": (frozenset({"child_id", "secret"}), frozenset()),
    "auth.response": (frozenset({"status"}), frozenset()),
    "scope.snapshot.request": (frozenset(), frozenset()),
    "scope.snapshot.response": (
        frozenset(
            {
                "session_id",
                "conversation_id",
                "active_message_ids",
                "selected_canvas_id",
                "selected_revision_id",
                "run_id",
                "selection_generation",
            }
        ),
        frozenset(),
    ),
    "canvas.list.request": (frozenset(), frozenset()),
    "canvas.list.response": (frozenset({"canvases"}), frozenset()),
    "canvas.read.request": (
        frozenset({"canvas_id"}),
        frozenset({"revision_id"}),
    ),
    "canvas.read.response": (
        frozenset(
            {
                "canvas_id",
                "revision_id",
                "title",
                "content_sha256",
                "source_bytes",
                "render_metadata",
            }
        ),
        frozenset(),
    ),
    "selection.request": (
        frozenset({"action", "expected_session_id", "expected_canvas_id", "expected_revision_id", "expected_selection_generation"}),
        frozenset({"canvas_id", "revision_id", "title"}),
    ),
    "selection.response": (
        frozenset({"canvas_id", "revision_id", "following", "selection_generation"}),
        frozenset(),
    ),
    "canvas.events": (
        frozenset({"event_id", "kind", "canvas_id", "revision_id", "metadata"}),
        frozenset(),
    ),
    "canvas.events.request": (frozenset(), frozenset({"after_event_id"})),
    "canvas.events.response": (frozenset({"events"}), frozenset()),
    "bridge.request": (frozenset({"request"}), frozenset()),
    "bridge.response": (
        frozenset({"request_id", "preparation_nonce", "presentation"}),
        frozenset(),
    ),
    "bridge.decision.request": (
        frozenset({"request_id", "preparation_nonce", "approved"}),
        frozenset(),
    ),
    "bridge.decision.response": (
        frozenset({"request_id", "status"}),
        frozenset(),
    ),
    "health.request": (frozenset(), frozenset()),
    "health.response": (frozenset({"status"}), frozenset()),
    "shutdown.request": (frozenset(), frozenset()),
    "shutdown.response": (frozenset({"status"}), frozenset()),
    "control.cancel": (frozenset({"request_id"}), frozenset()),
    "control.error": (frozenset({"code"}), frozenset()),
}


class ControlProtocolError(RuntimeError):
    """Bounded protocol failure identified only by a stable error code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not (1 <= len(value) <= 200):
        raise ControlProtocolError(f"invalid_{field_name}")
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise ControlProtocolError(f"invalid_{field_name}")
    return value


def _secret_bytes(value: object) -> bytes | None:
    """Return the bounded ASCII secret representation accepted by compare_digest."""

    if not isinstance(value, str) or not (32 <= len(value) <= 128):
        return None
    try:
        return value.encode("ascii")
    except UnicodeEncodeError:
        return None


def _validate_json(value: object, *, depth: int = 0) -> None:
    if depth > 16:
        raise ControlProtocolError("payload_too_deep")
    if value is None or isinstance(value, (str, bool, int)):
        if (
            isinstance(value, str)
            and len(value.encode("utf-8")) > MAX_CONTROL_VALUE_BYTES
        ):
            raise ControlProtocolError("payload_value_too_large")
        return
    if isinstance(value, list):
        if len(value) > 5000:
            raise ControlProtocolError("payload_collection_too_large")
        for child in value:
            _validate_json(child, depth=depth + 1)
        return
    if isinstance(value, Mapping):
        if len(value) > 5000:
            raise ControlProtocolError("payload_collection_too_large")
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > 100:
                raise ControlProtocolError("invalid_payload_key")
            _validate_json(child, depth=depth + 1)
        return
    raise ControlProtocolError("invalid_payload_value")


def _validate_payload(message_type: str, payload: object) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ControlProtocolError("invalid_payload")
    required, optional = _MESSAGE_FIELDS[message_type]
    keys = frozenset(payload)
    if missing := required - keys:
        del missing
        raise ControlProtocolError("missing_payload_field")
    if keys - required - optional:
        raise ControlProtocolError("unknown_payload_field")
    copied = dict(payload)
    _validate_json(copied)
    _validate_typed_fields(copied)
    return MappingProxyType(copied)


def _validate_typed_fields(payload: Mapping[str, Any]) -> None:
    """Apply the common scalar/container types used by protocol payloads."""

    string_fields = {
        "child_id",
        "secret",
        "status",
        "session_id",
        "conversation_id",
        "run_id",
        "canvas_id",
        "revision_id",
        "title",
        "content_sha256",
        "action",
        "event_id",
        "kind",
        "request_id",
        "preparation_nonce",
        "code",
    }
    nullable_string_fields = {
        "selected_canvas_id",
        "selected_revision_id",
        "after_event_id",
    }
    mapping_fields = {"render_metadata", "metadata", "request", "presentation"}
    for key, value in payload.items():
        if key in {
            "selection_generation",
            "expected_selection_generation",
            "expected_session_id",
            "expected_canvas_id",
            "expected_revision_id",
        }:
            try:
                validate_opaque_identifier(value, field_name="selection expectation")
            except ValueError:
                raise ControlProtocolError("invalid_payload_field") from None
        if key in string_fields and not isinstance(value, str):
            raise ControlProtocolError("invalid_payload_field")
        if (
            key in nullable_string_fields
            and value is not None
            and not isinstance(value, str)
        ):
            raise ControlProtocolError("invalid_payload_field")
        if key in mapping_fields and not isinstance(value, Mapping):
            raise ControlProtocolError("invalid_payload_field")
        if key in {"following", "approved"} and type(value) is not bool:
            raise ControlProtocolError("invalid_payload_field")
        if key in {"events", "canvases"} and not isinstance(value, list):
            raise ControlProtocolError("invalid_payload_field")
        if key == "source_bytes" and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ControlProtocolError("invalid_payload_field")
        if key == "active_message_ids" and (
            not isinstance(value, list)
            or any(not isinstance(item, str) for item in value)
        ):
            raise ControlProtocolError("invalid_payload_field")


@dataclass(frozen=True, slots=True)
class ControlMessage:
    """One strictly decoded protocol envelope."""

    version: int
    message_type: str
    request_id: str
    deadline_ms: int | None
    payload: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != CONTROL_PROTOCOL_VERSION:
            raise ControlProtocolError("unsupported_version")
        if (
            not isinstance(self.message_type, str)
            or self.message_type not in _MESSAGE_FIELDS
        ):
            raise ControlProtocolError("unsupported_type")
        _identifier(self.request_id, "request_id")
        if self.deadline_ms is not None and (
            not isinstance(self.deadline_ms, int)
            or isinstance(self.deadline_ms, bool)
            or self.deadline_ms <= 0
        ):
            raise ControlProtocolError("invalid_deadline")
        object.__setattr__(
            self, "payload", _validate_payload(self.message_type, self.payload)
        )


def encode_control_frame(message: ControlMessage) -> bytes:
    """Encode one message with a four-byte network-order length prefix."""

    if not isinstance(message, ControlMessage):
        raise TypeError("message must be a ControlMessage")
    body = json.dumps(
        {
            "version": message.version,
            "type": message.message_type,
            "request_id": message.request_id,
            "deadline_ms": message.deadline_ms,
            "payload": dict(message.payload),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(body) > MAX_CONTROL_FRAME_BYTES:
        raise ControlProtocolError("frame_too_large")
    return struct.pack(">I", len(body)) + body


def decode_control_frame(body: bytes) -> ControlMessage:
    """Decode and strictly validate one frame body."""

    if not isinstance(body, bytes):
        raise TypeError("body must be bytes")
    if len(body) > MAX_CONTROL_FRAME_BYTES:
        raise ControlProtocolError("frame_too_large")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise ControlProtocolError("invalid_json") from None
    if not isinstance(value, dict):
        raise ControlProtocolError("invalid_envelope")
    if set(value) - _ENVELOPE_FIELDS:
        raise ControlProtocolError("unknown_field")
    if set(value) != _ENVELOPE_FIELDS:
        raise ControlProtocolError("missing_field")
    return ControlMessage(
        version=value["version"],
        message_type=value["type"],
        request_id=value["request_id"],
        deadline_ms=value["deadline_ms"],
        payload=value["payload"],
    )


async def _read_message(reader: asyncio.StreamReader) -> ControlMessage:
    try:
        header = await reader.readexactly(4)
        size = struct.unpack(">I", header)[0]
        if size > MAX_CONTROL_FRAME_BYTES:
            raise ControlProtocolError("frame_too_large")
        return decode_control_frame(await reader.readexactly(size))
    except asyncio.IncompleteReadError:
        raise ControlProtocolError("connection_closed") from None


async def _write_message(
    writer: asyncio.StreamWriter,
    lock: asyncio.Lock,
    message: ControlMessage,
) -> None:
    encoded = encode_control_frame(message)
    async with lock:
        writer.write(encoded)
        try:
            await writer.drain()
        except (ConnectionError, RuntimeError):
            raise ControlProtocolError("connection_closed") from None


@dataclass(frozen=True, slots=True, repr=False)
class ChildControlLaunch:
    """Secret-bearing child spawn data; repr never exposes its values."""

    environment: Mapping[str, str] = field(repr=False)

    def __repr__(self) -> str:
        return "ChildControlLaunch(environment=<redacted>)"


@dataclass(slots=True)
class _PendingReply:
    expected_type: str
    future: asyncio.Future[ControlMessage]


@dataclass(slots=True)
class _ChildState:
    child_id: str
    secret: str = field(repr=False)
    generation: str
    connected: asyncio.Event = field(default_factory=asyncio.Event)
    writer: asyncio.StreamWriter | None = field(default=None, repr=False)
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    pending: dict[str, _PendingReply] = field(default_factory=dict, repr=False)
    cancelled_ids: OrderedDict[str, None] = field(
        default_factory=OrderedDict, repr=False
    )
    events: asyncio.Queue[ControlMessage] = field(
        repr=False, default_factory=asyncio.Queue
    )


class CanvasControlBroker:
    """Parent-owned loopback transport with isolated per-child state."""

    def __init__(
        self,
        *,
        max_pending_requests: int = DEFAULT_MAX_PENDING_REQUESTS,
        max_queued_events: int = DEFAULT_MAX_QUEUED_EVENTS,
    ) -> None:
        if not 1 <= max_pending_requests <= 256:
            raise ValueError("invalid pending-request limit")
        if not 1 <= max_queued_events <= 1024:
            raise ValueError("invalid event-queue limit")
        self._max_pending_requests = max_pending_requests
        self._max_queued_events = max_queued_events
        self._server: asyncio.AbstractServer | None = None
        self._host = "127.0.0.1"
        self._port: int | None = None
        self._children: dict[str, _ChildState] = {}
        self._connection_tasks: set[asyncio.Task[Any]] = set()

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await asyncio.start_server(
            self._accept_connection,
            host=self._host,
            port=0,
            limit=MAX_CONTROL_FRAME_BYTES + 4,
        )
        socket = self._server.sockets[0]
        address = socket.getsockname()
        if not ipaddress.ip_address(address[0]).is_loopback:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            raise ControlProtocolError("non_loopback_listener")
        self._port = int(address[1])

    def issue_child(self, child_id: str) -> ChildControlLaunch:
        """Mint a fresh launch secret, revoking any earlier incarnation."""

        _identifier(child_id, "child_id")
        if self._server is None or self._port is None:
            raise ControlProtocolError("broker_not_started")
        previous = self._children.get(child_id)
        if previous is not None:
            self._revoke_state(previous, "child_restarted")
        state = _ChildState(
            child_id=child_id,
            secret=secrets.token_urlsafe(32),
            generation=uuid4().hex,
            events=asyncio.Queue(maxsize=self._max_queued_events),
        )
        self._children[child_id] = state
        return ChildControlLaunch(
            MappingProxyType(
                {
                    _ENV_HOST: self._host,
                    _ENV_PORT: str(self._port),
                    _ENV_CHILD_ID: child_id,
                    _ENV_SECRET: state.secret,
                    _ENV_VERSION: str(CONTROL_PROTOCOL_VERSION),
                }
            )
        )

    async def revoke_child(self, child_id: str) -> None:
        state = self._children.pop(child_id, None)
        if state is not None:
            self._revoke_state(state, "child_revoked")

    def _revoke_state(self, state: _ChildState, code: str) -> None:
        state.secret = ""
        state.connected.clear()
        for pending in state.pending.values():
            if not pending.future.done():
                pending.future.set_exception(ControlProtocolError(code))
        state.pending.clear()
        state.cancelled_ids.clear()
        if state.writer is not None:
            state.writer.close()
            state.writer = None

    async def wait_connected(self, child_id: str, *, timeout: float) -> None:
        state = self._children.get(child_id)
        if state is None:
            raise ControlProtocolError("unknown_child")
        try:
            await asyncio.wait_for(state.connected.wait(), timeout)
        except TimeoutError:
            raise ControlProtocolError("connection_deadline_exceeded") from None

    async def request(
        self,
        child_id: str,
        message_type: str,
        payload: Mapping[str, Any],
        *,
        timeout: float,
    ) -> ControlMessage:
        if message_type not in _REQUEST_TYPES:
            raise ControlProtocolError("unsupported_request_type")
        if timeout <= 0 or timeout > 300:
            raise ValueError("timeout must be between zero and 300 seconds")
        state = self._children.get(child_id)
        if state is None or state.writer is None or not state.connected.is_set():
            raise ControlProtocolError("child_not_connected")
        if len(state.pending) >= self._max_pending_requests:
            raise ControlProtocolError("backpressure")
        request_id = f"request-{uuid4().hex}"
        deadline_ms = int((time.time() + timeout) * 1000)
        future: asyncio.Future[ControlMessage] = (
            asyncio.get_running_loop().create_future()
        )
        state.pending[request_id] = _PendingReply(_REQUEST_REPLY[message_type], future)
        try:
            await _write_message(
                state.writer,
                state.write_lock,
                ControlMessage(
                    CONTROL_PROTOCOL_VERSION,
                    message_type,
                    request_id,
                    deadline_ms,
                    payload,
                ),
            )
        except BaseException:
            state.pending.pop(request_id, None)
            if not future.done():
                future.cancel()
            raise
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout)
        except TimeoutError:
            await self._cancel_request(state, request_id, future)
            raise ControlProtocolError("deadline_exceeded") from None
        except asyncio.CancelledError:
            await self._cancel_request(state, request_id, future)
            raise

    async def _cancel_request(
        self,
        state: _ChildState,
        request_id: str,
        future: asyncio.Future[ControlMessage],
    ) -> None:
        state.pending.pop(request_id, None)
        state.cancelled_ids[request_id] = None
        state.cancelled_ids.move_to_end(request_id)
        while len(state.cancelled_ids) > self._max_pending_requests * 4:
            state.cancelled_ids.popitem(last=False)
        if not future.done():
            future.cancel()
        if state.writer is not None:
            try:
                await _write_message(
                    state.writer,
                    state.write_lock,
                    ControlMessage(
                        CONTROL_PROTOCOL_VERSION,
                        "control.cancel",
                        f"cancel-{uuid4().hex}",
                        None,
                        {"request_id": request_id},
                    ),
                )
            except ControlProtocolError:
                pass

    async def next_event(self, child_id: str, *, timeout: float) -> ControlMessage:
        state = self._children.get(child_id)
        if state is None:
            raise ControlProtocolError("unknown_child")
        return await asyncio.wait_for(state.events.get(), timeout)

    async def _accept_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._connection_tasks.add(task)
        try:
            peer = writer.get_extra_info("peername")
            if not peer or not ipaddress.ip_address(peer[0]).is_loopback:
                raise ControlProtocolError("non_loopback_peer")
            auth = await asyncio.wait_for(_read_message(reader), timeout=2)
            if auth.message_type != "auth.request":
                raise ControlProtocolError("authentication_required")
            child_id = auth.payload["child_id"]
            supplied = auth.payload["secret"]
            state = self._children.get(child_id) if isinstance(child_id, str) else None
            supplied_bytes = _secret_bytes(supplied)
            expected_bytes = _secret_bytes(state.secret) if state is not None else None
            if (
                state is None
                or supplied_bytes is None
                or expected_bytes is None
                or not hmac.compare_digest(expected_bytes, supplied_bytes)
                or state.writer is not None
            ):
                raise ControlProtocolError("authentication_failed")
            # A launch capability authenticates exactly one connection. A
            # reconnect requires AppService to mint a fresh incarnation.
            state.secret = ""
            state.writer = writer
            state.connected.set()
            await _write_message(
                writer,
                state.write_lock,
                ControlMessage(
                    CONTROL_PROTOCOL_VERSION,
                    "auth.response",
                    auth.request_id,
                    None,
                    {"status": "accepted"},
                ),
            )
            while True:
                message = await _read_message(reader)
                if message.message_type == "canvas.events":
                    try:
                        state.events.put_nowait(message)
                    except asyncio.QueueFull:
                        raise ControlProtocolError("event_backpressure") from None
                    continue
                if (
                    message.message_type in _RESPONSE_TYPES
                    or message.message_type == "control.error"
                ):
                    pending = state.pending.pop(message.request_id, None)
                    if pending is None:
                        if message.request_id in state.cancelled_ids:
                            state.cancelled_ids.pop(message.request_id, None)
                            continue
                        raise ControlProtocolError("out_of_order_reply")
                    if message.message_type == "control.error":
                        pending.future.set_exception(
                            ControlProtocolError(str(message.payload["code"]))
                        )
                    elif message.message_type != pending.expected_type:
                        pending.future.set_exception(
                            ControlProtocolError("out_of_order_reply")
                        )
                    else:
                        pending.future.set_result(message)
                    continue
                raise ControlProtocolError("unexpected_child_message")
        except (ControlProtocolError, TimeoutError) as error:
            code = (
                error.code
                if isinstance(error, ControlProtocolError)
                else "authentication_failed"
            )
            try:
                await _write_message(
                    writer,
                    asyncio.Lock(),
                    ControlMessage(
                        CONTROL_PROTOCOL_VERSION,
                        "control.error",
                        "request-auth",
                        None,
                        {"code": code},
                    ),
                )
            except ControlProtocolError:
                pass
        finally:
            for state in self._children.values():
                if state.writer is writer:
                    state.writer = None
                    state.connected.clear()
                    for pending in state.pending.values():
                        if not pending.future.done():
                            pending.future.set_exception(
                                ControlProtocolError("connection_closed")
                            )
                    state.pending.clear()
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            if task is not None:
                self._connection_tasks.discard(task)

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        for child_id in tuple(self._children):
            await self.revoke_child(child_id)
        tasks = tuple(self._connection_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


ControlRequestHandler = Callable[[ControlMessage], Awaitable[ControlMessage]]


class CanvasControlClient:
    """Child-side authenticated client that dispatches authority requests."""

    def __init__(
        self,
        environment: Mapping[str, str],
        *,
        handler: ControlRequestHandler | None = None,
        max_active_requests: int = DEFAULT_MAX_PENDING_REQUESTS,
    ) -> None:
        try:
            host = environment[_ENV_HOST]
            port = int(environment[_ENV_PORT])
            child_id = environment[_ENV_CHILD_ID]
            secret = environment[_ENV_SECRET]
            version = int(environment[_ENV_VERSION])
        except (KeyError, TypeError, ValueError):
            raise ControlProtocolError("invalid_spawn_environment") from None
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ControlProtocolError("non_loopback_endpoint")
        except ValueError:
            raise ControlProtocolError("invalid_spawn_environment") from None
        if version != CONTROL_PROTOCOL_VERSION:
            raise ControlProtocolError("unsupported_version")
        _identifier(child_id, "child_id")
        if _secret_bytes(secret) is None:
            raise ControlProtocolError("invalid_spawn_environment")
        if not 1 <= port <= 65535:
            raise ControlProtocolError("invalid_spawn_environment")
        if not 1 <= max_active_requests <= 256:
            raise ValueError("invalid active-request limit")
        self._host = host
        self._port = port
        self._child_id = child_id
        self._secret = secret
        self._handler = handler or self._default_handler
        self._max_active_requests = max_active_requests
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._write_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._requests: dict[str, asyncio.Task[None]] = {}
        self._disconnected = asyncio.Event()
        self._disconnected.set()

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
        *,
        handler: ControlRequestHandler | None = None,
    ) -> CanvasControlClient | None:
        """Return no client outside textual-serve child processes."""

        present = [
            key in environment
            for key in (_ENV_HOST, _ENV_PORT, _ENV_CHILD_ID, _ENV_SECRET, _ENV_VERSION)
        ]
        if not any(present):
            return None
        if not all(present):
            raise ControlProtocolError("invalid_spawn_environment")
        return cls(environment, handler=handler)

    @property
    def child_id(self) -> str:
        """Return the non-secret AppService identity bound at spawn."""

        return self._child_id

    async def start(self) -> None:
        if self._reader_task is not None:
            return
        self._reader, self._writer = await asyncio.open_connection(
            self._host, self._port, limit=MAX_CONTROL_FRAME_BYTES + 4
        )
        auth_id = f"auth-{uuid4().hex}"
        await _write_message(
            self._writer,
            self._write_lock,
            ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                "auth.request",
                auth_id,
                None,
                {"child_id": self._child_id, "secret": self._secret},
            ),
        )
        try:
            response = await asyncio.wait_for(_read_message(self._reader), timeout=2)
        except ControlProtocolError as error:
            await self._close_writer()
            if error.code == "connection_closed":
                raise ControlProtocolError("authentication_failed") from None
            raise
        if response.message_type == "control.error":
            await self._close_writer()
            raise ControlProtocolError(str(response.payload["code"]))
        if (
            response.message_type != "auth.response"
            or response.request_id != auth_id
            or response.payload.get("status") != "accepted"
        ):
            await self._close_writer()
            raise ControlProtocolError("authentication_failed")
        self._secret = ""
        self._disconnected.clear()
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"canvas-control-{self._child_id}"
        )

    async def send_event(self, payload: Mapping[str, Any]) -> None:
        writer = self._writer
        if (
            writer is None
            or self._reader_task is None
            or self._reader_task.done()
            or self._disconnected.is_set()
        ):
            raise ControlProtocolError("client_not_connected")
        event_id = payload.get("event_id")
        request_id = (
            f"event-{event_id}" if isinstance(event_id, str) else f"event-{uuid4().hex}"
        )
        await _write_message(
            writer,
            self._write_lock,
            ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                "canvas.events",
                request_id,
                None,
                payload,
            ),
        )

    async def wait_disconnected(self, *, timeout: float) -> None:
        """Wait until transport loss is observed by the authoritative child."""

        if timeout <= 0 or timeout > 300:
            raise ValueError("timeout must be between zero and 300 seconds")
        try:
            await asyncio.wait_for(self._disconnected.wait(), timeout)
        except TimeoutError:
            raise ControlProtocolError("connection_deadline_exceeded") from None

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                message = await _read_message(self._reader)
                if message.message_type == "control.cancel":
                    request_id = message.payload["request_id"]
                    task = self._requests.get(request_id)
                    if task is not None:
                        task.cancel()
                    continue
                if message.message_type not in _REQUEST_TYPES:
                    raise ControlProtocolError("unexpected_parent_message")
                if len(self._requests) >= self._max_active_requests:
                    await self._send_error(message.request_id, "backpressure")
                    continue
                if message.deadline_ms is None or message.deadline_ms <= int(
                    time.time() * 1000
                ):
                    await self._send_error(message.request_id, "deadline_exceeded")
                    continue
                task = asyncio.create_task(self._dispatch(message))
                self._requests[message.request_id] = task
                task.add_done_callback(
                    lambda _task, request_id=message.request_id: self._requests.pop(
                        request_id, None
                    )
                )
        except (ControlProtocolError, asyncio.CancelledError):
            pass
        finally:
            tasks = tuple(self._requests.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._requests.clear()
            writer = self._writer
            self._writer = None
            self._reader = None
            self._disconnected.set()
            if writer is not None:
                writer.close()

    async def _dispatch(self, message: ControlMessage) -> None:
        try:
            response = await self._handler(message)
            if not isinstance(response, ControlMessage):
                raise ControlProtocolError("invalid_handler_response")
            if response.request_id != message.request_id:
                raise ControlProtocolError("out_of_order_reply")
            assert self._writer is not None
            await _write_message(self._writer, self._write_lock, response)
        except asyncio.CancelledError:
            raise
        except ControlProtocolError as error:
            await self._send_error(message.request_id, error.code)
        except Exception:  # noqa: BLE001 - authority errors cross as bounded codes
            await self._send_error(message.request_id, "operation_failed")

    async def _default_handler(self, message: ControlMessage) -> ControlMessage:
        if message.message_type == "health.request":
            return ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                "health.response",
                message.request_id,
                None,
                {"status": "ok"},
            )
        if message.message_type == "shutdown.request":
            return ControlMessage(
                CONTROL_PROTOCOL_VERSION,
                "shutdown.response",
                message.request_id,
                None,
                {"status": "accepted"},
            )
        raise ControlProtocolError("authority_unavailable")

    async def _send_error(self, request_id: str, code: str) -> None:
        if self._writer is None:
            return
        try:
            await _write_message(
                self._writer,
                self._write_lock,
                ControlMessage(
                    CONTROL_PROTOCOL_VERSION,
                    "control.error",
                    request_id,
                    None,
                    {"code": _identifier(code, "error_code")},
                ),
            )
        except ControlProtocolError:
            pass

    async def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except ConnectionError:
                pass
            self._writer = None

    async def aclose(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
            self._reader_task = None
        tasks = tuple(self._requests.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._requests.clear()
        await self._close_writer()
        self._disconnected.set()


__all__ = [
    "CONTROL_PROTOCOL_VERSION",
    "MAX_CONTROL_FRAME_BYTES",
    "CanvasControlBroker",
    "CanvasControlClient",
    "ChildControlLaunch",
    "ControlMessage",
    "ControlProtocolError",
    "decode_control_frame",
    "encode_control_frame",
]
