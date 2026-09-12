"""Closed private voice records and bounded ownership, using only the stdlib.

Sequences are logical *per stream*, never physical wire order. A sender reserves
before pulling a producer. Publishing transfers that reservation into IPC custody;
writing or receiving does not return it. Only a validated cumulative consumer
receipt does. Session owners keep one window per fixed lane and retire it only
after actual consumption or matching fenced discard, never merely on cancellation.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import re
import struct
import threading
from types import MappingProxyType

from tldw_chatbook.Utils.local_stt_providers import LOCAL_STT_PROVIDERS

from .voice_process_types import VoiceTransportFailure


HEADER_LIMIT = 4096
PAYLOAD_LIMIT = 262144
PCM_LIMIT = 65536
TEXT_LIMIT = 65536
_PREFIX = struct.Struct("!II")
_PARENT = "parent_to_child"
_CHILD = "child_to_parent"
_BOTH = (_PARENT, _CHILD)
_COMMON = frozenset({"version", "op", "generation", "request_id", "sequence"})
_TURN = frozenset({"turn_id", "revision", "epoch"})
_REASONS = frozenset(
    {
        "stop",
        "escape",
        "microphone_disabled",
        "hands_free_exit",
        "navigation",
        "teardown",
    }
)
_OUTCOMES = frozenset({"clean", "force_closed", "detached", "failed"})
_FAULTS = frozenset(
    {
        "transport_failed",
        "protocol_invalid",
        "capacity_exceeded",
        "lease_expired",
        "startup_failed",
        "stt_failed",
        "native_failed",
        "shutdown_unconfirmed",
        "provider_failed",
        "tts_failed",
        "stale",
    }
)
_DIAGNOSTICS = frozenset(
    {
        "capture_progress",
        "provider_delta",
        "eligible_phrase",
        "synthesis_complete",
        "render_receipt",
        "capture_fault",
        "stt_ready",
        "session_ready",
    }
)


class ProtocolError(VoiceTransportFailure):
    """Fixed category only; never interpolate a record or underlying exception."""

    def __init__(self, code: str = "voice_protocol_invalid") -> None:
        if code not in {
            "voice_protocol_invalid",
            "voice_capacity_exceeded",
            "voice_transport_closed",
            "voice_transport_eof",
            "voice_transport_truncated",
            "voice_transport_failed",
        }:
            code = "voice_protocol_invalid"
        super().__init__(code)


def checked_epoch(value: object) -> int:
    """Validate exact nonnegative int64, rejecting bools."""
    if type(value) is not int or not 0 <= value < 2**63:
        raise ProtocolError()
    return value


def checked_pcm(payload: bytes) -> bytes:
    """Accept whole 10 ms, 48 kHz mono PCM16 frames only."""
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > PCM_LIMIT
        or len(payload) % 960
    ):
        raise ProtocolError()
    return payload


@dataclass(frozen=True, slots=True)
class _Operation:
    directions: tuple[str, ...]
    scope: str
    fields: tuple[tuple[str, str], ...] = ()
    payload: str = "none"
    cap: int = 0
    lane: str = "control"


# One closed table, intentionally no registration or arbitrary metadata mapping.
_OPERATIONS = MappingProxyType(
    {
        "bootstrap": _Operation(
            (_PARENT,),
            "session",
            (
                ("root", "root"),
                ("source", "digest"),
                ("native_abi", "int"),
                ("stt_provider", "provider"),
                ("stt_model", "model"),
                ("stt_device", "stt_device"),
                ("stt_compute_type", "stt_compute_type"),
                ("stt_precision", "stt_precision"),
                ("language", "language"),
                ("response_eagerness_ms", "eagerness"),
                ("aec_enabled", "bool"),
                ("vad_aggressiveness", "vad"),
                ("vad_preroll_ms", "preroll"),
            ),
        ),
        "hello": _Operation(
            (_CHILD,),
            "session",
            (("root", "root"), ("source", "digest"), ("native_abi", "int")),
        ),
        "stt_ready": _Operation((_CHILD,), "session", (("native_streaming", "bool"),)),
        "start": _Operation((_PARENT,), "session", (("capture_live", "bool"),)),
        "session_ready": _Operation((_CHILD,), "session"),
        "lease": _Operation((_PARENT,), "session", lane="lease"),
        "prepare": _Operation(
            (_CHILD,), "turn", payload="transcript", cap=PAYLOAD_LIMIT, lane="prepare"
        ),
        "prepared": _Operation(
            (_PARENT,),
            "turn",
            (
                ("request_handle", "handle"),
                ("context_handle", "handle"),
                ("decision", "decision"),
            ),
        ),
        "start_attempt": _Operation(
            (_CHILD,),
            "turn",
            (("request_handle", "handle"), ("context_handle", "handle")),
        ),
        "provider_delta": _Operation(
            (_PARENT,), "turn", payload="text", cap=4096, lane="provider"
        ),
        "provider_end": _Operation(
            (_PARENT,), "turn", (("last_sequence", "int"),), lane="provider_end"
        ),
        "tool_pending": _Operation(
            (_PARENT,), "turn", (("last_sequence", "int"),), lane="provider_end"
        ),
        "provider_failure": _Operation(
            (_PARENT,),
            "turn",
            (("last_sequence", "int"), ("code", "fault")),
            lane="provider_end",
        ),
        "cancel": _Operation(_BOTH, "turn", (("reason", "reason"),), lane="fence"),
        "cleanup": _Operation(
            (_PARENT,),
            "turn",
            (("outcome", "outcome"), ("last_sequence", "int")),
            lane="cleanup",
        ),
        "synthesize": _Operation((_CHILD,), "phrase", payload="text", cap=3072),
        "pcm": _Operation(
            (_PARENT,), "phrase", payload="pcm", cap=PCM_LIMIT, lane="pcm"
        ),
        "pcm_end": _Operation(
            (_PARENT,), "phrase", (("last_sequence", "int"),), lane="pcm_end"
        ),
        "tts_closed": _Operation(
            (_PARENT,),
            "phrase",
            (("outcome", "outcome"), ("last_sequence", "int")),
            lane="tts_closed",
        ),
        "terminal_propose": _Operation(
            (_CHILD,),
            "turn",
            (
                ("request_handle", "handle"),
                ("context_handle", "handle"),
                ("last_sequence", "int"),
                ("kind", "terminal_kind"),
                ("boundary_id", "optional_handle"),
                ("answer_sha256", "digest"),
                ("answer_chars", "int"),
            ),
            payload="transcript",
            cap=PAYLOAD_LIMIT,
            lane="terminal",
        ),
        "terminal_claim": _Operation(
            (_PARENT,),
            "turn",
            (("context_handle", "handle"), ("last_sequence", "int")),
            lane="terminal",
        ),
        "terminal_result": _Operation(
            (_PARENT,),
            "turn",
            (("context_handle", "handle"), ("disposition", "disposition")),
            lane="terminal_result",
        ),
        "preview": _Operation(
            (_CHILD,), "turn", payload="transcript", cap=PAYLOAD_LIMIT, lane="preview"
        ),
        "draft": _Operation(
            (_CHILD,),
            "turn",
            (("draft_slot", "draft_slot"),),
            payload="transcript",
            cap=PAYLOAD_LIMIT,
            lane="draft",
        ),
        "draft_recovery": _Operation(
            (_CHILD,),
            "turn",
            (("action", "recovery_action"), ("recovery_id", "positive_int")),
        ),
        "diagnostic": _Operation(
            _BOTH,
            "session",
            (("code", "diagnostic"), ("value", "int")),
            lane="diagnostic",
        ),
        "close": _Operation(_BOTH, "session", (("reason", "reason"),), lane="close"),
        "closed": _Operation(
            (_CHILD,), "session", (("outcome", "outcome"),), lane="closed"
        ),
        "resources_closed": _Operation(
            (_CHILD,), "session", (("outcome", "outcome"),), lane="resources_closed"
        ),
        "fault": _Operation(_BOTH, "session", (("code", "fault"),), lane="fault"),
        "credit": _Operation(
            _BOTH,
            "session",
            (
                ("lane", "lane"),
                ("stream_id", "handle"),
                ("stream_turn", "optional_turn"),
                ("stream_revision", "int"),
                ("stream_epoch", "int"),
                ("stream_phrase", "int"),
                ("ack_sequence", "int"),
                ("ack_bytes", "int"),
            ),
            lane="credit",
        ),
    }
)

# Per direction, including consumer custody. Projection slots are coalesced only
# while queued; their producer must wait for the current in-flight receipt.
_BUDGETS = MappingProxyType(
    {
        "control": (64, 4096),
        "prepare": (2, PAYLOAD_LIMIT),
        "terminal": (1, PAYLOAD_LIMIT),
        "provider": (8, 4096),
        "pcm": (8, PCM_LIMIT),
        "tts_closed": (1, 4096),
        "preview": (1, PAYLOAD_LIMIT),
        "draft_0": (1, PAYLOAD_LIMIT),
        "draft_1": (1, PAYLOAD_LIMIT),
        "diagnostic": (64, 4096),
    }
)
_PRIORITY = (
    "fault",
    "close",
    "fence",
    "closed",
    "resources_closed",
    "cleanup",
    "tts_closed",
    "terminal_result",
    "lease",
    "credit",
)
_ENDS = ("provider_end", "pcm_end")
_DRAFT_LANES = frozenset({"draft_0", "draft_1"})


def _valid_string(value: object, pattern: str, limit: int) -> bool:
    return (
        type(value) is str
        and len(value) <= limit
        and re.fullmatch(pattern, value) is not None
    )


def _check_scalar(value: object, kind: str) -> None:
    if kind.startswith("stt_"):
        choices = {
            "stt_device": {"auto", "cpu", "cuda", "mps"},
            "stt_compute_type": {
                "default",
                "auto",
                "int8",
                "int8_float16",
                "int8_float32",
                "int8_bfloat16",
                "int16",
                "float16",
                "float32",
                "bfloat16",
            },
            "stt_precision": {
                "fp16",
                "fp32",
                "bf16",
                "float16",
                "float32",
                "bfloat16",
                "int8",
                "f32",
            },
        }
        if value is not None and (type(value) is not str or value not in choices[kind]):
            raise ProtocolError()
        return
    if kind in {"int", "positive_int"}:
        checked_epoch(value)
        if kind == "positive_int" and value == 0:
            raise ProtocolError()
        return
    choices = {
        "reason": _REASONS,
        "outcome": _OUTCOMES,
        "fault": _FAULTS,
        "diagnostic": _DIAGNOSTICS,
        "decision": {"provisional", "waiting for stable turn"},
        "disposition": {"promoted", "recovery", "failed"},
        "lane": _BUDGETS.keys(),
        "provider": LOCAL_STT_PROVIDERS,
        "terminal_kind": {"promote", "accepted"},
        "recovery_action": {"request", "revoke"},
    }
    if kind in choices:
        valid = type(value) is str and value in choices[kind]
    elif kind == "bool":
        valid = type(value) is bool
    elif kind == "draft_slot":
        valid = type(value) is int and value in (0, 1)
    elif kind in {"vad", "preroll", "eagerness"}:
        low, high = {"vad": (0, 3), "preroll": (0, 1000), "eagerness": (500, 3000)}[
            kind
        ]
        valid = type(value) is int and low <= value <= high
    elif kind == "handle":
        valid = _valid_string(value, r"[0-9a-f]{32}", 32)
    elif kind == "optional_handle":
        valid = value is None or _valid_string(value, r"[0-9a-f]{32}", 32)
    elif kind == "digest":
        valid = _valid_string(value, r"[0-9a-f]{64}", 64)
    elif kind == "turn":
        valid = _valid_string(value, r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", 128)
    elif kind == "optional_turn":
        valid = value == "" or _valid_string(
            value, r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", 128
        )
    elif kind == "language":
        valid = _valid_string(value, r"[A-Za-z][A-Za-z0-9_-]{0,31}", 32)
    elif kind == "model":
        valid = value is None or (
            _valid_string(value, r"[A-Za-z0-9][A-Za-z0-9_./-]{0,255}", 256)
            and ".." not in value
        )
    elif kind == "root":
        # Attested source identity only, never a dispatch/open-file target.
        valid = (
            type(value) is str
            and 0 < len(value) <= 1024
            and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)
        )
    else:
        valid = False
    if not valid:
        raise ProtocolError()


@dataclass(frozen=True, slots=True)
class Record:
    """Immutable metadata and bytes; repr intentionally excludes private content."""

    header: Mapping[str, object] = field(repr=False)
    payload: bytes = field(default=b"", repr=False)
    _reservation: Reservation | None = field(
        default=None, repr=False, init=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.header, Mapping) or type(self.payload) is not bytes:
            raise ProtocolError()
        object.__setattr__(self, "header", MappingProxyType(dict(self.header)))


def _validate_header(header: object, payload_length: int, direction: str) -> _Operation:
    if type(header) is not dict or direction not in _BOTH:
        raise ProtocolError()
    op = header.get("op")
    if type(op) is not str or op not in _OPERATIONS:
        raise ProtocolError()
    rule = _OPERATIONS[op]
    expected = _COMMON | {name for name, _ in rule.fields}
    optional = (
        {"stt_device", "stt_compute_type", "stt_precision"}
        if op == "bootstrap"
        else set()
    )
    if rule.scope in {"turn", "phrase"}:
        expected |= _TURN
    if rule.scope == "phrase":
        expected |= {"phrase_id"}
    if (
        set(header) - optional != expected - optional
        or not set(header) <= expected
        or direction not in rule.directions
        or payload_length > rule.cap
    ):
        raise ProtocolError()
    if rule.payload == "pcm" and (payload_length == 0 or payload_length % 960):
        raise ProtocolError()
    if type(header["version"]) is not int or header["version"] != 1:
        raise ProtocolError()
    for name in ("generation", "sequence"):
        checked_epoch(header[name])
    _check_scalar(header["request_id"], "handle")
    if rule.scope in {"turn", "phrase"}:
        _check_scalar(header["turn_id"], "turn")
        checked_epoch(header["revision"])
        checked_epoch(header["epoch"])
    if rule.scope == "phrase":
        checked_epoch(header["phrase_id"])
    for name, kind in rule.fields:
        if name in optional and name not in header:
            continue
        _check_scalar(header[name], kind)
    return rule


def _check_payload(rule: _Operation, payload: bytes) -> None:
    if type(payload) is not bytes or len(payload) > rule.cap:
        raise ProtocolError()
    if rule.payload == "pcm":
        checked_pcm(payload)
    elif rule.payload in {"text", "transcript"}:
        try:
            value = payload.decode("utf-8", errors="strict")
        except UnicodeError:
            raise ProtocolError() from None
        if (rule.payload == "text" and not value) or len(value) > TEXT_LIMIT:
            raise ProtocolError()


def _header_bytes(record: Record, direction: str) -> bytes:
    rule = _validate_header(dict(record.header), len(record.payload), direction)
    _check_payload(rule, record.payload)
    try:
        raw = json.dumps(
            dict(record.header),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ProtocolError() from None
    if len(raw) > HEADER_LIMIT or (
        rule.lane in {"control", "diagnostic", "tts_closed"}
        and 8 + len(raw) + len(record.payload) > 4096
    ):
        raise ProtocolError()
    return raw


def encode_record(record: Record, direction: str) -> bytes:
    """Encode one bounded record. Writers use parts to avoid a second body copy."""
    raw = _header_bytes(record, direction)
    return _PREFIX.pack(len(raw), len(record.payload)) + raw + record.payload


def _charge(record: Record) -> int:
    op = record.header.get("op")
    if type(op) is not str or op not in _OPERATIONS:
        raise ProtocolError()
    rule = _OPERATIONS[op]
    header = _header_bytes(record, rule.directions[0])
    return (
        len(header) + 8 + len(record.payload)
        if rule.lane in {"control", "diagnostic", "tts_closed"}
        else len(record.payload)
    )


def _read_exact(
    read: Callable[[int], bytes], count: int, *, frame_boundary: bool = False
) -> bytes:
    body = bytearray()
    while len(body) < count:
        try:
            chunk = read(count - len(body))
        except InterruptedError:
            continue
        except Exception:
            raise ProtocolError("voice_transport_failed") from None
        if type(chunk) is not bytes or len(chunk) > count - len(body):
            raise ProtocolError()
        if not chunk:
            raise ProtocolError(
                "voice_transport_eof"
                if frame_boundary and not body
                else "voice_transport_truncated"
            )
        body.extend(chunk)
    return bytes(body)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ProtocolError()
        result[name] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ProtocolError()


def read_record(
    read: Callable[[int], bytes],
    direction: str,
    *,
    admit: Callable[[Mapping[str, object], int], None] | None = None,
) -> Record:
    """Read one bounded record, admitting before its body.

    Only an empty next-frame boundary is ordinary EOF. Any incomplete prefix,
    header or payload has the separate terminal truncated-frame category.
    """
    header_length, payload_length = _PREFIX.unpack(
        _read_exact(read, 8, frame_boundary=True)
    )
    if not 0 < header_length <= HEADER_LIMIT or payload_length > PAYLOAD_LIMIT:
        raise ProtocolError()
    raw = _read_exact(read, header_length)
    try:
        header = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeError, RecursionError):
        raise ProtocolError() from None
    rule = _validate_header(header, payload_length, direction)
    if (
        rule.lane in {"control", "diagnostic"}
        and 8 + header_length + payload_length > 4096
    ):
        raise ProtocolError()
    if admit is not None:
        admit(MappingProxyType(header), payload_length)
    payload = _read_exact(read, payload_length)
    _check_payload(rule, payload)
    return Record(header, payload)


@dataclass(frozen=True, slots=True)
class StreamKey:
    """One bounded logical stream; old keys can never replenish a replacement."""

    generation: int
    request_id: str
    lane: str
    turn_id: str = ""
    revision: int = 0
    epoch: int = 0
    phrase_id: int = 0

    def __post_init__(self) -> None:
        for value in (self.generation, self.revision, self.epoch, self.phrase_id):
            checked_epoch(value)
        _check_scalar(self.request_id, "handle")
        _check_scalar(self.lane, "lane")
        if self.turn_id:
            _check_scalar(self.turn_id, "turn")

    def matches(self, record: Record) -> bool:
        h = record.header
        if self.lane in _DRAFT_LANES:
            # One window lasts for this retained turn. Revision/attempt changes
            # update its content, not the stream identity used by wire credits.
            return self._record_lane(record) == self.lane and (
                h["generation"],
                h["request_id"],
                h.get("turn_id"),
            ) == (
                self.generation,
                self.request_id,
                self.turn_id,
            )
        if self.lane not in {"provider", "pcm", "draft_0", "draft_1"}:
            # These are session-wide streams. Request handles still identify the
            # individual operation; sequence/credit custody is session-wide.
            return h["generation"] == self.generation
        return (
            h["generation"],
            h["request_id"],
            h.get("turn_id", ""),
            h.get("revision", 0),
            h.get("epoch", 0),
            h.get("phrase_id", 0),
        ) == (
            self.generation,
            self.request_id,
            self.turn_id,
            self.revision,
            self.epoch,
            self.phrase_id,
        )

    def _record_lane(self, record: Record) -> str:
        """Resolve the draft wire operation to this explicitly retained slot."""
        lane = _OPERATIONS[record.header["op"]].lane
        if lane == "draft":
            _check_scalar(record.header.get("draft_slot"), "draft_slot")
            return f"draft_{record.header['draft_slot']}"
        return lane


@dataclass(slots=True)
class _CreditEntry:
    sequence: int
    size: int
    published: bool = False
    enqueued: bool = False
    write_state: str = "unstarted"
    can_complete: bool = False


class Reservation:
    """A single producer pull's capacity, held until consumption after publish."""

    def __init__(self, owner: CreditWindow, entry: _CreditEntry) -> None:
        self._owner = owner
        self._entry = entry
        self._enqueued = False

    @property
    def sequence(self) -> int:
        return self._entry.sequence

    def publish(self, record: Record) -> None:
        """Bind exact validated bytes/identity before enqueueing; keep the credit."""
        size = _charge(record)
        with self._owner._lock:
            self._owner._check_open()
            lane = self._owner.key._record_lane(record)
            if (
                self._entry.published
                or self._entry not in self._owner._entries
                or size > self._entry.size
                or record._reservation is not None
                or lane != self._owner.key.lane
                or not self._owner.key.matches(record)
                or record.header["sequence"] != self.sequence
            ):
                raise ProtocolError()
            self._entry.size = size
            self._entry.published = True
            object.__setattr__(record, "_reservation", self)

    def _begin_write(self) -> None:
        with self._owner._lock:
            # Closing producer admission does not retract a published frame.
            # It must still reach the receiver's matching fenced discard path.
            if not self._entry.enqueued or self._entry.write_state != "unstarted":
                raise ProtocolError()
            self._entry.write_state = "writing"

    def _before_write(self, *, can_complete: bool) -> None:
        with self._owner._lock:
            if self._entry.write_state != "writing":
                raise ProtocolError()
            # This is the last actual segment, not merely an active frame.
            # Closing admission cannot cancel an already-started frame's I/O.
            self._entry.can_complete = can_complete

    def _after_write(self, *, completed: bool) -> None:
        with self._owner._lock:
            self._entry.can_complete = completed
            pending = self._owner._pending_ack
            if not completed and pending is not None and pending[0] >= self.sequence:
                # A short write/EINTR proves the pending consumption receipt was
                # impossible. Never carry it forward into a later successful call.
                raise ProtocolError() from None

    def _finish_write(self) -> None:
        # Called only after PipeWriter has relinquished its Record reference and
        # local mailbox custody. A closed producer may still finish real custody.
        with self._owner._lock:
            if self._entry.write_state != "writing" or not self._entry.can_complete:
                raise ProtocolError()
            self._entry.write_state = "written"
            pending = self._owner._pending_ack
            if pending is not None:
                self._owner._settle_ack(*pending)

    def _fail_write(self) -> None:
        with self._owner._lock:
            self._entry.write_state = "failed"
            self._owner._pending_ack = None
            self._owner.close()

    def abandon(self) -> None:
        """Undo only an unpublished final reservation (EOF/cancel before enqueue)."""
        with self._owner._lock:
            if (
                self._entry.published
                or not self._owner._entries
                or self._owner._entries[-1] is not self._entry
            ):
                raise ProtocolError()
            self._owner._entries.pop()
            self._owner._next -= 1
            self._owner._wake()


class CreditWindow:
    """One fixed-lane end-to-end budget, with one bounded asynchronous waiter.

    The optional block count can narrow a lane, e.g. seven provider blocks when
    the real gateway owns the eighth. It cannot enlarge the protocol budget.
    """

    def __init__(self, key: StreamKey, *, blocks: int | None = None) -> None:
        maximum, self.block_bytes = _BUDGETS[key.lane]
        self.blocks = maximum if blocks is None else blocks
        if type(self.blocks) is not int or not 1 <= self.blocks <= maximum:
            raise ProtocolError()
        self.key = key
        self._entries: deque[_CreditEntry] = deque()
        self._next = 1
        self._ack_sequence = 0
        self._ack_bytes = 0
        self._pending_ack: tuple[int, int] | None = None
        self._closed = False
        self._lock = threading.RLock()
        self._waiter: asyncio.Future[None] | None = None
        self._wake_scheduled = False

    def _check_open(self) -> None:
        if self._closed:
            raise ProtocolError("voice_transport_closed")

    @property
    def outstanding(self) -> tuple[int, int]:
        with self._lock:
            return len(self._entries), sum(entry.size for entry in self._entries)

    async def reserve(self) -> Reservation:
        """Reserve a maximum block before invoking next/anext on the producer."""
        while True:
            with self._lock:
                self._check_open()
                permit = self.try_reserve()
                if permit is not None:
                    return permit
                if self._waiter is not None:
                    raise ProtocolError("voice_capacity_exceeded")
                waiter = asyncio.get_running_loop().create_future()
                self._waiter = waiter
            try:
                await waiter
            finally:
                with self._lock:
                    if self._waiter is waiter:
                        self._waiter = None

    def try_reserve(self) -> Reservation | None:
        """Nonblocking equivalent for producers already on their owner loop."""
        with self._lock:
            self._check_open()
            if len(self._entries) >= self.blocks:
                return None
            checked_epoch(self._next)
            entry = _CreditEntry(self._next, self.block_bytes)
            self._next += 1
            self._entries.append(entry)
            return Reservation(self, entry)

    def _wake(self) -> None:
        waiter = self._waiter
        if waiter is not None and not self._wake_scheduled:
            self._wake_scheduled = True

            def deliver() -> None:
                with self._lock:
                    self._wake_scheduled = False
                    current = self._waiter
                    if current is not None and not current.done():
                        current.set_result(None)

            try:
                waiter.get_loop().call_soon_threadsafe(deliver)
            except RuntimeError:
                self._wake_scheduled = False
                self._closed = True

    def acknowledge(self, key: StreamKey, sequence: int, consumed_bytes: int) -> None:
        """Validate a consumer receipt; defer a race with active writer custody.

        Unstarted frames cannot have been consumed. A reader can consume a final
        kernel write before the writer thread resumes, so one cumulative receipt
        may wait only for a call that could finish the entire frame. Prefix and
        intermediate-header calls cannot qualify. Credit never replenishes early.
        """
        checked_epoch(sequence)
        checked_epoch(consumed_bytes)
        with self._lock:
            # Fencing stops new admission, not truthful late cleanup accounting.
            newest_ack = (
                self._pending_ack[0] if self._pending_ack else self._ack_sequence
            )
            if key != self.key or sequence <= newest_ack:
                raise ProtocolError()
            entries = [entry for entry in self._entries if entry.sequence <= sequence]
            if (
                not entries
                or entries[-1].sequence != sequence
                or not all(
                    entry.write_state == "written"
                    or (entry.write_state == "writing" and entry.can_complete)
                    for entry in entries
                )
                or self._ack_bytes + sum(entry.size for entry in entries)
                != consumed_bytes
            ):
                raise ProtocolError()
            self._pending_ack = (sequence, consumed_bytes)
            self._settle_ack(sequence, consumed_bytes)

    def _settle_ack(self, sequence: int, consumed_bytes: int) -> None:
        entries = [entry for entry in self._entries if entry.sequence <= sequence]
        if not all(entry.write_state == "written" for entry in entries):
            return
        for _ in entries:
            self._entries.popleft()
        self._ack_sequence, self._ack_bytes = sequence, consumed_bytes
        self._pending_ack = None
        self._wake()

    def accept_credit(self, record: Record) -> None:
        """Validate the complete wire identity before accepting its cumulative ack."""
        _header_bytes(record, _PARENT)
        h = record.header
        if h["op"] != "credit":
            raise ProtocolError()
        key = StreamKey(
            h["generation"],
            h["stream_id"],
            h["lane"],
            h["stream_turn"],
            h["stream_revision"],
            h["stream_epoch"],
            h["stream_phrase"],
        )
        self.acknowledge(key, h["ack_sequence"], h["ack_bytes"])

    def close(self) -> None:
        """Fence admission and wake waiters with failure, retaining custody counts."""
        with self._lock:
            self._closed = True
            self._wake()


@dataclass(slots=True, eq=False)
class Delivery:
    """Consumer-owned data. Release only after real use or a matching fence."""

    record: Record | None = field(repr=False)
    _owner: ReceiveStream = field(repr=False)
    _sequence: int
    _size: int
    _mailbox: Mailbox | None = field(default=None, repr=False)
    _used: bool = False

    @property
    def discard_only(self) -> bool:
        """Check current fencing, including revocations received after delivery.

        Data consumers must check this before effect/render use. Pass the inbound
        mailbox to receive() so that a later fence remains visible here.
        """
        return self._mailbox is not None and self._mailbox.is_fenced(self._owner.key)

    def consume(self) -> tuple[int, int]:
        if self.discard_only:
            raise ProtocolError()
        return self._owner._consume(self)

    def discard(self, fenced_key: StreamKey) -> tuple[int, int]:
        if fenced_key != self._owner.key:
            raise ProtocolError()
        return self._owner._consume(self)


class ReceiveStream:
    """Bounded sequence/custody state with independent EOF and cleanup facts.

    Owned by one consumer loop. This has no parent authority: completion is a
    transport consumption barrier, never a promotion or actual-render receipt.
    """

    def __init__(self, key: StreamKey, *, blocks: int | None = None) -> None:
        maximum, self.block_bytes = _BUDGETS[key.lane]
        self.blocks = maximum if blocks is None else blocks
        if type(self.blocks) is not int or not 1 <= self.blocks <= maximum:
            raise ProtocolError()
        self.key = key
        self._pending: deque[Delivery] = deque()
        self._received = 0
        self._ack_sequence = 0
        self._ack_bytes = 0
        self._end: int | None = None
        self._cleanup: str | None = None

    @property
    def ack(self) -> tuple[int, int]:
        return self._ack_sequence, self._ack_bytes

    @property
    def complete(self) -> bool:
        return self.data_complete and (
            self.key.lane != "pcm" or self._cleanup == "clean"
        )

    @property
    def data_complete(self) -> bool:
        """The named final data sequence was consumed, independently of cleanup."""
        return self._end is not None and self._ack_sequence == self._end

    @property
    def cleanup_outcome(self) -> str | None:
        """Actual cleanup receipt, independent of producer exhaustion."""
        return self._cleanup

    def receive(self, record: Record, *, mailbox: Mailbox | None = None) -> Delivery:
        size = _charge(record)
        lane = self.key._record_lane(record)
        if (
            lane != self.key.lane
            or not self.key.matches(record)
            or (mailbox is not None and mailbox._lane(record.header) != lane)
            or (self._end is not None and record.header["sequence"] > self._end)
            or record.header["sequence"] != self._received + 1
        ):
            raise ProtocolError()
        if len(self._pending) >= self.blocks or size > self.block_bytes:
            raise ProtocolError("voice_capacity_exceeded")
        self._received += 1
        delivery = Delivery(record, self, self._received, size, mailbox)
        self._pending.append(delivery)
        return delivery

    def _consume(self, delivery: Delivery) -> tuple[int, int]:
        if delivery._used or delivery not in self._pending:
            raise ProtocolError()
        delivery._used = True
        if delivery._mailbox is not None:
            delivery._mailbox.release(delivery.record)
            delivery._mailbox = None
        delivery.record = None
        while self._pending and self._pending[0]._used:
            head = self._pending.popleft()
            self._ack_sequence = head._sequence
            self._ack_bytes += head._size
        return self.ack

    def end(self, last_sequence: int) -> bool:
        checked_epoch(last_sequence)
        if (
            self._end is not None
            or not self._received <= last_sequence <= self._received + self.blocks
        ):
            raise ProtocolError()
        self._end = last_sequence
        return self.complete

    def closed(self, outcome: str) -> None:
        _check_scalar(outcome, "outcome")
        if self._cleanup is not None:
            raise ProtocolError()
        self._cleanup = outcome


class Mailbox:
    """Fixed bounded scheduling lanes, shared by a pipe thread and its owner.

    This is the queue portion of custody. Dequeue does NOT acknowledge a sender
    reservation. Consumers retain their ReceiveStream Delivery until actual use.
    Reserved receipts are never overwritten: a second unconsumed authoritative
    receipt fails closed. Only fence/lease/credit/projection state can coalesce.
    """

    def __init__(
        self, direction: str, *, generation: int, outbound: bool = False
    ) -> None:
        if direction not in _BOTH:
            raise ProtocolError()
        self.direction = direction
        self.outbound = outbound
        self.generation = checked_epoch(generation)
        self._queues: dict[str, deque[Record]] = {
            lane: deque() for lane in (*_BUDGETS, *_PRIORITY, *_ENDS)
        }
        self._owned: dict[int, tuple[str, Record]] = {}
        self._windows: dict[str, CreditWindow] = {}
        self._data_sequences: dict[str, tuple[StreamKey, int]] = {}
        self._last_sequences: dict[str | tuple[str, str], int] = {}
        self._fenced_epoch = -1
        self._turns: list[str] = []
        # Metadata order is local; draft slots belong to the remote sender.
        # A retained old binding is a tombstone after its empty slot is reused.
        self._draft_slots: dict[str, int] = {}
        self._draft_owners: list[str | None] = [None, None]
        self._condition = threading.Condition()
        self._closed = False

    @property
    def count(self) -> int:
        with self._condition:
            return self.queued + len(self._owned)

    @property
    def queued(self) -> int:
        with self._condition:
            return sum(len(queue) for queue in self._queues.values())

    def retain_turn(self, turn_id: str) -> None:
        _check_scalar(turn_id, "turn")
        with self._condition:
            if turn_id not in self._turns:
                if "" in self._turns:
                    self._turns[self._turns.index("")] = turn_id
                    return
                if len(self._turns) >= 2:
                    raise ProtocolError("voice_capacity_exceeded")
                self._turns.append(turn_id)

    def retire_turn(self, turn_id: str) -> None:
        with self._condition:
            if turn_id not in self._turns:
                return  # Independent inbound/outbound retirement is idempotent.
            index = self._turns.index(turn_id)
            if any(
                record.header.get("turn_id") == turn_id
                for queue in self._queues.values()
                for record in queue
            ) or any(
                record.header.get("turn_id") == turn_id
                for _, record in self._owned.values()
            ):
                raise ProtocolError()
            # Keep two stable slots instead of shifting a live turn's identity.
            self._turns[index] = ""
            slot = self._draft_slots.pop(turn_id, None)
            if slot is not None:
                self._last_sequences.pop((f"draft_{slot}", turn_id), None)
                if self._draft_owners[slot] == turn_id:
                    self._draft_owners[slot] = None

    def _lane(self, header: Mapping[str, object]) -> str:
        # LifecyclePipe also resolves lanes before put's authoritative admission.
        with self._condition:
            lane = _OPERATIONS[header["op"]].lane
            if lane in {"draft", "preview"}:
                if header["turn_id"] not in self._turns:
                    raise ProtocolError("voice_capacity_exceeded")
            if lane == "draft":
                slot = header.get("draft_slot")
                _check_scalar(slot, "draft_slot")
                turn = header["turn_id"]
                lane = f"draft_{slot}"
                if turn in self._draft_slots:
                    if (
                        self._draft_slots[turn] != slot
                        or self._draft_owners[slot] != turn
                    ):
                        raise ProtocolError()
                elif self._queues[lane] or any(
                    owned_lane == lane for owned_lane, _ in self._owned.values()
                ):
                    raise ProtocolError("voice_capacity_exceeded")
            return lane

    @staticmethod
    def _sequence_key(lane, header):
        return (lane, header["turn_id"]) if lane in _DRAFT_LANES else lane

    def check_admission(
        self, header: Mapping[str, object], payload_length: int
    ) -> None:
        """Reject stale/capacity records before reading their potentially large body."""
        checked_epoch(payload_length)
        _validate_header(dict(header), payload_length, self.direction)
        with self._condition:
            if self._closed:
                raise ProtocolError("voice_transport_closed")
            if header["generation"] != self.generation:
                raise ProtocolError()
            lane = self._lane(header)
            if lane not in {"provider", "pcm", "credit", "diagnostic"} and header[
                "sequence"
            ] <= self._last_sequences.get(self._sequence_key(lane, header), -1):
                raise ProtocolError()
            queue = self._queues[lane]
            held = sum(owned_lane == lane for owned_lane, _ in self._owned.values())
            maximum = _BUDGETS.get(lane, (2 if lane == "cleanup" else 1, 0))[0]
            if lane == "credit":
                # One pending ack for each of the fixed lane names, even while
                # that lane's preceding ack is in a partial write/dispatch.
                return
            if lane in {"fence", "lease", "diagnostic"}:
                return
            if lane in {"preview", "draft_0", "draft_1"} and queue:
                return
            if len(queue) + held >= maximum:
                raise ProtocolError("voice_capacity_exceeded")
            if lane in {"provider", "pcm"}:
                previous = self._data_sequences.get(lane)
                if previous is None:
                    if header["sequence"] != 1 or header["epoch"] <= self._fenced_epoch:
                        raise ProtocolError()
                else:
                    key, sequence = previous
                    if (
                        not key.matches(Record(header))
                        or header["sequence"] != sequence + 1
                    ):
                        raise ProtocolError()

    def is_fenced(self, key: StreamKey) -> bool:
        """Whether this known data stream permits discard only, never effects."""
        with self._condition:
            previous = self._data_sequences.get(key.lane)
            return (
                previous is not None
                and previous[0] == key
                and key.epoch <= self._fenced_epoch
            )

    def fence_stream(self, key: StreamKey) -> None:
        """Locally revoke a known data epoch without inventing a wire sequence.

        The audio owner calls this before sending its cancel in the opposite
        direction. As with a received cancel, all known data at or below that
        session epoch becomes discard-only. Repeating the local fence is safe.
        """
        with self._condition:
            previous = self._data_sequences.get(key.lane)
            if previous is None or previous[0] != key:
                raise ProtocolError()
            self._fenced_epoch = max(self._fenced_epoch, key.epoch)

    def put(self, record: Record) -> None:
        """Nonblocking admission; authoritative overflow raises a terminal category."""
        _header_bytes(record, self.direction)
        with self._condition:
            self.check_admission(record.header, len(record.payload))
            lane = self._lane(record.header)
            queue = self._queues[lane]
            permit = record._reservation
            if self.outbound and lane in _BUDGETS:
                if (
                    permit is None
                    or permit._enqueued
                    or not permit._entry.published
                    or permit._owner.key.lane != lane
                ):
                    raise ProtocolError()
                permit._owner._check_open()
                previous = self._windows.get(lane)
                if previous is not None and previous is not permit._owner:
                    if previous.outstanding != (0, 0):
                        raise ProtocolError("voice_capacity_exceeded")
                    previous.close()
                self._windows[lane] = permit._owner
                permit._enqueued = True
                permit._entry.enqueued = True
            if (
                lane == "diagnostic"
                and len(queue)
                + sum(owned_lane == lane for owned_lane, _ in self._owned.values())
                >= 64
            ):
                return
            if lane == "credit":
                for old in tuple(queue):
                    if old.header["lane"] == record.header["lane"]:
                        if (
                            any(
                                old.header[field] != record.header[field]
                                for field in (
                                    "generation",
                                    "stream_id",
                                    "lane",
                                    "stream_turn",
                                    "stream_revision",
                                    "stream_epoch",
                                    "stream_phrase",
                                )
                            )
                            or record.header["ack_sequence"]
                            <= old.header["ack_sequence"]
                            or record.header["ack_bytes"] < old.header["ack_bytes"]
                        ):
                            raise ProtocolError()
                        queue.remove(old)
                        break
            if lane in {"fence", "lease", "preview", "draft_0", "draft_1"} and queue:
                old = queue[0]
                field = (
                    "epoch"
                    if lane == "fence"
                    else "revision"
                    if lane in {"preview", "draft_0", "draft_1"}
                    else "sequence"
                )
                if record.header[field] <= old.header[field]:
                    raise ProtocolError()
                queue.clear()
            queue.append(record)
            if lane in _DRAFT_LANES:
                turn = record.header["turn_id"]
                slot = record.header["draft_slot"]
                self._draft_slots[turn] = slot
                self._draft_owners[slot] = turn
            if lane not in {"provider", "pcm", "credit", "diagnostic"}:
                self._last_sequences[self._sequence_key(lane, record.header)] = (
                    record.header["sequence"]
                )
            if lane == "fence":
                self._fenced_epoch = max(self._fenced_epoch, record.header["epoch"])
            if lane in {"provider", "pcm"}:
                h = record.header
                self._data_sequences[lane] = (
                    StreamKey(
                        h["generation"],
                        h["request_id"],
                        lane,
                        h["turn_id"],
                        h["revision"],
                        h["epoch"],
                        h.get("phrase_id", 0),
                    ),
                    h["sequence"],
                )
            self._condition.notify()

    def replace_projection(self, record: Record) -> bool:
        """Replace only an unsent snapshot in place, using its original credit.

        False means the snapshot is already in flight. The producer must wait
        for its consumption credit before retaining/publishing another snapshot.
        """
        _header_bytes(record, self.direction)
        with self._condition:
            if record.header["generation"] != self.generation:
                raise ProtocolError()
            lane = self._lane(record.header)
            if lane not in {"preview", "draft_0", "draft_1"} or self._closed:
                raise ProtocolError()
            queue = self._queues[lane]
            if not queue:
                return False
            old = queue[0]
            if lane in _DRAFT_LANES and (
                record.header["request_id"],
                record.header["turn_id"],
            ) != (old.header["request_id"], old.header["turn_id"]):
                raise ProtocolError()
            if record.header["sequence"] != old.header["sequence"] or (
                record.header["epoch"],
                record.header["revision"],
            ) <= (old.header["epoch"], old.header["revision"]):
                raise ProtocolError()
            permit = old._reservation
            if self.outbound:
                if (
                    permit is None
                    or record._reservation is not None
                    or permit._owner.key.lane != lane
                    or not permit._owner.key.matches(record)
                ):
                    raise ProtocolError()
                with permit._owner._lock:
                    permit._owner._check_open()
                    permit._entry.size = _charge(record)
                    object.__setattr__(record, "_reservation", permit)
            queue[0] = record
            return True

    def open_stream(self, key: StreamKey) -> None:
        """Select a known data stream before its first data or overtaking fence.

        Buffered records of a fenced known stream remain admissible only for
        matching Delivery.discard. Unknown/retired identities never gain custody.
        Select a newer stream only after this lane's old custody retires.
        """
        with self._condition:
            if key.lane not in {"provider", "pcm"} or key.generation != self.generation:
                raise ProtocolError()
            if self._queues[key.lane] or any(
                lane == key.lane for lane, _ in self._owned.values()
            ):
                raise ProtocolError("voice_capacity_exceeded")
            window = self._windows.get(key.lane)
            if window is not None and window.outstanding != (0, 0):
                raise ProtocolError("voice_capacity_exceeded")
            previous = self._data_sequences.get(key.lane)
            if key.epoch <= self._fenced_epoch or (
                previous is not None
                and (key.epoch, key.phrase_id)
                <= (previous[0].epoch, previous[0].phrase_id)
            ):
                raise ProtocolError()
            self._data_sequences[key.lane] = (key, 0)

    def take(self, *, block: bool = False) -> Record | None:
        """Select revocations before new starts/claims; never preempt an in-flight frame."""
        with self._condition:
            if block:
                self._condition.wait_for(lambda: self._closed or self.queued > 0)
            for lane in (*_PRIORITY, *_BUDGETS, *_ENDS):
                queue = self._queues[lane]
                if queue:
                    if lane in {"fence", "lease", "credit"}:
                        # A partial frame cannot be preempted. Keep just one
                        # pending coalesced successor per fixed reserved slot.
                        if any(
                            owned_lane == lane
                            and (
                                lane != "credit"
                                or held.header["lane"] == queue[0].header["lane"]
                            )
                            for owned_lane, held in self._owned.values()
                        ):
                            continue
                    record = queue.popleft()
                    self._owned[id(record)] = (lane, record)
                    return record
            return None

    def release(self, record: Record) -> None:
        """Release local custody after write/use; this never returns sender credit."""
        with self._condition:
            owned = self._owned.get(id(record))
            if owned is None or owned[1] is not record:
                raise ProtocolError()
            del self._owned[id(record)]

    def release_reserved(self, record: Record) -> None:
        """Finish synchronous reserved-control dispatch if the handler has not."""
        with self._condition:
            owned = self._owned.get(id(record))
            if owned is not None and owned[0] not in _BUDGETS:
                del self._owned[id(record)]

    def close(self) -> None:
        with self._condition:
            self._closed = True
            for window in self._windows.values():
                window.close()
            self._condition.notify_all()
