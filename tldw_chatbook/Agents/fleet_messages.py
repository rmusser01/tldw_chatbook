"""Bounded, session-only progress inboxes and scoped capabilities (ADR-136).

The store lock owns all queue state and capability membership. Code holding it
never calls a coordinator, observer, or external service. Where both locks are
needed, the coordinator must acquire its lock first.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace

MAX_MESSAGE_CHARS = 2_000
MAX_ENVELOPE_CHARS = 4_000
MAX_CHILD_PENDING = 8
MAX_CHILD_PENDING_CHARS = 16_000
MAX_CHILD_LIFETIME = 32
MAX_CHILD_LIFETIME_CHARS = 64_000
MAX_INBOX_PENDING = 32
MAX_INBOX_PENDING_CHARS = 64_000
MAX_RUNTIME_PENDING = 256
MAX_RUNTIME_PENDING_CHARS = 512_000
MAX_COLLECTED = 4
MAX_RESULT_CHARS = 8_000
MAX_IDENTITY_CHARS = 128
MAX_AGENT_CHARS = 80


class MessageError(ValueError):
    """A fixed, body-free refusal code suitable for metadata projection."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class MessageIdentity:
    """Service-bound source identity, captured before a child can report."""

    handle_id: str
    run_id: str
    parent_run_id: str
    chain_id: str | None
    agent: str


@dataclass(frozen=True)
class ProgressMessage:
    """An admitted report; snapshot ordering is arrival ordering."""

    message_id: str
    identity: MessageIdentity
    body: str


@dataclass(frozen=True)
class CollectedMessages:
    """Complete serialized result and trusted collection counts."""

    content: str
    collected_count: int
    remaining_count: int


def _validate_text(
    value: object, limit: int, *, size_code: str = "invalid_message"
) -> str:
    if type(value) is not str or not value.strip():
        raise MessageError("invalid_message")
    if len(value) > limit:
        raise MessageError(size_code)
    if any(
        (ord(char) < 32 and char not in "\t\n\r") or ord(char) == 127 for char in value
    ):
        raise MessageError("invalid_message")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise MessageError("invalid_message") from None
    return value


def _json(value: object) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
    except (TypeError, ValueError):
        raise MessageError("invalid_message") from None


def _envelope(message: ProgressMessage) -> dict[str, str | None]:
    source = message.identity
    return {
        "message_id": message.message_id,
        "handle_id": source.handle_id,
        "run_id": source.run_id,
        "parent_run_id": source.parent_run_id,
        "chain_id": source.chain_id,
        "agent": source.agent,
        "body": message.body,
    }


@dataclass
class _SenderState:
    capability: MessageSender
    accepted_count: int = 0
    accepted_chars: int = 0


class MessageStore:
    """Own conversation inboxes and the runtime-wide pending allowance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inboxes: dict[str, MessageInbox] = {}
        self._closed = False
        self._pending_count = 0
        self._pending_chars = 0

    def open_inbox(self, conversation_id: str) -> MessageInbox:
        """Return the current inbox, creating it unless the store is closed."""
        _validate_text(conversation_id, MAX_IDENTITY_CHARS)
        with self._lock:
            if self._closed:
                raise MessageError("unavailable")
            inbox = self._inboxes.get(conversation_id)
            if inbox is None:
                inbox = MessageInbox(self, conversation_id)
                self._inboxes[conversation_id] = inbox
            return inbox

    def get_inbox(self, conversation_id: str) -> MessageInbox | None:
        """Look up a live inbox without allocating or reviving an owner."""
        with self._lock:
            return self._inboxes.get(conversation_id)

    def close_inbox(self, conversation_id: str) -> None:
        """Invalidate capabilities and release this inbox's pending allowance."""
        with self._lock:
            inbox = self._inboxes.pop(conversation_id, None)
            if inbox is not None:
                self._close_locked(inbox)

    def _close_locked(self, inbox: MessageInbox) -> None:
        self._pending_count -= len(inbox._messages)
        self._pending_chars -= sum(len(message.body) for message in inbox._messages)
        inbox._messages.clear()
        inbox._senders.clear()
        inbox._reader = None

    def close(self) -> None:
        """Permanently invalidate every inbox before external worker cleanup."""
        with self._lock:
            self._closed = True
            for inbox in self._inboxes.values():
                self._close_locked(inbox)
            self._inboxes.clear()

    def pending_counts(self) -> dict[str, int]:
        """Return detached, body-free counts for nonempty conversation inboxes."""
        with self._lock:
            return {
                key: len(inbox._messages)
                for key, inbox in self._inboxes.items()
                if inbox._messages
            }


class MessageInbox:
    """Conversation owner API; children receive only its bound sender."""

    def __init__(self, store: MessageStore, conversation_id: str) -> None:
        self._store = store
        self._conversation_id = conversation_id
        self._messages: list[ProgressMessage] = []
        self._senders: dict[str, _SenderState] = {}
        self._reader: MessageReader | None = None

    def _check_locked(self) -> None:
        if (
            self._store._closed
            or self._store._inboxes.get(self._conversation_id) is not self
        ):
            raise MessageError("unavailable")

    def sender(self, identity: MessageIdentity) -> MessageSender:
        """Bind a unique live child identity, refusing duplicate capabilities."""
        if type(identity) is not MessageIdentity:
            raise MessageError("invalid_message")
        for value in (identity.handle_id, identity.run_id, identity.parent_run_id):
            _validate_text(value, MAX_IDENTITY_CHARS)
        if identity.chain_id is not None:
            _validate_text(identity.chain_id, MAX_IDENTITY_CHARS)
        _validate_text(identity.agent, MAX_AGENT_CHARS)
        identity = replace(identity)
        with self._store._lock:
            self._check_locked()
            if identity.handle_id in self._senders or any(
                state.capability._identity.run_id == identity.run_id
                for state in self._senders.values()
            ):
                raise MessageError("unavailable")
            sender = MessageSender(self, identity)
            self._senders[identity.handle_id] = _SenderState(sender)
            return sender

    def reader(
        self, run_id: str, *, chain_id: str | None, automatic: bool
    ) -> MessageReader:
        """Bind one active primary; automatic readers require a known chain."""
        _validate_text(run_id, MAX_IDENTITY_CHARS)
        if chain_id is not None:
            _validate_text(chain_id, MAX_IDENTITY_CHARS)
        if automatic and chain_id is None:
            raise MessageError("unavailable")
        with self._store._lock:
            self._check_locked()
            if self._reader is not None:
                raise MessageError("reader_busy")
            reader = MessageReader(self, run_id, chain_id, automatic)
            self._reader = reader
            return reader

    def revoke_sender(self, handle_id: str) -> None:
        """Retire the child's live capability and lifetime counter."""
        with self._store._lock:
            self._check_locked()
            self._senders.pop(handle_id, None)

    def snapshot(self) -> tuple[ProgressMessage, ...]:
        """Inspect detached immutable envelopes without consuming them."""
        with self._store._lock:
            self._check_locked()
            return tuple(
                replace(message, identity=replace(message.identity))
                for message in self._messages
            )

    def discard(self, message_ids: Sequence[str]) -> int:
        """Remove selected snapshot IDs only, without refunding lifetime sends."""
        selected = set(message_ids)
        with self._store._lock:
            self._check_locked()
            return self._remove_locked(selected)

    def _remove_locked(self, selected: set[str]) -> int:
        removed = [
            message for message in self._messages if message.message_id in selected
        ]
        self._messages = [
            message for message in self._messages if message.message_id not in selected
        ]
        self._store._pending_count -= len(removed)
        self._store._pending_chars -= sum(len(message.body) for message in removed)
        return len(removed)


@dataclass(frozen=True, eq=False)
class MessageSender:
    """Report-only capability validated by exact owner membership."""

    _inbox: MessageInbox
    _identity: MessageIdentity

    def _is_bound_to(self, identity: MessageIdentity) -> bool:
        """Check an existing coordinator binding without renewing its authority."""
        inbox = self._inbox
        with inbox._store._lock:
            if (
                inbox._store._closed
                or inbox._store._inboxes.get(inbox._conversation_id) is not inbox
            ):
                return False
            state = inbox._senders.get(self._identity.handle_id)
            return (
                state is not None
                and state.capability is self
                and self._identity == identity
            )

    def send(self, message: object) -> str:
        """Atomically admit one complete report or return a fixed refusal."""
        body = _validate_text(message, MAX_MESSAGE_CHARS, size_code="message_too_large")
        report = ProgressMessage(uuid.uuid4().hex, self._identity, body)
        if len(_json(_envelope(report))) > MAX_ENVELOPE_CHARS:
            raise MessageError("message_too_large")
        inbox = self._inbox
        store = inbox._store
        with store._lock:
            inbox._check_locked()
            state = inbox._senders.get(self._identity.handle_id)
            if state is None or state.capability is not self:
                raise MessageError("unavailable")
            if (
                state.accepted_count >= MAX_CHILD_LIFETIME
                or state.accepted_chars + len(body) > MAX_CHILD_LIFETIME_CHARS
            ):
                raise MessageError("sender_limit")
            child = [
                m for m in inbox._messages if m.identity.run_id == self._identity.run_id
            ]
            if (
                len(child) >= MAX_CHILD_PENDING
                or sum(len(m.body) for m in child) + len(body) > MAX_CHILD_PENDING_CHARS
                or len(inbox._messages) >= MAX_INBOX_PENDING
                or sum(len(m.body) for m in inbox._messages) + len(body)
                > MAX_INBOX_PENDING_CHARS
                or store._pending_count >= MAX_RUNTIME_PENDING
                or store._pending_chars + len(body) > MAX_RUNTIME_PENDING_CHARS
            ):
                raise MessageError("queue_full")
            inbox._messages.append(report)
            state.accepted_count += 1
            state.accepted_chars += len(body)
            store._pending_count += 1
            store._pending_chars += len(body)
            return report.message_id

    def close(self) -> None:
        """Revoke this exact sender; never revoke a replacement capability."""
        inbox = self._inbox
        with inbox._store._lock:
            state = inbox._senders.get(self._identity.handle_id)
            if state is not None and state.capability is self:
                del inbox._senders[self._identity.handle_id]


@dataclass(frozen=True, eq=False)
class MessageReader:
    """Single-primary capability for eligible FIFO collection."""

    _inbox: MessageInbox
    _run_id: str
    _chain_id: str | None
    _automatic: bool

    def _eligible_locked(self) -> list[ProgressMessage]:
        self._inbox._check_locked()
        if self._inbox._reader is not self:
            raise MessageError("unavailable")
        return [
            m
            for m in self._inbox._messages
            if not self._automatic or m.identity.chain_id == self._chain_id
        ]

    def pending_count(self) -> int:
        """Return only reports eligible for this exact live primary."""
        with self._inbox._store._lock:
            return len(self._eligible_locked())

    def collect(self, max_chars: int = MAX_RESULT_CHARS) -> CollectedMessages:
        """Serialize whole eligible reports before consuming any queue entry."""
        if type(max_chars) is not int:
            raise MessageError("invalid_message")
        cap = min(max_chars, MAX_RESULT_CHARS) if max_chars > 0 else MAX_RESULT_CHARS
        with self._inbox._store._lock:
            eligible = self._eligible_locked()
            selected: list[ProgressMessage] = []
            content = _json(
                {"status": "collected", "messages": [], "remaining": len(eligible)}
            )
            if len(content) > cap:
                raise MessageError("result_limit_too_small")
            for message in eligible[:MAX_COLLECTED]:
                candidate = selected + [message]
                serialized = _json(
                    {
                        "status": "collected",
                        "messages": [_envelope(m) for m in candidate],
                        "remaining": len(eligible) - len(candidate),
                    }
                )
                if len(serialized) > cap:
                    if not selected:
                        raise MessageError("result_limit_too_small")
                    break
                selected = candidate
                content = serialized
            self._inbox._remove_locked({message.message_id for message in selected})
            return CollectedMessages(
                content, len(selected), len(eligible) - len(selected)
            )

    def close(self) -> None:
        """Release this reader slot; a stale close cannot affect its successor."""
        with self._inbox._store._lock:
            if self._inbox._reader is self:
                self._inbox._reader = None
