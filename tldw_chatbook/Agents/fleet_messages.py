"""Bounded progress inboxes and process-local capabilities (ADRs 136/199).

The store lock owns all queue state and capability membership. Code holding it
never calls a coordinator or observer. Owned synchronous SQLite leaves may
commit under this lock. Where both locks are
needed, the coordinator must acquire its lock first.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.DB.fleet_progress_repository import (
        FleetProgressPromotionContribution,
        FleetProgressRepository,
    )

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
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


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
        # ponytail: one writer lock serializes bounded queues and durable SQL;
        # use per-inbox locks only if measured SQL throughput warrants them.
        self._lock = threading.Lock()
        self._revoked = threading.Event()
        self._inboxes: dict[str, MessageInbox] = {}
        self._deferred_counts: dict[str, int] = {}
        self._closed = False
        self.on_enqueue: Callable[[str, str, MessageIdentity], None] | None = None
        self._pending_count = 0
        self._pending_chars = 0
        self._published: tuple[
            Mapping[str, tuple[MessageInbox, tuple[ProgressMessage, ...]]],
            Mapping[str, int],
        ] = (MappingProxyType({}), MappingProxyType({}))

    def open_inbox(
        self,
        conversation_id: str,
        *,
        repository: FleetProgressRepository | None = None,
        saved_conversation_id: str | None = None,
        messages: tuple[ProgressMessage, ...] = (),
        blocking: bool = True,
    ) -> MessageInbox | None:
        """Bind authoritative state; nonblocking preparation retries outside owners."""
        _validate_text(conversation_id, MAX_IDENTITY_CHARS)
        if not self._lock.acquire(blocking=blocking):
            return None
        try:
            if self._closed or self._revoked.is_set():
                raise MessageError("unavailable")
            inbox = self._inboxes.get(conversation_id)
            if inbox is not None and inbox._revoked.is_set():
                raise MessageError("unavailable")
            if inbox is None:
                chars = sum(len(message.body) for message in messages)
                if (
                    len(messages) > MAX_INBOX_PENDING
                    or chars > MAX_INBOX_PENDING_CHARS
                    or self._pending_count + len(messages) > MAX_RUNTIME_PENDING
                    or self._pending_chars + chars > MAX_RUNTIME_PENDING_CHARS
                ):
                    raise MessageError("queue_full")
                inbox = MessageInbox(self, conversation_id)
                inbox._messages = list(messages)
                self._inboxes[conversation_id] = inbox
                self._pending_count += len(messages)
                self._pending_chars += chars
            if repository is not None and inbox._repository is None:
                # Prepared loading never replaces an already admitted live queue.
                if inbox._messages and tuple(inbox._messages) != messages:
                    raise MessageError("unavailable")
                inbox._repository = repository
                inbox._saved_conversation_id = saved_conversation_id
            self._deferred_counts.pop(conversation_id, None)
            self._publish_locked()
            return inbox
        finally:
            self._lock.release()

    def defer_inbox(
        self, conversation_id: str, count: int, *, blocking: bool = True
    ) -> bool:
        """Retain only a bounded saved count when live body capacity is full."""
        _validate_text(conversation_id, MAX_IDENTITY_CHARS)
        if not 0 <= count <= MAX_INBOX_PENDING:
            raise MessageError("queue_full")
        if not self._lock.acquire(blocking=blocking):
            return False
        try:
            if self._closed or self._revoked.is_set():
                raise MessageError("unavailable")
            if conversation_id not in self._inboxes:
                self._deferred_counts[conversation_id] = count
                self._publish_locked()
            return True
        finally:
            self._lock.release()

    def wait_for_writer(self) -> None:
        """Wait outside lifecycle locks; the next binding must revalidate ownership."""
        with self._lock:
            pass

    def get_inbox(self, conversation_id: str) -> MessageInbox | None:
        """Read committed membership; mutation still rechecks the authoritative lock."""
        if self._revoked.is_set():
            return None
        view = self._published[0].get(conversation_id)
        return view[0] if view is not None and not view[0]._revoked.is_set() else None

    def begin_close_inbox(self, conversation_id: str) -> MessageInbox | None:
        """Revoke the exact authoritative inbox without waiting on a SQL writer.

        Native identity serializes release and binding. This deny-only capture
        never grants authority or consults the published UI observation. Its
        object identity also fences physical cleanup against a replacement.
        """
        inbox = self._inboxes.get(conversation_id)
        if inbox is not None:
            inbox._revoked.set()
        return inbox

    def finish_close_inbox(
        self, conversation_id: str, inbox: MessageInbox | None
    ) -> None:
        """Release allowance only after admitted work settles for this exact inbox."""
        with self._lock:
            if self._inboxes.get(conversation_id) is not inbox:
                return
            self._deferred_counts.pop(conversation_id, None)
            self._inboxes.pop(conversation_id, None)
            if inbox is not None:
                self._close_locked(inbox)
            self._publish_locked()

    def close_inbox(self, conversation_id: str) -> None:
        """Invalidate capabilities and release this inbox's pending allowance."""
        self.finish_close_inbox(
            conversation_id, self.begin_close_inbox(conversation_id)
        )

    def _close_locked(self, inbox: MessageInbox) -> None:
        inbox._revoked.set()
        self._pending_count -= len(inbox._messages)
        self._pending_chars -= sum(len(message.body) for message in inbox._messages)
        inbox._messages.clear()
        inbox._senders.clear()
        inbox._reader = None

    def begin_close(self) -> None:
        """Permanently revoke admission without waiting for an owned SQL leaf."""
        self._revoked.set()
        for inbox in tuple(self._inboxes.values()):
            inbox._revoked.set()

    def close(self) -> None:
        """Release revoked queues after admitted durable work physically settles."""
        self.begin_close()
        with self._lock:
            self._closed = True
            for inbox in self._inboxes.values():
                self._close_locked(inbox)
            self._inboxes.clear()
            self._deferred_counts.clear()
            self._publish_locked()

    def _publish_locked(self) -> None:
        """Replace bounded immutable observations after an owned mutation commits.

        The single reference replacement lets UI readers observe the last
        committed state while SQLite waits, without taking the writer lock.
        These views grant no sending, collection or discard authority.
        """
        inboxes = {
            key: (
                inbox,
                tuple(
                    replace(message, identity=replace(message.identity))
                    for message in inbox._messages
                ),
            )
            for key, inbox in self._inboxes.items()
        }
        counts = {key: count for key, count in self._deferred_counts.items() if count}
        counts.update((key, len(view[1])) for key, view in inboxes.items() if view[1])
        self._published = (MappingProxyType(inboxes), MappingProxyType(counts))

    def pending_counts(self) -> dict[str, int]:
        """Read detached body-free counts without waiting for a durable writer."""
        if self._revoked.is_set():
            return {}
        inboxes, counts = self._published
        return {
            key: count
            for key, count in counts.items()
            if key not in inboxes or not inboxes[key][0]._revoked.is_set()
        }


class MessageInbox:
    """Conversation owner API; children receive only its bound sender."""

    def __init__(self, store: MessageStore, conversation_id: str) -> None:
        self._store = store
        self._conversation_id = conversation_id
        self._messages: list[ProgressMessage] = []
        self._senders: dict[str, _SenderState] = {}
        self._reader: MessageReader | None = None
        self._revoked = threading.Event()
        self._repository: FleetProgressRepository | None = None
        self._saved_conversation_id: str | None = None
        self._durable_failed = False
        self._promotion = None

    @property
    def saved_conversation_id(self) -> str | None:
        """Read the durable binding without touching SQLite."""
        with self._store._lock:
            return self._saved_conversation_id

    def _check_locked(self) -> None:
        if (
            self._store._closed
            or self._store._revoked.is_set()
            or self._revoked.is_set()
            or self._store._inboxes.get(self._conversation_id) is not self
        ):
            raise MessageError("unavailable")

    def _check_mutation_locked(self) -> None:
        self._check_locked()
        if self._durable_failed:
            raise MessageError("durable_unavailable")
        if self._promotion is not None:
            raise MessageError("saving")

    def pending_metadata(self) -> tuple[tuple[str, MessageIdentity], ...]:
        """Read committed IDs/source metadata without bodies or consumption."""
        return tuple(
            (message.message_id, message.identity) for message in self.snapshot()
        )

    def prepare_promotion(self) -> FleetProgressPromotionContribution:
        """Freeze a temporary queue until its atomic Save commits or rolls back."""
        from tldw_chatbook.DB.fleet_progress_repository import (
            FleetProgressPromotionContribution,
        )

        with self._store._lock:
            self._check_mutation_locked()
            if self._repository is not None:
                raise MessageError("unavailable")
            contribution = FleetProgressPromotionContribution(tuple(self._messages))
            self._promotion = contribution
            return contribution

    def settle_promotion(
        self,
        contribution: FleetProgressPromotionContribution,
        *,
        repository: FleetProgressRepository | None = None,
        saved_conversation_id: str | None = None,
    ) -> None:
        """Publish committed storage without SQL; rollback retains every report."""
        with self._store._lock:
            if self._promotion is not contribution:
                return
            self._promotion = None
            if repository is not None:
                if saved_conversation_id is None:
                    self._durable_failed = True
                    return
                self._repository = repository
                self._saved_conversation_id = saved_conversation_id

    def _append_locked(self, message: ProgressMessage) -> None:
        self._check_mutation_locked()
        if self._repository is not None:
            try:
                self._repository.append(self._saved_conversation_id, message)
            except MessageError:
                raise
            except Exception:  # noqa: BLE001 - uncertain commits fail closed
                self._durable_failed = True
                raise MessageError("durable_unavailable") from None
        self._messages.append(message)

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
        """Inspect the immutable committed view without blocking behind SQLite."""
        view = self._store._published[0].get(self._conversation_id)
        if (
            view is None
            or view[0] is not self
            or self._store._revoked.is_set()
            or self._revoked.is_set()
        ):
            raise MessageError("unavailable")
        return view[1]

    def discard(self, message_ids: Sequence[str]) -> int:
        """Remove selected snapshot IDs only, without refunding lifetime sends."""
        selected = set(message_ids)
        with self._store._lock:
            self._check_locked()
            return self._remove_locked(selected)

    def _remove_locked(self, selected: set[str]) -> int:
        self._check_mutation_locked()
        removed = [
            message for message in self._messages if message.message_id in selected
        ]
        if removed and self._repository is not None:
            try:
                self._repository.remove(
                    self._saved_conversation_id,
                    tuple(message.message_id for message in removed),
                )
            except Exception:  # noqa: BLE001 - uncertain commits fail closed
                # Even an uncertain postcommit error must never replay collection.
                self._durable_failed = True
                raise MessageError("durable_unavailable") from None
        self._messages = [
            message for message in self._messages if message.message_id not in selected
        ]
        self._store._pending_count -= len(removed)
        self._store._pending_chars -= sum(len(message.body) for message in removed)
        self._store._publish_locked()
        return len(removed)


@dataclass(frozen=True, eq=False)
class MessageSender:
    """Report-only capability validated by exact owner membership."""

    _inbox: MessageInbox
    _identity: MessageIdentity
    _revoked: threading.Event = field(default_factory=threading.Event, repr=False)

    def _is_bound_to(self, identity: MessageIdentity) -> bool:
        """Check an existing coordinator binding without renewing its authority."""
        inbox = self._inbox
        with inbox._store._lock:
            if (
                self._revoked.is_set()
                or inbox._store._closed
                or inbox._store._revoked.is_set()
                or inbox._revoked.is_set()
                or inbox._store._inboxes.get(inbox._conversation_id) is not inbox
            ):
                return False
            state = inbox._senders.get(self._identity.handle_id)
            return (
                state is not None
                and state.capability is self
                and self._identity == identity
            )

    def _state_locked(self) -> _SenderState:
        """Check this exact capability while its inbox store lock is held."""
        self._inbox._check_locked()
        if self._revoked.is_set():
            raise MessageError("unavailable")
        state = self._inbox._senders.get(self._identity.handle_id)
        if state is None or state.capability is not self:
            raise MessageError("unavailable")
        return state

    def _allowance_locked(self, body: str) -> _SenderState:
        """Reserve no allowance until the caller's complete enqueue succeeds."""
        state = self._state_locked()
        if (
            state.accepted_count >= MAX_CHILD_LIFETIME
            or state.accepted_chars + len(body) > MAX_CHILD_LIFETIME_CHARS
        ):
            raise MessageError("sender_limit")
        return state

    def send(self, message: object) -> str:
        """Atomically admit one complete report or return a fixed refusal."""
        body = _validate_text(message, MAX_MESSAGE_CHARS, size_code="message_too_large")
        report = ProgressMessage(uuid.uuid4().hex, self._identity, body)
        if len(_json(_envelope(report))) > MAX_ENVELOPE_CHARS:
            raise MessageError("message_too_large")
        inbox = self._inbox
        store = inbox._store
        with store._lock:
            state = self._allowance_locked(body)
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
            inbox._append_locked(report)
            state.accepted_count += 1
            state.accepted_chars += len(body)
            store._pending_count += 1
            store._pending_chars += len(body)
            store._publish_locked()
            callback = store.on_enqueue
        if callback is not None:
            try:
                callback(inbox._conversation_id, report.message_id, report.identity)
            except Exception:  # noqa: BLE001 - observation cannot undo committed admission
                return report.message_id
        return report.message_id

    def close(self) -> None:
        """Revoke this exact sender; never revoke a replacement capability."""
        self._revoked.set()
        inbox = self._inbox
        # Denial is immediate; exact membership cleanup cannot block lifecycle
        # locks behind another inbox's admitted SQL. Store close owns remnants.
        if not inbox._store._lock.acquire(blocking=False):
            return
        try:
            state = inbox._senders.get(self._identity.handle_id)
            if state is not None and state.capability is self:
                del inbox._senders[self._identity.handle_id]
        finally:
            inbox._store._lock.release()


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
