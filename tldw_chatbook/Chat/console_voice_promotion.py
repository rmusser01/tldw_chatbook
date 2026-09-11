"""Immutable identities and values for completed speculative voice promotion."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from uuid import UUID, uuid4, uuid5

from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.console_voice_trace_gateway import (
    ProvisionalTraceEnvelope,
    ProvisionalTraceManifest,
    ProvisionalTraceRegistry,
    ProvisionalTraceUnavailable,
    VoiceTraceImportContext,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_voice_attempts import VoiceAttemptSnapshot


_PROMOTION_ID_MAX_LENGTH = 512
_OPAQUE_ID_MAX_LENGTH = 512
_VOICE_TEXT_MAX_LENGTH = 1_000_000
_USAGE_JSON_MAX_LENGTH = 1_000_000
_VOICE_PROMOTION_NAMESPACE = UUID("2844583c-2048-44d4-9f3c-086d2af71e16")


def _required_string(value: object, name: str, *, maximum: int) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded non-empty string")
    return value


def _optional_string(value: object, name: str, *, maximum: int) -> str | None:
    if value is None:
        return None
    return _required_string(value, name, maximum=maximum)


def _non_negative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class ConsoleSessionBindingOrigin:
    """Process-local session binding captured before voice provider dispatch."""

    session_id: str
    session_incarnation: int
    persisted_conversation_id: str | None
    conversation_binding_revision: int

    def __post_init__(self) -> None:
        _required_string(self.session_id, "session_id", maximum=_OPAQUE_ID_MAX_LENGTH)
        _positive_integer(self.session_incarnation, "session_incarnation")
        _optional_string(
            self.persisted_conversation_id,
            "persisted_conversation_id",
            maximum=_OPAQUE_ID_MAX_LENGTH,
        )
        _non_negative_integer(
            self.conversation_binding_revision,
            "conversation_binding_revision",
        )


@dataclass(frozen=True, slots=True)
class VoicePromotionContext:
    """Sealed completed no-tool voice pair offered for one promotion claim."""

    promotion_id: str
    attempt_id: str
    origin: ConsoleSessionBindingOrigin
    expected_native_leaf_id: str | None
    expected_persisted_leaf_id: str | None
    user_text: str = field(repr=False)
    assistant_text: str = field(repr=False)
    usage_json: str | None = field(repr=False)
    terminal_boundary_id: str
    capture_eligible_at_dispatch: bool

    def __post_init__(self) -> None:
        _required_string(
            self.promotion_id,
            "promotion_id",
            maximum=_PROMOTION_ID_MAX_LENGTH,
        )
        _required_string(self.attempt_id, "attempt_id", maximum=_OPAQUE_ID_MAX_LENGTH)
        if type(self.origin) is not ConsoleSessionBindingOrigin:
            raise TypeError("origin must be a ConsoleSessionBindingOrigin")
        _optional_string(
            self.expected_native_leaf_id,
            "expected_native_leaf_id",
            maximum=_OPAQUE_ID_MAX_LENGTH,
        )
        _optional_string(
            self.expected_persisted_leaf_id,
            "expected_persisted_leaf_id",
            maximum=_OPAQUE_ID_MAX_LENGTH,
        )
        _required_string(
            self.user_text,
            "user_text",
            maximum=_VOICE_TEXT_MAX_LENGTH,
        )
        _required_string(
            self.assistant_text,
            "assistant_text",
            maximum=_VOICE_TEXT_MAX_LENGTH,
        )
        canonical_usage = _optional_string(
            self.usage_json,
            "usage_json",
            maximum=_USAGE_JSON_MAX_LENGTH,
        )
        if canonical_usage is not None:
            restored_usage = ProviderUsage.from_json(canonical_usage)
            if restored_usage is None or restored_usage.to_json() != canonical_usage:
                raise ValueError("usage_json must be canonical ProviderUsage JSON")
        _required_string(
            self.terminal_boundary_id,
            "terminal_boundary_id",
            maximum=_OPAQUE_ID_MAX_LENGTH,
        )
        _boolean(
            self.capture_eligible_at_dispatch,
            "capture_eligible_at_dispatch",
        )


@dataclass(frozen=True, slots=True)
class ResolvedVoicePromotionDestination:
    """Store-resolved durable or temporary destination for one claimed pair."""

    session_id: str
    session_incarnation: int
    persisted_conversation_id: str | None
    expected_persisted_leaf_id: str | None
    capture_eligible_at_dispatch: bool

    def __post_init__(self) -> None:
        _required_string(self.session_id, "session_id", maximum=_OPAQUE_ID_MAX_LENGTH)
        _positive_integer(self.session_incarnation, "session_incarnation")
        _optional_string(
            self.persisted_conversation_id,
            "persisted_conversation_id",
            maximum=_OPAQUE_ID_MAX_LENGTH,
        )
        _optional_string(
            self.expected_persisted_leaf_id,
            "expected_persisted_leaf_id",
            maximum=_OPAQUE_ID_MAX_LENGTH,
        )
        _boolean(
            self.capture_eligible_at_dispatch,
            "capture_eligible_at_dispatch",
        )


class ConsoleVoicePromotionClaimStatus(str, Enum):
    """Content-safe result of a store-owned promotion-lease claim."""

    CLAIMED = "claimed"
    TRANSIENT_CONTENTION = "transient_contention"
    CONFLICT = "conflict"


class ConsoleVoicePromotionRecoveryKind(str, Enum):
    """Whether a failed projection may be retried without a provider."""

    TEMPORARY_RETRYABLE = "temporary_retryable"
    DURABLE_RETRYABLE = "durable_retryable"
    VALIDATION_BLOCKED = "validation_blocked"


@dataclass(frozen=True, slots=True)
class ConsoleVoicePromotionLease:
    """Opaque capability authorizing exactly one store publication attempt."""

    lease_id: str
    session_id: str
    session_incarnation: int
    lease_revision: int
    promotion_id: str
    expected_native_leaf_id: str | None
    destination: ResolvedVoicePromotionDestination

    def __post_init__(self) -> None:
        _required_string(self.lease_id, "lease_id", maximum=_OPAQUE_ID_MAX_LENGTH)
        _required_string(self.session_id, "session_id", maximum=_OPAQUE_ID_MAX_LENGTH)
        _positive_integer(self.session_incarnation, "session_incarnation")
        _positive_integer(self.lease_revision, "lease_revision")
        _required_string(
            self.promotion_id,
            "promotion_id",
            maximum=_PROMOTION_ID_MAX_LENGTH,
        )
        _optional_string(
            self.expected_native_leaf_id,
            "expected_native_leaf_id",
            maximum=_OPAQUE_ID_MAX_LENGTH,
        )
        if type(self.destination) is not ResolvedVoicePromotionDestination:
            raise TypeError("destination must be a ResolvedVoicePromotionDestination")
        if (
            self.destination.session_id != self.session_id
            or self.destination.session_incarnation != self.session_incarnation
        ):
            raise ValueError("destination must match lease session identity")


@dataclass(frozen=True, slots=True)
class ConsoleVoicePromotionClaim:
    """Typed lease-claim result that never carries protected turn content."""

    status: ConsoleVoicePromotionClaimStatus
    lease: ConsoleVoicePromotionLease | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not ConsoleVoicePromotionClaimStatus:
            raise TypeError("status must be a ConsoleVoicePromotionClaimStatus")
        if self.status is ConsoleVoicePromotionClaimStatus.CLAIMED:
            if type(self.lease) is not ConsoleVoicePromotionLease:
                raise ValueError("a claimed outcome requires its lease")
        elif self.lease is not None:
            raise ValueError("a non-claimed outcome cannot carry a lease")

    @classmethod
    def claimed(cls, lease: ConsoleVoicePromotionLease) -> "ConsoleVoicePromotionClaim":
        """Return the sole claim result that grants a publication capability."""
        return cls(ConsoleVoicePromotionClaimStatus.CLAIMED, lease)


@dataclass(frozen=True, slots=True)
class ConsoleVoicePromotionRecovery:
    """Exact provider-free retry projection retained after publication failure."""

    lease: ConsoleVoicePromotionLease
    context: VoicePromotionContext = field(repr=False)
    commit: "CompletedVoicePairCommit | None" = None
    kind: ConsoleVoicePromotionRecoveryKind = (
        ConsoleVoicePromotionRecoveryKind.TEMPORARY_RETRYABLE
    )

    def __post_init__(self) -> None:
        if type(self.lease) is not ConsoleVoicePromotionLease:
            raise TypeError("lease must be a ConsoleVoicePromotionLease")
        if type(self.context) is not VoicePromotionContext:
            raise TypeError("context must be a VoicePromotionContext")
        if self.lease.promotion_id != self.context.promotion_id:
            raise ValueError("recovery lease and context must share a promotion")
        if (
            self.commit is not None
            and type(self.commit) is not CompletedVoicePairCommit
        ):
            raise TypeError("commit must be a CompletedVoicePairCommit or None")
        if type(self.kind) is not ConsoleVoicePromotionRecoveryKind:
            raise TypeError("kind must be a ConsoleVoicePromotionRecoveryKind")
        if self.kind is ConsoleVoicePromotionRecoveryKind.DURABLE_RETRYABLE:
            if self.commit is None:
                raise ValueError("durable recovery requires its validated commit")
        elif self.commit is not None:
            raise ValueError("only durable recovery carries a commit")


@dataclass(frozen=True, slots=True)
class ConsoleVoicePromotionRebranch:
    """Fresh store-owned lease and frozen pair for an explicit rebranch."""

    lease: ConsoleVoicePromotionLease
    context: VoicePromotionContext = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.lease) is not ConsoleVoicePromotionLease:
            raise TypeError("lease must be a ConsoleVoicePromotionLease")
        if type(self.context) is not VoicePromotionContext:
            raise TypeError("context must be a VoicePromotionContext")
        if self.lease.promotion_id != self.context.promotion_id:
            raise ValueError("rebranch lease and context must share a promotion")
        if self.lease.session_id != self.context.origin.session_id:
            raise ValueError("rebranch lease and context must share a session")


@dataclass(frozen=True, slots=True)
class CompletedVoicePairCommit:
    """Durable identities returned after the completed-pair transaction commits."""

    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    terminal_receipt_id: str
    active_leaf_message_id: str
    already_committed: bool = False
    user_revision_id: str | None = None
    assistant_revision_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "conversation_id",
            "user_message_id",
            "assistant_message_id",
            "terminal_receipt_id",
            "active_leaf_message_id",
        ):
            _required_string(
                getattr(self, name),
                name,
                maximum=_OPAQUE_ID_MAX_LENGTH,
            )
        _boolean(self.already_committed, "already_committed")
        for name in ("user_revision_id", "assistant_revision_id"):
            _optional_string(
                getattr(self, name),
                name,
                maximum=_OPAQUE_ID_MAX_LENGTH,
            )
        if (self.user_revision_id is None) is not (self.assistant_revision_id is None):
            raise ValueError("voice revision identities must be supplied together")


@dataclass(frozen=True, slots=True)
class VoicePromotionIdentitySet:
    """Domain-separated durable and operation identities for one promotion."""

    user_message_id: str
    assistant_message_id: str
    terminal_receipt_id: str
    operation_id: str
    retry_id: str


class VoicePromotionClaimStatus(str, Enum):
    """Content-free result of the app-lifetime acceptance linearization."""

    CLAIMED = "claimed"
    TRANSIENT_CONTENTION = "transient_contention"
    CONFLICT = "conflict"
    QUIT_FENCED = "quit_fenced"
    CLOSE_FENCED = "close_fenced"
    UNAVAILABLE = "unavailable"


class VoicePromotionOutcomeStatus(str, Enum):
    """Content-free terminal state of one claimed promotion."""

    PROMOTED = "promoted"
    RECOVERY = "recovery"


@dataclass(frozen=True, slots=True)
class VoicePromotionClaim:
    """Opaque owner claim; protected turn content is excluded from its repr."""

    status: VoicePromotionClaimStatus
    promotion_id: str
    session_id: str
    claim_id: str | None = None
    lease: ConsoleVoicePromotionLease | None = field(default=None, repr=False)
    _context: VoicePromotionContext | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _store: Any | None = field(default=None, repr=False, compare=False)
    _owner_id: str | None = field(default=None, repr=False, compare=False)
    _owned: "_OwnedVoicePromotion | None" = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.status) is not VoicePromotionClaimStatus:
            raise TypeError("status must be a VoicePromotionClaimStatus")
        _required_string(
            self.promotion_id,
            "promotion_id",
            maximum=_PROMOTION_ID_MAX_LENGTH,
        )
        _required_string(self.session_id, "session_id", maximum=_OPAQUE_ID_MAX_LENGTH)
        if self.status is VoicePromotionClaimStatus.CLAIMED:
            _required_string(self.claim_id, "claim_id", maximum=_OPAQUE_ID_MAX_LENGTH)
            if type(self.lease) is not ConsoleVoicePromotionLease:
                raise ValueError("a claimed owner outcome requires its lease")
            if type(self._context) is not VoicePromotionContext:
                raise ValueError("a claimed owner outcome requires protected context")
            if self._store is None or self._owner_id is None or self._owned is None:
                raise ValueError("a claimed owner outcome requires exact owner custody")
        elif any(
            value is not None
            for value in (
                self.claim_id,
                self.lease,
                self._context,
                self._store,
                self._owner_id,
                self._owned,
            )
        ):
            raise ValueError("a refused owner claim cannot carry claim authority")


@dataclass(frozen=True, slots=True)
class VoicePromotionOutcome:
    """Content-free terminal projection for one promotion attempt."""

    status: VoicePromotionOutcomeStatus
    promotion_id: str
    session_id: str
    failure_code: str | None = None
    _commit: CompletedVoicePairCommit | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.status) is not VoicePromotionOutcomeStatus:
            raise TypeError("status must be a VoicePromotionOutcomeStatus")
        _required_string(
            self.promotion_id,
            "promotion_id",
            maximum=_PROMOTION_ID_MAX_LENGTH,
        )
        _required_string(self.session_id, "session_id", maximum=_OPAQUE_ID_MAX_LENGTH)
        _optional_string(
            self.failure_code,
            "failure_code",
            maximum=_OPAQUE_ID_MAX_LENGTH,
        )
        if (
            self._commit is not None
            and type(self._commit) is not CompletedVoicePairCommit
        ):
            raise TypeError("commit must be a CompletedVoicePairCommit or None")


@dataclass(frozen=True, slots=True)
class VoicePromotionOwnerStatus:
    """Content-free app-lifetime promotion status."""

    revision: int
    active_claim_count: int
    recovery_count: int
    quit_fenced: bool


@dataclass(frozen=True, slots=True)
class VoicePromotionQuitToken:
    """Opaque reversible fence for one graceful-quit attempt."""

    _owner_id: str = field(repr=False)
    _fence_id: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class VoicePromotionQuitPermit:
    """Opaque sealed proof that one exact quit fence reached quiescence."""

    _owner_id: str = field(repr=False)
    _fence_id: str = field(repr=False)
    _sealed_revision: int = field(repr=False)


@dataclass(frozen=True, slots=True)
class VoicePromotionSessionCloseToken:
    """Opaque exact fence preventing new claims for one closing session."""

    _owner_id: str = field(repr=False)
    _session_id: str = field(repr=False)
    _fence_id: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class VoicePromotionRecoveryHandle:
    """Opaque exact capability for one provider-free recovery decision."""

    promotion_id: str
    session_id: str
    _owner_id: str = field(repr=False)
    _recovery_id: str = field(repr=False)


@dataclass(slots=True)
class _OwnedVoicePromotion:
    claim: VoicePromotionClaim | None = None
    task: asyncio.Task[VoicePromotionOutcome] | None = None


@dataclass(frozen=True, slots=True)
class _OwnedVoiceRecovery:
    claim: VoicePromotionClaim = field(repr=False)
    handle: VoicePromotionRecoveryHandle


class VoicePromotionOwner:
    """App-lifetime custodian for claimed completed-pair publication."""

    def __init__(
        self,
        store_provider: Callable[[], Any],
        *,
        sync_runner: Callable[[Callable[[], Any]], Awaitable[Any]] | None = None,
    ) -> None:
        self._store_provider = store_provider
        self._sync_runner = sync_runner or asyncio.to_thread
        self._lock = threading.RLock()
        self._owner_id = str(uuid4())
        self._revision = 0
        self._claims: dict[str, _OwnedVoicePromotion] = {}
        self._claim_by_session: dict[str, str] = {}
        self._recoveries: dict[str, _OwnedVoiceRecovery] = {}
        self._recovery_decisions: set[str] = set()
        self._tasks: set[asyncio.Task[VoicePromotionOutcome]] = set()
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = (
            set()
        )
        self._quit_fence: tuple[str, str, int | None] | None = None
        self._quit_token: VoicePromotionQuitToken | None = None
        self._quit_permit: VoicePromotionQuitPermit | None = None
        self._session_close_tokens: dict[str, VoicePromotionSessionCloseToken] = {}

    @property
    def status(self) -> VoicePromotionOwnerStatus:
        """Return a content-free snapshot separate from protected recovery state."""

        with self._lock:
            return VoicePromotionOwnerStatus(
                revision=self._revision,
                active_claim_count=len(self._claims),
                recovery_count=len(self._recoveries),
                quit_fenced=self._quit_fence is not None,
            )

    def _notify_waiters_locked(self) -> None:
        waiters = tuple(self._waiters)
        for loop, waiter in waiters:
            if waiter.done() or loop.is_closed():
                continue
            loop.call_soon_threadsafe(self._resolve_waiter, waiter)

    @staticmethod
    def _resolve_waiter(waiter: asyncio.Future[None]) -> None:
        if not waiter.done():
            waiter.set_result(None)

    def _changed_locked(self) -> None:
        self._revision += 1
        self._notify_waiters_locked()

    @staticmethod
    def _map_store_claim_status(
        status: ConsoleVoicePromotionClaimStatus,
    ) -> VoicePromotionClaimStatus:
        if status is ConsoleVoicePromotionClaimStatus.TRANSIENT_CONTENTION:
            return VoicePromotionClaimStatus.TRANSIENT_CONTENTION
        return VoicePromotionClaimStatus.CONFLICT

    def try_claim(self, context: VoicePromotionContext) -> VoicePromotionClaim:
        """Synchronously acquire store authority and app-lifetime custody."""

        if type(context) is not VoicePromotionContext:
            raise TypeError("context must be a VoicePromotionContext")
        session_id = context.origin.session_id
        with self._lock:
            if self._quit_fence is not None:
                return VoicePromotionClaim(
                    VoicePromotionClaimStatus.QUIT_FENCED,
                    context.promotion_id,
                    session_id,
                )
            if session_id in self._session_close_tokens:
                return VoicePromotionClaim(
                    VoicePromotionClaimStatus.CLOSE_FENCED,
                    context.promotion_id,
                    session_id,
                )
            existing_id = self._claim_by_session.get(session_id)
            if existing_id is not None:
                existing = self._claims[existing_id].claim
                if existing._context == context:
                    return existing
                return VoicePromotionClaim(
                    VoicePromotionClaimStatus.TRANSIENT_CONTENTION,
                    context.promotion_id,
                    session_id,
                )
            if session_id in self._recoveries:
                return VoicePromotionClaim(
                    VoicePromotionClaimStatus.TRANSIENT_CONTENTION,
                    context.promotion_id,
                    session_id,
                )
            store = self._store_provider()
            if store is None:
                return VoicePromotionClaim(
                    VoicePromotionClaimStatus.UNAVAILABLE,
                    context.promotion_id,
                    session_id,
                )
            store_claim = store.try_claim_voice_promotion(context)
            if store_claim.status is not ConsoleVoicePromotionClaimStatus.CLAIMED:
                return VoicePromotionClaim(
                    self._map_store_claim_status(store_claim.status),
                    context.promotion_id,
                    session_id,
                )
            assert store_claim.lease is not None
            claim_id = str(uuid4())
            owned = _OwnedVoicePromotion()
            claim = VoicePromotionClaim(
                VoicePromotionClaimStatus.CLAIMED,
                context.promotion_id,
                session_id,
                claim_id,
                store_claim.lease,
                context,
                store,
                self._owner_id,
                owned,
            )
            owned.claim = claim
            self._claims[claim_id] = owned
            self._claim_by_session[session_id] = claim_id
            coroutine = self._run_promotion(claim)
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(
                    coroutine,
                    name=f"voice-promotion-{claim_id[:8]}",
                )
            except Exception:
                coroutine.close()
                self._remove_claim_locked(claim)
                store.abort_voice_promotion(store_claim.lease)
                return VoicePromotionClaim(
                    VoicePromotionClaimStatus.UNAVAILABLE,
                    context.promotion_id,
                    session_id,
                )
            owned.task = task
            self._tasks.add(task)
            task.add_done_callback(self._promotion_task_done)
            self._changed_locked()
            return claim

    async def promote(self, claim: VoicePromotionClaim) -> VoicePromotionOutcome:
        """Run one claimed publication under owner-rooted task custody."""

        if type(claim) is not VoicePromotionClaim:
            raise TypeError("claim must be a VoicePromotionClaim")
        if claim.status is not VoicePromotionClaimStatus.CLAIMED:
            raise RuntimeError("Voice promotion claim was not accepted.")
        owned = claim._owned
        with self._lock:
            if (
                claim._owner_id != self._owner_id
                or owned is None
                or owned.claim is not claim
                or owned.task is None
            ):
                raise RuntimeError("Voice promotion claim is stale.")
            task = owned.task
        return await asyncio.shield(task)

    def _promotion_task_done(self, task: asyncio.Task[VoicePromotionOutcome]) -> None:
        with self._lock:
            self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            return

    async def _run_promotion(self, claim: VoicePromotionClaim) -> VoicePromotionOutcome:
        assert claim.lease is not None and claim._context is not None
        store = claim._store
        assert store is not None
        failure_code: str | None = None
        committed: CompletedVoicePairCommit | None = None
        try:
            if claim.lease.destination.persisted_conversation_id is None:
                store.publish_temporary_voice_pair(claim.lease, claim._context)
            else:
                persistence = getattr(store, "persistence", None)
                commit_pair = getattr(persistence, "commit_completed_voice_pair", None)
                if not callable(commit_pair):
                    raise RuntimeError("Voice promotion persistence is unavailable.")

                def commit() -> CompletedVoicePairCommit:
                    return commit_pair(
                        destination=claim.lease.destination,
                        context=claim._context,
                    )

                try:
                    committed = await self._sync_runner(commit)
                except Exception:
                    # The first call may have committed before reporting failure.
                    # The persistence seam is idempotent and reconciles by promotion ID.
                    committed = await self._sync_runner(commit)
                store.publish_durable_voice_pair(
                    claim.lease,
                    committed,
                    claim._context,
                )
        except Exception as exc:  # noqa: BLE001 -- becomes protected recovery
            failure_code = type(exc).__name__
            with self._lock:
                self._recoveries[claim.session_id] = _OwnedVoiceRecovery(
                    claim,
                    VoicePromotionRecoveryHandle(
                        claim.promotion_id,
                        claim.session_id,
                        self._owner_id,
                        str(uuid4()),
                    ),
                )
                self._remove_claim_locked(claim)
                self._changed_locked()
            return VoicePromotionOutcome(
                VoicePromotionOutcomeStatus.RECOVERY,
                claim.promotion_id,
                claim.session_id,
                failure_code,
                committed,
            )
        with self._lock:
            self._remove_claim_locked(claim)
            self._changed_locked()
        return VoicePromotionOutcome(
            VoicePromotionOutcomeStatus.PROMOTED,
            claim.promotion_id,
            claim.session_id,
            _commit=committed,
        )

    def recovery_for_session(
        self,
        session_id: str,
    ) -> VoicePromotionRecoveryHandle | None:
        """Return the current content-free recovery capability for a session."""

        _required_string(session_id, "session_id", maximum=_OPAQUE_ID_MAX_LENGTH)
        with self._lock:
            recovery = self._recoveries.get(session_id)
            return None if recovery is None else recovery.handle

    def _recovery_locked(
        self,
        handle: VoicePromotionRecoveryHandle,
    ) -> _OwnedVoiceRecovery:
        if type(handle) is not VoicePromotionRecoveryHandle:
            raise TypeError("handle must be a VoicePromotionRecoveryHandle")
        recovery = self._recoveries.get(handle.session_id)
        if (
            handle._owner_id != self._owner_id
            or recovery is None
            or recovery.handle is not handle
        ):
            raise RuntimeError("Voice promotion recovery is stale.")
        if handle.session_id in self._recovery_decisions:
            raise RuntimeError("Voice promotion recovery decision is already active.")
        return recovery

    async def retry_recovery(
        self,
        handle: VoicePromotionRecoveryHandle,
    ) -> VoicePromotionOutcome:
        """Retry the exact frozen pair without another provider call."""

        with self._lock:
            recovery = self._recovery_locked(handle)
            claim = recovery.claim
            assert claim.claim_id is not None and claim._owned is not None
            self._recoveries.pop(handle.session_id, None)
            self._claims[claim.claim_id] = claim._owned
            self._claim_by_session[claim.session_id] = claim.claim_id
            coroutine = self._run_promotion(claim)
            try:
                task = asyncio.get_running_loop().create_task(
                    coroutine,
                    name=f"voice-promotion-retry-{claim.claim_id[:8]}",
                )
            except Exception:
                coroutine.close()
                self._remove_claim_locked(claim)
                self._recoveries[handle.session_id] = recovery
                raise RuntimeError(
                    "Voice promotion recovery retry is unavailable."
                ) from None
            claim._owned.task = task
            self._tasks.add(task)
            task.add_done_callback(self._promotion_task_done)
            self._changed_locked()
        return await asyncio.shield(task)

    def select_recovery_leaf(
        self,
        handle: VoicePromotionRecoveryHandle,
        message_id: str | None,
    ) -> bool:
        """Select an exact native parent under one live recovery capability."""

        with self._lock:
            recovery = self._recovery_locked(handle)
            claim = recovery.claim
            store = claim._store
            lease = claim.lease
            assert store is not None and lease is not None
            self._recovery_decisions.add(handle.session_id)
        try:
            return bool(
                store.select_voice_promotion_recovery_leaf(
                    lease,
                    message_id,
                )
            )
        finally:
            with self._lock:
                self._recovery_decisions.discard(handle.session_id)

    async def rebranch_recovery(
        self,
        handle: VoicePromotionRecoveryHandle,
    ) -> VoicePromotionOutcome:
        """Attach the exact frozen pair to the store-selected leaf and retry."""

        with self._lock:
            recovery = self._recovery_locked(handle)
            old_claim = recovery.claim
            store = old_claim._store
            lease = old_claim.lease
            assert store is not None and lease is not None
            self._recovery_decisions.add(handle.session_id)
        try:
            rebranch = store.rebranch_voice_promotion_recovery(lease)
        except BaseException:
            with self._lock:
                self._recovery_decisions.discard(handle.session_id)
            raise

        owned = _OwnedVoicePromotion()
        claim = VoicePromotionClaim(
            VoicePromotionClaimStatus.CLAIMED,
            old_claim.promotion_id,
            old_claim.session_id,
            str(uuid4()),
            rebranch.lease,
            rebranch.context,
            store,
            self._owner_id,
            owned,
        )
        owned.claim = claim
        coroutine = self._run_promotion(claim)
        with self._lock:
            try:
                try:
                    task = asyncio.get_running_loop().create_task(
                        coroutine,
                        name=f"voice-promotion-rebranch-{claim.claim_id[:8]}",
                    )
                except Exception:
                    coroutine.close()
                    self._recoveries[handle.session_id] = _OwnedVoiceRecovery(
                        claim,
                        handle,
                    )
                    self._changed_locked()
                    raise RuntimeError(
                        "Voice promotion recovery rebranch is unavailable."
                    ) from None
                self._recoveries.pop(handle.session_id, None)
                self._claims[claim.claim_id] = owned
                self._claim_by_session[claim.session_id] = claim.claim_id
                owned.task = task
                self._tasks.add(task)
                task.add_done_callback(self._promotion_task_done)
                self._changed_locked()
            finally:
                self._recovery_decisions.discard(handle.session_id)
        return await asyncio.shield(task)

    def discard_recovery(self, handle: VoicePromotionRecoveryHandle) -> bool:
        """Explicitly discard the exact frozen pair and release its store lease."""

        with self._lock:
            recovery = self._recovery_locked(handle)
            claim = recovery.claim
            store = claim._store
            lease = claim.lease
            assert store is not None and lease is not None
            self._recovery_decisions.add(handle.session_id)
        try:
            store_recovery = store.voice_promotion_recovery(lease)
            released = (
                store.discard_voice_promotion_recovery(lease)
                if store_recovery is not None
                else store.abort_voice_promotion(lease)
            )
        except BaseException:
            with self._lock:
                self._recovery_decisions.discard(handle.session_id)
            raise
        with self._lock:
            try:
                if not released:
                    return False
                if self._recoveries.get(handle.session_id) is recovery:
                    self._recoveries.pop(handle.session_id, None)
                    self._changed_locked()
                return True
            finally:
                self._recovery_decisions.discard(handle.session_id)

    def _remove_claim_locked(self, claim: VoicePromotionClaim) -> None:
        if claim.claim_id is not None:
            self._claims.pop(claim.claim_id, None)
        if self._claim_by_session.get(claim.session_id) == claim.claim_id:
            self._claim_by_session.pop(claim.session_id, None)

    async def _wait_for_change(
        self,
        timeout: float,
        *,
        after_revision: int | None = None,
    ) -> bool:
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        entry = (loop, waiter)
        with self._lock:
            if after_revision is not None and self._revision != after_revision:
                return True
            self._waiters.add(entry)
        try:
            await asyncio.wait_for(waiter, timeout=max(0.0, float(timeout)))
            return True
        except TimeoutError:
            return False
        finally:
            with self._lock:
                self._waiters.discard(entry)

    async def wait_for_session(self, session_id: str, timeout: float) -> bool:
        """Wait boundedly without cancelling the owned publication task."""

        _required_string(session_id, "session_id", maximum=_OPAQUE_ID_MAX_LENGTH)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(timeout))
        while True:
            with self._lock:
                if session_id in self._recoveries:
                    return False
                if session_id not in self._claim_by_session:
                    return True
                observed_revision = self._revision
            if not await self._wait_for_change(
                deadline - loop.time(),
                after_revision=observed_revision,
            ):
                return False

    def begin_session_close(self, session_id: str) -> VoicePromotionSessionCloseToken:
        """Fence new claims for one exact session before close draining starts."""

        _required_string(session_id, "session_id", maximum=_OPAQUE_ID_MAX_LENGTH)
        with self._lock:
            if session_id in self._session_close_tokens:
                raise RuntimeError("A voice-promotion session close is already active.")
            token = VoicePromotionSessionCloseToken(
                self._owner_id,
                session_id,
                str(uuid4()),
            )
            self._session_close_tokens[session_id] = token
            self._changed_locked()
            return token

    def _release_session_close(self, token: VoicePromotionSessionCloseToken) -> None:
        if type(token) is not VoicePromotionSessionCloseToken:
            raise TypeError("token must be a VoicePromotionSessionCloseToken")
        with self._lock:
            if self._session_close_tokens.get(token._session_id) is token:
                self._session_close_tokens.pop(token._session_id, None)
                self._changed_locked()

    def abort_session_close(self, token: VoicePromotionSessionCloseToken) -> None:
        """Release only the exact reversible session-close fence."""

        self._release_session_close(token)

    def complete_session_close(self, token: VoicePromotionSessionCloseToken) -> None:
        """Retire the exact fence after irreversible close settlement."""

        self._release_session_close(token)

    def begin_quit(self) -> VoicePromotionQuitToken:
        """Fence new claims with one reversible exact-token capability."""

        with self._lock:
            if self._quit_fence is not None:
                raise RuntimeError("A voice-promotion quit fence is already active.")
            fence_id = str(uuid4())
            self._quit_fence = (fence_id, "token", None)
            self._quit_token = VoicePromotionQuitToken(self._owner_id, fence_id)
            self._quit_permit = None
            self._changed_locked()
            return self._quit_token

    def _token_matches_locked(self, token: VoicePromotionQuitToken) -> bool:
        return bool(
            type(token) is VoicePromotionQuitToken
            and token is self._quit_token
            and self._quit_fence == (token._fence_id, "token", None)
        )

    async def wait_for_quiescence(
        self,
        token: VoicePromotionQuitToken,
        timeout: float,
    ) -> bool:
        """Wait for zero claims and recoveries under one exact quit fence."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(timeout))
        while True:
            with self._lock:
                if not self._token_matches_locked(token):
                    return False
                if self._recoveries:
                    return False
                if not self._claims:
                    return True
                observed_revision = self._revision
            if not await self._wait_for_change(
                deadline - loop.time(),
                after_revision=observed_revision,
            ):
                return False

    def seal_quiescent(
        self,
        token: VoicePromotionQuitToken,
    ) -> VoicePromotionQuitPermit:
        """CAS one exact reversible token into a revision-sealed permit."""

        with self._lock:
            if not self._token_matches_locked(token):
                raise RuntimeError("Voice-promotion quit token is stale.")
            if self._claims or self._recoveries:
                raise RuntimeError("Voice promotions are not quiescent.")
            sealed_revision = self._revision
            permit = VoicePromotionQuitPermit(
                self._owner_id,
                token._fence_id,
                sealed_revision,
            )
            self._quit_token = None
            self._quit_permit = permit
            self._quit_fence = (token._fence_id, "permit", sealed_revision)
            return permit

    def abort_quit(self, token_or_permit: object) -> None:
        """Idempotently release only the matching unconsumed quit fence."""

        with self._lock:
            matches = False
            if type(token_or_permit) is VoicePromotionQuitToken:
                matches = self._token_matches_locked(token_or_permit)
            elif type(token_or_permit) is VoicePromotionQuitPermit:
                permit = token_or_permit
                matches = bool(
                    permit is self._quit_permit
                    and self._quit_fence
                    == (permit._fence_id, "permit", permit._sealed_revision)
                )
            if matches:
                self._quit_fence = None
                self._quit_token = None
                self._quit_permit = None
                self._changed_locked()

    def consume_quit_permit(self, permit: VoicePromotionQuitPermit) -> None:
        """Consume one exact permit while retaining its terminal claim fence."""

        if type(permit) is not VoicePromotionQuitPermit:
            raise TypeError("permit must be a VoicePromotionQuitPermit")
        with self._lock:
            expected = (permit._fence_id, "permit", permit._sealed_revision)
            if (
                permit is not self._quit_permit
                or self._quit_fence != expected
                or permit._sealed_revision != self._revision
                or self._claims
                or self._recoveries
            ):
                raise RuntimeError("Voice-promotion quit permit is stale.")
            self._quit_fence = (permit._fence_id, "consumed", permit._sealed_revision)
            self._quit_permit = None
            self._changed_locked()


class VoiceWinningPromotion:
    """Claim one valid winner, publish its pair, then import trace best-effort."""

    def __init__(
        self,
        owner: VoicePromotionOwner,
        *,
        trace_registry: ProvisionalTraceRegistry | None = None,
        trace_context_factory: Callable[
            [VoicePromotionContext, CompletedVoicePairCommit],
            VoiceTraceImportContext,
        ]
        | None = None,
        trace_importer: Callable[
            [
                ProvisionalTraceManifest,
                tuple[ProvisionalTraceEnvelope, ...],
                VoiceTraceImportContext,
            ],
            Any,
        ]
        | None = None,
        sync_runner: Callable[[Callable[[], Any]], Awaitable[Any]] | None = None,
    ) -> None:
        if type(owner) is not VoicePromotionOwner:
            raise TypeError("owner must be a VoicePromotionOwner")
        if (trace_context_factory is None) is not (trace_importer is None):
            raise ValueError("trace context and importer must be supplied together")
        self._owner = owner
        self._trace_registry = trace_registry
        self._trace_context_factory = trace_context_factory
        self._trace_importer = trace_importer
        self._sync_runner = sync_runner or asyncio.to_thread
        self._settlement_tasks: set[asyncio.Task[VoicePromotionOutcome]] = set()

    def promote(
        self,
        context: VoicePromotionContext,
        snapshot: "VoiceAttemptSnapshot",
        *,
        on_claim: Callable[[VoicePromotionClaim], None] | None = None,
    ) -> Awaitable[VoicePromotionOutcome]:
        """Synchronously claim a sealed winner before returning async settlement."""

        if type(context) is not VoicePromotionContext:
            raise TypeError("context must be a VoicePromotionContext")
        if not all(
            hasattr(snapshot, name)
            for name in (
                "response_text",
                "tool_request",
                "speech_frozen",
                "trace_manifest",
                "trace_envelopes",
            )
        ):
            raise TypeError("snapshot must be a VoiceAttemptSnapshot")
        trace = self._validated_trace(context, snapshot)
        if (
            snapshot.tool_request is not None
            or snapshot.speech_frozen
            or snapshot.response_text != context.assistant_text
        ):
            self._abandon_trace(snapshot.trace_manifest)
            return self._refused(context, "invalid_winning_snapshot")
        claim = self._owner.try_claim(context)
        if claim.status is not VoicePromotionClaimStatus.CLAIMED:
            self._abandon_trace(snapshot.trace_manifest)
            return self._refused(context, f"claim_{claim.status.value}")
        settlement = self._finish(claim, context, trace)
        try:
            task = asyncio.get_running_loop().create_task(
                settlement,
                name=f"voice-winning-settlement-{context.promotion_id[:8]}",
            )
        except Exception:
            settlement.close()
            self._abandon_trace(snapshot.trace_manifest)
            # try_claim already retained the publication task. Notify before
            # constructing the observer, so a raising callback leaks no coroutine.
            if on_claim is not None:
                on_claim(claim)
            return self._owner.promote(claim)
        self._settlement_tasks.add(task)
        task.add_done_callback(self._settlement_done)
        # The original owner and trace settlement already hold custody. An
        # observer callback can fail or disappear without cancelling either.
        if on_claim is not None:
            on_claim(claim)
        return asyncio.shield(task)

    def _settlement_done(self, task: asyncio.Task[VoicePromotionOutcome]) -> None:
        self._settlement_tasks.discard(task)
        if task.cancelled():
            return
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.exception()

    def _validated_trace(
        self,
        context: VoicePromotionContext,
        snapshot: "VoiceAttemptSnapshot",
    ) -> tuple[ProvisionalTraceManifest, tuple[ProvisionalTraceEnvelope, ...]] | None:
        manifest = snapshot.trace_manifest
        envelopes = snapshot.trace_envelopes
        typed_envelopes = bool(
            type(envelopes) is tuple
            and all(
                type(envelope) is ProvisionalTraceEnvelope for envelope in envelopes
            )
        )
        valid = bool(
            context.capture_eligible_at_dispatch
            and type(manifest) is ProvisionalTraceManifest
            and typed_envelopes
            and manifest.promotion_id == context.promotion_id
            and manifest.attempt_id == context.attempt_id
            and manifest.expected_call_count == len(envelopes)
            and manifest.envelope_ids
            == tuple(envelope.envelope_id for envelope in envelopes)
        )
        if valid:
            assert manifest is not None
            return manifest, envelopes
        self._abandon_trace(manifest)
        return None

    async def _finish(
        self,
        claim: VoicePromotionClaim,
        context: VoicePromotionContext,
        trace: tuple[
            ProvisionalTraceManifest,
            tuple[ProvisionalTraceEnvelope, ...],
        ]
        | None,
    ) -> VoicePromotionOutcome:
        outcome = await self._owner.promote(claim)
        if trace is None:
            return outcome
        manifest, envelopes = trace
        commit = outcome._commit
        if (
            commit is None
            or commit.user_revision_id is None
            or commit.assistant_revision_id is None
            or self._trace_context_factory is None
            or self._trace_importer is None
        ):
            self._abandon_trace(manifest)
            return outcome
        try:
            trace_context = self._trace_context_factory(context, commit)
            if type(trace_context) is not VoiceTraceImportContext:
                raise TypeError("trace context factory returned an invalid value")

            def import_trace() -> Any:
                assert self._trace_importer is not None
                return self._trace_importer(manifest, envelopes, trace_context)

            result = await self._sync_runner(import_trace)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            self._abandon_trace(manifest)
            raise
        except BaseException:
            self._abandon_trace(manifest)
        return outcome

    def _abandon_trace(self, manifest: ProvisionalTraceManifest | None) -> None:
        registry = self._trace_registry
        if registry is None or type(manifest) is not ProvisionalTraceManifest:
            return
        with contextlib.suppress(ProvisionalTraceUnavailable):
            registry.abandon_manifest(manifest)

    @staticmethod
    async def _refused(
        context: VoicePromotionContext,
        failure_code: str,
    ) -> VoicePromotionOutcome:
        return VoicePromotionOutcome(
            VoicePromotionOutcomeStatus.RECOVERY,
            context.promotion_id,
            context.origin.session_id,
            failure_code,
        )


def new_voice_promotion_id() -> str:
    """Return one random opaque promotion identity."""

    return str(uuid4())


def derive_voice_promotion_identities(
    promotion_id: str,
) -> VoicePromotionIdentitySet:
    """Derive stable domain-separated identities solely from a promotion ID."""

    _required_string(
        promotion_id,
        "promotion_id",
        maximum=_PROMOTION_ID_MAX_LENGTH,
    )

    def derived(label: str) -> str:
        return str(uuid5(_VOICE_PROMOTION_NAMESPACE, f"{label}:{promotion_id}"))

    return VoicePromotionIdentitySet(
        user_message_id=derived("user-message"),
        assistant_message_id=derived("assistant-message"),
        terminal_receipt_id=derived("terminal-receipt"),
        operation_id=derived("operation"),
        retry_id=derived("retry"),
    )


__all__ = [
    "CompletedVoicePairCommit",
    "ConsoleSessionBindingOrigin",
    "ConsoleVoicePromotionClaim",
    "ConsoleVoicePromotionClaimStatus",
    "ConsoleVoicePromotionLease",
    "ConsoleVoicePromotionRecovery",
    "ConsoleVoicePromotionRebranch",
    "ConsoleVoicePromotionRecoveryKind",
    "ResolvedVoicePromotionDestination",
    "VoicePromotionContext",
    "VoicePromotionClaim",
    "VoicePromotionClaimStatus",
    "VoicePromotionIdentitySet",
    "VoicePromotionOutcome",
    "VoicePromotionOutcomeStatus",
    "VoicePromotionOwner",
    "VoicePromotionOwnerStatus",
    "VoicePromotionQuitPermit",
    "VoicePromotionQuitToken",
    "VoicePromotionRecoveryHandle",
    "VoicePromotionSessionCloseToken",
    "VoiceWinningPromotion",
    "derive_voice_promotion_identities",
    "new_voice_promotion_id",
]
