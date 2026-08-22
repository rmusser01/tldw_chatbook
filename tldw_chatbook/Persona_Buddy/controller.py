"""App-owned Persona Buddy state, resolution, and async lifetime."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from pathlib import Path
from threading import RLock
from typing import Generic, Literal, TypeVar

from tldw_chatbook.Persona_Visual.repository import (
    PersonaVisualGraph,
    PersonaVisualIdentity,
    PersonaVisualRepository,
)
from tldw_chatbook.Persona_Visual.runtime import (
    PersonaVisualCacheIdentity,
    PersonaVisualPortrait,
    PersonaVisualResolution,
    resolve_active_persona_visual,
)

from .preferences import (
    PersonaBuddyPreferences,
    PersonaBuddySelection,
    persist_persona_buddy_preferences,
)
from .rendering import (
    MAX_PERSONA_BUDDY_PREPARED_CELLS,
    MAX_PERSONA_BUDDY_PREPARED_FRAMES,
    PERSONA_BUDDY_FRAME_UNAVAILABLE,
    PersonaBuddyFrameError,
    PersonaBuddyPreparedFrame,
    prepare_persona_buddy_frame,
    prepare_persona_buddy_portrait,
)


_BUILTIN_STATES = frozenset(
    {
        "idle",
        "listening",
        "thinking",
        "speaking",
        "approval_needed",
        "tool_running",
        "wake_armed",
        "offline",
        "error",
    }
)
_CUSTOM_STATE_PATTERN = re.compile(r"[a-z][a-z0-9_.:-]{0,95}\Z")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_UNSAFE_STATE_PREFIXES = (
    "env:",
    "file:",
    "ftp:",
    "http:",
    "https:",
    "proc:",
    "ssh:",
)
_UNSAFE_STATE_MARKERS = (
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
)
_LIVE_STATES = frozenset({"idle", "listening", "thinking", "speaking"})
_TIMED_SOURCES = frozenset({"explicit", "timed"})
_CUSTOM_STATE_SOURCES = frozenset({"authored", *_TIMED_SOURCES})
_MAX_TTL_SECONDS = 3600.0

_T = TypeVar("_T")

_LeaseKind = Literal[
    "error",
    "approval",
    "timed",
    "authored",
    "tool",
    "wake",
    "voice",
    "offline",
    "idle",
]


class PersonaBuddyStateError(ValueError):
    """A fixed-category error for untrusted controller state input."""

    __slots__ = ("category",)

    def __init__(self, category: str = "persona_buddy_state_invalid") -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class BuddyDrainResult(Generic[_T]):
    """Settlement of one retained child, including delayed outer cancellation."""

    completed: bool
    value: _T | None = None
    error_category: str | None = None
    cancellation: BaseException | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class PersonaBuddyLeaseToken:
    """Exact capability for one source-and-owner lease revision."""

    source: str
    owner: str
    state: str
    lease_id: int
    expires_at: float | None = None


@dataclass(frozen=True, slots=True)
class PersonaBuddySnapshot:
    """Immutable controller state safe for app-owned consumers."""

    generation: int
    selection: PersonaBuddySelection | None
    state: str
    state_source: str | None = None
    state_owner: str | None = None
    enabled: bool = False
    open: bool = True
    collapsed: bool = False
    preferences_generation: int = 0
    profile_generation: int = 0
    viewport_generation: int = 0
    visual: PersonaBuddyVisualSnapshot | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class PersonaBuddyVisualSnapshot:
    """Path-free resolved visual state consumed by disposable views."""

    available: bool
    reason: str | None
    source: str
    persona_id: str | None
    persona_revision: int | None
    requested_state: str
    resolved_state: str | None
    animation_id: str | None
    graph_identity: PersonaVisualIdentity | None
    cache_identity: PersonaVisualCacheIdentity | None
    frames: tuple[PersonaBuddyPreparedFrame, ...] = field(repr=False)
    frame_rate: float | None = None
    loop: bool = False
    animate: bool = False


@dataclass(frozen=True, slots=True)
class _ResolutionTicket:
    selection: PersonaBuddySelection
    requested_state: str
    controller_generation: int
    preferences_generation: int
    profile_generation: int
    viewport_generation: int
    enabled: bool
    open: bool
    collapsed: bool
    reduced_motion: bool


@dataclass(frozen=True, slots=True)
class _LocalPersona:
    persona_id: str
    revision: int
    portrait: PersonaVisualPortrait | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _StateLease:
    token: PersonaBuddyLeaseToken
    kind: _LeaseKind


def _require_identifier(value: object) -> str:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise PersonaBuddyStateError()
    return value


def _require_state(value: object) -> str:
    if type(value) is not str or _CUSTOM_STATE_PATTERN.fullmatch(value) is None:
        raise PersonaBuddyStateError()
    if value in _BUILTIN_STATES:
        return value
    if value.startswith(_UNSAFE_STATE_PREFIXES):
        raise PersonaBuddyStateError()
    compact = re.sub(r"[._:-]+", "_", value)
    if any(marker in compact for marker in _UNSAFE_STATE_MARKERS):
        raise PersonaBuddyStateError()
    return value


def _require_expiration(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PersonaBuddyStateError()
    expiration = float(value)
    if not math.isfinite(expiration):
        raise PersonaBuddyStateError()
    return expiration


def _lease_kind(source: str, state: str) -> _LeaseKind:
    if state == "error":
        return "error"
    if state == "approval_needed":
        return "approval"
    if source in _TIMED_SOURCES:
        return "timed"
    if source == "authored":
        return "authored"
    if state == "tool_running":
        return "tool"
    if state == "wake_armed":
        return "wake"
    if state in _LIVE_STATES:
        return "voice" if state != "idle" else "idle"
    if state == "offline":
        return "offline"
    return "authored"


def _validate_state_authority(
    *,
    source: str,
    state: str,
    expires_at: float | None,
    now: float,
) -> None:
    if source in _TIMED_SOURCES and (expires_at is None or expires_at <= now):
        raise PersonaBuddyStateError()
    if state not in _BUILTIN_STATES and source not in _CUSTOM_STATE_SOURCES:
        raise PersonaBuddyStateError()


def load_local_persona_portrait(
    local_persona_service: object,
    persona_record: Mapping[str, object],
) -> PersonaVisualPortrait | None:
    """Load the linked local Character-card BLOB as a path-free portrait."""

    try:
        character_id = persona_record.get("character_card_id")
        getter = getattr(local_persona_service, "get_character", None)
        if type(character_id) is not int or character_id < 1 or not callable(getter):
            return None
        character = getter(character_id)
        if not isinstance(character, Mapping):
            return None
        image = character.get("image")
        revision = character.get("version")
        if (
            character.get("id") != character_id
            or type(revision) is not int
            or revision < 0
            or type(image) is not bytes
            or not image
        ):
            return None
        mime_type = _portrait_mime_type(image)
        if mime_type is None:
            return None
        return PersonaVisualPortrait(
            portrait_id=f"local-character:{character_id}",
            revision=revision,
            mime_type=mime_type,
            sha256=hashlib.sha256(image).hexdigest(),
            data=image,
        )
    except Exception:
        return None


def _portrait_mime_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class PersonaBuddyController:
    """Own one explicit selection and source-scoped operational-state leases."""

    def __init__(
        self,
        *,
        selection: PersonaBuddySelection | None = None,
        preferences: PersonaBuddyPreferences | None = None,
        clock: Callable[[], float] = time.monotonic,
        local_persona_service: object | None = None,
        profile_db: object | None = None,
        profile_root: str | Path | None = None,
        reduced_motion: bool | Callable[[], bool] = False,
        repository_factory: Callable[[object], PersonaVisualRepository] = (
            PersonaVisualRepository
        ),
        portrait_loader: (
            Callable[[Mapping[str, object]], PersonaVisualPortrait | None] | None
        ) = None,
        preference_writer: Callable[[PersonaBuddyPreferences], bool] = (
            persist_persona_buddy_preferences
        ),
        scheduler: Callable[[Callable[[], None]], object] | None = None,
        phase_barrier: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> None:
        if selection is not None and type(selection) is not PersonaBuddySelection:
            raise PersonaBuddyStateError()
        if preferences is not None and type(preferences) is not PersonaBuddyPreferences:
            raise PersonaBuddyStateError()
        if preferences is not None and selection is not None:
            raise PersonaBuddyStateError()
        if type(reduced_motion) is not bool and not callable(reduced_motion):
            raise PersonaBuddyStateError()
        self._lock = RLock()
        self._clock = clock
        self._preferences = preferences or PersonaBuddyPreferences(selection=selection)
        self._selection = self._preferences.selection
        self._leases: dict[tuple[str, str], _StateLease] = {}
        self._generation = 0
        self._preferences_generation = 0
        self._profile_generation = 0
        self._viewport_generation = 0
        self._next_lease_id = 0
        self._visual: PersonaBuddyVisualSnapshot | None = None
        self._local_persona_service = local_persona_service
        self._profile_db = profile_db
        self._profile_root = (
            Path(profile_root).resolve() if profile_root is not None else None
        )
        self._reduced_motion = reduced_motion
        self._repository_factory = repository_factory
        self._portrait_loader = portrait_loader
        self._preference_writer = preference_writer
        self._scheduler = scheduler
        self._phase_barrier = phase_barrier
        self._operation_lock = asyncio.Lock()
        self._owned_tasks: set[asyncio.Task[object]] = set()
        self._shutdown_requested = False
        self._shutdown_task: asyncio.Task[None] | None = None

    def snapshot(self) -> PersonaBuddySnapshot:
        """Return the current immutable selection and resolved state.

        Returns:
            The current content-free Buddy state snapshot.
        """

        with self._lock:
            self._discard_expired_locked()
            lease = self._resolve_locked()
            if lease is None:
                return PersonaBuddySnapshot(
                    generation=self._generation,
                    selection=self._selection,
                    state="idle",
                    enabled=self._preferences.enabled,
                    open=self._preferences.open,
                    collapsed=self._preferences.collapsed,
                    preferences_generation=self._preferences_generation,
                    profile_generation=self._profile_generation,
                    viewport_generation=self._viewport_generation,
                    visual=self._visual,
                )
            token = lease.token
            return PersonaBuddySnapshot(
                generation=self._generation,
                selection=self._selection,
                state=token.state,
                state_source=token.source,
                state_owner=token.owner,
                enabled=self._preferences.enabled,
                open=self._preferences.open,
                collapsed=self._preferences.collapsed,
                preferences_generation=self._preferences_generation,
                profile_generation=self._profile_generation,
                viewport_generation=self._viewport_generation,
                visual=self._visual,
            )

    def current_preferences(self) -> PersonaBuddyPreferences:
        """Return the exact immutable preference snapshot under the state lock."""

        with self._lock:
            return self._preferences

    def apply_preferences_patch(self, **changes: object) -> int:
        """Apply selected immutable fields immediately and return their revision.

        Persistence is deliberately separate: interactive views can publish the
        user's latest intent synchronously, then let an app-owned worker drain the
        blocking writer without carrying a stale whole-preference snapshot.
        """

        allowed = {"enabled", "selection", "open", "collapsed", "geometry"}
        if not changes or not set(changes) <= allowed:
            raise PersonaBuddyStateError()
        with self._lock:
            candidate = replace(self._preferences, **changes)
            self._apply_preferences_locked(candidate)
            return self._preferences_generation

    async def persist_preferences_revision(self, revision: int) -> bool:
        """Persist current merged preferences once for one applied revision.

        A failed write leaves the immediate in-memory intent authoritative; it is
        not rolled back over newer interaction fields. A later revision writes the
        then-current merged snapshot and converges durable state.
        """

        if type(revision) is not int or revision < 0:
            raise PersonaBuddyStateError()
        if self._shutdown_requested:
            raise RuntimeError("persona_buddy_shutdown")

        async def serialized() -> bool:
            async with self._operation_lock:
                with self._lock:
                    if revision > self._preferences_generation:
                        raise PersonaBuddyStateError()
                    candidate = self._preferences
                write = await self._drain_owned(
                    asyncio.to_thread(self._preference_writer, candidate),
                    name="preferences:patch-write",
                )
                return write.completed and write.value is True

        outcome = await self._drain_owned(
            serialized(), name=f"preferences:revision:{revision}"
        )
        if outcome.cancellation is not None:
            raise outcome.cancellation
        return outcome.completed and outcome.value is True

    def select_local_persona(self, persona_id: str) -> int:
        """Explicitly replace the selected local Persona and return generation."""

        selection = PersonaBuddySelection("local", persona_id)
        with self._lock:
            if selection != self._selection:
                self._selection = selection
                self._preferences = replace(self._preferences, selection=selection)
                self._preferences_generation += 1
                self._generation += 1
            return self._generation

    def invalidate_profile(self) -> int:
        """Advance local Persona/graph authority without clearing preferences."""

        with self._lock:
            self._profile_generation += 1
            self._generation += 1
            return self._profile_generation

    def set_viewport_generation(self, generation: int) -> int:
        """Record a current viewport generation for late-frame fencing."""

        if type(generation) is not int or generation < 0:
            raise PersonaBuddyStateError()
        with self._lock:
            if generation != self._viewport_generation:
                self._viewport_generation = generation
                self._generation += 1
            return self._viewport_generation

    def observe_persona(self, *, source: str, persona_id: object) -> int:
        """Observe UI/runtime selection without changing Buddy authority."""

        _require_identifier(source)
        del persona_id
        with self._lock:
            return self._generation

    def acquire_state(
        self,
        *,
        source: str,
        owner: str,
        state: str,
        expires_at: float | None = None,
    ) -> PersonaBuddyLeaseToken:
        """Acquire or replace one exact source-and-owner state lease."""

        normalized_source = _require_identifier(source)
        normalized_owner = _require_identifier(owner)
        normalized_state = _require_state(state)
        normalized_expiration = _require_expiration(expires_at)
        with self._lock:
            self._discard_expired_locked()
            _validate_state_authority(
                source=normalized_source,
                state=normalized_state,
                expires_at=normalized_expiration,
                now=self._clock(),
            )
            self._next_lease_id += 1
            token = PersonaBuddyLeaseToken(
                source=normalized_source,
                owner=normalized_owner,
                state=normalized_state,
                lease_id=self._next_lease_id,
                expires_at=normalized_expiration,
            )
            self._leases[(normalized_source, normalized_owner)] = _StateLease(
                token=token,
                kind=_lease_kind(normalized_source, normalized_state),
            )
            self._generation += 1
            return token

    def release_state(
        self,
        *,
        token: PersonaBuddyLeaseToken | None = None,
        source: str | None = None,
        owner: str | None = None,
    ) -> bool:
        """Release only an exact token or an exact source-and-owner pair."""

        with self._lock:
            self._discard_expired_locked()
            if token is not None:
                if type(token) is not PersonaBuddyLeaseToken:
                    return False
                key = (token.source, token.owner)
                current = self._leases.get(key)
                if current is None or current.token != token:
                    return False
            else:
                if source is None or owner is None:
                    return False
                try:
                    key = (_require_identifier(source), _require_identifier(owner))
                except PersonaBuddyStateError:
                    return False
                current = self._leases.get(key)
                if current is None:
                    return False
            del self._leases[key]
            self._generation += 1
            return True

    def set_timed_state(
        self,
        *,
        owner: str,
        state: str,
        ttl_seconds: float,
    ) -> PersonaBuddyLeaseToken:
        """Acquire a bounded explicit/custom state lease from the monotonic clock."""

        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, Real):
            raise PersonaBuddyStateError()
        ttl = float(ttl_seconds)
        if not math.isfinite(ttl) or not 0 < ttl <= _MAX_TTL_SECONDS:
            raise PersonaBuddyStateError()
        return self.acquire_state(
            source="timed",
            owner=owner,
            state=state,
            expires_at=self._clock() + ttl,
        )

    def set_authored_trigger(
        self,
        *,
        owner: str,
        state: str,
    ) -> PersonaBuddyLeaseToken:
        """Acquire or replace one authored Persona Visual trigger."""

        return self.acquire_state(source="authored", owner=owner, state=state)

    async def _drain_owned(
        self,
        awaitable: Awaitable[_T],
        *,
        name: str,
    ) -> BuddyDrainResult[_T]:
        """Shield and settle one registered child despite repeated cancellation."""

        async def runner() -> _T:
            return await awaitable

        task = asyncio.create_task(runner(), name=f"persona-buddy:{name}")
        self._owned_tasks.add(task)  # registered before this method's first await
        owner = asyncio.current_task()
        seen_cancellations = owner.cancelling() if owner is not None else 0
        outer_cancellation: asyncio.CancelledError | None = None
        try:
            while True:
                try:
                    value = await asyncio.shield(task)
                except asyncio.CancelledError as error:
                    cancellation_count = owner.cancelling() if owner is not None else 0
                    if cancellation_count > seen_cancellations:
                        outer_cancellation = outer_cancellation or error
                        seen_cancellations = cancellation_count
                        if not task.done():
                            continue
                    if task.cancelled():
                        return BuddyDrainResult(
                            completed=False,
                            error_category="persona_buddy_operation_cancelled",
                            cancellation=outer_cancellation,
                        )
                    if not task.done():
                        outer_cancellation = outer_cancellation or error
                        continue
                    try:
                        value = task.result()
                    except asyncio.CancelledError:
                        return BuddyDrainResult(
                            completed=False,
                            error_category="persona_buddy_operation_cancelled",
                            cancellation=outer_cancellation,
                        )
                    except Exception:
                        return BuddyDrainResult(
                            completed=False,
                            error_category="persona_buddy_operation_failed",
                            cancellation=outer_cancellation,
                        )
                except Exception:
                    return BuddyDrainResult(
                        completed=False,
                        error_category="persona_buddy_operation_failed",
                        cancellation=outer_cancellation,
                    )
                return BuddyDrainResult(
                    completed=True,
                    value=value,
                    cancellation=outer_cancellation,
                )
        finally:
            if task.done():
                self._owned_tasks.discard(task)

    async def run_serialized(
        self,
        operation: Callable[[], _T],
        *,
        name: str,
    ) -> _T:
        """Run one blocking same-owner operation off-loop and drain cancellation."""

        if self._shutdown_requested:
            raise RuntimeError("persona_buddy_shutdown")

        async def serialized() -> _T:
            async with self._operation_lock:
                outcome = await self._drain_owned(
                    asyncio.to_thread(operation), name=f"{name}:thread"
                )
                if not outcome.completed:
                    raise RuntimeError(outcome.error_category)
                return outcome.value  # type: ignore[return-value]

        outcome = await self._drain_owned(serialized(), name=name)
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if not outcome.completed:
            raise RuntimeError(outcome.error_category)
        return outcome.value  # type: ignore[return-value]

    async def update_preferences(
        self, preferences: PersonaBuddyPreferences
    ) -> PersonaBuddySnapshot:
        """Persist then apply one exact preference snapshot under owner serialization."""

        if type(preferences) is not PersonaBuddyPreferences:
            raise PersonaBuddyStateError()
        if self._shutdown_requested:
            raise RuntimeError("persona_buddy_shutdown")

        async def serialized() -> PersonaBuddySnapshot:
            async with self._operation_lock:
                expected_authority = self._preference_authority()
                candidate = preferences
                last_successful: PersonaBuddyPreferences | None = None
                while True:
                    written = await self._drain_owned(
                        asyncio.to_thread(self._preference_writer, candidate),
                        name="preferences:write",
                    )
                    if not written.completed or written.value is not True:
                        if last_successful is not None:
                            with self._lock:
                                self._apply_preferences_locked(last_successful)
                        return self.snapshot()
                    last_successful = candidate
                    with self._lock:
                        if expected_authority == self._preference_authority():
                            self._apply_preferences_locked(candidate)
                            return self.snapshot()
                        candidate = self._preferences
                        expected_authority = self._preference_authority()

        outcome = await self._drain_owned(serialized(), name="preferences")
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if not outcome.completed or outcome.value is None:
            return self.snapshot()
        return outcome.value

    async def resolve_current_visual(
        self,
        *,
        cols: int,
        lines: int,
    ) -> PersonaBuddyVisualSnapshot:
        """Resolve and prepare the selected local Persona's current state."""

        if self._shutdown_requested:
            return self._unavailable_snapshot(
                "persona_buddy_shutdown", requested_state="idle"
            )

        async def serialized() -> PersonaBuddyVisualSnapshot:
            async with self._operation_lock:
                ticket = self._resolution_ticket()
                if ticket is None:
                    return self._apply_unavailable(
                        "persona_buddy_persona_unavailable", requested_state="idle"
                    )

                persona_outcome = await self._drain_owned(
                    asyncio.to_thread(self._read_local_persona, ticket.selection),
                    name="resolve:persona-read",
                )
                if not self._is_current(ticket):
                    return self._stale_snapshot(ticket)
                if not persona_outcome.completed or persona_outcome.value is None:
                    return self._apply_unavailable(
                        "persona_buddy_persona_unavailable",
                        requested_state=ticket.requested_state,
                        ticket=ticket,
                    )
                persona = persona_outcome.value
                if not await self._after_phase("persona_read", ticket):
                    return self._stale_snapshot(ticket)

                graph_outcome = await self._drain_owned(
                    asyncio.to_thread(self._read_graph, persona.persona_id),
                    name="resolve:graph-read",
                )
                if not self._is_current(ticket):
                    return self._stale_snapshot(ticket)
                graph = graph_outcome.value if graph_outcome.completed else None
                if (
                    type(graph) is not PersonaVisualGraph
                    or graph.identity.persona_id != persona.persona_id
                    or graph.identity.persona_revision != persona.revision
                ):
                    return self._apply_unavailable(
                        "persona_buddy_binding_unavailable",
                        requested_state=ticket.requested_state,
                        ticket=ticket,
                        persona=persona,
                    )
                graph_identity = graph.identity
                if not await self._after_phase("graph_read", ticket):
                    return self._stale_snapshot(ticket)

                resolution_outcome = await self._drain_owned(
                    asyncio.to_thread(
                        self._resolve_runtime,
                        persona,
                        ticket.requested_state,
                        ticket.reduced_motion,
                    ),
                    name="resolve:runtime",
                )
                if not self._is_current(ticket):
                    return self._stale_snapshot(ticket)
                resolution = (
                    resolution_outcome.value if resolution_outcome.completed else None
                )
                if (
                    type(resolution) is not PersonaVisualResolution
                    or resolution.cache_identity.graph != graph_identity
                    or resolution.requested_state != ticket.requested_state
                    or resolution.cache_identity.requested_state
                    != ticket.requested_state
                    or resolution.cache_identity.reduced_motion != ticket.reduced_motion
                ):
                    return self._preserve_or_unavailable(
                        ticket=ticket,
                        persona=persona,
                        reason=PERSONA_BUDDY_FRAME_UNAVAILABLE,
                    )
                cache_identity = resolution.cache_identity
                if not await self._after_phase("runtime_resolve", ticket):
                    return self._stale_snapshot(ticket)

                prepared_outcome = await self._drain_owned(
                    asyncio.to_thread(
                        self._prepare_resolution,
                        resolution,
                        cols,
                        lines,
                    ),
                    name="resolve:frame-prepare",
                )
                if not self._is_current(ticket):
                    return self._stale_snapshot(ticket)
                if not await self._after_phase("frame_prepare", ticket):
                    return self._stale_snapshot(ticket)
                frames = prepared_outcome.value if prepared_outcome.completed else None
                if not self._prepared_matches(resolution, frames):
                    return self._preserve_or_unavailable(
                        ticket=ticket,
                        persona=persona,
                        reason=PERSONA_BUDDY_FRAME_UNAVAILABLE,
                    )
                if not frames:
                    return self._preserve_or_unavailable(
                        ticket=ticket,
                        persona=persona,
                        reason=PERSONA_BUDDY_FRAME_UNAVAILABLE,
                    )

                visual = PersonaBuddyVisualSnapshot(
                    available=True,
                    reason=resolution.reason,
                    source=resolution.source,
                    persona_id=persona.persona_id,
                    persona_revision=persona.revision,
                    requested_state=resolution.requested_state,
                    resolved_state=resolution.resolved_state,
                    animation_id=resolution.animation_id,
                    graph_identity=graph_identity,
                    cache_identity=cache_identity,
                    frames=frames,
                    frame_rate=resolution.frame_rate,
                    loop=resolution.loop,
                    animate=resolution.animate,
                )
                with self._lock:
                    if not self._is_current(ticket):
                        return self._stale_snapshot(ticket)
                    self._visual = visual
                return visual

        outcome = await self._drain_owned(serialized(), name="resolve")
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if not outcome.completed or outcome.value is None:
            return self._unavailable_snapshot(
                "persona_buddy_operation_failed", requested_state="idle"
            )
        return outcome.value

    async def shutdown(self) -> None:
        """Stop admission and drain every registered operation exactly once."""

        self._shutdown_requested = True
        with self._lock:
            self._generation += 1
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(
                self._shutdown_runner(), name="shutdown_persona_buddy_controller"
            )
            self._shutdown_task = task
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError as error:
                if task.done():
                    if task.cancelled():
                        raise
                    task.result()
                    cancellation = cancellation or error
                    break
                cancellation = cancellation or error
                continue
        if cancellation is not None:
            raise cancellation

    async def _shutdown_runner(self) -> None:
        current = asyncio.current_task()
        while True:
            tasks = tuple(
                task
                for task in self._owned_tasks
                if task is not current and not task.done()
            )
            if not tasks:
                break
            for task in tasks:
                try:
                    await asyncio.shield(task)
                except BaseException:
                    pass
        async with self._operation_lock:
            return

    def _resolution_ticket(self) -> _ResolutionTicket | None:
        reduced_motion = self._current_reduced_motion()
        with self._lock:
            self._discard_expired_locked()
            selection = self._selection
            if (
                selection is None
                or not self._preferences.enabled
                or not self._preferences.open
            ):
                return None
            lease = self._resolve_locked()
            return _ResolutionTicket(
                selection=selection,
                requested_state=lease.token.state if lease else "idle",
                controller_generation=self._generation,
                preferences_generation=self._preferences_generation,
                profile_generation=self._profile_generation,
                viewport_generation=self._viewport_generation,
                enabled=self._preferences.enabled,
                open=self._preferences.open,
                collapsed=self._preferences.collapsed,
                reduced_motion=reduced_motion,
            )

    def _is_current(self, ticket: _ResolutionTicket) -> bool:
        reduced_motion = self._current_reduced_motion()
        with self._lock:
            self._discard_expired_locked()
            lease = self._resolve_locked()
            return (
                not self._shutdown_requested
                and self._selection == ticket.selection
                and self._generation == ticket.controller_generation
                and self._preferences_generation == ticket.preferences_generation
                and self._profile_generation == ticket.profile_generation
                and self._viewport_generation == ticket.viewport_generation
                and self._preferences.enabled == ticket.enabled
                and self._preferences.open == ticket.open
                and self._preferences.collapsed == ticket.collapsed
                and (lease.token.state if lease else "idle") == ticket.requested_state
                and reduced_motion == ticket.reduced_motion
            )

    def _current_reduced_motion(self) -> bool:
        try:
            value = (
                self._reduced_motion()
                if callable(self._reduced_motion)
                else self._reduced_motion
            )
        except Exception:
            return True
        return value if type(value) is bool else True

    def _preference_authority(
        self,
    ) -> tuple[int, PersonaBuddyPreferences]:
        """Snapshot every controller authority relevant to a preference commit."""

        with self._lock:
            return self._preferences_generation, self._preferences

    def _apply_preferences_locked(self, preferences: PersonaBuddyPreferences) -> None:
        if preferences == self._preferences:
            return
        self._preferences = preferences
        self._selection = preferences.selection
        self._preferences_generation += 1
        self._generation += 1

    async def _after_phase(self, phase: str, ticket: _ResolutionTicket) -> bool:
        barrier = self._phase_barrier
        if barrier is not None:
            pending = barrier(phase)
            if inspect.isawaitable(pending):
                await pending
            if not self._is_current(ticket):
                return False
        return self._is_current(ticket)

    def _read_local_persona(
        self, selection: PersonaBuddySelection
    ) -> _LocalPersona | None:
        service = self._local_persona_service
        getter = getattr(service, "get_persona_profile", None)
        if not callable(getter):
            return None
        try:
            record = getter(selection.local_persona_id)
            if not isinstance(record, Mapping):
                return None
            persona_id = record.get("id")
            revision = record.get("version")
            if (
                type(persona_id) is not str
                or persona_id != selection.local_persona_id
                or type(revision) is not int
                or revision < 1
                or record.get("deleted", False) is not False
                or record.get("is_active", True) is not True
            ):
                return None
            portrait = self._portrait_loader(record) if self._portrait_loader else None
            if portrait is not None and type(portrait) is not PersonaVisualPortrait:
                portrait = None
            return _LocalPersona(persona_id, revision, portrait)
        except Exception:
            return None

    def _read_graph(self, persona_id: str) -> PersonaVisualGraph | None:
        if self._profile_db is None:
            return None
        try:
            return self._repository_factory(self._profile_db).get_active_persona_pack(
                persona_id
            )
        except Exception:
            return None

    def _resolve_runtime(
        self,
        persona: _LocalPersona,
        requested_state: str,
        reduced_motion: bool,
    ) -> PersonaVisualResolution | None:
        if self._profile_db is None or self._profile_root is None:
            return None
        try:
            return resolve_active_persona_visual(
                self._repository_factory(self._profile_db),
                persona.persona_id,
                self._profile_root,
                requested_state,
                portrait=persona.portrait,
                reduced_motion=reduced_motion,
            )
        except Exception:
            return None

    @staticmethod
    def _prepare_resolution(
        resolution: PersonaVisualResolution,
        cols: int,
        lines: int,
    ) -> tuple[PersonaBuddyPreparedFrame, ...] | None:
        try:
            if resolution.frames:
                if len(resolution.frames) > MAX_PERSONA_BUDDY_PREPARED_FRAMES:
                    return None
                prepared: list[PersonaBuddyPreparedFrame] = []
                prepared_cells = 0
                for frame in resolution.frames:
                    remaining_cells = MAX_PERSONA_BUDDY_PREPARED_CELLS - prepared_cells
                    if remaining_cells < 1:
                        return None
                    painted = prepare_persona_buddy_frame(
                        frame,
                        resolution_cache_identity=resolution.cache_identity,
                        cols=cols,
                        lines=lines,
                        max_cells=remaining_cells,
                    )
                    prepared_cells += painted.width * painted.height
                    prepared.append(painted)
                return tuple(prepared)
            if resolution.portrait is not None:
                return (
                    prepare_persona_buddy_portrait(
                        resolution.portrait,
                        resolution_cache_identity=resolution.cache_identity,
                        cols=cols,
                        lines=lines,
                    ),
                )
            return ()
        except PersonaBuddyFrameError:
            return None

    @staticmethod
    def _prepared_matches(
        resolution: PersonaVisualResolution,
        frames: tuple[PersonaBuddyPreparedFrame, ...] | None,
    ) -> bool:
        if frames is None:
            return False
        if resolution.frames:
            expected = tuple(
                (
                    frame.asset_id,
                    frame.asset_key,
                    frame.sha256,
                    frame.manifest_frame_index,
                    frame.selected_frame,
                )
                for frame in resolution.frames
            )
            actual = tuple(
                (
                    frame.asset_id,
                    frame.asset_key,
                    frame.asset_sha256,
                    frame.manifest_frame_index,
                    frame.selected_frame,
                )
                for frame in frames
            )
            return expected == actual and all(
                frame.cache_identity == resolution.cache_identity for frame in frames
            )
        if resolution.portrait is not None:
            return (
                len(frames) == 1
                and frames[0].asset_key == resolution.portrait.portrait_id
                and frames[0].asset_sha256 == resolution.portrait.sha256
                and frames[0].selected_frame == resolution.portrait.selected_frame
                and frames[0].cache_identity == resolution.cache_identity
            )
        return frames == ()

    def _preserve_or_unavailable(
        self,
        *,
        ticket: _ResolutionTicket,
        persona: _LocalPersona,
        reason: str,
    ) -> PersonaBuddyVisualSnapshot:
        if not self._is_current(ticket):
            return self._stale_snapshot(ticket)
        with self._lock:
            if not self._is_current(ticket):
                return self._stale_snapshot(ticket)
            previous = self._visual
            if (
                previous is not None
                and previous.available
                and previous.persona_id == persona.persona_id
                and previous.frames
            ):
                retained = replace(previous, reason=reason)
                self._visual = retained
                return retained
        return self._apply_unavailable(
            reason,
            requested_state=ticket.requested_state,
            ticket=ticket,
            persona=persona,
        )

    def _apply_unavailable(
        self,
        reason: str,
        *,
        requested_state: str,
        ticket: _ResolutionTicket | None = None,
        persona: _LocalPersona | None = None,
    ) -> PersonaBuddyVisualSnapshot:
        if ticket is not None and not self._is_current(ticket):
            return self._stale_snapshot(ticket)
        visual = self._unavailable_snapshot(
            reason,
            requested_state=requested_state,
            persona=persona,
        )
        with self._lock:
            if ticket is not None and not self._is_current(ticket):
                return self._stale_snapshot(ticket)
            self._visual = visual
        return visual

    @staticmethod
    def _unavailable_snapshot(
        reason: str,
        *,
        requested_state: str,
        persona: _LocalPersona | None = None,
    ) -> PersonaBuddyVisualSnapshot:
        return PersonaBuddyVisualSnapshot(
            available=False,
            reason=reason,
            source="unavailable",
            persona_id=persona.persona_id if persona else None,
            persona_revision=persona.revision if persona else None,
            requested_state=requested_state,
            resolved_state=None,
            animation_id=None,
            graph_identity=None,
            cache_identity=None,
            frames=(),
        )

    @staticmethod
    def _stale_snapshot(ticket: _ResolutionTicket) -> PersonaBuddyVisualSnapshot:
        return PersonaBuddyVisualSnapshot(
            available=False,
            reason="persona_buddy_resolution_stale",
            source="unavailable",
            persona_id=ticket.selection.local_persona_id,
            persona_revision=None,
            requested_state=ticket.requested_state,
            resolved_state=None,
            animation_id=None,
            graph_identity=None,
            cache_identity=None,
            frames=(),
        )

    def _discard_expired_locked(self) -> None:
        now = self._clock()
        expired = [
            key
            for key, lease in self._leases.items()
            if lease.token.expires_at is not None and lease.token.expires_at <= now
        ]
        if not expired:
            return
        for key in expired:
            del self._leases[key]
        self._generation += 1

    def _resolve_locked(self) -> _StateLease | None:
        leases = tuple(self._leases.values())
        for kind in ("error", "approval", "timed", "authored", "tool"):
            if match := _newest(leases, kind):
                return match

        non_idle_voice = tuple(
            lease
            for lease in leases
            if lease.kind == "voice" and lease.token.state != "idle"
        )
        if non_idle_voice:
            return max(non_idle_voice, key=lambda lease: lease.token.lease_id)
        if wake := _newest(leases, "wake"):
            return wake
        for kind in ("voice", "idle", "offline"):
            if match := _newest(leases, kind):
                return match
        return None


def _newest(leases: tuple[_StateLease, ...], kind: str) -> _StateLease | None:
    candidates = (lease for lease in leases if lease.kind == kind)
    return max(candidates, key=lambda lease: lease.token.lease_id, default=None)
