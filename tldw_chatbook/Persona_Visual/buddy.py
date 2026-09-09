"""App-owned, profile-local Persona Buddy state with no UI object ownership."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .assets import _open_profile_root
from .contracts import ALLOWED_TRIGGER_SOURCES, RESERVED_STATES, PersonaVisualTrigger
from .repository import (
    PersonaVisualGraph,
    PersonaVisualIdentity,
    PersonaVisualRepository,
)
from .runtime import (
    PersonaVisualPortrait,
    PersonaVisualResolution,
    resolve_active_persona_visual,
)

UNAVAILABLE_REASON = "persona_buddy_unavailable"
PREFERENCES_REASON = "persona_buddy_preferences_invalid"
_STATE = re.compile(r"[a-z][a-z0-9_.:-]{0,95}\Z")
_LIVE = {"idle", "listening", "thinking", "speaking"}


@dataclass(frozen=True, slots=True)
class BuddyPreferences:
    """Private UI preferences, deliberately separate from portable actor data."""

    source: str = "local"
    local_persona_id: str | None = None
    enabled: bool = False
    open: bool = True
    collapsed: bool = False
    x: int | None = None
    y: int | None = None
    width: int = 28
    height: int = 16


@dataclass(frozen=True, slots=True)
class BuddySnapshot:
    """One path-free result that the current view may apply after fencing itself."""

    persona_id: str
    name: str
    generation: int
    identity: PersonaVisualIdentity
    requested_state: str
    resolution: PersonaVisualResolution
    prepared: Any = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _Lease:
    state: str
    expires: float | None
    order: int
    category: str
    priority: int = 0


@dataclass(frozen=True, slots=True)
class _Authority:
    revision: int
    name: str
    graph: PersonaVisualGraph


class BuddyController:
    """Serialize local authority, decode, and private preference work across views.

    Signal methods must be called on the owning event loop and accept trusted app
    lifecycle events only. Render callbacks run on a worker and must return private
    immutable prepared data; they must never access a widget or mutate this owner.
    """

    def __init__(
        self,
        local_service: Any,
        repository: PersonaVisualRepository,
        profile_root: str | Path,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._service = local_service
        self._repository = repository
        self._profile_root = Path(profile_root).absolute()
        self._clock = clock
        self._preferences = BuddyPreferences()
        self._generation = 0
        self._reason: str | None = None
        self._lock = asyncio.Lock()
        self._loaded = False
        self._closed = False
        self._authority: _Authority | None = None
        self._leases: dict[str, _Lease] = {}
        self._trigger_leases: dict[str, _Lease] = {}
        self._order = 0
        self._triggers: tuple[PersonaVisualTrigger, ...] = ()

    @property
    def preferences(self) -> BuddyPreferences:
        return self._preferences

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def requested_state(self) -> str:
        """Resolve pinned precedence; one owner's release cannot erase another."""
        before = self._resolve_state()
        now = self._clock()
        for leases in (self._leases, self._trigger_leases):
            expired = [
                key
                for key, lease in leases.items()
                if lease.expires is not None and lease.expires <= now
            ]
            for key in expired:
                del leases[key]
        state = self._resolve_state()
        if state != before:
            self._generation += 1
        return state

    def _resolve_state(self) -> str:
        leases = (*self._leases.values(), *self._trigger_leases.values())
        for state in ("error", "approval_needed"):
            if any(lease.state == state for lease in leases):
                return state
        for category in ("explicit", "trigger"):
            candidates = [lease for lease in leases if lease.category == category]
            if candidates:
                return max(
                    candidates, key=lambda item: (item.priority, item.order)
                ).state
        if any(lease.state == "tool_running" for lease in leases):
            return "tool_running"
        live = [lease for lease in leases if lease.state in _LIVE]
        live_state = max(live, key=lambda item: item.order).state if live else "idle"
        if live_state == "idle" and any(
            lease.state == "wake_armed" for lease in leases
        ):
            return "wake_armed"
        if live:
            return live_state
        if any(lease.state == "offline" for lease in leases):
            return "offline"
        return "idle"

    def signal(
        self,
        source: str,
        state: str,
        *,
        ttl: float | None = 30.0,
        explicit: bool = False,
    ) -> None:
        """Acquire or renew a trusted source lease; custom states always expire."""
        custom = state not in RESERVED_STATES
        if (
            not isinstance(source, str)
            or not source
            or len(source) > 200
            or not isinstance(state, str)
            or _STATE.fullmatch(state) is None
            or type(explicit) is not bool
            or (ttl is None and (custom or explicit))
            or (
                ttl is not None
                and (
                    type(ttl) not in (int, float)
                    or not math.isfinite(ttl)
                    or not 0 < ttl <= 300
                )
            )
        ):
            raise ValueError("persona_buddy_signal_invalid")
        if self._closed:
            return
        previous_state = self.requested_state
        previous_lease = self._leases.get(source)
        self._order += 1
        self._leases[source] = _Lease(
            state,
            self._clock() + ttl if ttl is not None else None,
            self._order,
            "explicit" if custom or explicit else "builtin",
        )
        if (
            state in _LIVE
            and not explicit
            and (
                previous_lease is None
                or previous_lease.state != state
                or previous_lease.category != "builtin"
            )
        ):
            self._trigger_leases.pop(source, None)
            self.trigger("live_state", state, owner=source)
        if self.requested_state != previous_state:
            self._generation += 1

    def trigger(self, source: str, match: str, *, owner: str) -> bool:
        """Match a validated authored trigger against exact trusted app metadata."""
        if source not in ALLOWED_TRIGGER_SOURCES or not isinstance(match, str):
            return False
        candidates = [
            item
            for item in self._triggers
            if item.source == source and item.match == match
        ]
        if not candidates or self._closed:
            return False
        selected = max(candidates, key=lambda item: item.priority)
        if not isinstance(owner, str) or not owner or len(owner) > 200:
            return False
        previous_state = self.requested_state
        self._order += 1
        self._trigger_leases[owner] = _Lease(
            selected.state,
            self._clock() + selected.duration_ms / 1000,
            self._order,
            "trigger",
            selected.priority,
        )
        if self.requested_state != previous_state:
            self._generation += 1
        return True

    def release(self, source: str) -> None:
        """Release only the named lifecycle operation."""
        previous_state = self.requested_state
        self._leases.pop(source, None)
        self._trigger_leases.pop(source, None)
        if self.requested_state != previous_state:
            self._generation += 1

    async def _thread(self, function: Callable[[], Any]) -> Any:
        # Repeated cancellation must not release the owner while a worker still
        # reads DB/assets or prepares a frame for the replaced view.
        task = asyncio.create_task(asyncio.to_thread(function))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:  # noqa: BLE001 — private worker failures are path-free at this boundary.
                break
        if cancelled:
            # Consume a possible worker exception without surfacing private text.
            if not task.cancelled():
                task.exception()
            raise asyncio.CancelledError
        return task.result()

    async def load_preferences(self) -> BuddyPreferences:
        """Load once off-loop; invalid or unsafe data keeps the Buddy off."""
        async with self._lock:
            await self._load_locked()
            return self._preferences

    async def _load_locked(self) -> None:
        if self._loaded or self._closed:
            return
        try:
            preferences = await self._thread(self._read_preferences)
        except Exception:  # noqa: BLE001 — private worker failures are path-free at this boundary.
            self._reason = PREFERENCES_REASON
        else:
            if not self._closed:
                self._preferences = preferences
        self._loaded = True

    @contextmanager
    def _preferences_root(self) -> Iterator[int]:
        # The app owns profile creation. Never mkdir through an unchecked parent.
        descriptor = _open_profile_root(
            str(self._profile_root),
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise ValueError(PREFERENCES_REASON)
            yield descriptor
        finally:
            os.close(descriptor)

    def _read_preferences(self) -> BuddyPreferences:
        try:
            with self._preferences_root() as root:
                descriptor = os.open(
                    "persona_buddy.json",
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=root,
                )
                with os.fdopen(descriptor, "rb") as handle:
                    metadata = os.fstat(handle.fileno())
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_uid != os.geteuid()
                        or stat.S_IMODE(metadata.st_mode) != 0o600
                        or metadata.st_nlink != 1
                        or metadata.st_size > 8192
                    ):
                        raise ValueError(PREFERENCES_REASON)
                    data = handle.read(8193)
        except FileNotFoundError:
            return BuddyPreferences()
        if len(data) > 8192:
            raise ValueError(PREFERENCES_REASON)
        values = json.loads(data)
        if type(values) is not dict:
            raise ValueError(PREFERENCES_REASON)
        return _validated_preferences(BuddyPreferences(**values))

    def _write_preferences(self, preferences: BuddyPreferences) -> None:
        # Anchor creation, replacement and cleanup to the validated directory;
        # replacing the pathname during a write cannot redirect the operation.
        with self._preferences_root() as root:
            temporary = ".persona-buddy-" + secrets.token_hex(12)
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=root,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(asdict(preferences), handle, separators=(",", ":"))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(
                    temporary, "persona_buddy.json", src_dir_fd=root, dst_dir_fd=root
                )
            finally:
                try:
                    os.unlink(temporary, dir_fd=root)
                except FileNotFoundError:
                    pass

    def _read_authority(self, persona_id: str) -> _Authority | None:
        try:
            persona = self._service.get_persona_profile(persona_id)
            if (
                persona.get("id") != persona_id
                or persona.get("backend", "local") != "local"
                or persona.get("deleted", False)
                or persona.get("is_active", True) is not True
            ):
                return None
            revision = persona.get("version")
            if type(revision) is not int or revision < 1:
                return None
            graph = self._repository.get_active_persona_pack(persona_id)
            if graph is None or graph.identity.persona_id != persona_id:
                return None
            return _Authority(revision, str(persona.get("name", "Persona")), graph)
        except Exception:  # noqa: BLE001 — private worker failures are path-free at this boundary.
            return None

    def _adopt_authority(self, authority: _Authority | None) -> None:
        if self._authority != authority:
            self._authority = authority
            self._generation += 1
            self._triggers = (
                authority.graph.version.manifest.triggers if authority else ()
            )
            # Authored triggers from a replaced pack must not drive the new pack.
            self._trigger_leases.clear()

    async def select(self, persona_id: str, *, source: str = "local") -> bool:
        """Explicitly select and enable one eligible local Persona."""
        if (
            source != "local"
            or not isinstance(persona_id, str)
            or not persona_id
            or len(persona_id) > 200
        ):
            self._reason = "persona_buddy_local_required"
            return False
        self._generation += 1
        async with self._lock:
            await self._load_locked()
            if self._closed:
                return False
            authority = await self._thread(lambda: self._read_authority(persona_id))
            if authority is None or self._closed:
                self._reason = UNAVAILABLE_REASON
                return False
            preferences = replace(
                self._preferences,
                source="local",
                local_persona_id=persona_id,
                enabled=True,
                open=True,
            )
            try:
                await self._thread(lambda: self._write_preferences(preferences))
                current = await self._thread(lambda: self._read_authority(persona_id))
            except Exception:  # noqa: BLE001 — private worker failures are path-free at this boundary.
                self._reason = PREFERENCES_REASON
                return False
            if self._closed:
                return False
            self._preferences = preferences
            self._leases.clear()
            self._trigger_leases.clear()
            self._adopt_authority(current)
            self._generation += 1
            self._reason = None if current is not None else UNAVAILABLE_REASON
            return current is not None

    async def update_preferences(self, **values: Any) -> BuddyPreferences:
        """Persist geometry or visibility without allowing implicit retargeting."""
        if set(values) - {"enabled", "open", "collapsed", "x", "y", "width", "height"}:
            raise ValueError(PREFERENCES_REASON)
        self._generation += 1
        async with self._lock:
            await self._load_locked()
            if self._closed:
                return self._preferences
            preferences = _validated_preferences(replace(self._preferences, **values))
            if preferences.enabled and preferences.local_persona_id is None:
                raise ValueError(PREFERENCES_REASON)
            try:
                await self._thread(lambda: self._write_preferences(preferences))
            except Exception:  # noqa: BLE001 — private worker failures are path-free at this boundary.
                self._reason = PREFERENCES_REASON
                return self._preferences
            if not self._closed:
                self._preferences = preferences
                self._generation += 1
            return self._preferences

    async def refresh(
        self,
        width: int,
        height: int,
        *,
        reduced_motion: bool = False,
        prepare: Callable[[PersonaVisualResolution], Any] | None = None,
        portrait: PersonaVisualPortrait | None = None,
    ) -> BuddySnapshot | None:
        """Resolve and prepare off-loop, then reject stale authority or lease work."""
        async with self._lock:
            await self._load_locked()
            preferences = self._preferences
            if (
                self._closed
                or not preferences.enabled
                or not preferences.open
                or preferences.local_persona_id is None
                or width <= 0
                or height <= 0
            ):
                return None
            persona_id = preferences.local_persona_id
            requested_state = self.requested_state
            generation = self._generation

            def resolve():
                authority = self._read_authority(persona_id)
                if authority is None:
                    return None, None, None
                resolution = resolve_active_persona_visual(
                    self._repository,
                    persona_id,
                    self._profile_root,
                    requested_state,
                    reduced_motion=reduced_motion or preferences.collapsed,
                    portrait=portrait,
                )
                prepared = prepare(resolution) if prepare is not None else None
                current = self._read_authority(persona_id)
                if (
                    current != authority
                    or resolution.cache_identity.graph != authority.graph.identity
                ):
                    return current, None, None
                return authority, resolution, prepared

            try:
                authority, resolution, prepared = await self._thread(resolve)
            except Exception:  # noqa: BLE001 — private worker failures are path-free at this boundary.
                if generation == self._generation and not self._closed:
                    self._reason = "persona_buddy_frame_failed"
                return None
            if (
                self._closed
                or preferences != self._preferences
                or requested_state != self.requested_state
                or generation != self._generation
            ):
                return None
            self._adopt_authority(authority)
            if requested_state != self.requested_state:
                return None
            if authority is None or resolution is None:
                self._reason = UNAVAILABLE_REASON
                return None
            self._reason = resolution.reason
            # Valid authority survives missing frames, so the view can retain
            # its same-identity static image or display a labelled fallback.
            return BuddySnapshot(
                persona_id,
                authority.name,
                self._generation,
                authority.graph.identity,
                requested_state,
                resolution,
                prepared,
            )

    async def shutdown(self) -> None:
        """Fence pending results and drain this owner's current worker."""
        self._closed = True
        self._generation += 1
        async with self._lock:
            self._leases.clear()
            self._trigger_leases.clear()
            self._authority = None
            self._triggers = ()


def _validated_preferences(preferences: BuddyPreferences) -> BuddyPreferences:
    if (
        preferences.source != "local"
        or any(
            type(getattr(preferences, key)) is not bool
            for key in ("enabled", "open", "collapsed")
        )
        or (
            preferences.local_persona_id is not None
            and (
                not isinstance(preferences.local_persona_id, str)
                or not 0 < len(preferences.local_persona_id) <= 200
            )
        )
        or any(
            value is not None and (type(value) is not int or not 0 <= value <= 10000)
            for value in (preferences.x, preferences.y)
        )
        or any(
            type(value) is not int or not 1 <= value <= 10000
            for value in (preferences.width, preferences.height)
        )
        or (preferences.enabled and preferences.local_persona_id is None)
    ):
        raise ValueError(PREFERENCES_REASON)
    return preferences
