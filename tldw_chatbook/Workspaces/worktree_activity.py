"""Runtime-owned complete worktree reservations, without execution (ADR-285).

Scopes consume resolver-admitted absolute paths and captured root identities;
they confer no filesystem authority. Claims retire only after every attached
native owner positively drains and its applicable required writes and original
host handoffs settle.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING
from unicodedata import normalize
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Utils.filesystem_identity import (
    DirectoryChain,
    directory_identity_from_stat,
)

if TYPE_CHECKING:
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner


@dataclass(frozen=True, slots=True)
class WorktreeScope:
    """One resolver-admitted exact target or subtree under a captured root.

    Target metadata is captured before admission, never under the owner lock.
    Missing targets are valid create preconditions; other metadata failures
    refuse construction. Authority/exclusions remain the resolver's concern.
    """

    path: Path
    subtree: bool
    root: DirectoryChain = field(repr=False)
    _parts: tuple[str, ...] = field(init=False, repr=False)
    _locations: tuple[tuple[int, int, tuple[str, ...]], ...] = field(
        init=False, repr=False
    )
    _target_identity: tuple[int, int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or ".." in self.path.parts
            or "\x00" in str(self.path)
            or type(self.subtree) is not bool
            or not isinstance(self.root, DirectoryChain)
            or not self.root.canonical_root.is_absolute()
        ):
            raise ValueError("admitted canonical scope required")
        ancestors = (self.root.canonical_root, *self.root.canonical_root.parents)
        if len(self.root.identities) != len(ancestors):
            raise ValueError("complete admitted root identity required")
        try:
            self.path.relative_to(self.root.canonical_root)
        except ValueError:
            raise ValueError("scope is outside admitted root") from None
        locations = []
        for ancestor, identity in zip(ancestors, self.root.identities, strict=True):
            if not stat.S_ISDIR(identity.mode) or identity.reparse:
                raise ValueError("safe admitted root identity required")
            suffix = self.path.relative_to(ancestor).parts
            locations.append(
                (
                    identity.device,
                    identity.inode,
                    tuple(normalize("NFC", p).casefold() for p in suffix),
                )
            )
        # shortcut: case/Unicode aliases serialize; refine if distinct-name contention warrants it.
        object.__setattr__(
            self,
            "_parts",
            tuple(normalize("NFC", p).casefold() for p in self.path.parts),
        )
        object.__setattr__(self, "_locations", tuple(locations))
        try:
            metadata = os.lstat(self.path)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("scope target changed after resolution")
            identity = directory_identity_from_stat(metadata)
            if identity.reparse:
                raise ValueError("scope target changed after resolution")
            target_identity = (identity.device, identity.inode)
        except FileNotFoundError:
            target_identity = None
        except OSError:
            raise ValueError("scope target metadata unavailable") from None
        object.__setattr__(self, "_target_identity", target_identity)


def _path_overlap(
    first: tuple[str, ...], second: tuple[str, ...], first_tree: bool, second_tree: bool
) -> bool:
    return (
        first == second
        or (first_tree and second[: len(first)] == first)
        or (second_tree and first[: len(second)] == second)
    )


def _overlap(first: WorktreeScope, second: WorktreeScope) -> bool:
    if (
        first._target_identity is not None
        and first._target_identity == second._target_identity
    ) or _path_overlap(first._parts, second._parts, first.subtree, second.subtree):
        return True
    return any(
        (device_a, inode_a) == (device_b, inode_b)
        and _path_overlap(path_a, path_b, first.subtree, second.subtree)
        for device_a, inode_a, path_a in first._locations
        for device_b, inode_b, path_b in second._locations
    )


@dataclass(frozen=True, slots=True)
class WorktreeFootprint:
    """The entire immutable footprint, including runtime-wide opaque effects."""

    reads: tuple[WorktreeScope, ...] = ()
    writes: tuple[WorktreeScope, ...] = ()
    opaque: bool = False

    def __post_init__(self) -> None:
        if type(self.opaque) is not bool or any(
            not isinstance(scopes, tuple)
            or any(not isinstance(scope, WorktreeScope) for scope in scopes)
            for scopes in (self.reads, self.writes)
        ):
            raise ValueError("immutable admitted footprint required")

    def _conflicts(self, other: WorktreeFootprint) -> bool:
        if self.opaque or other.opaque:
            return True
        return any(
            _overlap(first, second)
            for writes, targets in (
                (self.writes, other.reads + other.writes),
                (other.writes, self.reads),
            )
            for first in writes
            for second in targets
        )


@dataclass(frozen=True, slots=True)
class WorktreeActivitySettlement:
    """Host-only durable-settlement handle for one actual native owner."""

    id: str
    _activity: WorktreeActivity = field(repr=False)

    def settle(self, *, durable: bool) -> None:
        """Record all required writes/host handoffs; false never releases a claim.

        Ordinary finite events with no durable payload must still conclusively
        deliver or discard their original host outcome. Workflow obligations
        remain durable; this handle does not invent an ordinary event journal.
        """
        self._activity._settle(self.id, durable=durable)


@dataclass(frozen=True, slots=True)
class WorktreeActivityClaim(WorktreeActivitySettlement):
    """Host launch custody for a complete footprint; never model-supplied IDs."""

    def bind_owner(self, owner: ExecutionOwner) -> None:
        """Bind real capacity admission before starting any worker/effects."""
        self._activity._bind_owner(self.id, owner)

    def attach_child(self, owner: ExecutionOwner) -> WorktreeActivitySettlement:
        """Retain this tree's claim for an admitted child before thread start."""
        return self._activity._attach_child(self.id, owner)


@dataclass
class _OwnerState:
    claim_id: str
    owner: ExecutionOwner
    public: WorktreeActivitySettlement
    drained: bool | None = None
    durable: bool = False


@dataclass
class _ClaimState:
    footprint: WorktreeFootprint
    custody: WorktreeActivityClaim
    members: dict[str, _OwnerState] = field(default_factory=dict)


class WorktreeActivity:
    """One runtime's all-or-none reservation owner; never launches or waits."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._claims: dict[str, _ClaimState] = {}
        self._members: dict[str, _OwnerState] = {}
        self._recovery: dict[str, WorktreeFootprint] = {}
        self._observers: set[Callable[[], None]] = set()

    def protect_recovery(self, workflow_id: str, footprint: WorktreeFootprint) -> None:
        """Retain startup uncertainty for this runtime; this only denies admission."""
        if not workflow_id or not isinstance(footprint, WorktreeFootprint):
            raise ValueError("recovery identity and complete footprint required")
        with self._lock:
            previous = self._recovery.get(workflow_id)
            if previous is not None and previous != footprint:
                raise ValueError("recovery protection cannot be replaced")
            self._recovery[workflow_id] = footprint

    def recovery_blockers(self, footprint: WorktreeFootprint) -> tuple[str, ...]:
        """Return body-free identities of unresolved startup protection."""
        with self._lock:
            return tuple(
                key
                for key, protected in self._recovery.items()
                if footprint._conflicts(protected)
            )

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Observe released activity; callbacks run outside the admission lock."""
        with self._lock:
            self._observers.add(callback)

        def unsubscribe() -> None:
            with self._lock:
                self._observers.discard(callback)

        return unsubscribe

    def _notify_released(self) -> None:
        with self._lock:
            observers = tuple(self._observers)
        for callback in observers:
            try:
                callback()
            except Exception:  # noqa: BLE001 - observers cannot prevent claim retirement
                logger.warning("Worktree activity observer failed")

    @contextmanager
    def prelaunch(
        self, *, footprint: WorktreeFootprint
    ) -> Iterator[WorktreeActivityClaim | None]:
        """Acquire before native capacity; unwind only a never-bound claim.

        The caller must bind the actually admitted owner before any work starts.
        Capacity refusal or an exception before binding releases this reservation;
        any outcome after binding requires physical drain and durable settlement.
        """
        if not isinstance(footprint, WorktreeFootprint):
            raise TypeError("footprint must be a WorktreeFootprint")
        claim = WorktreeActivityClaim(uuid4().hex, self)
        with self._lock:
            if any(
                footprint._conflicts(protected)
                for protected in (
                    *(s.footprint for s in self._claims.values()),
                    *self._recovery.values(),
                )
            ):
                claim = None
            else:
                self._claims[claim.id] = _ClaimState(footprint, claim)
        try:
            yield claim
        finally:
            released = False
            if claim is not None:
                with self._lock:
                    state = self._claims.get(claim.id)
                    if state is not None and not state.members:
                        del self._claims[claim.id]
                        released = True
            if released:
                self._notify_released()

    def try_acquire(
        self, *, owner: ExecutionOwner, footprint: WorktreeFootprint
    ) -> WorktreeActivityClaim | None:
        """Acquire a complete claim for an existing owner without waiting.

        Child launch admission must instead use prelaunch before native capacity.
        """
        with self.prelaunch(footprint=footprint) as claim:
            if claim is not None:
                claim.bind_owner(owner)
            return claim

    def _bind_owner(self, claim_id: str, owner: ExecutionOwner) -> None:
        from tldw_chatbook.Agents.execution_capacity import ExecutionOwner

        if not isinstance(owner, ExecutionOwner):
            raise TypeError("actual ExecutionOwner required")
        if not owner.is_admitted:
            raise ValueError("live admitted execution owner required")
        with self._lock:
            state = self._claims.get(claim_id)
            if state is None or state.members:
                raise ValueError("claim is expired or already bound")
            public = WorktreeActivitySettlement(claim_id, self)
            member = _OwnerState(claim_id, owner, public)
            state.members[claim_id] = member
            self._members[claim_id] = member
        owner.on_drained(lambda proven: self.record_drain(claim_id, proven=proven))

    def validate_claim(
        self,
        claim: WorktreeActivityClaim,
        *,
        owner: ExecutionOwner,
        footprint: WorktreeFootprint,
    ) -> None:
        """Verify original held custody and native membership without granting access."""
        if not owner.is_admitted:
            raise ValueError("activity claim owner is not admitted")
        with self._lock:
            state = (
                self._claims.get(claim.id)
                if type(claim) is WorktreeActivityClaim
                else None
            )
            member = state.members.get(claim.id) if state is not None else None
            if (
                state is None
                or state.custody is not claim
                or state.footprint != footprint
                or member is None
                or member.owner is not owner
                or member.drained is not None
            ):
                raise ValueError("live original activity claim custody required")

    def validate_opaque_claim(
        self, claim: WorktreeActivityClaim, *, owner: ExecutionOwner
    ) -> None:
        """Borrow only an original primary owner's already-complete opaque claim."""
        with self._lock:
            state = (
                self._claims.get(claim.id)
                if type(claim) is WorktreeActivityClaim
                else None
            )
            footprint = state.footprint if state is not None else None
        if footprint is None or not footprint.opaque:
            raise ValueError("original opaque snapshot coverage required")
        self.validate_claim(claim, owner=owner, footprint=footprint)

    def _attach_child(
        self, claim_id: str, owner: ExecutionOwner
    ) -> WorktreeActivitySettlement:
        from tldw_chatbook.Agents.execution_capacity import ExecutionOwner

        if not isinstance(owner, ExecutionOwner):
            raise TypeError("actual ExecutionOwner required")
        if not owner.is_admitted:
            raise ValueError("live admitted execution owner required")
        member_id = uuid4().hex
        public = WorktreeActivitySettlement(member_id, self)
        with self._lock:
            state = self._claims.get(claim_id)
            if state is None or not state.members:
                raise ValueError("bound live claim required")
            primary = state.members[claim_id]
            if not owner.child or owner.capacity is not primary.owner.capacity:
                raise ValueError("admitted child from the same runtime required")
            for existing in state.members.values():
                if existing.owner is owner:
                    return existing.public
            if primary.drained is not None:
                raise ValueError("parent execution has already drained")
            member = _OwnerState(claim_id, owner, public)
            state.members[member_id] = member
            self._members[member_id] = member
        owner.on_drained(lambda proven: self.record_drain(member_id, proven=proven))
        return public

    def record_drain(self, claim_id: str, *, proven: bool) -> None:
        """Internal native callback; idempotently retain negative drain proof."""
        with self._lock:
            member = self._members.get(claim_id)
            if member is None or member.drained is not None:
                return
            member.drained = proven is True
            released = self._retire_if_settled(member.claim_id)
        if released:
            self._notify_released()

    def _settle(self, member_id: str, *, durable: bool) -> None:
        if type(durable) is not bool:
            raise TypeError("durable must be a bool")
        with self._lock:
            member = self._members.get(member_id)
            if member is None:
                if member_id in self._claims:
                    raise ValueError("bind actual execution before settlement")
                return
            member.durable = member.durable or durable
            released = self._retire_if_settled(member.claim_id)
        if released:
            self._notify_released()

    def _retire_if_settled(self, claim_id: str) -> bool:
        state = self._claims[claim_id]
        if all(
            member.drained is True and member.durable
            for member in state.members.values()
        ):
            for member_id in state.members:
                del self._members[member_id]
            del self._claims[claim_id]
            return True
        return False
