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
    capture_directory_chain,
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


@dataclass(frozen=True, slots=True)
class _WaveFence:
    """Original window membership; ordering only, never filesystem authority."""

    wave: object
    footprint: WorktreeFootprint
    members: tuple[object, ...]


class WorktreeActivity:
    """One runtime's all-or-none reservation owner; never launches or waits."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._claims: dict[str, _ClaimState] = {}
        self._members: dict[str, _OwnerState] = {}
        self._recovery: dict[str, WorktreeFootprint] = {}
        self._waves: dict[object, _WaveFence] = {}
        self._ready_manual: set[object] = set()
        self._observers: set[Callable[[], None]] = set()

    def _set_ready_manual(self, source: object, *, ready: bool) -> None:
        """Publish the existing queue's body-free priority observation."""
        with self._lock:
            if ready:
                self._ready_manual.add(source)
            else:
                self._ready_manual.discard(source)
        if not ready:
            self._notify_released()

    def _open_wave(self, wave, footprint, members):
        """Install ordering atomically against ordinary claim admission."""
        from tldw_chatbook.Agents.workflow_wave import WorkflowWave

        if (
            type(wave) is not WorkflowWave
            or wave.runtime.worktree_activity is not self
        ):
            raise ValueError("original workflow wave required")
        wave.check_prepared_members(members)
        fence = _WaveFence(wave, footprint, tuple(members))
        self._prune_recovery()
        with self._lock:
            if wave in self._waves:
                raise ValueError("workflow wave already opened")
            if self._ready_manual or any(
                footprint._conflicts(protected)
                for protected in (
                    *(state.footprint for state in self._claims.values()),
                    *self._recovery.values(),
                    *(item.footprint for item in self._waves.values()),
                )
            ):
                return None
            self._waves[wave] = fence
        return fence

    def _close_wave(self, fence):
        """Retire only the exact original fence after its required disposition."""
        with self._lock:
            if self._waves.get(fence.wave) is not fence:
                raise ValueError("original workflow fence required")
            del self._waves[fence.wave]
        self._notify_released()

    @contextmanager
    def _mcp_permission_publication(self):
        # Protect the empty slot of an already admitted opaque owner.
        with self._lock:
            yield not any(state.footprint.opaque for state in self._claims.values())

    @contextmanager
    def _snapshot_prelaunch(self, source):
        """Finite original jobs conflict physically without acquiring a tree lane."""
        from .change_review_ownership import _SnapshotSource

        if type(source) is not _SnapshotSource or source.issuer.activity is not self:
            raise ValueError("original finite snapshot source required")
        source.check_source()
        with self._prelaunch(
            footprint=WorktreeFootprint(opaque=True), snapshot_source=source
        ) as claim:
            yield claim

    def protect_recovery(self, workflow_id: str, footprint: WorktreeFootprint) -> None:
        """Retain startup uncertainty for this runtime; this only denies admission."""
        if not workflow_id or not isinstance(footprint, WorktreeFootprint):
            raise ValueError("recovery identity and complete footprint required")
        self._prune_recovery()
        with self._lock:
            previous = self._recovery.get(workflow_id)
            if previous is not None and previous != footprint:
                raise ValueError("recovery protection cannot be replaced")
            self._recovery[workflow_id] = footprint

    def _prune_recovery(self) -> None:
        """Remove only historical scopes whose captured folder no longer matches."""
        with self._lock:
            saved = tuple(self._recovery.items())
        changed = False
        for workflow_id, footprint in saved:
            current = set()
            for root in {scope.root for scope in footprint.reads + footprint.writes}:
                try:
                    if capture_directory_chain(root.canonical_root) == root:
                        current.add(root)
                except (OSError, ValueError):
                    try:
                        identity = directory_identity_from_stat(
                            os.lstat(root.canonical_root)
                        )
                    except (FileNotFoundError, NotADirectoryError):
                        continue  # Confirmed absence retires this historical scope.
                    except (OSError, ValueError):
                        current.add(
                            root
                        )  # Unreadable metadata does not disprove ownership.
                    else:
                        if identity == root.identities[0]:
                            current.add(root)
            remaining = WorktreeFootprint(
                reads=tuple(
                    scope for scope in footprint.reads if scope.root in current
                ),
                writes=tuple(
                    scope for scope in footprint.writes if scope.root in current
                ),
                opaque=footprint.opaque,
            )
            if remaining == footprint:
                continue
            with self._lock:
                if self._recovery.get(workflow_id) is not footprint:
                    continue
                if remaining.reads or remaining.writes or remaining.opaque:
                    self._recovery[workflow_id] = remaining
                else:
                    del self._recovery[workflow_id]
                changed = True
        if changed:
            self._notify_released()

    def recovery_blockers(self, footprint: WorktreeFootprint) -> tuple[str, ...]:
        """Return body-free identities of unresolved startup protection."""
        self._prune_recovery()
        with self._lock:
            return tuple(
                key
                for key, protected in self._recovery.items()
                if footprint._conflicts(protected)
            )

    def release_recovery(self, workflow_id: str) -> bool:
        """Release reviewed history's overlay; actual native claims stay owned."""
        with self._lock:
            removed = self._recovery.pop(workflow_id, None) is not None
        if removed:
            self._notify_released()
        return removed

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
        self, *, footprint: WorktreeFootprint, _wave_member=None
    ) -> Iterator[WorktreeActivityClaim | None]:
        """Acquire before native capacity; unwind only a never-bound claim.

        The caller must bind the actually admitted owner before any work starts.
        Capacity refusal or an exception before binding releases this reservation;
        any outcome after binding requires physical drain and durable settlement.
        """
        with self._prelaunch(footprint=footprint, wave_member=_wave_member) as claim:
            yield claim

    @contextmanager
    def _prelaunch(self, *, footprint, wave_member=None, snapshot_source=None):
        if not isinstance(footprint, WorktreeFootprint):
            raise TypeError("footprint must be a WorktreeFootprint")
        self._prune_recovery()
        claim = WorktreeActivityClaim(uuid4().hex, self)
        with self._lock:
            foreign_wave = snapshot_source is None and any(
                footprint._conflicts(fence.footprint)
                and not any(wave_member is member for member in fence.members)
                for fence in self._waves.values()
            )
            if foreign_wave or any(
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

    def _release_workflow_observation(self, ticket):
        """Retire finite observation while the original window retains REQUIRED E."""
        from .change_review_ownership import _SnapshotTicket

        if type(ticket) is not _SnapshotTicket or ticket.claim is None:
            raise ValueError("original snapshot observation required")
        source = ticket.source
        wave = source.workflow_wave
        if (
            wave is None
            or wave.source is not source
            or wave.publication is None
            or wave.publication.reservation is not wave.reservation
        ):
            raise ValueError("original durable workflow window required")
        with self._lock:
            state = self._claims.get(ticket.claim.id)
            member = state.members.get(ticket.claim.id) if state is not None else None
            if (
                self._waves.get(wave) is not wave.fence
                or state is None
                or state.custody is not ticket.claim
                or member is None
                or member.owner is not ticket.owner
                or member.drained is not True
                or not ticket._worker_returned
                or not ticket._handoff_returned
                or not ticket._observation_returned
            ):
                raise ValueError("original workflow observation return unproven")
            # This finite claim has no required payload. The still-live fence
            # above retains the distinct window's failed required publication.
            del self._members[ticket.claim.id]
            del self._claims[ticket.claim.id]
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
