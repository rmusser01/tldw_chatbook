"""Process-memory ownership for one File Notes root session."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Literal, Protocol

SessionChangeAction = Literal[
    "created",
    "modified",
    "moved",
    "deleted",
    "restored",
]
SessionTransitionKind = Literal["root", "path", "source", "screen"]
_TRANSITION_KINDS = frozenset({"root", "path", "source", "screen"})


@dataclass(frozen=True, slots=True)
class SessionChange:
    """One successful filesystem mutation initiated by Chatbook."""

    action: SessionChangeAction
    relative_path: str
    destination_path: str | None = None


@dataclass(frozen=True, slots=True)
class SessionBinding:
    """Canonical selected root paired with its owner generation."""

    root_key: str
    generation: int


@dataclass(frozen=True, slots=True)
class SequencedSessionChange:
    """One session change in process-wide publication order."""

    sequence: int
    change: SessionChange


@dataclass(frozen=True, slots=True)
class FileNotesSessionSnapshot:
    """Immutable state for one requested root generation."""

    binding: SessionBinding
    changes: tuple[SequencedSessionChange, ...] = ()


class FileNotesGitServiceLifecycle(Protocol):
    """Narrow lifecycle boundary for an owner-attached Git service."""

    def shutdown(self) -> None:
        """Stop accepting work and settle retained service work."""


@dataclass(frozen=True, slots=True)
class SessionTransitionLease:
    """Idempotently releasable root/path/source/screen transition admission."""

    _owner: FileNotesSessionOwner = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)
    kind: SessionTransitionKind

    def release(self) -> None:
        """Release this transition admission once."""
        self._owner._release_transition(self._token)


@dataclass(frozen=True, slots=True)
class GitMutationLease:
    """Idempotently releasable Git mutation admission."""

    _owner: FileNotesSessionOwner = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def release(self) -> None:
        """Release this mutation admission once."""
        self._owner._release_mutation(self._token)


@dataclass(frozen=True, slots=True)
class GitStatusLease:
    """Idempotently releasable Git status admission."""

    _owner: FileNotesSessionOwner = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def release(self) -> None:
        """Release this status admission once."""
        self._owner._release_status(self._token)


class FileNotesSessionOwner:
    """Own root-scoped File Notes session state for one application process."""

    __slots__ = (
        "_binding",
        "_changes",
        "_generation",
        "_git_service",
        "_lock",
        "_mutation_token",
        "_next_sequence",
        "_shutdown",
        "_status_token",
        "_transition_tokens",
    )

    def __init__(self) -> None:
        self._lock = RLock()
        self._binding: SessionBinding | None = None
        self._generation = 0
        self._changes: list[SequencedSessionChange] = []
        self._next_sequence = 1
        self._transition_tokens: set[object] = set()
        self._mutation_token: object | None = None
        self._status_token: object | None = None
        self._shutdown = False
        self._git_service: FileNotesGitServiceLifecycle | None = None

    def select_root(self, root: str | Path) -> SessionBinding:
        """Select one canonical root, resetting state only when it changes.

        Args:
            root: Filesystem root to bind for this process session.

        Returns:
            The current immutable root-generation binding.

        Raises:
            RuntimeError: If the owner has already shut down.
        """
        root_key = str(Path(root).expanduser().resolve(strict=False))
        with self._lock:
            if self._shutdown:
                raise RuntimeError("File Notes session owner is shut down")
            if self._binding is not None and self._binding.root_key == root_key:
                return self._binding
            self._generation += 1
            self._binding = SessionBinding(root_key, self._generation)
            self._changes.clear()
            self._next_sequence = 1
            return self._binding

    def record_change(
        self,
        binding: SessionBinding,
        change: SessionChange,
    ) -> bool:
        """Publish one change if its root generation is still current."""
        with self._lock:
            if self._shutdown or binding != self._binding:
                return False
            self._changes.append(
                SequencedSessionChange(
                    sequence=self._next_sequence,
                    change=change,
                )
            )
            self._next_sequence += 1
            return True

    def snapshot(self, binding: SessionBinding) -> FileNotesSessionSnapshot:
        """Return an immutable snapshot without exposing another generation."""
        with self._lock:
            changes = tuple(self._changes) if binding == self._binding else ()
            return FileNotesSessionSnapshot(binding=binding, changes=changes)

    def try_acquire_transition(
        self,
        binding: SessionBinding,
        kind: SessionTransitionKind,
    ) -> SessionTransitionLease | None:
        """Try to admit a transition without awaiting."""
        if kind not in _TRANSITION_KINDS:
            raise ValueError(f"Unsupported File Notes transition kind: {kind}")
        with self._lock:
            if (
                self._shutdown
                or binding != self._binding
                or self._mutation_token is not None
            ):
                return None
            token = object()
            self._transition_tokens.add(token)
            return SessionTransitionLease(self, token, kind)

    def try_acquire_mutation(
        self,
        binding: SessionBinding,
    ) -> GitMutationLease | None:
        """Try to admit one mutation without awaiting."""
        with self._lock:
            if (
                self._shutdown
                or binding != self._binding
                or self._transition_tokens
                or self._mutation_token is not None
            ):
                return None
            token = object()
            self._mutation_token = token
            return GitMutationLease(self, token)

    def try_acquire_status(
        self,
        binding: SessionBinding,
    ) -> GitStatusLease | None:
        """Try to admit one status operation without awaiting."""
        with self._lock:
            if (
                self._shutdown
                or binding != self._binding
                or self._mutation_token is not None
                or self._status_token is not None
            ):
                return None
            token = object()
            self._status_token = token
            return GitStatusLease(self, token)

    def attach_git_service(self, service: FileNotesGitServiceLifecycle) -> None:
        """Attach the one optional service whose lifecycle this owner controls."""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("File Notes session owner is shut down")
            if self._git_service is not None:
                raise RuntimeError("A File Notes Git service is already attached")
            self._git_service = service

    def shutdown(self) -> None:
        """Seal admission and shut down the attached service exactly once."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            service = self._git_service
            self._git_service = None
        if service is not None:
            service.shutdown()

    def _release_transition(self, token: object) -> None:
        with self._lock:
            self._transition_tokens.discard(token)

    def _release_mutation(self, token: object) -> None:
        with self._lock:
            if self._mutation_token is token:
                self._mutation_token = None

    def _release_status(self, token: object) -> None:
        with self._lock:
            if self._status_token is token:
                self._status_token = None
