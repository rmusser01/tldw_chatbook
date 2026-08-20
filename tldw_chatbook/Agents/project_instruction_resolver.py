"""Secure, bounded discovery of repository-authored project instructions."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

InstructionKind = Literal["override", "standard"]
InstructionOutcomeCode = Literal[
    "omitted_byte_budget",
    "omitted_token_budget",
    "stale",
    "invalid",
    "resolution_failed",
]

_WINDOWS = os.name == "nt"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", None)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", None)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_BINARY = getattr(os, "O_BINARY", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


@dataclass(frozen=True, slots=True)
class InstructionSource:
    """One securely pinned project-instruction source."""

    canonical_path: Path = field(repr=False)
    relative_path: str
    scope: str
    kind: InstructionKind
    body: str = field(repr=False)
    byte_count: int
    digest: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class InstructionOutcome:
    """Content-free result for a source that was not delivered."""

    relative_path: str
    scope: str
    code: InstructionOutcomeCode


@dataclass(frozen=True, slots=True)
class StartupInstructionCandidate:
    """Securely pinned, byte-admitted startup resolver result."""

    binding_id: str
    binding_root: Path = field(repr=False)
    locator_fingerprint: str = field(repr=False)
    dispatch_started_wall_ns: int = field(repr=False)
    source: InstructionSource | None
    outcomes: tuple[InstructionOutcome, ...]


@dataclass(frozen=True, slots=True)
class InstructionChainDelivery:
    """Instruction digests and terminal outcomes delivered to one model chain."""

    source_digests: tuple[str, ...] = field(repr=False)
    outcomes: tuple[InstructionOutcome, ...]


@dataclass(frozen=True, slots=True)
class InstructionSnapshot:
    """Immutable project-instruction state for one Console dispatch."""

    binding_id: str
    binding_root: Path = field(repr=False)
    locator_fingerprint: str = field(repr=False)
    dispatch_started_wall_ns: int = field(repr=False)
    startup_source: InstructionSource | None
    global_outcomes: tuple[InstructionOutcome, ...]
    primary_delivery: InstructionChainDelivery
    warning_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _FallbackCondition:
    kind: Literal["absent", "empty"]
    file_identity: tuple[int, int, int, int, int] | None = None
    digest: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _ReadResult:
    source: InstructionSource | None = None
    outcome: InstructionOutcome | None = None
    fallback_condition: _FallbackCondition | None = None


class _UnsafeMetadata(Exception):
    pass


class ProjectInstructionResolver:
    """Resolve only the selected binding root's effective instruction file."""

    def resolve_startup(
        self,
        *,
        binding_id: str,
        binding_root: Path,
        locator_fingerprint: str,
        max_bytes: int,
        dispatch_started_wall_ns: int,
    ) -> StartupInstructionCandidate:
        """Resolve and securely pin the effective binding-root instructions.

        Args:
            binding_id: Selected workspace binding identity.
            binding_root: Canonical selected workspace locator.
            locator_fingerprint: Fingerprint captured when the binding was selected.
            max_bytes: Maximum raw bytes admitted for the startup source.
            dispatch_started_wall_ns: Dispatch wall-clock cutoff in nanoseconds.

        Returns:
            A byte-admitted candidate containing at most one root source.

        Raises:
            ValueError: If ``max_bytes`` is negative.
        """
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")

        root, expected_ancestors = _canonical_binding_root(binding_root)
        if expected_ancestors is None:
            return StartupInstructionCandidate(
                binding_id=binding_id,
                binding_root=root,
                locator_fingerprint=locator_fingerprint,
                dispatch_started_wall_ns=dispatch_started_wall_ns,
                source=None,
                outcomes=(InstructionOutcome(".", ".", "resolution_failed"),),
            )

        override = _read_candidate(
            root=root,
            filename="AGENTS.override.md",
            kind="override",
            max_bytes=max_bytes,
            dispatch_started_wall_ns=dispatch_started_wall_ns,
            expected_ancestors=expected_ancestors,
        )
        result = override
        if override.fallback_condition is not None:
            result = _read_candidate(
                root=root,
                filename="AGENTS.md",
                kind="standard",
                max_bytes=max_bytes,
                dispatch_started_wall_ns=dispatch_started_wall_ns,
                expected_ancestors=expected_ancestors,
            )
            rechecked_override = _read_candidate(
                root=root,
                filename="AGENTS.override.md",
                kind="override",
                max_bytes=max_bytes,
                dispatch_started_wall_ns=dispatch_started_wall_ns,
                expected_ancestors=expected_ancestors,
            )
            if rechecked_override.fallback_condition != override.fallback_condition:
                result = _fallback_changed_result(rechecked_override)

        return StartupInstructionCandidate(
            binding_id=binding_id,
            binding_root=root,
            locator_fingerprint=locator_fingerprint,
            dispatch_started_wall_ns=dispatch_started_wall_ns,
            source=result.source,
            outcomes=(result.outcome,) if result.outcome else (),
        )


def admit_sources(
    sources: Sequence[InstructionSource],
    safe_input_tokens: int,
    count_tokens: Callable[[InstructionSource], int],
) -> InstructionChainDelivery:
    """Admit whole sources under an injected model-token budget.

    Sources are supplied broad-to-specific. Admission considers them in reverse
    order so narrower guidance wins, then reports admitted digests in rendering
    order.

    Args:
        sources: Project sources ordered broad-to-specific.
        safe_input_tokens: Remaining safe provider input allowance.
        count_tokens: Pure estimator including any source wrapper overhead.

    Returns:
        The admitted source digests and whole-source omission outcomes.
    """
    remaining = max(0, safe_input_tokens)
    admitted: set[int] = set()
    omitted: set[int] = set()
    for index in range(len(sources) - 1, -1, -1):
        try:
            needed = count_tokens(sources[index])
        except Exception:
            omitted.add(index)
            continue
        if type(needed) is not int or needed <= 0:
            omitted.add(index)
            continue
        if needed <= remaining:
            admitted.add(index)
            remaining -= needed
        else:
            omitted.add(index)

    outcomes = tuple(
        InstructionOutcome(source.relative_path, source.scope, "omitted_token_budget")
        for index, source in enumerate(sources)
        if index in omitted
    )
    return InstructionChainDelivery(
        source_digests=tuple(
            source.digest for index, source in enumerate(sources) if index in admitted
        ),
        outcomes=outcomes,
    )


def _canonical_binding_root(
    binding_root: Path,
) -> tuple[Path, tuple[tuple[int, int, int], ...] | None]:
    lexical = _safe_absolute(binding_root)
    if lexical is None:
        return binding_root, None
    try:
        return lexical, _capture_ancestor_identities(lexical)
    except (OSError, RuntimeError, ValueError, _UnsafeMetadata):
        return lexical, None


def _safe_absolute(path: Path) -> Path | None:
    try:
        return path.absolute()
    except (OSError, RuntimeError, ValueError):
        return None


def _read_candidate(
    *,
    root: Path,
    filename: str,
    kind: InstructionKind,
    max_bytes: int,
    dispatch_started_wall_ns: int,
    expected_ancestors: tuple[tuple[int, int, int], ...],
) -> _ReadResult:
    path = root / filename

    def outcome(code: InstructionOutcomeCode) -> InstructionOutcome:
        return InstructionOutcome(filename, ".", code)

    try:
        if _capture_ancestor_identities(root) != expected_ancestors:
            raise _UnsafeMetadata
    except (OSError, _UnsafeMetadata):
        return _ReadResult(outcome=outcome("resolution_failed"))
    try:
        file_before = os.lstat(path)
    except FileNotFoundError:
        try:
            if _capture_ancestor_identities(root) != expected_ancestors:
                raise _UnsafeMetadata
        except (OSError, _UnsafeMetadata):
            return _ReadResult(outcome=outcome("resolution_failed"))
        return _ReadResult(fallback_condition=_FallbackCondition("absent"))
    except OSError:
        return _ReadResult(outcome=outcome("resolution_failed"))

    try:
        file_identity = _verified_state(file_before)
        if _capture_ancestor_identities(root) != expected_ancestors:
            raise _UnsafeMetadata
        if not stat.S_ISREG(file_before.st_mode):
            return _ReadResult(outcome=outcome("invalid"))
        if stat.S_ISLNK(file_before.st_mode) or _is_reparse(file_before):
            return _ReadResult(outcome=outcome("invalid"))
        if file_before.st_mtime_ns > dispatch_started_wall_ns:
            return _ReadResult(outcome=outcome("stale"))
        if file_before.st_size > max_bytes:
            return _ReadResult(outcome=outcome("omitted_byte_budget"))
    except _UnsafeMetadata:
        return _ReadResult(outcome=outcome("resolution_failed"))

    flags = os.O_RDONLY | _CLOEXEC | _BINARY | _NONBLOCK
    if _NOFOLLOW is not None:
        flags |= _NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if _is_reparse(opened) or _verified_state(opened) != file_identity:
                raise _UnsafeMetadata
            raw = _bounded_read(descriptor, max_bytes + 1)
            finished = os.fstat(descriptor)
        finally:
            os.close(descriptor)

        file_after = os.lstat(path)
        ancestors_after = _capture_ancestor_identities(root)
        if (
            _is_reparse(finished)
            or _is_reparse(file_after)
            or _verified_state(finished) != file_identity
            or _verified_state(file_after) != file_identity
            or ancestors_after != expected_ancestors
        ):
            raise _UnsafeMetadata
        if len(raw) > max_bytes:
            return _ReadResult(outcome=outcome("omitted_byte_budget"))
        body = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        return _ReadResult(outcome=outcome("invalid"))
    except (OSError, _UnsafeMetadata):
        return _ReadResult(outcome=outcome("resolution_failed"))

    if not body.strip():
        return _ReadResult(
            fallback_condition=_FallbackCondition(
                "empty",
                file_identity=file_identity,
                digest=hashlib.sha256(raw).hexdigest(),
            )
        )
    return _ReadResult(
        source=InstructionSource(
            canonical_path=path,
            relative_path=filename,
            scope=".",
            kind=kind,
            body=body,
            byte_count=len(raw),
            digest=hashlib.sha256(raw).hexdigest(),
        )
    )


def _fallback_changed_result(rechecked: _ReadResult) -> _ReadResult:
    if rechecked.outcome is not None:
        return rechecked
    return _ReadResult(
        outcome=InstructionOutcome("AGENTS.override.md", ".", "resolution_failed")
    )


def _bounded_read(descriptor: int, cap: int) -> bytes:
    chunks: list[bytes] = []
    remaining = cap
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _verified_state(value: object) -> tuple[int, int, int, int, int]:
    try:
        identity = (
            int(getattr(value, "st_dev")),
            int(getattr(value, "st_ino")),
            int(getattr(value, "st_mode")),
            int(getattr(value, "st_size")),
            int(getattr(value, "st_mtime_ns")),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise _UnsafeMetadata from error
    return identity


def _capture_ancestor_identities(
    root: Path,
) -> tuple[tuple[int, int, int], ...]:
    identities: list[tuple[int, int, int]] = []
    for ancestor in (root, *root.parents):
        value = os.lstat(ancestor)
        identity = _directory_identity(value)
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or _is_reparse(value)
        ):
            raise _UnsafeMetadata
        identities.append(identity)
    return tuple(identities)


def _directory_identity(value: object) -> tuple[int, int, int]:
    try:
        return (
            int(getattr(value, "st_dev")),
            int(getattr(value, "st_ino")),
            int(getattr(value, "st_mode")),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise _UnsafeMetadata from error


def _is_reparse(value: object) -> bool:
    if not _WINDOWS:
        return False
    try:
        attributes = getattr(value, "st_file_attributes")
        if attributes is None or _REPARSE_POINT is None:
            raise TypeError
        return bool(int(attributes) & int(_REPARSE_POINT))
    except Exception as error:
        raise _UnsafeMetadata from error
