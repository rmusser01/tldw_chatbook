"""Secure, bounded discovery of repository-authored project instructions."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from tldw_chatbook.Utils.filesystem_identity import (
    DirectoryIdentityError,
    directory_identity_from_stat,
)

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
class InstructionSourceMetadata:
    """Content-free source identity retained even when token-omitted."""

    relative_path: str
    scope: str
    byte_count: int


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
    startup_source_metadata: InstructionSourceMetadata | None = None


@dataclass(frozen=True, slots=True)
class NestedResolutionBatch:
    """Pinned nested sources and content-free terminal outcomes for one batch."""

    sources: tuple[InstructionSource, ...]
    outcomes: tuple[InstructionOutcome, ...]


@dataclass(frozen=True, slots=True)
class BindingRootIdentity:
    """Run-local selected-root identity pinned at dispatch construction."""

    canonical_root: Path = field(repr=False)
    ancestor_identities: tuple[tuple[int, int, int], ...] | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class InstructionPromotionSnapshot:
    """Content-bounded current state for one repository-instruction proposal."""

    binding_id: str
    binding_root: Path = field(repr=False)
    locator_fingerprint: str = field(repr=False)
    root_identity_digest: str = field(repr=False)
    target_relative_path: str
    expected_sha256: str | None = field(default=None, repr=False)
    expected_absent: bool = False
    effective_chain: tuple[tuple[str, InstructionKind, str], ...] = field(
        default=(), repr=False
    )
    effective_chain_digest: str = field(default="", repr=False)
    activation_revision: int = 0


class InstructionPromotionSnapshotError(RuntimeError):
    """Stable content-free refusal while reading a promotion target."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


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

    def snapshot_promotion_target(
        self,
        *,
        binding_id: str,
        binding_root: Path,
        locator_fingerprint: str,
        target_path: Path,
        activation_revision: int,
        max_bytes: int = 1024 * 1024,
    ) -> InstructionPromotionSnapshot:
        """Capture one eligible target and its currently applicable chain.

        The snapshot contains only paths and digests for the instruction chain;
        unrelated instruction bodies are never returned.
        """
        if not binding_id or not locator_fingerprint or activation_revision < 0:
            raise InstructionPromotionSnapshotError("authority_unavailable")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        root, expected_ancestors = _canonical_binding_root(binding_root)
        target = _safe_absolute(target_path)
        if expected_ancestors is None:
            raise InstructionPromotionSnapshotError("binding_changed")
        if (
            target is None
            or not target.is_relative_to(root)
            or target.name not in {"AGENTS.md", "AGENTS.override.md"}
        ):
            raise InstructionPromotionSnapshotError("ineligible_target")
        relative = target.relative_to(root)
        if not relative.parts or ".." in relative.parts:
            raise InstructionPromotionSnapshotError("ineligible_target")

        target_before = _read_promotion_target_state(
            root=root,
            target=target,
            expected_ancestors=expected_ancestors,
            max_bytes=max_bytes,
        )
        chain = _read_current_instruction_chain(
            root=root,
            target_directory=target.parent,
            expected_ancestors=expected_ancestors,
            max_bytes=max_bytes,
        )
        target_after = _read_promotion_target_state(
            root=root,
            target=target,
            expected_ancestors=expected_ancestors,
            max_bytes=max_bytes,
        )
        if target_before != target_after:
            raise InstructionPromotionSnapshotError("target_state_changed")
        try:
            if _capture_ancestor_identities(root) != expected_ancestors:
                raise InstructionPromotionSnapshotError("binding_changed")
        except OSError:
            raise InstructionPromotionSnapshotError("binding_changed") from None

        expected_sha256, expected_absent = target_before
        chain_metadata = tuple(
            (source.relative_path, source.kind, source.digest) for source in chain
        )
        return InstructionPromotionSnapshot(
            binding_id=binding_id,
            binding_root=root,
            locator_fingerprint=locator_fingerprint,
            root_identity_digest=_canonical_metadata_digest(expected_ancestors),
            target_relative_path=relative.as_posix(),
            expected_sha256=expected_sha256,
            expected_absent=expected_absent,
            effective_chain=chain_metadata,
            effective_chain_digest=_canonical_metadata_digest(chain_metadata),
            activation_revision=activation_revision,
        )

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

    def resolve_targets(
        self,
        binding_root: Path,
        targets: Sequence[Path],
        *,
        max_bytes: int,
        dispatch_started_wall_ns: int,
        pinned_by_canonical_path: Mapping[Path, InstructionSource],
        terminal_scopes: frozenset[str] = frozenset(),
        admission_bytes: int | None = None,
        expected_binding_identity: BindingRootIdentity | None = None,
    ) -> NestedResolutionBatch:
        """Resolve effective files on the union of root-to-target chains.

        ``targets`` are already-normalized directory scopes supplied by the
        path-aware tool owner. The binding root itself is excluded because its
        startup source is already pinned separately.

        Args:
            binding_root: Canonical selected instruction authority root.
            targets: Validated directory scopes required by the tool batch.
            max_bytes: Maximum raw bytes admitted across newly found sources.
            dispatch_started_wall_ns: Dispatch cutoff for stale-file checks.
            pinned_by_canonical_path: Sources already frozen for this dispatch.
            terminal_scopes: Scopes with a prior terminal no-content outcome.
            admission_bytes: Current cumulative ledger allowance. Defaults to
                ``max_bytes`` for standalone resolver calls.
            expected_binding_identity: Dispatch-owned selected-root identity;
                required when reusing any pinned source.

        Returns:
            Sources in broad-to-specific order plus content-free outcomes.

        Raises:
            ValueError: If ``max_bytes`` is negative.
        """
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if admission_bytes is not None and admission_bytes < 0:
            raise ValueError("admission_bytes must be non-negative")
        root, expected_root = _canonical_binding_root(binding_root)
        if pinned_by_canonical_path and expected_binding_identity is None:
            expected_root = None
        if expected_binding_identity is not None and (
            root != expected_binding_identity.canonical_root
            or expected_root is None
            or expected_root != expected_binding_identity.ancestor_identities
        ):
            expected_root = None
        if expected_root is None:
            return NestedResolutionBatch(
                (), (InstructionOutcome(".", ".", "resolution_failed"),)
            )

        directories: set[Path] = set()
        outcomes: list[InstructionOutcome] = []
        for target in targets:
            lexical = _safe_absolute(target)
            if lexical is None or not lexical.is_relative_to(root):
                outcomes.append(InstructionOutcome(".", ".", "resolution_failed"))
                continue
            current = root
            for part in lexical.relative_to(root).parts:
                current /= part
                try:
                    value = os.lstat(current)
                    if (
                        not stat.S_ISDIR(value.st_mode)
                        or stat.S_ISLNK(value.st_mode)
                        or _is_reparse(value)
                    ):
                        raise _UnsafeMetadata
                except FileNotFoundError:
                    break
                except (OSError, _UnsafeMetadata):
                    scope = current.relative_to(root).as_posix()
                    outcomes.append(
                        InstructionOutcome(
                            f"{scope}/AGENTS.md", scope, "resolution_failed"
                        )
                    )
                    break
                directories.add(current)

        found: list[tuple[InstructionSource, bool]] = []
        for directory in sorted(
            directories,
            key=lambda path: (len(path.relative_to(root).parts), path.as_posix()),
        ):
            scope = directory.relative_to(root).as_posix()
            if scope in terminal_scopes:
                continue
            result, was_pinned = _resolve_nested_directory(
                root=root,
                directory=directory,
                max_bytes=max_bytes,
                dispatch_started_wall_ns=dispatch_started_wall_ns,
                pinned_by_canonical_path=pinned_by_canonical_path,
                expected_binding_ancestors=expected_root,
            )
            if result.source is not None:
                found.append((result.source, was_pinned))
            elif result.outcome is not None:
                outcomes.append(result.outcome)

        remaining = max_bytes if admission_bytes is None else admission_bytes
        admitted: set[int] = {
            index for index, (_source, pinned) in enumerate(found) if pinned
        }
        new_indexes = [
            index for index, (_source, pinned) in enumerate(found) if not pinned
        ]
        for index in sorted(
            new_indexes,
            key=lambda item: (
                -len(Path(found[item][0].scope).parts),
                found[item][0].relative_path,
            ),
        ):
            source = found[index][0]
            if source.byte_count <= remaining:
                admitted.add(index)
                remaining -= source.byte_count
            else:
                outcomes.append(
                    InstructionOutcome(
                        source.relative_path,
                        source.scope,
                        "omitted_byte_budget",
                    )
                )

        sources = tuple(
            source for index, (source, _pinned) in enumerate(found) if index in admitted
        )
        return NestedResolutionBatch(
            sources=sources,
            outcomes=tuple(
                sorted(
                    dict.fromkeys(outcomes),
                    key=lambda item: (
                        len(Path(item.scope).parts),
                        item.relative_path,
                        item.code,
                    ),
                )
            ),
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


def _canonical_metadata_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise InstructionPromotionSnapshotError("snapshot_invalid") from error
    return hashlib.sha256(encoded).hexdigest()


def _read_promotion_target_state(
    *,
    root: Path,
    target: Path,
    expected_ancestors: tuple[tuple[int, int, int], ...],
    max_bytes: int,
) -> tuple[str | None, bool]:
    """Read one target through no-follow identity checks."""
    try:
        if _capture_ancestor_identities(root) != expected_ancestors:
            raise InstructionPromotionSnapshotError("binding_changed")
        before = os.lstat(target)
    except FileNotFoundError:
        try:
            if _capture_ancestor_identities(root) != expected_ancestors:
                raise InstructionPromotionSnapshotError("binding_changed")
        except OSError:
            raise InstructionPromotionSnapshotError("binding_changed") from None
        return None, True
    except InstructionPromotionSnapshotError:
        raise
    except OSError:
        raise InstructionPromotionSnapshotError("target_unavailable") from None
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or _is_reparse(before)
    ):
        raise InstructionPromotionSnapshotError("invalid_target")
    if before.st_size > max_bytes:
        raise InstructionPromotionSnapshotError("target_too_large")
    identity = _verified_state(before)
    flags = os.O_RDONLY | _CLOEXEC | _BINARY | _NONBLOCK
    if _NOFOLLOW is not None:
        flags |= _NOFOLLOW
    try:
        descriptor = os.open(target, flags)
        try:
            opened = os.fstat(descriptor)
            if _is_reparse(opened) or _verified_state(opened) != identity:
                raise _UnsafeMetadata
            raw = _bounded_read(descriptor, max_bytes + 1)
            finished = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = os.lstat(target)
        if (
            len(raw) > max_bytes
            or _is_reparse(after)
            or _verified_state(finished) != identity
            or _verified_state(after) != identity
            or _capture_ancestor_identities(root) != expected_ancestors
        ):
            raise _UnsafeMetadata
    except (OSError, _UnsafeMetadata):
        raise InstructionPromotionSnapshotError("target_state_changed") from None
    return hashlib.sha256(raw).hexdigest(), False


def _read_current_instruction_chain(
    *,
    root: Path,
    target_directory: Path,
    expected_ancestors: tuple[tuple[int, int, int], ...],
    max_bytes: int,
) -> tuple[InstructionSource, ...]:
    """Read the effective broad-to-specific chain for one target directory."""
    if not target_directory.is_relative_to(root):
        raise InstructionPromotionSnapshotError("outside_binding")
    directories = [root]
    current = root
    for part in target_directory.relative_to(root).parts:
        current /= part
        try:
            value = os.lstat(current)
        except OSError:
            raise InstructionPromotionSnapshotError("target_unavailable") from None
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or _is_reparse(value)
        ):
            raise InstructionPromotionSnapshotError("invalid_target")
        directories.append(current)

    cutoff = time.time_ns()
    sources: list[InstructionSource] = []
    for directory in directories:
        try:
            directory_ancestors = _capture_ancestor_identities(directory)
            depth = len(directory.relative_to(root).parts)
            if directory_ancestors[depth:] != expected_ancestors:
                raise _UnsafeMetadata
        except (OSError, RuntimeError, ValueError, _UnsafeMetadata):
            raise InstructionPromotionSnapshotError("binding_changed") from None
        scope = "." if directory == root else directory.relative_to(root).as_posix()
        prefix = "" if scope == "." else f"{scope}/"
        override = _read_candidate(
            root=directory,
            filename="AGENTS.override.md",
            kind="override",
            max_bytes=max_bytes,
            dispatch_started_wall_ns=cutoff,
            expected_ancestors=directory_ancestors,
            relative_path=f"{prefix}AGENTS.override.md",
            scope=scope,
        )
        result = override
        if override.fallback_condition is not None:
            result = _read_candidate(
                root=directory,
                filename="AGENTS.md",
                kind="standard",
                max_bytes=max_bytes,
                dispatch_started_wall_ns=cutoff,
                expected_ancestors=directory_ancestors,
                relative_path=f"{prefix}AGENTS.md",
                scope=scope,
            )
            rechecked = _read_candidate(
                root=directory,
                filename="AGENTS.override.md",
                kind="override",
                max_bytes=max_bytes,
                dispatch_started_wall_ns=cutoff,
                expected_ancestors=directory_ancestors,
                relative_path=f"{prefix}AGENTS.override.md",
                scope=scope,
            )
            if rechecked.fallback_condition != override.fallback_condition:
                raise InstructionPromotionSnapshotError("effective_chain_changed")
        if result.outcome is not None:
            raise InstructionPromotionSnapshotError(result.outcome.code)
        if result.source is not None:
            sources.append(result.source)
    return tuple(sources)


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


def capture_binding_root_identity(binding_root: Path) -> BindingRootIdentity:
    """Capture the selected root and ancestor identities for one dispatch.

    Args:
        binding_root: Canonical selected workspace root to pin.

    Returns:
        The lexical root plus its fail-closed ancestor identity chain. An
        unavailable chain is represented inside the returned value and makes
        later resolution ineligible.
    """
    root, ancestors = _canonical_binding_root(binding_root)
    return BindingRootIdentity(root, ancestors)


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
    relative_path: str | None = None,
    scope: str = ".",
) -> _ReadResult:
    path = root / filename
    displayed_path = relative_path or filename

    def outcome(code: InstructionOutcomeCode) -> InstructionOutcome:
        return InstructionOutcome(displayed_path, scope, code)

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
            relative_path=displayed_path,
            scope=scope,
            kind=kind,
            body=body,
            byte_count=len(raw),
            digest=hashlib.sha256(raw).hexdigest(),
        )
    )


def _resolve_nested_directory(
    *,
    root: Path,
    directory: Path,
    max_bytes: int,
    dispatch_started_wall_ns: int,
    pinned_by_canonical_path: Mapping[Path, InstructionSource],
    expected_binding_ancestors: tuple[tuple[int, int, int], ...],
) -> tuple[_ReadResult, bool]:
    scope = directory.relative_to(root).as_posix()
    override_path = directory / "AGENTS.override.md"
    standard_path = directory / "AGENTS.md"
    try:
        expected_ancestors = _capture_ancestor_identities(directory)
        depth = len(directory.relative_to(root).parts)
        if expected_ancestors[depth:] != expected_binding_ancestors:
            raise _UnsafeMetadata
    except (OSError, RuntimeError, ValueError, _UnsafeMetadata):
        return (
            _ReadResult(
                outcome=InstructionOutcome(
                    f"{scope}/AGENTS.md", scope, "resolution_failed"
                )
            ),
            False,
        )
    pinned_path = override_path
    pinned = pinned_by_canonical_path.get(pinned_path)
    if pinned is None:
        pinned_path = standard_path
        pinned = pinned_by_canonical_path.get(pinned_path)
    if pinned is not None:
        try:
            valid = _valid_pinned_source(
                root=root,
                directory=directory,
                pinned_path=pinned_path,
                source=pinned,
            )
            if (
                not valid
                or _capture_ancestor_identities(directory) != expected_ancestors
            ):
                raise _UnsafeMetadata
        except (OSError, RuntimeError, ValueError, _UnsafeMetadata):
            return (
                _ReadResult(
                    outcome=InstructionOutcome(
                        f"{scope}/AGENTS.md", scope, "resolution_failed"
                    )
                ),
                False,
            )
        return _ReadResult(source=pinned), True
    override_relative = f"{scope}/AGENTS.override.md"
    override = _read_candidate(
        root=directory,
        filename="AGENTS.override.md",
        kind="override",
        max_bytes=max_bytes,
        dispatch_started_wall_ns=dispatch_started_wall_ns,
        expected_ancestors=expected_ancestors,
        relative_path=override_relative,
        scope=scope,
    )
    result = override
    if override.fallback_condition is not None:
        result = _read_candidate(
            root=directory,
            filename="AGENTS.md",
            kind="standard",
            max_bytes=max_bytes,
            dispatch_started_wall_ns=dispatch_started_wall_ns,
            expected_ancestors=expected_ancestors,
            relative_path=f"{scope}/AGENTS.md",
            scope=scope,
        )
        rechecked_override = _read_candidate(
            root=directory,
            filename="AGENTS.override.md",
            kind="override",
            max_bytes=max_bytes,
            dispatch_started_wall_ns=dispatch_started_wall_ns,
            expected_ancestors=expected_ancestors,
            relative_path=override_relative,
            scope=scope,
        )
        if rechecked_override.fallback_condition != override.fallback_condition:
            result = _fallback_changed_result(
                rechecked_override,
                relative_path=override_relative,
                scope=scope,
            )
    return result, False


def _fallback_changed_result(
    rechecked: _ReadResult,
    *,
    relative_path: str = "AGENTS.override.md",
    scope: str = ".",
) -> _ReadResult:
    if rechecked.outcome is not None:
        return rechecked
    return _ReadResult(
        outcome=InstructionOutcome(relative_path, scope, "resolution_failed")
    )


def _valid_pinned_source(
    *,
    root: Path,
    directory: Path,
    pinned_path: Path,
    source: InstructionSource,
) -> bool:
    try:
        encoded = source.body.encode("utf-8")
    except (AttributeError, UnicodeEncodeError):
        return False
    content_matches = type(source.byte_count) is int and any(
        len(raw) == source.byte_count
        and hashlib.sha256(raw).hexdigest() == source.digest
        for raw in (encoded, b"\xef\xbb\xbf" + encoded)
    )
    expected_scope = directory.relative_to(root).as_posix()
    expected_relative = pinned_path.relative_to(root).as_posix()
    expected_kind: InstructionKind = (
        "override" if pinned_path.name == "AGENTS.override.md" else "standard"
    )
    return (
        pinned_path == source.canonical_path
        and source.canonical_path.is_absolute()
        and ".." not in source.canonical_path.parts
        and source.canonical_path.is_relative_to(root)
        and source.relative_path == expected_relative
        and source.scope == expected_scope
        and source.kind == expected_kind
        and content_matches
        and _safe_relative_label(source.relative_path)
        and _safe_relative_label(source.scope)
    )


def _safe_relative_label(value: str) -> bool:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        bool(value)
        and not posix.is_absolute()
        and not windows.is_absolute()
        and ".." not in posix.parts
        and ".." not in windows.parts
        and "\n" not in value
        and "\r" not in value
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
        try:
            identity = directory_identity_from_stat(value)
        except DirectoryIdentityError as error:
            raise _UnsafeMetadata from error
        if (
            not stat.S_ISDIR(value.st_mode)
            or stat.S_ISLNK(value.st_mode)
            or _is_reparse(value)
        ):
            raise _UnsafeMetadata
        identities.append((identity.device, identity.inode, identity.mode))
    return tuple(identities)


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
