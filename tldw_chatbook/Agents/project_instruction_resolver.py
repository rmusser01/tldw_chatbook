"""Secure, bounded discovery of repository-authored project instructions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from tldw_chatbook.Tools.remote_root_types import RemoteRoot
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
    binding_root: "Path | RemoteRoot" = field(repr=False)
    locator_fingerprint: str = field(repr=False)
    dispatch_started_wall_ns: int = field(repr=False)
    source: InstructionSource | None
    outcomes: tuple[InstructionOutcome, ...]
    excluded_dirs: frozenset[Path] = field(default=frozenset(), repr=False)
    #: Task 19: the executor-backed IO that produced a REMOTE candidate.
    #: Rides along (never compared, never rendered) so the run-local
    #: activation ledger resolves nested scopes through the same reader.
    remote_io: "RemoteInstructionIO | None" = field(
        default=None, compare=False, repr=False
    )


@dataclass(frozen=True, slots=True)
class InstructionChainDelivery:
    """Instruction digests and terminal outcomes delivered to one model chain."""

    source_digests: tuple[str, ...] = field(repr=False)
    outcomes: tuple[InstructionOutcome, ...]


@dataclass(frozen=True, slots=True)
class InstructionSnapshot:
    """Immutable project-instruction state for one Console dispatch."""

    binding_id: str
    binding_root: "Path | RemoteRoot" = field(repr=False)
    locator_fingerprint: str = field(repr=False)
    dispatch_started_wall_ns: int = field(repr=False)
    startup_source: InstructionSource | None
    global_outcomes: tuple[InstructionOutcome, ...]
    primary_delivery: InstructionChainDelivery
    warning_codes: tuple[str, ...]
    startup_source_metadata: InstructionSourceMetadata | None = None
    excluded_dirs: frozenset[Path] = field(default=frozenset(), repr=False)
    #: Task 19: the executor-backed IO for a REMOTE binding root; the
    #: activation ledger builds its nested resolver from this slot.
    remote_io: "RemoteInstructionIO | None" = field(
        default=None, compare=False, repr=False
    )


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


def path_is_excluded(path: Path, excluded: frozenset[Path]) -> bool:
    """True when ``path`` is or lies under a workspace-excluded directory.

    Deny-side comparison is COMPONENT-WISE CASEFOLDED, mirroring
    ``Utils.sensitive_paths._compare_key`` (TASK-19800): macOS and Windows
    filesystems are case-insensitive by default and ``Path.resolve()``
    preserves the caller's spelling, so a Settings-typed ``Docs`` exclusion
    must also exclude an on-disk ``docs/AGENTS.md``. Folding may
    over-refuse a genuinely distinct sibling on a case-sensitive
    filesystem -- the cheap direction for a denylist. Never reuse the
    folded form for confinement checks: folding loosens those (see
    ``_compare_key`` for why).

    Args:
        path: Candidate path to test (file or directory).
        excluded: Absolute resolved exclusion entries (files or
            directories) to compare against.

    Returns:
        True when ``path`` is or lies under any excluded entry; False when
        it does not or ``excluded`` is empty.
    """
    if not excluded:
        return False
    try:
        resolved_key = tuple(
            part.casefold() for part in path.resolve(strict=False).parts
        )
    except Exception:  # noqa: BLE001 - unresolvable candidate paths fail closed
        return True
    for entry in excluded:
        try:
            entry_key = tuple(
                part.casefold() for part in entry.resolve(strict=False).parts
            )
        except Exception:  # noqa: BLE001 - broad like ``_resolved`` (TASK-847):
            # symlink loops raise OSError/RuntimeError, embedded NULs raise
            # ValueError. Skipping the ENTRY is safe because an actual tool
            # touching such a path still fails closed via ``is_sensitive_path``,
            # which refuses any path it cannot resolve.
            continue
        if entry_key == resolved_key[: len(entry_key)]:
            return True
    return False


class ProjectInstructionResolver:
    """Resolve only the selected binding root's effective instruction file."""

    #: Class-level default so subclasses that override ``__init__``
    #: without chaining (test doubles do) keep local-IO behavior.
    _remote_io: "RemoteInstructionIO | None" = None

    def __init__(self, remote_io: "RemoteInstructionIO | None" = None) -> None:
        """Configure the resolver's IO strategy.

        Args:
            remote_io: Task 19's executor-backed IO for REMOTE binding
                roots. ``None`` (the default) keeps the laptop-fd
                behavior byte-identical: every pre-Task-19 call site
                constructs ``ProjectInstructionResolver()`` and only
                ever passes laptop ``Path`` roots.
        """
        self._remote_io = remote_io

    def snapshot_promotion_target(
        self,
        *,
        binding_id: str,
        binding_root: Path,
        locator_fingerprint: str,
        target_path: Path,
        activation_revision: int,
        max_bytes: int = 1024 * 1024,
        excluded_dirs: frozenset[Path] = frozenset(),
    ) -> InstructionPromotionSnapshot:
        """Capture one eligible target and its currently applicable chain.

        The snapshot contains only paths and digests for the instruction chain;
        unrelated instruction bodies are never returned.

        Args:
            excluded_dirs: Absolute resolved paths of user-excluded
                files/directories; excluded instruction candidates are
                treated as absent.
        """
        if not binding_id or not locator_fingerprint or activation_revision < 0:
            raise InstructionPromotionSnapshotError("authority_unavailable")
        if isinstance(binding_root, RemoteRoot):
            # Task 19 boundary: promotion (model-PROPOSED instruction
            # files) stays local-only in v1 — its fd-verified before/
            # after chain capture has no remote analogue yet. Refusing
            # fail-closed is safe: nothing was promoted, nothing changes.
            raise InstructionPromotionSnapshotError("ineligible_target")
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
        if path_is_excluded(target, excluded_dirs):
            raise InstructionPromotionSnapshotError("ineligible_target")
        try:
            from tldw_chatbook.Utils.path_validation import validate_path

            validated_parent = validate_path(
                target.parent,
                root,
                redact_paths=True,
                allow_hidden=True,
            )
        except (OSError, RuntimeError, ValueError):
            raise InstructionPromotionSnapshotError("ineligible_target") from None
        # Validate the parent, then restore the allowlisted final filename so
        # the descriptor reader below can still reject a final-component
        # symlink rather than silently following it.
        target = validated_parent / target.name
        relative = target.relative_to(root)

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
            excluded_dirs=excluded_dirs,
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
        excluded_dirs: frozenset[Path] = frozenset(),
    ) -> StartupInstructionCandidate:
        """Resolve and securely pin the effective binding-root instructions.

        Args:
            binding_id: Selected workspace binding identity.
            binding_root: Canonical selected workspace locator.
            locator_fingerprint: Fingerprint captured when the binding was selected.
            max_bytes: Maximum raw bytes admitted for the startup source.
            dispatch_started_wall_ns: Dispatch wall-clock cutoff in nanoseconds.
            excluded_dirs: Absolute resolved paths of user-excluded
                files/directories; excluded instruction candidates are
                treated as absent.

        Returns:
            A byte-admitted candidate containing at most one root source.

        Raises:
            ValueError: If ``max_bytes`` is negative.
        """
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")

        root, expected_ancestors = self._canonical_root(binding_root)
        if expected_ancestors is None:
            return StartupInstructionCandidate(
                binding_id=binding_id,
                binding_root=(
                    binding_root if isinstance(binding_root, RemoteRoot) else root
                ),
                locator_fingerprint=locator_fingerprint,
                dispatch_started_wall_ns=dispatch_started_wall_ns,
                source=None,
                outcomes=(InstructionOutcome(".", ".", "resolution_failed"),),
                excluded_dirs=excluded_dirs,
                remote_io=self._remote_io,
            )

        def read(filename: str, kind: InstructionKind) -> _ReadResult:
            return self._read_admitted(
                root=root,
                filename=filename,
                kind=kind,
                max_bytes=max_bytes,
                dispatch_started_wall_ns=dispatch_started_wall_ns,
                expected_ancestors=expected_ancestors,
                excluded_dirs=excluded_dirs,
                remote=isinstance(binding_root, RemoteRoot),
            )

        override = read("AGENTS.override.md", "override")
        result = override
        if override.fallback_condition is not None:
            result = read("AGENTS.md", "standard")
            rechecked_override = read("AGENTS.override.md", "override")
            if rechecked_override.fallback_condition != override.fallback_condition:
                result = _fallback_changed_result(rechecked_override)

        return StartupInstructionCandidate(
            binding_id=binding_id,
            binding_root=(
                binding_root if isinstance(binding_root, RemoteRoot) else root
            ),
            locator_fingerprint=locator_fingerprint,
            dispatch_started_wall_ns=dispatch_started_wall_ns,
            source=result.source,
            outcomes=(result.outcome,) if result.outcome else (),
            excluded_dirs=excluded_dirs,
            remote_io=self._remote_io,
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
        excluded_dirs: frozenset[Path] = frozenset(),
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
            excluded_dirs: Absolute resolved paths of user-excluded
                files/directories; excluded instruction candidates are
                treated as absent.

        Returns:
            Sources in broad-to-specific order plus content-free outcomes.

        Raises:
            ValueError: If ``max_bytes`` is negative.
        """
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if admission_bytes is not None and admission_bytes < 0:
            raise ValueError("admission_bytes must be non-negative")
        root, expected_root = self._canonical_root(binding_root)
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
            if self._remote_io is not None and isinstance(binding_root, RemoteRoot):
                # Remote walk: worker stats replace per-component laptop
                # lstat (see RemoteInstructionIO.walk_component_chain for
                # the documented weaker-granularity analogue).
                failure = self._remote_io.walk_component_chain(
                    root, lexical, excluded_dirs, directories
                )
                if failure is not None:
                    outcomes.append(failure)
                continue
            current = root
            for part in lexical.relative_to(root).parts:
                current /= part
                if path_is_excluded(current, excluded_dirs):
                    # Excluded directories and everything under them are
                    # treated exactly like nonexistent scopes: no lstat, no
                    # AGENTS.md read, no outcome.
                    break
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
                excluded_dirs=excluded_dirs,
                remote_io=(
                    self._remote_io
                    if isinstance(binding_root, RemoteRoot)
                    else None
                ),
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

    # -- Task 19: IO-strategy dispatch -------------------------------------

    def _canonical_root(
        self, binding_root: "Path | RemoteRoot"
    ) -> tuple[Path, tuple[tuple[int, int, int], ...] | None]:
        """Canonical binding root via the configured IO strategy.

        Local roots keep ``_canonical_binding_root`` verbatim. A
        :class:`~tldw_chatbook.Tools.remote_root_types.RemoteRoot` without
        a remote IO strategy fails CLOSED (``None`` ancestors) instead of
        touching the laptop's filesystem with a remote path.
        """
        if isinstance(binding_root, RemoteRoot):
            if self._remote_io is None:
                return Path(str(binding_root.root)), None
            return self._remote_io.canonical_root(binding_root)
        return _canonical_binding_root(binding_root)

    def _read_admitted(
        self,
        *,
        root: Path,
        filename: str,
        kind: InstructionKind,
        max_bytes: int,
        dispatch_started_wall_ns: int,
        expected_ancestors: tuple[tuple[int, int, int], ...],
        relative_path: str | None = None,
        scope: str = ".",
        excluded_dirs: frozenset[Path] = frozenset(),
        remote: bool = False,
    ) -> _ReadResult:
        """Read one candidate through the configured IO strategy.

        The LOCAL branch is the pre-Task-19 read verbatim (exclusion
        check + fd-pinned read). The REMOTE branch (``remote`` — set for
        :class:`RemoteRoot` dispatches) routes through the executor-backed
        reader, which applies the same exclusion and admission rules
        worker-side (see ``RemoteInstructionIO``).
        """
        if remote and self._remote_io is not None:
            return self._remote_io.read_candidate(
                root=root,
                filename=filename,
                kind=kind,
                max_bytes=max_bytes,
                dispatch_started_wall_ns=dispatch_started_wall_ns,
                relative_path=relative_path,
                scope=scope,
                excluded_dirs=excluded_dirs,
            )
        # Workspace-excluded candidates are skipped exactly like missing
        # files: never lstat'ed, opened, or admitted.
        if path_is_excluded(root / filename, excluded_dirs):
            return _ReadResult(fallback_condition=_FallbackCondition("absent"))
        return _read_candidate(
            root=root,
            filename=filename,
            kind=kind,
            max_bytes=max_bytes,
            dispatch_started_wall_ns=dispatch_started_wall_ns,
            expected_ancestors=expected_ancestors,
            relative_path=relative_path,
            scope=scope,
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
    excluded_dirs: frozenset[Path] = frozenset(),
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

        def read(filename: str, kind: InstructionKind) -> _ReadResult:
            # Workspace-excluded candidates are skipped exactly like missing
            # files: never lstat'ed, opened, or added to the chain.
            if path_is_excluded(directory / filename, excluded_dirs):
                return _ReadResult(fallback_condition=_FallbackCondition("absent"))
            return _read_candidate(
                root=directory,
                filename=filename,
                kind=kind,
                max_bytes=max_bytes,
                dispatch_started_wall_ns=cutoff,
                expected_ancestors=directory_ancestors,
                relative_path=f"{prefix}{filename}",
                scope=scope,
            )

        override = read("AGENTS.override.md", "override")
        result = override
        if override.fallback_condition is not None:
            result = read("AGENTS.md", "standard")
            rechecked = read("AGENTS.override.md", "override")
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


def capture_binding_root_identity(
    binding_root: "Path | RemoteRoot",
    *,
    remote_io: "RemoteInstructionIO | None" = None,
) -> BindingRootIdentity:
    """Capture the selected root and ancestor identities for one dispatch.

    Args:
        binding_root: Canonical selected workspace root to pin. A
            :class:`~tldw_chatbook.Tools.remote_root_types.RemoteRoot`
            requires ``remote_io``; its identity is the WORKER-REPORTED
            ping chain (the remote analogue of the laptop ancestor
            capture). A remote root without a reader fails closed.
        remote_io: Task 19's executor-backed IO for remote roots.

    Returns:
        The lexical root plus its fail-closed ancestor identity chain. An
        unavailable chain is represented inside the returned value and makes
        later resolution ineligible.
    """
    if isinstance(binding_root, RemoteRoot):
        if remote_io is None:
            return BindingRootIdentity(Path(str(binding_root.root)), None)
        root, ancestors = remote_io.canonical_root(binding_root)
        return BindingRootIdentity(root, ancestors)
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
    excluded_dirs: frozenset[Path] = frozenset(),
    remote_io: "RemoteInstructionIO | None" = None,
) -> tuple[_ReadResult, bool]:
    scope = directory.relative_to(root).as_posix()
    override_path = directory / "AGENTS.override.md"
    standard_path = directory / "AGENTS.md"
    # Workspace-excluded candidate files are skipped exactly like missing
    # ones: never lstat'ed, opened, or admitted -- the sibling candidate in
    # the same directory still resolves normally. Remote entries are the
    # Task 17 raw RELATIVE paths and match lexically against the
    # candidates' root-relative form.
    if remote_io is not None:
        override_excluded = remote_io.candidate_is_excluded(
            directory, "AGENTS.override.md", excluded_dirs
        )
        standard_excluded = remote_io.candidate_is_excluded(
            directory, "AGENTS.md", excluded_dirs
        )
    else:
        override_excluded = path_is_excluded(override_path, excluded_dirs)
        standard_excluded = path_is_excluded(standard_path, excluded_dirs)
    if remote_io is None:
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
    else:
        # Remote: per-directory (st_dev, st_ino) identities would need a
        # new wire op. The pinged ROOT chain above plus the worker's
        # per-call root pin (every stat/read below re-validates the whole
        # chain server-side before touching the filesystem) are the
        # documented weaker-granularity analogue.
        expected_ancestors = expected_binding_ancestors
    pinned_path = override_path
    pinned = None if override_excluded else pinned_by_canonical_path.get(pinned_path)
    if pinned is None:
        pinned_path = standard_path
        pinned = None if standard_excluded else pinned_by_canonical_path.get(pinned_path)
    if pinned is not None:
        try:
            valid = _valid_pinned_source(
                root=root,
                directory=directory,
                pinned_path=pinned_path,
                source=pinned,
            )
            if remote_io is None and (
                not valid
                or _capture_ancestor_identities(directory) != expected_ancestors
            ):
                raise _UnsafeMetadata
            if remote_io is not None and not valid:
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

    def read(
        filename: str, kind: InstructionKind, relative_path: str, excluded: bool
    ) -> _ReadResult:
        # Workspace-excluded candidates are skipped exactly like missing
        # files: never lstat'ed, opened, or admitted.
        if excluded:
            return _ReadResult(fallback_condition=_FallbackCondition("absent"))
        if remote_io is not None:
            return remote_io.read_candidate(
                root=directory,
                filename=filename,
                kind=kind,
                max_bytes=max_bytes,
                dispatch_started_wall_ns=dispatch_started_wall_ns,
                relative_path=relative_path,
                scope=scope,
            )
        return _read_candidate(
            root=directory,
            filename=filename,
            kind=kind,
            max_bytes=max_bytes,
            dispatch_started_wall_ns=dispatch_started_wall_ns,
            expected_ancestors=expected_ancestors,
            relative_path=relative_path,
            scope=scope,
        )

    override = read(
        "AGENTS.override.md", "override", override_relative, override_excluded
    )
    result = override
    if override.fallback_condition is not None:
        result = read(
            "AGENTS.md", "standard", f"{scope}/AGENTS.md", standard_excluded
        )
        rechecked_override = read(
            "AGENTS.override.md", "override", override_relative, override_excluded
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


# ---------------------------------------------------------------------------
# Task 19 (Phase 4b): executor-backed IO for remote binding roots
# ---------------------------------------------------------------------------

#: One rendered fs_read line: ``<1-based line number>\t<content>``.
_NUMBERED_LINE_PATTERN = re.compile(r"\A([0-9]+)\t(.*)\Z", re.DOTALL)

#: Worker render markers (see the bundle's ``_read_relative_file``).
_TRUNCATION_MARKER = "… [truncated]"
_EMPTY_FILE_NOTICE = "(empty file)"
_PAST_END_PREFIX = "(offset "

#: Hard ceiling on fs_read pages per candidate: each page renders at most
#: ``MAX_READ_CHARS`` (32 KiB) of content, so this bounds a runaway pager
#: far above the 1 MiB resolver byte ceiling before failing closed.
_MAX_FS_READ_PAGES = 1024


class _RemoteOpFailure(Exception):
    """One executor operation failure, with its typed wire code."""

    def __init__(self, code: str, text: str) -> None:
        self.code = str(code)
        self.text = str(text)
        super().__init__(self.text)


@dataclass(frozen=True, slots=True)
class _RemoteRead:
    """Classified outcome of one remote candidate read."""

    kind: Literal["ok", "absent", "omitted", "invalid", "failed"]
    body: str = ""
    digest: str = ""
    size: int = -1


@dataclass(frozen=True, slots=True)
class _RemoteStat:
    """Parsed ``stat_path`` view of one remote path."""

    kind: Literal["file", "directory", "other"]
    size: int
    modified_ns: int


def remote_path_is_excluded(
    relative_posix: str, excluded: frozenset[Path]
) -> bool:
    """Lexical deny-side exclusion match for one ROOT-RELATIVE remote path.

    Remote exclusion entries (Task 17) are the binding's RAW relative
    paths — the remote owns the filesystem, so there is nothing to
    resolve on the laptop. Matching mirrors ``path_is_excluded``'s
    component-wise CASEFOLDED prefix compare (deny-side only; folding may
    over-refuse, never under-refuse).
    """
    if not excluded:
        return False
    parts = tuple(
        part.casefold() for part in PurePosixPath(relative_posix).parts
    )
    for entry in excluded:
        if entry.is_absolute() or ".." in entry.parts:
            continue
        entry_parts = tuple(part.casefold() for part in entry.parts)
        if not entry_parts:
            # An exclusion of "." covers the whole binding root.
            return True
        if parts[: len(entry_parts)] == entry_parts:
            return True
    return False


class RemoteInstructionIO:
    """IO strategy that reads instruction candidates over the executor.

    LOCAL analogue (what this class replaces for remote roots, per
    ADR-068/069 and the spec section "AGENTS.md from remote"):

    * fd-pinned no-follow read + before/after identity re-checks
      → the WORKER-SIDE PIN: every dispatched op re-validates the pinged
      root identity chain server-side before touching the filesystem,
      and ``fs_read``'s CAS stamps (sha256 + size over the RAW bytes) are
      computed from ONE worker-side read, so the rendered body and its
      raw-byte identity are never a torn pair. A post-read ``stat_path``
      cross-checks size and freshness.
    * ancestor identity capture per directory
      → the ping chain pins the ROOT and its ancestors; per-subdirectory
      ``(st_dev, st_ino)`` identities would need a new wire op, so the
      reader uses the root chain + the candidate's own stat — a
      deliberately WEAKER granularity, bounded by worker confinement
      (every read stays inside the pinned root) and by the per-call root
      pin (a swapped root refuses every later op).

    Documented divergences from the local fd reader (all fail-safe or
    content-preserving directions, none of which weaken the untrusted-
    content posture — remote AGENTS.md never gains trust):

    * ``fs_read`` renders LINE-NUMBERED text: the admitted body is the
      reconstruction (line numbers stripped). ``\r\n`` and exotic line
      separators normalize to ``\n``; a UTF-8 BOM is stripped like the
      local ``utf-8-sig`` decode. ``byte_count``/``digest`` cover the
      ADMITTED body (self-consistent for the ledger, receipts and pinned
      re-validation); the worker-reported raw stamps pin change
      detection during the read itself.
    * Worker decode is lenient (U+FFFD replacement) where the local
      reader refuses invalid UTF-8; the binary (NUL) sniff still refuses
      binary files with ``invalid``. Unreadable-in-any-form candidates
      surface as content-free outcomes exactly like the local reader.
    * A symlinked candidate escaping the root is refused by worker
      containment as ``file not found`` (absent → fallback may proceed),
      where the local reader records ``invalid`` (fallback suppressed).
      Either way only worker-confined content is ever admitted.
    * The stale check compares the REMOTE file's ``mtime_ns`` (worker
      stat) against the LAPTOP dispatch clock, so host clock skew of
      ±N seconds shifts the cutoff by the same N — a bounded, benign
      divergence of the local same-clock comparison.

    Any executor failure — transport, pin, protocol — is source-closed
    (``resolution_failed``): per ADR-069's prep-failure posture an
    unreachable remote yields a content-free warning and the dispatch
    proceeds. NOTHING in this class raises past the resolver.
    """

    __slots__ = ("_ancestors", "_binding_id", "_executor", "_payload", "_root")

    def __init__(self, executor: Any, *, binding_id: str = "") -> None:
        """Bind one workspace executor.

        Args:
            executor: Duck-typed executor with ``ping()`` and
                ``execute(tool, args, *, intent)`` — both the loopback
                ``RemoteWorkspaceToolExecutor`` (dict results) and the
                controller's dispatch adapter (str results) are accepted.
            binding_id: Content-free diagnostics label.
        """
        self._executor = executor
        self._binding_id = str(binding_id)
        self._root: Path | None = None
        self._ancestors: tuple[tuple[int, int, int], ...] | None = None
        self._payload: Mapping[str, Any] | None = None

    # -- identity -----------------------------------------------------------

    @property
    def executor(self) -> Any:
        """The bound executor (test/diagnostics read-only access)."""
        return self._executor

    def ping_payload(self) -> Mapping[str, Any]:
        """The cached ping payload, capturing it on first use."""
        if self._payload is None:
            self._payload = self._executor.ping()
        return self._payload

    def canonical_root(
        self, descriptor: RemoteRoot
    ) -> tuple[Path, tuple[tuple[int, int, int], ...] | None]:
        """Pin the remote root's canonical path and ancestor chain (ping).

        Success is cached for the reader's lifetime: the per-op worker
        root pin is the LIVE re-validation between calls (a retargeted
        root refuses every later op), so re-pinging per call would only
        add a round trip. Failure is NOT cached — the next resolution
        re-attempts exactly like the local per-call capture.
        """
        if self._root is None:
            try:
                chain = self.ping_payload()["identity_chain"]
                ancestors = tuple(
                    (int(entry[1]), int(entry[2]), int(entry[3]))
                    for entry in chain
                )
                canonical = Path(str(chain[0][0]))
            except Exception:  # noqa: BLE001 - any ping failure is closed
                return Path(str(descriptor.root)), None
            self._root = canonical
            self._ancestors = ancestors
        return self._root, self._ancestors

    # -- relative-path helpers ----------------------------------------------

    def _relative(self, directory: Path) -> str:
        """Root-relative POSIX form of one opaque directory path."""
        if self._root is None or directory == self._root:
            return ""
        try:
            relative = directory.relative_to(self._root)
        except ValueError:
            return ""
        return relative.as_posix()

    def candidate_rel(self, directory: Path, filename: str) -> str:
        """The worker-facing root-relative candidate path."""
        directory_rel = self._relative(directory)
        return f"{directory_rel}/{filename}" if directory_rel else filename

    def candidate_is_excluded(
        self, directory: Path, filename: str, excluded: frozenset[Path]
    ) -> bool:
        """Lexical exclusion check for one remote candidate."""
        return remote_path_is_excluded(
            self.candidate_rel(directory, filename), excluded
        )

    # -- reads ---------------------------------------------------------------

    def read_candidate(
        self,
        *,
        root: Path,
        filename: str,
        kind: InstructionKind,
        max_bytes: int,
        dispatch_started_wall_ns: int,
        relative_path: str | None = None,
        scope: str = ".",
        excluded_dirs: frozenset[Path] = frozenset(),
    ) -> _ReadResult:
        """Read one candidate file through the executor (see class doc)."""
        candidate_rel = self.candidate_rel(root, filename)
        displayed_path = relative_path or filename

        def outcome(code: InstructionOutcomeCode) -> InstructionOutcome:
            return InstructionOutcome(displayed_path, scope, code)

        if remote_path_is_excluded(candidate_rel, excluded_dirs):
            return _ReadResult(fallback_condition=_FallbackCondition("absent"))

        read = self._fs_read_all(candidate_rel, max_bytes=max_bytes)
        if read.kind == "absent":
            return _ReadResult(fallback_condition=_FallbackCondition("absent"))
        if read.kind == "omitted":
            return _ReadResult(outcome=outcome("omitted_byte_budget"))
        if read.kind == "invalid":
            return _ReadResult(outcome=outcome("invalid"))
        if read.kind == "failed":
            return _ReadResult(outcome=outcome("resolution_failed"))

        stated = self._stat(candidate_rel)
        if not isinstance(stated, _RemoteStat):
            # Readable a moment ago, unstattable now: treat as changed
            # under us (the local reader's post-read lstat failure
            # analogue) — never as absent.
            return _ReadResult(outcome=outcome("resolution_failed"))
        if stated.kind != "file":
            return _ReadResult(outcome=outcome("invalid"))
        if stated.size != read.size:
            return _ReadResult(outcome=outcome("resolution_failed"))
        if stated.modified_ns > dispatch_started_wall_ns:
            return _ReadResult(outcome=outcome("stale"))
        if read.size > max_bytes:
            return _ReadResult(outcome=outcome("omitted_byte_budget"))

        body = read.body.removeprefix("\ufeff")
        encoded = body.encode("utf-8", errors="strict")
        if len(encoded) > max_bytes:
            return _ReadResult(outcome=outcome("omitted_byte_budget"))
        if not body.strip():
            return _ReadResult(
                fallback_condition=_FallbackCondition(
                    "empty",
                    file_identity=(
                        stated.modified_ns,
                        stated.size,
                        0,
                        0,
                        0,
                    ),
                    digest=read.digest,
                )
            )
        return _ReadResult(
            source=InstructionSource(
                canonical_path=root / filename,
                relative_path=displayed_path,
                scope=scope,
                kind=kind,
                body=body,
                byte_count=len(encoded),
                digest=hashlib.sha256(encoded).hexdigest(),
            )
        )

    # -- directory walk -------------------------------------------------------

    def walk_component_chain(
        self,
        root: Path,
        lexical: Path,
        excluded_dirs: frozenset[Path],
        directories: set[Path],
    ) -> InstructionOutcome | None:
        """Walk one target's directory chain via worker stats.

        Adds every EXISTING directory component to ``directories``.
        Returns ``None`` on benign stops (missing or excluded scope —
        mirroring the local walk's silent ``FileNotFoundError`` break) or
        a content-free ``resolution_failed`` outcome when a component is
        refused. Documented weaker granularity: a symlinked directory
        pointing INSIDE the root is followed (worker parity for fs ops);
        one escaping the root makes every read under it refuse, so no
        out-of-root content can activate.
        """
        current = root
        for part in lexical.relative_to(root).parts:
            current = current / part
            component_rel = self._relative(current)
            if remote_path_is_excluded(component_rel, excluded_dirs):
                return None
            stated = self._stat(component_rel)
            if not isinstance(stated, _RemoteStat):
                # Missing and unstatable are indistinguishable through
                # stat_path; the walk treats both as a nonexistent scope
                # (no activation, never wrong activation).
                return None
            if stated.kind != "directory":
                return InstructionOutcome(
                    f"{component_rel}/AGENTS.md",
                    component_rel,
                    "resolution_failed",
                )
            directories.add(current)
        return None

    # -- executor plumbing -----------------------------------------------------

    def _execute(self, tool: str, args: dict[str, Any]) -> str:
        """Dispatch one read op; normalize the result to its text form."""
        try:
            result = self._executor.execute(tool, args, intent="read")
        except Exception as error:  # noqa: BLE001 - any failure is closed
            code = str(getattr(error, "code", "") or "resolution_failed")
            raise _RemoteOpFailure(code, str(error)) from None
        if isinstance(result, Mapping):
            result = result.get("result")
        if not isinstance(result, str):
            raise _RemoteOpFailure("protocol_failure", "missing result text")
        return result

    def _fs_read_all(self, candidate_rel: str, *, max_bytes: int) -> _RemoteRead:
        """Read one candidate fully, paging past the render cap."""
        from tldw_chatbook.Tools.remote_workspace_executor import (
            split_fs_read_result,
        )

        collected: list[str] = []
        offset = 1
        for _page in range(_MAX_FS_READ_PAGES):
            try:
                result = self._execute(
                    "fs_read", {"path": candidate_rel, "offset": offset}
                )
            except _RemoteOpFailure as failure:
                return self._classify_read_failure(failure)
            split = split_fs_read_result(result)
            if split is None:
                return _RemoteRead("failed")
            body_text, digest, size = split
            if size > max_bytes:
                # The worker stamp reports the RAW size: cap before any
                # further paging (byte budget precedes reconstruction).
                return _RemoteRead("omitted", digest=digest, size=size)
            if body_text == _EMPTY_FILE_NOTICE:
                return _RemoteRead("ok", body="", digest=digest, size=size)
            page_offset = offset
            page = _parse_fs_read_page(body_text, expected_start=offset)
            if page is None:
                return _RemoteRead("failed")
            page_lines, next_number, truncated, past_end = page
            collected.extend(page_lines)
            if past_end or not truncated:
                return _RemoteRead(
                    "ok", body="\n".join(collected), digest=digest, size=size
                )
            # Truncation may have cut the render MID-LINE: the trailing
            # fragment (unparseable) was never collected, but the LAST
            # parseable line can also be content-cut with its number
            # intact — drop it too and resume from its line number.
            if page_lines:
                collected.pop()
                offset = next_number - 1
            else:
                offset = next_number
            if offset <= page_offset:
                # No forward progress (a single line too long for the
                # worker's render cap): the candidate is unrenderable —
                # fail closed instead of looping.
                return _RemoteRead("omitted")
        return _RemoteRead("omitted")

    @staticmethod
    def _classify_read_failure(failure: _RemoteOpFailure) -> _RemoteRead:
        """Map one typed fs_read refusal onto the local outcome codes."""
        if failure.code == "tool_failure":
            if failure.text.startswith("file not found"):
                return _RemoteRead("absent")
            if "too large to read" in failure.text:
                return _RemoteRead("omitted")
            if "appears to be binary" in failure.text:
                return _RemoteRead("invalid")
        return _RemoteRead("failed")

    def _stat(self, candidate_rel: str) -> _RemoteStat | None:
        """Stat one remote path; ``None`` when unstatably absent/refused."""
        try:
            result = self._execute("stat_path", {"path": candidate_rel})
        except _RemoteOpFailure:
            return None
        fields: dict[str, str] = {}
        for line in result.split("\n"):
            key, separator, value = line.partition(": ")
            if separator:
                fields[key] = value
        try:
            kind = fields["type"]
            if kind not in ("file", "directory", "other"):
                return None
            return _RemoteStat(
                kind=kind,  # type: ignore[arg-type]
                size=int(fields["size"]),
                modified_ns=int(fields["modified_ns"]),
            )
        except (KeyError, ValueError):
            return None


def _parse_fs_read_page(
    body_text: str, *, expected_start: int
) -> tuple[list[str], int, bool, bool] | None:
    """Parse one rendered fs_read page.

    Returns ``(lines, next_line_number, truncated, past_end)``, or
    ``None`` on a contract violation (a line that is neither numbered-
    with-sequential-index nor a recognized marker — fail closed). The
    render cap can cut a page MID-LINE, leaving a trailing fragment that
    is not a full numbered line; such a fragment is tolerated ONLY
    directly before the truncation marker and dropped — the pager
    re-reads from ``next_line_number``. A file line whose CONTENT is
    literally ``… [truncated]`` forces re-paging that converges to the
    page cap and then fails closed: omission, never wrong content.
    """
    lines: list[str] = []
    expected = expected_start
    truncated = False
    past_end = False
    raw = body_text.split("\n") if body_text else []
    index = 0
    while index < len(raw):
        line = raw[index]
        if line == _TRUNCATION_MARKER:
            truncated = True
            index += 1
            continue
        if line.startswith(_PAST_END_PREFIX) and line.endswith(")"):
            past_end = True
            index += 1
            continue
        match = _NUMBERED_LINE_PATTERN.match(line)
        if match is not None and int(match.group(1)) == expected:
            lines.append(match.group(2))
            expected += 1
            index += 1
            continue
        if index + 1 < len(raw) and raw[index + 1] == _TRUNCATION_MARKER:
            # The mid-line fragment the cap cut: drop it, resume from
            # its (implicit) line number on the next page.
            truncated = True
            index += 2
            continue
        return None
    return lines, expected, truncated, past_end
