"""Read-only migration of legacy Notes sync evidence into paused candidates."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path

from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncRootRecord,
)
from tldw_chatbook.Notes.notes_sync_filesystem import (
    NotesSyncFilesystemError,
    validate_sync_root_admission,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootState,
    NotesSyncSerializationProfile,
    normalize_notes_sync_relative_path,
    validate_notes_sync_digest,
    validate_notes_sync_opaque_id,
    validate_notes_sync_reason_code,
)
from tldw_chatbook.Notes.notes_sync_reconciler import (
    ReconciliationInput,
    ReconciliationPlan,
    assert_review_current,
    plan_reconciliation,
)
from tldw_chatbook.Utils.sensitive_paths import (
    find_root_binding_conflict,
    resolve_sensitive_context,
)


LEGACY_MIGRATION_REPORT_LIMIT = 200
# TASK-21112: the one-time first-boot evidence reads over chachanotes.db must
# be bounded. The limit is far beyond any observed legacy sync inventory; a
# store that exceeds it fails the snapshot with an explicit reason instead of
# silently migrating a truncated (and therefore wrong) subset.
LEGACY_SNAPSHOT_EVIDENCE_LIMIT = 10_000
_LEGACY_POLICY_KEYS = (
    "auto_sync_enabled",
    "sync_on_close",
    "conflict_resolution",
    "sync_direction",
)
_VALID_CONFLICT_POLICIES = frozenset({"ask", "disk_wins", "db_wins", "newer_wins"})
_VALID_DIRECTIONS = frozenset({"disk_to_db", "db_to_disk", "bidirectional"})


class LegacyNotesSyncSnapshotError(RuntimeError):
    """Legacy evidence could not be read without exposing private details."""


class LegacyNotesSyncMigrationError(RuntimeError):
    """Paused candidates could not be persisted atomically."""


@dataclass(frozen=True, slots=True, repr=False)
class _LegacyRootEvidence:
    source_kind: str
    source_id: str
    canonical_path: str | None
    root_identity_digest: str | None
    ancestor_identity_digests: tuple[str, ...]
    reason_code: str | None

    def __repr__(self) -> str:
        return f"_LegacyRootEvidence(source_kind={self.source_kind!r}, <private>)"


@dataclass(frozen=True, slots=True, repr=False)
class _LegacyNoteEvidence:
    note_id: object
    version: object
    canonical_root: str | None
    root_reason_code: str | None
    raw_relative_path: object
    content_digest: object
    file_identity_digest: str | None
    file_mode: int | None
    file_freshness_digest: str | None
    file_reason_code: str | None

    def __repr__(self) -> str:
        return "_LegacyNoteEvidence(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class LegacyNotesSyncSnapshot:
    """Frozen historical evidence; representation intentionally hides paths."""

    note_scope_id: str
    source_fingerprint: str
    roots: tuple[_LegacyRootEvidence, ...]
    notes: tuple[_LegacyNoteEvidence, ...]
    policy_issues: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.note_scope_id, field_name="note_scope_id")
        validate_notes_sync_digest(
            self.source_fingerprint,
            field_name="source_fingerprint",
        )

    def __repr__(self) -> str:
        return (
            "LegacyNotesSyncSnapshot("
            f"roots={len(self.roots)}, notes={len(self.notes)}, <private>)"
        )


@dataclass(frozen=True, slots=True)
class LegacyMigrationReportEntry:
    """One bounded path-free reason requiring migration review."""

    reason_code: str
    root_id: str | None = None
    binding_id: str | None = None

    def __post_init__(self) -> None:
        validate_notes_sync_reason_code(self.reason_code)
        if self.root_id is not None:
            validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        if self.binding_id is not None:
            validate_notes_sync_opaque_id(self.binding_id, field_name="binding_id")


@dataclass(frozen=True, slots=True, repr=False)
class LegacyNotesSyncMigrationPlan:
    """Pure paused-candidate plan with no mutation authority."""

    migration_id: str
    source_fingerprint: str
    roots: tuple[NotesSyncRootRecord, ...]
    bindings: tuple[NotesSyncBindingRecord, ...]
    report: tuple[LegacyMigrationReportEntry, ...]
    requires_fresh_dry_run: bool = True
    requires_explicit_activation: bool = True

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.migration_id, field_name="migration_id")
        validate_notes_sync_digest(
            self.source_fingerprint,
            field_name="source_fingerprint",
        )
        if type(self.roots) is not tuple or any(
            type(root) is not NotesSyncRootRecord for root in self.roots
        ):
            raise TypeError("roots must be a tuple of NotesSyncRootRecord values.")
        if type(self.bindings) is not tuple or any(
            type(binding) is not NotesSyncBindingRecord for binding in self.bindings
        ):
            raise TypeError(
                "bindings must be a tuple of NotesSyncBindingRecord values."
            )
        if type(self.report) is not tuple or any(
            type(item) is not LegacyMigrationReportEntry for item in self.report
        ):
            raise TypeError(
                "report must be a tuple of LegacyMigrationReportEntry values."
            )
        if len(self.report) > LEGACY_MIGRATION_REPORT_LIMIT:
            raise ValueError("migration report exceeds its fixed bound.")
        if any(root.state is not NotesSyncRootState.PAUSED for root in self.roots):
            raise ValueError("legacy roots must remain paused candidates.")
        if any(
            binding.state is not NotesSyncBindingState.CANDIDATE
            for binding in self.bindings
        ):
            raise ValueError("legacy bindings must remain candidates.")
        if (
            type(self.requires_fresh_dry_run) is not bool
            or not self.requires_fresh_dry_run
            or type(self.requires_explicit_activation) is not bool
            or not self.requires_explicit_activation
        ):
            raise ValueError("legacy candidates must retain both activation gates.")

    def __repr__(self) -> str:
        return (
            "LegacyNotesSyncMigrationPlan("
            f"roots={len(self.roots)}, bindings={len(self.bindings)}, "
            f"report={len(self.report)}, paused=True)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class LegacyNotesSyncMigrationResult:
    """Path-free result of one private migration transaction."""

    already_migrated: bool
    root_count: int
    binding_count: int
    report: tuple[LegacyMigrationReportEntry, ...]

    def __post_init__(self) -> None:
        if type(self.already_migrated) is not bool:
            raise TypeError("already_migrated must be a boolean.")
        if any(
            type(value) is not int or value < 0
            for value in (self.root_count, self.binding_count)
        ):
            raise ValueError("migration counts must be non-negative integers.")
        if type(self.report) is not tuple or any(
            type(item) is not LegacyMigrationReportEntry for item in self.report
        ):
            raise TypeError(
                "report must be a tuple of LegacyMigrationReportEntry values."
            )
        if len(self.report) > LEGACY_MIGRATION_REPORT_LIMIT:
            raise ValueError("migration report exceeds its fixed bound.")

    def __repr__(self) -> str:
        return (
            "LegacyNotesSyncMigrationResult("
            f"already_migrated={self.already_migrated}, "
            f"roots={self.root_count}, bindings={self.binding_count}, "
            f"report={len(self.report)})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class LegacyCandidateActivationAuthorization:
    """Path-free proof that a current dry-run received explicit approval."""

    root_id: str
    observation_token: str
    direction: NotesSyncDirection

    def __post_init__(self) -> None:
        validate_notes_sync_opaque_id(self.root_id, field_name="root_id")
        validate_notes_sync_digest(
            self.observation_token,
            field_name="observation_token",
        )
        if type(self.direction) is not NotesSyncDirection:
            raise TypeError("direction must be a NotesSyncDirection.")

    def __repr__(self) -> str:
        return (
            "LegacyCandidateActivationAuthorization("
            f"root_id={self.root_id!r}, direction={self.direction!r}, <private>)"
        )


def _json_value(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if type(value) is bytes:
        return {"bytes_digest": hashlib.sha256(value).hexdigest()}
    return {"invalid_type": type(value).__name__}


def _digest_payload(kind: str, *values: object, _read_budget=None) -> str:
    if _read_budget is not None:
        from tldw_chatbook.Backup_Recovery.mobile_runs_goal_budget import _serialization

        _read_budget.container(2)
        _read_budget.container(len(values))
        # Scalar native digest inputs use the exact existing JSON conversion.
        _read_budget.value(values)
        projected = {"kind": kind, "values": [_json_value(value) for value in values]}
        _serialization(projected, _read_budget, encoded=True)
    else:
        projected = {"kind": kind, "values": [_json_value(value) for value in values]}
    payload = json.dumps(
        projected,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _opaque_id(kind: str, *values: object, _read_budget=None) -> str:
    if _read_budget is not None:
        _read_budget.reserve(64 + 4 * (len(kind) + 8 + 40))
        _read_budget.reserve(64 + 4 * 40)
    return f"legacy-{kind}-{_digest_payload(kind, *values, _read_budget=_read_budget)[:40]}"


def _string_path(value: object) -> str | None:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if type(value) is not str or not value or len(value) > 4096 or "\x00" in value:
        return None
    return value


def _filesystem_paths_overlap(left: Path, right: Path) -> bool:
    if left.samefile(right):
        return True
    return any(parent.samefile(right) for parent in left.parents) or any(
        parent.samefile(left) for parent in right.parents
    )


def _sensitive_root_identity_conflict(root: Path) -> bool:
    context = resolve_sensitive_context()
    if find_root_binding_conflict(root, context) is not None:
        return True
    protected = (
        *context.dirs,
        *context.direct_child_denied_dirs,
        *context.files,
        *context.db_paths,
    )
    for target in protected:
        try:
            target.lstat()
        except FileNotFoundError:
            continue
        if _filesystem_paths_overlap(root, target):
            return True
    return False


def _root_evidence(
    raw_path: object,
    *,
    source_kind: str,
    source_id: str,
    sync_roots: Iterable[Path | str],
    file_notes_roots: Iterable[Path | str],
    private_paths: Iterable[Path | str],
    _read_budget=None,
) -> _LegacyRootEvidence:
    selected = _string_path(raw_path)
    if selected is None:
        return _LegacyRootEvidence(
            source_kind=source_kind,
            source_id=source_id,
            canonical_path=None,
            root_identity_digest=None,
            ancestor_identity_digests=(),
            reason_code="root_invalid",
        )
    path = Path(selected).expanduser()
    if not path.is_absolute():
        return _LegacyRootEvidence(
            source_kind=source_kind,
            source_id=source_id,
            canonical_path=None,
            root_identity_digest=None,
            ancestor_identity_digests=(),
            reason_code="root_invalid",
        )
    try:
        canonical = validate_sync_root_admission(
            path,
            sync_roots=sync_roots,
            file_notes_roots=file_notes_roots,
            private_paths=private_paths,
        )
    except NotesSyncFilesystemError as error:
        reason = (
            "root_missing"
            if error.reason_code in {"root_unavailable", "comparison_root_unavailable"}
            else error.reason_code
        )
        return _LegacyRootEvidence(
            source_kind=source_kind,
            source_id=source_id,
            canonical_path=None,
            root_identity_digest=None,
            ancestor_identity_digests=(),
            reason_code=reason,
        )
    try:
        for roots, reason in (
            (sync_roots, "root_overlap"),
            (file_notes_roots, "file_notes_overlap"),
            (private_paths, "private_path_overlap"),
        ):
            for other in roots:
                if _filesystem_paths_overlap(
                    canonical,
                    Path(other).expanduser().resolve(strict=True),
                ):
                    return _LegacyRootEvidence(
                        source_kind=source_kind,
                        source_id=source_id,
                        canonical_path=None,
                        root_identity_digest=None,
                        ancestor_identity_digests=(),
                        reason_code=reason,
                    )
        if _sensitive_root_identity_conflict(canonical):
            return _LegacyRootEvidence(
                source_kind=source_kind,
                source_id=source_id,
                canonical_path=None,
                root_identity_digest=None,
                ancestor_identity_digests=(),
                reason_code="private_path_overlap",
            )
        root_metadata = canonical.stat()
        root_identity_digest = _digest_payload(
            "root-identity",
            root_metadata.st_dev,
            root_metadata.st_ino,
            _read_budget=_read_budget,
        )
        ancestor_identity_digests = tuple(
            _digest_payload(
                "root-identity",
                metadata.st_dev,
                metadata.st_ino,
                _read_budget=_read_budget,
            )
            for metadata in (parent.stat() for parent in canonical.parents)
        )
    except OSError:
        return _LegacyRootEvidence(
            source_kind=source_kind,
            source_id=source_id,
            canonical_path=None,
            root_identity_digest=None,
            ancestor_identity_digests=(),
            reason_code="comparison_root_unavailable",
        )
    except Exception:
        return _LegacyRootEvidence(
            source_kind=source_kind,
            source_id=source_id,
            canonical_path=None,
            root_identity_digest=None,
            ancestor_identity_digests=(),
            reason_code="private_path_check_failed",
        )
    return _LegacyRootEvidence(
        source_kind=source_kind,
        source_id=source_id,
        canonical_path=str(canonical),
        root_identity_digest=root_identity_digest,
        ancestor_identity_digests=ancestor_identity_digests,
        reason_code=None,
    )


def _probe_note_file(
    canonical_root: str | None,
    raw_relative_path: object,
    raw_file_path: object,
    *,
    _read_budget=None,
) -> tuple[str | None, int | None, str | None, str | None]:
    if canonical_root is None:
        return None, None, None, None
    try:
        relative_path = normalize_notes_sync_relative_path(raw_relative_path)
    except (TypeError, ValueError):
        return None, None, None, "invalid_note_evidence"
    root = Path(canonical_root)
    expected = root.joinpath(*relative_path.split("/"))
    selected_text = _string_path(raw_file_path)
    selected = expected if selected_text is None else Path(selected_text).expanduser()
    if not selected.is_absolute():
        return None, None, None, "file_out_of_root"
    try:
        selected_canonical = selected.resolve(strict=False)
        expected_canonical = expected.resolve(strict=False)
    except OSError:
        return None, None, None, "file_unavailable"
    try:
        paths_match = selected.samefile(expected)
    except OSError:
        paths_match = os.path.normcase(
            os.fspath(selected_canonical)
        ) == os.path.normcase(os.fspath(expected_canonical))
    if not paths_match:
        return None, None, None, "file_out_of_root"

    current = root
    for component in relative_path.split("/")[:-1]:
        current /= component
        try:
            ancestor = current.lstat()
        except FileNotFoundError:
            return None, None, None, "file_missing"
        except OSError:
            return None, None, None, "file_unavailable"
        if (
            stat.S_ISLNK(ancestor.st_mode)
            or getattr(ancestor, "st_reparse_tag", 0)
            or not stat.S_ISDIR(ancestor.st_mode)
        ):
            return None, None, None, "unsafe_file_identity"
    try:
        metadata = selected.lstat()
    except FileNotFoundError:
        return None, None, None, "file_missing"
    except OSError:
        return None, None, None, "file_unavailable"
    identity = _digest_payload(
        "file-identity", metadata.st_dev, metadata.st_ino, _read_budget=_read_budget
    )
    if (
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_reparse_tag", 0)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        return (
            identity,
            stat.S_IMODE(metadata.st_mode),
            None,
            "unsafe_file_identity",
        )
    mode = stat.S_IMODE(metadata.st_mode)
    freshness = _digest_payload(
        "file-freshness",
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        mode,
        _read_budget=_read_budget,
    )
    return identity, mode, freshness, None


def _derive_row_root(raw_file_path: object, raw_relative_path: object) -> str | None:
    selected_text = _string_path(raw_file_path)
    if selected_text is None:
        return None
    selected = Path(selected_text).expanduser()
    if not selected.is_absolute():
        return None
    try:
        relative_path = normalize_notes_sync_relative_path(raw_relative_path)
    except (TypeError, ValueError):
        return None
    parts = relative_path.split("/")
    try:
        root = selected.parents[len(parts) - 1]
    except IndexError:
        return None
    expected = root.joinpath(*parts)
    if os.path.normcase(os.path.abspath(selected)) != os.path.normcase(
        os.path.abspath(expected)
    ):
        return None
    return str(root)


def _policy_snapshot(
    settings: Mapping[str, object],
) -> tuple[tuple[tuple[str, object], ...], tuple[str, ...]]:
    raw_notes = settings.get("notes")
    if raw_notes is None:
        return (), ()
    if not isinstance(raw_notes, Mapping):
        return (), ("invalid_legacy_policy",)
    values = tuple(
        (key, _json_value(raw_notes.get(key))) for key in _LEGACY_POLICY_KEYS
    )
    issues: list[str] = []
    for key, value in values:
        if value is None:
            continue
        valid = (
            (key in {"auto_sync_enabled", "sync_on_close"} and type(value) is bool)
            or (
                key == "conflict_resolution"
                and type(value) is str
                and value in _VALID_CONFLICT_POLICIES
            )
            or (
                key == "sync_direction"
                and type(value) is str
                and value in _VALID_DIRECTIONS
            )
        )
        if not valid:
            issues.append("invalid_legacy_policy")
    if any(value is not None for _key, value in values):
        issues.append("legacy_policy_ignored")
    return values, tuple(dict.fromkeys(issues))


def legacy_sync_directory_configured(settings: object) -> bool:
    """Return True when the legacy ``notes.sync_directory`` key is present.

    Key *presence* — not validity — mirrors this module's own config-root
    evidence test in :func:`snapshot_legacy_notes_sync`. The TASK-21112
    start gate uses this so the one-time migration path still runs for any
    profile that carries the key, without opening any database.

    Args:
        settings: The loaded application settings mapping.

    Returns:
        True when a ``notes`` mapping exists and contains ``sync_directory``.
    """

    if not isinstance(settings, Mapping):
        return False
    notes = settings.get("notes")
    return isinstance(notes, Mapping) and "sync_directory" in notes


def snapshot_legacy_notes_sync(
    legacy_connection: sqlite3.Connection,
    settings: Mapping[str, object],
    *,
    note_scope_id: str,
    sync_roots: Iterable[Path | str] = (),
    file_notes_roots: Iterable[Path | str] = (),
    private_paths: Iterable[Path | str] = (),
) -> LegacyNotesSyncSnapshot:
    """Read a frozen legacy snapshot using SELECT and filesystem metadata only."""

    if not isinstance(legacy_connection, sqlite3.Connection):
        raise TypeError("legacy_connection must be a sqlite3.Connection.")
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping.")
    validate_notes_sync_opaque_id(note_scope_id, field_name="note_scope_id")
    sync_roots = tuple(sync_roots)
    file_notes_roots = tuple(file_notes_roots)
    private_paths = tuple(private_paths)
    policy_values, policy_issues = _policy_snapshot(settings)
    raw_notes = settings.get("notes")
    roots: list[_LegacyRootEvidence] = []
    if isinstance(raw_notes, Mapping) and "sync_directory" in raw_notes:
        roots.append(
            _root_evidence(
                raw_notes.get("sync_directory"),
                source_kind="config",
                source_id="config",
                sync_roots=sync_roots,
                file_notes_roots=file_notes_roots,
                private_paths=private_paths,
            )
        )
    evidence_limit = LEGACY_SNAPSHOT_EVIDENCE_LIMIT
    try:
        note_rows = legacy_connection.execute(
            """
            SELECT id, version, file_path_on_disk, relative_file_path_on_disk,
                   sync_root_folder, last_synced_disk_file_hash,
                   last_synced_disk_file_mtime, is_externally_synced,
                   sync_strategy, sync_excluded, file_extension
            FROM notes
            WHERE deleted = 0 AND (
                is_externally_synced = 1
                OR file_path_on_disk IS NOT NULL
                OR relative_file_path_on_disk IS NOT NULL
                OR sync_root_folder IS NOT NULL
            )
            ORDER BY CAST(id AS TEXT), rowid
            LIMIT ?
            """,
            (evidence_limit + 1,),
        ).fetchall()
        session_rows = legacy_connection.execute(
            """
            SELECT session_id, sync_root_folder, sync_direction,
                   conflict_resolution, started_at, completed_at, status,
                   total_files, processed_files, conflicts_found, errors_count,
                   client_id, summary
            FROM sync_sessions
            ORDER BY started_at, CAST(session_id AS TEXT), rowid
            LIMIT ?
            """,
            (evidence_limit + 1,),
        ).fetchall()
    except sqlite3.Error:
        raise LegacyNotesSyncSnapshotError("legacy_snapshot_failed") from None
    if len(note_rows) > evidence_limit or len(session_rows) > evidence_limit:
        # Migrating a truncated evidence set would drop bindings and turn the
        # remainder into apparent new files; refuse loudly instead.
        raise LegacyNotesSyncSnapshotError("legacy_snapshot_overflow")

    notes: list[_LegacyNoteEvidence] = []
    for row in note_rows:
        source_id = _opaque_id("note-source", row[0])
        raw_root = _derive_row_root(row[2], row[3]) if row[4] is None else row[4]
        root = _root_evidence(
            raw_root,
            source_kind="row",
            source_id=source_id,
            sync_roots=sync_roots,
            file_notes_roots=file_notes_roots,
            private_paths=private_paths,
        )
        roots.append(root)
        identity, mode, freshness, file_reason = _probe_note_file(
            root.canonical_path,
            row[3],
            row[2],
        )
        notes.append(
            _LegacyNoteEvidence(
                note_id=row[0],
                version=row[1],
                canonical_root=root.canonical_path,
                root_reason_code=root.reason_code,
                raw_relative_path=row[3],
                content_digest=row[5],
                file_identity_digest=identity,
                file_mode=mode,
                file_freshness_digest=freshness,
                file_reason_code=file_reason,
            )
        )
    for row in session_rows:
        roots.append(
            _root_evidence(
                row[1],
                source_kind="session",
                source_id=_opaque_id("session-source", row[0]),
                sync_roots=sync_roots,
                file_notes_roots=file_notes_roots,
                private_paths=private_paths,
            )
        )
        if row[2] not in _VALID_DIRECTIONS or row[3] not in _VALID_CONFLICT_POLICIES:
            policy_issues = tuple((*policy_issues, "invalid_legacy_policy"))
        policy_issues = tuple((*policy_issues, "legacy_policy_ignored"))

    fingerprint_payload = {
        "config": policy_values,
        "config_root": (
            _json_value(raw_notes.get("sync_directory"))
            if isinstance(raw_notes, Mapping) and "sync_directory" in raw_notes
            else None
        ),
        "notes": [[_json_value(value) for value in row] for row in note_rows],
        "sessions": [[_json_value(value) for value in row] for row in session_rows],
        "root_admission": [
            [
                item.source_kind,
                item.source_id,
                item.canonical_path,
                item.root_identity_digest,
                item.ancestor_identity_digests,
                item.reason_code,
            ]
            for item in roots
        ],
        "file_probe": [
            [
                item.file_identity_digest,
                item.file_mode,
                item.file_freshness_digest,
                item.file_reason_code,
            ]
            for item in notes
        ],
    }
    source_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return LegacyNotesSyncSnapshot(
        note_scope_id=note_scope_id,
        source_fingerprint=source_fingerprint,
        roots=tuple(roots),
        notes=tuple(notes),
        policy_issues=tuple(dict.fromkeys(policy_issues)),
    )


def _bounded_report(
    values: Iterable[LegacyMigrationReportEntry],
) -> tuple[LegacyMigrationReportEntry, ...]:
    unique: list[LegacyMigrationReportEntry] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for item in values:
        key = (item.reason_code, item.root_id, item.binding_id)
        if key in seen:
            continue
        if len(unique) == LEGACY_MIGRATION_REPORT_LIMIT:
            unique[-1] = LegacyMigrationReportEntry("migration_report_truncated")
            break
        seen.add(key)
        unique.append(item)
    return tuple(unique)


def plan_legacy_notes_sync_migration(
    snapshot: LegacyNotesSyncSnapshot,
    *,
    _read_budget=None,
) -> LegacyNotesSyncMigrationPlan:
    """Purely translate a frozen legacy snapshot into inert candidate records."""

    if type(snapshot) is not LegacyNotesSyncSnapshot:
        raise TypeError("snapshot must be a LegacyNotesSyncSnapshot.")
    report: list[LegacyMigrationReportEntry] = [
        LegacyMigrationReportEntry(reason) for reason in snapshot.policy_issues
    ]
    grouped: dict[str, list[_LegacyRootEvidence]] = {}
    for evidence in snapshot.roots:
        if evidence.reason_code is not None:
            report.append(LegacyMigrationReportEntry(evidence.reason_code))
            continue
        if (
            evidence.canonical_path is not None
            and evidence.root_identity_digest is not None
        ):
            grouped.setdefault(evidence.root_identity_digest, []).append(evidence)

    overlapping: set[str] = set()
    identities = sorted(grouped)
    canonical_by_identity = {
        identity: min(
            item.canonical_path for item in evidence if item.canonical_path is not None
        )
        for identity, evidence in grouped.items()
    }
    ancestors_by_identity = {
        identity: {
            ancestor for item in evidence for ancestor in item.ancestor_identity_digests
        }
        for identity, evidence in grouped.items()
    }
    for index, left_identity in enumerate(identities):
        for right_identity in identities[index + 1 :]:
            if (
                left_identity in ancestors_by_identity[right_identity]
                or right_identity in ancestors_by_identity[left_identity]
            ):
                overlapping.update({left_identity, right_identity})

    roots: list[NotesSyncRootRecord] = []
    root_ids: dict[str, str] = {}
    for root_identity in identities:
        canonical_path = canonical_by_identity[root_identity]
        root_id = _opaque_id(
            "root", snapshot.note_scope_id, root_identity, _read_budget=_read_budget
        )
        if root_identity in overlapping:
            report.append(LegacyMigrationReportEntry("root_overlap", root_id=root_id))
            continue
        sources = {item.source_kind for item in grouped[root_identity]}
        reason = (
            f"{next(iter(sources))}_only_root"
            if len(sources) == 1
            else "multiple_legacy_sources"
        )
        report.append(LegacyMigrationReportEntry(reason, root_id=root_id))
        for evidence in grouped[root_identity]:
            assert evidence.canonical_path is not None
            root_ids[evidence.canonical_path] = root_id
        roots.append(
            NotesSyncRootRecord(
                root_id=root_id,
                note_scope_id=snapshot.note_scope_id,
                logical_folder_id=None,
                canonical_path=canonical_path,
                direction=NotesSyncDirection.BIDIRECTIONAL,
                state=NotesSyncRootState.PAUSED,
                last_status_code="migration_review_required",
            )
        )

    snapshot_identity_counts: dict[str, int] = {}
    for note in snapshot.notes:
        if note.file_identity_digest is not None:
            snapshot_identity_counts[note.file_identity_digest] = (
                snapshot_identity_counts.get(note.file_identity_digest, 0) + 1
            )
    duplicate_snapshot_identities = {
        identity for identity, count in snapshot_identity_counts.items() if count > 1
    }

    prepared: list[
        tuple[
            _LegacyNoteEvidence,
            str,
            str,
            str,
            int,
            str,
        ]
    ] = []
    for note in snapshot.notes:
        if note.root_reason_code is not None or note.canonical_root not in root_ids:
            continue
        try:
            note_id = validate_notes_sync_opaque_id(note.note_id, field_name="note_id")
            if type(note.version) is not int or note.version < 0:
                raise ValueError
            relative_path = normalize_notes_sync_relative_path(note.raw_relative_path)
            content_digest = validate_notes_sync_digest(
                note.content_digest,
                field_name="content_digest",
            )
        except (TypeError, ValueError):
            report.append(LegacyMigrationReportEntry("invalid_note_evidence"))
            continue
        if note.file_reason_code not in {None, "file_missing"}:
            report.append(LegacyMigrationReportEntry(note.file_reason_code))
            if note.file_identity_digest in duplicate_snapshot_identities:
                report.append(LegacyMigrationReportEntry("duplicate_file_identity"))
            continue
        root_id = root_ids[note.canonical_root]
        binding_id = _opaque_id(
            "binding", root_id, note_id, relative_path, _read_budget=_read_budget
        )
        if note.file_reason_code == "file_missing":
            report.append(
                LegacyMigrationReportEntry(
                    "file_missing",
                    root_id=root_id,
                    binding_id=binding_id,
                )
            )
            continue
        prepared.append(
            (
                note,
                root_id,
                note_id,
                relative_path,
                note.version,
                content_digest,
            )
        )

    duplicate_paths: set[tuple[str, str]] = set()
    duplicate_identities: set[str] = set()
    path_counts: dict[tuple[str, str], int] = {}
    identity_counts: dict[str, int] = {}
    for note, root_id, _note_id, relative_path, _version, _digest in prepared:
        path_key = (root_id, relative_path)
        path_counts[path_key] = path_counts.get(path_key, 0) + 1
        if note.file_identity_digest is not None:
            identity_counts[note.file_identity_digest] = (
                identity_counts.get(note.file_identity_digest, 0) + 1
            )
    duplicate_paths.update(key for key, count in path_counts.items() if count > 1)
    duplicate_identities.update(
        key for key, count in identity_counts.items() if count > 1
    )

    bindings: list[NotesSyncBindingRecord] = []
    for note, root_id, note_id, relative_path, version, content_digest in prepared:
        binding_id = _opaque_id(
            "binding", root_id, note_id, relative_path, _read_budget=_read_budget
        )
        if (root_id, relative_path) in duplicate_paths:
            report.append(
                LegacyMigrationReportEntry(
                    "duplicate_binding_path",
                    root_id=root_id,
                    binding_id=binding_id,
                )
            )
            continue
        if (
            note.file_identity_digest is not None
            and note.file_identity_digest in duplicate_identities
        ):
            report.append(
                LegacyMigrationReportEntry(
                    "duplicate_file_identity",
                    root_id=root_id,
                    binding_id=binding_id,
                )
            )
            continue
        if note.file_identity_digest is None or note.file_mode is None:
            report.append(
                LegacyMigrationReportEntry(
                    "incomplete_note_evidence",
                    root_id=root_id,
                    binding_id=binding_id,
                )
            )
            continue
        bindings.append(
            NotesSyncBindingRecord(
                binding_id=binding_id,
                root_id=root_id,
                note_scope_id=snapshot.note_scope_id,
                note_id=note_id,
                normalized_relative_path=relative_path,
                stable_identity_digest=note.file_identity_digest,
                state=NotesSyncBindingState.CANDIDATE,
                serialization=NotesSyncSerializationProfile(
                    utf8_bom=False,
                    newline="lf",
                    final_newline=False,
                    mode=note.file_mode,
                ),
                content_digest=content_digest,
                note_version=version,
            )
        )

    return LegacyNotesSyncMigrationPlan(
        migration_id=_opaque_id(
            "migration", snapshot.source_fingerprint, _read_budget=_read_budget
        ),
        source_fingerprint=snapshot.source_fingerprint,
        roots=tuple(sorted(roots, key=lambda item: item.root_id)),
        bindings=tuple(sorted(bindings, key=lambda item: item.binding_id)),
        report=_bounded_report(report),
    )


def persist_legacy_notes_sync_migration(
    store: NotesDeviceStateStore,
    plan: LegacyNotesSyncMigrationPlan,
) -> LegacyNotesSyncMigrationResult:
    """Persist one plan idempotently in exactly one private transaction."""

    if not isinstance(store, NotesDeviceStateStore):
        raise TypeError("store must be a NotesDeviceStateStore.")
    if type(plan) is not LegacyNotesSyncMigrationPlan:
        raise TypeError("plan must be a LegacyNotesSyncMigrationPlan.")
    try:
        with store.transaction(immediate=True) as connection:
            return _persist_legacy_plan(connection, plan)
    except Exception:
        raise LegacyNotesSyncMigrationError("legacy_migration_failed") from None


def _persist_legacy_plan(connection, plan):
    """Insert a native plan inside the caller's existing private transaction."""
    timestamp = max(1, time.time_ns())
    root_count = 0
    binding_count = 0
    exists = connection.execute(
        """
        SELECT 1 FROM notes_sync_legacy_migrations
        WHERE source_fingerprint = ?
        """,
        (plan.source_fingerprint,),
    ).fetchone()
    if exists is not None:
        return LegacyNotesSyncMigrationResult(True, 0, 0, plan.report)
    eligible_root_ids: set[str] = set()
    for root in plan.roots:
        existing_root = connection.execute(
            """
            SELECT note_scope_id, canonical_path, state
            FROM notes_sync_roots WHERE root_id = ?
            """,
            (root.root_id,),
        ).fetchone()
        if existing_root is None:
            connection.execute(
                """
                INSERT INTO notes_sync_roots (
                    root_id, note_scope_id, logical_folder_id, canonical_path,
                    remote_origin_id, direction, state, cursor, last_status_code,
                    created_at, updated_at
                ) VALUES (?, ?, NULL, ?, NULL, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    root.root_id,
                    root.note_scope_id,
                    root.canonical_path,
                    root.direction.value,
                    root.state.value,
                    root.last_status_code,
                    timestamp,
                    timestamp,
                ),
            )
            root_count += 1
            eligible_root_ids.add(root.root_id)
        elif tuple(existing_root[:2]) != (
            root.note_scope_id,
            root.canonical_path,
        ):
            raise LegacyNotesSyncMigrationError("legacy_candidate_collision")
        elif existing_root[2] == NotesSyncRootState.PAUSED.value:
            eligible_root_ids.add(root.root_id)
    for binding in plan.bindings:
        existing_binding = connection.execute(
            """
            SELECT root_id, note_scope_id, note_id,
                   normalized_relative_path
            FROM notes_sync_bindings WHERE binding_id = ?
            """,
            (binding.binding_id,),
        ).fetchone()
        if existing_binding is not None:
            if tuple(existing_binding) != (
                binding.root_id,
                binding.note_scope_id,
                binding.note_id,
                binding.normalized_relative_path,
            ):
                raise LegacyNotesSyncMigrationError("legacy_candidate_collision")
            continue
        if binding.root_id not in eligible_root_ids:
            continue
        connection.execute(
            """
            INSERT INTO notes_sync_bindings (
                binding_id, root_id, note_scope_id, note_id,
                normalized_relative_path, stable_identity_digest, state,
                utf8_bom, newline, final_newline, file_mode,
                content_digest, note_version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                binding.binding_id,
                binding.root_id,
                binding.note_scope_id,
                binding.note_id,
                binding.normalized_relative_path,
                binding.stable_identity_digest,
                binding.state.value,
                int(binding.serialization.utf8_bom),
                binding.serialization.newline,
                int(binding.serialization.final_newline),
                binding.serialization.mode,
                binding.content_digest,
                binding.note_version,
                timestamp,
                timestamp,
            ),
        )
        binding_count += 1
    connection.execute(
        """
        INSERT INTO notes_sync_legacy_migrations (
            migration_id, source_fingerprint, state, reason_code,
            created_at, updated_at
        ) VALUES (?, ?, 'pending_review', ?, ?, ?)
        """,
        (
            plan.migration_id,
            plan.source_fingerprint,
            ("migration_review_required" if plan.roots else "legacy_evidence_unusable"),
            timestamp,
            timestamp,
        ),
    )
    return LegacyNotesSyncMigrationResult(
        already_migrated=False,
        root_count=root_count,
        binding_count=binding_count,
        report=plan.report,
    )


def authorize_legacy_candidate_activation(
    root_id: str,
    *,
    dry_run: ReconciliationPlan,
    fresh_observations: ReconciliationInput,
    explicitly_approved: bool,
) -> LegacyCandidateActivationAuthorization:
    """Return activation proof only for an approved, complete, current dry-run."""

    validate_notes_sync_opaque_id(root_id, field_name="root_id")
    if type(dry_run) is not ReconciliationPlan:
        raise TypeError("dry_run must be a ReconciliationPlan.")
    if type(fresh_observations) is not ReconciliationInput:
        raise TypeError("fresh_observations must be a ReconciliationInput.")
    if type(explicitly_approved) is not bool:
        raise TypeError("explicitly_approved must be a boolean.")
    if not explicitly_approved:
        raise ValueError("explicit_activation_required")
    if dry_run.root_id != root_id or fresh_observations.root_id != root_id:
        raise ValueError("candidate_root_mismatch")
    assert_review_current(dry_run, fresh_observations)
    if dry_run != plan_reconciliation(fresh_observations):
        raise ValueError("dry_run_plan_mismatch")
    if (
        fresh_observations.observation_generation
        != fresh_observations.expected_generation
        or not fresh_observations.root_available
        or fresh_observations.root_overlap
        or dry_run.skips
        or dry_run.attention
        or dry_run.deletion_groups
    ):
        raise ValueError("complete_dry_run_required")
    return LegacyCandidateActivationAuthorization(
        root_id=root_id,
        observation_token=dry_run.observation_token,
        direction=fresh_observations.direction,
    )


__all__ = [
    "LEGACY_MIGRATION_REPORT_LIMIT",
    "LEGACY_SNAPSHOT_EVIDENCE_LIMIT",
    "LegacyCandidateActivationAuthorization",
    "LegacyMigrationReportEntry",
    "LegacyNotesSyncMigrationError",
    "LegacyNotesSyncMigrationPlan",
    "LegacyNotesSyncMigrationResult",
    "LegacyNotesSyncSnapshot",
    "LegacyNotesSyncSnapshotError",
    "authorize_legacy_candidate_activation",
    "legacy_sync_directory_configured",
    "persist_legacy_notes_sync_migration",
    "plan_legacy_notes_sync_migration",
    "snapshot_legacy_notes_sync",
]


def _import_root_paths(connection, budget):
    """Read all current root paths, regardless of state or identifier shape."""
    budget.container(2)
    count, width = connection.execute(
        "SELECT count(*),coalesce(sum(length(CAST(canonical_path AS BLOB))),0) FROM notes_sync_roots"
    ).fetchone()
    if count > LEGACY_SNAPSHOT_EVIDENCE_LIMIT:
        raise LegacyNotesSyncSnapshotError("legacy_snapshot_overflow")
    budget.container(count * 3)
    budget.reserve(width * 4)
    return tuple(
        row[0]
        for row in connection.execute(
            "SELECT canonical_path FROM notes_sync_roots ORDER BY root_id"
        ).fetchall()
    )


def _import_note_owned(connection, scope, note_id, budget):
    budget.container(3)
    return (
        connection.execute(
            "SELECT 1 FROM notes_sync_bindings WHERE note_scope_id=? AND note_id=? LIMIT 1",
            (scope, note_id),
        ).fetchone()
        is not None
    )


def _price_import_paths(paths, budget):
    from tldw_chatbook.Backup_Recovery.admission import _price_paths

    # Same native parse/resolve/parents unit, before each evidence call.
    _price_paths(paths, budget)


def _plan_imported_notes(
    connection, store, budget, *, note_scope_id, file_notes_roots=(), private_paths=()
):
    """Review only receipt-backed unrepresented links; never activate a root."""
    from tldw_chatbook.Backup_Recovery.mobile_runs_goal_budget import _serialization

    if not isinstance(store, NotesDeviceStateStore):
        raise TypeError("store must be a NotesDeviceStateStore")
    validate_notes_sync_opaque_id(note_scope_id, field_name="note_scope_id")
    # Identity-only no-op precedes the full receipt codec and note/file evidence.
    linked = "deleted=0 AND (is_externally_synced=1 OR file_path_on_disk IS NOT NULL OR relative_file_path_on_disk IS NOT NULL OR sync_root_folder IS NOT NULL)"
    budget.container(2)
    budget.value(linked)
    count, width = connection.execute(
        "SELECT count(*),coalesce(sum(length(CAST(id AS BLOB))),0) FROM notes WHERE deleted=0 AND (is_externally_synced=1 OR file_path_on_disk IS NOT NULL OR relative_file_path_on_disk IS NOT NULL OR sync_root_folder IS NOT NULL)"
    ).fetchone()
    if not count:
        return None, False
    if count > LEGACY_SNAPSHOT_EVIDENCE_LIMIT:
        raise LegacyNotesSyncSnapshotError("legacy_snapshot_overflow")
    budget.container(count * 3)
    budget.reserve(width * 4)
    candidates = connection.execute(
        "SELECT id FROM notes WHERE deleted=0 AND (is_externally_synced=1 OR file_path_on_disk IS NOT NULL OR relative_file_path_on_disk IS NOT NULL OR sync_root_folder IS NOT NULL)"
    ).fetchall()
    budget.container(count)
    with store.transaction() as private:
        unrepresented = tuple(
            row[0]
            for row in candidates
            if not _import_note_owned(private, note_scope_id, row[0], budget)
        )
    if not unrepresented:
        return None, False
    budget.container(1)
    if (
        connection.execute(
            "SELECT 1 FROM mobile_consolidation_batches WHERE substr(batch_key,1,4)='chat' LIMIT 1"
        ).fetchone()
        is None
    ):
        return None, False
    from tldw_chatbook.DB.mobile_consolidation_receipts import _imported_note_ids

    identities = _imported_note_ids(connection, budget)
    budget.container(len(unrepresented))
    pending = tuple(note for note in unrepresented if note in identities)
    if not pending:
        return None, False
    with store.transaction() as private:
        current_roots = _import_root_paths(private, budget)
    budget.container(len(pending) * 3)
    rows, roots, notes = [], [], []
    # Only fixed columns of a selected, unrepresented, current linked note.
    fields = (
        "id",
        "version",
        "file_path_on_disk",
        "relative_file_path_on_disk",
        "sync_root_folder",
        "last_synced_disk_file_hash",
    )
    budget.container(len(fields))
    for note_id in pending:
        budget.container(len(fields) + 2)
        size = connection.execute(
            "SELECT coalesce(length(CAST(id AS BLOB)),0),coalesce(length(CAST(version AS BLOB)),0),coalesce(length(CAST(file_path_on_disk AS BLOB)),0),coalesce(length(CAST(relative_file_path_on_disk AS BLOB)),0),coalesce(length(CAST(sync_root_folder AS BLOB)),0),coalesce(length(CAST(last_synced_disk_file_hash AS BLOB)),0) FROM notes WHERE id=? AND deleted=0 AND (is_externally_synced=1 OR file_path_on_disk IS NOT NULL OR relative_file_path_on_disk IS NOT NULL OR sync_root_folder IS NOT NULL)",
            (note_id,),
        ).fetchone()
        if size is None:
            continue
        budget.container(len(fields) * 2)
        budget.reserve(4 * sum(size))
        row = connection.execute(
            "SELECT id,version,file_path_on_disk,relative_file_path_on_disk,sync_root_folder,last_synced_disk_file_hash FROM notes WHERE id=? AND deleted=0 AND (is_externally_synced=1 OR file_path_on_disk IS NOT NULL OR relative_file_path_on_disk IS NOT NULL OR sync_root_folder IS NOT NULL)",
            (note_id,),
        ).fetchone()
        if row is None:
            raise LegacyNotesSyncSnapshotError("legacy_snapshot_changed")
        if len(rows) >= LEGACY_SNAPSHOT_EVIDENCE_LIMIT:
            raise LegacyNotesSyncSnapshotError("legacy_snapshot_overflow")
        budget.value(row)
        rows.append(tuple(row))
        source_id = _opaque_id("note-source", row[0], _read_budget=budget)
        # Native root/file probes retain their refusal and no-follow semantics.
        budget.container(3)
        path_inputs = tuple(value for value in row[2:5] if isinstance(value, str))
        _price_import_paths(path_inputs, budget)
        raw_root = _derive_row_root(row[2], row[3]) if row[4] is None else row[4]
        budget.container(
            len(current_roots) + len(file_notes_roots) + len(private_paths) + 1
        )
        _price_import_paths(
            (raw_root, *current_roots, *file_notes_roots, *private_paths), budget
        )
        root = _root_evidence(
            raw_root,
            source_kind="row",
            source_id=source_id,
            sync_roots=current_roots,
            file_notes_roots=file_notes_roots,
            private_paths=private_paths,
            _read_budget=budget,
        )
        budget.container(12)
        _price_import_paths(path_inputs, budget)
        identity, mode, freshness, reason = _probe_note_file(
            root.canonical_path, row[3], row[2], _read_budget=budget
        )
        roots.append(root)
        notes.append(
            _LegacyNoteEvidence(
                row[0],
                row[1],
                root.canonical_path,
                root.reason_code,
                row[3],
                row[5],
                identity,
                mode,
                freshness,
                reason,
            )
        )
    if not rows:
        return None, False
    # Native fingerprint framing over only this immutable selection, not sessions/config.
    budget.value(rows)
    budget.container(3)
    budget.container(len(roots))
    budget.container(len(notes))
    for _ in roots:
        budget.container(4)
    for _ in notes:
        budget.container(4)
    payload = {
        "notes": rows,
        "roots": [
            (
                root.canonical_path,
                root.root_identity_digest,
                root.ancestor_identity_digests,
                root.reason_code,
            )
            for root in roots
        ],
        "files": [
            (
                note.file_identity_digest,
                note.file_mode,
                note.file_freshness_digest,
                note.file_reason_code,
            )
            for note in notes
        ],
    }
    _serialization(payload, budget, encoded=True)
    fingerprint = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    budget.container(len(roots) + len(notes) + 6)
    snapshot = LegacyNotesSyncSnapshot(
        note_scope_id, fingerprint, tuple(roots), tuple(notes), ()
    )
    # Native planner models, grouping/index containers and pairwise root comparisons.
    budget.value(snapshot, text_copies=2)
    budget.container(32 * (len(roots) + len(notes)) + len(roots) ** 2)
    plan = plan_legacy_notes_sync_migration(snapshot, _read_budget=budget)
    warning = len(plan.bindings) != len(notes)
    if not plan.bindings:
        return None, warning
    budget.container(len(plan.bindings) + len(plan.roots))
    admitted_roots = {binding.root_id for binding in plan.bindings}
    plan = LegacyNotesSyncMigrationPlan(
        plan.migration_id,
        plan.source_fingerprint,
        tuple(root for root in plan.roots if root.root_id in admitted_roots),
        plan.bindings,
        plan.report,
    )
    return plan, warning


def _persist_imported_notes(
    store, plan, budget, *, note_scope_id, file_notes_roots=(), private_paths=()
):
    """Recheck current private ownership after the canonical snapshot has closed."""
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    if not isinstance(store, NotesDeviceStateStore):
        raise TypeError("store must be a NotesDeviceStateStore")
    if (
        type(plan) is not LegacyNotesSyncMigrationPlan
        or type(budget) is not _TraceReadBudget
    ):
        raise TypeError("native plan and read budget required")
    budget.value(
        plan
    )  # Native insertion bindings/record materializations before writes.
    with store.transaction(immediate=True) as private:
        current_roots = _import_root_paths(private, budget)
        for binding in plan.bindings:
            if _import_note_owned(private, note_scope_id, binding.note_id, budget):
                raise LegacyNotesSyncMigrationError("legacy_candidate_collision")
            budget.container(2)
            if (
                private.execute(
                    "SELECT 1 FROM notes_sync_bindings WHERE stable_identity_digest=? LIMIT 1",
                    (binding.stable_identity_digest,),
                ).fetchone()
                is not None
            ):
                raise LegacyNotesSyncMigrationError("legacy_candidate_collision")
        for root in plan.roots:
            budget.container(
                len(current_roots) + len(file_notes_roots) + len(private_paths) + 1
            )
            _price_import_paths(
                (
                    root.canonical_path,
                    *current_roots,
                    *file_notes_roots,
                    *private_paths,
                ),
                budget,
            )
            current = _root_evidence(
                root.canonical_path,
                source_kind="row",
                source_id=root.root_id,
                sync_roots=current_roots,
                file_notes_roots=file_notes_roots,
                private_paths=private_paths,
                _read_budget=budget,
            )
            if (
                current.reason_code is not None
                or _opaque_id(
                    "root",
                    note_scope_id,
                    current.root_identity_digest,
                    _read_budget=budget,
                )
                != root.root_id
            ):
                raise LegacyNotesSyncMigrationError("legacy_candidate_collision")
        _persist_legacy_plan(private, plan)


@contextmanager
def _core_import_read(core):
    """Borrow the actual selected Core; retire only a newly owned thread handle."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    if not isinstance(core, CharactersRAGDB):
        raise LegacyNotesSyncSnapshotError("canonical_owner_unavailable")
    before = getattr(core._local, "conn", None)
    try:
        with core.transaction() as cursor:
            yield cursor.connection
    finally:
        current = getattr(core._local, "conn", None)
        if current is not None and current is not before:
            core.close_connection()
            if getattr(core._local, "conn", None) is current:
                raise LegacyNotesSyncSnapshotError("canonical_reader_not_retired")


def _canonical_import_present(core):
    with _core_import_read(core) as connection:
        return (
            connection.execute(
                "SELECT 1 FROM mobile_consolidation_batches WHERE substr(batch_key,1,4)='chat' LIMIT 1"
            ).fetchone()
            is not None
        )
