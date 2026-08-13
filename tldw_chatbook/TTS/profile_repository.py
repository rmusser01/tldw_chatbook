"""Serialized lifecycle owner for the local TTS generation-profile store."""

from __future__ import annotations

import asyncio
import math
import os
import sqlite3
import stat
import threading
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, Literal, TypeVar, cast
from unicodedata import category as _unicode_category
from unicodedata import normalize as _unicode_normalize
from uuid import UUID, uuid4

from tldw_chatbook.DB.private_sqlite import (
    ProfileMigrationBoundaryDestination,
    backup_connection_to_private,
    backup_open_connections_to_private,
    close_profile_migration_destination,
    connect_private_sqlite,
    discard_profile_migration_destination,
    migrate_profile_store_to_candidate,
    open_canonical_profile_migration_destination,
)
import tldw_chatbook.TTS.profile_schema as _profile_schema
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_migration_candidate import (
    ProfileMigrationBoundary,
    ProfileMigrationBoundaryRequest,
    ProfileMigrationBoundarySnapshot,
    step_profile_migration_candidate,
)
from tldw_chatbook.TTS.profile_migration_journal import (
    PROFILE_MIGRATION_CANDIDATE_LEAVES,
    PROFILE_MIGRATION_ROLLBACK_LEAVES,
    ProfileMigrationPublicationSlot,
    ProfileMigrationPublicationStage,
)
from tldw_chatbook.TTS.profile_migration_namespace import (
    MigrationTombstoneKey,
    ParentAuthority,
    admit_zero_reusable_tombstone,
    prepare_reusable_tombstone,
    remove_exact,
    remove_zero_reusable_tombstone,
    require_reusable_tombstone,
)
from tldw_chatbook.TTS.profile_migration_publication import (
    prepare_profile_migration_artifact,
    publish_profile_migration,
    retain_profile_migration_destination,
)
from tldw_chatbook.TTS.profile_migration_recovery import (
    recover_profile_migration_publication,
)
from tldw_chatbook.TTS.profile_reference_audio import validate_canonical_reference_wav
from tldw_chatbook.TTS.profile_reference_storage import (
    ASSIGNED_PROFILE_WITH_REFERENCE_JOIN_SELECT,
    PROFILE_WITH_REFERENCE_SELECT,
    REFERENCE_PAYLOAD_SELECT,
    REFERENCE_TABLE,
    decode_reference_payload,
    decode_reference_summary,
    read_reference_blob,
    validate_reference_rows,
    write_reference_blob,
)
from tldw_chatbook.TTS.profile_reference_types import (
    MAX_REFERENCE_COUNT,
    MAX_REFERENCE_TOTAL_BYTES,
    CanonicalTTSCloneReference,
    TTSCloneReference,
    TTSCloneRecipeRequirement,
)
from tldw_chatbook.TTS.profile_schema import (
    CURRENT_PROFILE_SCHEMA_VERSION,
    ExactProfileStoreCleanupError,
    ExactProfileStoreAuthorityError,
    ExactProfileStoreNotCurrentError,
    PostInitProfileStoreAuthority,
    capture_post_init_profile_store_authority,
    decode_assigned_snapshot,
    decode_assignment,
    decode_profile,
    decode_utc_datetime,
    encode_assignment,
    encode_profile,
    encode_utc_datetime,
    encode_uuid,
    open_profile_store,
    open_exact_current_profile_store,
    revalidate_exact_current_profile_store,
    validate_profile_candidate,
    validate_profile_store_rows,
)
from tldw_chatbook.TTS.profile_store_lock import (
    ProfileStoreLease,
    ProfileStoreLockMode,
)
from tldw_chatbook.TTS.profile_types import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    FrozenJsonOptions,
    ProfileBackupReceipt,
    ProfileRepositoryState,
    ProfileRestoreReceipt,
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileCollisionSnapshot,
    TTSProfileDraft,
    TTSProfilePage,
)
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Utils import private_paths


_T = TypeVar("_T")
_PATH_TYPE = type(Path())
_CHARACTER_REF_TYPE = CharacterRef
_CANONICAL_REFERENCE_TYPE = CanonicalTTSCloneReference
_CLONE_RECIPE_REQUIREMENT_TYPE = TTSCloneRecipeRequirement
_TTS_GENERATION_PROFILE_TYPE = TTSGenerationProfile
_TTS_PROFILE_DRAFT_TYPE = TTSProfileDraft
_TTS_PROFILE_COLLISION_SNAPSHOT_TYPE = TTSProfileCollisionSnapshot
_MAX_SEARCH_CHARACTERS = 128
_MAX_NORMALIZED_SEARCH_CHARACTERS = 512
_MAX_NORMALIZED_SEARCH_BYTES = 2_048
_UNSAFE_SEARCH_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})
_unicode_ord = ord
_monotonic = time.monotonic
# SQLite extended result codes are ABI-stable.  Keeping the exact values here
# also supports Python builds that do not expose every named sqlite3 constant.
_SQLITE_CONSTRAINT_FOREIGNKEY = 787
_SQLITE_CONSTRAINT_PRIMARYKEY = 1_555
_SQLITE_CONSTRAINT_TRIGGER = 1_811
_SQLITE_CONSTRAINT_UNIQUE = 2_067
_STORE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_INITIALIZATION_LOCK_TIMEOUT_SECONDS = 0.1
_V2_MIGRATION_BACKUP_SUFFIX = ".pre-v3.sqlite3"
_V3_MIGRATION_BACKUP_SUFFIX = ".pre-v4.sqlite3"
_RESTORE_BACKUP_PAGE_BATCH = 64
_RESTORE_PROGRESS_OPCODE_INTERVAL = 1_000
_RESTORE_REBIND_TIMEOUT_SECONDS = 5.0
_TransactionOperation = Literal[
    "create",
    "read",
    "update",
    "delete",
    "assignment_set",
    "assignment_remove",
    "reference_set",
    "reference_remove",
]
TTSBundleImportChoice = Literal["create", "reuse", "copy"]
TTSBundleDependencyState = Literal["exact", "missing"]
TTSBundleImportResultKind = Literal["created", "reused", "stale_inspection"]
_PROFILE_SELECT = PROFILE_WITH_REFERENCE_SELECT
_BASE_PROFILE_SELECT = """
SELECT
    profile_id, display_name, normalized_name, provider_id, model_id, voice_id,
    response_format, speed, options_json, revision, created_at, updated_at
FROM tts_generation_profiles
"""
_ASSIGNMENT_SELECT = """
SELECT
    source,
    authority_id,
    character_id,
    profile_id,
    created_at,
    updated_at
FROM character_tts_assignments
"""


@dataclass(frozen=True, slots=True)
class _OperationAdmission(Generic[_T]):
    """One generation-bound worker submission awaiting publication."""

    generation: int
    future: Future[_T]


@dataclass(slots=True)
class _IntegrityEvidence:
    """Exact schema-owned values and statement error for one mutation."""

    profile_id: UUID | None
    normalized_name: str | None = None
    statement_error: sqlite3.IntegrityError | None = None


@dataclass(frozen=True, slots=True)
class _PersistedAssignment:
    """One fully decoded assignment row including persistence timestamps."""

    assignment: CharacterTTSAssignment
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class _DestinationSnapshot:
    """One canonical backup destination admitted before worker submission."""

    path: Path
    parent_identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _CandidateSnapshot:
    """One exact standalone restore candidate and its admission identity."""

    path: Path
    identity: tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True, repr=False)
class TTSBundleImportCommand:
    """Private canonical bundle values plus one explicit reviewed decision."""

    choice: TTSBundleImportChoice
    source_profile_id: UUID
    source_draft: TTSProfileDraft
    recipe_requirement: TTSCloneRecipeRequirement
    canonical_reference: CanonicalTTSCloneReference
    expected_generation: int
    reviewed_source_collisions: TTSProfileCollisionSnapshot
    copy_profile_id: UUID | None
    copy_display_name: str | None
    dependency_state: TTSBundleDependencyState
    inactive_consent: bool

    def __repr__(self) -> str:
        return "TTSBundleImportCommand(<private>)"


@dataclass(frozen=True, slots=True)
class TTSBundleImportRepositoryFacts:
    """Safe current repository collisions returned after a stale review."""

    source_collisions: TTSProfileCollisionSnapshot
    copy_collisions: TTSProfileCollisionSnapshot | None


@dataclass(frozen=True, slots=True)
class TTSBundleImportResult:
    """Atomic exact-reuse/create decision and repository-only stale facts."""

    kind: TTSBundleImportResultKind
    profile: TTSGenerationProfile | None
    repository_facts: TTSBundleImportRepositoryFacts | None = None

    def __post_init__(self) -> None:
        profile_result = (
            type(self.kind) is str
            and self.kind in {"created", "reused"}
            and type(self.profile) is _TTS_GENERATION_PROFILE_TYPE
            and self.repository_facts is None
        )
        stale_result = (
            type(self.kind) is str
            and self.kind == "stale_inspection"
            and self.profile is None
            and type(self.repository_facts) is TTSBundleImportRepositoryFacts
        )
        if not profile_result and not stale_result:
            raise _repository_error("operation_failed")


def _repository_error(code: str) -> ProfileRepositoryError:
    return ProfileRepositoryError(code)


def _decode_profile_with_reference_row(row: sqlite3.Row) -> TTSGenerationProfile:
    """Preserve the repository codec seam and add metadata-only reference state."""

    return replace(decode_profile(row), reference=decode_reference_summary(row))


def _decode_assigned_with_reference_row(
    row: sqlite3.Row,
) -> AssignedTTSProfileSnapshot:
    """Preserve the joined codec seam and add metadata-only reference state."""

    snapshot = decode_assigned_snapshot(row)
    return replace(
        snapshot,
        profile=replace(
            snapshot.profile,
            reference=decode_reference_summary(row),
        ),
    )


def _v2_migration_backup_path(active_path: Path) -> Path:
    """Return the fixed inert sibling used for one retained v2 snapshot."""

    return active_path.with_name(f"{active_path.name}{_V2_MIGRATION_BACKUP_SUFFIX}")


def _v3_migration_backup_path(active_path: Path) -> Path:
    """Return the fixed inert sibling used for one retained v3 snapshot."""

    return active_path.with_name(f"{active_path.name}{_V3_MIGRATION_BACKUP_SUFFIX}")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_exact_profile_id(value: object) -> UUID:
    if type(value) is not UUID:
        raise _repository_error("operation_failed")
    profile_id = cast(UUID, value)
    validation_error: BaseException | None = None
    validated: UUID | None = None
    try:
        validated = UUID(str(profile_id))
        if validated != profile_id:
            raise ValueError
    except BaseException as error:
        validation_error = error
    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("operation_failed")
    assert validated is not None
    return validated


def _validate_optional_profile_id(value: object) -> UUID | None:
    if value is None:
        return None
    return _validate_exact_profile_id(value)


def _validate_optional_profile(
    value: object,
) -> TTSGenerationProfile | None:
    """Return an exact canonical profile snapshot or reject the boundary."""

    if value is None:
        return None
    if type(value) is not _TTS_GENERATION_PROFILE_TYPE:
        raise _repository_error("operation_failed")
    profile = cast(TTSGenerationProfile, value)
    validation_error: BaseException | None = None
    validated: TTSGenerationProfile | None = None
    try:
        validated = TTSGenerationProfile(
            profile_id=profile.profile_id,
            display_name=profile.display_name,
            normalized_name=profile.normalized_name,
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            voice_id=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
            revision=profile.revision,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
            reference=profile.reference,
        )
        if validated != profile:
            raise ValueError
    except BaseException as error:
        validation_error = error
    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("operation_failed")
    assert validated is not None
    return validated


def _validate_draft(value: object) -> TTSProfileDraft:
    if type(value) is not _TTS_PROFILE_DRAFT_TYPE:
        raise _repository_error("operation_failed")
    draft = cast(TTSProfileDraft, value)
    validation_error: BaseException | None = None
    validated: TTSProfileDraft | None = None
    try:
        validated = TTSProfileDraft(
            display_name=draft.display_name,
            provider_id=draft.provider_id,
            model_id=draft.model_id,
            voice_id=draft.voice_id,
            response_format=draft.response_format,
            speed=draft.speed,
            options=draft.options,
        )
        if validated != draft:
            raise ValueError
    except BaseException as error:
        validation_error = error
    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("operation_failed")
    assert validated is not None
    return validated


def _validate_expected_revision(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise _repository_error("operation_failed")
    return cast(int, value)


def _validate_expected_generation(value: object) -> int:
    if type(value) is not int or value < 0:
        raise _repository_error("operation_failed")
    return cast(int, value)


def _validate_canonical_reference(value: object) -> CanonicalTTSCloneReference:
    """Return one exact copied canonical reference or reject the boundary."""

    if type(value) is not _CANONICAL_REFERENCE_TYPE:
        raise _repository_error("operation_failed")
    reference = cast(CanonicalTTSCloneReference, value)
    validation_error: BaseException | None = None
    validated: CanonicalTTSCloneReference | None = None
    try:
        validated = CanonicalTTSCloneReference(
            wav_bytes=reference.wav_bytes,
            reference_text=reference.reference_text,
            sha256=reference.sha256,
            byte_length=reference.byte_length,
            duration_ms=reference.duration_ms,
            sample_rate_hz=reference.sample_rate_hz,
            channels=reference.channels,
            sample_encoding=reference.sample_encoding,
        )
        metadata = validate_canonical_reference_wav(validated.wav_bytes)
        if (
            metadata.byte_length != validated.byte_length
            or metadata.duration_ms != validated.duration_ms
            or metadata.sample_rate_hz != validated.sample_rate_hz
            or metadata.channels != validated.channels
            or metadata.sample_encoding != validated.sample_encoding
        ):
            raise ValueError
        if validated != reference:
            raise ValueError
    except BaseException as error:
        validation_error = error
    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("operation_failed")
    assert validated is not None
    return validated


def _validate_recipe_requirement(
    value: object,
    *,
    model_id: str | None = None,
) -> TTSCloneRecipeRequirement:
    """Return a fresh model-coherent exact recipe requirement."""

    if type(value) is not _CLONE_RECIPE_REQUIREMENT_TYPE:
        raise _repository_error("operation_failed")
    requirement = cast(TTSCloneRecipeRequirement, value)
    try:
        validated = TTSCloneRecipeRequirement(
            recipe_id=requirement.recipe_id,
            recipe_revision=requirement.recipe_revision,
            model_id=requirement.model_id,
        )
    except Exception:
        raise _repository_error("operation_failed") from None
    if validated != requirement or (
        model_id is not None and validated.model_id != model_id
    ):
        raise _repository_error("operation_failed")
    return validated


def _validate_collision_snapshot(value: object) -> TTSProfileCollisionSnapshot:
    if type(value) is not _TTS_PROFILE_COLLISION_SNAPSHOT_TYPE:
        raise _repository_error("operation_failed")
    snapshot = cast(TTSProfileCollisionSnapshot, value)
    return TTSProfileCollisionSnapshot(
        _validate_optional_profile(snapshot.profile_id_match),
        _validate_optional_profile(snapshot.normalized_name_match),
    )


def _validate_bundle_import_command(value: object) -> TTSBundleImportCommand:
    """Canonicalize one reviewed import decision before worker admission."""

    if type(value) is not TTSBundleImportCommand:
        raise _repository_error("operation_failed")
    command = cast(TTSBundleImportCommand, value)
    if command.choice not in {"create", "reuse", "copy"}:
        raise _repository_error("operation_failed")
    source_draft = _validate_draft(command.source_draft)
    if (
        source_draft.provider_id != "audio_cpp"
        or source_draft.response_format != "wav"
        or source_draft.speed != 1.0
        or bool(source_draft.options)
    ):
        raise _repository_error("operation_failed")
    source_profile_id = _validate_exact_profile_id(command.source_profile_id)
    requirement = _validate_recipe_requirement(
        command.recipe_requirement,
        model_id=source_draft.model_id,
    )
    canonical = _validate_canonical_reference(command.canonical_reference)
    expected_generation = _validate_expected_generation(command.expected_generation)
    reviewed = _validate_collision_snapshot(command.reviewed_source_collisions)
    if command.dependency_state not in {"exact", "missing"}:
        raise _repository_error("operation_failed")
    if type(command.inactive_consent) is not bool:
        raise _repository_error("operation_failed")
    consent_required = command.dependency_state == "missing" and command.choice in {
        "create",
        "copy",
    }
    if command.inactive_consent is not consent_required:
        raise _repository_error("operation_failed")
    copy_profile_id = _validate_optional_profile_id(command.copy_profile_id)
    copy_display_name = command.copy_display_name
    if command.choice == "copy":
        if copy_profile_id is None or type(copy_display_name) is not str:
            raise _repository_error("operation_failed")
        copy_draft: TTSProfileDraft | None = None
        copy_validation_failed = False
        try:
            copy_draft = TTSProfileDraft(
                display_name=copy_display_name,
                provider_id=source_draft.provider_id,
                model_id=source_draft.model_id,
                voice_id=source_draft.voice_id,
                response_format=source_draft.response_format,
                speed=source_draft.speed,
                options=source_draft.options,
            )
        except Exception:
            copy_validation_failed = True
        if copy_validation_failed or copy_draft is None:
            raise _repository_error("operation_failed")
        copy_display_name = copy_draft.display_name
    elif copy_profile_id is not None or copy_display_name is not None:
        raise _repository_error("operation_failed")
    return TTSBundleImportCommand(
        choice=command.choice,
        source_profile_id=source_profile_id,
        source_draft=source_draft,
        recipe_requirement=requirement,
        canonical_reference=canonical,
        expected_generation=expected_generation,
        reviewed_source_collisions=reviewed,
        copy_profile_id=copy_profile_id,
        copy_display_name=copy_display_name,
        dependency_state=command.dependency_state,
        inactive_consent=command.inactive_consent,
    )


def _validate_character_ref(value: object) -> CharacterRef:
    if type(value) is not _CHARACTER_REF_TYPE:
        raise _repository_error("operation_failed")
    character_ref = cast(CharacterRef, value)
    validation_error: BaseException | None = None
    validated: CharacterRef | None = None
    try:
        validated = CharacterRef(
            source=character_ref.source,
            authority_id=character_ref.authority_id,
            character_id=character_ref.character_id,
        )
        if validated != character_ref:
            raise ValueError
    except BaseException as error:
        validation_error = error
    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("operation_failed")
    assert validated is not None
    return validated


def _is_unsafe_search_character(character: str) -> bool:
    category = _unicode_category(character)
    code_point = _unicode_ord(character)
    if type(category) is not str or type(code_point) is not int:
        raise ValueError
    return (
        category in _UNSAFE_SEARCH_CATEGORIES
        or 0xFDD0 <= code_point <= 0xFDEF
        or code_point & 0xFFFF in (0xFFFE, 0xFFFF)
    )


def _normalize_search(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or len(value) > _MAX_SEARCH_CHARACTERS:
        raise _repository_error("operation_failed")

    processing_error: BaseException | None = None
    raw_unsafe = False
    normalized_unsafe = False
    trimmed = ""
    normalized: str | None = None
    normalized_byte_count: int | None = None
    try:
        raw_unsafe = any(_is_unsafe_search_character(character) for character in value)
        if not raw_unsafe:
            trimmed = value.strip()
            if trimmed:
                normalized_value = _unicode_normalize("NFKC", trimmed)
                if type(normalized_value) is not str:
                    raise ValueError
                normalized = normalized_value.casefold()
                if type(normalized) is not str:
                    raise ValueError
                normalized_unsafe = any(
                    _is_unsafe_search_character(character) for character in normalized
                )
                if not normalized_unsafe:
                    normalized_byte_count = len(normalized.encode("utf-8"))
    except BaseException as error:
        processing_error = error

    if processing_error is not None:
        if not isinstance(processing_error, Exception):
            raise processing_error
        raise _repository_error("operation_failed")
    if raw_unsafe or normalized_unsafe:
        raise _repository_error("operation_failed")
    if not trimmed:
        return None
    assert normalized is not None
    assert normalized_byte_count is not None
    if (
        len(normalized) > _MAX_NORMALIZED_SEARCH_CHARACTERS
        or normalized_byte_count > _MAX_NORMALIZED_SEARCH_BYTES
    ):
        raise _repository_error("operation_failed")
    return normalized


def _validate_page_limit(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 100:
        raise _repository_error("operation_failed")
    return cast(int, value)


def _validate_page_offset(value: object) -> int:
    if type(value) is not int or value < 0:
        raise _repository_error("operation_failed")
    return cast(int, value)


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _full_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_database_path(database_path: Path, failure_code: str) -> Path:
    """Resolve one configured database path without creating filesystem state."""

    resolution_error: BaseException | None = None
    resolved: Path | None = None
    try:
        resolved = database_path.resolve(strict=False)
    except BaseException as error:
        resolution_error = error
    if resolution_error is not None:
        if not isinstance(resolution_error, Exception):
            raise resolution_error
        raise _repository_error(failure_code)
    if type(resolved) is not _PATH_TYPE or not resolved.is_absolute():
        raise _repository_error(failure_code)
    return resolved


def _reserved_store_paths(database_path: Path) -> tuple[Path, ...]:
    return (
        database_path,
        database_path.with_name(f"{database_path.name}.lock"),
        _v2_migration_backup_path(database_path),
        _v3_migration_backup_path(database_path),
        database_path.with_name(f".{database_path.name}.migration-publication.json"),
        *(
            database_path.with_name(leaf)
            for leaf in PROFILE_MIGRATION_CANDIDATE_LEAVES.values()
        ),
        *(
            database_path.with_name(leaf)
            for leaf in PROFILE_MIGRATION_ROLLBACK_LEAVES.values()
        ),
        *(
            database_path.with_name(f"{database_path.name}{suffix}")
            for suffix in _STORE_SIDECAR_SUFFIXES
        ),
    )


def _validate_backup_destination(
    destination: object,
    database_path: Path,
) -> _DestinationSnapshot:
    """Validate and canonicalize one safe publication target."""

    if type(destination) is not _PATH_TYPE:
        raise _repository_error("backup_failed")

    validation_error: BaseException | None = None
    snapshot: _DestinationSnapshot | None = None
    try:
        exact_destination = cast(Path, destination)
        validate_path_simple(exact_destination, require_exists=False)
        if os.path.lexists(exact_destination) and exact_destination.is_symlink():
            raise ValueError
        resolved_destination = exact_destination.resolve(strict=False)
        parent = resolved_destination.parent.resolve(strict=True)
        parent_state = parent.stat()
        if not stat.S_ISDIR(parent_state.st_mode):
            raise ValueError

        reserved = _reserved_store_paths(database_path)
        if resolved_destination in reserved:
            raise ValueError
        if os.path.lexists(resolved_destination):
            destination_state = resolved_destination.stat()
            if not stat.S_ISREG(destination_state.st_mode):
                raise ValueError
            destination_identity = _stat_identity(destination_state)
            for reserved_path in reserved:
                if not os.path.lexists(reserved_path):
                    continue
                if _stat_identity(reserved_path.stat()) == destination_identity:
                    raise ValueError
        snapshot = _DestinationSnapshot(
            path=resolved_destination,
            parent_identity=_stat_identity(parent_state),
        )
    except BaseException as error:
        validation_error = error

    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("backup_failed")
    assert snapshot is not None
    return snapshot


def _validate_restore_timeout(value: object) -> float:
    if type(value) not in (int, float):
        raise _repository_error("restore_failed")
    try:
        normalized = float(cast(int | float, value))
    except Exception:
        raise _repository_error("restore_failed") from None
    if not math.isfinite(normalized) or normalized <= 0:
        raise _repository_error("restore_failed")
    return normalized


def _validate_restore_candidate_path(
    candidate: object,
    database_path: Path,
) -> _CandidateSnapshot:
    """Validate one exact non-store regular-file identity without mutation."""

    if type(candidate) is not _PATH_TYPE:
        raise _repository_error("restore_failed")

    validation_error: BaseException | None = None
    snapshot: _CandidateSnapshot | None = None
    try:
        exact_candidate = cast(Path, candidate)
        validate_path_simple(exact_candidate, require_exists=True)
        if exact_candidate.is_symlink():
            raise ValueError
        resolved_candidate = exact_candidate.resolve(strict=True)
        candidate_state = resolved_candidate.stat()
        if not stat.S_ISREG(candidate_state.st_mode):
            raise ValueError
        reserved = _reserved_store_paths(database_path)
        if resolved_candidate in reserved:
            raise ValueError
        candidate_identity = _stat_identity(candidate_state)
        for reserved_path in reserved:
            if not os.path.lexists(reserved_path):
                continue
            if _stat_identity(reserved_path.stat()) == candidate_identity:
                raise ValueError
        snapshot = _CandidateSnapshot(
            path=resolved_candidate,
            identity=_full_stat_identity(candidate_state),
        )
    except BaseException as error:
        validation_error = error

    if validation_error is not None:
        if not isinstance(validation_error, Exception):
            raise validation_error
        raise _repository_error("restore_failed")
    assert snapshot is not None
    return snapshot


def _candidate_is_unchanged(candidate: _CandidateSnapshot) -> bool:
    try:
        return (
            not candidate.path.is_symlink()
            and _full_stat_identity(candidate.path.stat()) == candidate.identity
        )
    except Exception:
        return False


def _read_monotonic() -> float:
    timing_error: BaseException | None = None
    value: object = None
    try:
        value = _monotonic()
    except BaseException as error:
        timing_error = error
    if timing_error is not None:
        if not isinstance(timing_error, Exception):
            raise timing_error
        raise _repository_error("restore_failed")
    if type(value) not in (int, float):
        raise _repository_error("restore_failed")
    normalized = float(cast(int | float, value))
    if not math.isfinite(normalized):
        raise _repository_error("restore_failed")
    return normalized


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - _read_monotonic()
    if not math.isfinite(remaining):
        raise _repository_error("restore_failed")
    return remaining


def _require_restore_time(deadline: float) -> None:
    """Require positive remaining restore time at one worker boundary."""

    if _remaining_seconds(deadline) <= 0:
        raise _repository_error("restore_failed")


def _run_with_restore_progress(
    connection: sqlite3.Connection,
    deadline: float,
    operation: Callable[[], _T],
) -> _T:
    """Run SQLite work with one deadline-aware VM progress handler."""

    callback_error: BaseException | None = None
    body_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    progress_installed = False
    result: _T | None = None

    def interrupt_after_deadline() -> int:
        nonlocal callback_error
        try:
            _require_restore_time(deadline)
        except BaseException as error:
            callback_error = error
            return 1
        return 0

    try:
        _require_restore_time(deadline)
        connection.set_progress_handler(
            interrupt_after_deadline,
            _RESTORE_PROGRESS_OPCODE_INTERVAL,
        )
        progress_installed = True
        result = operation()
        _require_restore_time(deadline)
    except BaseException as error:
        body_error = error

    if progress_installed:
        try:
            connection.set_progress_handler(None, 0)
        except BaseException as error:
            cleanup_error = error

    if callback_error is not None:
        body_error = callback_error
    _raise_with_cleanup_precedence(body_error, cleanup_error)
    return cast(_T, result)


def _unlink_path_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _escape_like_literal(value: str) -> str:
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _read_sqlite_errorcode(error: sqlite3.IntegrityError) -> int | None:
    """Read one exact extended result code without inspecting error text."""

    code: object = None
    code_error: BaseException | None = None
    try:
        code = error.sqlite_errorcode
    except BaseException as caught:
        code_error = caught

    if code_error is not None:
        if not isinstance(code_error, Exception):
            raise code_error
        return None
    if type(code) is not int:
        return None
    return cast(int, code)


def _profile_conflict_evidence(
    connection: sqlite3.Connection,
    operation_kind: _TransactionOperation,
    profile_id: UUID | None,
    normalized_name: str | None,
) -> bool | None:
    """Check exact create/update conflict rows inside the failed transaction."""

    inspection_error: BaseException | None = None
    conflict_exists: bool | None = None
    try:
        if type(profile_id) is not UUID or type(normalized_name) is not str:
            raise ValueError
        encoded_profile_id = encode_uuid(profile_id)
        if operation_kind == "create":
            row = connection.execute(
                """
                SELECT profile_id, normalized_name
                FROM tts_generation_profiles
                WHERE profile_id = ? OR normalized_name = ?
                LIMIT 1
                """,
                (encoded_profile_id, normalized_name),
            ).fetchone()
        elif operation_kind == "update":
            row = connection.execute(
                """
                SELECT profile_id, normalized_name
                FROM tts_generation_profiles
                WHERE profile_id != ? AND normalized_name = ?
                LIMIT 1
                """,
                (encoded_profile_id, normalized_name),
            ).fetchone()
        else:
            raise ValueError

        if row is None:
            conflict_exists = False
        elif len(row) == 2 and type(row[0]) is str and type(row[1]) is str:
            stored_profile_id = cast(str, row[0])
            stored_normalized_name = cast(str, row[1])
            if operation_kind == "create" and (
                stored_profile_id == encoded_profile_id
                or stored_normalized_name == normalized_name
            ):
                conflict_exists = True
            elif (
                operation_kind == "update"
                and stored_profile_id != encoded_profile_id
                and stored_normalized_name == normalized_name
            ):
                conflict_exists = True
    except BaseException as error:
        inspection_error = error

    if inspection_error is not None:
        if not isinstance(inspection_error, Exception):
            raise inspection_error
        return None
    return conflict_exists


def _profile_has_assignment(
    connection: sqlite3.Connection,
    profile_id: UUID | None,
) -> bool | None:
    """Check one exact schema-owned delete restriction, failing closed."""

    inspection_error: BaseException | None = None
    assignment_exists: bool | None = None
    try:
        if type(profile_id) is not UUID:
            raise ValueError
        encoded_profile_id = encode_uuid(profile_id)
        row = connection.execute(
            """
            SELECT profile_id
            FROM character_tts_assignments
            WHERE profile_id = ?
            LIMIT 1
            """,
            (encoded_profile_id,),
        ).fetchone()
        if row is None:
            assignment_exists = False
        elif len(row) == 1 and type(row[0]) is str and row[0] == encoded_profile_id:
            assignment_exists = True
    except BaseException as error:
        inspection_error = error

    if inspection_error is not None:
        if not isinstance(inspection_error, Exception):
            raise inspection_error
        return None
    return assignment_exists


def _has_integrity_conflict_evidence(
    connection: sqlite3.Connection,
    error: sqlite3.IntegrityError,
    operation_kind: _TransactionOperation,
    evidence: _IntegrityEvidence | None,
) -> bool:
    """Require an exact extended code and matching row under the held lock."""

    sqlite_errorcode = _read_sqlite_errorcode(error)
    if not connection.in_transaction or evidence is None:
        return False
    if operation_kind in ("create", "update") and sqlite_errorcode in (
        _SQLITE_CONSTRAINT_PRIMARYKEY,
        _SQLITE_CONSTRAINT_UNIQUE,
    ):
        return (
            _profile_conflict_evidence(
                connection,
                operation_kind,
                evidence.profile_id,
                evidence.normalized_name,
            )
            is True
        )
    if operation_kind == "delete" and sqlite_errorcode in (
        _SQLITE_CONSTRAINT_FOREIGNKEY,
        _SQLITE_CONSTRAINT_TRIGGER,
    ):
        return _profile_has_assignment(connection, evidence.profile_id) is True
    return False


def _fresh_repository_error(
    error: ProfileRepositoryError,
) -> ProfileRepositoryError:
    """Recreate one structured error without its traceback, chain, or notes."""

    if isinstance(error, ExactProfileStoreAuthorityError):
        return ExactProfileStoreAuthorityError()

    code: object = "operation_failed"
    code_error: BaseException | None = None
    try:
        code = error.code
    except BaseException as caught:
        code_error = caught
    if code_error is not None and not isinstance(code_error, Exception):
        raise code_error
    if code_error is not None or type(code) is not str:
        code = "operation_failed"
    return ProfileRepositoryError(cast(str, code))


def _raise_operation_error(error: BaseException) -> None:
    """Preserve safe repository/control-flow errors and bound every other error."""

    if not isinstance(error, Exception):
        raise error
    if isinstance(error, ProfileRepositoryError):
        raise _fresh_repository_error(error)
    raise _repository_error("operation_failed")


def _raise_with_cleanup_precedence(
    primary_error: BaseException | None,
    *cleanup_errors: BaseException | None,
) -> None:
    """Apply the hardened cleanup precedence used by adjacent profile modules."""

    if primary_error is not None and not isinstance(primary_error, Exception):
        raise primary_error
    for cleanup_error in cleanup_errors:
        if cleanup_error is not None and not isinstance(cleanup_error, Exception):
            raise cleanup_error
    if any(cleanup_error is not None for cleanup_error in cleanup_errors):
        raise _repository_error("operation_failed")
    if isinstance(primary_error, ProfileRepositoryError):
        raise _fresh_repository_error(primary_error)
    if primary_error is not None:
        raise _repository_error("operation_failed")


def _raise_cleanup_errors(*errors: BaseException | None) -> None:
    """Preserve the first control-flow cleanup signal or report a safe failure."""

    for error in errors:
        if error is not None and not isinstance(error, Exception):
            raise error
    if any(error is not None for error in errors):
        raise _repository_error("operation_failed")


def _retrieve_future_exception(future: asyncio.Future[_T]) -> None:
    """Mark one wrapper exception retrieved without changing await behavior."""

    try:
        future.exception()
    except BaseException:
        pass


class TTSProfileRepository:
    """Own one serialized profile-store connection and its lifecycle generation.

    Construction is deliberately pure. The executor, worker thread, shared
    lease, filesystem, and SQLite connection are first touched by :meth:`open`.
    """

    def __init__(
        self,
        database_path: Path,
        *,
        _clock: Callable[[], datetime] | None = None,
        _uuid_factory: Callable[[], UUID] | None = None,
    ) -> None:
        """Create an initially closed, reopenable repository.

        Args:
            database_path: Exact local profile-store path.
            _clock: Private deterministic UTC-clock seam.
            _uuid_factory: Private deterministic UUID4 seam.

        Raises:
            ProfileRepositoryError: If a constructor input is invalid.
        """

        if (
            type(database_path) is not _PATH_TYPE
            or (_clock is not None and not callable(_clock))
            or (_uuid_factory is not None and not callable(_uuid_factory))
        ):
            raise _repository_error("operation_failed")

        self._database_path = database_path
        self._clock = _utc_now if _clock is None else _clock
        self._uuid_factory = uuid4 if _uuid_factory is None else _uuid_factory
        self._state = ProfileRepositoryState.CLOSED
        self._generation = 0
        self._terminal = False
        self._state_lock = threading.Lock()
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._lifecycle_lock: asyncio.Lock | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._executor_shutdown = False
        self._connection: sqlite3.Connection | None = None
        self._lease: ProfileStoreLease | None = None
        self._exact_authority_quarantined = False
        self._restore_sidecar_identities: dict[str, os.stat_result] = {}
        self._restore_parent_authority: ParentAuthority | None = None
        self._reusable_tombstones: dict[MigrationTombstoneKey, os.stat_result] = {}
        self._active_database_path: Path | None = None
        self._residual_cleanup_paths: tuple[Path, ...] = ()
        self._damaged_reference_profile_ids: set[UUID] = set()
        self._store_established = False
        self._pending_futures: set[Future[object]] = set()
        self._open_completion: asyncio.Task[ProfileStoreResult[None]] | None = None

    @property
    def state(self) -> ProfileRepositoryState:
        """Return the current public lifecycle state."""

        with self._state_lock:
            return self._state

    @property
    def generation(self) -> int:
        """Return the current monotonic lifecycle generation."""

        with self._state_lock:
            return self._generation

    @property
    def terminal(self) -> bool:
        """Return whether definitive close has made ``closed`` terminal."""

        with self._state_lock:
            return self._terminal

    def _active_path_for_operation(self, failure_code: str) -> Path:
        """Snapshot the open lifecycle's canonical path without filesystem I/O."""

        with self._state_lock:
            state_error = self._normal_state_error_locked()
            active_path = self._active_database_path
        if state_error is not None:
            raise _repository_error(state_error)
        if active_path is None:
            raise _repository_error(failure_code)
        return active_path

    def _require_configured_path_matches(
        self,
        active_path: Path,
        failure_code: str,
    ) -> None:
        """Fail closed when the configured path no longer resolves as opened."""

        current_path = _canonical_database_path(self._database_path, failure_code)
        if current_path != active_path:
            raise _repository_error(failure_code)

    def _worker_active_path(self) -> Path:
        """Return the worker-owned canonical path without re-resolving config."""

        active_path = self._active_database_path
        if active_path is None:
            raise _repository_error("invalid_state")
        return active_path

    def _clear_reference_damage_markers(self) -> None:
        """Clear markers at one lifecycle generation boundary."""

        with self._state_lock:
            self._damaged_reference_profile_ids.clear()

    def _discard_reference_damage_marker(self, profile_id: UUID) -> None:
        """Clear one repaired or removed reference marker."""

        with self._state_lock:
            self._damaged_reference_profile_ids.discard(profile_id)

    def _reference_damage_is_marked(
        self,
        profile_id: UUID,
        generation: int,
    ) -> bool:
        """Return whether this exact generation isolated the reference."""

        with self._state_lock:
            return (
                self._generation == generation
                and profile_id in self._damaged_reference_profile_ids
            )

    def _mark_reference_damage(self, profile_id: UUID, generation: int) -> None:
        """Mark row-local damage only while its admitted generation is live."""

        with self._state_lock:
            if (
                self._generation == generation
                and not self._terminal
                and self._state is ProfileRepositoryState.OPEN
            ):
                self._damaged_reference_profile_ids.add(profile_id)

    async def open(self) -> ProfileStoreResult[None]:
        """Open the profile store or retry one unavailable open attempt.

        Returns:
            The active lifecycle generation with a ``None`` value.

        Raises:
            ProfileRepositoryError: If the state is invalid, the store cannot
                be opened safely, or the repository was definitively closed.
            BaseException: A worker control-flow signal, after partial
                ownership has been cleaned.
        """

        lifecycle_lock = self._bind_or_check_loop()
        with self._state_lock:
            shared_completion = self._open_completion
        if shared_completion is not None:
            return await self._await_open_completion(shared_completion)

        async with lifecycle_lock:
            with self._state_lock:
                if self._terminal:
                    raise _repository_error("terminal")
                if self._state is ProfileRepositoryState.OPEN:
                    return ProfileStoreResult(
                        generation=self._generation,
                        value=None,
                    )
                state_error = self._open_state_error_locked()
                if state_error is not None:
                    raise _repository_error(state_error)
                self._generation += 1
                self._damaged_reference_profile_ids.clear()
                generation = self._generation
                executor = self._executor

            if executor is None:
                executor_error: BaseException | None = None
                created_executor: ThreadPoolExecutor | None = None
                try:
                    created_executor = ThreadPoolExecutor(max_workers=1)
                except BaseException as error:
                    executor_error = error

                if executor_error is not None:
                    with self._state_lock:
                        self._state = ProfileRepositoryState.UNAVAILABLE
                    _raise_operation_error(executor_error)
                assert created_executor is not None
                with self._state_lock:
                    self._executor = created_executor
                    self._executor_shutdown = False
                executor = created_executor

            submission_error: BaseException | None = None
            open_future: Future[None] | None = None
            try:
                open_future = executor.submit(self._worker_open)
            except BaseException as error:
                submission_error = error

            if submission_error is not None:
                with self._state_lock:
                    self._state = ProfileRepositoryState.UNAVAILABLE
                _raise_operation_error(submission_error)
            assert open_future is not None

            completion = asyncio.create_task(self._finish_open(generation, open_future))
            with self._state_lock:
                self._open_completion = completion
            return await self._await_open_completion(completion)

    async def _await_open_completion(
        self,
        completion: asyncio.Task[ProfileStoreResult[None]],
    ) -> ProfileStoreResult[None]:
        """Join one open attempt and clear its marker only after settlement."""

        self._bind_or_check_loop()
        try:
            return await self._await_lifecycle_completion(completion)
        finally:
            with self._state_lock:
                if completion.done() and self._open_completion is completion:
                    self._open_completion = None

    def _open_state_error_locked(self) -> str | None:
        if self._state is ProfileRepositoryState.RESTORING:
            return "restoring"
        if self._state not in (
            ProfileRepositoryState.CLOSED,
            ProfileRepositoryState.UNAVAILABLE,
        ):
            return "invalid_state"
        if self._executor_shutdown:
            return "terminal"
        return None

    async def _finish_open(
        self,
        generation: int,
        open_future: Future[None],
    ) -> ProfileStoreResult[None]:
        self._bind_or_check_loop()
        open_error: BaseException | None = None
        try:
            await asyncio.wrap_future(open_future)
        except BaseException as error:
            open_error = error

        with self._state_lock:
            generation_changed = self._generation != generation or self._terminal
            if open_error is None and not generation_changed:
                self._state = ProfileRepositoryState.OPEN
            else:
                self._state = ProfileRepositoryState.UNAVAILABLE

        if open_error is not None:
            _raise_operation_error(open_error)
        if generation_changed:
            raise _repository_error("stale")
        return ProfileStoreResult(generation=generation, value=None)

    def _worker_open(self) -> None:
        """Acquire shared ownership and open the long-lived connection."""

        self._clear_reference_damage_markers()
        if self._connection is not None or self._lease is not None:
            self._worker_cleanup()

        lease: ProfileStoreLease | None = None
        connection: sqlite3.Connection | None = None
        active_path: Path | None = None
        body_error: BaseException | None = None
        try:
            active_path = _canonical_database_path(
                self._database_path,
                "operation_failed",
            )
            expected_post_init_authority: PostInitProfileStoreAuthority | None = None
            shared = self._worker_open_if_proven_current(active_path)
            if shared is None:
                expected_post_init_authority = self._worker_initialize_store(
                    active_path,
                    allow_create=not self._store_established,
                )
                shared = self._worker_open_if_proven_current(
                    active_path,
                    expected_post_init_authority=expected_post_init_authority,
                )
                if shared is None:
                    raise _repository_error("operation_failed")
                lease, connection = shared
            else:
                lease, connection = shared
            if connection is None:
                raise _repository_error("operation_failed")
            validate_profile_store_rows(connection)
            self._require_configured_path_matches(
                active_path,
                "operation_failed",
            )
            revalidate_exact_current_profile_store(connection, active_path)
        except BaseException as error:
            body_error = error

        if body_error is None:
            assert lease is not None
            assert active_path is not None
            self._lease = lease
            self._connection = connection
            self._active_database_path = active_path
            self._store_established = True
            return

        connection_error: BaseException | None = None
        lease_error: BaseException | None = None
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                connection_error = error
                self._connection = connection
        if lease is not None and connection_error is not None:
            self._lease = lease
        elif lease is not None:
            try:
                lease.release()
            except BaseException as error:
                lease_error = error
                self._lease = lease
        if self._connection is not None or self._lease is not None:
            self._active_database_path = active_path
        else:
            self._active_database_path = None
        _raise_with_cleanup_precedence(
            body_error,
            connection_error,
            lease_error,
        )

    def _worker_open_if_proven_current(
        self,
        active_path: Path,
        *,
        expected_post_init_authority: PostInitProfileStoreAuthority | None = None,
    ) -> tuple[ProfileStoreLease, sqlite3.Connection] | None:
        """Open exact v4 while continuously holding one shared gate."""

        lease = ProfileStoreLease(active_path, ProfileStoreLockMode.SHARED)
        connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        release_error: BaseException | None = None
        try:
            lease.acquire()
            parent_fd, _leaf = private_paths._open_verified_parent(
                active_path,
                missing_leaf_allowed=True,
            )
            try:
                journal_leaf = f".{active_path.name}.migration-publication.json"
                for suffix in ("", *_STORE_SIDECAR_SUFFIXES):
                    try:
                        os.stat(
                            f"{journal_leaf}{suffix}",
                            dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        continue
                    self._worker_release_unproven_shared(lease, active_path)
                    return None
            finally:
                os.close(parent_fd)
            try:
                connection = open_exact_current_profile_store(
                    active_path,
                    expected_post_init_authority=expected_post_init_authority,
                )
            except ExactProfileStoreNotCurrentError:
                self._worker_release_unproven_shared(lease, active_path)
                return None
            except ExactProfileStoreCleanupError as error:
                connection = cast(sqlite3.Connection, error.connection)
                raise
            return lease, connection
        except BaseException as error:
            body_error = error

        if self._connection is not None:
            self._lease = lease
            self._active_database_path = active_path
            _raise_with_cleanup_precedence(body_error)
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                self._worker_retain_failed_connection(connection, active_path)
                self._lease = lease
                _raise_with_cleanup_precedence(body_error, error)
        try:
            lease.release()
        except BaseException as error:
            release_error = error
            self._lease = lease
            self._active_database_path = active_path
        _raise_with_cleanup_precedence(body_error, release_error)
        raise AssertionError("unreachable")

    def _worker_release_unproven_shared(
        self,
        lease: ProfileStoreLease,
        active_path: Path,
    ) -> None:
        try:
            lease.release()
        except BaseException:
            self._lease = lease
            self._active_database_path = active_path
            raise
        return None

    def _worker_retain_failed_connection(
        self,
        connection: sqlite3.Connection,
        active_path: Path,
    ) -> None:
        """Retain a live connection whose close did not complete."""

        if self._connection is not None and self._connection is not connection:
            raise _repository_error("unavailable")
        self._connection = connection
        self._active_database_path = active_path

    def _worker_initialize_store(
        self,
        active_path: Path,
        *,
        allow_create: bool = True,
    ) -> PostInitProfileStoreAuthority:
        """Create or migrate one store only while holding exclusive ownership."""

        lease = ProfileStoreLease(
            active_path,
            ProfileStoreLockMode.EXCLUSIVE,
            timeout_seconds=_INITIALIZATION_LOCK_TIMEOUT_SECONDS,
        )
        connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        connection_error: BaseException | None = None
        lease_error: BaseException | None = None
        authority: PostInitProfileStoreAuthority | None = None
        try:
            lease.acquire()
            recover_profile_migration_publication(active_path)
            source_version = self._worker_exact_schema_version(active_path)
            if source_version is None:
                if not allow_create:
                    raise _repository_error("missing")
                connection = open_profile_store(active_path)
            elif source_version < CURRENT_PROFILE_SCHEMA_VERSION:
                self._worker_publish_migrated_store(active_path, source_version)
                connection = open_profile_store(active_path, must_exist=True)
            elif source_version == CURRENT_PROFILE_SCHEMA_VERSION:
                connection = open_profile_store(active_path, must_exist=True)
            else:
                raise _repository_error("schema_unsupported")
            validate_profile_store_rows(connection)
            self._require_configured_path_matches(
                active_path,
                "operation_failed",
            )
        except BaseException as error:
            body_error = error

        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                connection_error = error
                self._connection = connection
                self._lease = lease
                self._active_database_path = active_path
        if body_error is None and connection_error is None:
            try:
                authority = capture_post_init_profile_store_authority(active_path)
            except BaseException as error:
                body_error = error
        if self._connection is not None:
            self._lease = lease
            self._active_database_path = active_path
        elif connection_error is None:
            try:
                lease.release()
            except BaseException as error:
                lease_error = error
                self._lease = lease
                self._active_database_path = active_path
        _raise_with_cleanup_precedence(
            body_error,
            connection_error,
            lease_error,
        )
        assert authority is not None
        return authority

    def _worker_publish_migrated_store(
        self,
        active_path: Path,
        source_version: int,
        *,
        source_path: Path | None = None,
        stage_hook: Callable[[ProfileMigrationPublicationStage], None] | None = None,
        progress_guard: Callable[[], None] | None = None,
    ) -> None:
        """Prepare and transactionally publish one exact v4 candidate set."""

        slot_for_boundary = {
            ProfileMigrationBoundary.PRE_V3: ProfileMigrationPublicationSlot.PRE_V3,
            ProfileMigrationBoundary.PRE_V4: ProfileMigrationPublicationSlot.PRE_V4,
        }
        tombstone_for_slot = {
            ProfileMigrationPublicationSlot.ACTIVE: MigrationTombstoneKey.ACTIVE_CANDIDATE,
            ProfileMigrationPublicationSlot.PRE_V3: MigrationTombstoneKey.PRE_V3_CANDIDATE,
            ProfileMigrationPublicationSlot.PRE_V4: MigrationTombstoneKey.PRE_V4_CANDIDATE,
        }
        backup_path_for_slot = {
            ProfileMigrationPublicationSlot.PRE_V3: _v2_migration_backup_path(
                active_path
            ),
            ProfileMigrationPublicationSlot.PRE_V4: _v3_migration_backup_path(
                active_path
            ),
        }
        active_candidate_path = active_path.with_name(
            PROFILE_MIGRATION_CANDIDATE_LEAVES[ProfileMigrationPublicationSlot.ACTIVE]
        )
        owners: list[
            tuple[ProfileMigrationBoundaryDestination, MigrationTombstoneKey]
        ] = []
        publication_started = False
        source: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        try:
            self._worker_prepare_reusable_tombstones(active_path)
            if source_path is None:
                source = connect_private_sqlite(
                    "tts.profile_migration_backup",
                    active_path,
                    read_only=True,
                    must_exist=True,
                    isolation_level=None,
                )
            else:
                source = connect_private_sqlite(
                    "tts.profile_restore_stage",
                    source_path,
                    read_only=True,
                    must_exist=True,
                    isolation_level=None,
                )
            source.row_factory = sqlite3.Row
            source.execute("PRAGMA foreign_keys = ON")
            if source_version == 0:
                if _profile_schema._user_schema_objects(source):
                    raise _repository_error("schema_partial")
                try:
                    _profile_schema._validate_full_integrity(source)
                except Exception:
                    raise _repository_error("schema_corrupt") from None
            else:
                _profile_schema.validate_profile_store_version(
                    source,
                    source_version,
                )
            active_owner = open_canonical_profile_migration_destination(
                active_candidate_path,
                schema_version=source_version,
                tombstone_key=tombstone_for_slot[
                    ProfileMigrationPublicationSlot.ACTIVE
                ],
            )
            owners.append(
                (
                    active_owner,
                    tombstone_for_slot[ProfileMigrationPublicationSlot.ACTIVE],
                )
            )
            boundary_owners: dict[
                ProfileMigrationPublicationSlot,
                ProfileMigrationBoundaryDestination,
            ] = {}

            def validate_candidate(candidate: sqlite3.Connection) -> None:
                _profile_schema.validate_profile_store_version(
                    candidate,
                    CURRENT_PROFILE_SCHEMA_VERSION,
                )
                validate_profile_store_rows(candidate)
                validate_reference_rows(candidate)

            def capture_boundary(
                snapshot: ProfileMigrationBoundarySnapshot,
                request: ProfileMigrationBoundaryRequest,
            ) -> None:
                slot = slot_for_boundary[request.kind]
                candidate_path = active_path.with_name(
                    PROFILE_MIGRATION_CANDIDATE_LEAVES[slot]
                )
                owner = open_canonical_profile_migration_destination(
                    candidate_path,
                    schema_version=request.schema_version,
                    tombstone_key=tombstone_for_slot[slot],
                )
                owners.append((owner, tombstone_for_slot[slot]))
                snapshot.backup_to(owner)
                boundary_owners[slot] = owner

            migrate_profile_store_to_candidate(
                source,
                active_owner,
                migrate=lambda candidate: step_profile_migration_candidate(
                    candidate,
                    boundary_sink=capture_boundary,
                ),
                validate=validate_candidate,
                progress_guard=progress_guard,
            )
            source.close()
            source = None

            active_artifact = prepare_profile_migration_artifact(
                active_candidate_path,
                slot=ProfileMigrationPublicationSlot.ACTIVE,
            )
            ordered_slots = tuple(
                slot
                for slot in (
                    ProfileMigrationPublicationSlot.PRE_V3,
                    ProfileMigrationPublicationSlot.PRE_V4,
                )
                if slot in boundary_owners
            )
            backup_artifacts = tuple(
                prepare_profile_migration_artifact(
                    active_path.with_name(PROFILE_MIGRATION_CANDIDATE_LEAVES[slot]),
                    slot=slot,
                )
                for slot in ordered_slots
            )
            active_destination = retain_profile_migration_destination(
                active_path,
                slot=ProfileMigrationPublicationSlot.ACTIVE,
                must_exist=True,
            )
            backup_destinations = tuple(
                retain_profile_migration_destination(
                    backup_path_for_slot[slot],
                    slot=slot,
                    must_exist=backup_path_for_slot[slot].exists(),
                )
                for slot in ordered_slots
            )
            publication_started = True
            publish_profile_migration(
                active_candidate=active_artifact,
                backup_candidates=backup_artifacts,
                active_destination=active_destination,
                backup_destinations=backup_destinations,
                stage_hook=stage_hook,
            )
            self._worker_refresh_reusable_tombstones(active_path)
        except BaseException as error:
            body_error = error

        if body_error is not None and publication_started:
            try:
                self._worker_refresh_reusable_tombstones(active_path)
            except BaseException as error:
                cleanup_errors: list[BaseException] = [error]
            else:
                cleanup_errors = []
        else:
            cleanup_errors = []

        if source is not None:
            try:
                source.close()
            except BaseException as error:
                cleanup_errors.append(error)
                self._worker_retain_failed_connection(source, active_path)
                source = None
        if body_error is not None and not publication_started:
            for owner, tombstone_key in reversed(owners):
                try:
                    discard_profile_migration_destination(
                        owner,
                        tombstone_key=tombstone_key,
                    )
                    holding = active_path.with_name(
                        f".profile-migration-{tombstone_key.value}.tombstone"
                    )
                    self._reusable_tombstones[tombstone_key] = holding.stat()
                except BaseException as error:
                    cleanup_errors.append(error)
        for owner, _tombstone_key in reversed(owners):
            try:
                close_profile_migration_destination(owner)
            except BaseException as error:
                cleanup_errors.append(error)
        for pending_error in (body_error, *cleanup_errors):
            if pending_error is not None and not isinstance(pending_error, Exception):
                raise pending_error
        if body_error is not None or cleanup_errors:
            if isinstance(body_error, ProfileRepositoryError):
                raise _repository_error(body_error.code)
            raise _repository_error("migration_failed")

    def _worker_prepare_reusable_tombstones(self, active_path: Path) -> None:
        """Turn retained exact cleanup evidence into zero reusable leaves."""

        parent = ParentAuthority(active_path.parent.stat())
        known = self._reusable_tombstones
        for key in MigrationTombstoneKey:
            holding = active_path.with_name(f".profile-migration-{key.value}.tombstone")
            try:
                observed = holding.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            expected = known.get(key)
            if expected is None:
                try:
                    known[key] = admit_zero_reusable_tombstone(
                        active_path,
                        parent_authority=parent,
                        tombstone_key=key,
                    )
                except Exception:
                    raise _repository_error("migration_failed") from None
            elif not private_paths._same_identity(observed, expected):
                raise _repository_error("migration_failed")
        moves = (
            (
                MigrationTombstoneKey.ACTIVE_ROLLBACK,
                MigrationTombstoneKey.ACTIVE_CANDIDATE,
            ),
            (
                MigrationTombstoneKey.PRE_V3_ROLLBACK,
                MigrationTombstoneKey.PRE_V3_CANDIDATE,
            ),
            (
                MigrationTombstoneKey.PRE_V4_ROLLBACK,
                MigrationTombstoneKey.PRE_V4_CANDIDATE,
            ),
            (MigrationTombstoneKey.JOURNAL, MigrationTombstoneKey.JOURNAL),
            (MigrationTombstoneKey.LIVE_WAL, MigrationTombstoneKey.LIVE_WAL),
            (MigrationTombstoneKey.LIVE_SHM, MigrationTombstoneKey.LIVE_SHM),
        )
        for source_key, destination_key in moves:
            expected = known.pop(source_key, None)
            if expected is None:
                continue
            try:
                prepared = prepare_reusable_tombstone(
                    active_path,
                    parent_authority=parent,
                    file_identity=expected,
                    source_key=source_key,
                    destination_key=destination_key,
                )
                if destination_key in {
                    MigrationTombstoneKey.LIVE_WAL,
                    MigrationTombstoneKey.LIVE_SHM,
                }:
                    remove_zero_reusable_tombstone(
                        active_path,
                        parent_authority=parent,
                        file_identity=prepared,
                        tombstone_key=destination_key,
                    )
                else:
                    known[destination_key] = prepared
            except Exception:
                raise _repository_error("migration_failed") from None

    def _worker_refresh_reusable_tombstones(self, active_path: Path) -> None:
        """Retain exact identities produced by a completed publication."""

        refreshed: dict[MigrationTombstoneKey, os.stat_result] = {}
        for key in MigrationTombstoneKey:
            holding = active_path.with_name(f".profile-migration-{key.value}.tombstone")
            try:
                observed = holding.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (
                private_paths._classify_private_file_stat(
                    observed,
                    expected_uid=os.geteuid(),
                )
                is not None
                or stat.S_IMODE(observed.st_mode) != 0o600
                or observed.st_nlink != 1
            ):
                raise _repository_error("migration_failed")
            refreshed[key] = observed
        self._reusable_tombstones = refreshed

    def _worker_settle_reusable_tombstones(
        self,
        active_path: Path,
        parent_fd: int,
    ) -> None:
        """Make retained cleanup evidence restart-safe before lease release."""

        parent = ParentAuthority(os.fstat(parent_fd))
        known = self._reusable_tombstones
        for key in MigrationTombstoneKey:
            holding = active_path.with_name(f".profile-migration-{key.value}.tombstone")
            try:
                observed = holding.stat(follow_symlinks=False)
            except FileNotFoundError:
                if key in known:
                    raise _repository_error("operation_failed")
                continue
            expected = known.get(key)
            if expected is None or not private_paths._same_identity(observed, expected):
                raise _repository_error("operation_failed")
            try:
                known[key] = prepare_reusable_tombstone(
                    active_path,
                    parent_authority=parent,
                    file_identity=expected,
                    source_key=key,
                    destination_key=key,
                )
            except Exception:
                raise _repository_error("operation_failed") from None

    def _worker_exact_schema_version(self, active_path: Path) -> int | None:
        """Read the exact version without authorizing migration or creation."""

        if not active_path.exists():
            return None
        connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        close_error: BaseException | None = None
        version: int | None = None
        try:
            connection = connect_private_sqlite(
                "tts.profile_migration_backup",
                active_path,
                read_only=True,
                must_exist=True,
                isolation_level=None,
            )
            value = connection.execute("PRAGMA user_version").fetchone()[0]
            if type(value) is not int:
                raise ValueError
            version = value
        except BaseException as error:
            body_error = error
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                close_error = error
                self._worker_retain_failed_connection(connection, active_path)
        for pending_error in (body_error, close_error):
            if pending_error is not None and not isinstance(pending_error, Exception):
                raise pending_error
        if body_error is not None or close_error is not None or version is None:
            raise _repository_error("migration_failed")
        return version

    async def create_profile(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID | None = None,
        *,
        expected_generation: int | None = None,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        """Create one immutable profile at revision 1.

        Args:
            draft: Exact validated profile draft.
            profile_id: Optional exact caller-selected UUID. When omitted, the
                repository generates a UUID4 on its serialized worker.
            expected_generation: Optional exact lifecycle generation when the
                create is derived from caller-held repository state.

        Returns:
            The active generation and exact persisted profile.

        Raises:
            ProfileRepositoryError: If inputs, state, persistence, or
                uniqueness checks fail safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_draft = _validate_draft(draft)
        validated_profile_id = _validate_optional_profile_id(profile_id)
        validated_generation = (
            None
            if expected_generation is None
            else _validate_expected_generation(expected_generation)
        )
        return await self._submit_operation(
            lambda connection: self._worker_create_profile(
                connection,
                validated_draft,
                validated_profile_id,
            ),
            expected_generation=validated_generation,
        )

    async def create_profile_with_reference(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID,
        canonical: CanonicalTTSCloneReference,
        recipe_requirement: TTSCloneRecipeRequirement,
        *,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        """Atomically create one profile and its canonical clone reference.

        Args:
            draft: Exact validated generation-profile draft.
            profile_id: Exact caller-selected UUID for the new profile.
            canonical: Fully validated source-independent clone reference.
            expected_generation: Exact active repository generation.

        Returns:
            The active generation and committed revision-2 profile summary.

        Raises:
            ProfileRepositoryError: If validation, freshness, uniqueness,
                quota, persistence, or round-trip verification fails.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_draft = _validate_draft(draft)
        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_reference = _validate_canonical_reference(canonical)
        validated_requirement = _validate_recipe_requirement(
            recipe_requirement,
            model_id=validated_draft.model_id,
        )
        validated_generation = _validate_expected_generation(expected_generation)
        return await self._submit_operation(
            lambda connection: self._worker_create_profile_with_reference(
                connection,
                validated_draft,
                validated_profile_id,
                validated_reference,
                validated_requirement,
                validated_generation,
            ),
            expected_generation=validated_generation,
        )

    async def create_profile_with_assignment(
        self,
        draft: TTSProfileDraft,
        profile_id: UUID,
        character_ref: CharacterRef,
        *,
        expected_generation: int,
        expected_current_profile_id: UUID | None,
    ) -> ProfileStoreResult[AssignedTTSProfileSnapshot]:
        """Atomically create one profile and set one exact assignment."""

        validated_draft = _validate_draft(draft)
        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_character_ref = _validate_character_ref(character_ref)
        validated_generation = _validate_expected_generation(expected_generation)
        validated_current_profile_id = _validate_optional_profile_id(
            expected_current_profile_id
        )
        return await self._submit_operation(
            lambda connection: self._worker_create_profile_with_assignment(
                connection,
                validated_draft,
                validated_profile_id,
                validated_character_ref,
                validated_generation,
                validated_current_profile_id,
            ),
            expected_generation=validated_generation,
        )

    async def commit_bundle_import(
        self,
        command: TTSBundleImportCommand,
    ) -> ProfileStoreResult[TTSBundleImportResult]:
        """Recheck and commit one explicit bundle decision in one transaction."""

        validated = _validate_bundle_import_command(command)
        return await self._submit_operation(
            lambda connection: self._worker_commit_bundle_import(
                connection,
                validated,
            ),
            expected_generation=validated.expected_generation,
        )

    async def get_profile(
        self,
        profile_id: UUID,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        """Load and fully decode one profile by exact UUID.

        Args:
            profile_id: Exact profile UUID.

        Returns:
            The active generation and immutable decoded profile.

        Raises:
            ProfileRepositoryError: If the input, state, row, or SQLite access
                fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_profile_id = _validate_exact_profile_id(profile_id)
        return await self._submit_operation(
            lambda connection: self._worker_get_profile(
                connection,
                validated_profile_id,
            )
        )

    async def get_profile_collisions(
        self,
        profile_id: UUID,
        draft: TTSProfileDraft,
    ) -> ProfileStoreResult[TTSProfileCollisionSnapshot]:
        """Read exact rows matching a portable UUID hint or normalized name."""

        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_draft = _validate_draft(draft)
        return await self._submit_operation(
            lambda connection: self._worker_get_profile_collisions(
                connection,
                validated_profile_id,
                validated_draft.normalized_name,
            )
        )

    async def list_profiles(
        self,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ProfileStoreResult[TTSProfilePage]:
        """List one stable bounded page and the full filtered result count.

        Search is trimmed, normalized with Unicode NFKC, and case-folded like
        persisted profile names. Empty or whitespace-only search lists all
        profiles. SQL LIKE metacharacters and the explicit escape character
        are always treated literally.

        Args:
            search: Optional exact string of at most 128 characters whose
                normalized form remains within the repository's bounded
                search policy.
            limit: Exact integer page size from 1 through 100.
            offset: Exact nonnegative integer result offset.

        Returns:
            The active generation and an immutable profile page.

        Raises:
            ProfileRepositoryError: If inputs, state, decoding, or SQLite
                access fail safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        normalized_search = _normalize_search(search)
        validated_limit = _validate_page_limit(limit)
        validated_offset = _validate_page_offset(offset)
        return await self._submit_operation(
            lambda connection: self._worker_list_profiles(
                connection,
                normalized_search,
                validated_limit,
                validated_offset,
            )
        )

    async def update_profile(
        self,
        profile_id: UUID,
        expected_revision: int,
        draft: TTSProfileDraft,
        *,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        """Replace one profile only at the exact editor revision.

        Args:
            profile_id: Exact profile UUID.
            expected_revision: Exact positive revision loaded by the editor.
            draft: Exact replacement profile draft.
            expected_generation: Exact nonnegative lifecycle generation loaded
                by the editor.

        Returns:
            The active generation and immutable updated profile.

        Raises:
            ProfileRepositoryError: If inputs, state, optimistic revision,
                uniqueness, row decoding, or SQLite access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_revision = _validate_expected_revision(expected_revision)
        validated_draft = _validate_draft(draft)
        validated_generation = _validate_expected_generation(expected_generation)
        return await self._submit_operation(
            lambda connection: self._worker_update_profile(
                connection,
                validated_profile_id,
                validated_revision,
                validated_draft,
            ),
            expected_generation=validated_generation,
        )

    async def delete_profile(
        self,
        profile_id: UUID,
        *,
        expected_generation: int,
    ) -> ProfileStoreResult[None]:
        """Delete exactly one unreferenced profile by UUID.

        Args:
            profile_id: Exact profile UUID.
            expected_generation: Exact nonnegative lifecycle generation loaded
                with the profile.

        Returns:
            The active generation paired with ``None``.

        Raises:
            ProfileRepositoryError: If the input or state is invalid, the row
                is missing or referenced, or SQLite access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_generation = _validate_expected_generation(expected_generation)
        return await self._submit_operation(
            lambda connection: self._worker_delete_profile(
                connection,
                validated_profile_id,
            ),
            expected_generation=validated_generation,
        )

    async def set_reference(
        self,
        profile_id: UUID,
        canonical: CanonicalTTSCloneReference,
        recipe_requirement: TTSCloneRecipeRequirement,
        *,
        expected_revision: int,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        """Atomically attach or replace one profile-owned clone reference.

        Args:
            profile_id: Exact profile UUID that owns the reference.
            canonical: Fully validated canonical clone-reference payload.
            expected_revision: Exact profile revision required for the update.
            expected_generation: Exact active repository generation.

        Returns:
            The active generation and updated profile.

        Raises:
            ProfileRepositoryError: If input, state, generation, revision, or
                SQLite access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_reference = _validate_canonical_reference(canonical)
        validated_requirement = _validate_recipe_requirement(
            recipe_requirement,
        )
        validated_revision = _validate_expected_revision(expected_revision)
        validated_generation = _validate_expected_generation(expected_generation)
        return await self._submit_operation(
            lambda connection: self._worker_set_reference(
                connection,
                validated_profile_id,
                validated_reference,
                validated_requirement,
                validated_revision,
            ),
            expected_generation=validated_generation,
        )

    async def remove_reference(
        self,
        profile_id: UUID,
        *,
        expected_revision: int,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSGenerationProfile]:
        """Atomically remove one profile-owned clone reference."""

        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_revision = _validate_expected_revision(expected_revision)
        validated_generation = _validate_expected_generation(expected_generation)
        return await self._submit_operation(
            lambda connection: self._worker_remove_reference(
                connection,
                validated_profile_id,
                validated_revision,
            ),
            expected_generation=validated_generation,
        )

    async def get_reference(
        self,
        profile_id: UUID,
        *,
        expected_revision: int,
        expected_generation: int,
    ) -> ProfileStoreResult[TTSCloneReference]:
        """Stream and fully revalidate one exact stored clone reference."""

        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_revision = _validate_expected_revision(expected_revision)
        validated_generation = _validate_expected_generation(expected_generation)
        return await self._submit_operation(
            lambda connection: self._worker_get_reference(
                connection,
                validated_profile_id,
                validated_revision,
                validated_generation,
            ),
            expected_generation=validated_generation,
        )

    async def assignment_count(
        self,
        profile_id: UUID,
    ) -> ProfileStoreResult[int]:
        """Count assignments to one existing profile across all authorities.

        Args:
            profile_id: Exact profile UUID.

        Returns:
            The active generation and nonnegative assignment count.

        Raises:
            ProfileRepositoryError: If the input, profile, count row, state, or
                SQLite access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_profile_id = _validate_exact_profile_id(profile_id)
        return await self._submit_operation(
            lambda connection: self._worker_assignment_count(
                connection,
                validated_profile_id,
            )
        )

    async def set_assignment(
        self,
        character_ref: CharacterRef,
        profile_id: UUID,
        *,
        expected_generation: int,
        expected_profile_revision: int,
        expected_current_profile_id: UUID | None,
        expected_profile: TTSGenerationProfile | None = None,
    ) -> ProfileStoreResult[CharacterTTSAssignment]:
        """Create or replace one exact authority-scoped assignment.

        Args:
            character_ref: Exact validated source, authority, and character.
            profile_id: Exact existing profile UUID.
            expected_generation: Exact nonnegative lifecycle generation loaded
                with the selected profile and current assignment.
            expected_profile_revision: Exact positive revision of the selected
                profile.
            expected_current_profile_id: Exact currently assigned profile UUID,
                or ``None`` when the character was observed as unassigned.
            expected_profile: Optional exact immutable selected-profile snapshot.
                When supplied, a delete/recreate with the same UUID and revision
                is rejected as a conflict.

        Returns:
            The active generation and persisted assignment.

        Raises:
            ProfileRepositoryError: If inputs, state, optimistic expectations,
                persistence, foreign-key checks, row decoding, or SQLite
                access fail safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_character_ref = _validate_character_ref(character_ref)
        validated_profile_id = _validate_exact_profile_id(profile_id)
        validated_generation = _validate_expected_generation(expected_generation)
        validated_profile_revision = _validate_expected_revision(
            expected_profile_revision
        )
        validated_current_profile_id = _validate_optional_profile_id(
            expected_current_profile_id
        )
        validated_expected_profile = _validate_optional_profile(expected_profile)
        return await self._submit_operation(
            lambda connection: self._worker_set_assignment(
                connection,
                validated_character_ref,
                validated_profile_id,
                validated_generation,
                validated_profile_revision,
                validated_current_profile_id,
                validated_expected_profile,
            ),
            expected_generation=validated_generation,
        )

    async def remove_assignment(
        self,
        character_ref: CharacterRef,
        *,
        expected_generation: int,
        expected_profile_id: UUID,
    ) -> ProfileStoreResult[None]:
        """Remove one exact authority-scoped assignment idempotently.

        Args:
            character_ref: Exact validated source, authority, and character.
            expected_generation: Exact nonnegative lifecycle generation loaded
                with the assignment.
            expected_profile_id: Exact profile UUID observed on the assignment.

        Returns:
            The active generation paired with ``None``.

        Raises:
            ProfileRepositoryError: If an input, state, optimistic expectation,
                persistence, or SQLite access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_character_ref = _validate_character_ref(character_ref)
        validated_generation = _validate_expected_generation(expected_generation)
        validated_profile_id = _validate_exact_profile_id(expected_profile_id)
        return await self._submit_operation(
            lambda connection: self._worker_remove_assignment(
                connection,
                validated_character_ref,
                validated_generation,
                validated_profile_id,
            ),
            expected_generation=validated_generation,
        )

    async def get_assigned_profile(
        self,
        character_ref: CharacterRef,
    ) -> ProfileStoreResult[AssignedTTSProfileSnapshot | None]:
        """Read one exact assignment and immutable profile revision by JOIN.

        Args:
            character_ref: Exact validated source, authority, and character.

        Returns:
            The active generation and joined snapshot, or ``None`` when the
            exact character is unassigned.

        Raises:
            ProfileRepositoryError: If the input, state, joined row, or SQLite
                access fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        validated_character_ref = _validate_character_ref(character_ref)
        return await self._submit_operation(
            lambda connection: self._worker_get_assigned_profile(
                connection,
                validated_character_ref,
            )
        )

    async def backup_to(
        self,
        destination: Path,
        *,
        timeout_seconds: int | float = 5.0,
    ) -> ProfileStoreResult[ProfileBackupReceipt]:
        """Publish one validated SQLite online-backup snapshot atomically.

        Args:
            destination: Exact non-store path for the completed snapshot.
            timeout_seconds: Positive finite backup and qualification budget.

        Returns:
            The active generation and safe backup metadata.

        Raises:
            ProfileRepositoryError: If path admission, state, backup,
                validation, or atomic publication fails safely.
            BaseException: A caller control-flow signal preserved by the
                serialized operation lane.
        """

        timing_error: BaseException | None = None
        timeout: float | None = None
        deadline: float | None = None
        try:
            timeout = _validate_restore_timeout(timeout_seconds)
            deadline = _read_monotonic() + timeout
            if not math.isfinite(deadline):
                raise ValueError
        except BaseException as error:
            timing_error = error
        if timing_error is not None and not isinstance(timing_error, Exception):
            raise timing_error
        if timing_error is not None:
            raise _repository_error("backup_failed")
        assert timeout is not None
        assert deadline is not None
        if type(destination) is not _PATH_TYPE:
            raise _repository_error("backup_failed")
        exact_destination = cast(Path, destination)
        active_path = self._active_path_for_operation("backup_failed")
        return await self._submit_operation(
            lambda connection: self._worker_backup_to(
                connection,
                exact_destination,
                active_path,
                deadline,
            )
        )

    def _worker_backup_to(
        self,
        connection: sqlite3.Connection,
        destination_path: Path,
        active_path: Path,
        deadline: float,
    ) -> ProfileBackupReceipt:
        """Create and atomically publish one worker-owned online backup."""

        destination = _validate_backup_destination(destination_path, active_path)
        temporary_path: Path | None = None
        destination_connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        published = False
        receipt: ProfileBackupReceipt | None = None
        try:
            _require_restore_time(deadline)
            if self._worker_active_path() != active_path:
                raise _repository_error("backup_failed")
            self._require_configured_path_matches(active_path, "backup_failed")
            # Validate the clock before any destination publication.
            created_at = self._clock()
            ProfileBackupReceipt(created_at=created_at, byte_count=0)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.path.name}.",
                suffix=".backup",
                dir=destination.path.parent,
            )
            temporary_path = Path(temporary_name)
            os.close(descriptor)
            destination_connection = connect_private_sqlite(
                "tts.profile_backup",
                temporary_path,
                must_exist=True,
                isolation_level=None,
            )
            self._worker_online_backup(
                connection,
                destination_connection,
                deadline=deadline,
            )
            destination_connection.close()
            destination_connection = None
            self._worker_validate_standalone_snapshot(
                temporary_path,
                deadline=deadline,
            )
            temporary_state = temporary_path.stat()
            if not stat.S_ISREG(temporary_state.st_mode):
                raise _repository_error("backup_failed")
            receipt = ProfileBackupReceipt(
                created_at=created_at,
                byte_count=temporary_state.st_size,
            )

            self._require_configured_path_matches(active_path, "backup_failed")
            current_destination = _validate_backup_destination(
                destination.path,
                active_path,
            )
            if current_destination.parent_identity != destination.parent_identity:
                raise _repository_error("backup_failed")
            _require_restore_time(deadline)
            _fsync_file(temporary_path)
            _require_restore_time(deadline)
            os.replace(temporary_path, destination.path)
            published = True
            _fsync_directory(destination.path.parent)
            # Publication is already committed at this point. A single fsync
            # syscall cannot be interrupted, so do not report a timeout after
            # a durable destination has become visible.
        except BaseException as error:
            body_error = error

        if destination_connection is not None:
            try:
                destination_connection.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if temporary_path is not None:
            if not published:
                try:
                    _unlink_path_if_present(temporary_path)
                except BaseException as error:
                    cleanup_errors.append(error)
            for suffix in _STORE_SIDECAR_SUFFIXES:
                try:
                    _unlink_path_if_present(
                        temporary_path.with_name(f"{temporary_path.name}{suffix}")
                    )
                except BaseException as error:
                    cleanup_errors.append(error)

        if body_error is not None or cleanup_errors:
            for candidate_error in (body_error, *cleanup_errors):
                if candidate_error is not None and not isinstance(
                    candidate_error,
                    Exception,
                ):
                    raise candidate_error
            raise _repository_error("backup_failed")
        assert receipt is not None
        return receipt

    def _worker_online_backup(
        self,
        source: sqlite3.Connection,
        destination: sqlite3.Connection,
        *,
        deadline: float | None = None,
    ) -> None:
        """Copy one complete SQLite snapshot through the online-backup API."""

        progress_guard = (
            None if deadline is None else lambda: _require_restore_time(deadline)
        )
        if progress_guard is not None:
            progress_guard()
        backup_open_connections_to_private(
            "tts.profile_backup",
            source,
            destination,
            progress_guard=progress_guard,
        )
        if progress_guard is not None:
            progress_guard()

    def _worker_require_full_integrity(
        self,
        connection: sqlite3.Connection,
        *,
        deadline: float | None = None,
    ) -> None:
        """Require SQLite's full integrity check on one worker-owned handle."""

        callback_error: BaseException | None = None
        body_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        results: list[object] | None = None
        progress_installed = False

        def interrupt_after_deadline() -> int:
            nonlocal callback_error
            try:
                assert deadline is not None
                _require_restore_time(deadline)
            except BaseException as error:
                callback_error = error
                return 1
            return 0

        try:
            if deadline is not None:
                _require_restore_time(deadline)
                connection.set_progress_handler(
                    interrupt_after_deadline,
                    _RESTORE_PROGRESS_OPCODE_INTERVAL,
                )
                progress_installed = True
            results = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        except BaseException as error:
            body_error = error
        if progress_installed:
            try:
                connection.set_progress_handler(None, 0)
            except BaseException as error:
                cleanup_error = error

        if callback_error is not None:
            body_error = callback_error
        elif body_error is None and deadline is not None:
            try:
                _require_restore_time(deadline)
            except BaseException as error:
                body_error = error

        mapped_errors: list[BaseException | None] = []
        for candidate_error in (body_error, cleanup_error):
            if candidate_error is None or not isinstance(candidate_error, Exception):
                mapped_errors.append(candidate_error)
            elif isinstance(candidate_error, ProfileRepositoryError):
                mapped_errors.append(candidate_error)
            else:
                mapped_errors.append(_repository_error("schema_corrupt"))
        _raise_with_cleanup_precedence(*mapped_errors)
        if results != ["ok"]:
            raise _repository_error("schema_corrupt")

    def _worker_validate_standalone_snapshot(
        self,
        path: Path,
        *,
        deadline: float | None = None,
    ) -> None:
        """Run shared schema/domain checks plus full integrity on one snapshot."""

        check_deadline = (
            None if deadline is None else lambda: _require_restore_time(deadline)
        )
        validate_profile_candidate(
            path,
            check_deadline=check_deadline,
        )
        connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        close_error: BaseException | None = None
        try:
            if check_deadline is not None:
                check_deadline()
            connection = connect_private_sqlite(
                "tts.profile_snapshot",
                path,
                must_exist=True,
                read_only=True,
                immutable=True,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            if check_deadline is not None:
                check_deadline()
            self._worker_require_full_integrity(
                connection,
                deadline=deadline,
            )
            validate_reference_rows(
                connection,
                check_deadline=check_deadline,
            )
        except BaseException as error:
            body_error = error
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                close_error = error
        _raise_with_cleanup_precedence(body_error, close_error)

    def _worker_restore(
        self,
        candidate_path: Path,
        deadline: float,
        generation: int,
        active_path: Path,
    ) -> ProfileRestoreReceipt:
        """Run one exclusive staged restore and race-free shared rebind."""

        return self._worker_restore_with_publication(
            candidate_path,
            deadline,
            generation,
            active_path,
        )

    def _worker_restore_with_publication(
        self,
        candidate_path: Path,
        deadline: float,
        generation: int,
        active_path: Path,
    ) -> ProfileRestoreReceipt:
        """Restore through the same recoverable publication used by migration."""

        self._clear_reference_damage_markers()
        candidate: _CandidateSnapshot | None = None
        restored_at: datetime | None = None
        exclusive_lease: ProfileStoreLease | None = None
        primary_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        recovery_path: Path | None = None
        try:
            restored_at = self._clock()
            self._require_configured_path_matches(active_path, "restore_failed")
            candidate = _validate_restore_candidate_path(candidate_path, active_path)
            _require_restore_time(deadline)
            self._worker_close_for_restore(deadline)
            remaining = _remaining_seconds(deadline)
            if remaining <= 0:
                raise _repository_error("restore_failed")
            exclusive_lease = ProfileStoreLease(
                active_path,
                ProfileStoreLockMode.EXCLUSIVE,
                timeout_seconds=remaining,
            )
            exclusive_lease.acquire()
            _require_restore_time(deadline)
            recover_profile_migration_publication(active_path)
            parent_authority = ParentAuthority(active_path.parent.stat())
            for key in MigrationTombstoneKey:
                require_reusable_tombstone(
                    active_path,
                    parent_authority=parent_authority,
                    tombstone_key=key,
                )
            self._worker_remove_live_sidecars(deadline=deadline)
            if not _candidate_is_unchanged(candidate):
                raise _repository_error("restore_failed")
            source_version = self._worker_schema_version_for_restore(candidate.path)
            if not 0 <= source_version <= CURRENT_PROFILE_SCHEMA_VERSION:
                raise _repository_error("schema_unsupported")
            self._worker_validate_restore_source(candidate.path, source_version)
            self._require_configured_path_matches(active_path, "restore_failed")
            if not _candidate_is_unchanged(candidate):
                raise _repository_error("restore_failed")

            def prepublication_guard(
                stage: ProfileMigrationPublicationStage,
            ) -> None:
                nonlocal recovery_path
                if stage is not ProfileMigrationPublicationStage.PREFLIGHT:
                    return
                _require_restore_time(deadline)
                self._require_configured_path_matches(
                    active_path,
                    "restore_failed",
                )
                if not _candidate_is_unchanged(candidate):
                    raise _repository_error("restore_failed")
                recovery_path = self._worker_create_recovery_backup(
                    cast(datetime, restored_at),
                    deadline,
                )

            self._worker_publish_migrated_store(
                active_path,
                source_version,
                source_path=candidate.path,
                stage_hook=prepublication_guard,
                progress_guard=lambda: _require_restore_time(deadline),
            )
            exclusive_lease.release()
            exclusive_lease = None
            self._worker_rebind_current_store()
            connection = self._connection
            if connection is None:
                raise _repository_error("restore_failed")
            profile_count, assignment_count = self._worker_store_counts(connection)
            with self._state_lock:
                if (
                    self._generation != generation
                    or self._terminal
                    or self._state is not ProfileRepositoryState.RESTORING
                ):
                    raise _repository_error("stale")
                self._state = ProfileRepositoryState.OPEN
            return ProfileRestoreReceipt(
                restored_at=cast(datetime, restored_at),
                profile_count=profile_count,
                assignment_count=assignment_count,
            )
        except BaseException as error:
            if isinstance(error, ExactProfileStoreAuthorityError):
                self._worker_seal_exact_authority(error)
            primary_error = error

        retained_failed_connection = False
        if exclusive_lease is not None:
            if self._connection is not None and self._lease is None:
                self._lease = exclusive_lease
                self._active_database_path = active_path
                exclusive_lease = None
                retained_failed_connection = True
            else:
                try:
                    exclusive_lease.release()
                except BaseException as error:
                    cleanup_errors.append(error)
                    self._lease = exclusive_lease
                    self._active_database_path = active_path
        rebound_ok = False
        if (
            not cleanup_errors
            and not retained_failed_connection
            and not (
                isinstance(primary_error, ProfileRepositoryError)
                and primary_error.code == "unavailable"
            )
        ):
            try:
                self._worker_rebind_current_store()
                rebound_ok = True
            except BaseException as error:
                cleanup_errors.append(error)
        with self._state_lock:
            if (
                self._generation == generation
                and not self._terminal
                and self._state is ProfileRepositoryState.RESTORING
            ):
                self._state = (
                    ProfileRepositoryState.OPEN
                    if rebound_ok
                    else ProfileRepositoryState.UNAVAILABLE
                )
        _ = recovery_path
        for pending_error in (primary_error, *cleanup_errors):
            if pending_error is not None and not isinstance(pending_error, Exception):
                raise pending_error
        if (
            isinstance(primary_error, ProfileRepositoryError)
            and primary_error.code == "unavailable"
        ):
            raise _repository_error("unavailable")
        if rebound_ok and isinstance(primary_error, ProfileRepositoryError):
            if primary_error.code in {
                "corrupt_data",
                "lock_timeout",
                "schema_corrupt",
                "schema_partial",
                "schema_unsupported",
                "stale",
            }:
                raise _repository_error(primary_error.code)
            raise _repository_error("restore_failed")
        raise _repository_error("restore_failed")

    def _worker_schema_version_for_restore(self, candidate_path: Path) -> int:
        """Read one admitted restore candidate without authorizing migration."""

        connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        version: int | None = None
        try:
            connection = connect_private_sqlite(
                "tts.profile_restore_stage",
                candidate_path,
                read_only=True,
                must_exist=True,
                isolation_level=None,
            )
            row = connection.execute("PRAGMA user_version").fetchone()
            if row is None or len(row) != 1 or type(row[0]) is not int:
                raise ValueError
            version = cast(int, row[0])
        except BaseException as error:
            body_error = error
        close_error: BaseException | None = None
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                close_error = error
                self._worker_retain_failed_connection(
                    connection,
                    self._worker_active_path(),
                )
        for pending_error in (body_error, close_error):
            if pending_error is not None and not isinstance(pending_error, Exception):
                raise pending_error
        if body_error is not None or close_error is not None or version is None:
            raise _repository_error("schema_corrupt")
        return version

    def _worker_validate_restore_source(
        self,
        candidate_path: Path,
        source_version: int,
    ) -> None:
        """Fully qualify an incoming source before creating recovery evidence."""

        connection: sqlite3.Connection | None = None
        body_error: BaseException | None = None
        try:
            connection = connect_private_sqlite(
                "tts.profile_restore_stage",
                candidate_path,
                read_only=True,
                must_exist=True,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            if source_version == 0:
                if _profile_schema._user_schema_objects(connection):
                    raise _repository_error("schema_corrupt")
                _profile_schema._validate_full_integrity(connection)
            else:
                _profile_schema.validate_profile_store_version(
                    connection,
                    source_version,
                )
                self._worker_require_full_integrity(connection)
                validate_profile_store_rows(connection)
                if source_version >= 3:
                    validate_reference_rows(connection)
                self._worker_store_counts(connection)
        except BaseException as error:
            body_error = error
        close_error: BaseException | None = None
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                close_error = error
                self._worker_retain_failed_connection(
                    connection,
                    self._worker_active_path(),
                )
        for pending_error in (body_error, close_error):
            if pending_error is not None and not isinstance(pending_error, Exception):
                raise pending_error
        if isinstance(body_error, ProfileRepositoryError):
            raise _repository_error(body_error.code)
        if body_error is not None or close_error is not None:
            raise _repository_error("schema_corrupt")

    def _worker_close_for_restore(self, deadline: float) -> None:
        connection = self._connection
        lease = self._lease
        if connection is None or lease is None:
            raise _repository_error("invalid_state")
        self._worker_revalidate_exact_authority(connection)
        exact_connection = cast(object, connection)
        sidecar_fds = getattr(exact_connection, "sidecar_fds", {})
        parent_fd = getattr(exact_connection, "parent_fd", -1)
        if isinstance(sidecar_fds, dict) and type(parent_fd) is int and parent_fd >= 0:
            self._restore_sidecar_identities = {
                suffix: os.fstat(descriptor)
                for suffix, descriptor in sidecar_fds.items()
            }
            self._restore_parent_authority = ParentAuthority(os.fstat(parent_fd))
        _require_restore_time(deadline)
        timeout_row = connection.execute("PRAGMA busy_timeout").fetchone()
        if (
            timeout_row is None
            or len(timeout_row) != 1
            or type(timeout_row[0]) is not int
            or timeout_row[0] < 0
        ):
            raise _repository_error("restore_failed")
        original_timeout_ms = cast(int, timeout_row[0])
        remaining = _remaining_seconds(deadline)
        if remaining <= 0:
            raise _repository_error("restore_failed")
        bounded_timeout_ms = min(original_timeout_ms, int(remaining * 1_000))
        checkpoint: object = None
        body_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            connection.execute(f"PRAGMA busy_timeout = {bounded_timeout_ms}")
            checkpoint = _run_with_restore_progress(
                connection,
                deadline,
                lambda: connection.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone(),
            )
        except BaseException as error:
            body_error = error
        try:
            connection.execute(f"PRAGMA busy_timeout = {original_timeout_ms}")
        except BaseException as error:
            cleanup_error = error
        _raise_with_cleanup_precedence(body_error, cleanup_error)
        if (
            checkpoint is None
            or not isinstance(checkpoint, (tuple, sqlite3.Row))
            or len(checkpoint) != 3
            or any(type(value) is not int for value in checkpoint)
            or tuple(checkpoint) != (0, 0, 0)
        ):
            raise _repository_error("restore_failed")
        self._worker_revalidate_exact_authority(connection)
        connection.close()
        self._connection = None
        lease.release()
        self._lease = None
        _require_restore_time(deadline)

    def _worker_create_recovery_backup(
        self,
        restored_at: datetime,
        deadline: float,
    ) -> Path:
        active_path = self._worker_active_path()
        source: sqlite3.Connection | None = None
        recovery_path: Path | None = None
        body_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        try:
            _require_restore_time(deadline)
            timestamp = restored_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            descriptor, recovery_name = tempfile.mkstemp(
                prefix=f"{active_path.name}.pre-restore-{timestamp}-",
                suffix=".recovery.sqlite3",
                dir=active_path.parent,
            )
            recovery_path = Path(recovery_name)
            os.close(descriptor)
            source = open_profile_store(
                active_path,
                must_exist=True,
                check_deadline=lambda: _require_restore_time(deadline),
            )
            backup_connection_to_private(
                "tts.profile_recovery",
                source,
                active_path,
                recovery_path,
                progress_guard=lambda: _require_restore_time(deadline),
            )
            try:
                source.close()
            except BaseException:
                self._connection = source
                source = None
                raise
            else:
                source = None
            self._worker_validate_standalone_snapshot(
                recovery_path,
                deadline=deadline,
            )
            _require_restore_time(deadline)
            _fsync_file(recovery_path)
            _require_restore_time(deadline)
            _fsync_directory(recovery_path.parent)
            _require_restore_time(deadline)
        except BaseException as error:
            body_error = error

        for connection in (source,):
            if connection is None:
                continue
            try:
                connection.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if body_error is not None or cleanup_errors:
            if recovery_path is not None:
                cleanup_errors.extend(
                    self._worker_remove_temporary_store(recovery_path)
                )
            _raise_with_cleanup_precedence(body_error, *cleanup_errors)
        assert recovery_path is not None
        return recovery_path

    def _worker_remove_live_sidecars(self, *, deadline: float) -> None:
        database_path = self._worker_active_path()
        rollback_journal = database_path.with_name(f"{database_path.name}-journal")
        _require_restore_time(deadline)
        try:
            rollback_journal.lstat()
        except FileNotFoundError:
            pass
        else:
            raise _repository_error("restore_failed")
        _require_restore_time(deadline)
        tombstones = {
            "-wal": MigrationTombstoneKey.LIVE_WAL,
            "-shm": MigrationTombstoneKey.LIVE_SHM,
        }
        parent_authority = self._restore_parent_authority
        expected_sidecars = self._restore_sidecar_identities
        for suffix in ("-wal", "-shm"):
            sidecar = database_path.with_name(f"{database_path.name}{suffix}")
            _require_restore_time(deadline)
            try:
                state = sidecar.lstat()
            except FileNotFoundError:
                _require_restore_time(deadline)
                continue
            _require_restore_time(deadline)
            expected = expected_sidecars.get(suffix)
            if (
                parent_authority is None
                or expected is None
                or not private_paths._same_identity(state, expected)
                or private_paths._classify_private_file_stat(
                    state,
                    expected_uid=os.geteuid(),
                )
                is not None
                or stat.S_IMODE(state.st_mode) != 0o600
            ):
                raise _repository_error("restore_failed")
            _require_restore_time(deadline)
            try:
                remove_exact(
                    sidecar,
                    parent_authority=parent_authority,
                    file_identity=expected,
                    tombstone_key=tombstones[suffix],
                )
                holding = database_path.with_name(
                    f".profile-migration-{tombstones[suffix].value}.tombstone"
                )
                self._reusable_tombstones[tombstones[suffix]] = holding.stat()
            except BaseException as error:
                if not isinstance(error, Exception):
                    raise
                raise _repository_error("restore_failed") from None
            _require_restore_time(deadline)
        self._restore_sidecar_identities = {}
        self._restore_parent_authority = None

    def _worker_remove_temporary_store(self, path: Path) -> list[BaseException]:
        errors: list[BaseException] = []
        for target in (
            path,
            *(
                path.with_name(f"{path.name}{suffix}")
                for suffix in _STORE_SIDECAR_SUFFIXES
            ),
        ):
            try:
                _unlink_path_if_present(target)
            except BaseException as error:
                errors.append(error)
        return errors

    def _worker_store_counts(
        self,
        connection: sqlite3.Connection,
        *,
        deadline: float | None = None,
    ) -> tuple[int, int]:
        def read_counts() -> tuple[int, int]:
            counts: list[int] = []
            for statement in (
                "SELECT COUNT(*) FROM tts_generation_profiles",
                "SELECT COUNT(*) FROM character_tts_assignments",
            ):
                if deadline is not None:
                    _require_restore_time(deadline)
                row = connection.execute(statement).fetchone()
                if (
                    row is None
                    or len(row) != 1
                    or type(row[0]) is not int
                    or row[0] < 0
                ):
                    raise _repository_error("corrupt_data")
                counts.append(cast(int, row[0]))
            if deadline is not None:
                _require_restore_time(deadline)
            return (counts[0], counts[1])

        if deadline is None:
            return read_counts()
        return _run_with_restore_progress(
            connection,
            deadline,
            read_counts,
        )

    def _worker_rebind_current_store(self) -> None:
        active_path = self._worker_active_path()
        if self._connection is not None and self._lease is not None:
            self._worker_revalidate_exact_authority(self._connection)
            validate_profile_store_rows(self._connection)
            validate_reference_rows(self._connection)
            self._worker_store_counts(self._connection)
            self._worker_revalidate_exact_authority(self._connection)
            return

        cleanup_error: BaseException | None = None
        try:
            self._worker_cleanup()
        except BaseException as error:
            cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error

        shared = self._worker_open_if_proven_current(active_path)
        if shared is None:
            expected = self._worker_initialize_store(active_path, allow_create=False)
            shared = self._worker_open_if_proven_current(
                active_path,
                expected_post_init_authority=expected,
            )
        if shared is None:
            raise _repository_error("restore_failed")
        lease, connection = shared
        body_error: BaseException | None = None
        try:
            self._worker_revalidate_exact_authority(connection, active_path)
            validate_profile_store_rows(connection)
            validate_reference_rows(connection)
            self._worker_store_counts(connection)
            self._worker_revalidate_exact_authority(connection, active_path)
        except BaseException as error:
            body_error = error
        if body_error is None:
            assert connection is not None
            self._lease = lease
            self._connection = connection
            self._active_database_path = active_path
            return

        connection_error: BaseException | None = None
        lease_error: BaseException | None = None
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                connection_error = error
                self._connection = connection
                self._lease = lease
        if connection_error is None:
            try:
                lease.release()
            except BaseException as error:
                lease_error = error
                self._lease = lease
        if self._connection is not None or self._lease is not None:
            self._active_database_path = active_path
        else:
            self._active_database_path = None
        _raise_with_cleanup_precedence(body_error, connection_error, lease_error)

    async def restore_from(
        self,
        candidate: Path,
        timeout_seconds: int | float = 5.0,
    ) -> ProfileStoreResult[ProfileRestoreReceipt]:
        """Atomically restore one validated standalone profile-store snapshot.

        The timeout is enforced cooperatively between bounded SQLite backup
        page batches, SQLite VM progress callbacks, row decodes, and filesystem
        boundaries. One in-flight kernel filesystem call cannot be interrupted.

        Args:
            candidate: Exact standalone candidate path.
            timeout_seconds: Positive finite quiescence/exclusive-lock budget.

        Returns:
            The admitted lifecycle generation and safe restore metadata.

        Raises:
            ProfileRepositoryError: If admission, quiescence, validation,
                locking, replacement, or lifecycle rebind fails safely.
            BaseException: A caller control-flow signal after lifecycle
                settlement and cleanup.
        """

        timeout = _validate_restore_timeout(timeout_seconds)
        if type(candidate) is not _PATH_TYPE:
            raise _repository_error("restore_failed")
        exact_candidate = cast(Path, candidate)
        active_path = self._active_path_for_operation("restore_failed")
        deadline = _read_monotonic() + timeout
        if not math.isfinite(deadline):
            raise _repository_error("restore_failed")

        lifecycle_lock = self._bind_or_check_loop()
        remaining = _remaining_seconds(deadline)
        if remaining <= 0:
            raise _repository_error("restore_failed")
        try:
            await asyncio.wait_for(lifecycle_lock.acquire(), timeout=remaining)
        except TimeoutError:
            raise _repository_error("restore_failed") from None

        try:
            with self._state_lock:
                state_error = self._normal_state_error_locked()
                if state_error is not None:
                    raise _repository_error(state_error)
                self._state = ProfileRepositoryState.RESTORING
                self._generation += 1
                self._damaged_reference_profile_ids.clear()
                generation = self._generation
                pending = tuple(self._pending_futures)
                executor = self._executor

            setup_error: BaseException | None = None
            completion: (
                asyncio.Task[ProfileStoreResult[ProfileRestoreReceipt]] | None
            ) = None
            try:
                for future in pending:
                    future.cancel()

                completion = asyncio.create_task(
                    self._finish_restore(
                        exact_candidate,
                        deadline,
                        generation,
                        pending,
                        executor,
                        active_path,
                    )
                )
            except BaseException as error:
                setup_error = error
            if setup_error is not None:
                with self._state_lock:
                    if (
                        self._generation == generation
                        and not self._terminal
                        and self._state is ProfileRepositoryState.RESTORING
                    ):
                        self._state = ProfileRepositoryState.OPEN
                if not isinstance(setup_error, Exception):
                    raise setup_error
                raise _repository_error("restore_failed")
            assert completion is not None
            return await self._await_lifecycle_completion(completion)
        finally:
            lifecycle_lock.release()

    async def _finish_restore(
        self,
        candidate: Path,
        deadline: float,
        generation: int,
        pending: tuple[Future[object], ...],
        executor: ThreadPoolExecutor | None,
        active_path: Path,
    ) -> ProfileStoreResult[ProfileRestoreReceipt]:
        """Quiesce old work, run restore on the worker, and publish safely."""

        self._bind_or_check_loop()
        pre_worker_error: BaseException | None = None
        restore_future: Future[ProfileRestoreReceipt] | None = None
        try:
            running = tuple(future for future in pending if not future.done())
            if running:
                wrappers = tuple(asyncio.wrap_future(future) for future in running)
                drain = asyncio.gather(*wrappers, return_exceptions=True)
                remaining = _remaining_seconds(deadline)
                if remaining <= 0:
                    raise _repository_error("restore_failed")
                try:
                    await asyncio.wait_for(asyncio.shield(drain), timeout=remaining)
                except TimeoutError:
                    raise _repository_error("restore_failed") from None

            remaining = _remaining_seconds(deadline)
            if remaining <= 0 or executor is None:
                raise _repository_error("restore_failed")
            restore_future = executor.submit(
                self._worker_restore,
                candidate,
                deadline,
                generation,
                active_path,
            )
        except BaseException as error:
            pre_worker_error = error

        if pre_worker_error is not None:
            with self._state_lock:
                if (
                    self._generation == generation
                    and not self._terminal
                    and self._state is ProfileRepositoryState.RESTORING
                ):
                    self._state = ProfileRepositoryState.OPEN
            if not isinstance(pre_worker_error, Exception):
                raise pre_worker_error
            raise _repository_error("restore_failed")
        assert restore_future is not None

        worker_error: BaseException | None = None
        receipt: ProfileRestoreReceipt | None = None
        try:
            receipt = await asyncio.wrap_future(restore_future)
        except BaseException as error:
            worker_error = error
        if worker_error is not None:
            _raise_operation_error(worker_error)

        with self._state_lock:
            if (
                self._generation != generation
                or self._terminal
                or self._state is not ProfileRepositoryState.OPEN
            ):
                raise _repository_error("stale")
        assert receipt is not None
        return ProfileStoreResult(generation=generation, value=receipt)

    def _worker_create_profile(
        self,
        connection: sqlite3.Connection,
        draft: TTSProfileDraft,
        profile_id: UUID | None,
    ) -> TTSGenerationProfile:
        evidence = _IntegrityEvidence(
            profile_id=profile_id,
            normalized_name=draft.normalized_name,
        )
        return self._worker_transaction(
            connection,
            lambda: self._worker_insert_profile(
                connection,
                draft,
                profile_id,
                evidence,
            ),
            operation_kind="create",
            immediate=True,
            integrity_evidence=evidence,
        )

    def _worker_insert_profile(
        self,
        connection: sqlite3.Connection,
        draft: TTSProfileDraft,
        profile_id: UUID | None,
        evidence: _IntegrityEvidence,
    ) -> TTSGenerationProfile:
        persisted_id = profile_id if profile_id is not None else self._worker_new_uuid()
        evidence.profile_id = persisted_id
        timestamp = self._clock()
        profile = TTSGenerationProfile(
            profile_id=persisted_id,
            display_name=draft.display_name,
            normalized_name=draft.normalized_name,
            provider_id=draft.provider_id,
            model_id=draft.model_id,
            voice_id=draft.voice_id,
            response_format=draft.response_format,
            speed=draft.speed,
            options=cast(FrozenJsonOptions, draft.options),
            revision=1,
            created_at=timestamp,
            updated_at=timestamp,
        )
        parameters = encode_profile(profile)
        try:
            connection.execute(
                """
                INSERT INTO tts_generation_profiles (
                    profile_id,
                    display_name,
                    normalized_name,
                    provider_id,
                    model_id,
                    voice_id,
                    response_format,
                    speed,
                    options_json,
                    revision,
                    created_at,
                    updated_at
                ) VALUES (
                    :profile_id,
                    :display_name,
                    :normalized_name,
                    :provider_id,
                    :model_id,
                    :voice_id,
                    :response_format,
                    :speed,
                    :options_json,
                    :revision,
                    :created_at,
                    :updated_at
                )
                """,
                parameters,
            )
        except sqlite3.IntegrityError as error:
            evidence.statement_error = error
            raise
        return self._worker_require_round_trip(connection, persisted_id, profile)

    def _worker_create_profile_with_reference(
        self,
        connection: sqlite3.Connection,
        draft: TTSProfileDraft,
        profile_id: UUID,
        canonical: CanonicalTTSCloneReference,
        recipe_requirement: TTSCloneRecipeRequirement,
        expected_generation: int,
    ) -> TTSGenerationProfile:
        evidence = _IntegrityEvidence(
            profile_id=profile_id,
            normalized_name=draft.normalized_name,
        )

        def create_with_reference() -> TTSGenerationProfile:
            self._worker_require_generation(expected_generation)
            profile = self._worker_insert_profile(
                connection,
                draft,
                profile_id,
                evidence,
            )
            return self._worker_put_reference(
                connection,
                profile.profile_id,
                canonical,
                recipe_requirement,
                profile.revision,
            )

        created = self._worker_transaction(
            connection,
            create_with_reference,
            operation_kind="create",
            immediate=True,
            integrity_evidence=evidence,
        )
        self._discard_reference_damage_marker(profile_id)
        return created

    def _worker_create_profile_with_assignment(
        self,
        connection: sqlite3.Connection,
        draft: TTSProfileDraft,
        profile_id: UUID,
        character_ref: CharacterRef,
        expected_generation: int,
        expected_current_profile_id: UUID | None,
    ) -> AssignedTTSProfileSnapshot:
        evidence = _IntegrityEvidence(
            profile_id=profile_id,
            normalized_name=draft.normalized_name,
        )

        def create_and_assign() -> AssignedTTSProfileSnapshot:
            self._worker_require_generation(expected_generation)
            profile = self._worker_insert_profile(
                connection,
                draft,
                profile_id,
                evidence,
            )
            assignment = self._worker_set_assignment_exact(
                connection,
                character_ref,
                profile.profile_id,
                expected_generation,
                profile.revision,
                expected_current_profile_id,
                profile,
            )
            return AssignedTTSProfileSnapshot(
                assignment=assignment,
                profile=profile,
            )

        return self._worker_transaction(
            connection,
            create_and_assign,
            operation_kind="create",
            immediate=True,
            integrity_evidence=evidence,
        )

    def _worker_commit_bundle_import(
        self,
        connection: sqlite3.Connection,
        command: TTSBundleImportCommand,
    ) -> TTSBundleImportResult:
        evidence = _IntegrityEvidence(
            profile_id=command.source_profile_id,
            normalized_name=command.source_draft.normalized_name,
        )

        def commit_import() -> TTSBundleImportResult:
            self._worker_require_generation(command.expected_generation)
            collisions = self._worker_read_profile_collisions(
                connection,
                command.source_profile_id,
                command.source_draft.normalized_name,
            )
            if collisions != command.reviewed_source_collisions:
                return TTSBundleImportResult(
                    kind="stale_inspection",
                    profile=None,
                    repository_facts=TTSBundleImportRepositoryFacts(
                        source_collisions=collisions,
                        copy_collisions=None,
                    ),
                )
            if command.choice == "reuse":
                candidate = collisions.profile_id_match
                if (
                    candidate is not None
                    and candidate == collisions.normalized_name_match
                    and self._worker_bundle_profile_matches(
                        connection,
                        candidate,
                        command,
                    )
                ):
                    return TTSBundleImportResult(kind="reused", profile=candidate)
                return TTSBundleImportResult(
                    kind="stale_inspection",
                    profile=None,
                    repository_facts=TTSBundleImportRepositoryFacts(
                        source_collisions=collisions,
                        copy_collisions=None,
                    ),
                )
            if command.choice == "copy":
                assert command.copy_profile_id is not None
                assert command.copy_display_name is not None
                copy_draft = TTSProfileDraft(
                    display_name=command.copy_display_name,
                    provider_id=command.source_draft.provider_id,
                    model_id=command.source_draft.model_id,
                    voice_id=command.source_draft.voice_id,
                    response_format=command.source_draft.response_format,
                    speed=command.source_draft.speed,
                    options=command.source_draft.options,
                )
                copy_collisions = self._worker_read_profile_collisions(
                    connection,
                    command.copy_profile_id,
                    copy_draft.normalized_name,
                )
                if (
                    collisions.profile_id_match is None
                    and collisions.normalized_name_match is None
                ) or (
                    copy_collisions.profile_id_match is not None
                    or copy_collisions.normalized_name_match is not None
                ):
                    return TTSBundleImportResult(
                        kind="stale_inspection",
                        profile=None,
                        repository_facts=TTSBundleImportRepositoryFacts(
                            source_collisions=collisions,
                            copy_collisions=copy_collisions,
                        ),
                    )
                evidence.profile_id = command.copy_profile_id
                evidence.normalized_name = copy_draft.normalized_name
                profile = self._worker_insert_profile(
                    connection,
                    copy_draft,
                    command.copy_profile_id,
                    evidence,
                )
                created = self._worker_put_reference(
                    connection,
                    profile.profile_id,
                    command.canonical_reference,
                    command.recipe_requirement,
                    profile.revision,
                )
                return TTSBundleImportResult(kind="created", profile=created)
            if command.choice != "create" or (
                collisions.profile_id_match is not None
                or collisions.normalized_name_match is not None
            ):
                return TTSBundleImportResult(
                    kind="stale_inspection",
                    profile=None,
                    repository_facts=TTSBundleImportRepositoryFacts(
                        source_collisions=collisions,
                        copy_collisions=None,
                    ),
                )
            profile = self._worker_insert_profile(
                connection,
                command.source_draft,
                command.source_profile_id,
                evidence,
            )
            created = self._worker_put_reference(
                connection,
                profile.profile_id,
                command.canonical_reference,
                command.recipe_requirement,
                profile.revision,
            )
            return TTSBundleImportResult(kind="created", profile=created)

        result = self._worker_transaction(
            connection,
            commit_import,
            operation_kind="create",
            immediate=True,
            integrity_evidence=evidence,
        )
        if result.profile is not None:
            self._discard_reference_damage_marker(result.profile.profile_id)
        return result

    def _worker_bundle_profile_matches(
        self,
        connection: sqlite3.Connection,
        profile: TTSGenerationProfile,
        command: TTSBundleImportCommand,
    ) -> bool:
        """Compare complete public/private bundle equality below the boundary."""

        draft = command.source_draft
        if (
            profile.profile_id != command.source_profile_id
            or profile.display_name != draft.display_name
            or (
                profile.provider_id,
                profile.model_id,
                profile.voice_id,
                profile.response_format,
                profile.speed,
                profile.options,
            )
            != (
                draft.provider_id,
                draft.model_id,
                draft.voice_id,
                draft.response_format,
                draft.speed,
                draft.options,
            )
            or profile.reference is None
            or profile.reference.recipe_requirement is None
            or profile.reference.recipe_requirement != command.recipe_requirement
        ):
            return False
        row = connection.execute(
            f"{REFERENCE_PAYLOAD_SELECT} WHERE profile_id = ?",
            (encode_uuid(profile.profile_id),),
        ).fetchone()
        if row is None:
            return False
        rowid = row["reference_rowid"]
        byte_length = row["reference_byte_length"]
        if (
            type(rowid) is not int
            or rowid <= 0
            or type(byte_length) is not int
            or byte_length <= 0
        ):
            return False
        payload = read_reference_blob(connection, rowid, byte_length)
        reference = decode_reference_payload(row, payload)
        canonical = command.canonical_reference
        return (
            reference.recipe_requirement == command.recipe_requirement
            and reference.reference_text == canonical.reference_text
            and reference.sha256 == canonical.sha256
            and reference.wav_bytes == canonical.wav_bytes
        )

    def _worker_get_profile(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
    ) -> TTSGenerationProfile:
        row = connection.execute(
            f"{_PROFILE_SELECT} WHERE p.profile_id = ?",
            (encode_uuid(profile_id),),
        ).fetchone()
        if row is None:
            raise _repository_error("missing")
        return _decode_profile_with_reference_row(row)

    def _worker_get_base_profile(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
    ) -> TTSGenerationProfile:
        row = connection.execute(
            f"{_BASE_PROFILE_SELECT} WHERE profile_id = ?",
            (encode_uuid(profile_id),),
        ).fetchone()
        if row is None:
            raise _repository_error("missing")
        return decode_profile(row)

    def _worker_get_profile_collisions(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        normalized_name: str,
    ) -> TTSProfileCollisionSnapshot:
        def read_collisions() -> TTSProfileCollisionSnapshot:
            return self._worker_read_profile_collisions(
                connection,
                profile_id,
                normalized_name,
            )

        return self._worker_transaction(
            connection,
            read_collisions,
            operation_kind="read",
            immediate=False,
        )

    def _worker_read_profile_collisions(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        normalized_name: str,
    ) -> TTSProfileCollisionSnapshot:
        """Read current collision facts inside a caller-owned transaction."""

        profile_id_row = connection.execute(
            f"{_PROFILE_SELECT} WHERE p.profile_id = ?",
            (encode_uuid(profile_id),),
        ).fetchone()
        normalized_name_row = connection.execute(
            f"{_PROFILE_SELECT} WHERE p.normalized_name = ?",
            (normalized_name,),
        ).fetchone()
        return TTSProfileCollisionSnapshot(
            profile_id_match=(
                None
                if profile_id_row is None
                else _decode_profile_with_reference_row(profile_id_row)
            ),
            normalized_name_match=(
                None
                if normalized_name_row is None
                else _decode_profile_with_reference_row(normalized_name_row)
            ),
        )

    def _worker_list_profiles(
        self,
        connection: sqlite3.Connection,
        normalized_search: str | None,
        limit: int,
        offset: int,
    ) -> TTSProfilePage:
        def read_page() -> TTSProfilePage:
            if normalized_search is None:
                where_clause = ""
                filter_parameters: tuple[object, ...] = ()
            else:
                where_clause = " WHERE p.normalized_name LIKE ? ESCAPE '!'"
                filter_parameters = (f"%{_escape_like_literal(normalized_search)}%",)

            count_row = connection.execute(
                f"SELECT COUNT(*) FROM tts_generation_profiles AS p{where_clause}",
                filter_parameters,
            ).fetchone()
            if (
                count_row is None
                or len(count_row) != 1
                or type(count_row[0]) is not int
                or count_row[0] < 0
            ):
                raise _repository_error("corrupt_data")
            total = cast(int, count_row[0])
            rows = connection.execute(
                (
                    f"{_PROFILE_SELECT}{where_clause} "
                    "ORDER BY p.normalized_name ASC, p.profile_id ASC "
                    "LIMIT ? OFFSET ?"
                ),
                (*filter_parameters, limit, offset),
            ).fetchall()
            profiles = tuple(_decode_profile_with_reference_row(row) for row in rows)
            return TTSProfilePage(profiles=profiles, total=total)

        return self._worker_transaction(
            connection,
            read_page,
            operation_kind="read",
            immediate=False,
        )

    def _worker_update_profile(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        expected_revision: int,
        draft: TTSProfileDraft,
    ) -> TTSGenerationProfile:
        evidence = _IntegrityEvidence(
            profile_id=profile_id,
            normalized_name=draft.normalized_name,
        )

        def update() -> TTSGenerationProfile:
            stored = self._worker_get_profile(connection, profile_id)
            if stored.revision != expected_revision:
                raise _repository_error("conflict")
            if stored.reference is not None and (
                stored.provider_id,
                stored.model_id,
                stored.voice_id,
                stored.response_format,
                stored.speed,
                stored.options,
            ) != (
                draft.provider_id,
                draft.model_id,
                draft.voice_id,
                draft.response_format,
                draft.speed,
                draft.options,
            ):
                raise _repository_error("conflict")
            updated = TTSGenerationProfile(
                profile_id=profile_id,
                display_name=draft.display_name,
                normalized_name=draft.normalized_name,
                provider_id=draft.provider_id,
                model_id=draft.model_id,
                voice_id=draft.voice_id,
                response_format=draft.response_format,
                speed=draft.speed,
                options=cast(FrozenJsonOptions, draft.options),
                revision=stored.revision + 1,
                created_at=stored.created_at,
                updated_at=self._clock(),
                reference=stored.reference,
            )
            parameters = encode_profile(updated)
            parameters["expected_revision"] = expected_revision
            try:
                cursor = connection.execute(
                    """
                    UPDATE tts_generation_profiles
                    SET
                        display_name = :display_name,
                        normalized_name = :normalized_name,
                        provider_id = :provider_id,
                        model_id = :model_id,
                        voice_id = :voice_id,
                        response_format = :response_format,
                        speed = :speed,
                        options_json = :options_json,
                        revision = :revision,
                        updated_at = :updated_at
                    WHERE profile_id = :profile_id
                        AND revision = :expected_revision
                    """,
                    parameters,
                )
            except sqlite3.IntegrityError as error:
                evidence.statement_error = error
                raise
            if cursor.rowcount != 1:
                raise _repository_error("conflict")
            return self._worker_require_round_trip(
                connection,
                profile_id,
                updated,
            )

        return self._worker_transaction(
            connection,
            update,
            operation_kind="update",
            immediate=True,
            integrity_evidence=evidence,
        )

    def _worker_delete_profile(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
    ) -> None:
        evidence = _IntegrityEvidence(profile_id=profile_id)

        def delete() -> None:
            self._worker_get_profile(connection, profile_id)
            encoded_profile_id = encode_uuid(profile_id)
            try:
                cursor = connection.execute(
                    "DELETE FROM tts_generation_profiles WHERE profile_id = ?",
                    (encoded_profile_id,),
                )
            except sqlite3.IntegrityError as error:
                evidence.statement_error = error
                raise
            if cursor.rowcount == 0:
                raise _repository_error("missing")
            if cursor.rowcount != 1:
                raise _repository_error("corrupt_data")

        self._worker_transaction(
            connection,
            delete,
            operation_kind="delete",
            immediate=True,
            integrity_evidence=evidence,
        )
        self._discard_reference_damage_marker(profile_id)

    def _worker_bump_reference_revision(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        expected_revision: int,
        timestamp: datetime,
    ) -> TTSGenerationProfile:
        cursor = connection.execute(
            """
            UPDATE tts_generation_profiles
            SET revision = revision + 1, updated_at = ?
            WHERE profile_id = ? AND revision = ?
            """,
            (
                encode_utc_datetime(timestamp),
                encode_uuid(profile_id),
                expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise _repository_error("conflict")
        profile = self._worker_get_profile(connection, profile_id)
        if profile.revision != expected_revision + 1:
            raise _repository_error("corrupt_data")
        return profile

    def _worker_put_reference(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        canonical: CanonicalTTSCloneReference,
        recipe_requirement: TTSCloneRecipeRequirement,
        expected_revision: int,
    ) -> TTSGenerationProfile:
        """Insert a first reference inside the caller-owned transaction."""

        profile = self._worker_get_base_profile(connection, profile_id)
        if profile.revision != expected_revision:
            raise _repository_error("conflict")
        encoded_profile_id = encode_uuid(profile_id)
        existing = connection.execute(
            f"SELECT 1 FROM {REFERENCE_TABLE} WHERE profile_id = ?",
            (encoded_profile_id,),
        ).fetchone()
        if existing is not None:
            raise _repository_error("conflict")
        quota = connection.execute(
            f"SELECT COUNT(*), COALESCE(SUM(byte_length), 0) FROM {REFERENCE_TABLE}"
        ).fetchone()
        if (
            quota is None
            or len(quota) != 2
            or type(quota[0]) is not int
            or type(quota[1]) is not int
            or quota[0] < 0
            or quota[1] < 0
        ):
            raise _repository_error("corrupt_data")
        if (
            cast(int, quota[0]) + 1 > MAX_REFERENCE_COUNT
            or cast(int, quota[1]) + canonical.byte_length > MAX_REFERENCE_TOTAL_BYTES
        ):
            raise _repository_error("reference_quota")

        reference_id = self._worker_new_uuid()
        timestamp = self._clock()
        cursor = connection.execute(
            f"""
            INSERT INTO {REFERENCE_TABLE} (
                profile_id, reference_id, wav_bytes, reference_text, sha256,
                byte_length, duration_ms, sample_rate_hz, channels,
                sample_encoding, created_at, updated_at, recipe_id,
                recipe_revision
            ) VALUES (?, ?, zeroblob(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                encoded_profile_id,
                encode_uuid(reference_id),
                canonical.byte_length,
                canonical.reference_text,
                canonical.sha256,
                canonical.byte_length,
                canonical.duration_ms,
                canonical.sample_rate_hz,
                canonical.channels,
                canonical.sample_encoding,
                encode_utc_datetime(timestamp),
                encode_utc_datetime(timestamp),
                recipe_requirement.recipe_id,
                recipe_requirement.recipe_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise _repository_error("operation_failed")
        rowid_row = connection.execute(
            f"SELECT rowid FROM {REFERENCE_TABLE} WHERE profile_id = ?",
            (encoded_profile_id,),
        ).fetchone()
        if (
            rowid_row is None
            or len(rowid_row) != 1
            or type(rowid_row[0]) is not int
            or rowid_row[0] <= 0
        ):
            raise _repository_error("corrupt_data")
        write_reference_blob(connection, cast(int, rowid_row[0]), canonical.wav_bytes)
        updated = self._worker_bump_reference_revision(
            connection,
            profile_id,
            expected_revision,
            timestamp,
        )
        if (
            updated.reference is None
            or updated.reference.reference_id != reference_id
            or updated.reference.byte_length != canonical.byte_length
        ):
            raise _repository_error("corrupt_data")
        return updated

    def _worker_set_reference(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        canonical: CanonicalTTSCloneReference,
        recipe_requirement: TTSCloneRecipeRequirement,
        expected_revision: int,
    ) -> TTSGenerationProfile:
        def set_exact() -> TTSGenerationProfile:
            profile = self._worker_get_base_profile(connection, profile_id)
            if profile.revision != expected_revision:
                raise _repository_error("conflict")
            if recipe_requirement.model_id != profile.model_id:
                raise _repository_error("operation_failed")
            encoded_profile_id = encode_uuid(profile_id)
            existing_row = connection.execute(
                f"SELECT byte_length FROM {REFERENCE_TABLE} WHERE profile_id = ?",
                (encoded_profile_id,),
            ).fetchone()
            existing_bytes = 0
            if existing_row is not None:
                if (
                    len(existing_row) != 1
                    or type(existing_row[0]) is not int
                    or existing_row[0] <= 0
                ):
                    raise _repository_error("corrupt_data")
                existing_bytes = cast(int, existing_row[0])
            quota_row = connection.execute(
                f"SELECT COUNT(*), COALESCE(SUM(byte_length), 0) FROM {REFERENCE_TABLE}"
            ).fetchone()
            if (
                quota_row is None
                or len(quota_row) != 2
                or type(quota_row[0]) is not int
                or type(quota_row[1]) is not int
                or quota_row[0] < 0
                or quota_row[1] < 0
            ):
                raise _repository_error("corrupt_data")
            prospective_count = cast(int, quota_row[0]) + (
                0 if existing_row is not None else 1
            )
            prospective_bytes = (
                cast(int, quota_row[1]) - existing_bytes + canonical.byte_length
            )
            if (
                prospective_count > MAX_REFERENCE_COUNT
                or prospective_bytes > MAX_REFERENCE_TOTAL_BYTES
            ):
                raise _repository_error("reference_quota")

            reference_id = self._worker_new_uuid()
            timestamp = self._clock()
            cursor = connection.execute(
                f"""
                INSERT INTO {REFERENCE_TABLE} (
                    profile_id,
                    reference_id,
                    wav_bytes,
                    reference_text,
                    sha256,
                    byte_length,
                    duration_ms,
                    sample_rate_hz,
                    channels,
                    sample_encoding,
                    created_at,
                    updated_at,
                    recipe_id,
                    recipe_revision
                ) VALUES (?, ?, zeroblob(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    reference_id = excluded.reference_id,
                    wav_bytes = excluded.wav_bytes,
                    reference_text = excluded.reference_text,
                    sha256 = excluded.sha256,
                    byte_length = excluded.byte_length,
                    duration_ms = excluded.duration_ms,
                    sample_rate_hz = excluded.sample_rate_hz,
                    channels = excluded.channels,
                    sample_encoding = excluded.sample_encoding,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    recipe_id = excluded.recipe_id,
                    recipe_revision = excluded.recipe_revision
                """,
                (
                    encoded_profile_id,
                    encode_uuid(reference_id),
                    canonical.byte_length,
                    canonical.reference_text,
                    canonical.sha256,
                    canonical.byte_length,
                    canonical.duration_ms,
                    canonical.sample_rate_hz,
                    canonical.channels,
                    canonical.sample_encoding,
                    encode_utc_datetime(timestamp),
                    encode_utc_datetime(timestamp),
                    recipe_requirement.recipe_id,
                    recipe_requirement.recipe_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise _repository_error("operation_failed")
            rowid_row = connection.execute(
                f"SELECT rowid FROM {REFERENCE_TABLE} WHERE profile_id = ?",
                (encoded_profile_id,),
            ).fetchone()
            if (
                rowid_row is None
                or len(rowid_row) != 1
                or type(rowid_row[0]) is not int
                or rowid_row[0] <= 0
            ):
                raise _repository_error("corrupt_data")
            write_reference_blob(
                connection, cast(int, rowid_row[0]), canonical.wav_bytes
            )
            updated = self._worker_bump_reference_revision(
                connection,
                profile_id,
                expected_revision,
                timestamp,
            )
            if (
                updated.reference is None
                or updated.reference.reference_id != reference_id
                or updated.reference.byte_length != canonical.byte_length
            ):
                raise _repository_error("corrupt_data")
            self._discard_reference_damage_marker(profile_id)
            return updated

        return self._worker_transaction(
            connection,
            set_exact,
            operation_kind="reference_set",
            immediate=True,
        )

    def _worker_remove_reference(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        expected_revision: int,
    ) -> TTSGenerationProfile:
        def remove_exact() -> TTSGenerationProfile:
            profile = self._worker_get_base_profile(connection, profile_id)
            if profile.revision != expected_revision:
                raise _repository_error("conflict")
            cursor = connection.execute(
                f"DELETE FROM {REFERENCE_TABLE} WHERE profile_id = ?",
                (encode_uuid(profile_id),),
            )
            if cursor.rowcount == 0:
                raise _repository_error("missing")
            if cursor.rowcount != 1:
                raise _repository_error("corrupt_data")
            updated = self._worker_bump_reference_revision(
                connection,
                profile_id,
                expected_revision,
                self._clock(),
            )
            if updated.reference is not None:
                raise _repository_error("corrupt_data")
            self._discard_reference_damage_marker(profile_id)
            return updated

        return self._worker_transaction(
            connection,
            remove_exact,
            operation_kind="reference_remove",
            immediate=True,
        )

    def _worker_get_reference(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        expected_revision: int,
        expected_generation: int,
    ) -> TTSCloneReference:
        def read_exact() -> TTSCloneReference:
            if self._reference_damage_is_marked(profile_id, expected_generation):
                raise _repository_error("reference_unavailable")
            profile = self._worker_get_base_profile(connection, profile_id)
            if profile.revision != expected_revision:
                raise _repository_error("conflict")
            structural_error = False
            try:
                row = connection.execute(
                    f"{REFERENCE_PAYLOAD_SELECT} WHERE profile_id = ?",
                    (encode_uuid(profile_id),),
                ).fetchone()
            except sqlite3.DatabaseError:
                structural_error = True
                row = None
            if structural_error:
                raise _repository_error("schema_corrupt") from None
            if row is None:
                raise _repository_error("missing")
            rowid = row["reference_rowid"]
            byte_length = row["reference_byte_length"]
            if (
                type(rowid) is not int
                or rowid <= 0
                or type(byte_length) is not int
                or byte_length <= 0
            ):
                raise _repository_error("reference_unavailable")
            try:
                payload = read_reference_blob(connection, rowid, byte_length)
            except sqlite3.DatabaseError:
                raise _repository_error("schema_corrupt") from None
            reference = decode_reference_payload(row, payload)
            validation_error: BaseException | None = None
            metadata = None
            try:
                metadata = validate_canonical_reference_wav(payload)
            except BaseException as error:
                validation_error = error
            if validation_error is not None and not isinstance(
                validation_error, Exception
            ):
                raise validation_error
            if (
                validation_error is not None
                or metadata is None
                or metadata.byte_length != reference.summary.byte_length
                or metadata.duration_ms != reference.summary.duration_ms
                or metadata.sample_rate_hz != reference.summary.sample_rate_hz
                or metadata.channels != reference.summary.channels
                or metadata.sample_encoding != reference.summary.sample_encoding
            ):
                raise _repository_error("reference_unavailable")
            return reference

        try:
            return self._worker_transaction(
                connection,
                read_exact,
                operation_kind="read",
                immediate=False,
            )
        except ProfileRepositoryError as error:
            if error.code == "schema_corrupt":
                with self._state_lock:
                    if (
                        not self._terminal
                        and self._state is ProfileRepositoryState.OPEN
                    ):
                        self._state = ProfileRepositoryState.UNAVAILABLE
                cleanup_error: BaseException | None = None
                try:
                    self._worker_cleanup()
                except BaseException as caught:
                    cleanup_error = caught
                _raise_with_cleanup_precedence(error, cleanup_error)
            if error.code == "reference_unavailable":
                self._mark_reference_damage(profile_id, expected_generation)
            raise

    def _worker_assignment_count(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
    ) -> int:
        def count() -> int:
            self._worker_get_profile(connection, profile_id)
            row = connection.execute(
                """
                SELECT COUNT(*)
                FROM character_tts_assignments
                WHERE profile_id = ?
                """,
                (encode_uuid(profile_id),),
            ).fetchone()
            if row is None or len(row) != 1 or type(row[0]) is not int or row[0] < 0:
                raise _repository_error("corrupt_data")
            return cast(int, row[0])

        return self._worker_transaction(
            connection,
            count,
            operation_kind="read",
            immediate=False,
        )

    def _worker_require_generation(self, expected_generation: int) -> None:
        with self._state_lock:
            state_error = self._worker_state_error_locked(expected_generation)
        if state_error is not None:
            raise _repository_error(state_error)

    def _worker_set_assignment(
        self,
        connection: sqlite3.Connection,
        character_ref: CharacterRef,
        profile_id: UUID,
        expected_generation: int,
        expected_profile_revision: int,
        expected_current_profile_id: UUID | None,
        expected_profile: TTSGenerationProfile | None,
    ) -> CharacterTTSAssignment:
        return self._worker_transaction(
            connection,
            lambda: self._worker_set_assignment_exact(
                connection,
                character_ref,
                profile_id,
                expected_generation,
                expected_profile_revision,
                expected_current_profile_id,
                expected_profile,
            ),
            operation_kind="assignment_set",
            immediate=True,
        )

    def _worker_set_assignment_exact(
        self,
        connection: sqlite3.Connection,
        character_ref: CharacterRef,
        profile_id: UUID,
        expected_generation: int,
        expected_profile_revision: int,
        expected_current_profile_id: UUID | None,
        expected_profile: TTSGenerationProfile | None,
    ) -> CharacterTTSAssignment:
        self._worker_require_generation(expected_generation)
        selected_profile = self._worker_get_profile(connection, profile_id)
        if selected_profile.revision != expected_profile_revision:
            raise _repository_error("conflict")
        if expected_profile is not None and selected_profile != expected_profile:
            raise _repository_error("conflict")
        existing = self._worker_get_persisted_assignment(connection, character_ref)
        current_profile_id = (
            None if existing is None else existing.assignment.profile_id
        )
        if current_profile_id != expected_current_profile_id:
            raise _repository_error("conflict")
        assignment = CharacterTTSAssignment(
            character_ref=character_ref,
            profile_id=profile_id,
        )
        timestamp = self._clock()
        created_at = timestamp if existing is None else existing.created_at
        updated_at = (
            timestamp if existing is None else max(existing.updated_at, timestamp)
        )
        expected = _PersistedAssignment(
            assignment=assignment,
            created_at=created_at,
            updated_at=updated_at,
        )
        parameters = encode_assignment(
            assignment,
            created_at=created_at,
            updated_at=updated_at,
        )
        cursor = connection.execute(
            """
            INSERT INTO character_tts_assignments (
                source,
                authority_id,
                character_id,
                profile_id,
                created_at,
                updated_at
            ) VALUES (
                :source,
                :authority_id,
                :character_id,
                :profile_id,
                :created_at,
                :updated_at
            )
            ON CONFLICT(source, authority_id, character_id)
            DO UPDATE SET
                profile_id = excluded.profile_id,
                updated_at = excluded.updated_at
            """,
            parameters,
        )
        if cursor.rowcount != 1:
            raise _repository_error("corrupt_data")
        persisted = self._worker_get_persisted_assignment(connection, character_ref)
        if persisted != expected:
            raise _repository_error("corrupt_data")
        return persisted.assignment

    def _worker_remove_assignment(
        self,
        connection: sqlite3.Connection,
        character_ref: CharacterRef,
        expected_generation: int,
        expected_profile_id: UUID,
    ) -> None:
        def remove_exact() -> None:
            self._worker_require_generation(expected_generation)
            existing = self._worker_get_persisted_assignment(
                connection,
                character_ref,
            )
            if existing is None:
                return
            if existing.assignment.profile_id != expected_profile_id:
                raise _repository_error("conflict")
            cursor = connection.execute(
                """
                DELETE FROM character_tts_assignments
                WHERE source = ?
                    AND authority_id = ?
                    AND character_id = ?
                    AND profile_id = ?
                """,
                (
                    character_ref.source,
                    character_ref.authority_id,
                    character_ref.character_id,
                    encode_uuid(expected_profile_id),
                ),
            )
            if cursor.rowcount != 1:
                raise _repository_error("corrupt_data")
            if (
                self._worker_get_persisted_assignment(connection, character_ref)
                is not None
            ):
                raise _repository_error("corrupt_data")

        self._worker_transaction(
            connection,
            remove_exact,
            operation_kind="assignment_remove",
            immediate=True,
        )

    def _worker_get_persisted_assignment(
        self,
        connection: sqlite3.Connection,
        character_ref: CharacterRef,
    ) -> _PersistedAssignment | None:
        row = connection.execute(
            (
                f"{_ASSIGNMENT_SELECT} "
                "WHERE source = ? AND authority_id = ? AND character_id = ?"
            ),
            (
                character_ref.source,
                character_ref.authority_id,
                character_ref.character_id,
            ),
        ).fetchone()
        if row is None:
            return None
        assignment = decode_assignment(row)
        if assignment.character_ref != character_ref:
            raise _repository_error("corrupt_data")
        created_at = decode_utc_datetime(row["created_at"])
        updated_at = decode_utc_datetime(row["updated_at"])
        if created_at > updated_at:
            raise _repository_error("corrupt_data")
        return _PersistedAssignment(
            assignment=assignment,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _worker_get_assigned_profile(
        self,
        connection: sqlite3.Connection,
        character_ref: CharacterRef,
    ) -> AssignedTTSProfileSnapshot | None:
        row = connection.execute(
            (
                f"{ASSIGNED_PROFILE_WITH_REFERENCE_JOIN_SELECT} "
                "WHERE a.source = ? "
                "AND a.authority_id = ? "
                "AND a.character_id = ?"
            ),
            (
                character_ref.source,
                character_ref.authority_id,
                character_ref.character_id,
            ),
        ).fetchone()
        if row is None:
            return None
        snapshot = _decode_assigned_with_reference_row(row)
        if snapshot.assignment.character_ref != character_ref:
            raise _repository_error("corrupt_data")
        return snapshot

    def _worker_require_round_trip(
        self,
        connection: sqlite3.Connection,
        profile_id: UUID,
        expected: TTSGenerationProfile,
    ) -> TTSGenerationProfile:
        decoded = self._worker_get_profile(connection, profile_id)
        if decoded != expected:
            raise _repository_error("corrupt_data")
        return decoded

    def _worker_new_uuid(self) -> UUID:
        generated = self._uuid_factory()
        if type(generated) is not UUID or generated.version != 4:
            raise _repository_error("operation_failed")
        return generated

    def _worker_transaction(
        self,
        connection: sqlite3.Connection,
        operation: Callable[[], _T],
        *,
        operation_kind: _TransactionOperation,
        immediate: bool,
        integrity_evidence: _IntegrityEvidence | None = None,
    ) -> _T:
        body_error: BaseException | None = None
        value: _T | None = None
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            value = operation()
            self._worker_revalidate_exact_authority(connection)
            self._commit_transaction(connection)
        except BaseException as error:
            body_error = error

        if body_error is None:
            return cast(_T, value)

        if isinstance(body_error, ExactProfileStoreAuthorityError):
            _raise_with_cleanup_precedence(body_error)

        integrity_conflict = False
        classification_error: BaseException | None = None
        if (
            isinstance(body_error, sqlite3.IntegrityError)
            and integrity_evidence is not None
            and integrity_evidence.statement_error is body_error
        ):
            try:
                integrity_conflict = _has_integrity_conflict_evidence(
                    connection,
                    body_error,
                    operation_kind,
                    integrity_evidence,
                )
            except BaseException as error:
                classification_error = error

        rollback_error: BaseException | None = None
        try:
            self._rollback_transaction(connection)
            if connection.in_transaction:
                raise _repository_error("operation_failed")
        except BaseException as error:
            rollback_error = error

        cleanup_error: BaseException | None = None
        if rollback_error is not None:
            with self._state_lock:
                if not self._terminal and self._state is ProfileRepositoryState.OPEN:
                    self._state = ProfileRepositoryState.UNAVAILABLE
            try:
                self._worker_cleanup()
            except BaseException as error:
                cleanup_error = error

        if isinstance(body_error, sqlite3.IntegrityError):
            if classification_error is not None:
                body_error = classification_error
            else:
                body_error = _repository_error(
                    "conflict" if integrity_conflict else "operation_failed"
                )
        _raise_with_cleanup_precedence(
            body_error,
            rollback_error,
            cleanup_error,
        )
        raise AssertionError("unreachable")

    def _commit_transaction(self, connection: sqlite3.Connection) -> None:
        """Commit one worker-owned transaction.

        This small boundary also permits deterministic fault injection without
        exposing a public repository test hook.
        """

        connection.commit()

    def _rollback_transaction(self, connection: sqlite3.Connection) -> None:
        """Roll back one worker-owned transaction."""

        connection.rollback()

    async def _submit_operation(
        self,
        operation: Callable[[sqlite3.Connection], _T],
        *,
        expected_generation: int | None = None,
    ) -> ProfileStoreResult[_T]:
        """Submit and publish one normal generation-bound operation."""

        self._bind_or_check_loop()
        admission = self._admit_operation(
            operation,
            expected_generation=expected_generation,
        )
        return await self._publish_operation(admission)

    def _admit_operation(
        self,
        operation: Callable[[sqlite3.Connection], _T],
        *,
        expected_generation: int | None = None,
    ) -> _OperationAdmission[_T]:
        """Synchronously capture state/generation and register a worker future."""

        self._bind_or_check_loop()
        if not callable(operation):
            raise _repository_error("operation_failed")

        submission_error: BaseException | None = None
        future: Future[_T] | None = None
        with self._state_lock:
            state_error = self._normal_state_error_locked()
            if state_error is not None:
                raise _repository_error(state_error)
            generation = self._generation
            if expected_generation is not None and expected_generation != generation:
                raise _repository_error("stale")
            executor = self._executor
            if executor is None or self._executor_shutdown:
                raise _repository_error("invalid_state")
            try:
                future = executor.submit(
                    self._worker_operation,
                    generation,
                    operation,
                )
            except BaseException as error:
                submission_error = error
            if future is not None:
                self._pending_futures.add(cast(Future[object], future))

        if submission_error is not None:
            _raise_operation_error(submission_error)
        assert future is not None
        future.add_done_callback(self._discard_pending_future)
        return _OperationAdmission(generation=generation, future=future)

    def _normal_state_error_locked(self) -> str | None:
        if self._terminal:
            return "terminal"
        if self._state is ProfileRepositoryState.CLOSED:
            return "closed"
        if self._state is ProfileRepositoryState.RESTORING:
            return "restoring"
        if self._state is ProfileRepositoryState.UNAVAILABLE:
            return "unavailable"
        if self._state is not ProfileRepositoryState.OPEN:
            return "invalid_state"
        return None

    def _discard_pending_future(self, future: Future[_T]) -> None:
        with self._state_lock:
            self._pending_futures.discard(cast(Future[object], future))

    def _worker_operation(
        self,
        generation: int,
        operation: Callable[[sqlite3.Connection], _T],
    ) -> _T:
        """Check freshness immediately before invoking one SQLite operation."""

        with self._state_lock:
            state_error = self._worker_state_error_locked(generation)
            connection = self._connection
        if state_error is not None:
            raise _repository_error(state_error)
        if connection is None:
            raise _repository_error("invalid_state")

        operation_error: BaseException | None = None
        value: _T | None = None
        try:
            self._worker_revalidate_exact_authority(connection)
            value = operation(connection)
            self._worker_revalidate_exact_authority(connection)
        except BaseException as error:
            operation_error = error
        if operation_error is not None:
            if isinstance(operation_error, ExactProfileStoreAuthorityError):
                self._worker_seal_exact_authority(operation_error)
            _raise_operation_error(operation_error)
        return cast(_T, value)

    def _worker_revalidate_exact_authority(
        self,
        connection: sqlite3.Connection,
        active_path: Path | None = None,
    ) -> None:
        """Fence one worker operation against live namespace substitution."""

        selected_path = (
            self._active_database_path if active_path is None else active_path
        )
        revalidate_exact_current_profile_store(connection, selected_path)

    def _worker_seal_exact_authority(
        self,
        error: ExactProfileStoreAuthorityError,
    ) -> None:
        """Quarantine exact authority without invoking SQLite cleanup."""

        with self._state_lock:
            if not self._terminal:
                self._state = ProfileRepositoryState.UNAVAILABLE
            self._exact_authority_quarantined = True
        _raise_with_cleanup_precedence(error)

    def _worker_state_error_locked(self, generation: int) -> str | None:
        if generation != self._generation:
            return "stale"
        return self._normal_state_error_locked()

    async def _publish_operation(
        self,
        admission: _OperationAdmission[_T],
    ) -> ProfileStoreResult[_T]:
        """Await a shielded worker future and publish only if it remains current."""

        self._bind_or_check_loop()
        wrapped_future = asyncio.wrap_future(admission.future)
        wrapped_future.add_done_callback(_retrieve_future_exception)
        worker_cancelled = False
        worker_error: BaseException | None = None
        try:
            value = await asyncio.shield(wrapped_future)
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling() > 0:
                raise
            worker_cancelled = wrapped_future.cancelled()
            if not worker_cancelled:
                raise
            value = cast(_T, None)
        except BaseException as error:
            worker_error = error
            value = cast(_T, None)

        if worker_cancelled:
            raise _repository_error("stale")
        if worker_error is not None:
            _raise_operation_error(worker_error)

        with self._state_lock:
            state_error = self._worker_state_error_locked(admission.generation)
        if state_error is not None:
            raise _repository_error(state_error)
        return ProfileStoreResult(
            generation=admission.generation,
            value=value,
        )

    async def close(self) -> ProfileStoreResult[None]:
        """Close safely, retaining quarantined authority until it matches again.

        Exact namespace loss makes SQLite cleanup unsafe because close may
        mutate or remove a foreign WAL/SHM cohort.  In that state this method
        reports ``operation_failed`` and retains the worker, connection, and
        lease.  A later call may settle cleanup after the exact inode cohort is
        restored; otherwise process exit is the only non-mutating release.
        """

        lifecycle_lock = self._bind_or_check_loop()
        async with lifecycle_lock:
            with self._state_lock:
                if self._terminal:
                    if not self._exact_authority_quarantined:
                        return ProfileStoreResult(
                            generation=self._generation,
                            value=None,
                        )
                    generation = self._generation
                else:
                    self._generation += 1
                    self._damaged_reference_profile_ids.clear()
                    generation = self._generation
                    self._terminal = True
                    self._state = ProfileRepositoryState.CLOSED
                executor = self._executor
                pending = tuple(self._pending_futures)
                authority_quarantined = self._exact_authority_quarantined

            for future in pending:
                future.cancel()

            if executor is None:
                if authority_quarantined:
                    raise _repository_error("operation_failed")
                return ProfileStoreResult(generation=generation, value=None)

            completion = asyncio.create_task(self._finish_close(executor, pending))
            await self._await_lifecycle_completion(completion)
            return ProfileStoreResult(generation=generation, value=None)

    async def _finish_close(
        self,
        executor: ThreadPoolExecutor,
        pending: tuple[Future[object], ...],
    ) -> None:
        """Drain admitted work, clean worker ownership, and shut down off-loop."""

        self._bind_or_check_loop()
        if pending:
            await asyncio.gather(
                *(asyncio.shield(asyncio.wrap_future(future)) for future in pending),
                return_exceptions=True,
            )

        cleanup_error: BaseException | None = None
        cleanup_future: Future[None] | None = None
        try:
            cleanup_future = executor.submit(self._worker_cleanup)
        except BaseException as error:
            cleanup_error = error

        if cleanup_future is not None:
            try:
                await asyncio.wrap_future(cleanup_future)
            except BaseException as error:
                cleanup_error = error

        with self._state_lock:
            authority_quarantined = self._exact_authority_quarantined
        if cleanup_error is not None and authority_quarantined:
            _raise_cleanup_errors(cleanup_error)

        shutdown_error: BaseException | None = None
        with self._state_lock:
            self._executor_shutdown = True
        try:
            await asyncio.to_thread(
                executor.shutdown,
                wait=True,
                cancel_futures=True,
            )
        except BaseException as error:
            shutdown_error = error
        finally:
            with self._state_lock:
                if self._executor is executor:
                    self._executor = None

        _raise_cleanup_errors(cleanup_error, shutdown_error)

    def _worker_cleanup(self) -> None:
        """Close SQLite before its lease, only under revalidated authority."""

        self._clear_reference_damage_markers()
        connection = self._connection
        lease = self._lease
        connection_error: BaseException | None = None
        residual_error: BaseException | None = None
        lease_error: BaseException | None = None

        if connection is not None:
            try:
                self._worker_revalidate_exact_authority(connection)
            except ExactProfileStoreAuthorityError:
                with self._state_lock:
                    self._exact_authority_quarantined = True
                raise
            with self._state_lock:
                self._exact_authority_quarantined = False
            parent_fd = getattr(cast(object, connection), "parent_fd", -1)
            active_path = self._active_database_path
            if (
                self._reusable_tombstones
                and active_path is not None
                and type(parent_fd) is int
                and parent_fd >= 0
            ):
                try:
                    self._worker_settle_reusable_tombstones(active_path, parent_fd)
                except BaseException:
                    with self._state_lock:
                        self._exact_authority_quarantined = True
                    raise
        elif self._exact_authority_quarantined:
            raise _repository_error("operation_failed")

        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                connection_error = error
            if connection_error is None:
                self._connection = None

        if connection_error is None and self._residual_cleanup_paths:
            for path in self._residual_cleanup_paths:
                try:
                    _unlink_path_if_present(path)
                except BaseException as error:
                    if residual_error is None:
                        residual_error = error
            if residual_error is None:
                self._residual_cleanup_paths = ()

        if lease is not None and connection_error is None and residual_error is None:
            try:
                lease.release()
            except BaseException as error:
                lease_error = error
            if lease_error is None:
                self._lease = None

        if (
            self._connection is None
            and self._lease is None
            and not self._residual_cleanup_paths
        ):
            self._active_database_path = None

        _raise_cleanup_errors(connection_error, residual_error, lease_error)

    async def _await_lifecycle_completion(
        self,
        completion: asyncio.Task[_T],
    ) -> _T:
        """Delay caller cancellation until a lifecycle transition settles."""

        self._bind_or_check_loop()
        cancellation: asyncio.CancelledError | None = None
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except BaseException:
                break

        completion_error: BaseException | None = None
        result: _T | None = None
        try:
            result = completion.result()
        except BaseException as error:
            completion_error = error

        if cancellation is not None:
            raise cancellation
        if completion_error is not None:
            _raise_operation_error(completion_error)
        return cast(_T, result)

    def _bind_or_check_loop(self) -> asyncio.Lock:
        """Bind first async use and reject every later foreign-loop caller."""

        running_loop: asyncio.AbstractEventLoop | None = None
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        if running_loop is None:
            raise _repository_error("invalid_state")

        wrong_loop = False
        lifecycle_lock: asyncio.Lock | None = None
        with self._state_lock:
            if self._owner_loop is None:
                lifecycle_lock = asyncio.Lock()
                self._owner_loop = running_loop
                self._lifecycle_lock = lifecycle_lock
            elif self._owner_loop is not running_loop:
                wrong_loop = True
            else:
                lifecycle_lock = self._lifecycle_lock

        if wrong_loop or lifecycle_lock is None:
            raise _repository_error("invalid_state")
        return lifecycle_lock
