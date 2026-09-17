"""Safe, structured errors for TTS generation-profile operations."""

from __future__ import annotations

from tldw_chatbook.Utils.platform_files import os
from typing import Protocol


class _MigrationCleanupOwner(Protocol):
    def close(self) -> None: ...


_VALIDATION_CODES = frozenset(
    {
        "audio_cpp",
        "assignment",
        "assignment_count",
        "availability",
        "authority_id",
        "byte_count",
        "catalog_revision",
        "character_id",
        "choice",
        "configuration_revision",
        "created_at",
        "display_name",
        "generation",
        "model_id",
        "normalized_name",
        "options",
        "profile_id",
        "profile_count",
        "provider_id",
        "recovery_action",
        "reference_id",
        "reference_invalid",
        "reference_text",
        "profiles",
        "response_format",
        "revision",
        "restored_at",
        "source",
        "speed",
        "timestamps",
        "total",
        "updated_at",
        "voice_id",
        "byte_length",
        "duration_ms",
        "sample_rate_hz",
        "channels",
        "sample_encoding",
    }
)
_REPOSITORY_CODES = frozenset(
    {
        "backup_failed",
        "closed",
        "conflict",
        "corrupt_data",
        "invalid_state",
        "lock_timeout",
        "migration_failed",
        "missing",
        "operation_failed",
        "reference_quota",
        "reference_unavailable",
        "restore_failed",
        "restoring",
        "restart_required",
        "runtime_unsupported",
        "schema_corrupt",
        "schema_partial",
        "schema_unsupported",
        "stale",
        "terminal",
        "unavailable",
    }
)
_SERVICE_CODES = frozenset(
    {
        "artifact_ineligible",
        "operation_failed",
        "profile_unavailable",
        "profile_unverified",
        "stale_configuration",
        "unsupported_profile",
    }
)


class _ProfileError(Exception):
    """Base class for errors whose public detail is deliberately bounded."""

    __slots__ = ("code",)

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ProfileValidationError(_ProfileError, ValueError):
    """A value-independent failure at the profile-domain boundary."""

    def __init__(self, code: str) -> None:
        safe_code = (
            code if type(code) is str and code in _VALIDATION_CODES else "options"
        )
        super().__init__(safe_code, f"TTS profile validation failed: {safe_code}")

    def __reduce__(self) -> tuple[type["ProfileValidationError"], tuple[str]]:
        return (ProfileValidationError, (self.code,))


class ProfileRepositoryError(_ProfileError, RuntimeError):
    """A value-independent profile repository failure."""

    def __init__(self, code: str) -> None:
        safe_code = (
            code
            if type(code) is str and code in _REPOSITORY_CODES
            else "operation_failed"
        )
        message = (
            "TTS profile repository unavailable: restart is required."
            if safe_code == "restart_required"
            else "TTS profile repository unavailable: SQLite runtime lacks required "
            "close-policy support."
            if safe_code == "runtime_unsupported"
            else f"TTS profile repository failed: {safe_code}"
        )
        super().__init__(safe_code, message)

    def __reduce__(self) -> tuple[type["ProfileRepositoryError"], tuple[str]]:
        return (ProfileRepositoryError, (self.code,))


class ProfileMigrationCleanupError(ProfileRepositoryError):
    """Retain teardown authority after an exclusive migration close failed."""

    def __init__(
        self, owner: _MigrationCleanupOwner, *, code: str = "unavailable"
    ) -> None:
        super().__init__(code)
        self.owner = owner


def _migration_cleanup_owner(
    error: BaseException | None,
) -> _MigrationCleanupOwner | None:
    if error is None:
        return None
    metadata = BaseException.__dict__["__dict__"].__get__(error, BaseException)
    cleanup = (
        error
        if isinstance(error, ProfileMigrationCleanupError)
        else metadata.get("_profile_migration_cleanup_error")
    )
    if isinstance(cleanup, ProfileMigrationCleanupError):
        return BaseException.__dict__["__dict__"].__get__(cleanup, BaseException)[
            "owner"
        ]
    return None


def _raise_migration_cleanup_failure(
    owner: _MigrationCleanupOwner,
    *errors: BaseException | None,
    code: str = "unavailable",
) -> None:
    owners = []
    for error in errors:
        previous = _migration_cleanup_owner(error)
        if previous is not None and all(
            previous is not retained for retained in owners
        ):
            owners.append(previous)
    if all(owner is not retained for retained in owners):
        owners.append(owner)
    cleanup = ProfileMigrationCleanupError(
        owners[0] if len(owners) == 1 else _MigrationCleanupGroup(*owners), code=code
    )
    for error in errors:
        if error is not None and not isinstance(error, Exception):
            metadata = BaseException.__dict__["__dict__"].__get__(error, BaseException)
            metadata["_profile_migration_cleanup_error"] = cleanup
            raise error from None
    try:
        raise cleanup from None
    except ProfileMigrationCleanupError:
        # Even callers unwinding an active native exception expose no chain.
        cleanup.__context__ = None
        cleanup.__cause__ = None
        raise


class _MigrationCleanupGroup:
    def __init__(self, *owners: _MigrationCleanupOwner) -> None:
        self._owners = list(owners)

    def __repr__(self) -> str:
        return "_MigrationCleanupGroup(<private>)"

    def close(self) -> None:
        while self._owners:
            self._owners[0].close()
            self._owners.pop(0)


class _ProfileMigrationValidationOwner:
    """One exclusive view and its pins; retries only settle native teardown."""

    def __init__(self, file_fd: int, parent_fd: int = -1, *, native=None) -> None:
        self.native = native
        self.reader_outcome = None
        self.connection = None
        self.file_fd = file_fd
        self.parent_fd = parent_fd

    def __repr__(self) -> str:
        return "_ProfileMigrationValidationOwner(<private>)"

    def close_sqlite(self, body_error: BaseException | None = None) -> None:
        if self.connection is not None:
            close_error = None
            try:
                if self.native is not None and self.reader_outcome is not None:
                    self.native.retry_close_reader(self.connection, self.reader_outcome)
                else:
                    self.connection.close()
            except BaseException as error:  # noqa: BLE001 - preserve native ownership and control flow
                close_error = error
            if close_error is not None:
                _raise_migration_cleanup_failure(self, body_error, close_error)
            self.connection = None

    def close(self) -> None:
        self.close_sqlite()
        for attribute in ("file_fd", "parent_fd"):
            descriptor = getattr(self, attribute)
            if descriptor >= 0:
                # A raw close may consume the descriptor even when it errors.
                setattr(self, attribute, -1)
                try:
                    if self.native is not None:
                        self.native.close(os.close, descriptor)
                    else:
                        os.close(descriptor)
                except BaseException as error:  # noqa: BLE001 - retain remaining teardown authority
                    _raise_migration_cleanup_failure(self, error)
        if self.native is not None:
            self.native.finish()
            if self.native.active:
                _raise_migration_cleanup_failure(self, *self.native.errors)


class ProfileServiceError(_ProfileError, RuntimeError):
    """A value-independent failure at the profile-service boundary."""

    def __init__(self, code: str) -> None:
        safe_code = (
            code if type(code) is str and code in _SERVICE_CODES else "operation_failed"
        )
        super().__init__(safe_code, f"TTS profile service failed: {safe_code}")

    def __reduce__(self) -> tuple[type["ProfileServiceError"], tuple[str]]:
        return (ProfileServiceError, (self.code,))
