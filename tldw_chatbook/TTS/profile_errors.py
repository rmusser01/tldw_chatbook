"""Safe, structured errors for TTS generation-profile operations."""

from __future__ import annotations


_VALIDATION_CODES = frozenset(
    {
        "audio_cpp",
        "assignment",
        "assignment_count",
        "authority_id",
        "byte_count",
        "character_id",
        "created_at",
        "display_name",
        "generation",
        "model_id",
        "normalized_name",
        "options",
        "profile_id",
        "profile_count",
        "provider_id",
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
    }
)
_REPOSITORY_CODES = frozenset(
    {
        "backup_failed",
        "closed",
        "conflict",
        "invalid_state",
        "lock_timeout",
        "missing",
        "operation_failed",
        "restore_failed",
        "restoring",
        "stale",
        "terminal",
        "unavailable",
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
        super().__init__(safe_code, f"TTS profile repository failed: {safe_code}")

    def __reduce__(self) -> tuple[type["ProfileRepositoryError"], tuple[str]]:
        return (ProfileRepositoryError, (self.code,))
