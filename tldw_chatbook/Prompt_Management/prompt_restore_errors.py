"""Bounded, privacy-safe retained Prompt restore failures."""

from __future__ import annotations

from enum import StrEnum


class PromptRestoreErrorCode(StrEnum):
    """Stable restore failure categories safe to translate into fixed UI copy."""

    EXPECTED_VERSION = "expected_version"
    NAME_CONFLICT = "name_conflict"
    VALIDATION = "validation"


class PromptRestoreError(ValueError):
    """A classified restore failure whose arbitrary source text is never exposed."""

    def __init__(self, code: PromptRestoreErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


def prompt_restore_error_from_conflict(error: object) -> PromptRestoreError | None:
    """Translate only known DB conflict codes into bounded restore categories."""
    code = getattr(error, "code", None)
    if code == PromptRestoreErrorCode.EXPECTED_VERSION:
        return PromptRestoreError(PromptRestoreErrorCode.EXPECTED_VERSION)
    if code == PromptRestoreErrorCode.NAME_CONFLICT:
        return PromptRestoreError(PromptRestoreErrorCode.NAME_CONFLICT)
    return None
